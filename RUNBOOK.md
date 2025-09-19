# RUNBOOK — Commands & How-To

This is your copy‑paste guide from **zero → demo → screenshots → cleanup**.

This runbook is a copy-paste guide for local smoke tests, container builds, Kubernetes (kind) deployment, HPA autoscaling, troubleshooting, and cleanup. It assumes macOS on Apple Silicon but notes Linux/GPU variations.

---

## Prereqs

- macOS with Apple Silicon (instructions also fine on Linux; adjust platforms)
- Docker Desktop (or Colima) **running**
- Homebrew
- kind, kubectl, helm
- Python 3.12 recommended

```bash
brew install --cask docker
open -a Docker  # start Docker Desktop

brew install kind kubectl helm

# optional but handy
brew install git-filter-repo
```

---

## Local Smoke Test

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Torch first (CPU wheel)
python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu

# Service deps
python -m pip install -r service/requirements.txt
python -m pip install ./service/kernels || true

# Run locally
python -m uvicorn service.app:app --port 8000 --reload

# In a new terminal:
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}'
curl -s -X POST localhost:8000/kernel_demo -H 'content-type: application/json' \
  -d '{"data":[-1,2,-3,4]}'
```

---

## Containers

```bash
# Build for linux/arm64 to match kind on Apple Silicon
docker buildx build --platform linux/arm64 -f Dockerfile.service  -t ai-infer:latest --load .
docker buildx build --platform linux/arm64 -f Dockerfile.loadgen  -t ai-loadgen:latest --load .
```

---

## Kind + Metrics Server

```bash
kind create cluster --name aiinfer

# Load local images into kind
kind load docker-image ai-infer:latest --name aiinfer
kind load docker-image ai-loadgen:latest --name aiinfer

# Metrics server (required for HPA)
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/
helm upgrade --install metrics-server metrics-server/metrics-server \
  -n kube-system --create-namespace \
  --set args="{--kubelet-insecure-tls,--kubelet-preferred-address-types=InternalIP,Hostname,InternalDNS,ExternalDNS,ExternalIP}"

kubectl -n kube-system rollout status deploy/metrics-server
kubectl get apiservices | grep metrics
```

> If `kubectl top nodes` fails or HPA events show “no metrics returned,” wait a minute or patch metrics-server with:
>
> ```bash
> kubectl -n kube-system patch deploy metrics-server --type='json' -p='[
>   {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"},
>   {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-preferred-address-types=InternalIP,Hostname,InternalDNS,ExternalDNS,ExternalIP"},
>   {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-use-node-status-port"}
> ]'
> kubectl -n kube-system rollout status deploy/metrics-server
> ```

---

## Deploy (Option A: Plain YAML)

```bash
# If missing, create k8s/ai-infer.yaml from this repo (or README)
kubectl apply -f k8s/ai-infer.yaml
kubectl get pods -w
```

## Deploy (Option B: Helm, local chart at repo root)

```bash
# Ensure .helmignore excludes .venv and large files
helm install ai-infer . \
  --set image.service=ai-infer:latest \
  --set image.loadgen=ai-loadgen:latest \
  --set serviceMonitor.enabled=false
```

---

## Validate In-Cluster

```bash
# Terminal A
kubectl port-forward svc/ai-infer-service 8080:8080

# Terminal B
curl -s localhost:8080/healthz
curl -s -X POST localhost:8080/predict -H 'content-type: application/json' \
  -d '{"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}'
```

---

## Trigger Autoscaling

```bash
# Increase load
kubectl set env deploy/ai-loadgen QPS=300 CONCURRENCY=80

# Watch scaling
kubectl get hpa -w
kubectl get pods -w
```

Take screenshots of:
- `kubectl get hpa`
- `kubectl describe hpa ai-infer-service | sed -n '1,120p'`
- `kubectl get pods -o wide`
- API responses (`/healthz`, `/predict`)

---

## Troubleshooting

- **CrashLoopBackOff for service**: Ensure the container CMD is `python -m uvicorn service.app:app ...` and `service/__init__.py` exists. Set `ENV PYTHONPATH=/app` in `Dockerfile.service`.
- **ImportError: attempted relative import**: Use absolute import in `service/app.py`: `from service.model import predict`.
- **Image not found in cluster**: Re-run `kind load docker-image ...` for both images, then `kubectl rollout restart deploy/ai-infer-service`.
- **HPA not scaling**: Ensure metrics-server is Ready; lower target utilization (to 10–20%) and/or increase QPS/CONCURRENCY.
- **Helm ServiceMonitor error**: Disable with `--set serviceMonitor.enabled=false` or remove the template.
- **Git push blocked by large files**: Add `.gitignore` for `.venv/`, run `git filter-repo --strip-blobs-bigger-than 50M` and force-push.

---

## Cleanup

```bash
# If using plain YAML
kubectl delete -f k8s/ai-infer.yaml

# If using Helm
helm uninstall ai-infer
helm uninstall metrics-server -n kube-system

# Remove cluster
kind delete cluster --name aiinfer
```


