# AI Inference Microservice

A tiny, production-flavored inference microservice that exercises the following stack listed:
**Python + PyTorch + FastAPI**, **C++/pybind11** (custom op), **Golang** load generator with **Prometheus** metrics, **Docker**, **Kubernetes**, and **Horizontal Pod Autoscaling (HPA)**. CUDA/Triton/CUTLASS/vLLM are stubbed with clear next steps and documented in the roadmap.

> Built overnight to showcase end‑to‑end skills: model serving, custom kernels, containerization, K8s deployment, autoscaling, and observability hooks.

---

## Highlights

- **FastAPI + PyTorch** inference service (`/predict`)
- **Custom C++ op (pybind11)** exposed at `/kernel_demo` (falls back to NumPy if the wheel isn’t available)
- **/healthz** exposes feature flags (e.g., `cuda`, `kernel`)
- **Go load-generator** with Prometheus metrics on `:9090/metrics`
- **Kubernetes + HPA** scales the service from 1 → N based on CPU
- **Kind-ready** development flow and **metrics-server** recipe
- **Open Source** friendly: MIT license, readable structure, and a clear **roadmap** to CUDA/Triton/CUTLASS/vLLM

---

## Repo Structure (key files)

```
.
├─ service/
│  ├─ app.py                 # FastAPI app (predict, kernel_demo, healthz)
│  ├─ model.py               # Tiny MLP
│  ├─ requirements.txt       # Runtime deps
│  └─ kernels/               # pybind11 extension (CPU-only build path)
│
├─ go-loadgen/               # Go-based load generator (Prometheus metrics)
├─ Dockerfile.service        # Service container (runs service.app:app)
├─ Dockerfile.loadgen        # Loadgen container
├─ k8s/ai-infer.yaml         # (Optional) Plain manifests for quick deploys
├─ Chart.yaml, templates/, values.yaml  # (Optional) Helm chart at repo root
├─ Makefile                  # (Optional) helpers
└─ LICENSE                   # MIT
```

> You can re-create `k8s/ai-infer.yaml` from the **RUNBOOK.md** below.

---

## Quick Start (Local)

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip setuptools wheel

# Python 3.12 recommended (avoids PyO3 issues on 3.13)
# torch first (CPU wheel OK on macOS/arm64)
python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r service/requirements.txt
python -m pip install ./service/kernels || true

# run
python -m uvicorn service.app:app --port 8000 --reload

# new terminal
curl -s localhost:8000/healthz
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}'
curl -s -X POST localhost:8000/kernel_demo -H 'content-type: application/json' \
  -d '{"data":[-1,2,-3,4]}'
```

---

## Kubernetes Demo (Kind + HPA)

> Apple Silicon/macOS shown. Docker Desktop or Colima must be running.

```bash
# Build images (arm64 for kind’s node on Apple Silicon)
docker buildx build --platform linux/arm64 -f Dockerfile.service  -t ai-infer:latest --load .
docker buildx build --platform linux/arm64 -f Dockerfile.loadgen  -t ai-loadgen:latest --load .

# Kind cluster
kind create cluster --name aiinfer
kind load docker-image ai-infer:latest --name aiinfer
kind load docker-image ai-loadgen:latest --name aiinfer

# Metrics server (needed for HPA)
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/
helm upgrade --install metrics-server metrics-server/metrics-server \
  -n kube-system --create-namespace \
  --set args="{--kubelet-insecure-tls,--kubelet-preferred-address-types=InternalIP,Hostname,InternalDNS,ExternalDNS,ExternalIP}"

kubectl -n kube-system rollout status deploy/metrics-server
kubectl get apiservices | grep metrics
```

### Deploy (Option A: plain manifests)

```bash
kubectl apply -f k8s/ai-infer.yaml
kubectl get pods -w
```

### Deploy (Option B: Helm, local chart)

> Ensure `.helmignore` excludes `.venv/` and other large files. Then:

```bash
helm install ai-infer . \
  --set image.service=ai-infer:latest \
  --set image.loadgen=ai-loadgen:latest \
  --set serviceMonitor.enabled=false
```

### Test in-cluster

```bash
# In terminal A
kubectl port-forward svc/ai-infer-service 8080:8080

# In terminal B
curl -s localhost:8080/healthz
curl -s -X POST localhost:8080/predict -H 'content-type: application/json' \
  -d '{"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}'
```

### Trigger HPA scaling

```bash
# Bump load to trigger HPA
kubectl set env deploy/ai-loadgen QPS=300 CONCURRENCY=80

# Watch
kubectl get hpa -w
kubectl get pods -w
```

**Expected**: replicas increase from 1 → 2+ within ~1–2 minutes once metrics flow.

---

## Results

See [**RESULTS.md**](RESULTS.md) for the exact outputs captured during a run (HPA 1→5 replicas, pods, and API responses).

---

## Roadmap (CUDA/Triton/CUTLASS/vLLM)

- **CUDA kernel path**: Add a `CMakeLists.txt`/`setup.py` to compile a CUDA op (`.cu`), gated behind an env flag; expose via FastAPI like `/kernel_cuda_demo`.
- **CUTLASS**: Showcase a GEMM using CUTLASS (CUDA-only) with a small benchmark vs. PyTorch matmul.
- **Triton**: Add a simple Triton kernel (ReLU / GEMM tile), include a CPU fallback; enable with `TRITON_ENABLED=1`.
- **vLLM**: Add a second FastAPI route that proxies to vLLM for text generation (CPU install path for demo; GPU path documented). Include a `/vllm/healthz` and basic streaming.
- **Observability**: Optional ServiceMonitor (guarded by `values.yaml`) when Prometheus Operator is installed; add `/metrics` exporter to the service too.
- **CI**: GitHub Actions for docker builds and basic unit tests; kind e2e workflow (optional).

---

## License

MIT — see `LICENSE`.
