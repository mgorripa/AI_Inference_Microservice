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
AI_Inference_Microservice/
│
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml
│       ├── docker.yml
│       └── gpu-ci.yml
│
├── benchmarks/
│   ├── __init__.py
│   ├── README.md
│   ├── common.py
│   ├── benchmark_api.py
│   ├── benchmark_model.py
│   ├── benchmark_batching.py
│   ├── benchmark_devices.py
│   ├── benchmark_kernel.py
│   ├── benchmark_quantization.py
│   ├── analyze_results.py
│   ├── schemas.py
│   └── results/
│       └── .gitkeep
│
├── configs/
│   ├── development.env
│   ├── production.env
│   └── benchmark.env
│
├── docs/
│   ├── architecture.md
│   ├── benchmarking.md
│   ├── deployment.md
│   ├── design-decisions.md
│   ├── gpu-roadmap.md
│   └── images/
│       └── architecture.svg
│
├── go-loadgen/
│   ├── cmd/
│   │   └── loadgen/
│   │       └── main.go
│   ├── internal/
│   │   ├── client/
│   │   │   ├── client.go
│   │   │   └── client_test.go
│   │   ├── config/
│   │   │   ├── config.go
│   │   │   └── config_test.go
│   │   ├── load/
│   │   │   ├── runner.go
│   │   │   └── runner_test.go
│   │   ├── metrics/
│   │   │   └── metrics.go
│   │   ├── payload/
│   │   │   ├── generator.go
│   │   │   └── generator_test.go
│   │   └── report/
│   │       ├── report.go
│   │       └── report_test.go
│   ├── go.mod
│   ├── go.sum
│   └── README.md
│
├── gpu/
│   ├── README.md
│   ├── cuda/
│   │   ├── CMakeLists.txt
│   │   ├── bindings.cpp
│   │   ├── bias_gelu.cu
│   │   ├── bias_gelu.cuh
│   │   └── tests/
│   │       └── test_cuda_kernel.py
│   ├── triton/
│   │   ├── __init__.py
│   │   ├── bias_gelu.py
│   │   └── test_bias_gelu.py
│   └── cutlass/
│       ├── README.md
│       └── gemm_example.cu
│
├── k8s/
│   ├── base/
│   │   ├── configmap.yaml
│   │   ├── deployment.yaml
│   │   ├── hpa.yaml
│   │   ├── service.yaml
│   │   ├── serviceaccount.yaml
│   │   └── kustomization.yaml
│   ├── overlays/
│   │   ├── local/
│   │   │   ├── kustomization.yaml
│   │   │   └── patch-local.yaml
│   │   └── gpu/
│   │       ├── kustomization.yaml
│   │       └── patch-gpu.yaml
│   └── monitoring/
│       ├── servicemonitor.yaml
│       └── prometheus-rules.yaml
│
├── profiling/
│   ├── README.md
│   ├── profile_model.py
│   ├── profile_service.py
│   ├── profile_batching.py
│   └── traces/
│       └── .gitkeep
│
├── scripts/
│   ├── wait_for_service.sh
│   ├── smoke_test.sh
│   ├── run_benchmarks.sh
│   ├── collect_system_info.py
│   ├── create_kind_cluster.sh
│   └── validate_results.py
│
├── service/
│   ├── __init__.py
│   ├── app.py
│   ├── lifecycle.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   ├── error_handlers.py
│   │   ├── middleware.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── inference.py
│   │       ├── kernel.py
│   │       ├── metadata.py
│   │       └── metrics.py
│   │
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── pytorch_backend.py
│   │   ├── cpu_backend.py
│   │   ├── mps_backend.py
│   │   ├── cuda_backend.py
│   │   └── quantized_cpu_backend.py
│   │
│   ├── batching/
│   │   ├── __init__.py
│   │   ├── batcher.py
│   │   ├── policies.py
│   │   ├── queue.py
│   │   └── request_item.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   │
│   ├── kernels/
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── src/
│   │   │   ├── bindings.cpp
│   │   │   ├── relu.cpp
│   │   │   ├── relu.hpp
│   │   │   ├── bias_gelu.cpp
│   │   │   └── bias_gelu.hpp
│   │   └── tests/
│   │       └── test_cpp_extension.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── prometheus.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── loader.py
│   │   ├── metadata.py
│   │   ├── schemas.py
│   │   └── distilbert.py
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── request_context.py
│   │   └── timing.py
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   └── limits.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── device.py
│       ├── json_safe.py
│       └── system_info.py
│
├── tests/
│   ├── conftest.py
│   ├── integration/
│   │   ├── test_api.py
│   │   ├── test_batching_api.py
│   │   ├── test_concurrency.py
│   │   └── test_metrics.py
│   ├── unit/
│   │   ├── test_backends.py
│   │   ├── test_batcher.py
│   │   ├── test_config.py
│   │   ├── test_device.py
│   │   ├── test_error_handlers.py
│   │   ├── test_json_safe.py
│   │   ├── test_model.py
│   │   └── test_schemas.py
│   └── e2e/
│       ├── test_docker.py
│       └── test_kubernetes.py
│
├── charts/
│   └── ai-infer/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── _helpers.tpl
│           ├── configmap.yaml
│           ├── deployment.yaml
│           ├── hpa.yaml
│           ├── service.yaml
│           ├── serviceaccount.yaml
│           └── servicemonitor.yaml
│
├── .dockerignore
├── .editorconfig
├── .env.example
├── .gitignore
├── .helmignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile.loadgen
├── Dockerfile.service
├── LICENSE
├── Makefile
├── README.md
├── RESULTS.md
├── RUNBOOK.md
├── SECURITY.md
├── pyproject.toml
├── requirements-dev.txt
└── requirements.txt
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
