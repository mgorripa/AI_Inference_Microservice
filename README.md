# Overnight AI Inference Microservice

**Built after reading Terry’s hiring post** — an end‑to‑end demo of AI infra:
- PyTorch + FastAPI model serving
- Custom C++/pybind11 op (CPU today; CUDA path sketched)
- Optional Triton kernel + vLLM route
- Docker + Kubernetes via Helm
- HPA scaling under Go loadgen
- Prometheus metrics
- Open source (MIT)

## Quickstart (CPU‑only)
```bash
make dev # (optional) local venv
make build # build Docker images
make kind # create kind cluster & load images
make helm-install # deploy service + loadgen + HPA
kubectl get pods
kubectl get hpa
kubectl port-forward svc/ai-infer-service 8080:8080
curl -X POST localhost:8080/predict -H 'content-type: application/json' -d '{"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}'