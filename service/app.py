#FastAPI + metrics + CUDA fallback
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import numpy as np
import torch
from service.model import predict

# Try to import custom kernel extension
try:
    from kernels import binding as kbinding
    KERNEL_AVAILABLE = True
except Exception as e:
    KERNEL_AVAILABLE = False

app = FastAPI(title="AI Inference Microservice")

# Prometheus metrics
REQS = Counter("requests_total", "Total requests", ["endpoint"])
LAT = Histogram("request_latency_seconds", "Latency", ["endpoint"])


class PredictIn(BaseModel):
    x: list[float]  # Input vector of length 16


@app.get("/healthz")
def healthz():
    """Liveness/Readiness endpoint."""
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "kernel": KERNEL_AVAILABLE,
    }


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def do_predict(inp: PredictIn):
    """Run inference through the TinyMLP model."""
    REQS.labels("predict").inc()
    with LAT.labels("predict").time():
        x = torch.tensor(inp.x, dtype=torch.float32).view(1, -1)
        y = predict(x)
        probs = torch.softmax(y, dim=-1).squeeze().tolist()
        return {"probs": probs}


class KernelIn(BaseModel):
    data: list[float]


@app.post("/kernel_demo")
def kernel_demo(inp: KernelIn):
    """Demonstrate using a custom CUDA/C++ kernel (or fallback to numpy)."""
    REQS.labels("kernel_demo").inc()
    with LAT.labels("kernel_demo").time():
        arr = np.array(inp.data, dtype=np.float32)
        if KERNEL_AVAILABLE:
            out = kbinding.vec_relu(arr)
            mode = "cuda_or_cpp"
        else:
            out = np.maximum(arr, 0)
            mode = "numpy_fallback"
        return {"mode": mode, "out": out.tolist()}

@app.get("/")
def index():
    return {"ok": True, "try": ["/healthz", "/predict", "/kernel_demo"]}
