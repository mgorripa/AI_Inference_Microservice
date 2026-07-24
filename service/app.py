
from __future__ import annotations

import math
import time
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from service.model import predict

# The current native extension is a CPU C++/pybind11 extension.
# It is not a CUDA implementation.
try:
    from kernels import binding as kbinding

    KERNEL_AVAILABLE = True
    KERNEL_BACKEND = "cpp_cpu"
    KERNEL_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - behavior depends on local build environment
    kbinding = None
    KERNEL_AVAILABLE = False
    KERNEL_BACKEND = "numpy_fallback"
    KERNEL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


APP_NAME = "AI Inference Microservice"
APP_VERSION = "1.0.0"
EXPECTED_INPUT_SIZE = 16
MAX_KERNEL_INPUT_SIZE = 100_000

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="CPU-based PyTorch inference service with a C++/NumPy ReLU demo.",
)


# -----------------------------
# Prometheus metrics
# -----------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests processed.",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Total HTTP request duration in seconds.",
    ["method", "endpoint"],
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of model inference requests.",
    ["result"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "inference_duration_seconds",
    "Time spent only inside model inference.",
)

KERNEL_REQUESTS_TOTAL = Counter(
    "kernel_requests_total",
    "Total number of custom-kernel demo requests.",
    ["backend", "result"],
)

KERNEL_DURATION_SECONDS = Histogram(
    "kernel_duration_seconds",
    "Time spent applying ReLU in the selected backend.",
    ["backend"],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of HTTP requests currently being processed.",
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model module loaded successfully: 1 for yes, 0 for no.",
)

KERNEL_BACKEND_INFO = Gauge(
    "kernel_backend_info",
    "Information metric identifying the active kernel backend.",
    ["backend"],
)

# Importing service.model and predict above succeeded, so the model is available.
MODEL_LOADED.set(1)

# Publish exactly one active backend label.
KERNEL_BACKEND_INFO.labels(backend=KERNEL_BACKEND).set(1)


# -----------------------------
# Request and response schemas
# -----------------------------

FiniteFloat = Annotated[float, Field(strict=False)]


class PredictIn(BaseModel):
    """Input schema for the /predict endpoint."""

    # Reject unexpected JSON fields instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid")

    # The existing model expects exactly 16 numeric features.
    x: list[FiniteFloat] = Field(
        ...,
        min_length=EXPECTED_INPUT_SIZE,
        max_length=EXPECTED_INPUT_SIZE,
        description="Exactly 16 finite numeric input features.",
    )

    @field_validator("x")
    @classmethod
    def reject_nan_and_infinity(cls, values: list[float]) -> list[float]:
        """Reject NaN and infinity because they can corrupt model output."""
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("All input features must be finite numbers.")
        return [float(value) for value in values]


class PredictOut(BaseModel):
    """Response schema for the /predict endpoint."""

    probabilities: list[float]
    predicted_class: int
    device: str


class KernelIn(BaseModel):
    """Input schema for the /kernel_demo endpoint."""

    model_config = ConfigDict(extra="forbid")

    data: list[FiniteFloat] = Field(
        ...,
        min_length=1,
        max_length=MAX_KERNEL_INPUT_SIZE,
        description="One or more finite values on which ReLU is applied.",
    )

    @field_validator("data")
    @classmethod
    def reject_nan_and_infinity(cls, values: list[float]) -> list[float]:
        """Reject NaN and infinity before sending data to C++ or NumPy."""
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("All kernel input values must be finite numbers.")
        return [float(value) for value in values]


class KernelOut(BaseModel):
    """Response schema for the /kernel_demo endpoint."""

    backend: str
    output: list[float]


# -----------------------------
# Middleware
# -----------------------------

@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    """Measure every HTTP request, including requests that fail validation."""
    endpoint = request.url.path
    method = request.method
    started_at = time.perf_counter()

    ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        return response
    finally:
        # If an unexpected failure occurs before a response exists, record 500.
        status_code = str(locals().get("response").status_code if "response" in locals() else 500)

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            endpoint=endpoint,
        ).observe(time.perf_counter() - started_at)

        ACTIVE_REQUESTS.dec()


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def index() -> dict[str, object]:
    """List the useful routes for a developer opening the service root."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "device": "cpu",
        "routes": ["/healthz", "/predict", "/kernel_demo", "/metrics", "/docs"],
    }


@app.get("/healthz")
def healthz() -> dict[str, object]:
    """Return operational status without implying that CUDA is in use."""
    response: dict[str, object] = {
        "status": "healthy",
        "model_loaded": True,
        "device": "cpu",
        "kernel_backend": KERNEL_BACKEND,
        "version": APP_VERSION,
    }

    # Show the native-extension import error only when the fallback is active.
    # This is useful during laptop debugging.
    if KERNEL_IMPORT_ERROR is not None:
        response["kernel_import_error"] = KERNEL_IMPORT_ERROR

    return response


@app.get("/metrics")
def metrics() -> Response:
    """Expose all Prometheus metrics in the standard text format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/predict",
    response_model=PredictOut,
    status_code=status.HTTP_200_OK,
)
def do_predict(inp: PredictIn) -> PredictOut:
    """Run one prediction through the existing TinyMLP model."""
    try:
        # Convert the validated 16-value list into a single model input row.
        model_input = torch.tensor(
            inp.x,
            dtype=torch.float32,
            device="cpu",
        ).reshape(1, EXPECTED_INPUT_SIZE)

        inference_started_at = time.perf_counter()

        # service.model.predict is expected to perform inference without gradients.
        logits = predict(model_input)

        INFERENCE_DURATION_SECONDS.observe(
            time.perf_counter() - inference_started_at
        )

        # Convert two model logits into probabilities that sum to approximately 1.
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        # Ensure the returned tensor has the expected shape.
        if probabilities.ndim != 1 or probabilities.numel() != 2:
            raise RuntimeError(
                "The model must return two output logits for one input row."
            )

        probability_list = [float(value) for value in probabilities.tolist()]
        predicted_class = int(torch.argmax(probabilities).item())

        INFERENCE_REQUESTS_TOTAL.labels(result="success").inc()

        return PredictOut(
            probabilities=probability_list,
            predicted_class=predicted_class,
            device="cpu",
        )

    except HTTPException:
        raise
    except Exception as exc:
        INFERENCE_REQUESTS_TOTAL.labels(result="error").inc()

        # Do not expose a raw Python stack trace to the API client.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {type(exc).__name__}",
        ) from exc


@app.post(
    "/kernel_demo",
    response_model=KernelOut,
    status_code=status.HTTP_200_OK,
)
def kernel_demo(inp: KernelIn) -> KernelOut:
    """Apply ReLU with the C++ CPU extension or the explicit NumPy fallback."""
    values = np.asarray(inp.data, dtype=np.float32)

    started_at = time.perf_counter()

    try:
        if KERNEL_AVAILABLE:
            # The extension is a CPU C++ implementation exposed with pybind11.
            output = kbinding.vec_relu(values)
        else:
            # This fallback keeps the endpoint usable when the extension is absent.
            output = np.maximum(values, 0)

        KERNEL_DURATION_SECONDS.labels(
            backend=KERNEL_BACKEND
        ).observe(time.perf_counter() - started_at)

        KERNEL_REQUESTS_TOTAL.labels(
            backend=KERNEL_BACKEND,
            result="success",
        ).inc()

        return KernelOut(
            backend=KERNEL_BACKEND,
            output=np.asarray(output, dtype=np.float32).tolist(),
        )

    except Exception as exc:
        KERNEL_REQUESTS_TOTAL.labels(
            backend=KERNEL_BACKEND,
            result="error",
        ).inc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kernel execution failed: {type(exc).__name__}",
        ) from exc
