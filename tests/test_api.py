"""API tests for the FastAPI inference service."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from service.app import EXPECTED_INPUT_SIZE, app

client = TestClient(app)


def valid_input() -> list[float]:
    """Return one valid 16-feature request payload."""
    return [float(index) for index in range(EXPECTED_INPUT_SIZE)]


def test_index_returns_service_information() -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()

    assert body["name"] == "AI Inference Microservice"
    assert body["device"] == "cpu"
    assert "/predict" in body["routes"]


def test_healthz_returns_cpu_status() -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"
    assert body["kernel_backend"] in {"cpp_cpu", "numpy_fallback"}


def test_predict_accepts_exactly_16_numbers() -> None:
    response = client.post("/predict", json={"x": valid_input()})

    assert response.status_code == 200
    body = response.json()

    assert len(body["probabilities"]) == 2
    assert body["predicted_class"] in {0, 1}
    assert body["device"] == "cpu"

    assert math.isclose(
        sum(body["probabilities"]),
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    )


def test_predict_rejects_too_few_values() -> None:
    response = client.post("/predict", json={"x": [1.0, 2.0]})

    assert response.status_code == 422


def test_predict_rejects_too_many_values() -> None:
    response = client.post(
        "/predict",
        json={"x": [0.0] * (EXPECTED_INPUT_SIZE + 1)},
    )

    assert response.status_code == 422


def test_predict_rejects_non_numeric_value() -> None:
    values: list[object] = valid_input()
    values[0] = "not-a-number"

    response = client.post("/predict", json={"x": values})

    assert response.status_code == 422


def test_predict_rejects_nan() -> None:
    # JSON does not officially support NaN, so send raw content to exercise validation.
    response = client.post(
        "/predict",
        content='{"x":[NaN,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code in {400, 422}


def test_predict_rejects_extra_fields() -> None:
    response = client.post(
        "/predict",
        json={"x": valid_input(), "unexpected": True},
    )

    assert response.status_code == 422


def test_kernel_demo_applies_relu() -> None:
    response = client.post(
        "/kernel_demo",
        json={"data": [-3.0, -0.5, 0.0, 2.0, 8.0]},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["backend"] in {"cpp_cpu", "numpy_fallback"}
    assert body["output"] == [0.0, 0.0, 0.0, 2.0, 8.0]


def test_kernel_demo_rejects_empty_input() -> None:
    response = client.post("/kernel_demo", json={"data": []})

    assert response.status_code == 422


def test_metrics_endpoint_returns_prometheus_text() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text
    assert "inference_requests_total" in response.text
