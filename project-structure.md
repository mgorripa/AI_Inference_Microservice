# Project Structure

This document describes the directory layout for **ai-infer-microservice**.

```text
ai-infer-microservice/
  LICENSE
  README.md
  Makefile
  Dockerfile.service
  Dockerfile.loadgen
  .github/workflows/ci.yaml
  diagrams/
    arch.drawio
  service/
    app.py
    model.py
    requirements.txt
    triton_kernel.py  # optional
    vllm_stub.py      # optional
    kernels/
      setup.py        # builds pybind11 extension (optional)
      binding.cpp
      kernel.cu
  go-loadgen/
    main.go
    go.mod
  deploy/
    charts/ai-infer/
      Chart.yaml
      values.yaml
      templates/
        deployment.yaml
        service.yaml
        hpa.yaml
        servicemonitor.yaml  # optional (Prometheus Operator)
```