# Use Bash because several recipes rely on Bash syntax.
SHELL := /bin/bash

# Fail a recipe when any command in a pipeline fails.
.SHELLFLAGS := -eu -o pipefail -c

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
VENV_UVICORN := $(VENV)/bin/uvicorn
VENV_PYTEST := $(VENV)/bin/pytest
VENV_RUFF := $(VENV)/bin/ruff

IMAGE_SERVICE ?= ai-infer:latest
IMAGE_LOADGEN ?= ai-loadgen:latest
KIND_CLUSTER ?= aiinfer

# The Helm chart is stored at the repository root.
CHART_DIR := .

.PHONY: help setup install-kernel test lint run \
        docker-build docker-run docker-stop \
        kind-create kind-load metrics-server \
        deploy deploy-manifest deploy-helm \
        load-test status clean down

help:
	@echo "Available targets:"
	@echo "  make setup            Create .venv, install dependencies, compile C++ extension"
	@echo "  make test             Run all Python tests"
	@echo "  make lint             Run Ruff checks"
	@echo "  make run              Start FastAPI locally on port 8000"
	@echo "  make docker-build     Build service and load-generator images"
	@echo "  make docker-run       Run the service image on localhost:8080"
	@echo "  make kind-create      Create the local Kind cluster"
	@echo "  make kind-load        Load local images into Kind"
	@echo "  make metrics-server   Install metrics-server for HPA"
	@echo "  make deploy-manifest  Deploy k8s/ai-infer.yaml"
	@echo "  make deploy-helm      Deploy the root Helm chart"
	@echo "  make load-test        Increase load-generator traffic"
	@echo "  make status           Display pods, services, deployments and HPA"
	@echo "  make down             Remove Helm release and Kind cluster"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel
	$(VENV_PYTHON) -m pip install \
		torch==2.8.0 \
		--index-url https://download.pytorch.org/whl/cpu
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt
	$(VENV_PYTHON) -m pip install ./service/kernels

install-kernel:
	$(VENV_PYTHON) -m pip install --force-reinstall ./service/kernels

test:
	$(VENV_PYTEST) -q --cov=service --cov-report=term-missing

lint:
	$(VENV_RUFF) check service tests

run:
	$(VENV_UVICORN) service.app:app --host 127.0.0.1 --port 8000 --reload

docker-build:
	docker build -f Dockerfile.service -t $(IMAGE_SERVICE) .
	docker build -f Dockerfile.loadgen -t $(IMAGE_LOADGEN) .

docker-run:
	docker run --rm \
		--name ai-infer-local \
		-p 8080:8080 \
		$(IMAGE_SERVICE)

docker-stop:
	docker rm -f ai-infer-local 2>/dev/null || true

kind-create:
	kind get clusters | grep -qx "$(KIND_CLUSTER)" || \
		kind create cluster --name $(KIND_CLUSTER)

kind-load:
	kind load docker-image $(IMAGE_SERVICE) --name $(KIND_CLUSTER)
	kind load docker-image $(IMAGE_LOADGEN) --name $(KIND_CLUSTER)

metrics-server:
	helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/ --force-update
	helm upgrade --install metrics-server metrics-server/metrics-server \
		--namespace kube-system \
		--create-namespace \
		--set 'args={--kubelet-insecure-tls,--kubelet-preferred-address-types=InternalIP,Hostname,InternalDNS,ExternalDNS,ExternalIP}'
	kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s

deploy: docker-build kind-create kind-load metrics-server deploy-helm

deploy-manifest:
	kubectl apply -f k8s/ai-infer.yaml
	kubectl rollout status deployment/ai-infer --timeout=180s
	kubectl rollout status deployment/ai-loadgen --timeout=180s

deploy-helm:
	helm upgrade --install ai-infer $(CHART_DIR) \
		--set image.service=$(IMAGE_SERVICE) \
		--set image.loadgen=$(IMAGE_LOADGEN) \
		--set serviceMonitor.enabled=false
	kubectl rollout status deployment/ai-infer --timeout=180s
	kubectl rollout status deployment/ai-loadgen --timeout=180s

load-test:
	kubectl set env deployment/ai-loadgen \
		QPS=100 \
		CONCURRENCY=20 \
		DURATION_SECONDS=0
	kubectl rollout status deployment/ai-loadgen --timeout=180s

status:
	kubectl get deployments
	kubectl get pods -o wide
	kubectl get services
	kubectl get hpa

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

down:
	helm uninstall ai-infer 2>/dev/null || true
	kubectl delete -f k8s/ai-infer.yaml --ignore-not-found 2>/dev/null || true
	kind delete cluster --name $(KIND_CLUSTER) 2>/dev/null || true
