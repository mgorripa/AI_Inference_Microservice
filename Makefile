.PHONY: dev build push kind up down helm-install helm-upgrade helm-uninstall loadgen

IMAGE_SERVICE ?= ghcr.io/youruser/ai-infer-service:latest
IMAGE_LOADGEN ?= ghcr.io/youruser/ai-loadgen:latest
KIND_CLUSTER  ?= aiinfer
CHART_DIR     = deploy/charts/ai-infer

dev:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip wheel build
	. .venv/bin/activate && pip install -r service/requirements.txt

build:
	docker build -f Dockerfile.service -t $(IMAGE_SERVICE) .
	docker build -f Dockerfile.loadgen -t $(IMAGE_LOADGEN) .

push:
	docker push $(IMAGE_SERVICE)
	docker push $(IMAGE_LOADGEN)

kind:
	kind create cluster --name $(KIND_CLUSTER) || true
	# load local images into kind if not pushing to a registry
	kind load docker-image $(IMAGE_SERVICE) --name $(KIND_CLUSTER)
	kind load docker-image $(IMAGE_LOADGEN) --name $(KIND_CLUSTER)

helm-install:
	helm install ai-infer $(CHART_DIR) \
		--set image.service=$(IMAGE_SERVICE) \
		--set image.loadgen=$(IMAGE_LOADGEN)

helm-upgrade:
	helm upgrade ai-infer $(CHART_DIR) \
		--set image.service=$(IMAGE_SERVICE) \
		--set image.loadgen=$(IMAGE_LOADGEN)

helm-uninstall:
	helm uninstall ai-infer || true

up: build kind helm-install

down:
	helm-uninstall ; kind delete cluster --name $(KIND_CLUSTER) || true
