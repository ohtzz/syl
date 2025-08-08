DOCKER_HUB_USER := ohtz
REPO := syl
TAG := latest
PLATFORMS := linux/amd64,linux/arm64


.PHONY: format
format:
	ruff format . --exclude syl/cli/syl.py

.PHONY: test-pypi
test-pypi:
	rm -rf dist/ build/ *.egg-info/
	python -m build
	python -m twine upload --repository testpypi dist/*

.PHONY: release-pypi
release-pypi:
	rm -rf dist/ build/ *.egg-info/
	python -m build
	python -m twine upload --repository pypi dist/*

.PHONY: build-test
build-normal:
	docker build -f ./docker/base/Dockerfile -t ohtz/syl-test:base-latest .
	docker build --no-cache -f ./docker/server/Dockerfile -t ohtz/syl-test:server-latest .
	docker build --no-cache -f ./docker/index/Dockerfile -t ohtz/syl-test:index-latest .
	docker build --no-cache -f ./docker/watcher/Dockerfile -t ohtz/syl-test:watcher-latest .

.PHONY: build-base
build-base:
	@echo "Building base image with all ML dependencies..."
	@docker buildx create --name multiplatform --driver docker-container --use 2>/dev/null || docker buildx use multiplatform
	@docker buildx inspect --bootstrap >/dev/null 2>&1
	docker buildx build \
		--platform $(PLATFORMS) \
		-f ./docker/base/Dockerfile \
		-t $(DOCKER_HUB_USER)/$(REPO):base-latest \
		--push \
		.

.PHONY: build-service-%
build-service-%:
	@echo "Building $* service image..."
	docker buildx build \
		--platform $(PLATFORMS) \
		--no-cache \
		-f ./docker/$*/Dockerfile \
		-t $(DOCKER_HUB_USER)/$(REPO):$*-latest \
		--push \
		.

.PHONY: build-services
build-services: build-service-index build-service-server build-service-watcher

.PHONY: build-all
build-all: build-base build-services

.PHONY: build-base-local
build-base-local:
	docker buildx build \
		--platform $(shell docker version --format '{{.Server.Os}}/{{.Server.Arch}}') \
		--no-cache \
		-f ./docker/base/Dockerfile \
		-t $(DOCKER_HUB_USER)/$(REPO):base-latest \
		--load \
		.

.PHONY: build-services-local
build-services-local: build-service-index-local build-service-server-local build-service-watcher-local
	-

.PHONY: build-service-%-local
build-service-%-local: build-base-local
	docker buildx build \
		--platform $(shell docker version --format '{{.Server.Os}}/{{.Server.Arch}}') \
		--no-cache \
		-f ./docker/$*/Dockerfile \
		-t $(DOCKER_HUB_USER)/$(REPO):$*-latest \
		--load \
		.

.PHONY: test-index
test-index: build-service-index-local
	docker run --rm -p 8000:8000 $(DOCKER_HUB_USER)/$(REPO):index-latest

.PHONY: test-server
test-server: build-service-server-local
	docker run --rm -p 8001:8001 -p 9000:9000 $(DOCKER_HUB_USER)/$(REPO):server-latest

.PHONY: test-watcher
test-watcher: build-service-watcher-local
	docker run --rm $(DOCKER_HUB_USER)/$(REPO):watcher-latest

.PHONY: quick-build
quick-build: build-services
