# Makefile for flow_models: build docker image for remotely training/logging model

# Environment variables AWS_ACCT_ID and AWS_REGION are expected to exist
# (check env vars before importing support makefile)
include awsbatch-support/makefile-support.mk

# These two are system commands called in install-dev below...
# (verify in a python environment to gate installation of packages)
check_venv := $(shell if [ -n "$$VIRTUAL_ENV" ]; then echo "1"; else echo "0"; fi)
# (extract app version from setup.cfg for docker image labeling)
version := v$(shell grep -E 'current_version\s*=' setup.cfg | cut -d '=' -f2 | tr -d ' ')


install-dev:
ifeq ($(check_venv), 1)
	@echo "Installing dev packages with pip..."
	pip install -r requirements-dev.txt
else
	@echo "Not in a python virtual environment. Skipping pip install of dev packages."
endif

build-cpu:
	# For CPU package of TF for dev/testing on local instance
	# docker build --build-arg TENSORFLOW_PKG=tensorflow-cpu==2.11.0 -t $(ECR_REPO):$(version)-cpu .
	docker build --build-arg TENSORFLOW_PKG=tensorflow-cpu==2.12.0 -t $(ECR_REPO):$(version)-cpu .

run-local:
	# Run/test the batch job on AWS local CPU-only instance
	docker run --rm -it flow_models:$(version)-cpu


build-gpu:
	# Note generally this macro wouldn't be used at all, because this builds locally.
	# For GPU package of TF for runs on GPU-based instance
	# (fyi won't even build on my low-end cpu-only instance, so need some way to handle that)
	docker build --build-arg TENSORFLOW_PKG=tensorflow==2.12.0 -t $(ECR_REPO):$(version)-gpu .

