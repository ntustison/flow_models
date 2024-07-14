# Makefile for flow_models

# Environment variables AWS_ACCT_ID and AWS_REGION are expected to exist
# (check env vars before importing support makefile)
include awsbatch-support/makefile-awsbatchsupport.mk

# These are system commands used in macros below...
# (verify in a python environment to gate installation of packages)
check_venv := $(shell if [ -n "$$VIRTUAL_ENV" ]; then echo "1"; else echo "0"; fi)
# (verify in repo's root directory)
check_repo_root := $(shell if [ "$$(basename $$(pwd))" = "flow_models" ]; then echo "1"; else echo "0"; fi)
# (extract app version from setup.cfg for docker image labeling)
version := v$(shell grep -E 'current_version\s*=' setup.cfg | cut -d '=' -f2 | tr -d ' ')

create-env:
ifeq ($(check_repo_root), 1)
	@next_venv=$$(python3 -c "import os; max_val = max([int(d.replace('.venv', '')) for d in os.listdir('.') if d.startswith('.venv') and d.replace('.venv', '').isdigit()] + [0]); print(f'.venv{max_val+1}')"); \
	echo "Creating and installing new python environment ${PWD}/$$next_venv..."; \
	bash -c "python3 -m venv $$next_venv && source $${next_venv}/bin/activate && pip install -r requirements.txt"
else
	@echo "Not in root directory of flow_models repo."
endif

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
	# Note generally this macro would not be used at all, because it builds locally and
	# this is a super intentive install/build due to the gpu version of tensorflow.
	# This is why the CodeBuild-based build process was created; I mainly use that instead.
	# (e.g. this literally won't even build at all on my lower-end cpu-only instance, but
	# it builds just fine on my heavier, gpu-based instance.)
	docker build --build-arg TENSORFLOW_PKG=tensorflow==2.12.0 -t $(ECR_REPO):$(version)-gpu .

unittests:                                                            
	python -m unittest -v                                             
