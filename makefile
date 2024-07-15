# Makefile for flow_models

# Environment variables AWS_ACCT_ID and AWS_REGION are expected to exist
# (check env vars before importing support makefile)
include awsbatch-support/makefile-awsbatchsupport.mk

# These are system commands used in macros below.
# (verify in a python environment to gate installation of packages)
check_venv := $(shell if [ -n "$$VIRTUAL_ENV" ]; then echo "1"; else echo "0"; fi)
# (verify in repo's root directory)
check_repo_root := $(shell if [ "$$(basename $$(pwd))" = "flow_models" ]; then echo "1"; else echo "0"; fi)
# (extract app version from setup.cfg for docker image labeling)
version := v$(shell grep -E 'current_version\s*=' setup.cfg | cut -d '=' -f2 | tr -d ' ')

create-env:
ifeq ($(check_repo_root), 1)
	@next_venv=$$(python3 -c "import os; max_val = max([int(d.replace('.venv', '')) for d in os.listdir('.') if d.startswith('.venv') and d.replace('.venv', '').isdigit()] + [0]); print(f'.venv{max_val+1}')"); \
	echo "Creating/installing new python env ${PWD}/$$next_venv"; \
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

unittests:                                                            
	python -m unittest -v                                             

build-cpu:
	# For CPU package of TF for dev/testing on local instance - for testing
	docker build --build-arg TENSORFLOW_PKG=tensorflow-cpu==2.12.0 -t $(ECR_REPO):$(version)-cpu .

run-local:
	# Run/test the batch job on AWS local CPU-only instance - for testing
	docker run --rm -it flow_models:$(version)-cpu

build-and-push-local-image: build-gpu push-to-ecr

define-ecr-repo: create-ecr-repo list-ecr-repos  # very rarely

define-roles: create-codebuild-role create-batch-role list-roles  # very rarely

define-the-compute: create-compute-env create-job-queue register-job-definition  # rarely

build-image-and-run-job: create-project run-batch  # regularly

# Occasional commands to run manually:
# check-job-status, cancel-job
# list-roles
# delete-roles


# ensures all entries run every time since these aren't files
.PHONY: create-env install-dev unittests build-cpu run-local \
	build-and-push-local-image define-ecr-repo-and-role \
	define-the-compute build-image-and-run-job
