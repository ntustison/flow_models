# Makefile for flow_models: build docker image for remotely training/logging model

#AWS_ACCT_ID and AWS_REGION are environment variables
ECR_REPO = flow_models
ECR_REPO_URI = ${AWS_ACCT_ID}.dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO)

# Verify we're in a python environment to gate installation of packages
check_venv := $(shell if [ -n "$$VIRTUAL_ENV" ]; then echo "1"; else echo "0"; fi)
# Extract app version from setup.cfg for docker image labeling
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
	# For GPU package of TF for runs on GPU-based instance
	# docker build --build-arg TENSORFLOW_PKG=tensorflow==2.11.0 -t $(ECR_REPO):$(version)-gpu .
	docker build --build-arg TENSORFLOW_PKG=tensorflow==2.12.0 -t $(ECR_REPO):$(version)-gpu .

create-ecr-repo:
	# Create the repo in ECR where this app's docker images will be held on AWS
	aws ecr create-repository --repository-name $(ECR_REPO) --region $(AWS_REGION)

list-ecr-repos:
	# List the repo in ECR to confirm this one is there after create-ecr-repo
	aws ecr describe-repositories

push-to-ecr:
	# Push docker image to AWS ECR
ifndef MODE
	@echo "This makefile macro must be called as:"                                       
	@echo "  make push-to-ecr MODE=GPU   # or MODE=CPU"
	@echo "The MODE determines whether the CPU or GPU version of the app image get pushed to ECR."
	@echo                                                                                
endif                                                                                    
	docker tag $(ECR_REPO):$(version)-$${MODE} $(ECR_REPO_URI):latest  # tag the cpu or gpu image as 'latest'
	aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $(ECR_REPO_URI)  # login to ECR
	docker push $(ECR_REPO_URI):latest  # push the image

create-compute-env:
	# Create compute environment
	aws batch create-compute-environment --compute-environment-name GPUEnvironment --type MANAGED --compute-resources type=EC2,allocationStrategy=BEST_FIT_PROGRESSIVE,minvCpus=0,maxvCpus=16,desiredvCpus=0,instanceTypes=p2,p3,g4,subnets=subnet-12345,securityGroupIds=sg-12345 --service-role arn:aws:iam::$(AWS_ACCT_ID):role/AWSBatchServiceRole

create-job-queue:
	# Create job queue
	aws batch create-job-queue --job-queue-name GPUJobQueue --compute-environment-order order=1,computeEnvironment=GPUEnvironment --priority 1

register-job-definition:
	# Register job definition
	aws batch register-job-definition --cli-input-json file://job_definition.json

run-batch: create-compute-env create-job-queue register-job-definition
	# Run the batch job on AWS remote GPU instance
	aws batch submit-job --job-name MyGPUJob --job-queue GPUJobQueue --job-definition GPUJobDefinition
	# Command returns immediately and provides a job-id, to enter for check-job-status and cancel-job macros.
	# Job log output is found in the AWS CloudWatch Console:
	#     https://us-west-2.console.aws.amazon.com/cloudwatch/home?region=us-west-2

check-job-status:
	# Check status of a run-batch job that's still in progress, based on JOBID from run-batch
ifndef JOBID
	@echo "This makefile macro must be called as:"                                       
	@echo "  make check-job-status JOBID=12345678  # comes from output of `make run-batch`"
	@echo                                                                                
endif                                                                                    
	aws batch describe-jobs --jobs $${JOBID}

cancel-job:
	# Cancel a run-batch job that's still in progress, based on JOBID from run-batch
ifndef JOBID
	@echo "This makefile macro must be called as:"                                       
	@echo "  make cancel-job JOBID=12345678  # comes from output of `make run-batch`"
	@echo                                                                                
endif                                                                                    
	aws batch cancel-job --job-id $${JOBID} --reason "Cancelling job"

