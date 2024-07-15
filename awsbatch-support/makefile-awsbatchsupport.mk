ECR_REPO = flow_models
ECR_REPO_URI = ${AWS_ACCT_ID}.dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO)

list-ecr-repos:
	# List the repo in ECR to confirm this one is there after create-ecr-repo
	aws ecr describe-repositories --no-cli-pager

create-ecr-repo:
	# Create the repo in ECR where this app's docker images will be held on AWS
	aws ecr create-repository --repository-name $(ECR_REPO) --region $(AWS_REGION)



create-codebuild-role: check-codebuild-role-exists
	# generates CODEBUILD_SERVICE_ROLE_ARN needed for create-project.
	# read it from output and put into CODEBUILD_SERVICE_ROLE_ARN env var.
	@echo "Creating role CodeBuildServiceRole..."
	@aws iam create-role \
		--role-name CodeBuildServiceRole --no-cli-pager \
		--assume-role-policy-document file://awsbatch-support/codebuild-trust-policy.json
	@echo "Role CodeBuildServiceRole created successfully."
	@aws iam attach-role-policy \
		--role-name CodeBuildServiceRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
	@echo "AmazonEC2ContainerRegistryPowerUser successfully attached to role CodeBuildServiceRole."
	@aws iam attach-role-policy \
	    --role-name CodeBuildServiceRole \
		--policy-arn arn:aws:iam::aws:policy/AWSCodeBuildAdminAccess
	@echo "AWSCodeBuildAdminAccess successfully attached to role CodeBuildServiceRole."

check-codebuild-role-exists:
	# supports create-role with a verification check
	@aws iam get-role --role-name CodeBuildServiceRole > /dev/null 2>&1; \
	if [ $$? -eq 0 ]; then \
		echo "Role CodeBuildServiceRole already exists. Skipping creation."; \
		exit 1; \
	fi



create-batch-role: check-batch-role-exists
	# generates AWSBATCH_SERVICE_ROLE_ARN needed for create-compute-environment.
	# read it from output and put into AWSBATCH_SERVICE_ROLE_ARN env var.
	@echo "Creating role AWSBatchServiceRole..."
	@aws iam create-role \
		--role-name AWSBatchServiceRole --no-cli-pager \
		--assume-role-policy-document file://awsbatch-support/batch-trust-policy.json
	@echo "Role AWSBatchServiceRole created successfully."
	@aws iam attach-role-policy \
		--role-name AWSBatchServiceRole \
		--policy-arn arn:aws:iam::aws:policy/AWSBatchFullAccess
	@echo "AWSBatchFullAccess successfully attached to role AWSBatchServiceRole."
	# TODO: Check if a policy with less access than AWSBatchFullAccess would work.


check-batch-role-exists:
	# supports create-role with a verification check
	@aws iam get-role --role-name AWSBatchServiceRole > /dev/null 2>&1; \
	if [ $$? -eq 0 ]; then \
		echo "Role AWSBatchServiceRole already exists. Skipping creation."; \
		exit 1; \
	fi



create-compute-env:
	# Create compute environment
	aws batch create-compute-environment --compute-environment-name GPUEnvironment --type MANAGED \
		--compute-resources type=EC2,allocationStrategy=BEST_FIT_PROGRESSIVE,minvCpus=4,maxvCpus=4,desiredvCpus=4,instanceTypes=g4dn.xlarge,subnets=subnet-12345,securityGroupIds=sg-12345 \
		--service-role arn:aws:iam::$(AWS_ACCT_ID):role/AWSBatchServiceRole

	# to set up to use much-cheaper spot instances later:
	# aws batch create-compute-environment --compute-environment-name GPUEnvironment --type MANAGED \
	#	--compute-resources type=SPOT,allocationStrategy=SPOT_CAPACITY_OPTIMIZED,minvCpus=4,maxvCpus=4,desiredvCpus=4,instanceTypes=g4dn.xlarge,subnets=subnet-12345,securityGroupIds=sg-12345,spotIamFleetRole=arn:aws:iam::$(AWS_ACCT_ID):role/AWSBatchServiceRole \
	#	--service-role arn:aws:iam::$(AWS_ACCT_ID):role/AWSBatchServiceRole

create-job-queue:
	# Create job queue
	aws batch create-job-queue --job-queue-name GPUJobQueue --compute-environment-order order=1,computeEnvironment=GPUEnvironment --priority 1

register-job-definition:
	# Register job definition
	aws batch register-job-definition --cli-input-json file://awsbatch-support/job_definition.json



create-project: check-service-role
	# gets code from github branch, builds docker image, and pushes it to ECR repo
	aws codebuild create-project \
	    --name "ExampleProject" \
	    --source "type=GITHUB,location=https://github.com/username/repository.git,buildSpec=buildspec.yml" \
	    --artifacts "type=NO_ARTIFACTS" \
	    --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:4.0,computeType=BUILD_GENERAL1_SMALL" \
		--service-role arn:aws:iam::$(AWS_ACCT_ID):role/CodeBuildServiceRole

	    # or if need auth later:
	    # --source "type=GITHUB,location=https://github.com/username/repository.git,buildSpec=buildspec.yml,auth={type=OAUTH,resource=token}" \

check-service-role:
	# supports create-project with a verification check
	@if [ -z "${CODEBUILD_SERVICE_ROLE_ARN}" ]; then \
		echo "ERROR: CODEBUILD_SERVICE_ROLE_ARN is not set. Set this variable to the ARN of the CodeBuild service role." >&2; \
		exit 1; \
	fi



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

list-roles:
	@echo export CODEBUILD_SERVICE_ROLE_ARN=$$(aws iam get-role --role-name CodeBuildServiceRole | jq .Role.Arn)
	@echo export AWSBATCH_SERVICE_ROLE_ARN=$$(aws iam get-role --role-name AWSBatchServiceRole | jq .Role.Arn)
	@aws iam list-attached-role-policies --role-name CodeBuildServiceRole --no-cli-pager
	@aws iam list-attached-role-policies --role-name AWSBatchServiceRole --no-cli-pager

delete-roles:
	# Before we can delete the roles, we must detach their policies from them:
	aws iam detach-role-policy --role-name CodeBuildServiceRole --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
	aws iam detach-role-policy --role-name CodeBuildServiceRole --policy-arn "arn:aws:iam::aws:policy/AWSCodeBuildAdminAccess"
	aws iam detach-role-policy --role-name AWSBatchServiceRole --policy-arn "arn:aws:iam::aws:policy/AWSBatchServiceRolePolicy"
	# Deleting CodeBuildServiceRole and AWSBatchServiceRole
	@aws iam delete-role --role-name CodeBuildServiceRole
	@aws iam delete-role --role-name AWSBatchServiceRole


# Last two macros below are rarely used; only realistically run on dedicated GPU instance:

build-gpu:
	# Note generally this macro would not be used at all, because it builds locally and
	# this is a super intentive install/build due to the gpu version of tensorflow.
	# This is why the CodeBuild-based build process was created; I mainly use that instead.
	# (e.g. this literally won't even build at all on my lower-end cpu-only instance, but
	# it builds just fine on my heavier, gpu-based instance.)
	docker build --build-arg TENSORFLOW_PKG=tensorflow==2.12.0 -t $(ECR_REPO):$(version)-gpu .

push-to-ecr:
	# Push docker image to AWS ECR - only relevant to locally-built images.  Rarely used.
ifndef MODE
	@echo "This makefile macro must be called as:"                                       
	@echo "  make push-to-ecr MODE=GPU   # or MODE=CPU"
	@echo "The MODE determines whether the CPU or GPU version of the app image get pushed to ECR."
	@echo                                                                                
endif                                                                                    
	docker tag $(ECR_REPO):$(version)-$${MODE} $(ECR_REPO_URI):latest  # tag the cpu or gpu image as 'latest'
	aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $(ECR_REPO_URI)  # login to ECR
	docker push $(ECR_REPO_URI):latest  # push the image


# ensures all entries run every time since these aren't files
.PHONY: create-role check-role-exists create-project check-service-role \
	create-ecr-repo list-ecr-repos create-compute-env create-job-queue \
	register-job-definition run-batch check-job-status cancel-job \
	build-gpu push-to-ecr
