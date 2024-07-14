ECR_REPO = flow_models
ECR_REPO_URI = ${AWS_ACCT_ID}.dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO)

create-role: check-role-exists
	@echo "Creating role CodeBuildServiceRole..."
	@aws iam create-role \
		--role-name CodeBuildServiceRole \
		--assume-role-policy-document file://trust-policy.json | tee create-role-output.txt
	@echo "Role CodeBuildServiceRole created successfully."

check-role-exists:
	@aws iam get-role --role-name CodeBuildServiceRole > /dev/null 2>&1; \
	if [ $$? -eq 0 ]; then \
		echo "Role CodeBuildServiceRole already exists. Skipping creation."; \
		exit 1; \
	fi

create-project: check-service-role
	aws codebuild create-project \
	    --name "ExampleProject" \
	    --source "type=GITHUB,location=https://github.com/username/repository.git,buildSpec=buildspec.yml" \
        # --source "type=GITHUB,location=https://github.com/username/repository.git,buildSpec=buildspec.yml,auth={type=OAUTH,resource=token}" \
	    --artifacts "type=NO_ARTIFACTS" \
	    --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:4.0,computeType=BUILD_GENERAL1_SMALL" \
	    --service-role ${CODEBUILD_SERVICE_ROLE_ARN}

check-service-role:
	@if [ -z "${CODEBUILD_SERVICE_ROLE_ARN}" ]; then \
		echo "ERROR: CODEBUILD_SERVICE_ROLE_ARN is not set. Set this variable to the ARN of the CodeBuild service role." >&2; \
		exit 1; \
	fi

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
	aws batch register-job-definition --cli-input-json file://batchsupport/job_definition.json

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


# All the macros in this makefile-batchsupport should be listed here in this
# .PHONY list because they're not files, they're process we need to run every time.
.PHONY: create-project check-service-role submit-job create-role check-role-exists
