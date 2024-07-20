ECR_REPO = flow_models
export ECR_REPO_URI = ${AWS_ACCT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}


list-ecr-repos:
	# List the repo in ECR to confirm this one is there after create-ecr-repo
	aws ecr describe-repositories --no-cli-pager

create-ecr-repo:
	# Create the repo in ECR where this app's docker images will be held on AWS
	aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION}

create-codebuild-role: check-codebuild-role-exists
	# generates CODEBUILD_SERVICE_ROLE_ARN needed for create-project.
	# read it from output and put into CODEBUILD_SERVICE_ROLE_ARN env var.
	@echo "Creating role CodeBuildServiceRole..."
	@aws iam create-role \
		--role-name CodeBuildServiceRole --no-cli-pager \
		--assume-role-policy-document file://awsbatch-support/codebuild-trust-policy.json
	@echo "Role CodeBuildServiceRole created successfully."
	@aws iam create-policy --policy-name CodeBuildCloudWatchPolicy \
		--policy-document file://awsbatch-support/codebuild-cloudwatch-policy.json
	@aws iam attach-role-policy --role-name CodeBuildServiceRole \
		--policy-arn arn:aws:iam::${AWS_ACCT_ID}:policy/CodeBuildCloudWatchPolicy
	@echo "CodeBuildCloudWatchPolicy successfully attached to role CodeBuildServiceRole."
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

create-batch-instance-profile:
	@aws iam create-role --role-name BatchInstanceRole --no-cli-pager \
		--assume-role-policy-document file://awsbatch-support/instance-trust-policy.json
	@aws iam attach-role-policy --role-name BatchInstanceRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
	@aws iam create-instance-profile --instance-profile-name BatchInstanceProfile
	@aws iam add-role-to-instance-profile --instance-profile-name BatchInstanceProfile \
		--role-name BatchInstanceRole

	#@aws iam get-instance-profile --instance-profile-name BatchInstanceProfile



# Create compute environment - g4dn.xlarge have 4 vCpus so pinning vCpus to 4.
# Assumes pre-existing AWS subnet within your VPC, and a simple security group
# with no inbound rules and default outbound 0.0.0.0; name this group something
# like aws-batch-instance, add to the inbound rules of your mlflow instance,
# and set env vars ${AWSBATCH_SUBNET} and ${AWSBATCH_SG} in your env so this
# script can use them below.
# To create sg via cli instead of manually in aws dashboard:
# aws ec2 create-security-group --group-name aws-batch-instance --description "Security group for AWS Batch compute" --vpc-id your-vpc-id
create-compute-env:
	aws batch create-compute-environment --compute-environment-name GPUEnvironment --type MANAGED \
		--compute-resources type=EC2,allocationStrategy=BEST_FIT_PROGRESSIVE,minvCpus=4,maxvCpus=4,desiredvCpus=4,instanceTypes=g4dn.xlarge,subnets=${AWSBATCH_SUBNET},securityGroupIds=${AWSBATCH_SG},instanceRole=arn:aws:iam::$(AWS_ACCT_ID):instance-profile/BatchInstanceProfile \
		--service-role ""
	    # Blank string tells AWSbatch to auto-generate the service role AWSBatchServiceRole as a service-linked-role.

	# to set up to use much-cheaper spot instances later:
	# aws batch create-compute-environment --compute-environment-name GPUEnvironment --type MANAGED \
	#	--compute-resources type=SPOT,allocationStrategy=SPOT_CAPACITY_OPTIMIZED,minvCpus=4,maxvCpus=4,desiredvCpus=4,instanceTypes=g4dn.xlarge,subnets=${AWSBATCH_SUBNET},securityGroupIds=${AWSBATCH_SG},spotIamFleetRole=arn:aws:iam::$(AWS_ACCT_ID):role/AWSBatchServiceRole \
    #   --service-role ""

	@aws batch describe-compute-environments --compute-environments GPUEnvironment


list-compute-resources:
	@-aws batch describe-compute-environments --query 'computeEnvironments[*].[status, computeEnvironmentName, statusReason]' --output json | jq -r '["compute-env"] + .[] | @tsv'
	@-aws batch describe-job-definitions --query 'jobDefinitions[*].[status, jobDefinitionName, revision]' --output json | jq -r '.[] | ["job-definition"] + . | @tsv'
	@-aws autoscaling describe-auto-scaling-groups --query 'AutoScalingGroups[*].[Status, HealthStatus, AutoScalingGroupName]' --output json | jq -r '["autoscale-group"] + .[] | @tsv'
	@-aws batch describe-job-queues --query 'jobQueues[*].[state, jobQueueName]' --output json | jq -r '["job-queue"] + .[] | @tsv'
	@-aws ec2 describe-instances --query 'Reservations[*].Instances[*].[State.Name, InstanceId, LaunchTime, Tags[?Key==`Name`].Value | [0]]' --output json | jq -r '.[] | ["ec2-instance"] + .[] | @tsv'

delete-compute-resources1:
	@-aws batch update-job-queue --job-queue GPUJobQueue --state DISABLED
	@-aws batch delete-job-queue --job-queue GPUJobQueue
	@-aws batch deregister-job-definition --job-definition GPUJobDefinition:1  # assumes revision 1 here but use whatever seen in describe-job-definitions
	@-aws batch update-compute-environment --compute-environment GPUEnvironment --state DISABLED

# wait for the state of update-compute-resources1 to settle first, then run this:
delete-compute-resources2:
	@-aws batch delete-compute-environment --compute-environment GPUEnvironment


create-job-queue:
	# Create job queue
	aws batch create-job-queue --job-queue-name GPUJobQueue \
		--compute-environment-order order=1,computeEnvironment=GPUEnvironment \
		--priority 1

register-job-definition:
	# Register job definition.  The aws command requires an ECR_REPO_URI in this json
	# file which varies per user, so before submitting json file, substitute in the
	# $ECR_REPO_URI environment variable.
	envsubst < awsbatch-support/job_definition_template.json > /tmp/job-definition.json \
	&& aws batch register-job-definition --cli-input-json file:///tmp/job-definition.json
	#rm /tmp/job_definition.json


create-project: check-service-role
	# gets code from github branch, builds docker image, and pushes it to ECR repo
	aws codebuild create-project \
	    --name "flow_models_build" \
		--source "type=GITHUB,location=https://github.com/aganse/flow_models.git,buildspec=awsbatch-support/buildspec.yml" \
	    --artifacts "type=NO_ARTIFACTS" \
	    --environment "type=LINUX_CONTAINER,image=aws/codebuild/standard:4.0,computeType=BUILD_GENERAL1_SMALL,environmentVariables=[{name='ECR_REPO_URI', value='${ECR_REPO_URI}'},{name='DEVICE', value='${DEVICE}'}]" \
		--service-role arn:aws:iam::${AWS_ACCT_ID}:role/CodeBuildServiceRole
	    # (if needed auth later, we can append ",auth={type=OAUTH,resource=token}" to --source arg)

check-service-role:
	# supports create-project with a verification check
	@if [ -z "${CODEBUILD_SERVICE_ROLE_ARN}" ]; then \
		echo "ERROR: CODEBUILD_SERVICE_ROLE_ARN is not set. Set this variable to the ARN of the CodeBuild service role." >&2; \
		exit 1; \
	fi

run-build:
	@aws codebuild start-build --project-name flow_models_build

	# and to check build status in cli:
	# aws codebuild batch-get-builds --ids <arn:etc.etc.etc from start-build output, or from console>

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



push-to-ecr:
	# Push docker image to AWS ECR - eg could build it on g4dn.xlarge instance and push to ECR,
	# then after that could run image in AWS Batch for future runs.
ifndef DEVICE
	@echo "This makefile macro must be called as:"                                       
	@echo "  make push-to-ecr DEVICE=gpu   # or DEVICE=cpu"
	@echo "The DEVICE determines whether the CPU or GPU version of the app image gets pushed to ECR."
	@echo                                                                                
endif                                                                                    
	docker tag ${ECR_REPO}:${version}-$${DEVICE} ${ECR_REPO_URI}:latest  # tag the cpu or gpu image as 'latest'
	aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin ${ECR_REPO_URI}  # login to ECR
	docker push ${ECR_REPO_URI}:latest  # push the image


# ensures all entries run every time since these aren't files
.PHONY: create-role check-role-exists create-project check-service-role \
	create-ecr-repo list-ecr-repos create-compute-env create-job-queue \
	register-job-definition run-batch check-job-status cancel-job push-to-ecr
