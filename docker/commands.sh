docker build --network sagemaker -t falcon3-10b-inference .

sagemaker-user@default:~$ docker rmi 837369895783.dkr.ecr.region.amazonaws.com/falcon3-10b-inference:latest
Untagged: 837369895783.dkr.ecr.region.amazonaws.com/falcon3-10b-inference:latest

sagemaker-user@default:~$ aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 837369895783.dkr.ecr.us-east-1.amazonaws.com
WARNING! Your password will be stored unencrypted in /home/sagemaker-user/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credential-stores

Login Succeeded

sagemaker-user@default:~$ docker tag falcon3-10b-inference:latest 837369895783.dkr.ecr.us-east-1.amazonaws.com/falcon3-10b-inference:latest
sagemaker-user@default:~$ docker images
REPOSITORY                                                           TAG       IMAGE ID       CREATED          SIZE
837369895783.dkr.ecr.us-east-1.amazonaws.com/falcon3-10b-inference   latest    1086e125886f   35 minutes ago   7.4GB
falcon3-10b-inference                                                latest    1086e125886f   35 minutes ago   7.4GB

{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:PutImage"
            ],
            "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<repository-name>"
        }
    ]
}

docker push 837369895783.dkr.ecr.us-east-1.amazonaws.com/falcon3-10b-inference:latest
