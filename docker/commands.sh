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

sagemaker-user@default:~$ docker push 837369895783.dkr.ecr.us-east-1.amazonaws.com/falcon3-10b-inference:latest
The push refers to repository [837369895783.dkr.ecr.us-east-1.amazonaws.com/falcon3-10b-inference]
ecb610846dc8: Pushed 
3addcb90c002: Pushed 
bd831d67c76e: Pushed 
1b60f954bada: Pushed 
85e6d40487f0: Pushed 
067ea27560c1: Pushed 
7fb1037e08b3: Pushed 
14cbeede8d6e: Pushed 
ae2d55769c5e: Pushed 
e2ef8a51359d: Pushed 
latest: digest: sha256:2f142d6868f81ade85eabce64dcdcdc0e840c58e6db2373295fa11391fd3ed59 size: 2417

https://github.com/aws/sagemaker-pytorch-inference-toolkit


