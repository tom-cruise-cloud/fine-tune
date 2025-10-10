docker build --network sagemaker -t falcon3-10b-inference .

docker run --network sagemaker falcon3-10b-inference

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

Add permission to the role: 
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
            "Resource": "arn:aws:ecr:us-east-1:837369895783.dkr.ecr.us-east-1.amazonaws.com/falcon3-10b-inference"
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

https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/multi_model_bring_your_own/container/Dockerfile


(.fine_tune) PS D:\Users\yan.gong\.fine_tune> flask --app web.py run --host=0.0.0.0 --port=8080
 * Serving Flask app 'web.py'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
 * Running on http://172.17.8.189:8080
Press CTRL+C to quit
127.0.0.1 - - [01/Oct/2025 10:51:58] "POST /api/data HTTP/1.1" 200 -
127.0.0.1 - - [01/Oct/2025 10:56:14] "POST /api/data HTTP/1.1" 200 -


PS D:\Users\yan.gong> Invoke-WebRequest -Uri http://127.0.0.1:8080/api/data -Method Post -ContentType "application/json" -body '{"role": "user", "content": "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK"}'


StatusCode        : 200
StatusDescription : OK
Content           : {"content":"2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK","role":"user"}

RawContent        : HTTP/1.1 200 OK
                    Connection: close
                    Content-Length: 142
                    Content-Type: application/json
                    Date: Wed, 01 Oct 2025 14:57:06 GMT
                    Server: Werkzeug/3.1.3 Python/3.13.5

                    {"content":"2 123456789010 eni-123...
Forms             : {}
Headers           : {[Connection, close], [Content-Length, 142], [Content-Type, application/json], [Date, Wed, 01 Oct 2025 14:57:06 GMT]...}
Images            : {}
InputFields       : {}
Links             : {}
ParsedHtml        : System.__ComObject
RawContentLength  : 142

sagemaker-user@default:~$ curl -X POST -H "Content-Type: application/json" http://127.0.0.1:8080/invocations -d '{"role": "user", "content": "2 123456789010 eni-123
5b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK"}'
{"content":"2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK","role":"user"}

curl -X POST -H "Content-Type: application/json" http://127.0.0.1:8080/invocations -d '{"role": "user", "content": "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK"}'

curl -X GET -H "Content-Type: application/json" http://127.0.0.1:8080/ping
{"mimetype":"application/json","status":200}

sudo kill -9 $(pidof nginx)

aws console - sagemaker ai - inference - serverless

ml instance details: 
https://aws.amazon.com/sagemaker/ai/pricing/

Async invocation config

curl -X POST -H "Content-Type: application/json" https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/falcon3-7b/invocations -d '{"role": "user", "content": "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK"}'
{"message":"Missing Authentication Token"}
