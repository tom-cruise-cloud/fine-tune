import boto3
import tarfile
from datasets import load_dataset
from trl import SFTTrainer
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

data = [
    {"instruction": "version", "response": "The VPC Flow Logs version. If you use the default format, the version is , If you use a custom format, the version is the highest version among the specified fields.  Parquet data type: INT"},
    {"instruction": "account-id", "response": "The AWS account ID of the owner of the source network interface for which traffic is recorded. If the network interface is created by an AWS service, for example when creating a VPC endpoint or Network Load Balancer, the record might display unknown for this field. Parquet data type: STRING"},
    {"instruction": "interface-id", "response": "The ID of the network interface for which the traffic is recorded. Parquet data type: STRING"},
    {"instruction": "srcaddr", "response": "For incoming traffic, this is the IP address of the source of traffic. For outgoing traffic, this is the private IPv, address or the IPv, address of the network interface sending the traffic. See also pkt-srcaddr. Parquet data type: STRING"},
    {"instruction": "dstaddr", "response": "The destination address for outgoing traffic, or the IPv, or IPv, address of the network interface for incoming traffic on the network interface. The IPv, address of the network interface is always its private IPv, address. See also pkt-dstaddr. Parquet data type: STRING"},
    {"instruction": "srcport", "response": "The source port of the traffic. Parquet data type: INT"},
    {"instruction": "dstport", "response": "The destination port of the traffic. Parquet data type: INT"},
    {"instruction": "protocol", "response": "The IANA protocol number of the traffic. For more information, see Assigned Internet Protocol Numbers. Parquet data type: INT"},
    {"instruction": "packets", "response": "The number of packets transferred during the flow. Parquet data type: INT"},
    {"instruction": "bytes", "response": "The number of bytes transferred during the flow. Parquet data type: INT"},
    {"instruction": "start", "response": "The time, in Unix seconds, when the first packet of the flow was received within the aggregation interval. This might be up to ,0 seconds after the packet was transmitted or received on the network interface. Parquet data type: INT"},
    {"instruction": "end", "response": "The time, in Unix seconds, when the last packet of the flow was received within the aggregation interval. This might be up to ,0 seconds after the packet was transmitted or received on the network interface. Parquet data type: INT"},
    {"instruction": "action", "response": "The action that is associated with the traffic: ACCEPT — The traffic was accepted. REJECT — The traffic was rejected. For example, the traffic was not allowed by the security groups or network ACLs, or packets arrived after the connection was closed. Parquet data type: STRING"},
    {"instruction": "log-status", "response": "The logging status of the flow log: OK  Data is logging normally to the chosen destinations. NODATA  There was no network traffic to or from the network interface during the aggregation interval. SKIPDATA  Some flow log records were skipped during the aggregation interval. This might be because of an internal capacity constraint, or an internal error. Some flow log records may be skipped during the aggregation interval (see log-status in Available fields). This may be caused by an internal AWS capacity constraint or internal error. If you are using AWS Cost Explorer to view VPC flow log charges and some flow logs are skipped during the flow log aggregation interval, the number of flow logs reported in AWS Cost Explorer will be higher than the number of flow logs published by Amazon VPC. Parquet data type: STRING"},
    {"instruction": "vpc-id", "response": "The ID of the VPC that contains the network interface for which the traffic is recorded. Parquet data type: STRING"},
    {"instruction": "subnet-id", "response": "The ID of the subnet that contains the network interface for which the traffic is recorded. Parquet data type: STRING"},
    {"instruction": "instance-id", "response": "The ID of the instance that's associated with network interface for which the traffic is recorded, if the instance is owned by you. Returns a '-' symbol for a requester-managed network interface; for example, the network interface for a NAT gateway. Parquet data type: STRING"},
    {"instruction": "tcp-flags", "response": "The bitmask value for the following TCP flags FIN SYN RST SYN-ACK Parquet data type: STRING"},
    {"instruction": "TCP-flag-value", "response": "If no supported flags are recorded, the TCP flag value is 0. For example, since tcp-flags does not support logging ACK or PSH flags, records for traffic with these unsupported flags will result in tcp-flags value 0. If, however, an unsupported flag is accompanied by a supported flag, we will report the value of the supported flag. For example, if ACK is a part of SYN-ACK, it reports 1,. And if there is a record like SYN+ECE, since SYN is a supported flag and ECE is not, the TCP flag value is ,. If for some reason the flag combination is invalid and the value cannot be calculated, the value is '-'. If no flags are sent, the TCP flag value is 0. TCP flags can be OR-ed during the aggregation interval. For short connections, the flags might be set on the same line in the flow log record, for example, 19 for SYN-ACK and FIN, and , for SYN and FIN. For an example, see TCP flag sequence. For general information about TCP flags (such as the meaning of flags like FIN, SYN, and ACK), see TCP segment structure on Wikipedia. Parquet data type: INT"},
    {"instruction": "type", "response": "The type of traffic. The possible values are: IPv, | IPv, | EFA. For more information, see Elastic Fabric Adapter. Parquet data type: STRING"},
    {"instruction": "pkt-srcaddr", "response": "The packet-level (original) source IP address of the traffic. Use this field with the srcaddr field to distinguish between the IP address of an intermediate layer through which traffic flows, and the original source IP address of the traffic. For example, when traffic flows through a network interface for a NAT gateway, or where the IP address of a pod in Amazon EKS is different from the IP address of the network interface of the instance node on which the pod is running (for communication within a VPC). Parquet data type: STRING"},
    {"instruction": "pkt-dstaddr", "response": "The packet-level (original) destination IP address for the traffic. Use this field with the dstaddr field to distinguish between the IP address of an intermediate layer through which traffic flows, and the final destination IP address of the traffic. For example, when traffic flows through a network interface for a NAT gateway, or where the IP address of a pod in Amazon EKS is different from the IP address of the network interface of the instance node on which the pod is running (for communication within a VPC). Parquet data type: STRING"},
    {"instruction": "region", "response": "The Region that contains the network interface for which traffic is recorded. Parquet data type: STRING"},
    {"instruction": "az-id", "response": "The ID of the Availability Zone that contains the network interface for which traffic is recorded. If the traffic is from a sublocation, the record displays a '-' symbol for this field. Parquet data type: STRING"},
    {"instruction": "sublocation-type", "response": "The type of sublocation that's returned in the sublocation-id field. The possible values are: wavelength | outpost | localzone. If the traffic is not from a sublocation, the record displays a '-' symbol for this field. Parquet data type: STRING"},
    {"instruction": "sublocation-id", "response": "The ID of the sublocation that contains the network interface for which traffic is recorded. If the traffic is not from a sublocation, the record displays a '-' symbol for this field. Parquet data type: STRING"},
    {"instruction": "pkt-src-aws-service", "response": "The name of the subset of IP address ranges for the pkt-srcaddr field, if the source IP address is for an AWS service. If the source IP address belongs to an overlapped range, pkt-src-aws-service shows only one of the AWS service codes. The possible values are: AMAZON | AMAZON_APPFLOW | AMAZON_CONNECT | API_GATEWAY | AURORA_DSQL | CHIME_MEETINGS | CHIME_VOICECONNECTOR | CLOUD9 | CLOUDFRONT | CLOUDFRONT_ORIGIN_FACING | CODEBUILD | DYNAMODB | EBS | EC, | EC,_INSTANCE_CONNECT | GLOBALACCELERATOR | IVS_LOW_LATENCY | IVS_REALTIME | KINESIS_VIDEO_STREAMS | MEDIA_PACKAGE_V, | ROUTE,, | ROUTE,,_HEALTHCHECKS | ROUTE,,_HEALTHCHECKS_PUBLISHING | ROUTE,,_RESOLVER | S, | WORKSPACES_GATEWAYS. Parquet data type: STRING"},
    {"instruction": "pkt-dst-aws-service", "response": "The name of the subset of IP address ranges for the pkt-dstaddr field, if the destination IP address is for an AWS service. For a list of possible values, see the pkt-src-aws-service field. Parquet data type: STRING"},
    {"instruction": "flow-direction", "response": "The direction of the flow with respect to the interface where traffic is captured. The possible values are: ingress | egress. Parquet data type: STRING"},
    {"instruction": "traffic-path", "response": "The path that egress traffic takes to the destination. To determine whether the traffic is egress traffic, check the flow-direction field. The possible values are as follows. If none of the values apply, the field is set to -. 1 — Through another resource in the same VPC, including resources that create a network interface in the VPC 2 — Through an internet gateway or a gateway VPC endpoint 3 — Through a virtual private gateway 4 — Through an intra-region VPC peering connection 5 — Through an inter-region VPC peering connection 6 — Through a local gateway 7 — Through a gateway VPC endpoint (Nitro-based instances only) 8 — Through an internet gateway (Nitro-based instances only) Parquet data type: INT"},
    {"instruction": "ecs-cluster-arn", "response": "AWS Resource Name (ARN) of the ECS cluster if the traffic is from a running ECS task. To include this field in your subscription, you need permission to call ecs:ListClusters. Parquet data type: STRING"},
    {"instruction": "ecs-cluster-name", "response": "Name of the ECS cluster if the traffic is from a running ECS task. To include this field in your subscription, you need permission to call ecs:ListClusters. Parquet data type: STRING"},
    {"instruction": "ecs-container-instance-arn", "response": "ARN of the ECS container instance if the traffic is from a running ECS task on an EC, instance. If the capacity provider is AWS Fargate, this field will be '-'. To include this field in your subscription, you need permission to call ecs:ListClusters and ecs:ListContainerInstances. Parquet data type: STRING"},
    {"instruction": "ecs-container-instance-id", "response": "ID of the ECS container instance if the traffic is from a running ECS task on an EC, instance. If the capacity provider is AWS Fargate, this field will be '-'. To include this field in your subscription, you need permission to call ecs:ListClusters and ecs:ListContainerInstances. Parquet data type: STRING"},
    {"instruction": "ecs-container-id", "response": "Docker runtime ID of the container if the traffic is from a running ECS task. If there are one or more containers in the ECS task, this will be the docker runtime ID of the first container. To include this field in your subscription, you need permission to call ecs:ListClusters. Parquet data type: STRING"},
    {"instruction": "ecs-second-container-id", "response": "Docker runtime ID of the container if the traffic is from a running ECS task. If there are more than one containers in the ECS task, this will be the Docker runtime ID of the second container. To include this field in your subscription, you need permission to call ecs:ListClusters. Parquet data type: STRING"},
    {"instruction": "ecs-service-name", "response": "Name of the ECS service if the traffic is from a running ECS task and the ECS task is started by an ECS service. If the ECS task is not started by an ECS service, this field will be '-'. To include this field in your subscription, you need permission to call ecs:ListClusters and ecs:ListServices. Parquet data type: STRING"},
    {"instruction": "ecs-task-definition-arn", "response": "ARN of the ECS task definition if the traffic is from a running ECS task. To include this field in your subscription, you need permission to call ecs:ListClusters and ecs:ListTaskDefinitions Parquet data type: STRING"},
    {"instruction": "ecs-task-arn", "response": "ARN of the ECS task if the traffic is from a running ECS task. To include this field in your subscription, you need permission to call ecs:ListClusters and ecs:ListTasks. Parquet data type: STRING"},
    {"instruction": "ecs-task-id", "response": "ID of the ECS task if the traffic is from a running ECS task. To include this field in your subscription, you need permission to call ecs:ListClusters and ecs:ListTasks. Parquet data type: STRING"},
    {"instruction": "reject-reason", "response": "Reason why traffic was rejected. Possible values: BPA. Returns a '-' for any other reject reason. For more information about VPC Block Public Access (BPA), see Block public access to VPCs and subnets. Parquet data type: STRING"},
]

model_name = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Set the device to CUDA if available, otherwise CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

train_dataset = Dataset.from_dict({"instruction": [d["instruction"] for d in data], "response": [d["response"] for d in data]})
# print(dataset[0])
# Convert to Hugging Face Dataset

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["instruction"], text_target=examples["response"], truncation=True, padding='max_length', max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
print(tokenized_train_dataset)

training_args = TrainingArguments(output_dir="./results", num_train_epochs=3)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=tokenized_train_dataset,
    args=training_args,
    # formatting_func=lambda x: f"### Instruction:\n{x['instruction']}\n### Output:\n{x['response']}",
)

trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
