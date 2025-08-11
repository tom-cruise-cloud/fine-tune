import boto3
import json
from bert_score import score

AWS_BEARER_TOKEN_BEDROCK="bedrock-api-key-888888888888888"
# Initialize the Amazon Bedrock agent runtime client
bedrock_agent_client = boto3.client("bedrock-agent-runtime")

kb_id = "xxxxxxxxxxxx"
model_arn = "arn:aws:bedrock:us-east-1:888888888888:inference-profile/us.anthropic.claude-opus-4-20250514-v1:0"  
# Or another suitable model like "amazon.titan-text-express-v1" or similar

def retrieve_and_generate(input_query, knowledge_base_id, model_arn):   
    response = bedrock_agent_client.retrieve_and_generate(
        input={
            'text': input_query
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': knowledge_base_id,
                'modelArn': model_arn,
		'generationConfiguration': {
                  'inferenceConfig': {
                    'textInferenceConfig': {
                        'maxTokens': 1200,
			'temperature': 0.5
                    },
                },
              },
            },
     	},
    )
    return response['output']['text']

query = "provide troubleshooting steps and root cause analyze for '2 123456789010 eni-123456789abcdef01 172.31.16.139 203.0.113.12 12345 80 6 20 1200 1432917027 1432917142 REJECT NODATA'"
#query = "Why packet is dropped?"
#query = "Why packet is reject? Please provide troubleshooting steps?"
#query = "provide root cause analysis for REJECT and provide troubleshooting steps"
#query = "Provide root cause analysis for packet is dropped and provide troubleshooting steps"

generated_response = retrieve_and_generate(query, kb_id, model_arn)
print(f"User Query: {query}")
print("")
print("")
print(f"Generated Response: {generated_response}")

#print("\n")
#print(f"Evaluating...")
#scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#scores = scorer.score(query, generated_response)
#for key in scores:
#    print(f'{key}: {scores[key]}')

print("\n")
print(f"Evaluating...")
P, R, F1 = score([query], [generated_response], lang="en", verbose=True)
print(f"Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
