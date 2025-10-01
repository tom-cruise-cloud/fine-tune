from transformers import AutoTokenizer, AutoModelForCausalLM
import flask
import json
import os

# model_name='./fine_tuned_model_v1/'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     # device_map="auto"
# )

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# prompt = "2 123456789010 eni-1235b8ca123456789 172.31.9.69 172.31.9.12 49761 3389 6 20 4249 1418530010 1418530070 REJECT OK"

# messages = [
#     {"role": "user", "content": prompt}
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt")

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     attention_mask=model_inputs['attention_mask'],
#     pad_token_id=tokenizer.eos_token_id, 
#     max_new_tokens=1024,
#     do_sample=True,
#     top_k=50,
#     top_p=0.95,
#     temperature=0.85,
# )

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("response: " + response)

# Flask app for handling requests
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint.
    """
    # Check if the model is loaded
    health = model_name is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    model_name='./fine_tuned_model_v1/'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        # device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    """
    Inference endpoint.
    """
    # request_content_type = flask.request.content_type
    # accept_header = flask.request.headers.get('Accept', 'application/json')
    input_data = flask.request.get_data().decode("utf-8")
    # Process input data (e.g., JSON parsing, data transformation)
    # Example:
    data = json.loads(input_data)
    
    text = tokenizer.apply_chat_template(
        data,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs['attention_mask'],
        pad_token_id=tokenizer.eos_token_id, 
        max_new_tokens=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.85,
    )

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("response: " + response)
    return response
    
    # try:
    #     input_data = input_fn(flask.request.data, request_content_type)
    #     prediction = predict_fn(input_data, model)
    #     return output_fn(prediction, accept_header)
    # except Exception as e:
    #     return flask.Response(response=str(e), status=500, mimetype='text/plain')

# Initialize the model when the application starts
with app.app_context():
    model_name='./fine_tuned_model_v1/' # SageMaker mounts model artifacts here
