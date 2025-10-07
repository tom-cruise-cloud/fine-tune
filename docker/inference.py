from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    model_name = './fine_tuned_model_v1/'
    health = model_name is not None
    status = 200 if health else 404
    response = {
        "status": status,
        "mimetype": "application/json"
    }
    return response

@app.route('/invocations', methods=['POST'])
def invocations():
    model_name = './fine_tuned_model_v1/'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        # device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_toke

    input_data = request.get_data().decode("utf-8")
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
    return response

with app.app_context():
    model_name = './fine_tuned_model_v1/'
