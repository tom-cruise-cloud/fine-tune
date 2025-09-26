import os
import json
from transformers import pipeline

def model_fn(model_dir):
    """
    Loads the pre-trained model from the model_dir.
    """
    # Assuming your model is a Hugging Face model stored in model_dir
    model = pipeline("text-generation", model=model_dir)
    return model

def input_fn(request_body, content_type):
    """
    Deserializes the input data.
    """
    if content_type == 'application/json':
        data = json.loads(request_body)
        return data['inputs']
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_object, model):
    """
    Performs inference using the loaded model.
    """
    # Generate text using the Hugging Face pipeline
    generated_text = model(input_object, max_new_tokens=50)
    return generated_text[0]['generated_text']

def output_fn(prediction, accept_type):
    """
    Serializes the prediction result.
    """
    if accept_type == 'application/json':
        return json.dumps({'generated_text': prediction}), accept_type
    else:
        raise ValueError(f"Unsupported accept type: {accept_type}")
