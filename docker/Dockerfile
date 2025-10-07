# Use a Python 3.10 base image
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# # Copy your requirements.txt and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# # Copy your inference script and model files
COPY inference.py ./
COPY wsgi.py ./
COPY serve /usr/local/bin/
COPY ./fine_tuned_model_v1/ ./fine_tuned_model_v1/

EXPOSE 8080

CMD ["serve"]
