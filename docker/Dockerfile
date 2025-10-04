FROM python:3.10-slim

# RUN apt-get update && apt-get install -y \
#     nginx \
#     gunicorn \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# Set environment variables (optional, but good practice)
# ENV PYTHONUNBUFFERED 1
# ENV APP_HOME /app
# WORKDIR $APP_HOME

WORKDIR /app

# RUN apt-get update && apt-get install -y --no-install-recommends

# Copy your requirements.txt and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt

# Copy your inference script and model files
COPY inference.py ./
COPY serve ./
COPY ./fine_tuned_model_v1/ ./fine_tuned_model_v1/
# COPY nginx.conf /etc/nginx/nginx.conf
# Expose the port your inference server will listen on (if applicable)
EXPOSE 8080

# Define the command to run your inference script
# ENTRYPOINT ["python", "inference.py"]

CMD ["./serve"]
