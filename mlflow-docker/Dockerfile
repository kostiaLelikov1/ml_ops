FROM python:3.8-slim
# Install python packages
COPY requirements.txt /tmp
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /tmp/requirements.txt
