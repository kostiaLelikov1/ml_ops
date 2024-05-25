FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
