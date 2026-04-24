FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn>=0.34.0 \
    boto3>=1.36.0 \
    pydantic>=2.10.0 \
    grpcio>=1.70.0

COPY main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
