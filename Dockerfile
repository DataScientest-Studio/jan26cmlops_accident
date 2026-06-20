FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Pré-installer torch CPU-only pour éviter ~3-6 Go de dépendances CUDA
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY src/ ./src/

RUN dvc init --no-scm && \
    dvc remote add -d myremote /app/dvc-storage 

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.models.api:api", "--host", "0.0.0.0", "--port", "8000"]
