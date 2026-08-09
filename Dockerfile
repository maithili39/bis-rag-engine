FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# Pre-download embedding model at build time (avoids 420MB cold-start download)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# Copy source code
COPY src/ ./src/
COPY api.py build_vectorstore.py inference.py eval_script.py ./
COPY known_codes.json ./
COPY chromadb/ ./chromadb/

# Copy data files and build the vectorstore at image build time
# If dataset.pdf is present, vectorstore will be generated; otherwise build proceeds in degraded mode.
COPY data/ ./data/
RUN python build_vectorstore.py && rm -f ./data/*.pdf

EXPOSE 8000

CMD ["python", "api.py"]
