FROM python:3.12-slim

WORKDIR /app

# Install system libraries needed for audio processing
# libsndfile1 = reading audio files
# ffmpeg = audio format conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ML models during build so they're baked into the image
# This means no download wait when the app starts
RUN python -c "from transformers import WhisperProcessor, WhisperForConditionalGeneration; WhisperProcessor.from_pretrained('openai/whisper-medium'); WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
RUN python -c "from transformers import pipeline; pipeline('ner', model='d4data/biomedical-ner-all', aggregation_strategy='simple')"

# Set UTF-8 locale to prevent ASCII encoding errors
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# Copy your app code into the container
COPY . .

# Create directories the app writes to
RUN mkdir -p audio faiss_transcript_store

# HuggingFace Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
