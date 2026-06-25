---
title: MedScribe
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# MedScribe — Clinical Documentation Intelligence

An end-to-end clinical documentation system that transcribes doctor-patient consultations, extracts medical entities, generates structured SOAP notes, and provides entity-aware retrieval-augmented generation (RAG) for querying patient records.

## What It Does

1. **Record or upload** consultation audio in the browser.
2. **Transcribe** using Whisper (openai/whisper-medium) on CPU.
3. **Extract medical entities** — medications, conditions, symptoms, procedures, vitals, dosages — using a biomedical NER model (d4data/biomedical-ner-all).
4. **Generate SOAP notes** — structured Subjective/Objective/Assessment/Plan from the transcript and extracted entities.
5. **Query the transcript** — ask clinical questions with entity-aware RAG retrieval, powered by Groq (llama-3.3-70b-versatile).
6. **Edit and save** — correct transcription errors; the FAISS index and entities rebuild automatically.

## Architecture

```text
Audio (mic/upload)
  → Whisper STT (openai/whisper-medium)
  → Chunking (30s windows, 2s overlap)
  → FAISS vector index (cosine similarity, BAAI/bge-small-en-v1.5)
  → Medical NER (d4data/biomedical-ner-all)
  → Entity-aware hybrid retrieval (55% semantic + 20% lexical + 20% entity boost + 5% rank)
  → LLM answer generation (Groq llama-3.3-70b-versatile)
  → SOAP note generation (LLM + extracted entities)
```

## Live Demo

- **Frontend:** Deployed on Vercel
- **Backend API:** Deployed on HuggingFace Spaces (Docker)

## Project Structure

```text
.
├── main.py                     # FastAPI backend (API server)
├── app.py                      # CLI interactive RAG mode
├── audio_rag.py                # CLI: ingest, ask, evaluate commands
├── Dockerfile                  # HuggingFace Spaces deployment
├── src/
│   ├── audio_rag_pipeline.py   # Core pipeline: chunking, transcription, FAISS, RAG, entity-aware retrieval
│   ├── vectorstore.py          # FAISS vector store (cosine similarity via IndexFlatIP)
│   ├── medical_ner.py          # Medical NER using d4data/biomedical-ner-all
│   ├── soap_generator.py       # SOAP note generation from transcript + entities
│   └── rag_metrics.py          # Evaluation: faithfulness, recall@K, semantic similarity
├── stt_whisper/
│   └── inference.py            # Whisper model loading and transcription
├── frontend/
│   ├── src/App.jsx             # React app (record, upload, transcript, entities, SOAP, query)
│   └── package.json            # Frontend dependencies
└── audio/                      # Runtime data (transcripts, entities, audio files)
```

## Tech Stack

### Backend
- **FastAPI** + Uvicorn
- **Whisper** (openai/whisper-medium) — speech-to-text
- **BAAI/bge-small-en-v1.5** — text embeddings
- **FAISS** (IndexFlatIP, cosine similarity) — vector search
- **d4data/biomedical-ner-all** — medical named entity recognition
- **LangChain + Groq** (llama-3.3-70b-versatile) — LLM for RAG and SOAP generation
- **librosa / soundfile** — audio processing

### Frontend
- **React** + Vite
- **Axios** — API calls
- **Lucide React** — icons

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status, index availability |
| `POST` | `/process-audio` | Upload audio → transcribe, index, extract entities, generate summary |
| `POST` | `/save-transcript` | Save edited transcript, rebuild index and entities |
| `POST` | `/query` | Ask a question over the indexed transcript |
| `GET` | `/entities` | Get extracted medical entities |
| `POST` | `/soap-note` | Generate SOAP note from transcript + entities |
| `POST` | `/transcribe` | Transcribe audio without indexing |

## Local Development

### Backend

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend at `http://localhost:5173`, backend at `http://localhost:8000`.

## Deployment

### Backend — HuggingFace Spaces (Free)

The Dockerfile pre-downloads all ML models at build time (Whisper, embeddings, NER). No runtime downloads.

1. Create a HuggingFace Space (Docker SDK, CPU basic).
2. Push code to the Space.
3. Set `GROQ_API_KEY` as a secret in Space settings.

### Frontend — Vercel (Free)

1. Import the repo on Vercel with root directory set to `frontend`.
2. Set `VITE_API_BASE_URL` to your HuggingFace Space URL.

## CLI Usage

### Ingest Audio

```bash
python audio_rag.py ingest \
  --audio-path /path/to/consultation.wav \
  --persist-dir faiss_transcript_store
```

### Ask Questions

```bash
python audio_rag.py ask \
  --question "What were the patient's symptoms?" \
  --persist-dir faiss_transcript_store
```

### Evaluate RAG Quality

```bash
python audio_rag.py evaluate \
  --eval-file audio/eval_cases.jsonl \
  --persist-dir faiss_transcript_store \
  --output-path audio/eval_results.json
```

Metrics: Faithfulness, Retrieval Recall@K, Semantic Answer Similarity, Query-Answer Embedding Similarity.

## Key Engineering Decisions

- **Cosine similarity over L2 distance** — embedding models are trained with cosine objective; L2 penalizes magnitude differences that don't reflect semantic distance.
- **Entity-aware reranking** — when a query mentions a medical entity (drug, condition), chunks containing that entity get boosted. Prevents the common failure where vector similarity returns semantically close but factually wrong chunks.
- **SOAP generation uses extracted entities** — NER output is passed alongside the transcript to the LLM, improving structured note accuracy.
- **Grounded answers only** — RAG prompt strictly prevents the LLM from adding medical knowledge beyond what's in the transcript. No hallucinated medical advice.
- **Metadata stored as JSON** — replaced pickle serialization with JSON for security (pickle allows arbitrary code execution on untrusted data).

## Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Speech-to-Text | openai/whisper-medium | ~1.5GB | Transcription |
| Embeddings | BAAI/bge-small-en-v1.5 | ~130MB | Vector search |
| Medical NER | d4data/biomedical-ner-all | ~440MB | Entity extraction |
| LLM | llama-3.3-70b-versatile (Groq) | API | Answer generation, SOAP notes |

All models are free and open-source. LLM runs via Groq free tier.
