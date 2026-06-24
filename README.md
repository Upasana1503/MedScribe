# Medical Transcription RAG

This project is a proof-of-concept medical transcription and retrieval app. It records consultation audio in the browser, sends it to a FastAPI backend, transcribes the audio locally with Whisper, stores the transcript in a FAISS vector index, and uses a Groq-hosted LLM to answer or summarize from the indexed transcript.

The frontend also lets you edit the generated transcription after processing. When you click `Confirm & Save`, the corrected transcript is saved and the FAISS index is rebuilt from the edited text, so later RAG queries use the corrected version.

## Features

1. Browser-based audio recording with a React + Vite frontend.
2. Audio upload to a FastAPI backend.
3. Local Whisper transcription on CPU.
4. Transcript chunking and FAISS vector indexing.
5. RAG response generation with Groq.
6. Editable transcription review screen.
7. Save endpoint for corrected transcripts.
8. CLI tools for ingesting audio, asking questions, and evaluating RAG quality.

## Project Structure

```text
.
├── main.py                     # FastAPI backend used by the frontend
├── app.py                      # CLI interactive RAG mode
├── audio_rag.py                # CLI ingest, ask, and evaluate commands
├── src/
│   ├── audio_rag_pipeline.py   # Audio chunking, transcription, FAISS build, RAG logic
│   ├── vectorstore.py          # FAISS vector store wrapper
│   └── rag_metrics.py          # Evaluation metrics
├── stt_whisper/
│   └── inference.py            # Whisper model loading and transcription helpers
├── frontend/
│   ├── src/App.jsx             # React app
│   └── package.json            # Frontend scripts and dependencies
├── audio/
│   ├── transcript_chunks.jsonl # Saved transcript chunks
│   └── edited_transcript.txt   # Saved corrected transcript
└── faiss_transcript_store/     # Saved FAISS index and metadata
```

## Requirements

- Python 3.12
- Node.js and npm
- A Groq API key for RAG answer generation
- Internet access on first model setup, unless the required Hugging Face models are already cached locally

The backend uses:

- FastAPI
- Uvicorn
- Transformers Whisper
- Sentence Transformers
- FAISS
- LangChain Groq

The frontend uses:

- React
- Vite
- Axios
- Lucide React

## Environment Variables

Set your Groq key before starting the backend:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Whisper is configured to use local Hugging Face files by default:

```bash
HF_LOCAL_FILES_ONLY=1
```

If this is your first run and the Whisper model is not already cached, start the backend with downloads enabled:

```bash
export HF_LOCAL_FILES_ONLY=0
```

After the model has downloaded, you can switch back to local-only mode if you want.

## How to Run the Full App

Open two terminals from the project root:

```bash
cd /Users/sahukaraprakash/Downloads/MEDICAL_TRANSCRIPTION_RAG-frontendretry
```

### 1. Start the Backend

If the existing `venv` folder is available:

```bash
source venv/bin/activate
export GROQ_API_KEY="your_groq_api_key_here"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If you need to create a fresh virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key_here"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend URL:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

### 2. Start the Frontend

In a second terminal:

```bash
cd /Users/sahukaraprakash/Downloads/MEDICAL_TRANSCRIPTION_RAG-frontendretry/frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

## Frontend Workflow

1. Open `http://localhost:5173`.
2. Click `Start Recording`.
3. Allow microphone access in the browser.
4. Speak the consultation audio.
5. Click `Stop & Process`.
6. Wait for transcription and RAG processing.
7. Review the transcription in the text editor.
8. Type any corrections for mispronounced or misprocessed words.
9. Click `Confirm & Save`.

The saved edited transcript is written to:

```text
audio/edited_transcript.txt
```

The corrected transcript is also written into:

```text
audio/transcript_chunks.jsonl
```

The FAISS index is rebuilt in:

```text
faiss_transcript_store/
```

## Backend API

### `GET /health`

Checks whether the backend is running and whether a transcript index is available.

Example response:

```json
{
  "status": "ok",
  "index_ready": true,
  "rag_loaded": true
}
```

### `POST /process-audio`

Accepts an uploaded audio file, transcribes it, builds the FAISS index, and returns both transcript text and a RAG-generated response.

Form data:

- `file`: audio file
- `question`: optional custom summary/question prompt

### `POST /save-transcript`

Saves the edited transcript and rebuilds the FAISS index from the corrected text.

JSON body:

```json
{
  "transcript": "Corrected transcript text here"
}
```

### `POST /query`

Asks a question over the saved transcript index.

JSON body:

```json
{
  "query": "What diagnosis was discussed?",
  "top_k": 5
}
```

## CLI Usage

The project can also be used without the frontend.

### Ingest Audio and Build FAISS Index

```bash
python audio_rag.py ingest \
  --audio-path /absolute/path/to/consultation.wav \
  --persist-dir faiss_transcript_store \
  --transcript-path audio/transcript_chunks.jsonl \
  --chunk-seconds 30 \
  --overlap-seconds 2 \
  --language en
```

### Ask Questions Over the Indexed Transcript

```bash
python audio_rag.py ask \
  --question "What were the patient's symptoms?" \
  --persist-dir faiss_transcript_store \
  --top-k 5
```

### Interactive CLI Mode

```bash
python app.py
```

You can optionally paste an audio file path to ingest, then ask questions in a loop. Type `exit` to quit.

### Evaluate RAG Quality

Create a JSONL eval file:

```json
{"question":"What is the main complaint?","reference_answer":"The patient reports chest pain and shortness of breath.","relevant_chunk_ids":[0,1]}
{"question":"What treatment was advised?","reference_answer":"The patient was advised medication and follow-up.","relevant_chunk_ids":[2,3]}
```

Run evaluation:

```bash
python audio_rag.py evaluate \
  --eval-file audio/eval_cases.jsonl \
  --persist-dir faiss_transcript_store \
  --top-k 5 \
  --output-path audio/eval_results.json
```

Implemented metrics:

1. Faithfulness
2. Retrieval Recall@K
3. Semantic Answer Similarity
4. Query-Answer Embedding Similarity

## Troubleshooting

### Frontend says processing failed

Check that the backend is running:

```text
http://localhost:8000/health
```

Also check the backend terminal for the real error message.

### `GROQ_API_KEY` error

Set the key before running the backend:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

Then restart Uvicorn.

### Whisper model fails to load

If this is the first run, allow Hugging Face downloads:

```bash
export HF_LOCAL_FILES_ONLY=0
```

Then restart the backend.

### Frontend cannot reach backend

The frontend defaults to:

```text
http://localhost:8000
```

If your backend runs somewhere else, create `frontend/.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

Restart the Vite dev server after changing `.env`.

### Browser microphone does not work

Use a modern browser and allow microphone permissions. Browser microphone APIs usually require `localhost` or HTTPS, so use:

```text
http://localhost:5173
```

## Build Check

To verify the frontend compiles:

```bash
cd frontend
npm run build
```

To verify the backend imports:

```bash
venv/bin/python -c "import main; print('main import ok')"
```
