from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from stt_whisper.inference import transcribe_audio
from src.audio_rag_pipeline import (
    AudioChunk,
    TranscriptRAG,
    build_faiss_from_transcriptions,
    extract_and_save_entities,
    save_transcriptions_jsonl,
    transcribe_long_audio,
)
from src.medical_ner import MedicalEntities, load_entities
from src.soap_generator import SOAPNote, generate_soap_note


APP_TITLE = "Medical Transcription + RAG API"
APP_VERSION = "1.0.0"

DEFAULT_PERSIST_DIR = "faiss_transcript_store"
DEFAULT_TRANSCRIPT_PATH = "audio/transcript_chunks.jsonl"
DEFAULT_EDITED_TRANSCRIPT_PATH = "audio/edited_transcript.txt"
DEFAULT_ENTITIES_PATH = "audio/medical_entities.json"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TOP_K = 5
DEFAULT_PROCESS_AUDIO_QUERY = (
    "Provide a concise clinical summary from this consultation transcript. "
    "Include chief complaints, important findings, diagnosis, and treatment plan."
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to ask over indexed transcript")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str


class TranscribeResponse(BaseModel):
    transcript: str
    confidence: float


class ProcessAudioResponse(BaseModel):
    transcript: str
    confidence: float
    rag_response: str
    entities: dict = {}


class EntitiesResponse(BaseModel):
    entities: dict


class SOAPNoteResponse(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    raw_text: str


class SaveTranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Corrected transcript text to save")


class SaveTranscriptResponse(BaseModel):
    message: str
    transcript_path: str


app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://medical-transcription-rag.vercel.app/","http://localhost:5173"],  # For PoC, allowing all origins. Could be ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singleton-like state (loaded once and reused)
_rag: Optional[TranscriptRAG] = None
_rag_lock = asyncio.Lock()
_transcription_lock = asyncio.Lock()


def _index_exists(persist_dir: str) -> bool:
    has_index = Path(persist_dir, "faiss.index").is_file()
    has_meta = Path(persist_dir, "metadata.json").is_file() or Path(persist_dir, "metadata.pkl").is_file()
    return has_index and has_meta


def _create_rag() -> TranscriptRAG:
    return TranscriptRAG(
        persist_dir=DEFAULT_PERSIST_DIR,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        llm_model=DEFAULT_LLM_MODEL,
    )


async def _ensure_rag_loaded() -> TranscriptRAG:
    global _rag
    if _rag is not None:
        return _rag

    async with _rag_lock:
        if _rag is not None:
            return _rag
        if not _index_exists(DEFAULT_PERSIST_DIR):
            raise HTTPException(
                status_code=400,
                detail=(
                    "FAISS index is not available. Run /process-audio first "
                    "to transcribe and build the index."
                ),
            )
        try:
            _rag = await asyncio.to_thread(_create_rag)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG: {exc}") from exc
    return _rag


async def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "uploaded_audio").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".wav") as temp_file:
        temp_path = temp_file.name
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            temp_file.write(chunk)
    await upload.close()
    return temp_path


def _safe_remove(path: str) -> None:
    with suppress(FileNotFoundError):
        os.remove(path)


@app.on_event("startup")
async def startup_event() -> None:
    # Whisper model is loaded once by stt_whisper.inference import.
    # Preload RAG once if index already exists to avoid first-request delay.
    global _rag
    if _index_exists(DEFAULT_PERSIST_DIR):
        try:
            _rag = await asyncio.to_thread(_create_rag)
        except Exception:
            # Keep API alive; /query will return a detailed error when invoked.
            _rag = None


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "index_ready": _index_exists(DEFAULT_PERSIST_DIR),
        "rag_loaded": _rag is not None,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscribeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    temp_path = await _save_upload_to_temp(file)
    try:
        async with _transcription_lock:
            transcript, confidence = await asyncio.to_thread(transcribe_audio, temp_path)
        return TranscribeResponse(transcript=transcript, confidence=float(confidence))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        _safe_remove(temp_path)


@app.post("/process-audio", response_model=ProcessAudioResponse)
async def process_audio(
    file: UploadFile = File(...),
    question: str = Form(DEFAULT_PROCESS_AUDIO_QUERY),
) -> ProcessAudioResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    temp_path = await _save_upload_to_temp(file)
    try:
        async with _transcription_lock:
            chunks = await asyncio.to_thread(transcribe_long_audio, temp_path)

        if not chunks:
            raise HTTPException(status_code=400, detail="No transcript chunks generated from audio.")

        transcript = "\n".join(chunk.text for chunk in chunks if chunk.text.strip()).strip()
        confidence = sum(chunk.confidence for chunk in chunks) / len(chunks)

        await asyncio.to_thread(save_transcriptions_jsonl, chunks, DEFAULT_TRANSCRIPT_PATH)
        await asyncio.to_thread(
            build_faiss_from_transcriptions,
            chunks,
            DEFAULT_PERSIST_DIR,
            DEFAULT_EMBEDDING_MODEL,
        )

        entities = await asyncio.to_thread(
            extract_and_save_entities, chunks, DEFAULT_ENTITIES_PATH
        )

        global _rag
        async with _rag_lock:
            _rag = await asyncio.to_thread(_create_rag)
            rag_response = await asyncio.to_thread(_rag.ask, question, DEFAULT_TOP_K)

        return ProcessAudioResponse(
            transcript=transcript,
            confidence=float(confidence),
            rag_response=rag_response,
            entities=entities.to_dict(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(exc).encode('utf-8', errors='replace').decode('utf-8')}",
        ) from exc
    finally:
        _safe_remove(temp_path)


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> QueryResponse:
    rag = await _ensure_rag_loaded()
    try:
        async with _rag_lock:
            answer = await asyncio.to_thread(rag.ask, payload.query, payload.top_k)
        return QueryResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {exc}") from exc


@app.post("/save-transcript", response_model=SaveTranscriptResponse)
async def save_transcript(payload: SaveTranscriptRequest) -> SaveTranscriptResponse:
    transcript = payload.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    try:
        Path(DEFAULT_EDITED_TRANSCRIPT_PATH).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(Path(DEFAULT_EDITED_TRANSCRIPT_PATH).write_text, transcript, "utf-8")

        paragraphs = [p.strip() for p in transcript.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [transcript]
        chunks = [
            AudioChunk(
                chunk_id=idx,
                start_sec=0.0,
                end_sec=0.0,
                text=para,
                confidence=1.0,
            )
            for idx, para in enumerate(paragraphs)
        ]
        await asyncio.to_thread(save_transcriptions_jsonl, chunks, DEFAULT_TRANSCRIPT_PATH)
        await asyncio.to_thread(
            build_faiss_from_transcriptions,
            chunks,
            DEFAULT_PERSIST_DIR,
            DEFAULT_EMBEDDING_MODEL,
        )

        await asyncio.to_thread(
            extract_and_save_entities, chunks, DEFAULT_ENTITIES_PATH
        )

        global _rag
        async with _rag_lock:
            _rag = None
            if os.getenv("GROQ_API_KEY"):
                _rag = await asyncio.to_thread(_create_rag)

        return SaveTranscriptResponse(
            message="Transcript saved successfully.",
            transcript_path=DEFAULT_EDITED_TRANSCRIPT_PATH,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save transcript: {exc}") from exc


@app.get("/entities", response_model=EntitiesResponse)
async def get_entities() -> EntitiesResponse:
    entities = load_entities(DEFAULT_ENTITIES_PATH)
    if entities.is_empty():
        raise HTTPException(
            status_code=404,
            detail="No medical entities found. Process audio first.",
        )
    return EntitiesResponse(entities=entities.to_dict())


@app.post("/soap-note", response_model=SOAPNoteResponse)
async def soap_note() -> SOAPNoteResponse:
    rag = await _ensure_rag_loaded()

    transcript_path = Path(DEFAULT_EDITED_TRANSCRIPT_PATH)
    if not transcript_path.is_file():
        transcript_path = Path(DEFAULT_TRANSCRIPT_PATH)

    if not transcript_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="No transcript available. Process audio first.",
        )

    transcript_text = await asyncio.to_thread(transcript_path.read_text, "utf-8")
    entities = load_entities(DEFAULT_ENTITIES_PATH)

    try:
        note = await asyncio.to_thread(
            generate_soap_note, rag.llm, transcript_text, entities
        )
        return SOAPNoteResponse(**note.to_dict())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SOAP note generation failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT",8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
