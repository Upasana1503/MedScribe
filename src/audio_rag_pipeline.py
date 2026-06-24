from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.medical_ner import MedicalEntities, extract_entities, load_entities, save_entities
from src.vectorstore import FaissVectorStore

DEFAULT_ENTITIES_PATH = "audio/medical_entities.json"

load_dotenv()

DEFAULT_MEDICAL_WHISPER_PROMPT = (
    "ROLE:\n"
    "You are a medical transcript normalization system.\n\n"
    "CONTEXT:\n"
    "The input is an automatic speech recognition transcript\n"
    "from a clinical consultation.\n\n"
    "OBJECTIVE:\n"
    "Correct spelling errors in medical terminology, drug names,\n"
    "diagnoses, and anatomical terms.\n"
    "Ensure the entire transcript is preserved in full.\n\n"
    "STRICT RULES:\n"
    "- Preserve meaning exactly.\n"
    "- Do NOT add new medical information.\n"
    "- Do NOT remove any part of the transcript.\n"
    "- Do NOT summarize or shorten the content.\n"
    "- Preserve dosages, numbers, and units exactly.\n\n"
    "OUTPUT:\n"
    "Return the complete corrected transcript in plain text.\n"
    "No explanations."
)

CLINICAL_RAG_PROMPT_TEMPLATE = (
    "ROLE:\n"
    "You are a clinical medical assistant.\n\n"
    "CONTEXT:\n"
    "You are provided excerpts from a patient's consultation history.\n\n"
    "OBJECTIVE:\n"
    "Answer the doctor's question using only the provided patient records.\n\n"
    "STRICT RULES:\n"
    "- Use only information explicitly present in the retrieved text.\n"
    "- You may logically compare or contrast the retrieved information with the question.\n"
    "- Do NOT add new medical knowledge beyond what is in the records.\n"
    "- If the topic is not mentioned at all in the retrieved text, respond exactly with:\n"
    "  \"No relevant information found in consultation history.\"\n"
    "- Be concise and clinically precise.\n\n"
    "Retrieved Consultation Excerpts:\n"
    "{retrieved_context}\n\n"
    "Doctor's Question:\n"
    "{question}\n\n"
    "Answer:"
)

NO_RELEVANT_INFO_RESPONSE = "No relevant information found in consultation history."


@dataclass
class AudioChunk:
    chunk_id: int
    start_sec: float
    end_sec: float
    text: str
    confidence: float


def _iter_audio_chunks(
    audio_path: str,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
    target_sr: int = 16000,
) -> Iterable[tuple[int, np.ndarray, int, float, float]]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")

    with sf.SoundFile(audio_path) as audio_file:
        orig_sr = audio_file.samplerate
        total_frames = len(audio_file)
        chunk_frames = int(chunk_seconds * orig_sr)
        overlap_frames = int(overlap_seconds * orig_sr)
        hop_frames = max(chunk_frames - overlap_frames, 1)

        chunk_id = 0
        start_frame = 0
        while start_frame < total_frames:
            audio_file.seek(start_frame)
            frames_to_read = min(chunk_frames, total_frames - start_frame)
            audio = audio_file.read(frames_to_read, dtype="float32", always_2d=False)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if orig_sr != target_sr:
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

            start_sec = start_frame / orig_sr
            end_sec = (start_frame + frames_to_read) / orig_sr

            yield chunk_id, audio.astype(np.float32), target_sr, start_sec, end_sec

            chunk_id += 1
            if start_frame + frames_to_read >= total_frames:
                break
            start_frame += hop_frames


def transcribe_long_audio(
    audio_path: str,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
    language: str = "en",
    whisper_prompt: str | None = DEFAULT_MEDICAL_WHISPER_PROMPT,
) -> List[AudioChunk]:
    # Import here so "ask" mode does not pay Whisper model load cost.
    from stt_whisper.inference import transcribe_audio_array

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcriptions: List[AudioChunk] = []
    for chunk_id, audio, sr, start_sec, end_sec in _iter_audio_chunks(
        audio_path=audio_path,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    ):
        print(f"[INFO] Transcribing chunk {chunk_id} ({start_sec:.1f}s - {end_sec:.1f}s)")
        text, confidence = transcribe_audio_array(
            audio,
            sr=sr,
            language=language,
            prompt_hint=whisper_prompt,
        )
        transcriptions.append(
            AudioChunk(
                chunk_id=chunk_id,
                start_sec=start_sec,
                end_sec=end_sec,
                text=text.strip(),
                confidence=confidence,
            )
        )
    return transcriptions


def save_transcriptions_jsonl(chunks: List[AudioChunk], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        for chunk in chunks:
            file_obj.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved transcript chunks to {output_path}")


def _chunks_to_segments(
    chunks: List[AudioChunk],
    segment_chars: int = 1200,
    segment_overlap: int = 200,
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=segment_chars,
        chunk_overlap=segment_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    segments: List[Dict[str, Any]] = []
    for chunk in chunks:
        raw_text = chunk.text.strip()
        if not raw_text:
            continue
        split_texts = splitter.split_text(raw_text)
        for segment_id, segment_text in enumerate(split_texts):
            segments.append(
                {
                    "text": segment_text.strip(),
                    "chunk_id": chunk.chunk_id,
                    "segment_id": segment_id,
                    "start_sec": chunk.start_sec,
                    "end_sec": chunk.end_sec,
                }
            )
    return segments


def extract_and_save_entities(
    chunks: List[AudioChunk],
    entities_path: str = DEFAULT_ENTITIES_PATH,
) -> MedicalEntities:
    full_text = "\n".join(c.text for c in chunks if c.text.strip())
    if not full_text.strip():
        return MedicalEntities()
    print("[INFO] Extracting medical entities from transcript...")
    entities = extract_entities(full_text)
    save_entities(entities, entities_path)
    return entities


def build_faiss_from_transcriptions(
    chunks: List[AudioChunk],
    persist_dir: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    segment_chars: int = 1200,
    segment_overlap: int = 200,
) -> None:
    segments = _chunks_to_segments(
        chunks=chunks,
        segment_chars=segment_chars,
        segment_overlap=segment_overlap,
    )
    if not segments:
        raise ValueError("No transcript segments were generated for indexing.")

    texts = [segment["text"] for segment in segments]
    metadatas = segments
    vectorstore = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
    vectorstore.build_from_texts(texts=texts, metadatas=metadatas)


class TranscriptRAG:
    def __init__(
        self,
        persist_dir: str = "faiss_transcript_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
        groq_api_key: str | None = None,
        entities_path: str = DEFAULT_ENTITIES_PATH,
    ):
        self.vectorstore = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
        self.vectorstore.load()

        api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(groq_api_key=api_key, model_name=llm_model)

        self.entities = load_entities(entities_path)
        self._entity_terms = self._build_entity_lookup()

    def _build_entity_lookup(self) -> set[str]:
        terms = set()
        for items in self.entities.to_dict().values():
            for item in items:
                terms.add(item.lower())
        return terms

    def _find_query_entities(self, question: str) -> set[str]:
        q_lower = question.lower()
        return {term for term in self._entity_terms if term in q_lower}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    def _rerank_results(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_tokens = set(self._tokenize(question))
        query_entities = self._find_query_entities(question)
        ranked = []
        total = len(candidates)

        for idx, item in enumerate(candidates):
            meta = item.get("metadata") or {}
            text = str(meta.get("text", ""))
            text_lower = text.lower()
            text_tokens = set(self._tokenize(text))
            lexical = (len(query_tokens & text_tokens) / len(query_tokens)) if query_tokens else 0.0

            # Cosine similarity from FAISS IndexFlatIP. Higher = more similar.
            semantic = float(item.get("score", 0.0))
            semantic = max(0.0, min(1.0, semantic))

            # Entity boost: if query mentions a medical entity, boost chunks containing it.
            entity_boost = 0.0
            if query_entities:
                matched = sum(1 for ent in query_entities if ent in text_lower)
                entity_boost = matched / len(query_entities)

            # Rank prior keeps earlier semantic neighbors slightly preferred.
            rank_prior = 1.0 - (idx / max(total, 1))

            score = 0.55 * semantic + 0.20 * lexical + 0.20 * entity_boost + 0.05 * rank_prior
            ranked.append((score, item))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:top_k]]

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        log: bool = True,
        candidate_multiplier: int = 4,
    ) -> List[Dict[str, Any]]:
        candidate_k = max(top_k, top_k * max(candidate_multiplier, 1))
        candidates = self.vectorstore.query(question, top_k=candidate_k, log=log)
        return self._rerank_results(question, candidates, top_k=top_k)

    def _build_context_blocks(self, results: List[Dict[str, Any]]) -> List[str]:
        context_blocks = []
        for result in results:
            meta = result.get("metadata") or {}
            text = meta.get("text", "").strip()
            if not text:
                continue
            start_sec = float(meta.get("start_sec", 0.0))
            end_sec = float(meta.get("end_sec", 0.0))
            chunk_id = meta.get("chunk_id", "n/a")
            context_blocks.append(
                f"[chunk={chunk_id} time={start_sec:.1f}-{end_sec:.1f}s]\n{text}"
            )
        return context_blocks

    def _generate_answer(self, question: str, context_blocks: List[str]) -> str:
        if not context_blocks:
            return NO_RELEVANT_INFO_RESPONSE

        context = "\n\n".join(context_blocks)
        prompt = CLINICAL_RAG_PROMPT_TEMPLATE.format(
            retrieved_context=context,
            question=question,
        )
        response = self.llm.invoke(prompt)
        return response.content

    def ask(self, question: str, top_k: int = 5) -> str:
        results = self.retrieve(question, top_k=top_k, log=True)
        context_blocks = self._build_context_blocks(results)
        return self._generate_answer(question, context_blocks)

    def ask_with_details(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        results = self.retrieve(question, top_k=top_k, log=True)
        context_blocks = self._build_context_blocks(results)
        answer = self._generate_answer(question, context_blocks)

        retrieved = []
        for result in results:
            meta = result.get("metadata") or {}
            retrieved.append(
                {
                    "chunk_id": meta.get("chunk_id"),
                    "segment_id": meta.get("segment_id"),
                    "start_sec": meta.get("start_sec"),
                    "end_sec": meta.get("end_sec"),
                    "text": meta.get("text", ""),
                }
            )

        return {
            "question": question,
            "answer": answer,
            "retrieved": retrieved,
            "context_blocks": context_blocks,
        }
