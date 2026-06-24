import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import json
import os

from src.audio_rag_pipeline import (
    TranscriptRAG,
    build_faiss_from_transcriptions,
    save_transcriptions_jsonl,
    transcribe_long_audio,
)
from src.rag_metrics import (
    embedding_cosine_similarity,
    estimate_relevant_chunk_ids,
    faithfulness_score,
    generate_reference_answer,
    recall_at_k,
)

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".webm",
}


def _read_full_transcript(transcript_path: str) -> str:
    if not os.path.isfile(transcript_path):
        return ""

    chunks = []
    with open(transcript_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = str(payload.get("text", "")).strip()
            chunk_id = payload.get("chunk_id")
            if text:
                chunks.append((chunk_id, text))

    if not chunks:
        return ""

    try:
        chunks.sort(key=lambda item: int(item[0]))
    except (TypeError, ValueError):
        pass

    return "\n".join(text for _, text in chunks)


if __name__ == "__main__":
    print("Transcript RAG (Interactive)")
    print("Paste an audio file path to ingest now, or press Enter to use existing index.")
    print("Type your question and press Enter. Type 'exit' to quit.")

    persist_dir = "faiss_transcript_store"
    top_k = 8
    transcript_path = "audio/transcript_chunks.jsonl"

    audio_path = input("\nAudio file path (optional): ").strip().strip("\"'")
    if audio_path:
        if not os.path.isfile(audio_path):
            print(f"Audio file not found: {audio_path}")
            raise SystemExit(1)
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(AUDIO_EXTENSIONS))
            print(f"Invalid audio file type: '{ext or 'none'}'.")
            print(f"Use one of: {allowed}")
            raise SystemExit(1)

        print("\n[INFO] Transcribing and indexing audio...")
        transcriptions = transcribe_long_audio(audio_path=audio_path)
        save_transcriptions_jsonl(transcriptions, transcript_path)
        build_faiss_from_transcriptions(
            chunks=transcriptions,
            persist_dir=persist_dir,
        )
        print(f"[INFO] Ingestion complete. Index saved to '{persist_dir}'.")

    try:
        rag = TranscriptRAG(persist_dir=persist_dir)
    except Exception as exc:
        print(f"Failed to load transcript index from '{persist_dir}': {exc}")
        print("Run ingest first using audio_rag.py ingest.")
        raise SystemExit(1)

    while True:
        question = input("\nQuestion: ").strip()
        lowered = question.lower()
        if lowered in {"exit", "quit"}:
            print("Exiting.")
            break
        if not question:
            print("Please enter a non-empty question.")
            continue
        wants_full_transcript = (
            lowered in {"transcription", "full transcription", "full transcript", "transcript"}
            or ("transcript" in lowered)
            or ("transcription" in lowered)
        )
        if wants_full_transcript:
            full_text = _read_full_transcript(transcript_path)
            if not full_text:
                print("No transcript found. Ingest an audio file first.")
                continue
            print("\n[FULL TRANSCRIPTION]")
            print(full_text)
            continue

        details = rag.ask_with_details(question, top_k=top_k)
        answer = details["answer"]

        faithfulness = faithfulness_score(
            llm=rag.llm,
            question=question,
            answer=answer,
            contexts=details["context_blocks"],
        )

        retrieved_chunk_ids = []
        for item in details["retrieved"]:
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                continue
            try:
                retrieved_chunk_ids.append(int(chunk_id))
            except (TypeError, ValueError):
                continue

        recall_pool_k = max(top_k * 3, 12)
        recall_pool_results = rag.retrieve(question, top_k=recall_pool_k, log=False)
        recall_candidates = []
        for result in recall_pool_results:
            meta = result.get("metadata") or {}
            chunk_id = meta.get("chunk_id")
            text = meta.get("text", "")
            recall_candidates.append({"chunk_id": chunk_id, "text": text})

        relevant_chunk_ids = estimate_relevant_chunk_ids(
            llm=rag.llm,
            question=question,
            chunk_candidates=recall_candidates,
        )
        recall = recall_at_k(
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=relevant_chunk_ids,
            k=top_k,
        ) if relevant_chunk_ids else 0.0

        reference_answer = generate_reference_answer(
            llm=rag.llm,
            question=question,
            contexts=details["context_blocks"],
        )
        semantic_answer_similarity = embedding_cosine_similarity(
            embed_model=rag.vectorstore.model,
            text_a=answer,
            text_b=reference_answer,
        ) if reference_answer else 0.0

        query_answer_similarity = embedding_cosine_similarity(
            embed_model=rag.vectorstore.model,
            text_a=question,
            text_b=answer,
        )

        print("\n[ANSWER]")
        print(answer)
        print("\n[METRICS]")
        print(f"Faithfulness: {faithfulness:.3f}")
        print(f"Recall@{top_k}: {recall:.3f}")
        print(f"Semantic Answer Similarity: {semantic_answer_similarity:.3f}")
        print(f"Query-Answer Embedding Similarity: {query_answer_similarity:.3f}")
 
