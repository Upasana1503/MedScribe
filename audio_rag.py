import argparse
import json
from statistics import mean

from src.rag_metrics import (
    embedding_cosine_similarity,
    faithfulness_score,
    parse_eval_case,
    recall_at_k,
)

from src.audio_rag_pipeline import (
    DEFAULT_MEDICAL_WHISPER_PROMPT,
    TranscriptRAG,
    build_faiss_from_transcriptions,
    save_transcriptions_jsonl,
    transcribe_long_audio,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-audio RAG pipeline (Whisper + FAISS + LLM)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Chunk and transcribe audio, then index transcript in FAISS")
    ingest.add_argument("--audio-path", required=True, help="Path to source audio file")
    ingest.add_argument("--persist-dir", default="faiss_transcript_store", help="FAISS output directory")
    ingest.add_argument("--transcript-path", default="audio/transcript_chunks.jsonl", help="Transcript JSONL output file")
    ingest.add_argument("--chunk-seconds", type=float, default=30.0, help="Audio chunk length in seconds")
    ingest.add_argument("--overlap-seconds", type=float, default=2.0, help="Overlap between chunk windows")
    ingest.add_argument("--language", default="en", help="Whisper transcription language")
    ingest.add_argument(
        "--whisper-prompt",
        default=DEFAULT_MEDICAL_WHISPER_PROMPT,
        help="Prompt hint passed to Whisper decoding",
    )
    ingest.add_argument("--segment-chars", type=int, default=1200, help="Transcript segment size in characters")
    ingest.add_argument("--segment-overlap", type=int, default=200, help="Segment overlap in characters")
    ingest.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model name")

    ask = subparsers.add_parser("ask", help="Ask question over indexed transcript")
    ask.add_argument("--question", required=True, help="Question to answer")
    ask.add_argument("--persist-dir", default="faiss_transcript_store", help="FAISS directory to query")
    ask.add_argument("--top-k", type=int, default=5, help="Top-K retrieved transcript segments")
    ask.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    ask.add_argument("--llm-model", default="llama-3.3-70b-versatile", help="LLM model name")
    ask.add_argument(
        "--reference-answer",
        default=None,
        help="Optional ground-truth answer for this question (used for Semantic Answer Similarity)",
    )
    ask.add_argument(
        "--relevant-chunk-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional relevant chunk IDs for this question (used for Recall@K)",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate RAG with Faithfulness, Recall@K, Semantic Answer Similarity, and Query-Answer Embedding Similarity",
    )
    evaluate.add_argument("--eval-file", required=True, help="JSONL file with evaluation cases")
    evaluate.add_argument("--persist-dir", default="faiss_transcript_store", help="FAISS directory to query")
    evaluate.add_argument("--top-k", type=int, default=5, help="Top-K retrieved transcript segments")
    evaluate.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    evaluate.add_argument("--llm-model", default="llama-3.3-70b-versatile", help="LLM model name")
    evaluate.add_argument("--output-path", default="audio/eval_results.json", help="Path to save evaluation JSON")

    return parser


def run_ingest(args: argparse.Namespace) -> None:
    transcriptions = transcribe_long_audio(
        audio_path=args.audio_path,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        language=args.language,
        whisper_prompt=args.whisper_prompt,
    )
    save_transcriptions_jsonl(transcriptions, args.transcript_path)
    build_faiss_from_transcriptions(
        chunks=transcriptions,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        segment_chars=args.segment_chars,
        segment_overlap=args.segment_overlap,
    )
    print("[INFO] Ingestion complete.")


def run_ask(args: argparse.Namespace) -> None:
    rag = TranscriptRAG(
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
    )
    details = rag.ask_with_details(args.question, top_k=args.top_k)
    answer = details["answer"]
    print("\n[ANSWER]")
    print(answer)

    retrieved_chunk_ids = []
    for item in details["retrieved"]:
        chunk_id = item.get("chunk_id")
        if chunk_id is None:
            continue
        try:
            retrieved_chunk_ids.append(int(chunk_id))
        except (TypeError, ValueError):
            continue

    faithfulness = faithfulness_score(
        llm=rag.llm,
        question=args.question,
        answer=answer,
        contexts=details["context_blocks"],
    )

    semantic_answer_similarity = None
    if args.reference_answer:
        semantic_answer_similarity = embedding_cosine_similarity(
            embed_model=rag.vectorstore.model,
            text_a=answer,
            text_b=args.reference_answer,
        )

    recall = None
    if args.relevant_chunk_ids:
        recall = recall_at_k(retrieved_chunk_ids, args.relevant_chunk_ids, k=args.top_k)

    query_answer_similarity = embedding_cosine_similarity(
        embed_model=rag.vectorstore.model,
        text_a=args.question,
        text_b=answer,
    )

    print("\n[METRICS]")
    print(f"Faithfulness: {faithfulness:.3f}")
    print(
        f"Recall@{args.top_k}: "
        f"{f'{recall:.3f}' if recall is not None else 'n/a (provide --relevant-chunk-ids)'}"
    )
    print(
        "Semantic Answer Similarity: "
        f"{f'{semantic_answer_similarity:.3f}' if semantic_answer_similarity is not None else 'n/a (provide --reference-answer)'}"
    )
    print(f"Query-Answer Embedding Similarity: {query_answer_similarity:.3f}")


def run_evaluate(args: argparse.Namespace) -> None:
    rag = TranscriptRAG(
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
    )

    cases = []
    with open(args.eval_file, "r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                case = parse_eval_case(payload)
            except ValueError as exc:
                raise ValueError(f"Invalid eval case at line {line_no}: {exc}") from exc
            cases.append(case)

    if not cases:
        raise ValueError("No evaluation cases found in eval file.")

    per_case = []
    recall_scores = []
    faithfulness_scores = []
    semantic_scores = []
    query_answer_scores = []

    for idx, case in enumerate(cases, start=1):
        details = rag.ask_with_details(case.question, top_k=args.top_k)
        answer = details["answer"]

        retrieved_chunk_ids = []
        for item in details["retrieved"]:
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                continue
            try:
                retrieved_chunk_ids.append(int(chunk_id))
            except (TypeError, ValueError):
                continue

        case_recall = None
        if case.relevant_chunk_ids:
            case_recall = recall_at_k(retrieved_chunk_ids, case.relevant_chunk_ids, k=args.top_k)
            recall_scores.append(case_recall)

        case_faithfulness = faithfulness_score(
            llm=rag.llm,
            question=case.question,
            answer=answer,
            contexts=details["context_blocks"],
        )
        faithfulness_scores.append(case_faithfulness)

        case_semantic = embedding_cosine_similarity(
            embed_model=rag.vectorstore.model,
            text_a=answer,
            text_b=case.reference_answer,
        )
        semantic_scores.append(case_semantic)

        case_query_answer = embedding_cosine_similarity(
            embed_model=rag.vectorstore.model,
            text_a=case.question,
            text_b=answer,
        )
        query_answer_scores.append(case_query_answer)

        result = {
            "case_id": idx,
            "question": case.question,
            "reference_answer": case.reference_answer,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "relevant_chunk_ids": case.relevant_chunk_ids,
            "recall_at_k": case_recall,
            "faithfulness": case_faithfulness,
            "semantic_answer_similarity": case_semantic,
            "query_answer_embedding_similarity": case_query_answer,
            "answer": answer,
        }
        per_case.append(result)
        print(
            f"[EVAL] case={idx} "
            f"Recall@{args.top_k}={case_recall if case_recall is not None else 'n/a'} "
            f"Faithfulness={case_faithfulness:.3f} "
            f"SemanticAnsSim={case_semantic:.3f} "
            f"QueryAnsSim={case_query_answer:.3f}"
        )

    summary = {
        "num_cases": len(per_case),
        "top_k": args.top_k,
        "recall_at_k_mean": mean(recall_scores) if recall_scores else None,
        "faithfulness_mean": mean(faithfulness_scores) if faithfulness_scores else 0.0,
        "semantic_answer_similarity_mean": mean(semantic_scores) if semantic_scores else 0.0,
        "query_answer_embedding_similarity_mean": mean(query_answer_scores) if query_answer_scores else 0.0,
    }

    output = {"summary": summary, "cases": per_case}
    with open(args.output_path, "w", encoding="utf-8") as file_obj:
        json.dump(output, file_obj, indent=2, ensure_ascii=True)

    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2))
    print(f"\n[INFO] Saved evaluation results to {args.output_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "ask":
        run_ask(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()