from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    return normalized.split() if normalized else []


def answer_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    common = 0
    for token, count in pred_counts.items():
        common += min(count, ref_counts.get(token, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_chunk_ids: Sequence[int], relevant_chunk_ids: Sequence[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    relevant = set(int(x) for x in relevant_chunk_ids)
    if not relevant:
        return 0.0

    retrieved_top_k = [int(x) for x in retrieved_chunk_ids[:k]]
    hits = relevant.intersection(retrieved_top_k)
    return len(hits) / len(relevant)


def _extract_first_float(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def faithfulness_score(
    llm: Any,
    question: str,
    answer: str,
    contexts: Iterable[str],
) -> float:
    joined_context = "\n\n".join(contexts)
    if not joined_context.strip():
        return 0.0

    prompt = (
        "You are grading groundedness for a RAG answer.\n"
        "Score faithfulness from 0.0 to 1.0 where:\n"
        "1.0 means every factual claim in the answer is supported by the provided context,\n"
        "0.0 means unsupported or contradicted.\n"
        "Return strict JSON only: {\"faithfulness\": <float>}.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Context:\n{joined_context}\n"
    )

    response = llm.invoke(prompt)
    raw_text = getattr(response, "content", str(response)).strip()

    score: float | None = None
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            value = parsed.get("faithfulness")
            if value is not None:
                score = float(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        score = None

    if score is None:
        score = _extract_first_float(raw_text)
    if score is None:
        return 0.0

    return max(0.0, min(1.0, score))


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def estimate_relevant_chunk_ids(
    llm: Any,
    question: str,
    chunk_candidates: Sequence[Dict[str, Any]],
) -> List[int]:
    if not chunk_candidates:
        return []

    filtered = []
    seen = set()
    for item in chunk_candidates:
        chunk_id = item.get("chunk_id")
        text = str(item.get("text", "")).strip()
        if chunk_id is None or not text:
            continue
        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        filtered.append({"chunk_id": chunk_id, "text": text[:600]})

    if not filtered:
        return []

    chunks_block = "\n\n".join(
        f"chunk_id={item['chunk_id']}\n{item['text']}" for item in filtered
    )

    prompt = (
        "You are identifying relevant transcript chunks for a question.\n"
        "From the provided chunk list, return only chunk IDs relevant to answering the question.\n"
        "Return strict JSON only: {\"relevant_chunk_ids\": [int, ...]}.\n\n"
        f"Question:\n{question}\n\n"
        f"Chunks:\n{chunks_block}\n"
    )

    response = llm.invoke(prompt)
    raw_text = getattr(response, "content", str(response)).strip()
    parsed = _extract_json_object(raw_text)
    if not parsed:
        return []

    values = parsed.get("relevant_chunk_ids")
    if not isinstance(values, list):
        return []

    allowed = {item["chunk_id"] for item in filtered}
    relevant: List[int] = []
    for value in values:
        try:
            chunk_id = int(value)
        except (TypeError, ValueError):
            continue
        if chunk_id in allowed and chunk_id not in relevant:
            relevant.append(chunk_id)
    return relevant


def generate_reference_answer(
    llm: Any,
    question: str,
    contexts: Iterable[str],
) -> str:
    joined_context = "\n\n".join(contexts).strip()
    if not joined_context:
        return ""

    prompt = (
        "You are writing an ideal reference answer for evaluation.\n"
        "Use only the provided context.\n"
        "Write a concise factual answer in plain text.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{joined_context}\n"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()


def embedding_cosine_similarity(embed_model: Any, text_a: str, text_b: str) -> float:
    text_a = str(text_a or "").strip()
    text_b = str(text_b or "").strip()
    if not text_a or not text_b:
        return 0.0

    vectors = embed_model.encode([text_a, text_b])
    a = np.asarray(vectors[0], dtype=np.float32)
    b = np.asarray(vectors[1], dtype=np.float32)

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0

    score = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, score))


@dataclass
class EvaluationCase:
    question: str
    reference_answer: str
    relevant_chunk_ids: List[int]


def parse_eval_case(payload: Dict[str, Any]) -> EvaluationCase:
    question = str(payload.get("question", "")).strip()
    reference_answer = str(
        payload.get("reference_answer", payload.get("ground_truth_answer", payload.get("answer", "")))
    ).strip()

    raw_ids = payload.get("relevant_chunk_ids", [])
    if raw_ids is None:
        raw_ids = []
    if not isinstance(raw_ids, list):
        raise ValueError("relevant_chunk_ids must be a list of integers")

    relevant_chunk_ids: List[int] = []
    for item in raw_ids:
        relevant_chunk_ids.append(int(item))

    if not question:
        raise ValueError("question is required")
    if not reference_answer:
        raise ValueError("reference_answer (or ground_truth_answer/answer) is required")

    return EvaluationCase(
        question=question,
        reference_answer=reference_answer,
        relevant_chunk_ids=relevant_chunk_ids,
    )
