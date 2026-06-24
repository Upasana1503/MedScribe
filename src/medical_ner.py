from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from transformers import pipeline

NER_MODEL = "d4data/biomedical-ner-all"

_ner_pipeline = None


def _get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        print(f"[INFO] Loading medical NER model: {NER_MODEL}")
        _ner_pipeline = pipeline(
            "ner",
            model=NER_MODEL,
            aggregation_strategy="simple",
        )
        print("[INFO] Medical NER model loaded.")
    return _ner_pipeline


ENTITY_GROUP_MAP = {
    "Detailed_description": "findings",
    "Biological_structure": "anatomy",
    "Disease_disorder": "conditions",
    "Medication": "medications",
    "Sign_symptom": "symptoms",
    "Therapeutic_procedure": "procedures",
    "Diagnostic_procedure": "procedures",
    "Lab_value": "vitals",
    "Clinical_event": "findings",
    "Dosage": "dosages",
    "Duration": "temporal",
    "Frequency": "temporal",
    "Date": "temporal",
    "Age": "demographics",
    "Sex": "demographics",
    "Activity": "findings",
    "Severity": "findings",
    "Area": "anatomy",
    "Shape": "findings",
    "Color": "findings",
    "Mass": "findings",
    "Volume": "findings",
    "Distance": "findings",
    "Administration": "medications",
    "Family_history": "history",
    "History": "history",
    "Outcome": "findings",
    "Subject": "demographics",
    "Nonbiological_location": "findings",
    "Biological_attribute": "findings",
    "Quantitative_concept": "findings",
    "Qualitative_concept": "findings",
    "Coreference": "findings",
    "Other_entity": "findings",
    "Other_event": "findings",
}


@dataclass
class MedicalEntities:
    medications: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    vitals: List[str] = field(default_factory=list)
    anatomy: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    demographics: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)
    temporal: List[str] = field(default_factory=list)
    dosages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {k: v for k, v in asdict(self).items() if v}

    def merge(self, other: MedicalEntities) -> None:
        for category in asdict(self):
            existing = getattr(self, category)
            new_items = getattr(other, category)
            seen = {item.lower() for item in existing}
            for item in new_items:
                if item.lower() not in seen:
                    existing.append(item)
                    seen.add(item.lower())

    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in asdict(self).values())


def extract_entities(text: str, min_score: float = 0.5) -> MedicalEntities:
    if not text or not text.strip():
        return MedicalEntities()

    ner = _get_ner_pipeline()
    entities = MedicalEntities()

    chunks = _split_for_ner(text, max_length=512)
    seen: Dict[str, set] = {}

    for chunk in chunks:
        results = ner(chunk)
        for ent in results:
            if ent["score"] < min_score:
                continue

            word = ent["word"].strip()
            if len(word) < 2:
                continue

            group = ent.get("entity_group", "")
            category = ENTITY_GROUP_MAP.get(group, "findings")

            if category not in seen:
                seen[category] = set()

            word_lower = word.lower()
            if word_lower not in seen[category]:
                seen[category].add(word_lower)
                getattr(entities, category).append(word)

    return entities


def _split_for_ner(text: str, max_length: int = 512) -> List[str]:
    words = text.split()
    if len(words) <= max_length:
        return [text]

    chunks = []
    current: List[str] = []
    for word in words:
        current.append(word)
        if len(current) >= max_length:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def save_entities(entities: MedicalEntities, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entities.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved medical entities to {output_path}")


def load_entities(input_path: str) -> MedicalEntities:
    if not Path(input_path).is_file():
        return MedicalEntities()
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entities = MedicalEntities()
    for category, items in data.items():
        if hasattr(entities, category) and isinstance(items, list):
            setattr(entities, category, items)
    return entities
