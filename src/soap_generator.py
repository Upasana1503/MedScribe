from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.medical_ner import MedicalEntities

SOAP_PROMPT_TEMPLATE = (
    "ROLE:\n"
    "You are a clinical documentation assistant.\n\n"
    "CONTEXT:\n"
    "You are given a raw transcript from a doctor-patient consultation\n"
    "and a list of medical entities extracted from it.\n\n"
    "OBJECTIVE:\n"
    "Generate a structured clinical SOAP note from this consultation.\n\n"
    "FORMAT:\n"
    "SUBJECTIVE:\n"
    "[Patient's chief complaint, history of present illness, symptoms reported by patient,\n"
    " relevant past medical history, family history, social history, review of systems.\n"
    " Use only what the patient or doctor stated.]\n\n"
    "OBJECTIVE:\n"
    "[Physical examination findings, vital signs, lab results, imaging results,\n"
    " any measurable or observable data mentioned.]\n\n"
    "ASSESSMENT:\n"
    "[Clinical diagnosis or differential diagnoses discussed,\n"
    " clinical reasoning linking subjective and objective findings.]\n\n"
    "PLAN:\n"
    "[Treatment plan, medications prescribed with dosages,\n"
    " procedures ordered, follow-up instructions, referrals,\n"
    " patient education provided.]\n\n"
    "STRICT RULES:\n"
    "- Use ONLY information from the transcript and extracted entities.\n"
    "- Do NOT invent or assume any medical information.\n"
    "- If a SOAP section has no relevant information, write: \"Not documented.\"\n"
    "- Be concise and clinically precise.\n"
    "- Include specific drug names, dosages, and frequencies when mentioned.\n"
    "- Use standard medical abbreviations where appropriate.\n\n"
    "EXTRACTED MEDICAL ENTITIES:\n"
    "{entities_block}\n\n"
    "CONSULTATION TRANSCRIPT:\n"
    "{transcript}\n\n"
    "SOAP NOTE:"
)


@dataclass
class SOAPNote:
    subjective: str
    objective: str
    assessment: str
    plan: str
    raw_text: str

    def to_dict(self) -> dict:
        return {
            "subjective": self.subjective,
            "objective": self.objective,
            "assessment": self.assessment,
            "plan": self.plan,
            "raw_text": self.raw_text,
        }


def _format_entities_block(entities: MedicalEntities) -> str:
    data = entities.to_dict()
    if not data:
        return "No entities extracted."
    lines = []
    for category, items in data.items():
        label = category.replace("_", " ").title()
        lines.append(f"- {label}: {', '.join(items)}")
    return "\n".join(lines)


def _parse_soap_sections(raw: str) -> SOAPNote:
    sections = {
        "subjective": "",
        "objective": "",
        "assessment": "",
        "plan": "",
    }

    current_section = None
    current_lines = []

    for line in raw.split("\n"):
        stripped = line.strip().rstrip(":")
        lower = stripped.lower()

        if lower in ("subjective", "s"):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "subjective"
            current_lines = []
        elif lower in ("objective", "o"):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "objective"
            current_lines = []
        elif lower in ("assessment", "a"):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "assessment"
            current_lines = []
        elif lower in ("plan", "p"):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = "plan"
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return SOAPNote(
        subjective=sections["subjective"],
        objective=sections["objective"],
        assessment=sections["assessment"],
        plan=sections["plan"],
        raw_text=raw.strip(),
    )


def generate_soap_note(
    llm: Any,
    transcript: str,
    entities: MedicalEntities,
) -> SOAPNote:
    entities_block = _format_entities_block(entities)
    prompt = SOAP_PROMPT_TEMPLATE.format(
        entities_block=entities_block,
        transcript=transcript[:8000],
    )
    response = llm.invoke(prompt)
    raw_text = getattr(response, "content", str(response)).strip()
    return _parse_soap_sections(raw_text)
