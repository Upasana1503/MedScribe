import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DEVICE = "cpu"
MODEL_NAME = "openai/whisper-medium"  
LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
print("[INFO] Loading Whisper model...")

try:
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True,
        local_files_only=LOCAL_FILES_ONLY,
    )
except Exception as exc:
    mode = "local cache only" if LOCAL_FILES_ONLY else "online download enabled"
    raise RuntimeError(
        f"Failed to load Whisper model '{MODEL_NAME}' ({mode}). "
        "If this is first run, set HF_LOCAL_FILES_ONLY=0 and ensure internet connectivity."
    ) from exc

model.eval()
model.to("cpu")

DEFAULT_MEDICAL_WHISPER_PROMPT = (
    "ROLE:\n"
    "You are a medical transcript normalization system.\n\n"
    "CONTEXT:\n"
    "The input is an automatic speech recognition transcript\n"
    "from a clinical consultation.\n\n"
    "OBJECTIVE:\n"
    "Correct spelling errors in medical terminology, drug names,\n"
    "diagnoses, and anatomical terms.\n\n"
    "STRICT RULES:\n"
    "- Preserve meaning exactly.\n"
    "- Do NOT add new medical information.\n"
    "- Do NOT remove content.\n"
    "- Preserve dosages, numbers, and units exactly.\n\n"
    "OUTPUT:\n"
    "Return only the corrected transcript in plain text.\n"
    "No explanations."
)


def _prepare_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio is None:
        raise ValueError("audio is None")

    audio = np.asarray(audio)
    if audio.ndim > 1:
        # Convert multi-channel to mono
        audio = audio.mean(axis=1)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    return audio.astype(np.float32)


def _build_generate_kwargs(language: str, prompt_hint: str | None) -> dict:
    generate_kwargs = {
        "task": "transcribe",
        "language": language,
    }

    prompt_text = (prompt_hint or "").strip()
    if not prompt_text:
        return generate_kwargs

    get_prompt_ids = getattr(processor, "get_prompt_ids", None)
    if callable(get_prompt_ids):
        try:
            prompt_ids = get_prompt_ids(prompt_text)
            if prompt_ids is not None:
                generate_kwargs["prompt_ids"] = prompt_ids
        except Exception:
            # Fall back to plain decoding when prompt tokenization is unavailable.
            pass
    return generate_kwargs


def transcribe_audio_array(
    audio: np.ndarray,
    sr: int = 16000,
    language: str = "en",
    prompt_hint: str | None = DEFAULT_MEDICAL_WHISPER_PROMPT,
):
    audio = _prepare_audio(audio, sr)

    # Prepare Whisper input
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)
    generate_kwargs = _build_generate_kwargs(language=language, prompt_hint=prompt_hint)

    with torch.no_grad():
        try:
            predicted_ids = model.generate(input_features, **generate_kwargs)
        except TypeError:
            # Backward compatibility for transformer versions without prompt_ids support.
            generate_kwargs.pop("prompt_ids", None)
            predicted_ids = model.generate(input_features, **generate_kwargs)

    text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    confidence = 1.0  # Whisper doesn't expose confidence cleanly

    return text.strip(), confidence


def transcribe_audio(
    audio_path,
    language="en",
    prompt_hint: str | None = DEFAULT_MEDICAL_WHISPER_PROMPT,
):
    # Load audio from file, then reuse array path
    audio, sr = librosa.load(audio_path, sr=None)
    return transcribe_audio_array(audio, sr=sr, language=language, prompt_hint=prompt_hint)
