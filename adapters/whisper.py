from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import math
import shutil
import threading
import time
from pathlib import Path
from typing import Any


class WhisperAlignmentError(RuntimeError):
    pass


class WhisperAlignmentAdapter:
    """Verified local Whisper large-v3-turbo word-alignment boundary.

    This mirrors Kestrel's offline checkpoint contract. The model is released
    after every alignment so Wan remains the sole long-lived GPU workload.
    """

    MODEL_NAME = "large-v3-turbo"
    MODEL_SHA256 = "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a"
    MODEL_ID = "whisper:large-v3-turbo"
    MAX_TIMINGS = 20_000

    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        self.model_path = model_root / f"{self.MODEL_NAME}.pt"
        self._verified_signature: tuple[int, int] | None = None
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()

    def available(self) -> bool:
        return self.model_path.is_file()

    def _verify(self) -> None:
        try:
            stat = self.model_path.stat()
        except FileNotFoundError as exc:
            raise WhisperAlignmentError(
                f"Whisper {self.MODEL_NAME} is missing at {self.model_path}."
            ) from exc
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature == self._verified_signature:
            return
        digest = hashlib.sha256()
        with self.model_path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != self.MODEL_SHA256:
            raise WhisperAlignmentError("Whisper large-v3-turbo failed its offline integrity check.")
        self._verified_signature = signature

    @staticmethod
    def _finite(value: Any, fallback: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        return number if math.isfinite(number) and number >= 0.0 else fallback

    def _alignment_parts(self, audio_path: Path) -> tuple[Path | None, list[dict[str, Any]]]:
        part_dir = audio_path.parent / f".{audio_path.stem}_parts"
        manifest = part_dir / "alignment.json"
        if not manifest.is_file():
            return None, []
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            raw_parts = payload.get("parts")
            if not isinstance(raw_parts, list) or not raw_parts:
                raise ValueError("empty parts")
            parts: list[dict[str, Any]] = []
            for raw in raw_parts:
                path = (part_dir / str(raw["file"])).resolve()
                path.relative_to(part_dir.resolve())
                if not path.is_file():
                    raise FileNotFoundError(path)
                parts.append({
                    "path": path,
                    "text": str(raw.get("text", "")).strip(),
                    "language": str(raw.get("language", "auto")).strip().lower(),
                    "start": self._finite(raw.get("start")),
                })
            return part_dir, parts
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WhisperAlignmentError("The bilingual narration alignment manifest is invalid.") from exc

    def _align_sync(self, audio_path: Path, expected_text: str, language: str) -> dict[str, Any]:
        with self._thread_lock:
            self._verify()
            try:
                import torch
                import whisper
            except ImportError as exc:
                raise WhisperAlignmentError(
                    "The ComfyUI Python environment with openai-whisper is required for word alignment."
                ) from exc

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = None
            part_dir: Path | None = None
            started = time.perf_counter()
            try:
                part_dir, parts = self._alignment_parts(audio_path)
                if not parts:
                    parts = [{
                        "path": audio_path,
                        "text": expected_text,
                        "language": language,
                        "start": 0.0,
                    }]
                model = whisper.load_model(
                    self.MODEL_NAME,
                    device=device,
                    download_root=str(self.model_root),
                )
                segments: list[dict[str, Any]] = []
                words: list[dict[str, Any]] = []
                transcripts: list[str] = []
                detected_languages: list[str] = []
                for part_index, part in enumerate(parts):
                    part_language = str(part["language"])
                    result = model.transcribe(
                        str(part["path"]),
                        language=None if part_language in {"", "auto", "na"} else part_language,
                        initial_prompt=part["text"] or None,
                        word_timestamps=True,
                        condition_on_previous_text=False,
                        fp16=device == "cuda",
                        verbose=None,
                    )
                    offset = float(part["start"])
                    transcript = str(result.get("text", "")).strip()
                    if transcript:
                        transcripts.append(transcript)
                    detected_languages.append(str(result.get("language", part_language)))
                    for raw_segment in result.get("segments", []):
                        segment = {
                            "value": str(raw_segment.get("text", "")).strip(),
                            "start": offset + self._finite(raw_segment.get("start")),
                            "end": offset + self._finite(raw_segment.get("end")),
                            "part": part_index,
                            "language": part_language,
                        }
                        segment["end"] = max(segment["start"], segment["end"])
                        if segment["value"]:
                            segments.append(segment)
                        for raw_word in raw_segment.get("words") or []:
                            word = {
                                "value": str(raw_word.get("word", "")).strip(),
                                "start": offset + self._finite(raw_word.get("start")),
                                "end": offset + self._finite(raw_word.get("end")),
                                "part": part_index,
                                "language": part_language,
                            }
                            word["end"] = max(word["start"], word["end"])
                            if word["value"]:
                                words.append(word)
                if not words or len(words) > self.MAX_TIMINGS:
                    raise WhisperAlignmentError("Whisper returned no safe word alignment for this scene.")
                return {
                    "model": self.MODEL_ID,
                    "transcript": " ".join(transcripts),
                    "segments": segments,
                    "words": words,
                    "parts": len(parts),
                    "languages": detected_languages,
                    "device": device,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            finally:
                if model is not None:
                    del model
                gc.collect()
                if "torch" in locals() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if part_dir is not None:
                    shutil.rmtree(part_dir, ignore_errors=True)

    async def align(self, audio_path: Path, expected_text: str, language: str = "auto") -> dict[str, Any]:
        if not audio_path.is_file():
            raise WhisperAlignmentError(f"Narration audio is missing: {audio_path}")
        async with self._lock:
            return await asyncio.to_thread(self._align_sync, audio_path, expected_text, language)
