from __future__ import annotations

import math
import re
import secrets
from typing import Any


MODES = {"story", "interactive", "my_story"}
AUDIENCES = {"young", "family", "teen", "adult"}
VOICES = {f"{kind}{number}" for kind in ("F", "M") for number in range(1, 6)}
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish · suomi",
    "fr": "French · français",
    "de": "German · Deutsch",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian · italiano",
    "ja": "Japanese · 日本語",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "pl": "Polish",
    "pt": "Portuguese · português",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish · español",
    "sv": "Swedish · svenska",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
}
LANGUAGES = set(LANGUAGE_NAMES)
TRANSLATION_LANGUAGES = set(LANGUAGES)
DEFAULT_TRANSLATION_LANGUAGE = "fi"
CANONICAL_LANGUAGE = "en"

CINEMA_DEFAULTS = {
    "width": 480,
    "height": 272,
    "frames": 81,
    "fps": 16,
    "min_words": 80,
    "max_words": 110,
    "max_slow": 8.0,
}
DEFAULT_CONTEXT_COMPACTION_SCENES = 30
MAX_PROMPT_CHARS = 1_200
MAX_STORY_BYTES = 64 * 1024 * 1024


class TheaterError(RuntimeError):
    pass


def split_narration_sentences(text: str) -> list[str]:
    """Split multilingual narration without an online tokenizer."""
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return []
    sentences: list[str] = []
    start = 0
    index = 0
    closers = {'"', "'", "\u201d", "\u2019", "\u00bb", ")", "]", "}", "\u300d", "\u300f"}
    while index < len(normalized):
        character = normalized[index]
        is_cjk_end = character in "\u3002\uff01\uff1f"
        is_spaced_end = character in ".!?" and (
            index + 1 == len(normalized)
            or normalized[index + 1].isspace()
            or normalized[index + 1] in closers
        )
        if is_cjk_end or is_spaced_end:
            end = index + 1
            while end < len(normalized) and normalized[end] in closers:
                end += 1
            if is_cjk_end or end == len(normalized) or normalized[end].isspace():
                sentence = normalized[start:end].strip()
                if sentence:
                    sentences.append(sentence)
                while end < len(normalized) and normalized[end].isspace():
                    end += 1
                start = end
                index = end
                continue
        index += 1
    tail = normalized[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def spoken_word_count(text: str, language: str = "en") -> int:
    """Count stable duration units without treating unspaced Japanese as one word."""
    value = str(text)
    if str(language).lower() == "ja":
        japanese = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", value)
        remainder = re.sub(r"[\u3040-\u30ff\u3400-\u9fff]", " ", value)
        latin_words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", remainder, flags=re.UNICODE)
        return math.ceil(len(japanese) / 2) + len(latin_words)
    return len(re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", value, flags=re.UNICODE))


def translation_language(config: dict[str, Any]) -> str:
    source = str(config.get("language", "en")).lower()
    target = str(config.get("translation_language") or "").lower()
    if target in TRANSLATION_LANGUAGES and target != source:
        return target
    return "en" if source == DEFAULT_TRANSLATION_LANGUAGE else DEFAULT_TRANSLATION_LANGUAGE


def planning_language(config: dict[str, Any]) -> str:
    """Generated worlds are canonical English; pasted narration stays immutable."""
    return str(config.get("language", "en")).lower() if config.get("mode") == "my_story" else CANONICAL_LANGUAGE


def _remaining(expected: float, started: float, now: float) -> float:
    if expected <= 0:
        return 0.0
    return max(0.0, expected - max(0.0, now - started)) if started else expected


def attention_eta(state: dict[str, Any], now: float) -> int | None:
    """Estimate time to playable media from rolling, local measurements only."""
    status = str(state.get("status", ""))
    metrics = state.get("metrics", {}) if isinstance(state.get("metrics"), dict) else {}
    planner = float(metrics.get("planner_cycle_ema") or metrics.get("planner_elapsed_ema") or 0)
    translation = float(metrics.get("translation_cycle_ema") or metrics.get("translation_elapsed_ema") or 0)
    video = float(metrics.get("video_seconds_ema") or metrics.get("last_video_seconds") or 0)
    speech = float(metrics.get("tts_seconds_ema") or metrics.get("last_tts_seconds") or 0)
    alignment = float(metrics.get("alignment_seconds_ema") or 0)
    assembly = float(metrics.get("assembly_seconds_ema") or metrics.get("last_assembly_seconds") or 0)
    planner_started = float(metrics.get("planner_request_started_at") or metrics.get("planner_cycle_started_at") or 0)
    translation_started = float(metrics.get("translation_request_started_at") or metrics.get("translation_cycle_started_at") or 0)
    video_started = float(metrics.get("video_started_at") or 0)
    speech_started = float(metrics.get("tts_started_at") or 0)
    alignment_started = float(metrics.get("alignment_started_at") or 0)
    assembly_started = float(metrics.get("assembly_started_at") or 0)

    if status in {"starting", "planning"}:
        total = _remaining(planner, planner_started, now) + _remaining(translation, translation_started, now)
        total += max(video, speech) + alignment + assembly
    elif status == "generating":
        total = max(_remaining(video, video_started, now), _remaining(speech, speech_started, now))
        total += alignment + assembly
    elif status == "narrating":
        total = _remaining(speech, speech_started, now) + alignment + assembly
    elif status == "aligning":
        total = _remaining(alignment, alignment_started, now) + assembly
    elif status == "buffering":
        total = _remaining(assembly, assembly_started, now)
    else:
        total = float(metrics.get("completion_interval_ema") or 0)
    return max(1, round(total)) if total > 0 else None


def quality_settings(config: dict[str, Any]) -> dict[str, Any]:
    custom = config.get("quality_settings")
    return {**CINEMA_DEFAULTS, **custom} if isinstance(custom, dict) else dict(CINEMA_DEFAULTS)


def narration_word_limits(config: dict[str, Any]) -> tuple[int, int]:
    quality = quality_settings(config)
    minimum = max(12, math.ceil(float(quality["min_words"]) / 2.1))
    maximum = max(minimum, math.floor(float(quality["max_words"]) / 2.1))
    return minimum, maximum


def spoken_text(pairs: list[dict[str, str]]) -> str:
    """Return the exact source/translation sequence sent to the speech adapter."""
    parts: list[str] = []
    for pair in pairs:
        original = str(pair.get("original", "")).strip()
        translation = str(pair.get("translation", "")).strip()
        if original:
            parts.append(original)
        if translation:
            parts.append(translation)
    return " ".join(parts)


def normalize_story_text(text: str) -> str:
    value = re.sub(r"([。！？])(?=\S)", r"\1 ", "" if text is None else str(text))
    return re.sub(r"\s+", " ", value).strip()


def take_story_chunk(text: str, language: str, minimum: int, maximum: int, *, final: bool) -> tuple[str, int]:
    """Select one narration-sized prefix and report consumed characters.

    Sentence boundaries win. A single unusually long sentence is cut on a word
    boundary so a pasted book can always continue with bounded memory.
    """
    leading = len(text) - len(text.lstrip())
    value = text[leading:]
    if not value:
        return "", len(text)
    sentences = split_narration_sentences(value)
    selected: list[str] = []
    selected_words = 0
    for index, sentence in enumerate(sentences):
        words = spoken_word_count(sentence, language)
        if selected and selected_words + words > maximum:
            break
        if not selected and words > maximum:
            if language == "ja":
                char_limit = max(1, maximum * 2)
                chunk = sentence[:char_limit].rstrip()
            else:
                matches = list(re.finditer(r"[^\W_]+(?:['’-][^\W_]+)*", sentence, flags=re.UNICODE))
                cut = matches[min(maximum, len(matches)) - 1].end() if matches else len(sentence)
                chunk = sentence[:cut].rstrip()
            return chunk, leading + len(chunk)
        selected.append(sentence)
        selected_words += words
        if selected_words >= minimum:
            next_sentence = sentences[index + 1] if index + 1 < len(sentences) else ""
            if not next_sentence or selected_words + spoken_word_count(next_sentence, language) > maximum:
                break
    chunk = " ".join(selected).strip()
    if not chunk or (not final and selected_words < minimum and len(sentences) == 1):
        return "", 0
    return chunk, leading + len(chunk)


def validate_story_request(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Request body must be a JSON object.")
    mode = str(raw.get("mode", "story"))
    if mode not in MODES:
        raise ValueError("Choose Pure Story, Interactive Character Show, or My story.")

    story_text = normalize_story_text(raw.get("story_text", "")) if mode == "my_story" else ""
    prompt = str(raw.get("prompt", "")).strip()
    if mode == "my_story":
        if not story_text:
            raise ValueError("My story is empty.")
        if len(story_text.encode("utf-8")) > MAX_STORY_BYTES:
            raise ValueError("My story is larger than the 64 MiB local limit.")
        prompt = "My story"
    else:
        if not prompt:
            raise ValueError("Write a story idea first.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError("The story idea is too long (maximum 1,200 characters).")

    audience = str(raw.get("audience", "family"))
    if audience not in AUDIENCES:
        raise ValueError("Choose a supported audience level.")
    voice = str(raw.get("voice", "M1")).upper()
    language = str(raw.get("language", "en")).lower()
    target_language = str(raw.get("translation_language") or "").lower()
    if not target_language:
        target_language = "en" if language == DEFAULT_TRANSLATION_LANGUAGE else DEFAULT_TRANSLATION_LANGUAGE
    if voice not in VOICES:
        raise ValueError("Choose a supported Supertonic voice.")
    if language not in LANGUAGES:
        raise ValueError("Choose a supported narration language.")
    if target_language not in TRANSLATION_LANGUAGES:
        raise ValueError("Choose a supported translation language.")
    if target_language == language:
        raise ValueError("The translation language must differ from the story language.")

    supplied = raw.get("quality_settings", {})
    if not isinstance(supplied, dict):
        raise ValueError("Advanced generation settings must be an object.")
    try:
        width = int(supplied.get("width", CINEMA_DEFAULTS["width"]))
        height = int(supplied.get("height", CINEMA_DEFAULTS["height"]))
        frames = int(supplied.get("frames", CINEMA_DEFAULTS["frames"]))
        fps = int(supplied.get("fps", CINEMA_DEFAULTS["fps"]))
        min_words = int(supplied.get("min_words", CINEMA_DEFAULTS["min_words"]))
        max_words = int(supplied.get("max_words", CINEMA_DEFAULTS["max_words"]))
        max_slow = float(supplied.get("max_slow", CINEMA_DEFAULTS["max_slow"]))
        compaction = int(raw.get("context_compaction_scenes", DEFAULT_CONTEXT_COMPACTION_SCENES))
        seed = int(raw.get("seed", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Advanced generation values must be numeric.") from exc
    if width < 192 or width > 832 or width % 16:
        raise ValueError("Theater width must be a multiple of 16 between 192 and 832.")
    if height < 192 or height > 832 or height % 16:
        raise ValueError("Theater height must be a multiple of 16 between 192 and 832.")
    if frames < 9 or frames > 81 or (frames - 1) % 4:
        raise ValueError("Theater source frames must be 9-81 and follow the 4n+1 rule.")
    if fps < 1 or fps > 60:
        raise ValueError("Theater playback FPS must be between 1 and 60.")
    if min_words < 30 or min_words > 1_200:
        raise ValueError("Minimum narration words must be between 30 and 1,200.")
    if max_words < min_words or max_words > 2_400:
        raise ValueError("Maximum narration words must be at least the minimum and no more than 2,400.")
    if max_slow < 1 or max_slow > 20:
        raise ValueError("Maximum slow-motion must be between 1x and 20x.")
    if compaction != 0 and not 5 <= compaction <= 200:
        raise ValueError("Context compaction must be 0 (interval off) or between 5 and 200 scenes.")
    if seed < 0:
        seed = secrets.randbelow(2**31 - 1)

    config = {
        "prompt": prompt,
        "quality_settings": {
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "min_words": min_words,
            "max_words": max_words,
            "max_slow": max_slow,
        },
        "mode": mode,
        "audience": audience,
        "seed": seed,
        "voice": voice,
        "language": language,
        "translation_language": target_language,
        "context_compaction_scenes": compaction,
    }
    if story_text:
        config["_story_text"] = story_text
    return config
