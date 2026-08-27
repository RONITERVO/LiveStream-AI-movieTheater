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


def quality_settings(config: dict[str, Any]) -> dict[str, Any]:
    custom = config.get("quality_settings")
    return {**CINEMA_DEFAULTS, **custom} if isinstance(custom, dict) else dict(CINEMA_DEFAULTS)


def narration_word_limits(config: dict[str, Any]) -> tuple[int, int]:
    quality = quality_settings(config)
    minimum = max(12, math.ceil(float(quality["min_words"]) / 2.1))
    maximum = max(minimum, math.floor(float(quality["max_words"]) / 2.1))
    return minimum, maximum


def narration_safety_limits(config: dict[str, Any]) -> tuple[int, int]:
    """Return the hard source-language duration envelope accepted by playback."""
    minimum, maximum = narration_word_limits(config)
    return max(8, math.floor(minimum * 0.70)), math.ceil(maximum * 1.30)


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


def _story_prefix_at_word_limit(text: str, language: str, limit: int) -> str:
    """Cut an unavoidable long passage without changing any source characters."""
    if language == "ja":
        low, high = 1, len(text)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = text[:middle].rstrip()
            if spoken_word_count(candidate, language) <= limit:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best
    matches = list(re.finditer(r"[^\W_]+(?:['’-][^\W_]+)*", text, flags=re.UNICODE))
    if not matches:
        return text.rstrip()
    return text[:matches[min(limit, len(matches)) - 1].end()].rstrip()


def take_story_chunk(
    text: str,
    language: str,
    minimum: int,
    maximum: int,
    *,
    final: bool,
    accepted_minimum: int | None = None,
    accepted_maximum: int | None = None,
) -> tuple[str, int]:
    """Select one narration-sized prefix and report consumed characters.

    Sentence boundaries win. ``minimum`` and ``maximum`` are the preferred live
    duration target; the accepted bounds are the hard playback envelope. A word
    boundary is used only when no sentence boundary can satisfy that envelope.
    """
    leading = len(text) - len(text.lstrip())
    value = text[leading:]
    if not value:
        return "", len(text)
    hard_minimum = minimum if accepted_minimum is None else int(accepted_minimum)
    hard_maximum = maximum if accepted_maximum is None else int(accepted_maximum)
    if not 1 <= hard_minimum <= minimum <= maximum <= hard_maximum:
        raise ValueError("Story chunk limits must nest inside the accepted duration envelope.")
    sentences = split_narration_sentences(value)
    sentence_counts = [spoken_word_count(sentence, language) for sentence in sentences]
    total_words = sum(sentence_counts)
    if not final and total_words < hard_minimum:
        return "", 0

    # A final short passage is valid: preserving the user's ending outranks padding
    # or rejecting the story. It is identified explicitly by the source cursor.
    if final and total_words <= hard_maximum:
        return value.strip(), leading + len(value.rstrip())

    candidates: list[tuple[int, int]] = []
    prefix_words = 0
    for index, words in enumerate(sentence_counts, start=1):
        prefix_words += words
        if prefix_words > hard_maximum:
            break
        if prefix_words < hard_minimum:
            continue
        remaining_words = total_words - prefix_words
        if final and 0 < remaining_words < hard_minimum:
            continue
        candidates.append((index, prefix_words))

    if candidates:
        def distance(candidate: tuple[int, int]) -> tuple[int, int]:
            words = candidate[1]
            outside = minimum - words if words < minimum else words - maximum if words > maximum else 0
            return outside, abs(maximum - words)

        sentence_count, _ = min(candidates, key=distance)
        chunk = " ".join(sentences[:sentence_count]).strip()
        return chunk, leading + len(chunk)

    cut_words = min(maximum, hard_maximum)
    if final:
        remaining_words = total_words - cut_words
        if 0 < remaining_words < hard_minimum:
            cut_words = total_words - hard_minimum
    cut_words = max(hard_minimum, min(hard_maximum, cut_words))
    chunk = _story_prefix_at_word_limit(value, language, cut_words)
    if not chunk:
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
