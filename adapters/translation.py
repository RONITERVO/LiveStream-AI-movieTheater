from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any


STORY_WORDS = {
    "ar": "قصة",
    "bg": "история",
    "hr": "priča",
    "cs": "příběh",
    "da": "historie",
    "nl": "verhaal",
    "en": "story",
    "et": "lugu",
    "fi": "tarina",
    "fr": "histoire",
    "de": "Geschichte",
    "el": "ιστορία",
    "hi": "कहानी",
    "hu": "történet",
    "id": "cerita",
    "it": "storia",
    "ja": "物語",
    "ko": "이야기",
    "lv": "stāsts",
    "lt": "istorija",
    "pl": "historia",
    "pt": "história",
    "ro": "poveste",
    "ru": "история",
    "sk": "príbeh",
    "sl": "zgodba",
    "es": "historia",
    "sv": "berättelse",
    "tr": "hikâye",
    "uk": "історія",
    "vi": "câu chuyện",
}


class TranslationAdapterError(RuntimeError):
    pass


def starter_pair(source_language: str, translation_language: str) -> dict[str, str]:
    """Return an immediate, offline pair while the first scene is localized."""
    return {
        "source": STORY_WORDS.get(str(source_language).lower(), "story"),
        "translation": STORY_WORDS.get(str(translation_language).lower(), "story"),
    }


def _json_object(text: str) -> dict[str, Any]:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise TranslationAdapterError("localization did not return a JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise TranslationAdapterError("localization did not return one JSON object")
    return parsed


def _learning_pairs(
    value: Any, fallback: dict[str, str], source_corpus: str, translation_corpus: str,
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    source_search = re.sub(r"\s+", " ", source_corpus).casefold()
    translation_search = re.sub(r"\s+", " ", translation_corpus).casefold()
    if isinstance(value, list):
        for item in value[:24]:
            if not isinstance(item, dict):
                continue
            source = re.sub(r"\s+", " ", str(item.get("source", ""))).strip()[:80]
            translation = re.sub(r"\s+", " ", str(item.get("translation", ""))).strip()[:80]
            key = (source.casefold(), translation.casefold())
            if (
                not source or not translation or source.casefold() == translation.casefold() or key in seen
                or source.casefold() not in source_search or translation.casefold() not in translation_search
            ):
                continue
            seen.add(key)
            pairs.append({"source": source, "translation": translation})
            if len(pairs) == 16:
                break
    fallback_key = (fallback["source"].casefold(), fallback["translation"].casefold())
    if (
        fallback["source"] and fallback["translation"]
        and fallback["source"].casefold() != fallback["translation"].casefold()
        and fallback_key not in seen
    ):
        pairs.insert(0, fallback)
    return pairs[:16]


class TranslationAdapter:
    """Strict localization boundary around the shared resident Gemma runtime.

    Story planning stays canonical English. This adapter owns multilingual
    prompting, schema repair, exact sentence alignment and learning-card data.
    """

    def __init__(self, runtime: Any, language_names: dict[str, str]) -> None:
        self.runtime = runtime
        self.language_names = language_names

    def _name(self, language: str) -> str:
        return self.language_names.get(language, language)

    async def localize_scene(
        self,
        *,
        title: str,
        sentences: list[str],
        input_language: str,
        source_language: str,
        translation_language: str,
        log_path: Path | None = None,
        scene_number: int | None = None,
    ) -> dict[str, Any]:
        if not sentences or len(sentences) > 48:
            raise TranslationAdapterError("localization requires 1-48 source sentences")
        if source_language == translation_language:
            raise TranslationAdapterError("localization languages must be distinct")

        input_name = self._name(input_language)
        source_name = self._name(source_language)
        translation_name = self._name(translation_language)
        numbered = [{"id": index, "text": sentence} for index, sentence in enumerate(sentences, 1)]
        system = (
            "You are the strict localization adapter for a completely offline bilingual story theater. "
            f"Input is {input_name} [{input_language}]. The source narration must be {source_name} "
            f"[{source_language}] and its learning translation must be {translation_name} "
            f"[{translation_language}]. Return only valid JSON. Preserve meaning, names, dialogue, tone, facts, "
            "sentence ids and sentence count exactly. When an output language equals the input language, copy that "
            "input text verbatim. Never combine or split sentences and never add teaching commentary to narration."
        )
        request = (
            "Localize the title and numbered sentences into both requested output roles. Also select 8-16 useful "
            "single-word or very short phrase pairs that occur in the localized text; prefer concrete, memorable words "
            "and exclude names. Return "
            "{title_source,title_translation,sentences:[{id,source,translation}],"
            "learning_pairs:[{source,translation}]} only.\n\n"
            f"TITLE: {title}\nSENTENCES: {json.dumps(numbered, ensure_ascii=False)}"
        )
        last_error: Exception | None = None
        for attempt in range(1, 4):
            attempt_request = request
            if last_error:
                attempt_request += f"\n\nThe previous output was rejected: {last_error}. Repair only that defect."
            input_units = sum(max(len(item.split()), math.ceil(len(item) / 4)) for item in sentences)
            content, metrics = await self.runtime.complete(
                [{"role": "system", "content": system}, {"role": "user", "content": attempt_request}],
                max_tokens=min(2400, max(650, input_units * 5 + 350)),
            )
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({
                        "time": time.time(), "scene": scene_number, "attempt": attempt, "content": content,
                    }, ensure_ascii=False) + "\n")
            try:
                value = _json_object(content)
                localized = value.get("sentences")
                if not isinstance(localized, list) or len(localized) != len(sentences):
                    raise TranslationAdapterError("localization changed the sentence count")
                aligned: list[dict[str, str]] = []
                for expected_id, (original, item) in enumerate(zip(sentences, localized), 1):
                    if not isinstance(item, dict) or int(item.get("id", -1)) != expected_id:
                        raise TranslationAdapterError("localization changed sentence ids or order")
                    source = original if source_language == input_language else str(item.get("source", "")).strip()
                    translation = (
                        original if translation_language == input_language
                        else str(item.get("translation", "")).strip()
                    )
                    if not source or not translation:
                        raise TranslationAdapterError(f"localized sentence {expected_id} was empty")
                    aligned.append({"original": source, "translation": translation})

                source_title = title if source_language == input_language else str(value.get("title_source", "")).strip()
                translated_title = (
                    title if translation_language == input_language
                    else str(value.get("title_translation", "")).strip()
                )
                if not source_title or not translated_title:
                    raise TranslationAdapterError("localized title was empty")
                fallback = {"source": source_title, "translation": translated_title}
                source_corpus = " ".join([source_title, *(item["original"] for item in aligned)])
                translation_corpus = " ".join([translated_title, *(item["translation"] for item in aligned)])
                return {
                    "title": source_title,
                    "translated_title": translated_title,
                    "sentences": aligned,
                    "learning_pairs": _learning_pairs(
                        value.get("learning_pairs"), fallback, source_corpus, translation_corpus,
                    ),
                    "metrics": dict(metrics),
                    "attempt": attempt,
                }
            except (TranslationAdapterError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise TranslationAdapterError(f"localization failed closed: {last_error}")
