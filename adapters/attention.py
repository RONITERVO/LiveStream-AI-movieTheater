from __future__ import annotations

import time
from typing import Any

from adapters.translation import starter_pair
from story_domain import CANONICAL_LANGUAGE, attention_eta, translation_language


class AttentionAdapter:
    """Project durable theater state into the tiny loading-free UI contract."""

    def project(self, state: dict[str, Any]) -> dict[str, Any]:
        result = dict(state)
        result["title"] = state.get("display_title") or state.get("title")
        config = state.get("config", {})
        source_language = str(config.get("language", CANONICAL_LANGUAGE))
        target_language = translation_language(config)
        rendered = {int(item.get("number", 0)) for item in state.get("segments", [])}
        candidates = [
            item for item in state.get("planned", [])
            if int(item.get("number", 0)) not in rendered and item.get("learning_pairs")
        ]
        if not candidates:
            candidates = [item for item in reversed(state.get("segments", [])) if item.get("learning_pairs")]
        pairs = list(candidates[0].get("learning_pairs", [])) if candidates else []
        if not pairs:
            pairs = [starter_pair(source_language, target_language)]
        status = str(state.get("status", ""))
        scene_number = int(
            state.get("rendering_scene") or state.get("assembling_scene") or state.get("current_scene") or 1
        )
        detail = {
            "starting": "Starting local models",
            "planning": "Writing in English",
            "generating": f"Creating visual {scene_number}",
            "narrating": f"Recording scene {scene_number}",
            "aligning": f"Aligning every word in scene {scene_number}",
            "buffering": f"Preparing scene {scene_number}",
            "running": f"Preparing scene {scene_number + 1}",
            "complete": "Story complete",
            "failed": "Stream paused",
            "stopped": "Stream stopped",
            "interrupted": "Stream interrupted",
        }.get(status, "Preparing the story")
        metrics = state.get("metrics", {}) if isinstance(state.get("metrics"), dict) else {}
        result["attention"] = {
            "pairs": pairs[:16],
            "detail": detail,
            "eta_seconds": attention_eta(state, time.time()),
            "average_scene_seconds": round(float(metrics.get("completion_interval_ema") or 0)) or None,
            "source_language": source_language,
            "translation_language": target_language,
        }
        return result
