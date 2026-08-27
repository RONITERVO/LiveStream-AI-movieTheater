from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from adapters.local_models import StoryRuntime, SupertonicRuntime
from adapters.story_source import StorySourceAdapter, StorySourceFinished
from adapters.whisper import WhisperAlignmentAdapter
from process_utils import hidden_process_kwargs
from story_domain import (
    CINEMA_DEFAULTS,
    DEFAULT_CONTEXT_COMPACTION_SCENES,
    LANGUAGE_NAMES,
    TheaterError,
    narration_word_limits,
    narration_safety_limits,
    quality_settings,
    spoken_text,
    spoken_word_count,
    split_narration_sentences,
    translation_language,
)


LOGGER = logging.getLogger("wan-video-ui.theater")
THEATER_VERSION = 4


class GpuReleaseError(TheaterError):
    """The CUDA writer could not prove that Wan's VRAM is available again."""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _slug(value: str, fallback: str = "endless-story") -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return clean[:55] or fallback


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise TheaterError("The local story writer did not return a JSON object.")
    return json.loads(text[start : end + 1])


class TheaterManager:
    ACTIVE_STATUSES = {"starting", "planning", "generating", "narrating", "aligning", "buffering", "running"}
    GPU_BURST_TARGET = 3
    GPU_REFILL_POLL_SECONDS = 5.0
    GPU_REFILL_MIN_PREDICTED_WAIT = 12.0
    DEFAULT_CONTEXT_COMPACTION_SCENES = DEFAULT_CONTEXT_COMPACTION_SCENES
    COVERAGE_TARGET = 1.08
    DEFAULT_TTS_SPEED = 1.05
    MIN_TTS_SPEED = 0.96
    MAX_TTS_SPEED = 1.05
    FFMPEG_INTERRUPTED_EXIT_CODES = {-15, 255, 0xC000013A}
    DEFAULT_BILINGUAL_SECONDS_PER_WORD = 0.53
    LIVE_DIRECTIVE_MAX_CHARS = 500
    LIVE_DIRECTIVE_ACTIVE_LIMIT = 12
    LIVE_DIRECTIVE_HISTORY_LIMIT = 100
    LANGUAGE_NAMES = LANGUAGE_NAMES
    CINEMA_DEFAULTS = CINEMA_DEFAULTS

    @classmethod
    def quality_settings(cls, config: dict[str, Any]) -> dict[str, Any]:
        return quality_settings(config)

    @staticmethod
    def translation_language(config: dict[str, Any]) -> str:
        return translation_language(config)

    @classmethod
    def narration_word_limits(cls, config: dict[str, Any]) -> tuple[int, int]:
        return narration_word_limits(config)

    def __init__(
        self, app_dir: Path, output_root: Path, story_model_root: Path, llama_runtime_root: Path,
        supertonic_root: Path, whisper_root: Path, controller: Any,
        video_prompt_builder: Callable[[dict[str, Any]], dict[str, Any]],
        cuda_llama_runtime_root: Path | None = None,
    ) -> None:
        self.app_dir = app_dir
        self.output_root = output_root
        self.root = output_root / "wan_theater"
        self.root.mkdir(parents=True, exist_ok=True)
        self.controller = controller
        self.video_prompt_builder = video_prompt_builder
        self.writer = StoryRuntime(
            app_dir, story_model_root, llama_runtime_root, cuda_llama_runtime_root,
        )
        self.supertonic = SupertonicRuntime(app_dir, supertonic_root)
        self.whisper = WhisperAlignmentAdapter(whisper_root)
        self.story_source = StorySourceAdapter()
        self.sessions: dict[str, dict[str, Any]] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.steering_events: dict[str, asyncio.Event] = {}

    def _dir(self, session_id: str) -> Path:
        return self.root / session_id

    def load_existing(self) -> None:
        for progress in self.root.glob("*/session.json"):
            try:
                state = json.loads(progress.read_text(encoding="utf-8"))
                if int(state.get("version", 0)) != THEATER_VERSION:
                    continue
                if state.get("config", {}).get("mode") not in {"story", "interactive", "my_story"}:
                    continue
                if state.get("status") not in {"stopped", "failed", "complete"}:
                    state["status"] = "interrupted"
                    state["message"] = "The app stopped. Saved scenes remain playable and the story can continue."
                    self._save(state)
                self.sessions[state["id"]] = state
                self.steering_events[state["id"]] = asyncio.Event()
            except Exception:
                LOGGER.exception("Could not load theater session %s", progress)

    def _save(self, state: dict[str, Any]) -> None:
        state["updated"] = time.time()
        _atomic_json(self._dir(state["id"]) / "session.json", state)
        self._write_playlist(state)

    def _write_playlist(self, state: dict[str, Any]) -> None:
        directory = self._dir(state["id"])
        segments = [item for item in state.get("segments", []) if item.get("path")]
        m3u = ["#EXTM3U", "#EXT-X-VERSION:3", "#EXT-X-PLAYLIST-TYPE:EVENT", "#EXT-X-TARGETDURATION:600", "#EXT-X-MEDIA-SEQUENCE:0"]
        for item in segments:
            m3u.extend([f"#EXTINF:{float(item['duration']):.3f},{item['title']}", f"segments/{Path(item['path']).name}"])
        if state.get("status") in {"stopped", "complete"}:
            m3u.append("#EXT-X-ENDLIST")
        (directory / "session.m3u8").write_text("\n".join(m3u) + "\n", encoding="utf-8")
        archive = {
            "format": "Wan Endless Theater", "version": THEATER_VERSION,
            "id": state["id"], "title": state.get("title"), "prompt": state["config"]["prompt"],
            "config": state["config"],
            "bible": state.get("bible"), "story_source": state.get("story_source"),
            "story_summary": state.get("story_summary"),
            "continuity_memory": state.get("continuity_memory", {}),
            "context_compacted_through_scene": state.get("context_compacted_through_scene", 0),
            "context_compaction_events": state.get("metrics", {}).get("context_compaction_events", []),
            "live_directives": state.get("live_directives", []),
            "segments": segments,
            "total_duration": state.get("total_duration", 0), "updated": state.get("updated"),
        }
        _atomic_json(directory / "archive.json", archive)

    def recent(self) -> list[dict[str, Any]]:
        return sorted(self.sessions.values(), key=lambda item: item.get("created", 0), reverse=True)[:20]

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self.sessions.get(session_id)

    def active(self) -> dict[str, Any] | None:
        return next((s for s in self.sessions.values() if s.get("status") in self.ACTIVE_STATUSES), None)

    def _launch(self, state: dict[str, Any]) -> None:
        session_id = state["id"]
        self.steering_events.setdefault(session_id, asyncio.Event())
        task = asyncio.create_task(self._run(state), name=f"theater-{session_id}")
        self.tasks[session_id] = task
        task.add_done_callback(lambda finished, sid=session_id: self._task_finished(sid, finished))

    def _task_finished(self, session_id: str, task: asyncio.Task[None]) -> None:
        """Never leave an exited background worker looking active in the UI."""
        state = self.sessions.get(session_id)
        if not state or task.cancelled() or state.get("status") not in self.ACTIVE_STATUSES:
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        detail = str(error) if error else "The theater worker stopped before the next scene was produced."
        LOGGER.error("Theater session %s background worker exited unexpectedly: %s", session_id, detail)
        state["status"] = "failed"
        state["message"] = detail
        state["error"] = detail
        self._save(state)

    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        if self.active():
            raise TheaterError("An endless theater session is already running.")
        session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        story_text = str(config.pop("_story_text", ""))
        state = {
            "id": session_id, "version": THEATER_VERSION, "created": time.time(),
            "status": "starting", "message": f"Loading {self.writer.model_label} into system RAM...",
            "config": config, "title": None, "bible": None, "planned": [], "segments": [],
            "live_directives": [],
            "current_scene": 0, "total_duration": 0.0, "buffer_seconds": 0.0,
            "metrics": {
                "planner_tps": 0.0, "production_ema": 0.0, "coverage_ratio": 0.0,
                "writer_parallel_slots": self.writer.parallel_slots,
                "parallel_translation": True,
                "gpu_feed_wait_seconds": 0.0,
                "completion_interval_ema": 0.0, "speech_seconds_per_word_ema": 0.0,
                "spoken_word_multiplier_ema": 0.0, "last_narration_speed": self.DEFAULT_TTS_SPEED,
                "coverage_target": self.COVERAGE_TARGET,
                "writer_mode": "starting",
                "gpu_burst_available": self.writer.gpu_available,
                "gpu_burst_target": self.GPU_BURST_TARGET,
                "context_compaction_interval": int(
                    config.get("context_compaction_scenes", self.DEFAULT_CONTEXT_COMPACTION_SCENES)
                ),
            },
        }
        directory = self._dir(session_id)
        for sub in ("raw", "audio", "segments", "work", "logs"):
            (directory / sub).mkdir(parents=True, exist_ok=True)
        if story_text:
            state["story_source"] = self.story_source.persist(directory, story_text)
        self.sessions[session_id] = state
        self.steering_events[session_id] = asyncio.Event()
        self._save(state)
        self._launch(state)
        return state

    @staticmethod
    def _planning_context_snapshot(state: dict[str, Any]) -> dict[str, Any]:
        """Capture mutable causal state so speculative scenes can be replaced safely."""
        metrics = state.get("metrics", {})
        return {
            "story_summary": state.get("story_summary"),
            "has_continuity_memory": "continuity_memory" in state,
            "continuity_memory": state.get("continuity_memory"),
            "has_compaction_boundary": "context_compacted_through_scene" in state,
            "context_compacted_through_scene": state.get("context_compacted_through_scene"),
            "context_compaction_metrics": {
                key: value for key, value in metrics.items() if "context_compaction" in key
            },
            "story_source": dict(state.get("story_source", {})),
        }

    @staticmethod
    def _restore_planning_context(state: dict[str, Any], snapshot: dict[str, Any]) -> None:
        state["story_summary"] = snapshot.get("story_summary")
        if snapshot.get("has_continuity_memory"):
            state["continuity_memory"] = snapshot.get("continuity_memory") or {}
        else:
            state.pop("continuity_memory", None)
        if snapshot.get("has_compaction_boundary"):
            state["context_compacted_through_scene"] = snapshot.get("context_compacted_through_scene")
        else:
            state.pop("context_compacted_through_scene", None)
        metrics = state.setdefault("metrics", {})
        for key in list(metrics):
            if "context_compaction" in key:
                metrics.pop(key, None)
        metrics.update(snapshot.get("context_compaction_metrics") or {})
        if snapshot.get("story_source"):
            state["story_source"] = dict(snapshot["story_source"])

    @staticmethod
    def _steering_context(state: dict[str, Any], number: int) -> tuple[str, list[str]]:
        selected = [
            item for item in state.get("live_directives", [])
            if item.get("status") in {"pending", "active"}
            and number >= int(item.get("activation_scene") or 1)
        ]
        if not selected:
            return "", []
        payload = [
            {"id": item["id"], "scope": item["scope"], "instruction": item["text"]}
            for item in selected
        ]
        prompt = (
            "\nLIVE WORLD EVENTS AND DIRECTIONS:\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "Make each next_scene event observably affect this scene now rather than postponing it. "
            "Treat each audience_message as delayed viewer speech addressed to the recurring host: the host must "
            "acknowledge its specific meaning and answer it naturally in the spoken narration during this scene, "
            "without pretending the exchange is instantaneous or blindly making a viewer suggestion physically true. "
            "Treat persistent rules as active world constraints. Integrate directions causally while preserving the "
            "fixed premise contract, established identity, continuity, and audience level.\n"
        )
        return prompt, [item["id"] for item in selected]

    def _log_live_directive(self, state: dict[str, Any], event: dict[str, Any]) -> None:
        log_path = self._dir(state["id"]) / "logs" / "live_directives.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _trim_live_directives(self, state: dict[str, Any]) -> None:
        directives = list(state.get("live_directives", []))
        if len(directives) <= self.LIVE_DIRECTIVE_HISTORY_LIMIT:
            return
        live = [item for item in directives if item.get("status") in {"pending", "active"}]
        terminal = [item for item in directives if item.get("status") not in {"pending", "active"}]
        keep_terminal = max(0, self.LIVE_DIRECTIVE_HISTORY_LIMIT - len(live))
        state["live_directives"] = terminal[-keep_terminal:] + live if keep_terminal else live

    def add_live_directive(
        self, session_id: str, text: str, scope: str = "next_scene", delivery: str = "after_buffer",
    ) -> dict[str, Any]:
        state = self.sessions.get(session_id)
        if not state:
            raise TheaterError("Theater session not found.")
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            raise TheaterError("Enter a live direction.")
        if len(cleaned) > self.LIVE_DIRECTIVE_MAX_CHARS:
            raise TheaterError(f"Live directions can contain at most {self.LIVE_DIRECTIVE_MAX_CHARS} characters.")
        if scope not in {"audience_message", "next_scene", "persistent"}:
            raise TheaterError("Choose a chat message, one-scene event, or persistent world rule.")
        if scope == "audience_message" and state.get("config", {}).get("mode") != "interactive":
            raise TheaterError("Chat messages to the host require the Interactive character experience.")
        if delivery not in {"after_buffer", "next_unrendered"}:
            raise TheaterError("Choose delayed steering or next-unrendered steering.")
        active_count = sum(
            item.get("status") in {"pending", "active"} for item in state.get("live_directives", [])
        )
        if active_count >= self.LIVE_DIRECTIVE_ACTIVE_LIMIT:
            raise TheaterError(
                f"Remove or let an event finish before adding more than {self.LIVE_DIRECTIVE_ACTIVE_LIMIT} live directions."
            )
        planned_through = max((int(item["number"]) for item in state.get("planned", [])), default=0)
        planning_in_flight = bool(state.get("metrics", {}).get("planner_cycle_started_at"))
        bootstrap_in_flight = not planned_through and bool(state.get("bootstrap_scene"))
        reserved_through = planned_through + (1 if planning_in_flight else 0)
        if bootstrap_in_flight:
            reserved_through = max(reserved_through, 1)
        activation_scene = reserved_through + 1 if delivery == "after_buffer" else 1
        directive = {
            "id": secrets.token_hex(6), "text": cleaned, "scope": scope,
            "status": "active" if scope == "persistent" else "pending",
            "delivery": delivery, "activation_scene": activation_scene, "created_at": time.time(),
        }
        state.setdefault("live_directives", []).append(directive)
        self._trim_live_directives(state)
        self._log_live_directive(state, {"action": "added", **directive})
        task = self.tasks.get(session_id)
        if task and not task.done() and delivery == "next_unrendered":
            self.steering_events.setdefault(session_id, asyncio.Event()).set()
            state["message"] = "Live direction queued; speculative text will be replaced before the next render."
        elif (not task or task.done()) and delivery == "next_unrendered":
            self._rollback_speculative_plans(state, {int(item["number"]) for item in state.get("segments", [])})
            state["message"] = "Live direction saved; it will affect the next scene when this theater resumes."
        else:
            state["message"] = (
                f"Live direction queued for scene {activation_scene}; all existing planned work will finish first."
            )
        self._save(state)
        return state

    def remove_live_directive(self, session_id: str, directive_id: str) -> dict[str, Any]:
        state = self.sessions.get(session_id)
        if not state:
            raise TheaterError("Theater session not found.")
        directive = next(
            (item for item in state.get("live_directives", []) if item.get("id") == directive_id), None,
        )
        if not directive or directive.get("status") not in {"pending", "active"}:
            raise TheaterError("That live direction is no longer active.")
        directive.update(status="removed", removed_at=time.time())
        self._log_live_directive(state, {"action": "removed", **directive})
        task = self.tasks.get(session_id)
        if directive.get("delivery") == "next_unrendered":
            if task and not task.done():
                self.steering_events.setdefault(session_id, asyncio.Event()).set()
            else:
                self._rollback_speculative_plans(state, {int(item["number"]) for item in state.get("segments", [])})
            state["message"] = "The world rule was removed from future unrendered scenes."
        else:
            state["message"] = "The delayed rule was removed; existing planned scenes remain unchanged."
        self._save(state)
        return state

    def _mark_directives_applied(self, state: dict[str, Any], scene: dict[str, Any]) -> None:
        directive_ids = set(scene.get("_live_directive_ids") or [])
        if not directive_ids:
            return
        applied_at = time.time()
        for item in state.get("live_directives", []):
            if item.get("id") not in directive_ids:
                continue
            if item.get("scope") in {"audience_message", "next_scene"} and item.get("status") == "pending":
                item.update(status="applied", applied_scene=int(scene["number"]), applied_at=applied_at)
                self._log_live_directive(state, {"action": "applied", **item})
            elif item.get("scope") == "persistent" and not item.get("first_applied_scene"):
                item.update(first_applied_scene=int(scene["number"]), activated_at=applied_at)
                self._log_live_directive(state, {"action": "activated", **item})

    def _rollback_speculative_plans(self, state: dict[str, Any], protected_numbers: set[int]) -> int:
        """Discard only an unrendered suffix with a known causal-state checkpoint."""
        planned = list(state.get("planned", []))
        discarded = [item for item in planned if int(item["number"]) not in protected_numbers]
        if not discarded:
            return 0
        first = discarded[0]
        snapshot = first.get("_planning_context_before")
        if not isinstance(snapshot, dict):
            state.setdefault("metrics", {})["live_steering_legacy_delay_through_scene"] = int(discarded[-1]["number"])
            return 0
        discarded_numbers = {int(item["number"]) for item in discarded}
        self._restore_planning_context(state, snapshot)
        state["planned"] = [item for item in planned if int(item["number"]) not in discarded_numbers]
        for item in state.get("live_directives", []):
            if item.get("status") == "applied" and int(item.get("applied_scene") or 0) in discarded_numbers:
                item["status"] = "pending"
                item.pop("applied_scene", None)
                item.pop("applied_at", None)
            if int(item.get("first_applied_scene") or 0) in discarded_numbers:
                item.pop("first_applied_scene", None)
                item.pop("activated_at", None)
        metrics = state.setdefault("metrics", {})
        metrics["live_steering_revision"] = int(metrics.get("live_steering_revision") or 0) + 1
        metrics["live_steering_discarded_plans"] = int(metrics.get("live_steering_discarded_plans") or 0) + len(discarded)
        metrics["last_live_steering_scene"] = min(discarded_numbers)
        metrics.pop("live_steering_legacy_delay_through_scene", None)
        metrics.pop("planner_repair_reason", None)
        return len(discarded)

    def resume(self, session_id: str) -> dict[str, Any]:
        state = self.sessions.get(session_id)
        if not state:
            raise TheaterError("Theater session not found.")
        task = self.tasks.get(session_id)
        if task and not task.done():
            return state
        state["status"] = "starting"
        state["message"] = "Continuing from the saved story archive..."
        self._save(state)
        self._launch(state)
        return state

    async def stop(self, session_id: str) -> dict[str, Any]:
        state = self.sessions.get(session_id)
        if not state:
            raise TheaterError("Theater session not found.")
        task = self.tasks.get(session_id)
        if task and not task.done():
            await self.controller.interrupt()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        state["status"] = "stopped"
        state["message"] = "The theater stopped safely. Every completed scene is archived."
        self._save(state)
        return state

    async def shutdown(self) -> None:
        for session_id, task in list(self.tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                state = self.sessions.get(session_id)
                if state:
                    state["status"] = "interrupted"
                    state["message"] = "The app stopped; completed scenes were kept."
                    self._save(state)
        await self.writer.stop_all()
        await self.supertonic.stop()

    def _system_prompt(self, config: dict[str, Any]) -> str:
        age = config.get("audience", "family")
        mode = config.get("mode", "story")
        language = config.get("language", "en")
        language_name = self.LANGUAGE_NAMES.get(language, language)
        interactive_contract = (
            " This is an interactive character show with one stable primary on-screen host. Keep the host's identity, "
            "appearance, voice, relationships, setting and ongoing activity consistent. Narration is primarily the "
            "host's natural first-person speech to the viewer, without narrator labels or stage directions. When no "
            "viewer message is eligible, the host continues the activity, reflects, tells the unfolding story, or "
            "invites a future response without stalling. Viewer messages are delayed turns, not real-time perception; "
            "never claim to see, hear or monitor the viewer."
            if mode == "interactive" else ""
        )
        source_contract = (
            " The narration comes from the user's immutable My story source. Never rewrite, continue, summarize, censor, "
            "or translate that narration in a scene-planning response. Only infer compact visual metadata and continuity "
            "from the exact supplied passage."
            if mode == "my_story" else
            "Maintain strict causal continuity, stable identities, geography, wardrobe, tone and facts. Every scene must "
            "change the situation and use a new action, composition and sensory motif. Treat the user's seed as a binding "
            "premise contract. Preserve every explicit character count, named role, required object, action, event and "
            "setting. Never delay, remove, reverse or contradict an explicit premise event through an invented continuity "
            "rule. Explicit premise requirements outrank stylistic invention. "
        )
        return (
            "You are the resident writer for a completely offline audiovisual story theater. "
            f"MANDATORY OUTPUT LANGUAGE: {language_name} [{language}]. Every natural-language JSON string value, "
            "including titles, names, roles, descriptions, beats, narration, actions and summaries, must be written "
            f"only in {language_name}. Do not translate the user's story into English. Keep JSON keys in English. "
            "Return only valid JSON. "
            f"{source_contract}"
            "Never recap at length, reset the plot, or reuse an earlier event. Keep presentation "
            f"appropriate for audience={age}, and mode={mode}. "
            "Narration must be natural spoken prose. "
            "Use complete sentences separated by spaces and avoid abbreviations that end in a period."
            f"{interactive_contract}"
        )

    def _translation_system_prompt(self, config: dict[str, Any]) -> str:
        source = str(config.get("language", "en")).lower()
        target = self.translation_language(config)
        source_name = self.LANGUAGE_NAMES.get(source, source)
        target_name = self.LANGUAGE_NAMES.get(target, target)
        return (
            "You are the translation stage of a completely offline language-learning story theater. "
            f"Translate from {source_name} [{source}] to {target_name} [{target}]. Return only valid JSON. "
            "Preserve meaning, names, dialogue, tone and verified facts exactly. Use natural, concise spoken language. "
            "Never add explanations, omit details, combine sentences, split sentences, or change the supplied ids."
        )

    @staticmethod
    def _scene_object(value: Any, context: str) -> dict[str, Any]:
        """Accept the two common JSON shapes emitted by small local models."""
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            candidates = [item for item in value if isinstance(item, dict)]
            if len(candidates) == 1:
                return candidates[0]
        raise TheaterError(f"The local writer's {context} did not contain exactly one scene object.")

    @staticmethod
    def _cast_text(bible: dict[str, Any]) -> str:
        """Render both the new structured cast and legacy protagonist bibles."""
        value = bible.get("protagonists", bible.get("protagonist", "the established main cast"))
        if not isinstance(value, list):
            return str(value)
        members: list[str] = []
        for item in value:
            if isinstance(item, dict):
                details = [str(item.get(key, "")).strip() for key in ("name", "role", "appearance")]
                members.append(", ".join(part for part in details if part))
            elif str(item).strip():
                members.append(str(item).strip())
        return "; ".join(members) or "the established main cast"

    async def _prepare_narration(self, state: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
        """Create the durable, sentence-aligned transcript used by UI and TTS."""
        originals = split_narration_sentences(str(scene.get("narration", "")))
        if not originals:
            raise TheaterError(f"Scene {scene.get('number')} contains no speakable narration sentences.")
        if len(originals) > 48:
            raise TheaterError(f"Scene {scene.get('number')} contains too many narration sentences.")
        config = state["config"]
        source_language = str(config.get("language", "en")).lower()
        target_language = self.translation_language(config)
        scene["source_language"] = source_language
        scene["translation_language"] = target_language
        numbered = [{"id": index, "text": sentence} for index, sentence in enumerate(originals, 1)]
        request = (
            "Translate the title and every numbered narration sentence. Keep the exact sentence count and ids. "
            "The result is read aloud immediately after each original sentence, so translations must be concise and "
            "must not contain teaching commentary. Return "
            "{title_translation,sentences:[{id,translation}]} only.\n\n"
            f"TITLE: {scene.get('title', '')}\nSENTENCES: {json.dumps(numbered, ensure_ascii=False)}"
        )
        last_error: Exception | None = None
        for attempt in range(1, 4):
            state["message"] = (
                f"The local writer is aligning scene {scene['number']} sentence translations "
                f"(attempt {attempt}/3)..."
            )
            state.setdefault("metrics", {})["planner_stage"] = "translation"
            state["metrics"]["translation_attempt"] = attempt
            self._save(state)
            state_metrics = state.setdefault("metrics", {})
            state_metrics["translation_request_started_at"] = time.time()
            try:
                content, metrics = await self.writer.complete([
                    {"role": "system", "content": self._translation_system_prompt(config)},
                    {"role": "user", "content": request},
                ], max_tokens=min(2200, max(500, len(str(scene["narration"]).split()) * 5 + 250)))
            finally:
                state_metrics.pop("translation_request_started_at", None)
            with (self._dir(state["id"]) / "logs" / "translation_raw.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({
                    "time": time.time(), "number": scene["number"], "attempt": attempt, "content": content,
                }, ensure_ascii=False) + "\n")
            try:
                value = _json_object(content)
                translated_title = str(value.get("title_translation", "")).strip()
                translations = value.get("sentences")
                if not translated_title or not isinstance(translations, list) or len(translations) != len(originals):
                    raise TheaterError("translation output did not preserve the title and sentence count")
                aligned: list[dict[str, str]] = []
                for expected_id, (original, translated) in enumerate(zip(originals, translations), 1):
                    if not isinstance(translated, dict) or int(translated.get("id", -1)) != expected_id:
                        raise TheaterError("translation output changed sentence ids or order")
                    text = str(translated.get("translation", "")).strip()
                    if not text:
                        raise TheaterError(f"translation sentence {expected_id} was empty")
                    aligned.append({"original": original, "translation": text})
                scene["translated_title"] = translated_title
                scene["narration_sentences"] = aligned
                scene["source_word_count"] = sum(
                    spoken_word_count(pair["original"], source_language) for pair in aligned
                )
                scene["translation_word_count"] = sum(
                    spoken_word_count(pair["translation"], target_language) for pair in aligned
                )
                scene["total_spoken_words"] = scene["source_word_count"] + scene["translation_word_count"]
                scene["translation_metrics"] = dict(metrics)
                state["metrics"]["translation_tps"] = metrics["tokens_per_second"]
                state["metrics"]["translation_elapsed_seconds"] = metrics["elapsed_seconds"]
                if getattr(getattr(self, "writer", None), "profile", "cpu") == "cpu":
                    previous_translation = float(state["metrics"].get("translation_elapsed_ema") or 0)
                    state["metrics"]["translation_elapsed_ema"] = round(
                        metrics["elapsed_seconds"] if not previous_translation
                        else previous_translation * 0.7 + metrics["elapsed_seconds"] * 0.3,
                        3,
                    )
                else:
                    state["metrics"]["gpu_translation_elapsed_seconds"] = metrics["elapsed_seconds"]
                state["metrics"]["translation_prompt_tokens"] = metrics["prompt_tokens"]
                return scene
            except (TheaterError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise TheaterError(f"Sentence translation of scene {scene['number']} failed closed: {last_error}")

    async def _bootstrap(self, state: dict[str, Any]) -> dict[str, Any]:
        config = state["config"]
        language_name = self.LANGUAGE_NAMES.get(config.get("language", "en"), config.get("language", "en"))
        minimum_words, maximum_words = self.narration_word_limits(config)
        safety_minimum, safety_maximum = narration_safety_limits(config)
        opening_maximum = min(maximum_words, minimum_words + 80)
        opening_sentences = max(3, min(10, math.ceil(minimum_words / 7)))
        opening_sentence_minimum = max(4, math.ceil(minimum_words / opening_sentences))
        opening_sentence_maximum = max(
            opening_sentence_minimum, math.floor(opening_maximum / opening_sentences),
        )
        interactive_opening = (
            "This is an interactive character show. Establish one stable primary on-screen host from the seed, a "
            "recognizable place and an ongoing activity that can continue indefinitely. Scene 1 is the host's concise "
            "spoken opening: introduce who they are and what they are doing, welcome delayed viewer messages, and make "
            "clear that viewers can chat or influence later moments. Keep narration in the host's first-person voice.\n"
            if config.get("mode") == "interactive" else ""
        )
        source_chunk = ""
        source_cursor = 0
        if config.get("mode") == "my_story":
            source = state.get("story_source")
            if not isinstance(source, dict):
                raise TheaterError("My story source is missing from this session.")
            source_chunk, source_cursor = self.story_source.next_chunk(
                self._dir(state["id"]),
                source,
                language=str(config.get("language", "en")),
                minimum=minimum_words,
                maximum=opening_maximum,
                accepted_minimum=safety_minimum,
                accepted_maximum=safety_maximum,
            )
            request = (
                "Analyze the exact opening passage from My story. Infer a compact visual bible and one filmable scene "
                "without rewriting or returning the narration. Preserve names, events, chronology, tone and facts. "
                f"Write metadata only in {language_name}. Return "
                "{title,bible:{protagonists:[{name,role,appearance}],world,visual_style,premise_contract:[...],"
                "continuity_rules:[...]},story_summary,scene:{number,title,beat,visual_action,camera}} only.\n\n"
                f"EXACT IMMUTABLE NARRATION:\n{source_chunk}"
            )
        else:
            request = (
                f"Create an endless story from this seed: {config['prompt']}\n"
                f"Write every natural-language value only in {language_name}; English is forbidden except for JSON keys.\n"
                "First extract the seed's non-negotiable requirements into premise_contract. Scene 1 must visibly "
                "establish every explicitly requested main character and the immediate central situation or threat. "
                "Give every recurring character a stable name, role and visual appearance.\n"
                f"{interactive_opening}"
                f"Write {minimum_words}-{opening_maximum} narration words for scene 1. This is a hard playback-duration "
                f"budget; use exactly {opening_sentences} complete sentences with {opening_sentence_minimum}-"
                f"{opening_sentence_maximum} words each and no recap or filler.\n"
                "Return {title,bible:{protagonists:[{name,role,appearance}],world,visual_style,premise_contract:[...],"
                "continuity_rules:[...]},story_summary,scene:{number,title,beat,narration,visual_action,camera}}. "
                "visual_action must contain one filmable action and no visible text."
            )
        content, metrics = await self.writer.complete([
            {"role": "system", "content": self._system_prompt(config)},
            {"role": "user", "content": request},
        ], max_tokens=1300)
        value = _json_object(content)
        for key in ("title", "bible", "story_summary", "scene"):
            if key not in value:
                raise TheaterError(f"The local writer's story opening is missing {key}.")
        if not isinstance(value["bible"], dict):
            raise TheaterError("The local writer did not return a story bible object.")
        bible = value["bible"]
        required_bible = ("protagonists", "world", "visual_style", "premise_contract", "continuity_rules")
        if any(key not in bible for key in required_bible):
            raise TheaterError("The local writer's story bible did not preserve the full premise contract.")
        if config.get("mode") == "interactive":
            bible["experience"] = "interactive_character"
            bible["interaction_contract"] = [
                "The primary host directly addresses delayed viewer messages in spoken narration.",
                "The host never claims real-time sight, hearing, monitoring, or immediate response.",
                "Without a viewer message, the host continues the established activity and open-ended show.",
            ]
        value["scene"] = self._scene_object(value["scene"], "bootstrap")
        if source_chunk:
            value["scene"]["narration"] = source_chunk
            value["scene"]["source_cursor_end"] = source_cursor
            state["story_source"]["cursor"] = source_cursor
            state["story_source"]["progress"] = self.story_source.progress(state["story_source"])
        value["scene"]["planner_metrics"] = dict(metrics)
        state["metrics"]["planner_tps"] = metrics["tokens_per_second"]
        return value

    def _target_total_words(self, state: dict[str, Any]) -> int:
        """Translate measured completed-scene cadence into a bounded speech budget."""
        quality = self.quality_settings(state["config"])
        minimum, maximum = int(quality["min_words"]), int(quality["max_words"])
        metrics = state.setdefault("metrics", {})
        cadence = float(metrics.get("completion_interval_ema") or metrics.get("production_ema") or 0)
        if not cadence:
            return minimum
        fallback = self.DEFAULT_BILINGUAL_SECONDS_PER_WORD
        seconds_per_word = max(0.08, float(metrics.get("speech_seconds_per_word_ema") or fallback))
        target = round(cadence * self.COVERAGE_TARGET / seconds_per_word)
        return int(max(minimum, min(maximum, target)))

    def _target_words(self, state: dict[str, Any]) -> int:
        minimum, maximum = self.narration_word_limits(state["config"])
        metrics = state.setdefault("metrics", {})
        default_multiplier = 2.1
        multiplier = max(1.0, min(3.5, float(metrics.get("spoken_word_multiplier_ema") or default_multiplier)))
        target = round(self._target_total_words(state) / multiplier)
        return int(max(minimum, min(maximum, target)))

    def _narration_request_limits(self, state: dict[str, Any]) -> tuple[int, int]:
        """Stay near the live target without leaving the configured word budget."""
        minimum_words, maximum_words = self.narration_word_limits(state["config"])
        words = self._target_words(state)
        margin = min(8, max(3, math.ceil((maximum_words - minimum_words) * 0.20)))
        return max(minimum_words, words - margin), min(maximum_words, words + margin)

    def _narration_speed(self, state: dict[str, Any], scene: dict[str, Any]) -> float:
        """Use bounded voice pacing only to correct the residual duration error."""
        metrics = state.setdefault("metrics", {})
        cadence = float(metrics.get("completion_interval_ema") or metrics.get("production_ema") or 0)
        seconds_per_word = float(metrics.get("speech_seconds_per_word_ema") or 0)
        words = int(scene.get("total_spoken_words") or 0)
        previous_speed = float(metrics.get("last_narration_speed") or self.DEFAULT_TTS_SPEED)
        if not cadence or not seconds_per_word or not words:
            return self.DEFAULT_TTS_SPEED
        target_duration = cadence * self.COVERAGE_TARGET
        predicted_duration = words * seconds_per_word
        requested = previous_speed * predicted_duration / max(0.1, target_duration)
        return round(max(self.MIN_TTS_SPEED, min(self.MAX_TTS_SPEED, requested)), 3)

    def _context_compaction_due(self, state: dict[str, Any]) -> bool:
        if not state.get("bible"):
            return False
        interval = int(
            state.get("config", {}).get("context_compaction_scenes", self.DEFAULT_CONTEXT_COMPACTION_SCENES)
        )
        if interval <= 0:
            return False
        through_scene = max((int(item["number"]) for item in state.get("planned", [])), default=0)
        last_attempt = int(state.get("metrics", {}).get("last_context_compaction_attempt_scene") or 0)
        return through_scene > 0 and through_scene - last_attempt >= interval

    @staticmethod
    def _validated_continuity_memory(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TheaterError("context compaction did not return structured continuity memory")
        characters = value.get("character_states", [])
        threads = value.get("active_threads", [])
        facts = value.get("continuity_facts", [])
        if not isinstance(characters, list) or not isinstance(threads, list) or not isinstance(facts, list):
            raise TheaterError("context compaction returned malformed continuity lists")
        clean_characters: list[dict[str, Any]] = []
        for item in characters[:16]:
            if not isinstance(item, dict) or not str(item.get("name", "")).strip():
                raise TheaterError("context compaction returned a character state without a name")
            possessions = item.get("possessions", [])
            if not isinstance(possessions, list):
                raise TheaterError("context compaction returned malformed character possessions")
            clean_characters.append({
                "name": str(item["name"]).strip()[:120],
                "state": str(item.get("state", "")).strip()[:500],
                "location": str(item.get("location", "")).strip()[:240],
                "possessions": [str(entry).strip()[:160] for entry in possessions[:12] if str(entry).strip()],
            })

        def clean_strings(items: list[Any], limit: int) -> list[str]:
            return [str(item).strip()[:400] for item in items[:limit] if str(item).strip()]

        return {
            "character_states": clean_characters,
            "active_threads": clean_strings(threads, 16),
            "continuity_facts": clean_strings(facts, 28),
        }

    async def _compact_story_context(
        self, state: dict[str, Any], *, reason: str,
    ) -> bool:
        """Replace cumulative prose with bounded causal and entity continuity state."""
        through_scene = max((int(item["number"]) for item in state.get("planned", [])), default=0)
        if not through_scene or not state.get("bible"):
            return False
        metrics = state.setdefault("metrics", {})
        metrics["last_context_compaction_attempt_scene"] = through_scene
        summary = str(state.get("story_summary") or "").strip()
        if len(summary) > 24000:
            summary = summary[:8000] + "\n[older middle compressed by input bound]\n" + summary[-16000:]
        recent = [
            {
                "number": item.get("number"), "title": item.get("title"), "beat": item.get("beat"),
                "narration": item.get("narration"), "visual_action": item.get("visual_action"),
            }
            for item in state.get("planned", [])[-12:]
        ]
        planner_descriptors = [
            {
                "number": item.get("number"), "title": item.get("title"), "beat": item.get("beat"),
                "visual_action": item.get("visual_action"),
            }
            for item in state.get("planned", [])
        ]
        request = (
            "Compact the evolving story context for future scene planning. This is continuity bookkeeping, not a new "
            "story scene. Preserve every unresolved goal, promise, threat and causal dependency; each recurring "
            "character's current condition, location, relationships and possessions; irreversible world changes; and "
            "all facts needed to obey the premise contract. Never invent an event, resolve an open thread, retell scene "
            "prose, or weaken the fixed bible. Write story_summary as at most 250 words describing the current situation. "
            "Return {story_summary,continuity_memory:{character_states:[{name,state,location,possessions:[...]}],"
            "active_threads:[...],continuity_facts:[...]}} only. Keep every list item concise.\n\n"
            f"FIXED BIBLE:\n{json.dumps(state['bible'], ensure_ascii=False)}\n\n"
            f"PREVIOUS CONTINUITY MEMORY:\n{json.dumps(state.get('continuity_memory', {}), ensure_ascii=False)}\n\n"
            f"CURRENT ROLLING SUMMARY:\n{summary}\n\n"
            f"RECENT CAUSAL SCENES:\n{json.dumps(recent, ensure_ascii=False)}"
        )
        before_chars = (
            len(summary)
            + len(json.dumps(state.get("continuity_memory", {}), ensure_ascii=False))
            + len(json.dumps(planner_descriptors[-10:], ensure_ascii=False))
        )
        writer_profile = getattr(self.writer, "profile", "cpu")
        compaction_started = time.perf_counter()
        metrics["context_compaction_started_at"] = time.time()
        last_error: Exception | None = None
        for attempt in range(1, 4):
            state["message"] = (
                f"Compacting continuity through scene {through_scene} on {writer_profile.upper()} "
                f"(attempt {attempt}/3)..."
            )
            metrics.update({"planner_stage": "context_compaction", "context_compaction_reason": reason})
            self._save(state)
            try:
                attempt_request = request
                if last_error:
                    attempt_request += (
                        f"\n\nThe previous compaction was rejected: {last_error}. "
                        "Make the summary and continuity lists materially shorter without dropping required state."
                    )
                content, completion_metrics = await self.writer.complete([
                    {"role": "system", "content": self._system_prompt(state["config"])},
                    {"role": "user", "content": attempt_request},
                ], max_tokens=1100)
                with (self._dir(state["id"]) / "logs" / "context_compaction_raw.jsonl").open(
                    "a", encoding="utf-8",
                ) as log:
                    log.write(json.dumps({
                        "time": time.time(), "through_scene": through_scene, "reason": reason,
                        "attempt": attempt, "content": content,
                    }, ensure_ascii=False) + "\n")
                value = _json_object(content)
                compact_summary = str(value.get("story_summary", "")).strip()
                if not compact_summary or spoken_word_count(
                    compact_summary, state["config"].get("language", "en"),
                ) > 300:
                    raise TheaterError("context compaction returned an empty or oversized story summary")
                continuity_memory = self._validated_continuity_memory(value.get("continuity_memory"))
                after_chars = (
                    len(compact_summary)
                    + len(json.dumps(continuity_memory, ensure_ascii=False))
                    + len(json.dumps(planner_descriptors[-3:], ensure_ascii=False))
                )
                if after_chars > max(1200, math.ceil(before_chars * 1.10)):
                    raise TheaterError(
                        f"context compaction grew the planning payload from {before_chars} to {after_chars} characters"
                    )
                state["story_summary"] = compact_summary
                state["continuity_memory"] = continuity_memory
                state["context_compacted_through_scene"] = through_scene
                metrics.update({
                    "context_compaction_count": int(metrics.get("context_compaction_count") or 0) + 1,
                    "last_context_compaction_scene": through_scene,
                    "context_compaction_elapsed_seconds": completion_metrics["elapsed_seconds"],
                    "context_compaction_prompt_tokens": completion_metrics["prompt_tokens"],
                    "context_compaction_before_chars": before_chars,
                    "context_compaction_after_chars": after_chars,
                    "context_compaction_profile": writer_profile,
                })
                total_compaction_seconds = time.perf_counter() - compaction_started
                if writer_profile == "cpu":
                    previous_compaction = float(metrics.get("context_compaction_elapsed_ema") or 0)
                    metrics["context_compaction_elapsed_ema"] = round(
                        total_compaction_seconds if not previous_compaction
                        else previous_compaction * 0.7 + total_compaction_seconds * 0.3,
                        3,
                    )
                else:
                    metrics["gpu_context_compaction_seconds"] = round(total_compaction_seconds, 3)
                metrics.pop("context_compaction_error", None)
                metrics.pop("context_compaction_started_at", None)
                events = list(metrics.get("context_compaction_events") or [])[-19:]
                events.append({
                    "through_scene": through_scene, "reason": reason, "profile": writer_profile,
                    "completed_at": time.time(), "before_chars": before_chars, "after_chars": after_chars,
                    "elapsed_seconds": completion_metrics["elapsed_seconds"],
                })
                metrics["context_compaction_events"] = events
                self._save(state)
                return True
            except asyncio.CancelledError:
                metrics.pop("context_compaction_started_at", None)
                raise
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Context compaction through scene %s attempt %s failed: %s",
                    through_scene, attempt, exc,
                )
        metrics.update({
            "context_compaction_failures": int(metrics.get("context_compaction_failures") or 0) + 1,
            "context_compaction_error": str(last_error)[:500],
        })
        metrics.pop("context_compaction_started_at", None)
        self._save(state)
        return False

    @staticmethod
    def _recent_scene_context(
        state: dict[str, Any], recent: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Use three pre-compaction anchors, then grow back to the normal ten-scene window."""
        compacted_through = int(state.get("context_compacted_through_scene") or 0)
        if not compacted_through:
            return recent[-10:]
        anchors = [item for item in recent if int(item["number"]) <= compacted_through][-3:]
        after_compaction = [item for item in recent if int(item["number"]) > compacted_through]
        return (anchors + after_compaction)[-10:]

    async def _plan_next(self, state: dict[str, Any], number: int, recent: list[dict[str, Any]]) -> dict[str, Any]:
        planning_context_before = self._planning_context_snapshot(state)
        steering_prompt, live_directive_ids = self._steering_context(state, number)
        my_story = state.get("config", {}).get("mode") == "my_story"
        words = self._target_words(state)
        request_minimum, request_maximum = self._narration_request_limits(state)
        safety_minimum, safety_maximum = narration_safety_limits(state["config"])
        sentence_count = max(3, min(10, math.ceil(words / 7)))
        sentence_minimum = max(4, math.ceil(request_minimum / sentence_count))
        sentence_maximum = max(sentence_minimum, math.floor(request_maximum / sentence_count))
        language = state["config"].get("language", "en")
        language_name = self.LANGUAGE_NAMES.get(language, language)
        recent_context = self._recent_scene_context(state, recent)
        prior = [
            {"number": s["number"], "title": s["title"], "beat": s["beat"], "visual_action": s["visual_action"]}
            for s in recent_context
        ]
        used_hashes = [s.get("asset_fingerprint") for s in state.get("planned", [])[-30:]]
        source_chunk = ""
        source_cursor = 0
        shared_context = (
            f"Story bible: {json.dumps(state['bible'], ensure_ascii=False)}\n"
            f"Current story summary: {state.get('story_summary')}\n"
            f"Structured continuity memory: {json.dumps(state.get('continuity_memory', {}), ensure_ascii=False)}\n"
            f"Recent scenes: {json.dumps(prior, ensure_ascii=False)}\n"
        )
        if my_story:
            source = state.get("story_source")
            if not isinstance(source, dict):
                raise TheaterError("My story source is missing from this session.")
            source_chunk, source_cursor = self.story_source.next_chunk(
                self._dir(state["id"]),
                source,
                language=str(language),
                minimum=request_minimum,
                maximum=request_maximum,
                accepted_minimum=safety_minimum,
                accepted_maximum=safety_maximum,
            )
            request = (
                f"{shared_context}"
                f"Analyze exact My story passage {number}. Never rewrite or return its narration. Write metadata only "
                f"in {language_name}, preserve the source chronology, and update story_summary to a compact current-state "
                "summary of at most 250 words. Live direction may affect staging and imagery but never the supplied words. "
                f"Avoid these prior asset fingerprints: {used_hashes}. Return "
                "{story_summary,scene:{number,title,beat,visual_action,camera}} only.\n\n"
                f"{steering_prompt}EXACT IMMUTABLE NARRATION:\n{source_chunk}"
            )
        else:
            request = (
                f"{shared_context}"
                f"Write every natural-language value only in {language_name}; do not switch to English. "
                f"Create scene {number} with {request_minimum}-{request_maximum} source-language narration words. "
                f"This is a hard playback-duration budget: use about {sentence_count} complete sentences "
                f"({sentence_minimum}-{sentence_maximum} words each), make every sentence advance the action, and do not "
                "use recap or filler. Obey every premise_contract item and continuity rule, follow causally, introduce a "
                "new meaningful development, and remain open-ended. Replace story_summary with a compact current-state "
                "summary of at most 250 words. Keep the JSON compact and do not add fields. "
                f"Avoid these prior asset fingerprints: {used_hashes}. Return "
                "{story_summary,scene:{number,title,beat,narration,visual_action,camera}} only.\n\n"
                f"{steering_prompt}"
            )
        repair_reason = str(state.get("metrics", {}).get("planner_repair_reason") or "").strip()
        if repair_reason:
            request += f"\nThe previous output was rejected: {repair_reason}. Correct that exact validation failure."
        state_metrics = state.setdefault("metrics", {})
        state_metrics["planner_request_started_at"] = time.time()
        try:
            content, metrics = await self.writer.complete([
                {"role": "system", "content": self._system_prompt(state["config"])},
                {"role": "user", "content": request},
            ], max_tokens=min(1800, words * 3 + 450))
        finally:
            state_metrics.pop("planner_request_started_at", None)
        with (self._dir(state["id"]) / "logs" / "planner_raw.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"time": time.time(), "number": number, "content": content}, ensure_ascii=False) + "\n")
        value = _json_object(content)
        scene = self._scene_object(value.get("scene", value), f"scene {number} plan")
        scene["number"] = number
        if source_chunk:
            scene["narration"] = source_chunk
            scene["source_cursor_end"] = source_cursor
        required = ("title", "beat", "narration", "visual_action", "camera")
        if any(not str(scene.get(key, "")).strip() for key in required):
            raise TheaterError(f"The local writer's scene {number} is missing required story fields.")
        narration_words = spoken_word_count(scene["narration"], language)
        narration_sentences = split_narration_sentences(scene["narration"])
        if not narration_sentences:
            raise TheaterError(f"The local writer's scene {number} contains no speakable narration sentences.")
        if len(narration_sentences) > 48:
            raise TheaterError(f"The local writer returned too many narration sentences ({len(narration_sentences)}) for scene {number}.")
        configured_minimum, configured_maximum = self.narration_word_limits(state["config"])
        safety_minimum, safety_maximum = narration_safety_limits(state["config"])
        final_source_chunk = bool(
            source_chunk and source_cursor >= int(state.get("story_source", {}).get("bytes") or 0)
        )
        if (narration_words < safety_minimum and not final_source_chunk) or narration_words > safety_maximum:
            source_name = "My story source" if source_chunk else "The local writer"
            raise TheaterError(
                f"{source_name} returned {narration_words} narration words; "
                f"the safe duration envelope requires {safety_minimum}-{safety_maximum}"
            )
        fingerprint_text = f"{scene['beat']}|{scene['visual_action']}|{scene['camera']}".lower()
        scene["asset_fingerprint"] = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()[:16]
        if scene["asset_fingerprint"] in {s.get("asset_fingerprint") for s in state.get("planned", [])}:
            scene["visual_action"] += f" The action resolves with a unique physical change specific to scene {number}."
            scene["asset_fingerprint"] = hashlib.sha256((fingerprint_text + str(number)).encode()).hexdigest()[:16]
        scene["planner_metrics"] = {
            **metrics,
            "target_total_spoken_words": self._target_total_words(state),
            "requested_source_words_min": request_minimum,
            "requested_source_words_max": request_maximum,
            "accepted_source_words": narration_words,
            "configured_source_budget_met": configured_minimum <= narration_words <= configured_maximum,
        }
        scene["_planning_context_before"] = planning_context_before
        scene["_live_directive_ids"] = live_directive_ids
        state["story_summary"] = str(value.get("story_summary") or state.get("story_summary"))
        if source_chunk:
            state["story_source"]["cursor"] = source_cursor
            state["story_source"]["progress"] = self.story_source.progress(state["story_source"])
        state["metrics"]["planner_tps"] = metrics["tokens_per_second"]
        state["metrics"]["planner_elapsed_seconds"] = metrics["elapsed_seconds"]
        if getattr(getattr(self, "writer", None), "profile", "cpu") == "cpu":
            previous_planner = float(state["metrics"].get("planner_elapsed_ema") or 0)
            state["metrics"]["planner_elapsed_ema"] = round(
                metrics["elapsed_seconds"] if not previous_planner
                else previous_planner * 0.7 + metrics["elapsed_seconds"] * 0.3,
                3,
            )
        else:
            state["metrics"]["gpu_planner_elapsed_seconds"] = metrics["elapsed_seconds"]
        state["metrics"]["planner_prompt_tokens"] = metrics["prompt_tokens"]
        return scene

    async def _planner_loop(self, state: dict[str, Any], queue: asyncio.Queue[dict[str, Any]]) -> None:
        while True:
            # Keep two source-language plans waiting for the translation worker.
            # A separate bounded ready queue limits total look-ahead downstream.
            while queue.qsize() >= 2:
                await asyncio.sleep(0.5)
            number = len(state.get("planned", [])) + 1
            if number == 1 and state.get("bootstrap_scene"):
                # Keep the saved bootstrap scene until it has passed review. Previously an
                # exception here killed this child task while _run waited on an empty queue.
                try:
                    scene = dict(self._scene_object(state["bootstrap_scene"], "saved bootstrap"))
                    scene["number"] = 1
                    text = f"{scene.get('beat')}|{scene.get('visual_action')}|{scene.get('camera')}".lower()
                    scene["asset_fingerprint"] = hashlib.sha256(text.encode()).hexdigest()[:16]
                    state.pop("bootstrap_scene", None)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    LOGGER.exception("Local writer first-scene planning failed")
                    await queue.put({"_error": f"The local writer could not verify scene 1: {exc}"})
                    return
            else:
                if self._context_compaction_due(state):
                    await self._compact_story_context(state, reason="interval")
                last_error: Exception | None = None
                scene = None
                for attempt in range(1, 4):
                    cycle_started = time.perf_counter()
                    state.setdefault("metrics", {})["planner_cycle_started_at"] = time.time()
                    try:
                        state.setdefault("metrics", {})["planner_attempt"] = attempt
                        scene = await self._plan_next(state, number, state.get("planned", []))
                        cycle_seconds = time.perf_counter() - cycle_started
                        state["metrics"]["planner_cycle_seconds"] = round(cycle_seconds, 3)
                        if getattr(getattr(self, "writer", None), "profile", "cpu") == "cpu":
                            previous_cycle = float(state["metrics"].get("planner_cycle_ema") or 0)
                            state["metrics"]["planner_cycle_ema"] = round(
                                cycle_seconds if not previous_cycle
                                else previous_cycle * 0.7 + cycle_seconds * 0.3,
                                3,
                            )
                        else:
                            state["metrics"]["gpu_planner_cycle_seconds"] = round(cycle_seconds, 3)
                        state["metrics"].pop("planner_repair_reason", None)
                        break
                    except StorySourceFinished:
                        state["story_source"]["complete"] = True
                        state["story_source"]["progress"] = 1.0
                        self._save(state)
                        await queue.put({"_complete": True})
                        return
                    except Exception as exc:
                        last_error = exc
                        state.setdefault("metrics", {})["planner_repair_reason"] = str(exc)[:300]
                        LOGGER.exception("Local writer scene %s planning attempt %s failed", number, attempt)
                        state["message"] = f"The local writer is repairing scene {number}'s structured plan (attempt {attempt}/3)..."
                        self._save(state)
                        await asyncio.sleep(0.5)
                    finally:
                        state["metrics"].pop("planner_cycle_started_at", None)
                if scene is None:
                    await queue.put({"_error": f"The local writer could not structure scene {number}: {last_error}"})
                    return
            state.setdefault("planned", []).append(scene)
            self._mark_directives_applied(state, scene)
            state.setdefault("metrics", {})["source_plan_queue"] = queue.qsize() + 1
            state["message"] = "The local model planned the next scene; translation can run beside the following plan."
            self._save(state)
            await queue.put(scene)

    async def _translation_loop(
        self, state: dict[str, Any], source_queue: asyncio.Queue[dict[str, Any]],
        ready_queue: asyncio.Queue[dict[str, Any]], planner_task: asyncio.Task[None],
    ) -> None:
        """Prepare narration while the other Gemma slot plans the next scene."""
        while True:
            scene = await self._next_planned_scene(source_queue, planner_task)
            cycle_started = time.perf_counter()
            state.setdefault("metrics", {})["translation_cycle_started_at"] = time.time()
            try:
                if scene.get("_error"):
                    await ready_queue.put(scene)
                    return
                if scene.get("_complete"):
                    await ready_queue.put(scene)
                    return
                prepared = await self._prepare_narration(state, scene)
                cycle_seconds = time.perf_counter() - cycle_started
                cycle_metrics = {
                    "parallel_translation": True,
                    "source_plan_queue": source_queue.qsize(),
                    "translated_scene_queue": ready_queue.qsize() + 1,
                    "translation_cycle_seconds": round(cycle_seconds, 3),
                }
                if getattr(getattr(self, "writer", None), "profile", "cpu") == "cpu":
                    previous_cycle = float(state["metrics"].get("translation_cycle_ema") or 0)
                    cycle_metrics["translation_cycle_ema"] = round(
                        cycle_seconds if not previous_cycle
                        else previous_cycle * 0.7 + cycle_seconds * 0.3,
                        3,
                    )
                else:
                    cycle_metrics["gpu_translation_cycle_seconds"] = round(cycle_seconds, 3)
                state["metrics"].update(cycle_metrics)
                state["message"] = (
                    f"Scene {prepared['number']} is translation-ready while Gemma continues planning ahead."
                )
                self._save(state)
                await ready_queue.put(prepared)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Local writer scene %s translation failed", scene.get("number"))
                await ready_queue.put({"_error": str(exc)})
                return
            finally:
                state["metrics"].pop("translation_cycle_started_at", None)
                source_queue.task_done()

    @staticmethod
    async def _next_planned_scene(
        queue: asyncio.Queue[dict[str, Any]], planner_task: asyncio.Task[None],
    ) -> dict[str, Any]:
        """Wait for a scene while also supervising the producer task."""
        get_task = asyncio.create_task(queue.get())
        try:
            done, _ = await asyncio.wait({get_task, planner_task}, return_when=asyncio.FIRST_COMPLETED)
            if get_task in done:
                return get_task.result()

            # Let a queue.put performed immediately before a clean return wake its waiter.
            await asyncio.sleep(0)
            if get_task.done():
                return get_task.result()
            if planner_task.cancelled():
                raise asyncio.CancelledError
            error = planner_task.exception()
            if error:
                raise TheaterError(f"The local story planner crashed: {error}") from error
            raise TheaterError("The local story planner stopped before it supplied another scene.")
        finally:
            if not get_task.done():
                get_task.cancel()
            await asyncio.gather(get_task, return_exceptions=True)

    @classmethod
    def _narration_is_prepared(cls, config: dict[str, Any], scene: dict[str, Any]) -> bool:
        """Identify render-ready saved plans without repeating translation after resume."""
        pairs = scene.get("narration_sentences")
        if not isinstance(pairs, list) or not pairs:
            return False
        if any(not str(pair.get("original", "")).strip() for pair in pairs if isinstance(pair, dict)):
            return False
        if any(not isinstance(pair, dict) for pair in pairs):
            return False
        return bool(str(scene.get("translated_title", "")).strip()) and all(
            str(pair.get("translation", "")).strip() for pair in pairs
        )

    def _visual_prompt(self, state: dict[str, Any], scene: dict[str, Any]) -> str:
        bible = state["bible"]
        rules = "; ".join(bible.get("continuity_rules", []))
        premise = "; ".join(bible.get("premise_contract", []))
        cast = self._cast_text(bible)
        host_framing = (
            "Keep the primary host clearly recognizable and present, with a natural near-camera eyeline while they "
            "continue the scene's physical activity. "
            if state.get("config", {}).get("mode") == "interactive" else ""
        )
        return (
            f"{bible['visual_style']}. {scene['camera']}. In {bible['world']}. "
            f"The recurring cast is: {cast}. {host_framing}{scene['visual_action']}. Binding premise: {premise}. "
            f"Strict continuity: {rules}. Same faces, ages, bodies, wardrobe, props, architecture and palette. "
            "One clear action, natural restrained motion, coherent anatomy, no duplicate characters, "
            "no visible words, no subtitles, no logo, no watermark."
        )

    async def _wait_prompt(self, prompt_id: str, state: dict[str, Any]) -> list[dict[str, str]]:
        while True:
            result = await self.controller.job(prompt_id)
            if result["state"] == "complete":
                return result.get("files", [])
            if result["state"] == "failed":
                raise TheaterError(str(result.get("error") or "ComfyUI scene failed."))
            await asyncio.sleep(1)

    async def _duration(self, path: Path) -> float:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise TheaterError("FFprobe is required for theater synchronization.")
        process = await asyncio.create_subprocess_exec(
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **hidden_process_kwargs(),
        )
        out, _ = await process.communicate()
        if process.returncode:
            raise TheaterError(f"Could not inspect {path.name}.")
        return float(out.decode().strip())

    async def _run_ffmpeg(self, args: list[str], log_path: Path) -> None:
        for attempt in range(1, 3):
            with log_path.open("ab") as log:
                process = await asyncio.create_subprocess_exec(
                    *args, stdout=log, stderr=log,
                    **hidden_process_kwargs(),
                )
                try:
                    code = await process.wait()
                except asyncio.CancelledError:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                    raise
            if code == 0:
                return
            if code not in self.FFMPEG_INTERRUPTED_EXIT_CODES or attempt == 2:
                raise TheaterError(f"FFmpeg failed while synchronizing the theater (exit {code}).")
            LOGGER.warning("FFmpeg was externally interrupted (exit %s); retrying once.", code)
            await asyncio.sleep(0.25)

    @classmethod
    def _visual_sync_filter(
        cls, raw_duration: float, audio_duration: float, quality: dict[str, Any],
    ) -> tuple[str, float, bool]:
        """Build one interpolation/coverage graph so each scene is encoded once."""
        fps = int(quality["fps"])
        slow_duration = min(audio_duration, raw_duration * float(quality["max_slow"]))
        ratio = slow_duration / max(
            0.05, raw_duration * (int(quality["frames"]) - 1) / int(quality["frames"]),
        )
        slow_text = f"{slow_duration:.6f}"
        base = (
            f"setpts={ratio:.8f}*PTS,tpad=stop_mode=clone:stop_duration=2,"
            f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir,"
            f"tpad=stop_mode=clone:stop_duration={slow_text},trim=duration={slow_text},"
            "setpts=PTS-STARTPTS"
        )
        repeated = audio_duration > slow_duration + 0.15
        if not repeated:
            return f"[0:v]{base},fps={fps},format=yuv420p[v]", slow_duration, False
        cycle_frames = max(2, math.ceil(slow_duration * fps) * 2)
        graph = (
            f"[0:v]{base},split=2[f][r];"
            "[r]reverse,setpts=PTS-STARTPTS[rr];"
            "[f][rr]concat=n=2:v=1:a=0[cycle];"
            f"[cycle]loop=loop=-1:size={cycle_frames}:start=0,"
            f"trim=duration={audio_duration:.6f},setpts=PTS-STARTPTS,"
            f"fps={fps},format=yuv420p[v]"
        )
        return graph, slow_duration, True

    async def _synchronize(
        self, state: dict[str, Any], scene: dict[str, Any], raw_video: Path, audio: Path,
    ) -> tuple[Path, dict[str, Any]]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise TheaterError("FFmpeg is required for theater playback.")
        directory = self._dir(state["id"])
        work = directory / "work"
        number = int(scene["number"])
        raw_duration = await self._duration(raw_video)
        audio_duration = await self._duration(audio)
        quality = self.quality_settings(state["config"])
        fps = int(quality["fps"])
        visual_filter, slow_duration, repeated = self._visual_sync_filter(
            raw_duration, audio_duration, quality,
        )
        segment = directory / "segments" / f"scene_{number:05d}.mp4"
        staged = work / f"scene_{number:05d}_synchronized.mp4"
        await self._run_ffmpeg([
            ffmpeg, "-y", "-i", str(raw_video), "-i", str(audio),
            "-filter_complex", visual_filter, "-map", "[v]", "-map", "1:a:0",
            "-t", f"{audio_duration:.3f}", "-threads", "4", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-video_track_timescale", "90000",
            "-c:a", "aac", "-b:a", "160k",
            "-af", f"afade=t=in:st=0:d=0.08,afade=t=out:st={max(0.05,audio_duration-0.16):.6f}:d=0.14",
            "-movflags", "+faststart", str(staged),
        ], directory / "logs" / "ffmpeg.log")
        await asyncio.to_thread(staged.replace, segment)
        return segment, {
            "raw_video_duration": round(raw_duration, 3), "duration": round(audio_duration, 3),
            "slow_duration": round(slow_duration, 3), "motion_repeated": repeated,
            "estimated_motion_cycles": round(audio_duration / max(0.1, slow_duration * 2 if repeated else slow_duration), 2),
        }

    async def _render_scene(self, state: dict[str, Any], scene: dict[str, Any]) -> dict[str, Any]:
        """Create raw video and narration, leaving CPU muxing to another worker."""
        number = int(scene["number"])
        quality = self.quality_settings(state["config"])
        directory = self._dir(state["id"])
        cycle_started = time.perf_counter()
        audio_rel = f"wan_theater/{state['id']}/audio/scene_{number:05d}.wav"
        audio_path = self.output_root / audio_rel
        pairs = scene.get("narration_sentences")
        if not isinstance(pairs, list) or not pairs:
            pairs = [{"original": sentence} for sentence in split_narration_sentences(scene["narration"])]
        translation_language = self.translation_language(state["config"])
        if any(not str(pair.get("translation", "")).strip() for pair in pairs):
            raise TheaterError(f"Scene {number} is missing an aligned translation and cannot be narrated safely.")
        narration_speed = self._narration_speed(state, scene)
        scene["narration_speed"] = narration_speed
        tts_task = asyncio.create_task(self.supertonic.synthesize_alternating(
            pairs, audio_path,
            voice=str(state["config"].get("voice", "M1")),
            original_language=str(state["config"].get("language", "en")),
            translation_language=translation_language,
            speed=narration_speed,
        ))
        try:
            async with self.controller.workflow_lock:
                state["rendering_scene"] = number
                state.update(status="generating", current_scene=number, message=f"Wan is creating new visual {number} while Supertonic records narration on CPU...")
                self._save(state)
                video_config = {
                    "prompt": self._visual_prompt(state, scene),
                    "negative": "identity drift, changed clothing, changed location, duplicate person, extra characters, static, blurry, low detail, subtitles, watermark, text, malformed anatomy, NSFW",
                    "width": quality["width"], "height": quality["height"], "frames": quality["frames"],
                    "fps": quality["fps"], "seed": int(state["config"]["seed"]) + number * 9973,
                    "filename_prefix": f"wan_theater/{state['id']}/raw/scene_{number:05d}",
                }
                video_started = time.perf_counter()
                reply = await self.controller.submit(self.video_prompt_builder(video_config), f"theater-video-{state['id']}")
                files = await self._wait_prompt(reply["prompt_id"], state)
                video_rel = next((f["path"] for f in reversed(files) if f["filename"].lower().endswith((".mp4", ".webm"))), None)
                if not video_rel:
                    raise TheaterError(f"Scene {number} produced no video.")
                video_seconds = time.perf_counter() - video_started

            if not tts_task.done():
                state.update(status="narrating", message=f"Visual {number} is ready; finishing its CPU neural narration...")
                self._save(state)
            tts_seconds = await tts_task
            state.update(status="aligning", message=f"Whisper is aligning every spoken word in scene {number}...")
            self._save(state)
            async with self.controller.workflow_lock:
                await self.controller.free_models()
                alignment = await self.whisper.align(
                    audio_path,
                    spoken_text(pairs),
                    "auto",
                )
        except (Exception, asyncio.CancelledError):
            if not tts_task.done():
                tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
            raise
        finally:
            state.pop("rendering_scene", None)

        return {
            "scene": scene, "audio_rel": audio_rel, "audio_path": audio_path,
            "video_rel": video_rel, "video_seconds": video_seconds,
            "tts_seconds": tts_seconds, "cycle_started": cycle_started,
            "ready_seconds": time.perf_counter() - cycle_started,
            "alignment": alignment,
        }

    async def _assemble_scene(self, state: dict[str, Any], work_item: dict[str, Any]) -> None:
        """Stretch, interpolate, mux, and archive while Wan renders the next clip."""
        scene = work_item["scene"]
        number = int(scene["number"])
        state["assembling_scene"] = number
        if not state.get("rendering_scene"):
            state.update(status="buffering", message=f"Synchronizing scene {number} while preparing the next visual...")
            self._save(state)
        assembly_started = time.perf_counter()
        segment, sync = await self._synchronize(
            state, scene, self.output_root / work_item["video_rel"], work_item["audio_path"],
        )
        assembly_seconds = time.perf_counter() - assembly_started
        cycle_seconds = time.perf_counter() - float(work_item["cycle_started"])
        relative = str(segment.relative_to(self.output_root)).replace("\\", "/")
        completed_at = time.time()
        entry = {
            "number": number, "title": scene["title"], "beat": scene["beat"],
            "narration": scene["narration"],
            "translated_title": scene.get("translated_title", ""),
            "narration_sentences": scene.get("narration_sentences", []),
            "spoken_text": spoken_text(scene.get("narration_sentences", [])),
            "word_timestamps": work_item["alignment"]["words"],
            "speech_segments": work_item["alignment"]["segments"],
            "alignment_model": work_item["alignment"]["model"],
            "alignment_seconds": work_item["alignment"]["elapsed_seconds"],
            "source_word_count": scene.get("source_word_count"),
            "translation_word_count": scene.get("translation_word_count"),
            "total_spoken_words": scene.get("total_spoken_words"),
            "narration_speed": scene.get("narration_speed", self.DEFAULT_TTS_SPEED),
            "planner_metrics": scene.get("planner_metrics", {}),
            "translation_metrics": scene.get("translation_metrics", {}),
            "gpu_feed_wait_seconds": round(float(scene.get("gpu_feed_wait_seconds") or 0), 3),
            "source_language": scene.get("source_language", state["config"].get("language", "en")),
            "translation_language": scene.get("translation_language", ""),
            "visual_action": scene["visual_action"], "path": relative,
            "raw_video_path": work_item["video_rel"], "audio_path": work_item["audio_rel"], "created": completed_at,
            "asset_fingerprint": scene["asset_fingerprint"], **sync,
            "live_directive_ids": list(scene.get("_live_directive_ids") or []),
            "production_seconds": round(cycle_seconds, 3),
            "video_generation_seconds": round(float(work_item["video_seconds"]), 3),
            "tts_generation_seconds": round(float(work_item["tts_seconds"]), 3),
            "assembly_seconds": round(assembly_seconds, 3),
        }
        state.setdefault("segments", []).append(entry)
        for planned_scene in state.get("planned", []):
            if int(planned_scene["number"]) == number:
                planned_scene.pop("_planning_context_before", None)
                planned_scene.pop("_live_directive_ids", None)
                break
        state["total_duration"] = round(sum(float(item["duration"]) for item in state["segments"]), 3)
        previous_entry = state["segments"][-2] if len(state["segments"]) > 1 else None
        run_started_at = float(state.get("metrics", {}).get("run_started_at") or 0)
        previous_is_current_run = bool(
            previous_entry and (not run_started_at or float(previous_entry["created"]) >= run_started_at)
        )
        completion_interval = (
            completed_at - float(previous_entry["created"]) if previous_is_current_run
            else cycle_seconds
        )
        metrics = state.setdefault("metrics", {})
        previous_interval = float(metrics.get("completion_interval_ema") or 0)
        interval_ema = (
            completion_interval if not previous_interval
            else previous_interval * 0.65 + completion_interval * 0.35
        )
        total_words = int(entry.get("total_spoken_words") or 0)
        source_words = int(entry.get("source_word_count") or 0)
        seconds_per_word = float(entry["duration"]) / total_words if total_words else 0.0
        previous_spw = float(metrics.get("speech_seconds_per_word_ema") or 0)
        spw_ema = (
            seconds_per_word if not previous_spw else previous_spw * 0.65 + seconds_per_word * 0.35
        ) if seconds_per_word else previous_spw
        multiplier = total_words / source_words if source_words else 1.0
        previous_multiplier = float(metrics.get("spoken_word_multiplier_ema") or 0)
        multiplier_ema = (
            multiplier if not previous_multiplier else previous_multiplier * 0.65 + multiplier * 0.35
        )
        state["metrics"].update({
            "production_ema": round(interval_ema, 3),
            "completion_interval_ema": round(interval_ema, 3),
            "last_completion_interval_seconds": round(completion_interval, 3),
            "coverage_ratio": round(float(entry["duration"]) / max(0.1, interval_ema), 3),
            "speech_seconds_per_word_ema": round(spw_ema, 5),
            "spoken_word_multiplier_ema": round(multiplier_ema, 3),
            "last_narration_speed": float(entry["narration_speed"]),
            "coverage_target": self.COVERAGE_TARGET,
            "last_video_seconds": round(float(work_item["video_seconds"]), 3),
            "last_tts_seconds": round(float(work_item["tts_seconds"]), 3),
            "last_assembly_seconds": round(assembly_seconds, 3),
            "parallel_planner": True, "continuous_gpu_pipeline": True,
        })
        state["metrics"]["target_total_spoken_words"] = self._target_total_words(state)
        state.pop("assembling_scene", None)
        if state.get("rendering_scene"):
            state["status"] = "generating"
            state["message"] = f"Scene {number} is archived while Wan continues visual {state['rendering_scene']}."
        else:
            state["status"] = "running"
            state["message"] = f"Scene {number} is ready and archived; the next unique visual is starting."
        self._save(state)

    async def _assembly_loop(self, state: dict[str, Any], queue: asyncio.Queue[dict[str, Any]]) -> None:
        while True:
            work_item = await queue.get()
            try:
                await self._assemble_scene(state, work_item)
            finally:
                queue.task_done()

    def _pending_prepared_count(
        self, state: dict[str, Any], excluded_numbers: set[int] | None = None,
    ) -> int:
        produced = {int(item["number"]) for item in state.get("segments", [])}
        produced.update(excluded_numbers or set())
        return sum(
            1 for scene in state.get("planned", [])
            if int(scene["number"]) not in produced
            and self._narration_is_prepared(state["config"], scene)
        )

    async def _fill_story_buffer(
        self, state: dict[str, Any], target: int, excluded_numbers: set[int] | None = None,
    ) -> int:
        """Use the active writer profile to durably prepare a bounded scene batch."""
        produced = {int(item["number"]) for item in state.get("segments", [])}
        produced.update(excluded_numbers or set())
        pending = [
            scene for scene in state.get("planned", [])
            if int(scene["number"]) not in produced
        ]
        capacity = max(target + 2, len(pending) + 2)
        source_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=capacity)
        ready_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=capacity)
        for scene in pending:
            if not self._narration_is_prepared(state["config"], scene):
                await source_queue.put(scene)

        planner_task = asyncio.create_task(
            self._planner_loop(state, source_queue), name=f"gpu-burst-planner-{state['id']}",
        )
        translation_task = asyncio.create_task(
            self._translation_loop(state, source_queue, ready_queue, planner_task),
            name=f"gpu-burst-translator-{state['id']}",
        )
        try:
            while self._pending_prepared_count(state, excluded_numbers) < target:
                scene = await self._next_planned_scene(ready_queue, translation_task)
                if scene.get("_error"):
                    raise TheaterError(scene["_error"])
                if scene.get("_complete"):
                    return self._pending_prepared_count(state, excluded_numbers)
                state.setdefault("metrics", {})["gpu_burst_prepared_scenes"] = self._pending_prepared_count(
                    state, excluded_numbers,
                )
                self._save(state)
            return self._pending_prepared_count(state, excluded_numbers)
        finally:
            planner_task.cancel()
            translation_task.cancel()
            await asyncio.gather(planner_task, translation_task, return_exceptions=True)

    async def _prime_gpu_story_buffer(
        self, state: dict[str, Any], *, reason: str = "opening",
        excluded_numbers: set[int] | None = None,
    ) -> int:
        """Build a bounded text buffer on CUDA, then synchronously release VRAM."""
        metrics = state.setdefault("metrics", {})
        burst_started = time.perf_counter()
        load_started = burst_started
        load_seconds = 0.0
        offload_seconds = 0.0
        prepared = self._pending_prepared_count(state, excluded_numbers)
        async with self.controller.workflow_lock:
            await self.controller.wait_until_idle()
            await self.controller.free_models()
            try:
                state.update(
                    status="planning",
                    message=(
                        f"Loading {self.writer.model_label} on the RTX GPU to "
                        f"{'refill' if reason == 'adaptive_refill' else 'prepare'} "
                        f"{self.GPU_BURST_TARGET} scenes ahead..."
                    ),
                )
                metrics.update({"writer_mode": "gpu_burst", "gpu_burst_active": True})
                self._save(state)
                await self.writer.start(self._dir(state["id"]) / "logs", profile="gpu")
                load_seconds = time.perf_counter() - load_started
                if not state.get("bible"):
                    bootstrap = await self._bootstrap(state)
                    state.update(
                        title=bootstrap["title"], bible=bootstrap["bible"],
                        story_summary=bootstrap["story_summary"], bootstrap_scene=bootstrap["scene"],
                    )
                    self._save(state)
                if reason == "adaptive_refill":
                    await self._compact_story_context(state, reason="adaptive_refill")
                elif self._context_compaction_due(state):
                    await self._compact_story_context(state, reason="gpu_resume_interval")
                prepared = await self._fill_story_buffer(
                    state, self.GPU_BURST_TARGET, excluded_numbers,
                )
            finally:
                offload_started = time.perf_counter()
                try:
                    await self.writer.stop("gpu")
                    if await self.writer.healthy("gpu"):
                        raise GpuReleaseError(
                            "The CUDA story writer is still running, so Wan was not started to avoid a VRAM collision."
                        )
                except GpuReleaseError:
                    raise
                except Exception as exc:
                    raise GpuReleaseError(
                        f"The CUDA story writer could not be stopped safely: {exc}"
                    ) from exc
                finally:
                    offload_seconds = time.perf_counter() - offload_started
                    self.writer.activate("cpu")
                    metrics["gpu_burst_active"] = False

        total_seconds = round(time.perf_counter() - burst_started, 3)
        metrics.update({
            "gpu_burst_completed": True,
            "gpu_burst_prepared_scenes": prepared,
            "gpu_burst_load_seconds": round(load_seconds, 3),
            "gpu_burst_offload_seconds": round(offload_seconds, 3),
            "gpu_burst_total_seconds": total_seconds,
            "gpu_burst_count": int(metrics.get("gpu_burst_count") or 0) + 1,
        })
        if reason == "adaptive_refill":
            metrics["gpu_refill_count"] = int(metrics.get("gpu_refill_count") or 0) + 1
        events = list(metrics.get("gpu_burst_events") or [])[-19:]
        events.append({
            "reason": reason, "completed_at": time.time(), "prepared_scenes": prepared,
            "load_seconds": round(load_seconds, 3), "offload_seconds": round(offload_seconds, 3),
            "total_seconds": total_seconds,
        })
        metrics["gpu_burst_events"] = events
        self._save(state)
        return prepared

    def _restore_story_workers(
        self, state: dict[str, Any], excluded_numbers: set[int],
    ) -> tuple[
        asyncio.Queue[dict[str, Any]], asyncio.Queue[dict[str, Any]],
        asyncio.Task[None], asyncio.Task[None],
    ]:
        """Rebuild bounded CPU queues solely from the durable plan archive."""
        completed = {int(item["number"]) for item in state.get("segments", [])}
        completed.update(excluded_numbers)
        pending = [
            item for item in state.get("planned", [])
            if int(item["number"]) not in completed
        ]
        capacity = max(3, len(pending) + 1)
        source_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=capacity)
        ready_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=capacity)
        for scene in pending:
            target = ready_queue if self._narration_is_prepared(state["config"], scene) else source_queue
            target.put_nowait(scene)
        planner_task = asyncio.create_task(
            self._planner_loop(state, source_queue), name=f"cpu-planner-{state['id']}",
        )
        translation_task = asyncio.create_task(
            self._translation_loop(state, source_queue, ready_queue, planner_task),
            name=f"cpu-translator-{state['id']}",
        )
        return source_queue, ready_queue, planner_task, translation_task

    @staticmethod
    async def _cancel_story_workers(
        planner_task: asyncio.Task[None] | None, translation_task: asyncio.Task[None] | None,
    ) -> None:
        workers = [task for task in (planner_task, translation_task) if task]
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def _predicted_cpu_ready_wait(
        self, state: dict[str, Any], source_queue: asyncio.Queue[dict[str, Any]],
    ) -> float:
        """Estimate the next validated scene wait from live and completed CPU stages."""
        metrics = state.setdefault("metrics", {})
        now = time.time()
        planner_seconds = float(
            metrics.get("planner_cycle_ema") or metrics.get("planner_elapsed_ema")
            or metrics.get("planner_elapsed_seconds") or 0
        )
        translation_seconds = float(
            metrics.get("translation_cycle_ema") or metrics.get("translation_elapsed_ema")
            or metrics.get("translation_elapsed_seconds") or 0
        )
        translation_started = float(
            metrics.get("translation_cycle_started_at") or metrics.get("translation_request_started_at") or 0
        )
        compaction_started = float(metrics.get("context_compaction_started_at") or 0)
        planner_started = float(
            metrics.get("planner_cycle_started_at") or metrics.get("planner_request_started_at") or 0
        )
        if translation_started:
            return max(0.0, translation_seconds - max(0.0, now - translation_started))
        if source_queue.qsize():
            return translation_seconds
        if compaction_started:
            expected_compaction = float(metrics.get("context_compaction_elapsed_ema") or 0)
            if not expected_compaction:
                expected_compaction = planner_seconds * 1.25
            compaction_remaining = max(0.0, expected_compaction - max(0.0, now - compaction_started))
            return compaction_remaining + planner_seconds + translation_seconds
        if planner_started:
            planner_remaining = max(0.0, planner_seconds - max(0.0, now - planner_started))
            return planner_remaining + translation_seconds
        return 0.0

    def _should_refill_on_gpu(
        self, state: dict[str, Any], ready_queue: asyncio.Queue[dict[str, Any]],
        source_queue: asyncio.Queue[dict[str, Any]],
    ) -> bool:
        """Refill only when measured CPU wait is worse than the bounded GPU swap."""
        metrics = state.setdefault("metrics", {})
        if (
            not self.writer.gpu_available or not ready_queue.empty()
            or metrics.get("gpu_refill_disabled") or metrics.get("gpu_burst_fallback_reason")
            or state.get("story_source", {}).get("complete")
        ):
            return False
        predicted = self._predicted_cpu_ready_wait(state, source_queue)
        measured_burst = float(metrics.get("gpu_burst_total_seconds") or 0)
        threshold = max(
            self.GPU_REFILL_MIN_PREDICTED_WAIT,
            measured_burst * 0.75 if measured_burst else self.GPU_REFILL_MIN_PREDICTED_WAIT,
        )
        metrics["gpu_refill_predicted_cpu_wait"] = round(predicted, 3)
        metrics["gpu_refill_trigger_seconds"] = round(threshold, 3)
        return predicted >= threshold

    async def _run(self, state: dict[str, Any]) -> None:
        planner_task: asyncio.Task[None] | None = None
        translation_task: asyncio.Task[None] | None = None
        assembly_task: asyncio.Task[None] | None = None
        try:
            state.setdefault("metrics", {})["run_started_at"] = time.time()
            state.update(status="starting", message="Loading the neural voice...")
            self._save(state)
            await self.supertonic.start(self._dir(state["id"]) / "logs")
            if self.writer.gpu_available and self._pending_prepared_count(state) < self.GPU_BURST_TARGET:
                try:
                    await self._prime_gpu_story_buffer(state)
                except asyncio.CancelledError:
                    raise
                except GpuReleaseError:
                    raise
                except Exception as exc:
                    LOGGER.exception("CUDA story-buffer burst failed; continuing with the same Gemma model on CPU")
                    state.setdefault("metrics", {}).update({
                        "gpu_burst_active": False,
                        "gpu_burst_completed": False,
                        "gpu_burst_fallback_reason": str(exc)[:500],
                        "gpu_refill_disabled": True,
                    })
                    state["message"] = "The CUDA preload was unavailable; continuing safely with Gemma on CPU."
                    self._save(state)

            state.update(status="starting", message="Starting the resident CPU writer for continuous scene planning...")
            state.setdefault("metrics", {})["writer_mode"] = "cpu_sustain"
            self._save(state)
            await self.writer.start(self._dir(state["id"]) / "logs", profile="cpu")
            if not state.get("bible"):
                state.update(status="planning", message=f"{self.writer.model_label} is creating the endless story bible and first scene...")
                self._save(state)
                bootstrap = await self._bootstrap(state)
                state.update(
                    title=bootstrap["title"], bible=bootstrap["bible"], story_summary=bootstrap["story_summary"],
                    bootstrap_scene=bootstrap["scene"],
                )
                self._save(state)
            rendered_numbers = {int(item["number"]) for item in state.get("segments", [])}
            assembly_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=3)
            source_queue, ready_queue, planner_task, translation_task = self._restore_story_workers(
                state, rendered_numbers,
            )
            assembly_task = asyncio.create_task(self._assembly_loop(state, assembly_queue))
            steering_event = self.steering_events.setdefault(state["id"], asyncio.Event())

            async def apply_live_steering() -> bool:
                nonlocal source_queue, ready_queue, planner_task, translation_task
                if not steering_event.is_set():
                    return False
                await self._cancel_story_workers(planner_task, translation_task)
                planner_task = None
                translation_task = None
                discarded = self._rollback_speculative_plans(state, rendered_numbers)
                steering_event.clear()
                state.setdefault("metrics", {})["last_live_steering_applied_at"] = time.time()
                state["message"] = (
                    f"Live direction accepted; replaced {discarded} speculative scene plan"
                    f"{'s' if discarded != 1 else ''}. Completed media was kept."
                )
                source_queue, ready_queue, planner_task, translation_task = self._restore_story_workers(
                    state, rendered_numbers,
                )
                self._save(state)
                return True

            while True:
                if assembly_task.done():
                    error = assembly_task.exception()
                    if error:
                        raise TheaterError(f"The scene assembly worker failed: {error}") from error
                    raise TheaterError("The scene assembly worker stopped unexpectedly.")
                await apply_live_steering()
                ready_wait_started = time.perf_counter()
                while True:
                    if self._should_refill_on_gpu(state, ready_queue, source_queue):
                        predicted_wait = float(state["metrics"].get("gpu_refill_predicted_cpu_wait") or 0)
                        await self._cancel_story_workers(planner_task, translation_task)
                        planner_task = None
                        translation_task = None
                        state.update(
                            status="planning",
                            message=(
                                f"The CPU writer predicts a {predicted_wait:.0f}-second wait; "
                                "using the idle RTX window to refill three scenes..."
                            ),
                        )
                        self._save(state)
                        try:
                            await self._prime_gpu_story_buffer(
                                state, reason="adaptive_refill", excluded_numbers=rendered_numbers,
                            )
                        except asyncio.CancelledError:
                            raise
                        except GpuReleaseError:
                            raise
                        except Exception as exc:
                            LOGGER.exception("Adaptive CUDA story refill failed; disabling it for this session")
                            state["metrics"].update({
                                "gpu_refill_disabled": True,
                                "gpu_refill_disabled_reason": str(exc)[:500],
                            })
                        finally:
                            state["metrics"]["writer_mode"] = "cpu_sustain"
                            source_queue, ready_queue, planner_task, translation_task = self._restore_story_workers(
                                state, rendered_numbers,
                            )
                            self._save(state)
                        continue
                    try:
                        scene = await asyncio.wait_for(
                            self._next_planned_scene(ready_queue, translation_task),
                            timeout=self.GPU_REFILL_POLL_SECONDS,
                        )
                        break
                    except asyncio.TimeoutError:
                        continue
                ready_wait_seconds = time.perf_counter() - ready_wait_started
                if scene.get("_error"):
                    raise TheaterError(scene["_error"])
                if scene.get("_complete"):
                    await assembly_queue.join()
                    state.pop("rendering_scene", None)
                    state.pop("assembling_scene", None)
                    state["status"] = "complete"
                    state["message"] = "My story is complete. Every scene is archived."
                    self._save(state)
                    return
                if await apply_live_steering():
                    continue
                if int(scene["number"]) in rendered_numbers:
                    continue
                scene["gpu_feed_wait_seconds"] = round(ready_wait_seconds, 3)
                state["metrics"]["last_gpu_feed_wait_seconds"] = round(ready_wait_seconds, 3)
                state["metrics"]["gpu_feed_wait_seconds"] = round(
                    float(state["metrics"].get("gpu_feed_wait_seconds") or 0) + ready_wait_seconds,
                    3,
                )
                state["metrics"]["translated_scene_queue"] = ready_queue.qsize()
                work_item = await self._render_scene(state, scene)
                rendered_numbers.add(int(scene["number"]))
                if state.get("metrics", {}).get("live_steering_legacy_delay_through_scene"):
                    steering_event.set()
                if assembly_task.done():
                    error = assembly_task.exception()
                    if error:
                        raise TheaterError(f"The scene assembly worker failed: {error}") from error
                    raise TheaterError("The scene assembly worker stopped unexpectedly.")
                await assembly_queue.put(work_item)
        except asyncio.CancelledError:
            for child in (planner_task, translation_task, assembly_task):
                if child:
                    child.cancel()
            await asyncio.gather(
                *(child for child in (planner_task, translation_task, assembly_task) if child),
                return_exceptions=True,
            )
            raise
        except Exception as exc:
            for child in (planner_task, translation_task, assembly_task):
                if child:
                    child.cancel()
            await asyncio.gather(
                *(child for child in (planner_task, translation_task, assembly_task) if child),
                return_exceptions=True,
            )
            LOGGER.exception("Theater session %s failed", state["id"])
            state["status"] = "failed"
            state["message"] = str(exc)
            state["error"] = str(exc)
            self._save(state)
