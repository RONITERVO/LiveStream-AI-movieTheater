from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from aiohttp import web

from adapters.comfy import ComfyAdapter, build_wan_prompt
from story_domain import (
    CINEMA_DEFAULTS,
    DEFAULT_TRANSLATION_LANGUAGE,
    LANGUAGE_NAMES,
    TRANSLATION_LANGUAGES,
    VOICES,
    TheaterError,
    validate_story_request,
)
from theater_pipeline import TheaterManager


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


AI_ROOT = _env_path("WAN_AI_ROOT", Path(r"D:\AI"))
LOCAL_AI_ROOT = _env_path("WAN_LOCAL_AI_ROOT", Path(r"D:\LocalAI"))
COMFY_ROOT = _env_path("WAN_COMFY_ROOT", AI_ROOT / "ComfyUI")
STORY_MODEL_ROOT = _env_path("WAN_GEMMA4_ROOT", LOCAL_AI_ROOT / "Gemma4E4B")
LLAMA_RUNTIME_ROOT = _env_path("WAN_LLAMA_RUNTIME_ROOT", STORY_MODEL_ROOT)
CUDA_LLAMA_RUNTIME_ROOT = _env_path("WAN_CUDA_LLAMA_RUNTIME_ROOT", LOCAL_AI_ROOT / "Bonsai27B")
SUPERTONIC_ROOT = _env_path("WAN_SUPERTONIC_ROOT", LOCAL_AI_ROOT / "Supertonic3")
WHISPER_ROOT = _env_path("WAN_WHISPER_ROOT", COMFY_ROOT / "models" / "stt" / "whisper")
COMFY_URL = os.environ.get("WAN_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
OUTPUT_ROOT = _env_path("WAN_OUTPUT_ROOT", COMFY_ROOT / "output")
HOST = os.environ.get("WAN_HOST", "127.0.0.1")
PORT = int(os.environ.get("WAN_PORT", "7868"))

LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "wan-video-ui.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("wan-video-ui")
RUNTIME_STATE = web.AppKey("runtime_state", dict)

CONTROLLER = ComfyAdapter(COMFY_ROOT, COMFY_URL, LOG_DIR)
THEATER = TheaterManager(
    APP_DIR,
    OUTPUT_ROOT,
    STORY_MODEL_ROOT,
    LLAMA_RUNTIME_ROOT,
    SUPERTONIC_ROOT,
    WHISPER_ROOT,
    CONTROLLER,
    build_wan_prompt,
    cuda_llama_runtime_root=CUDA_LLAMA_RUNTIME_ROOT,
)


async def index(_: web.Request) -> web.FileResponse:
    return web.FileResponse(STATIC_DIR / "index.html")


async def api_status(_: web.Request) -> web.Response:
    stats = await CONTROLLER.stats()
    if not stats:
        return web.json_response({
            "ready": False,
            "starting": CONTROLLER.starting,
            "word_alignment": THEATER.whisper.available(),
        })
    device = (stats.get("devices") or [{}])[0]
    system = stats.get("system", {})
    return web.json_response({
        "ready": True,
        "starting": False,
        "device": device.get("name", "NVIDIA GPU"),
        "vram_gb": round(float(device.get("vram_total", 0)) / (1024**3), 1),
        "comfy_version": system.get("comfyui_version", "unknown"),
        "word_alignment": THEATER.whisper.available(),
    })


async def api_config(_: web.Request) -> web.Response:
    mode_labels = {
        "story": "Pure Story",
        "interactive": "Interactive Character Show",
        "my_story": "My story",
    }
    audience_labels = {
        "young": "Young",
        "family": "Family",
        "teen": "Teen",
        "adult": "Adult",
    }
    return web.json_response({
        "modes": [
            {"value": value, "label": mode_labels[value]}
            for value in ("story", "interactive", "my_story")
        ],
        "audiences": [
            {"value": value, "label": audience_labels[value]}
            for value in ("young", "family", "teen", "adult")
        ],
        "languages": [
            {
                "value": value,
                "label": label,
                "flag": f"flag-{value}",
                "translation": value in TRANSLATION_LANGUAGES,
            }
            for value, label in LANGUAGE_NAMES.items()
        ],
        "default_translation_language": DEFAULT_TRANSLATION_LANGUAGE,
        "voices": sorted(VOICES, key=lambda value: (value[0], int(value[1:]))),
        "quality": CINEMA_DEFAULTS,
    })


async def api_video(request: web.Request) -> web.StreamResponse:
    relative = request.query.get("path", "")
    if not relative:
        raise web.HTTPBadRequest(text="Missing path")
    candidate = (OUTPUT_ROOT / relative).resolve()
    try:
        candidate.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise web.HTTPForbidden(text="Invalid output path") from exc
    allowed = {".mp4", ".webm", ".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    if candidate.suffix.lower() not in allowed or not candidate.is_file():
        raise web.HTTPNotFound(text="Media not found")
    return web.FileResponse(candidate)


async def api_theater_start(request: web.Request) -> web.Response:
    try:
        raw = await request.json()
        return web.json_response(THEATER.start(validate_story_request(raw)))
    except (TheaterError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_recent(_: web.Request) -> web.Response:
    return web.json_response({"sessions": THEATER.recent()})


async def api_theater_status(request: web.Request) -> web.Response:
    state = THEATER.get(request.match_info["session_id"])
    if not state:
        raise web.HTTPNotFound(text="Theater session not found")
    return web.json_response(state)


async def api_theater_stop(request: web.Request) -> web.Response:
    try:
        return web.json_response(await THEATER.stop(request.match_info["session_id"]))
    except TheaterError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_resume(request: web.Request) -> web.Response:
    try:
        return web.json_response(THEATER.resume(request.match_info["session_id"]))
    except TheaterError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_live_directive(request: web.Request) -> web.Response:
    try:
        raw = await request.json()
        if not isinstance(raw, dict):
            raise ValueError("Request body must be a JSON object.")
        return web.json_response(THEATER.add_live_directive(
            request.match_info["session_id"],
            raw.get("text", ""),
            str(raw.get("scope", "next_scene")),
            str(raw.get("delivery", "after_buffer")),
        ))
    except (TheaterError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_remove_directive(request: web.Request) -> web.Response:
    try:
        return web.json_response(THEATER.remove_live_directive(
            request.match_info["session_id"],
            request.match_info["directive_id"],
        ))
    except TheaterError as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def api_theater_voice_preview(request: web.Request) -> web.Response:
    try:
        raw = await request.json()
        voice = str(raw.get("voice", "M1")).upper()
        language = str(raw.get("language", "en")).lower()
        if voice not in VOICES or language not in LANGUAGE_NAMES:
            raise ValueError("Choose a supported voice and language.")
        samples = {
            "fi": "Tervetuloa loputtomaan teatteriin. Jokainen uusi kohtaus jatkaa tarinaa.",
            "en": "Welcome to the endless theater. Every new scene continues the story.",
        }
        text = samples.get(language, samples["en"])
        relative = f"wan_theater/_voice_previews/{voice}_{language}.wav"
        output = OUTPUT_ROOT / relative
        if not output.exists():
            await THEATER.supertonic.start(LOG_DIR)
            await THEATER.supertonic.synthesize(text, output, voice=voice, language=language)
        return web.json_response({"path": relative, "voice": voice, "language": language})
    except (TheaterError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


def _is_local_exit_request(request: web.Request) -> bool:
    remote = request.remote or ""
    same_machine = remote in {"127.0.0.1", "::1"} or remote.startswith("::ffff:127.")
    return same_machine and request.headers.get("X-Wan-Local-Exit") == "release-owned-resources"


async def _release_owned_resources(app: web.Application) -> None:
    runtime = app[RUNTIME_STATE]

    async def cancel_background_start() -> None:
        task = runtime.get("comfy_start_task")
        if task and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    steps = (
        cancel_background_start,
        CONTROLLER.interrupt,
        THEATER.shutdown,
        CONTROLLER.free_models,
        CONTROLLER.stop_owned_process,
    )
    for action in steps:
        try:
            await action()
        except Exception:
            LOGGER.exception("Could not release an app-owned resource")
    runtime["shutdown_complete"] = True
    runtime["shutdown_event"].set()


async def api_shutdown(request: web.Request) -> web.Response:
    if not _is_local_exit_request(request):
        raise web.HTTPForbidden(text="Local exit authorization required")
    runtime = request.app[RUNTIME_STATE]
    if not runtime.get("resource_release_task"):
        runtime["resource_release_task"] = asyncio.create_task(
            _release_owned_resources(request.app),
            name="release-owned-resources",
        )
    return web.json_response(
        {"status": "shutting_down", "message": "Archiving work and releasing RAM and VRAM."},
        status=202,
    )


async def on_startup(app: web.Application) -> None:
    await CONTROLLER.open()
    THEATER.load_existing()
    (APP_DIR / "wan-video-ui.pid").write_text(str(os.getpid()), encoding="utf-8")
    app[RUNTIME_STATE]["comfy_start_task"] = asyncio.create_task(
        CONTROLLER.ensure_ready(),
        name="comfy-background-start",
    )


async def on_cleanup(app: web.Application) -> None:
    task = app[RUNTIME_STATE].get("resource_release_task")
    if task:
        await task
    else:
        await _release_owned_resources(app)
    await CONTROLLER.close()
    (APP_DIR / "wan-video-ui.pid").unlink(missing_ok=True)


def create_app() -> web.Application:
    app = web.Application(client_max_size=65 * 1024 * 1024)
    app[RUNTIME_STATE] = {
        "shutdown_event": asyncio.Event(),
        "shutdown_complete": False,
        "resource_release_task": None,
        "comfy_start_task": None,
    }
    app.router.add_get("/", index)
    app.router.add_get("/api/config", api_config)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/video", api_video)
    app.router.add_post("/api/theater", api_theater_start)
    app.router.add_get("/api/theater", api_theater_recent)
    app.router.add_post("/api/theater/voice-preview", api_theater_voice_preview)
    app.router.add_get("/api/theater/{session_id}", api_theater_status)
    app.router.add_post("/api/theater/{session_id}/stop", api_theater_stop)
    app.router.add_post("/api/theater/{session_id}/resume", api_theater_resume)
    app.router.add_post("/api/theater/{session_id}/directives", api_theater_live_directive)
    app.router.add_delete("/api/theater/{session_id}/directives/{directive_id}", api_theater_remove_directive)
    app.router.add_post("/api/shutdown", api_shutdown)
    app.router.add_static("/static/", STATIC_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


async def _serve() -> None:
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=HOST, port=PORT)
    await site.start()
    try:
        await app[RUNTIME_STATE]["shutdown_event"].wait()
    finally:
        await runner.cleanup()


def main() -> None:
    if not COMFY_ROOT.exists():
        raise SystemExit(f"ComfyUI was not found at {COMFY_ROOT}")
    LOGGER.info("Starting Wan Endless Theater on http://%s:%s", HOST, PORT)
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
