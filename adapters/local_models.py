from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from process_utils import terminate_process_tree
from story_domain import LANGUAGES, TRANSLATION_LANGUAGES, VOICES, TheaterError


class StoryRuntime:
    """Owned llama.cpp boundary for the one supported Gemma story model."""

    GEMMA4_E4B_ALIAS = "gemma4-e4b-theater"

    def __init__(
        self,
        app_dir: Path,
        model_root: Path,
        runtime_root: Path,
        cuda_runtime_root: Path | None = None,
    ) -> None:
        self.model_root = model_root
        self.runtime_root = runtime_root
        self.cuda_runtime_root = cuda_runtime_root or runtime_root
        self.profile = "cpu"
        self.urls = {"cpu": "http://127.0.0.1:8083", "gpu": "http://127.0.0.1:18083"}
        self.url = self.urls[self.profile]
        self.processes: dict[str, subprocess.Popen[bytes] | None] = {"cpu": None, "gpu": None}
        self.start_locks = {"cpu": asyncio.Lock(), "gpu": asyncio.Lock()}
        self.pid_files = {
            "cpu": app_dir / "theater-story-writer.pid",
            "gpu": app_dir / "theater-story-writer-gpu.pid",
        }
        self.process: subprocess.Popen[bytes] | None = None
        self.pid_file = self.pid_files["cpu"]
        self.model = model_root / "models" / "gemma-4-E4B-it-Q4_K_M.gguf"
        self.model_alias = self.GEMMA4_E4B_ALIAS
        self.model_label = "Gemma 4 E4B Q4_K_M"
        self.threads = 8
        self.parallel_slots = 2
        self.context_tokens_per_slot = 16_384
        self.sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "presence_penalty": 0.0}

    @property
    def gpu_available(self) -> bool:
        runtime = self.cuda_runtime_root / "runtime"
        return (runtime / "llama-server.exe").exists() and (runtime / "ggml-cuda.dll").exists()

    def activate(self, profile: str) -> None:
        if profile not in self.urls:
            raise ValueError(f"Unknown story-writer profile: {profile}")
        self.profile = profile
        self.url = self.urls[profile]
        self.process = self.processes[profile]
        self.pid_file = self.pid_files[profile]

    def _server_args(self, server: Path, profile: str = "cpu") -> list[str]:
        args = [
            str(server), "-m", str(self.model), "--alias", self.model_alias,
            "--host", "127.0.0.1", "--port", "18083" if profile == "gpu" else "8083",
            "-ngl", "99" if profile == "gpu" else "0",
            "-t", str(self.threads), "-tb", str(self.threads),
            "-c", str(self.context_tokens_per_slot * self.parallel_slots),
            "--parallel", str(self.parallel_slots), "--batch-size", "512", "--ubatch-size", "128",
            "--no-mmap", "--jinja", "--reasoning", "off", "--metrics",
        ]
        if profile == "gpu":
            args.extend(["-fa", "on", "-ctk", "f16", "-ctv", "f16", "--kv-offload", "--op-offload"])
        return args

    async def healthy(self, profile: str | None = None) -> bool:
        url = self.urls[profile or self.profile]
        try:
            async with ClientSession(timeout=ClientTimeout(total=2)) as session:
                async with session.get(f"{url}/health") as response:
                    if response.status != 200 or (await response.json()).get("status") != "ok":
                        return False
                async with session.get(f"{url}/v1/models") as response:
                    data = await response.json(content_type=None)
                    return response.status == 200 and any(
                        item.get("id") == self.model_alias for item in data.get("data", [])
                    )
        except Exception:
            return False

    async def start(self, log_dir: Path, profile: str = "cpu") -> None:
        if profile == "gpu" and not self.gpu_available:
            raise TheaterError(
                "The optional CUDA story-writer runtime is unavailable. Expected llama-server.exe and "
                f"ggml-cuda.dll under {self.cuda_runtime_root / 'runtime'}."
            )
        self.activate(profile)
        if await self.healthy(profile):
            if profile == "gpu" and self.processes[profile] is None:
                raise TheaterError(
                    "A CUDA story-writer server is already using port 18083 but is not owned by this app."
                )
            return
        async with self.start_locks[profile]:
            if await self.healthy(profile):
                return
            runtime_root = self.cuda_runtime_root if profile == "gpu" else self.runtime_root
            server = runtime_root / "runtime" / "llama-server.exe"
            if not server.exists() or not self.model.exists():
                raise TheaterError(
                    f"Gemma 4 E4B is required at {self.model} with llama.cpp at {server}."
                )
            log_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            device = "0" if profile == "gpu" else ""
            env.update({
                "GGML_CUDA_VISIBLE_DEVICES": device,
                "CUDA_VISIBLE_DEVICES": device,
                "LLAMA_ARG_CHAT_TEMPLATE_KWARGS": '{"enable_thinking":false}',
            })
            suffix = "-gpu" if profile == "gpu" else ""
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with (log_dir / f"writer{suffix}.out.log").open("ab") as stdout, (
                log_dir / f"writer{suffix}.err.log"
            ).open("ab") as stderr:
                process = subprocess.Popen(
                    self._server_args(server, profile),
                    cwd=runtime_root / "runtime",
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
            self.processes[profile] = process
            self.activate(profile)
            self.pid_files[profile].write_text(str(process.pid), encoding="utf-8")
            for _ in range(180):
                await asyncio.sleep(0.5)
                if await self.healthy(profile):
                    return
                if process.poll() is not None:
                    raise TheaterError(f"{self.model_label} exited while loading. Check writer{suffix}.err.log.")
            raise TheaterError(f"{self.model_label} did not become ready within 90 seconds.")

    async def stop(self, profile: str | None = None) -> None:
        selected = profile or self.profile
        process = self.processes[selected]
        if process and process.poll() is None:
            await terminate_process_tree(process)
        self.processes[selected] = None
        self.pid_files[selected].unlink(missing_ok=True)
        if self.profile == selected:
            self.activate("cpu")

    async def stop_all(self) -> None:
        await self.stop("gpu")
        await self.stop("cpu")

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 900,
    ) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        body = {
            "model": self.model_alias,
            "messages": messages,
            **self.sampling,
            "repeat_penalty": 1.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with ClientSession(timeout=ClientTimeout(total=300)) as session:
            async with session.post(f"{self.url}/v1/chat/completions", json=body) as response:
                data = await response.json(content_type=None)
                if response.status != 200:
                    raise TheaterError(data.get("error", {}).get("message") or str(data))
        usage = data.get("usage", {})
        elapsed = max(0.001, time.perf_counter() - started)
        return data["choices"][0]["message"]["content"], {
            "elapsed_seconds": round(elapsed, 3),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "tokens_per_second": round(int(usage.get("completion_tokens", 0)) / elapsed, 2),
        }


class SupertonicRuntime:
    """Owned, CPU-only Supertonic narration boundary."""

    VOICES = VOICES
    LANGUAGES = LANGUAGES
    TRANSLATION_LANGUAGES = TRANSLATION_LANGUAGES

    def __init__(self, app_dir: Path, root: Path) -> None:
        self.root = root
        self.url = "http://127.0.0.1:8084"
        self.process: subprocess.Popen[bytes] | None = None
        self.start_lock = asyncio.Lock()
        self.pid_file = app_dir / "theater-supertonic.pid"

    async def healthy(self) -> bool:
        try:
            async with ClientSession(timeout=ClientTimeout(total=2)) as session:
                async with session.get(f"{self.url}/v1/health") as response:
                    data = await response.json(content_type=None)
                    return response.status == 200 and data.get("status") == "ok"
        except Exception:
            return False

    async def start(self, log_dir: Path) -> None:
        if await self.healthy():
            return
        async with self.start_lock:
            if await self.healthy():
                return
            server = self.root / ".venv" / "Scripts" / "supertonic.exe"
            assets = self.root / "assets"
            if not server.exists() or not (assets / "onnx" / "vocoder.onnx").exists():
                raise TheaterError(f"Supertonic 3 is not installed in {self.root}.")
            log_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({
                "SUPERTONIC_CACHE_DIR": str(assets),
                "CUDA_VISIBLE_DEVICES": "",
                "ORT_DISABLE_ALL_CUDA": "1",
            })
            args = [
                str(server), "serve", "--host", "127.0.0.1", "--port", "8084",
                "--model", "supertonic-3", "--log-level", "warning",
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with (log_dir / "supertonic.out.log").open("ab") as stdout, (
                log_dir / "supertonic.err.log"
            ).open("ab") as stderr:
                self.process = subprocess.Popen(
                    args,
                    cwd=self.root,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                )
            self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
            for _ in range(180):
                await asyncio.sleep(0.25)
                if await self.healthy():
                    return
                if self.process.poll() is not None:
                    raise TheaterError("Supertonic 3 exited while loading. Check its theater log.")
            raise TheaterError("Supertonic 3 did not become ready within 45 seconds.")

    async def synthesize(
        self,
        text: str,
        output: Path,
        *,
        voice: str,
        language: str,
        speed: float = 1.05,
    ) -> float:
        started = time.perf_counter()
        body = {
            "text": text,
            "voice": voice,
            "lang": language,
            "speed": round(max(0.90, min(1.10, float(speed))), 3),
            "steps": 8,
            "silence_duration": 0.22,
            "response_format": "wav",
        }
        async with ClientSession(timeout=ClientTimeout(total=300)) as session:
            async with session.post(f"{self.url}/v1/tts", json=body) as response:
                data = await response.read()
                if response.status != 200:
                    raise TheaterError(f"Supertonic narration failed: {data.decode(errors='replace')[:500]}")
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(output.write_bytes, data)
        return time.perf_counter() - started

    @staticmethod
    def _concatenate_wavs(parts: list[Path], output: Path) -> None:
        if not parts:
            raise TheaterError("Bilingual narration contained no audio parts.")
        output.parent.mkdir(parents=True, exist_ok=True)
        expected: tuple[int, int, int, str] | None = None
        with wave.open(str(output), "wb") as destination:
            for part in parts:
                with wave.open(str(part), "rb") as source:
                    parameters = (
                        source.getnchannels(),
                        source.getsampwidth(),
                        source.getframerate(),
                        source.getcomptype(),
                    )
                    if expected is None:
                        expected = parameters
                        destination.setnchannels(parameters[0])
                        destination.setsampwidth(parameters[1])
                        destination.setframerate(parameters[2])
                        destination.setcomptype(parameters[3], source.getcompname())
                    elif parameters != expected:
                        raise TheaterError("Supertonic returned incompatible WAV formats for bilingual narration.")
                    destination.writeframes(source.readframes(source.getnframes()))

    async def synthesize_alternating(
        self,
        pairs: list[dict[str, str]],
        output: Path,
        *,
        voice: str,
        original_language: str,
        translation_language: str,
        speed: float = 1.05,
    ) -> float:
        if not translation_language:
            text = " ".join(str(pair.get("original", "")).strip() for pair in pairs).strip()
            return await self.synthesize(text, output, voice=voice, language=original_language, speed=speed)
        started = time.perf_counter()
        part_dir = output.parent / f".{output.stem}_parts"
        await asyncio.to_thread(shutil.rmtree, part_dir, True)
        part_dir.mkdir(parents=True, exist_ok=True)
        parts: list[Path] = []
        part_specs: list[dict[str, Any]] = []
        completed = False
        try:
            for index, pair in enumerate(pairs, 1):
                original = str(pair.get("original", "")).strip()
                translated = str(pair.get("translation", "")).strip()
                if not original or not translated:
                    raise TheaterError(f"Bilingual sentence {index} is incomplete.")
                for suffix, text, language in (
                    ("original", original, original_language),
                    ("translation", translated, translation_language),
                ):
                    part = part_dir / f"{index:03d}_{suffix}.wav"
                    await self.synthesize(text, part, voice=voice, language=language, speed=speed)
                    parts.append(part)
                    with wave.open(str(part), "rb") as source:
                        duration = source.getnframes() / max(1, source.getframerate())
                    part_specs.append({
                        "file": part.name,
                        "text": text,
                        "language": language,
                        "duration": duration,
                    })
            await asyncio.to_thread(self._concatenate_wavs, parts, output)
            offset = 0.0
            for spec in part_specs:
                spec["start"] = offset
                offset += float(spec.pop("duration"))
                spec["end"] = offset
            (part_dir / "alignment.json").write_text(
                json.dumps({"parts": part_specs}, ensure_ascii=False),
                encoding="utf-8",
            )
            completed = True
        finally:
            if not completed:
                await asyncio.to_thread(shutil.rmtree, part_dir, True)
        return time.perf_counter() - started

    async def stop(self) -> None:
        if self.process and self.process.poll() is None:
            await terminate_process_tree(self.process)
        self.process = None
        self.pid_file.unlink(missing_ok=True)
