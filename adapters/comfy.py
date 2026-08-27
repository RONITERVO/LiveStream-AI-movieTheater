from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from process_utils import hidden_process_kwargs, terminate_process_tree


DEFAULT_NEGATIVE = (
    "overexposed, static, blurry, low detail, subtitles, watermark, text, painting, "
    "still image, washed out, worst quality, low quality, JPEG artifacts, distorted "
    "hands, malformed fingers, deformed face, fused limbs, cluttered background, NSFW"
)


def build_wan_prompt(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    width = int(config["width"])
    height = int(config["height"])
    frames = int(config["frames"])
    fps = float(config["fps"])
    positive = str(config["prompt"]).strip()
    negative = str(config.get("negative", DEFAULT_NEGATIVE)).strip()
    prefix = str(config["filename_prefix"])
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default",
        }},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 0]}},
        "4": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {
            "width": width, "height": height, "length": frames, "batch_size": 1,
        }},
        "5": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default",
        }},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["5", 0],
            "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
            "strength_model": 1.0,
        }},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["6", 0], "shift": 5.0}},
        "8": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["7", 0], "add_noise": "enable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "start_at_step": 0, "end_at_step": 2,
            "return_with_leftover_noise": "enable",
        }},
        "9": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default",
        }},
        "10": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["9", 0],
            "lora_name": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
            "strength_model": 1.0,
        }},
        "11": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["10", 0], "shift": 5.0}},
        "12": {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["11", 0], "add_noise": "disable", "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["8", 0], "start_at_step": 2, "end_at_step": 4,
            "return_with_leftover_noise": "disable",
        }},
        "13": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["13", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": fps}},
        "16": {"class_type": "SaveVideo", "inputs": {
            "video": ["15", 0], "filename_prefix": prefix, "format": "mp4", "codec": "auto",
        }},
    }


class ComfyAdapter:
    """Loopback-only lifecycle and workflow boundary for ComfyUI."""

    def __init__(self, root: Path, url: str, log_dir: Path) -> None:
        self.root = root
        self.url = url.rstrip("/")
        self.log_dir = log_dir
        self.session: ClientSession | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.start_lock = asyncio.Lock()
        self.workflow_lock = asyncio.Lock()
        self.starting = False

    async def open(self) -> None:
        self.session = ClientSession(timeout=ClientTimeout(total=15))

    async def close(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None

    async def stop_owned_process(self) -> None:
        await terminate_process_tree(self.process)
        self.process = None

    async def stats(self) -> dict[str, Any] | None:
        assert self.session
        try:
            async with self.session.get(f"{self.url}/system_stats", timeout=3) as response:
                return await response.json() if response.status == 200 else None
        except Exception:
            return None

    async def ensure_ready(self) -> dict[str, Any]:
        current = await self.stats()
        if current:
            return current
        async with self.start_lock:
            current = await self.stats()
            if current:
                return current
            self.starting = True
            try:
                self._start_process()
                for _ in range(120):
                    await asyncio.sleep(0.75)
                    current = await self.stats()
                    if current:
                        return current
                    if self.process and self.process.poll() is not None:
                        raise RuntimeError("ComfyUI exited during startup. Check the UI logs folder.")
                raise RuntimeError("ComfyUI did not become ready within 90 seconds.")
            finally:
                self.starting = False

    def _start_process(self) -> None:
        python = self.root / ".venv" / "Scripts" / "python.exe"
        if not python.exists():
            raise RuntimeError(f"ComfyUI Python was not found at {python}")
        env = os.environ.copy()
        env.update({
            "PYTHONUTF8": "1",
            "HF_HOME": str(self.root / ".cache" / "huggingface"),
            "TRANSFORMERS_CACHE": str(self.root / ".cache" / "huggingface"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        })
        args = [
            str(python), "main.py", "--listen", "127.0.0.1", "--port", "8188", "--cuda-device", "0",
            "--enable-dynamic-vram", "--lowvram", "--reserve-vram", "1.5", "--cache-none",
            "--preview-method", "auto", "--fast-disk",
        ]
        with (self.log_dir / "comfyui.out.log").open("ab") as stdout, (
            self.log_dir / "comfyui.err.log"
        ).open("ab") as stderr:
            self.process = subprocess.Popen(
                args, cwd=self.root, env=env, stdout=stdout, stderr=stderr,
                **hidden_process_kwargs(),
            )

    async def submit(self, prompt: dict[str, Any], client_id: str) -> dict[str, Any]:
        assert self.session
        await self.ensure_ready()
        async with self.session.post(
            f"{self.url}/prompt", json={"prompt": prompt, "client_id": client_id}, timeout=30,
        ) as response:
            data = await response.json(content_type=None)
            if response.status != 200:
                raise RuntimeError(data.get("error", {}).get("message") or json.dumps(data))
            return data

    async def free_models(self) -> None:
        assert self.session
        try:
            async with self.session.post(
                f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=30,
            ) as response:
                await response.read()
        except Exception:
            return

    async def wait_until_idle(self, timeout_seconds: float = 900) -> None:
        assert self.session
        deadline = time.monotonic() + timeout_seconds
        while True:
            async with self.session.get(f"{self.url}/queue", timeout=10) as response:
                queue = await response.json(content_type=None)
            if not queue.get("queue_running") and not queue.get("queue_pending"):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the existing ComfyUI queue to become idle.")
            await asyncio.sleep(1)

    async def interrupt(self) -> None:
        assert self.session
        try:
            async with self.session.post(f"{self.url}/interrupt", timeout=10) as response:
                await response.read()
            async with self.session.post(f"{self.url}/queue", json={"clear": True}, timeout=10) as response:
                await response.read()
        except Exception:
            return

    async def job(self, prompt_id: str) -> dict[str, Any]:
        assert self.session
        async with self.session.get(f"{self.url}/history/{prompt_id}") as response:
            history = await response.json(content_type=None)
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            files: list[dict[str, str]] = []
            for output in item.get("outputs", {}).values():
                for value in output.values():
                    if not isinstance(value, list):
                        continue
                    for entry in value:
                        if isinstance(entry, dict) and entry.get("filename"):
                            relative = str(Path(entry.get("subfolder", "")) / entry["filename"])
                            files.append({"path": relative.replace("\\", "/"), "filename": entry["filename"]})
            error = next((
                message[1] for message in reversed(status.get("messages", []))
                if isinstance(message, list) and message and message[0] == "execution_error"
            ), None)
            return {
                "state": "failed" if error else ("complete" if status.get("completed") else "running"),
                "files": files,
                "error": error,
            }
        async with self.session.get(f"{self.url}/queue") as response:
            queue = await response.json(content_type=None)
        if any(len(item) > 1 and item[1] == prompt_id for item in queue.get("queue_running", [])):
            return {"state": "running", "files": []}
        if any(len(item) > 1 and item[1] == prompt_id for item in queue.get("queue_pending", [])):
            return {"state": "queued", "files": []}
        return {"state": "waiting", "files": []}
