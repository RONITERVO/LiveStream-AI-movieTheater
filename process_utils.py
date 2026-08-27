from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any


def hidden_process_kwargs() -> dict[str, Any]:
    """Create a Windows process without allocating or flashing a console host."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


async def terminate_process_tree(process: Any, timeout: float = 10.0) -> None:
    """Terminate one owned process and its descendants without matching by name."""
    if not process or process.poll() is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            **hidden_process_kwargs(),
        )
        await killer.wait()
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await asyncio.to_thread(process.wait)
        return
    process.terminate()
    try:
        await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await asyncio.to_thread(process.wait)
