# Wan Story Theater

A Windows-first, fully local narrated story theater powered by Wan 2.2, Gemma 4 E4B, Supertonic 3, and Whisper large-v3-turbo. After the models are installed, generation and playback use no cloud API, account, analytics, or telemetry. The app binds to `http://127.0.0.1:7868/`.

This repository contains application code only. Model weights, generated media, and third-party runtimes are excluded.

## Product surface

- **Pure Story** creates a continuing narrated story world.
- **Interactive Character Show** keeps a recurring host, setting, activity, live chat, and scene-level direction.
- **My Story** narrates pasted text exactly. The source is written once to a per-session file and read through a byte cursor, so book-sized input is not copied through every request or polling response.
- **Audience levels** remain available for young, family, teen, and adult experiences.
- **Narration** retains all 31 installed Supertonic languages and all ten voices.
- **Language-learning translation** speaks and displays each source sentence followed by its paired translation in any supported translation language.
- **Advanced Quality** retains resolution, source frames, playback FPS, word limits, maximum slow motion, context compaction, and deterministic or random seeds.
- **Exact word highlighting** uses Whisper timestamps aligned back to the displayed source and translation words. Clicking a word seeks the scene audio.
- **Durable output** uses ordinary JSON, MP4, WAV, SRT, M3U8, and UTF-8 text files.

There are no lesson, educational-grounding, or dream-mode branches in the pre-release contract.

## Architecture

The boundaries are intentionally narrow:

| Layer | Responsibility |
| --- | --- |
| `story_domain.py` | Pure validation, language rules, sentence pairing, budgets, and spoken-text order |
| `theater_pipeline.py` | Purely directed scene orchestration and durable session state |
| `adapters/` | Gemma, Supertonic, ComfyUI/Wan, Whisper, filesystem, process, and HTTP complexity |
| `app.py` | Aiohttp bootstrap and request/response boundary |
| `static/` | A backend-configured player and form with no product-policy duplication |

The session format is version 4. Earlier sessions are deliberately not migrated by this build; the pre-cleanup code and complete Git history are kept in the separately created archive tag and bundle.

## Requirements

- Windows 10 or 11
- NVIDIA GPU with at least 12 GB VRAM for the validated Wan 480×272 workflow; 16–24 GB is preferable
- 32 GB system RAM minimum; 64 GB preferable
- FFmpeg and FFprobe on `PATH`
- A working ComfyUI virtual environment containing PyTorch, CUDA, and OpenAI Whisper
- Gemma 4 E4B, Supertonic 3, and the required Wan 2.2 ComfyUI models

Default local layout:

```text
D:\AI\ComfyUI\
├── .venv\Scripts\python.exe
└── models\stt\whisper\large-v3-turbo.pt

D:\LocalAI\Gemma4E4B\
├── models\gemma-4-E4B-it-Q4_K_M.gguf
└── runtime\llama-server.exe

D:\LocalAI\Supertonic3\
```

The verified Whisper checkpoint SHA-256 is:

```text
aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a
```

The adapter requests `word_timestamps=True`, keeps exact segment and word timings, aligns recognized words to the visible bilingual token sequence, bridges short recognition gaps, and unloads Whisper before Wan resumes GPU work.

## Run

```powershell
python -m pip install -r requirements.txt
python app.py
```

On the tested layout, `Start Wan Video UI.cmd` uses ComfyUI's virtual-environment Python. `Launch Wan Video UI.vbs` starts it hidden, waits for readiness, and opens the local app. Launchers inherit `WAN_*` environment variables; `.env.example` is documentation and is not loaded automatically.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WAN_AI_ROOT` | `D:\AI` | AI application directory |
| `WAN_LOCAL_AI_ROOT` | `D:\LocalAI` | Local-model directory |
| `WAN_COMFY_ROOT` | `D:\AI\ComfyUI` | ComfyUI installation |
| `WAN_GEMMA4_ROOT` | `D:\LocalAI\Gemma4E4B` | Gemma model and llama.cpp runtime |
| `WAN_LLAMA_RUNTIME_ROOT` | same as Gemma root | Optional separate llama.cpp runtime |
| `WAN_CUDA_LLAMA_RUNTIME_ROOT` | `D:\LocalAI\Bonsai27B` | Optional bounded CUDA Gemma burst runtime |
| `WAN_SUPERTONIC_ROOT` | `D:\LocalAI\Supertonic3` | Offline neural speech service |
| `WAN_WHISPER_ROOT` | `<ComfyUI>\models\stt\whisper` | Verified word-alignment checkpoint directory |
| `WAN_COMFY_URL` | `http://127.0.0.1:8188` | ComfyUI endpoint |
| `WAN_OUTPUT_ROOT` | `<ComfyUI>\output` | Generated output root |
| `WAN_HOST` | `127.0.0.1` | Application bind address |
| `WAN_PORT` | `7868` | Application port |

Saved sessions live under `<WAN_OUTPUT_ROOT>\wan_theater`. A My Story session also contains `source.txt`; the polled session JSON contains only its cursor and progress metadata.

## Playback and synchronization

Gemma planning, Supertonic speech, Wan rendering, and FFmpeg assembly use bounded queues. Playback waits for synchronized media. The assembler first stretches unique forward motion; when narration is longer, it uses forward/reverse coverage rather than freezing the final frame.

In bilingual playback, the domain layer defines one exact sequence: source sentence, translated sentence, then the next pair. The same function supplies TTS and Whisper, while the browser maps the returned word timestamps back to those rendered tokens. My Story follows that identical path; the model may plan visual staging but cannot rewrite pasted narration.

## Development checks

```powershell
D:\AI\ComfyUI\.venv\Scripts\python.exe -m py_compile app.py theater_pipeline.py story_domain.py adapters\*.py test_prompt.py
D:\AI\ComfyUI\.venv\Scripts\python.exe -m unittest -v test_prompt.py
node --check static\js\highlight.js
node --check static\js\theater.js
```

The focused tests do not download models or require a GPU.

## Security

The app and owned services bind to loopback by default. There is no authentication layer; do not expose their ports to an untrusted network. The app forces Hugging Face and Transformers offline mode when it launches ComfyUI.
