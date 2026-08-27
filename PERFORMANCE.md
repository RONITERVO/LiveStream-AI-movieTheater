# Runtime boundaries and performance

The runtime optimizes for sustained local playback and recoverability, not a single synthetic generation score.

## Resource ownership

| Work | Primary resource | Boundary |
| --- | --- | --- |
| Story planning and optional translation | CPU/RAM, optional bounded GPU burst | `StoryRuntime` adapter |
| Speech synthesis | CPU/RAM | `SupertonicRuntime` adapter |
| Video generation | GPU/VRAM | `ComfyAdapter` |
| Word alignment | GPU/VRAM for a short post-render pass | `WhisperAlignmentAdapter` |
| Media synchronization | CPU | FFmpeg assembly in the orchestrator |

Queues remain bounded. ComfyUI submissions are serialized so two Wan jobs cannot compete for a 12 GB GPU. The optional Gemma GPU burst and Whisper alignment both wait on the same GPU boundary, release their model before Wan continues, and fall back to a durable failure message rather than corrupting completed media.

Whisper large-v3-turbo is faster than real time on the reference machine, but it still adds work after TTS and rendering. Alignment is run per completed scene because it produces the exact timings needed by playback and avoids holding either the complete book or a resident ASR model in memory.

## Narration duration

Advanced quality settings define a total spoken-word budget. In bilingual mode, source planning receives a reduced budget so source plus translation remain inside the selected total. The duration controller learns completed-scene cadence, speech seconds per word, and the actual expansion ratio for the selected language pair.

Supertonic pacing stays in the narrow 0.96–1.05 range. Large mismatches are handled visually: unique motion is slowed first, then forward/reverse coverage fills any remaining narration duration in one FFmpeg graph.

## My Story scaling

Pasted source text is normalized and persisted once as UTF-8 `source.txt`. Planning state contains only a byte cursor, total byte count, and completion flag. The source adapter reads a bounded window, selects whole narration units, advances by encoded byte length, and preserves unspaced Japanese text as well as space-delimited languages. This makes memory and request size depend on one scene rather than the size of the book.

## What to measure

Use completed segment timestamps and media duration, not component token rates alone:

- finished playable seconds per wall-clock minute;
- time to first playable scene;
- translated-scene queue depth and GPU feed wait;
- planner, translation, TTS, Wan, Whisper, and assembly elapsed time;
- source, translated, and total spoken words;
- playback coverage and buffer trend;
- live-direction creation, application, and segment completion timestamps.

Any optimization should preserve exact pasted narration, bilingual sentence order, archive durability, and deterministic advanced-quality settings before its throughput result is considered valid.
