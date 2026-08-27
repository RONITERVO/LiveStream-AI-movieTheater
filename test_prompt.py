import asyncio
import json
import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from adapters.comfy import build_wan_prompt
from adapters.local_models import StoryRuntime, SupertonicRuntime
from adapters.story_source import StorySourceAdapter, StorySourceFinished
from adapters.translation import TranslationAdapter, starter_pair
from adapters.whisper import WhisperAlignmentAdapter
from app import _is_local_exit_request, api_config, create_app
from process_utils import hidden_process_kwargs
from story_domain import (
    AUDIENCES,
    LANGUAGE_NAMES,
    VOICES,
    attention_eta,
    narration_word_limits,
    normalize_story_text,
    planning_language,
    spoken_text,
    spoken_word_count,
    split_narration_sentences,
    translation_language,
    validate_story_request,
)
from theater_pipeline import TheaterError, TheaterManager


class ProductSurfaceTests(unittest.TestCase):
    def test_only_the_three_story_sources_are_accepted(self):
        for mode in ("story", "interactive"):
            self.assertEqual(validate_story_request({"mode": mode, "prompt": "A lighthouse"})["mode"], mode)
        my_story = validate_story_request({"mode": "my_story", "story_text": "A complete opening sentence."})
        self.assertEqual(my_story["mode"], "my_story")
        self.assertEqual(my_story["prompt"], "My story")
        for removed in ("dream", "edutainment", "lesson"):
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                validate_story_request({"mode": removed, "prompt": "No"})

    def test_all_audiences_languages_and_voices_remain_available(self):
        self.assertEqual(AUDIENCES, {"young", "family", "teen", "adult"})
        self.assertEqual(VOICES, {f"{kind}{number}" for kind in ("F", "M") for number in range(1, 6)})
        self.assertEqual(len(LANGUAGE_NAMES), 31)
        self.assertEqual(SupertonicRuntime.LANGUAGES, set(LANGUAGE_NAMES))
        for audience in AUDIENCES:
            config = validate_story_request({
                "prompt": "A river journey",
                "audience": audience,
                "language": "ja",
                "translation_language": "fi",
                "voice": "F5",
            })
            self.assertEqual(config["audience"], audience)

    def test_learning_focus_is_not_part_of_the_contract(self):
        config = validate_story_request({"prompt": "A quiet story", "learning_focus": "astronomy"})
        self.assertNotIn("learning_focus", config)

    def test_every_story_has_a_distinct_translation_language(self):
        english = validate_story_request({"prompt": "A quiet story", "language": "en"})
        finnish = validate_story_request({"prompt": "Hiljainen tarina", "language": "fi"})
        self.assertEqual(english["translation_language"], "fi")
        self.assertEqual(finnish["translation_language"], "en")
        self.assertEqual(translation_language({"language": "en"}), "fi")
        with self.assertRaises(ValueError):
            validate_story_request({
                "prompt": "A quiet story", "language": "en", "translation_language": "en",
            })

    def test_generated_planning_is_canonical_english_but_my_story_is_immutable(self):
        self.assertEqual(planning_language({"mode": "story", "language": "fi"}), "en")
        self.assertEqual(planning_language({"mode": "interactive", "language": "ja"}), "en")
        self.assertEqual(planning_language({"mode": "my_story", "language": "fi"}), "fi")
        manager = TheaterManager.__new__(TheaterManager)
        prompt = manager._system_prompt({"mode": "story", "language": "ja", "audience": "family"})
        self.assertIn("MANDATORY OUTPUT LANGUAGE: English [en]", prompt)
        self.assertNotIn("MANDATORY OUTPUT LANGUAGE: Japanese", prompt)

    def test_advanced_quality_values_are_preserved_and_bounded(self):
        settings = {
            "width": 832, "height": 816, "frames": 77, "fps": 60,
            "min_words": 1200, "max_words": 2400, "max_slow": 20,
        }
        config = validate_story_request({
            "prompt": "A city in rain",
            "quality_settings": settings,
            "context_compaction_scenes": 200,
            "seed": 42,
        })
        self.assertEqual(config["quality_settings"], settings)
        self.assertEqual(config["context_compaction_scenes"], 200)
        with self.assertRaises(ValueError):
            validate_story_request({"prompt": "A city", "quality_settings": {"frames": 80}})

    def test_bilingual_budget_keeps_total_spoken_duration_bounded(self):
        config = validate_story_request({
            "prompt": "A story", "language": "en", "translation_language": "fi",
            "quality_settings": {"min_words": 80, "max_words": 110},
        })
        self.assertEqual(narration_word_limits(config), (39, 52))
        self.assertEqual(spoken_text([
            {"original": "One.", "translation": "Yksi."},
            {"original": "Two.", "translation": "Kaksi."},
        ]), "One. Yksi. Two. Kaksi.")

    def test_sentence_and_word_helpers_are_multilingual(self):
        self.assertEqual(
            split_narration_sentences('She said, "Go now." 次へ進む。最後だ！'),
            ['She said, "Go now."', "次へ進む。", "最後だ！"],
        )
        self.assertGreater(spoken_word_count("静かな庭を歩いて星を見る。", "ja"), 1)

    def test_browser_options_are_owned_by_the_backend(self):
        response = asyncio.run(api_config(None))
        data = json.loads(response.text)
        flags = (Path(__file__).parent / "static" / "flags.svg").read_text(encoding="utf-8")
        self.assertEqual([item["value"] for item in data["modes"]], ["story", "interactive", "my_story"])
        self.assertEqual(len(data["voices"]), 10)
        self.assertEqual(len(data["languages"]), len(LANGUAGE_NAMES))
        self.assertEqual(data["default_translation_language"], "fi")
        self.assertTrue(all(item["starter"] for item in data["languages"]))
        self.assertTrue(all(f'id="{item["flag"]}"' in flags for item in data["languages"]))


class StorySourceTests(unittest.TestCase):
    def test_pre_cleanup_sessions_are_not_silently_migrated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_dir = root / "old"
            session_dir.mkdir()
            (session_dir / "session.json").write_text(json.dumps({
                "id": "old", "version": 3, "status": "stopped",
                "config": {"mode": "story"},
            }), encoding="utf-8")
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            manager.sessions = {}
            manager.steering_events = {}
            manager.load_existing()
            self.assertEqual(manager.sessions, {})

    def test_large_story_is_persisted_once_and_replayed_in_order(self):
        original = " ".join(
            f"Sentence {number} keeps the long pasted book moving in exact order."
            for number in range(1, 401)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = StorySourceAdapter()
            source = adapter.persist(root, original)
            chunks = []
            while True:
                try:
                    chunk, cursor = adapter.next_chunk(
                        root, source, language="en", minimum=30, maximum=45,
                    )
                except StorySourceFinished:
                    break
                chunks.append(chunk)
                source["cursor"] = cursor
            self.assertEqual(" ".join(chunks), normalize_story_text(original))
            self.assertEqual(source["cursor"], source["bytes"])
            self.assertGreater(len(chunks), 50)

    def test_unspaced_japanese_is_chunked_without_loss(self):
        original = "彼は静かな庭を歩いた。空には星が光っていた。" * 80
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = StorySourceAdapter()
            source = adapter.persist(root, original)
            chunks = []
            while True:
                try:
                    chunk, cursor = adapter.next_chunk(
                        root, source, language="ja", minimum=20, maximum=35,
                    )
                except StorySourceFinished:
                    break
                chunks.append(chunk)
                source["cursor"] = cursor
            self.assertEqual(" ".join(chunks), normalize_story_text(original))

    def test_manager_removes_book_text_from_polled_state(self):
        class Writer:
            model_label = "Gemma"
            parallel_slots = 2
            gpu_available = False

        with tempfile.TemporaryDirectory() as directory:
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = Path(directory)
            manager.writer = Writer()
            manager.sessions = {}
            manager.tasks = {}
            manager.steering_events = {}
            manager.story_source = StorySourceAdapter()
            manager._launch = lambda _state: None
            config = validate_story_request({
                "mode": "my_story",
                "story_text": "One complete sentence. Another complete sentence.",
            })
            state = manager.start(config)
            self.assertNotIn("_story_text", state["config"])
            self.assertTrue((manager._dir(state["id"]) / "source.txt").is_file())
            self.assertNotIn("One complete sentence", json.dumps(state))

    def test_story_cursor_is_part_of_speculative_rollback(self):
        state = {
            "story_summary": "before",
            "story_source": {"cursor": 120, "bytes": 900, "complete": False},
            "metrics": {},
        }
        snapshot = TheaterManager._planning_context_snapshot(state)
        state["story_source"]["cursor"] = 450
        TheaterManager._restore_planning_context(state, snapshot)
        self.assertEqual(state["story_source"]["cursor"], 120)

    def test_my_story_bootstrap_never_uses_model_narration(self):
        class Writer:
            profile = "cpu"

            async def complete(self, _messages, max_tokens=0):
                return json.dumps({
                    "title": "Book",
                    "bible": {
                        "protagonists": [{"name": "Ada", "role": "lead", "appearance": "red coat"}],
                        "world": "Town",
                        "visual_style": "cinematic",
                        "premise_contract": ["Follow the book"],
                        "continuity_rules": ["Keep Ada recognizable"],
                    },
                    "story_summary": "Ada begins.",
                    "scene": {
                        "number": 1,
                        "title": "Opening",
                        "beat": "Ada walks",
                        "narration": "MODEL TEXT MUST NOT SURVIVE.",
                        "visual_action": "Ada walks along a road",
                        "camera": "tracking",
                    },
                }), {"tokens_per_second": 1, "elapsed_seconds": 1, "prompt_tokens": 1}

        async def exercise(root):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root
            manager.story_source = StorySourceAdapter()
            manager.writer = Writer()
            session = root / "session"
            (session / "logs").mkdir(parents=True)
            exact = "Ada opens the old gate. She follows the river toward the tower. " * 12
            config = validate_story_request({"mode": "my_story", "story_text": exact})
            raw = config.pop("_story_text")
            state = {
                "id": "session", "config": config,
                "story_source": manager.story_source.persist(session, raw),
                "metrics": {}, "planned": [], "segments": [],
            }
            return await manager._bootstrap(state)

        with tempfile.TemporaryDirectory() as directory:
            result = asyncio.run(exercise(Path(directory)))
        self.assertNotIn("MODEL TEXT", result["scene"]["narration"])
        self.assertTrue(result["scene"]["narration"].startswith("Ada opens the old gate."))


class AdapterAndOrchestrationTests(unittest.TestCase):
    def test_translation_adapter_localizes_both_roles_and_keeps_english_exact(self):
        class Runtime:
            def __init__(self):
                self.messages = None

            async def complete(self, messages, max_tokens=0):
                self.messages = messages
                return json.dumps({
                    "title_source": "Aamu",
                    "title_translation": "MODEL MUST NOT REWRITE INPUT",
                    "sentences": [
                        {"id": 1, "source": "Aurinko nousee.", "translation": "WRONG"},
                        {"id": 2, "source": "Lintu laulaa.", "translation": "WRONG"},
                    ],
                    "learning_pairs": [
                        {"source": "aurinko", "translation": "sun"},
                        {"source": "aurinko", "translation": "sun"},
                        {"source": "", "translation": "empty"},
                    ],
                }), {"tokens_per_second": 2, "elapsed_seconds": 3, "prompt_tokens": 4}

        async def exercise():
            runtime = Runtime()
            adapter = TranslationAdapter(runtime, LANGUAGE_NAMES)
            result = await adapter.localize_scene(
                title="Morning",
                sentences=["The sun rises.", "A bird sings."],
                input_language="en",
                source_language="fi",
                translation_language="en",
            )
            return runtime, result

        runtime, result = asyncio.run(exercise())
        self.assertEqual(result["title"], "Aamu")
        self.assertEqual(result["translated_title"], "Morning")
        self.assertEqual(
            [item["translation"] for item in result["sentences"]],
            ["The sun rises.", "A bird sings."],
        )
        self.assertEqual(result["learning_pairs"][1], {"source": "aurinko", "translation": "sun"})
        self.assertIn("Input is English [en]", runtime.messages[0]["content"])

    def test_my_story_localization_cannot_rewrite_the_pasted_source(self):
        class Runtime:
            async def complete(self, _messages, max_tokens=0):
                return json.dumps({
                    "title_source": "WRONG",
                    "title_translation": "Morning",
                    "sentences": [{
                        "id": 1, "source": "MODEL REWRITE", "translation": "The sun rises.",
                    }],
                    "learning_pairs": [{"source": "aurinko", "translation": "sun"}],
                }), {"tokens_per_second": 2, "elapsed_seconds": 3, "prompt_tokens": 4}

        async def exercise():
            return await TranslationAdapter(Runtime(), LANGUAGE_NAMES).localize_scene(
                title="Aamu",
                sentences=["Aurinko nousee."],
                input_language="fi",
                source_language="fi",
                translation_language="en",
            )

        result = asyncio.run(exercise())
        self.assertEqual(result["title"], "Aamu")
        self.assertEqual(result["sentences"][0]["original"], "Aurinko nousee.")
        self.assertEqual(result["sentences"][0]["translation"], "The sun rises.")

    def test_attention_contract_uses_next_scene_pairs_and_measured_eta(self):
        state = {
            "status": "generating", "current_scene": 2, "rendering_scene": 2,
            "config": {"language": "en", "translation_language": "fi"},
            "planned": [{"number": 2, "learning_pairs": [{"source": "river", "translation": "joki"}]}],
            "segments": [{"number": 1}],
            "metrics": {
                "video_seconds_ema": 100, "video_started_at": 70,
                "tts_seconds_ema": 20, "tts_started_at": 70,
                "alignment_seconds_ema": 5, "assembly_seconds_ema": 5,
                "completion_interval_ema": 125,
            },
        }
        self.assertEqual(attention_eta(state, 100), 80)
        manager = TheaterManager.__new__(TheaterManager)
        with patch("adapters.attention.time.time", return_value=100):
            public = manager.public_state(state)
        self.assertEqual(public["attention"]["pairs"], [{"source": "river", "translation": "joki"}])
        self.assertEqual(public["attention"]["eta_seconds"], 80)
        self.assertEqual(public["attention"]["average_scene_seconds"], 125)
        self.assertEqual(starter_pair("en", "fi"), {"source": "story", "translation": "tarina"})

    def test_windows_child_processes_are_fully_hidden(self):
        kwargs = hidden_process_kwargs()
        if os.name == "nt":
            self.assertTrue(kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW)
            self.assertTrue(kwargs["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW)
            self.assertEqual(kwargs["startupinfo"].wShowWindow, subprocess.SW_HIDE)
        else:
            self.assertEqual(kwargs, {})

    def test_wan_graph_keeps_the_validated_advanced_inputs(self):
        graph = build_wan_prompt({
            "prompt": "cinematic rain", "negative": "text", "width": 480, "height": 272,
            "frames": 81, "fps": 16, "seed": 9, "filename_prefix": "test/scene",
        })
        self.assertEqual(graph["4"]["inputs"]["length"], 81)
        self.assertEqual(graph["8"]["inputs"]["noise_seed"], 9)
        self.assertEqual(graph["16"]["class_type"], "SaveVideo")

    def test_gemma_adapter_has_one_supported_model_and_bounded_profiles(self):
        runtime = StoryRuntime(Path("."), Path("models"), Path("runtime"), Path("cuda"))
        self.assertEqual(runtime.model.name, "gemma-4-E4B-it-Q4_K_M.gguf")
        self.assertEqual(runtime.parallel_slots, 2)
        self.assertEqual(runtime._server_args(Path("llama-server.exe"), "cpu")[8], "8083")

    def test_bilingual_speech_keeps_language_specific_alignment_parts(self):
        async def exercise(root):
            runtime = SupertonicRuntime(root, root)

            async def synthesize(_text, output, **_kwargs):
                output.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(output), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16_000)
                    audio.writeframes(b"\0\0" * 1_600)
                return 0.01

            runtime.synthesize = synthesize
            output = root / "scene.wav"
            await runtime.synthesize_alternating(
                [{"original": "Hello.", "translation": "Hei."}], output,
                voice="M1", original_language="en", translation_language="fi",
            )
            adapter = WhisperAlignmentAdapter(root)
            part_dir, parts = adapter._alignment_parts(output)
            return output, part_dir, parts

        with tempfile.TemporaryDirectory() as directory:
            output, part_dir, parts = asyncio.run(exercise(Path(directory)))
            self.assertTrue(output.is_file())
            self.assertTrue(part_dir.is_dir())
            self.assertEqual([part["language"] for part in parts], ["en", "fi"])
            self.assertEqual([part["text"] for part in parts], ["Hello.", "Hei."])
            self.assertAlmostEqual(parts[1]["start"], 0.1, places=3)

    def test_visual_sync_uses_slow_motion_then_forward_reverse_coverage(self):
        slow, duration, repeated = TheaterManager._visual_sync_filter(
            5.0, 4.0, {"fps": 16, "frames": 81, "max_slow": 8},
        )
        self.assertFalse(repeated)
        self.assertIn("minterpolate", slow)
        repeated_graph, _, repeated = TheaterManager._visual_sync_filter(
            2.0, 20.0, {"fps": 16, "frames": 81, "max_slow": 4},
        )
        self.assertTrue(repeated)
        self.assertIn("reverse", repeated_graph)

    def test_render_aligns_the_exact_bilingual_playback_sequence(self):
        class Controller:
            def __init__(self):
                self.workflow_lock = asyncio.Lock()
                self.freed = 0

            async def submit(self, _prompt, _client_id):
                return {"prompt_id": "p"}

            async def job(self, _prompt_id):
                return {"state": "complete", "files": [{"path": "raw.mp4", "filename": "raw.mp4"}]}

            async def free_models(self):
                self.freed += 1

        class Speech:
            async def synthesize_alternating(self, _pairs, output, **_kwargs):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"RIFF")
                return 0.1

        class Whisper:
            def __init__(self):
                self.expected = None

            async def align(self, _audio, expected, language):
                self.expected = (expected, language)
                return {
                    "model": "whisper:large-v3-turbo", "segments": [],
                    "words": [{"value": "Hello", "start": 0, "end": .3}],
                    "elapsed_seconds": .2,
                }

        async def exercise(root):
            manager = TheaterManager.__new__(TheaterManager)
            manager.root = root / "wan_theater"
            manager.output_root = root
            manager.controller = Controller()
            manager.supertonic = Speech()
            manager.whisper = Whisper()
            manager.video_prompt_builder = lambda value: value
            manager._save = lambda _state: None
            (manager.root / "s" / "audio").mkdir(parents=True)
            state = {
                "id": "s", "metrics": {},
                "config": validate_story_request({
                    "prompt": "Story", "language": "en", "translation_language": "fi",
                }),
                "bible": {
                    "visual_style": "film", "world": "room", "protagonists": ["Ada"],
                    "premise_contract": [], "continuity_rules": [],
                },
            }
            scene = {
                "number": 1, "title": "One", "beat": "Greeting", "narration": "Hello.",
                "visual_action": "Ada waves", "camera": "medium",
                "narration_sentences": [{"original": "Hello.", "translation": "Hei."}],
                "total_spoken_words": 2,
            }
            work = await manager._render_scene(state, scene)
            return manager, work

        with tempfile.TemporaryDirectory() as directory:
            manager, work = asyncio.run(exercise(Path(directory)))
        self.assertEqual(manager.whisper.expected, ("Hello. Hei.", "auto"))
        self.assertEqual(work["alignment"]["model"], WhisperAlignmentAdapter.MODEL_ID)
        self.assertEqual(manager.controller.freed, 1)

    def test_kestrel_sequence_alignment_keeps_later_words_anchored(self):
        script = r"""
const fs = require('fs');
global.window = {};
eval(fs.readFileSync('static/js/highlight.js', 'utf8'));
const timings = ['oxygen','concentration','nineteen','point','eight','percent','is','safe','power','recovers']
  .map((value, index) => ({value, start:index, end:index+1}));
process.stdout.write(JSON.stringify(window.TheaterHighlight.alignTimings(
  'O concentration 19.8 percent is safe. Power recovers.', timings
)));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=Path(__file__).parent,
            check=True, capture_output=True, text=True,
        )
        mapped = json.loads(result.stdout)
        self.assertEqual(mapped[1], 1)
        self.assertEqual(mapped[5], 4)
        self.assertEqual(mapped[-1], 8)
        self.assertTrue(all(right >= left for left, right in zip(mapped, mapped[1:])))

    def test_interlude_helpers_keep_one_small_valid_card_contract(self):
        script = r"""
const fs = require('fs');
global.window = {};
eval(fs.readFileSync('static/js/interlude.js', 'utf8'));
const element = () => ({hidden:false, textContent:'', classList:{add(){}, remove(){}}});
const elements = {
  shell:element(), window:element(), transcript:element(), root:element(), pair:element(),
  source:element(), translation:element(), meta:element()
};
const learner = new window.LearningInterlude(elements);
learner.show({pairs:[{source:'river', translation:'joki'}], detail:'Creating visual 2', eta_seconds:125});
learner.hide(true);
process.stdout.write(JSON.stringify({
  duration: window.TheaterInterlude.formatDuration(125),
  dwell: window.TheaterInterlude.pairDwell({source:'river', translation:'joki'}),
  interrupted: !learner.active && elements.root.hidden && !elements.window.hidden,
  pairs: window.TheaterInterlude.validPairs([
    {source:'river', translation:'joki'},
    {source:'', translation:'empty'},
    null,
  ])
}));
"""
        result = subprocess.run(
            ["node", "-e", script], cwd=Path(__file__).parent,
            check=True, capture_output=True, text=True,
        )
        value = json.loads(result.stdout)
        self.assertEqual(value["duration"], "2:05")
        self.assertGreaterEqual(value["dwell"], 2600)
        self.assertTrue(value["interrupted"])
        self.assertEqual(value["pairs"], [{"source": "river", "translation": "joki"}])


class AppBoundaryTests(unittest.TestCase):
    def test_exit_requires_loopback_and_explicit_header(self):
        class Request:
            def __init__(self, remote, header):
                self.remote = remote
                self.headers = {"X-Wan-Local-Exit": header} if header else {}

        self.assertTrue(_is_local_exit_request(Request("127.0.0.1", "release-owned-resources")))
        self.assertTrue(_is_local_exit_request(Request("::1", "release-owned-resources")))
        self.assertFalse(_is_local_exit_request(Request("127.0.0.1", "")))
        self.assertFalse(_is_local_exit_request(Request("192.168.1.20", "release-owned-resources")))

    def test_routes_keep_only_the_focused_theater_api(self):
        routes = {route.resource.canonical for route in create_app().router.routes()}
        self.assertIn("/api/theater", routes)
        self.assertIn("/api/theater/{session_id}/directives", routes)
        self.assertIn("/api/shutdown", routes)
        self.assertNotIn("/api/learning", routes)

    def test_ui_has_no_removed_product_modes_or_learning_focus(self):
        html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (Path(__file__).parent / "static" / "js" / "theater.js").read_text(encoding="utf-8")
        interlude = (Path(__file__).parent / "static" / "js" / "interlude.js").read_text(encoding="utf-8")
        combined = (html + javascript).lower()
        for removed in ("learning focus", "educational adventure", "story-led lesson", "endless dream"):
            self.assertNotIn(removed, combined)
        self.assertIn("my story", combined)
        self.assertIn("advanced quality", combined)
        self.assertNotIn("speak story language only", combined)
        self.assertIn("quickLanguageSelect", html)
        self.assertIn("quickTranslationSelect", html)
        self.assertIn("learningInterlude", html)
        self.assertNotIn("spinner", html)
        self.assertNotIn("playerWaiting", html)
        self.assertIn("window.LearningInterlude", interlude)
        self.assertIn("response.status === 404", javascript)
        self.assertIn("localStorage.removeItem('wanTheaterSession')", javascript)


if __name__ == "__main__":
    unittest.main()
    planning_language,
