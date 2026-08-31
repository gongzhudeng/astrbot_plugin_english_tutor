from __future__ import annotations

import asyncio
import inspect
import tempfile
import unittest
from pathlib import Path

from astrbot_plugin_english_tutor.main import EnglishTutorPlugin
from astrbot_plugin_english_tutor.storage import TutorStore


class FakeEvent:
    unified_msg_origin = "platform:private:english-user"
    message_str = "Please save this for me."


class EnglishTutorToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plugin = object.__new__(EnglishTutorPlugin)
        self.plugin.store = TutorStore(Path(self.temp_dir.name) / "tutor.db")
        self.event = FakeEvent()

    def tearDown(self) -> None:
        self.plugin.store.close()
        self.temp_dir.cleanup()

    def test_tool_signatures_are_split_by_action(self) -> None:
        expected = {
            "english_quiz_material": {"self", "event"},
            "english_save_sentence": {"self", "event", "sentence", "note", "dialog"},
            "english_log_error": {"self", "event", "sentence", "correction", "note"},
            "english_add_vocab": {"self", "event", "sentence", "note"},
            "english_mark_review": {
                "self",
                "event",
                "sentence",
                "kind",
                "remembered",
            },
            "english_lookup": {"self", "event", "kind", "date"},
        }
        for name, parameters in expected.items():
            with self.subTest(name=name):
                method = getattr(EnglishTutorPlugin, name)
                self.assertEqual(set(inspect.signature(method).parameters), parameters)
                self.assertLess(len((inspect.getdoc(method) or "").split("\n\n", 1)[0]), 40)

        source = Path(__file__).with_name("main.py").read_text(encoding="utf-8")
        self.assertNotIn('@filter.llm_tool(name="english_notebook")', source)

    def test_split_tools_reuse_storage_workflow(self) -> None:
        saved = asyncio.run(
            self.plugin.english_save_sentence(
                self.event,
                "That makes sense.",
                "这说得通",
                "A: Is the plan clear?\nB: Yes, that makes sense.",
            )
        )
        self.assertIn("已收藏", saved)

        vocab = asyncio.run(
            self.plugin.english_add_vocab(self.event, "concise", "简洁的")
        )
        self.assertIn("已收录", vocab)

        error = asyncio.run(
            self.plugin.english_log_error(
                self.event,
                "He go to school.",
                "He goes to school.",
                "第三人称单数",
            )
        )
        self.assertIn("已记录错误", error)

        lookup = asyncio.run(self.plugin.english_lookup(self.event, "errors"))
        self.assertIn("He goes to school.", lookup)

        reviewed = asyncio.run(
            self.plugin.english_mark_review(
                self.event,
                "concise",
                "vocab",
                True,
            )
        )
        self.assertIn("记住了", reviewed)

        quiz = asyncio.run(self.plugin.english_quiz_material(self.event))
        self.assertIn("【出题材料】", quiz)

    def test_version_and_repository_metadata(self) -> None:
        metadata = Path(__file__).with_name("metadata.yaml").read_text(encoding="utf-8")
        source = Path(__file__).with_name("main.py").read_text(encoding="utf-8")
        self.assertIn("version: 0.5.0", metadata)
        self.assertIn('"0.5.0"', source)
        self.assertIn(
            "https://github.com/gongzhudeng/astrbot_plugin_english_tutor",
            metadata,
        )


if __name__ == "__main__":
    unittest.main()
