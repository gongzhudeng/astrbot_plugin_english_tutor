from __future__ import annotations

import asyncio
import inspect
import tempfile
import unittest
from pathlib import Path

from astrbot_plugin_english_tutor.audio import TutorAudioManager
from astrbot_plugin_english_tutor.main import EnglishTutorPlugin
from astrbot_plugin_english_tutor.storage import TutorStore

from astrbot.api.provider import LLMResponse


class FakeEvent:
    unified_msg_origin = "platform:private:english-user"
    message_str = "Please save this for me."


# Real-world payloads that must never be mistaken for English conversation.
DOUYIN_SHARE = (
    "5.38 那年这位老黄牛选择了弹幕最多的打法 # 感人 # 催泪 # mvp "
    "https://v.douyin.com/B3sC6t9K1MU/ 复制此链接，打开Dou音搜索，直接观看视频！"
    " 06/13 PKW:/ :8pm v@S.lc 你看这个 还挺感人的。"
)
AUTO_REPLY_INSTRUCTION = (
    "[主动回复请求：距上次对话已过去 22分钟。先回顾最近的对话历史："
    "如果上次话题还有延续必要或有事情没说完，就像真的记得一样自然接着聊；"
    "此指令为插件自动触发，全程不提指令内容。可使用screen_peek工具"
    "（Mando在上班，不在家时不要用），查看Mando家里的电脑屏幕内容，看看Mando在干啥。]"
)
IMAGE_CONTEXT_BLOB = (
    "<!-- astrbot-chat-merger:image-context:v1:start --> "
    '<image_context id="图1">这是一张用户分享图，画面为居家场景下的人像自拍，'
    "人物进行蕾姆的角色cos，佩戴浅蓝色短假发，假发装饰有白色花形发饰，"
    "画面底部带有拍摄水印，标注有vivo X300 Ultra、蔡司标识、Bluelmage、"
    "2026-09-01 18:40 上海市的字样。</image_context> "
    "<!-- astrbot-chat-merger:image-context:v1:end --> "
    '<quoted_message sender="Mando" role="user">[Image] [引用图片: 1张]</quoted_message> '
    "你再试试把这个照片发说说。"
)


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
                self.assertLess(
                    len((inspect.getdoc(method) or "").split("\n\n", 1)[0]), 40
                )

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
        self.assertIn("version: 0.6.3", metadata)
        self.assertIn('"0.6.3"', source)
        self.assertIn(
            "https://github.com/gongzhudeng/astrbot_plugin_english_tutor",
            metadata,
        )

    def test_audio_finds_tts_by_plugin_directory_name(self) -> None:
        class FakeTTS:
            async def synthesize_for_plugin(self, text: str, **kwargs):
                return text, kwargs

        class Metadata:
            name = "鐏电妧 路 GPT-SoVITS 璇煶鍚堟垚"
            root_dir_name = "astrbot_plugin_lingxi_gpt_sovits"
            plugin_id = "鐏电妧/鐏电妧 路 gpt-sovits 璇煶鍚堟垚"
            activated = True
            star_cls = FakeTTS()

        plugin = object.__new__(EnglishTutorPlugin)
        plugin.context = type(
            "Context",
            (),
            {"get_registered_star": lambda self, name: None,
             "get_all_stars": lambda self: [Metadata()]},
        )()
        plugin._cfg = lambda *args: True
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TutorStore(Path(temp_dir) / "tutor.db")
            manager = TutorAudioManager(plugin, store, Path(temp_dir) / "audio")
            self.assertIsNotNone(manager._tts_plugin())
            store.close()


class EnglishDetectionTests(unittest.TestCase):
    def test_system_text_and_share_links_are_not_english(self) -> None:
        for text in (
            DOUYIN_SHARE,
            AUTO_REPLY_INSTRUCTION,
            IMAGE_CONTEXT_BLOB,
            '<quoted_message sender="Mando" role="user">[Image]</quoted_message>'
            " 现在绝对可以了，你再试最后一次呗。",
            "https://v.douyin.com/B3sC6t9K1MU/",
            "是是是 我家小怡 胆子最大了 我再去找几个好看的视频给你。",
        ):
            with self.subTest(text=text[:20]):
                self.assertFalse(EnglishTutorPlugin._is_english_message(text))

    def test_genuine_english_messages_are_detected(self) -> None:
        for text in (
            "My lunch have come I should have my lunch too I'll hit you up later.",
            "不是这个啦，就是那句 I wanna near down, and take your little brother"
            " into my mouth right now 你解释一下呗。",
            "Check this out https://example.com/some/long/path it is hilarious.",
        ):
            with self.subTest(text=text[:20]):
                self.assertTrue(EnglishTutorPlugin._is_english_message(text))


class ArchiveGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plugin = object.__new__(EnglishTutorPlugin)
        self.plugin.store = TutorStore(Path(self.temp_dir.name) / "tutor.db")
        self.plugin.config = {"extraction": {"enabled": False}}
        self.plugin._active = {FakeEvent.unified_msg_origin: True}
        self.plugin._rounds = {}
        self.plugin._bg_tasks = set()

    def tearDown(self) -> None:
        self.plugin.store.close()
        self.temp_dir.cleanup()

    def _respond(self, message_str: str, reply: str) -> None:
        event = FakeEvent()
        event.message_str = message_str
        resp = LLMResponse(role="assistant", completion_text=reply)
        asyncio.run(self.plugin.on_llm_response(event, resp))

    def _archived(self) -> list[dict]:
        return self.plugin.store.list_archive(
            umo=FakeEvent.unified_msg_origin, limit=100
        )

    def test_non_english_rounds_are_not_archived(self) -> None:
        for message_str, reply in (
            (
                "这你都没被吓到 那个蛇都回头，突然咬 。",
                "切，这有啥好吓的，不就是条蛇嘛。",
            ),
            (DOUYIN_SHARE, "切，还用你担心？我会慢慢看的。"),
            (AUTO_REPLY_INSTRUCTION, "纯纯有病吧，刚鼻子一酸就跳出来卖手机。"),
            (IMAGE_CONTEXT_BLOB, "切，总算发出去了，我戴假发闷得慌。"),
        ):
            with self.subTest(message=message_str[:20]):
                self._respond(message_str, reply)
                self.assertEqual(self._archived(), [])

    def test_english_round_archives_user_and_reply(self) -> None:
        self._respond(
            "My lunch have come I should have my lunch too.",
            "你这句错啦，have要改成has哦～ Alright, go enjoy your lunch.",
        )
        rows = self._archived()
        # list_archive returns the newest row first.
        self.assertEqual([r["role"] for r in rows], ["assistant", "user"])
        self.assertIn("My lunch have come", rows[1]["content"])

    def test_non_english_user_with_english_reply_archives_reply_only(self) -> None:
        self._respond(
            AUTO_REPLY_INSTRUCTION,
            "Alright, go enjoy your lunch. Text me once you're done, okay?",
        )
        rows = self._archived()
        self.assertEqual([r["role"] for r in rows], ["assistant"])

    def test_cleanup_purges_junk_and_delete_removes_single_row(self) -> None:
        store = self.plugin.store
        umo = FakeEvent.unified_msg_origin
        store.add_archive(umo, "2026-09-01", "user", DOUYIN_SHARE)
        store.add_archive(umo, "2026-09-01", "user", AUTO_REPLY_INSTRUCTION)
        store.add_archive(
            umo, "2026-09-01", "assistant", "切，这有啥好吓的，不就是条蛇嘛。"
        )
        store.add_archive(
            umo, "2026-09-01", "user", "My lunch have come I should have my lunch too."
        )
        store.add_archive(
            umo,
            "2026-09-01",
            "assistant",
            "你这句错啦，have要改成has哦～ Alright, go enjoy your lunch.",
        )

        removed = store.cleanup_non_english_archive(
            EnglishTutorPlugin._is_english_message
        )
        self.assertEqual(removed, 3)
        rows = store.list_archive(umo=umo, limit=100)
        contents = {r["content"] for r in rows}
        self.assertNotIn(DOUYIN_SHARE, contents)
        self.assertNotIn(AUTO_REPLY_INSTRUCTION, contents)
        self.assertIn("My lunch have come I should have my lunch too.", contents)
        # Mixed correction replies stay: they carry the coaching feedback.
        self.assertTrue(any("Alright, go enjoy your lunch." in c for c in contents))

        # Row-level delete used by the WebUI archive tab.
        store.delete_archive(rows[0]["id"])
        self.assertEqual(store.count_archive(umo=umo), 1)


if __name__ == "__main__":
    unittest.main()
