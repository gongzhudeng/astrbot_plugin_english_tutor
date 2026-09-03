"""English tutor plugin for AstrBot.

Features:
- English mode with auto detection and cache-friendly prompt injection
- Conversation archiving (English sessions only, per date)
- Notebook: error journal / favorite sentences / vocabulary, with light
  spaced-repetition review scheduling
- Periodic background extraction of learning records (model fallback chain)
- Daily scheduled practice generation (text / rendered card image push)
- WebUI management page (see pages/dashboard/index.html)
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.agent.message import TextPart

from . import web_api
from .audio import TutorAudioManager
from .storage import TutorStore

PLUGIN_NAME = "astrbot_plugin_english_tutor"
PLUGIN_DIR = Path(__file__).parent

# Marker used to locate and strip this plugin's own system-prompt block, so
# toggling the injection or its position never leaves stale content behind.
_SYSTEM_BLOCK_START = "<!-- ENGLISH_TUTOR_PROMPT_BEGIN -->"
_SYSTEM_BLOCK_END = "<!-- /ENGLISH_TUTOR_PROMPT_END -->"
# Run late so other plugins finish mutating system_prompt before we clean up.
_INJECTION_PRIORITY = -10001

WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

DEFAULT_PROMPT_TEMPLATE = (
    "你是一位专业的英语老师。请为我生成今天的英语学习内容。\n"
    "难度定位：{difficulty}\n主题：{topic}\n"
    "内容形式：句子练习或简短情景对话，根据主题选择更合适的一种。\n"
    "要求：表达地道、贴近日常实用场景，略高于我当前水平一点点；"
    "结合我近期的错误和收藏的句子，优先覆盖我还没掌握的表达。"
)

# Appended after the user-editable template; braces stay literal (no .format).
OUTPUT_FORMAT_SPEC = """
【输出格式要求】只输出一个 JSON 对象，禁止输出任何解释或其他文字：
{"type": "sentences 或 dialogue", "items": [...]}
- 句子练习：type 为 "sentences"，items 每项为 {"en": "英文句子", "zh": "中文翻译", "note": "一句话用法提示"}，共 {count} 条。
- 情景对话：type 为 "dialogue"，items 每项为 {"speaker": "说话人名字", "en": "英文台词", "zh": "中文翻译"}，共 4~8 轮。"""

DEFAULT_SENTENCE_FILL_PROMPT = (
    "请为下面这个英语句子生成学习笔记。\n句子：{text}\n"
    "要求：note 为中文翻译 + 一句话用法说明；dialog 为一段包含该句的 3~4 句简短情景对话"
    '（英文，每行一句，以"说话人: 内容"格式分行）；context 为两三句中文描述的使用场景。'
)

DEFAULT_VOCAB_FILL_PROMPT = (
    "请为下面这个英语单词或短语生成学习笔记。\n单词：{text}\n"
    "要求：meaning 为中文释义（可多个词性）；example 为一个地道英文例句；"
    'dialog 为一段使用该词的 3~4 句简短情景对话（英文，每行一句，以"说话人: 内容"格式分行）；'
    "context 为一句话中文说明它的使用场景。"
)

# Fixed machine-readable envelope appended after the editable fill prompts;
# \\n inside keeps literal backslash-n visible to the model for line breaks.
SENTENCE_FILL_SPEC = (
    "\n\n【输出格式要求】只输出一个 JSON 对象，禁止任何其他文字：\n"
    '{"note": "中文翻译+用法说明", "dialog": "说话人: 台词\\n说话人: 台词（3~4 句）", "context": "使用场景"}'
)
VOCAB_FILL_SPEC = (
    "\n\n【输出格式要求】只输出一个 JSON 对象，禁止任何其他文字：\n"
    '{"meaning": "中文释义", "example": "地道英文例句", "dialog": "说话人: 台词\\n说话人: 台词（3~4 句）", "context": "使用场景"}'
)

EXTRACTION_PROMPT = """你是英语学习记录员。下面是一段用户与英语教练的对话，请从中提取学习数据，只输出 JSON。

【对话记录】
{conversation}

【已有未解决错误清单】
{existing}

提取要求：
1. errors：用户的英语错误（教练纠正过的，或明显的语法/用词/搭配错误）。如果某条错误与"已有清单"中的某条实质相同，把清单中该条的原文填入 repeat_of 字段；否则 repeat_of 留空。
2. sentences：对话中值得收藏的地道句子或表达，最多 3 条，宁缺毋滥；没有就给空数组。每条请附 dialog 字段：一段 3~4 句、包含该句的简短情景对话（英文，每行一句，格式"说话人: 台词"），展示它的实际用法。
3. vocab：值得学习的生词或地道短语，最多 5 个；没有就给空数组。
4. 只提取英语学习相关内容，不要记录日常生活事实。

只输出如下格式的 JSON，不要有任何其他文字：
{"errors": [{"original": "用户原句", "corrected": "正确句子", "explanation": "中文简要解释", "category": "错误类型，如 时态/介词/搭配", "repeat_of": ""}],
 "sentences": [{"sentence": "英文句子", "note": "中文说明", "dialog": "Anna: ...\\nBen: ..."}],
 "vocab": [{"word": "单词或短语", "meaning": "中文释义", "example": "英文例句"}]}"""


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def fmt_date_cn(date_str: str) -> str:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return date_str
    return f"{d.year}年{d.month}月{d.day}日"


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first JSON object from an LLM reply."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    payload = text[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        # Models sometimes emit real newlines/tabs inside JSON strings (e.g.
        # multi-line dialog); strict=False tolerates those control characters.
        try:
            data = json.loads(payload, strict=False)
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


@register(
    PLUGIN_NAME,
    "灵犀",
    "AI 英语私教：对话纠错、错误日记、句子收藏、单词本、对话存档、每日练习生成。",
    "0.6.0",
)
class EnglishTutorPlugin(Star):
    """英语私教插件主类。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.store: TutorStore | None = None
        self._modes: dict[str, str] = {}
        self._active: dict[str, bool] = {}
        self._windows: dict[str, deque[bool]] = {}
        self._rounds: dict[str, int] = {}
        self._daily_task: asyncio.Task | None = None
        self._bg_tasks: set[asyncio.Task] = set()
        self.audio_manager: TutorAudioManager | None = None
        self._tutor_block = ""
        self._card_template = ""
        self._stats_template = ""

    # ==================== lifecycle ====================

    async def initialize(self) -> None:
        data_dir = StarTools.get_data_dir(PLUGIN_NAME)
        self.store = TutorStore(data_dir / "tutor.db")
        self.audio_manager = TutorAudioManager(self, self.store, data_dir / "audio")
        # One-shot purge of rows recorded by pre-0.5.1 versions, which archived
        # share links and system-injected prompts as English conversation.
        if not await self.get_kv_data("archive_noise_cleaned", False):
            removed = self.store.cleanup_non_english_archive(self._is_english_message)
            if removed:
                logger.info(
                    f"[english_tutor] purged {removed} non-English archived rows"
                )
            await self.put_kv_data("archive_noise_cleaned", True)
        self._tutor_block = self._build_tutor_block()
        self._card_template = (PLUGIN_DIR / "templates" / "daily_card.html").read_text(
            encoding="utf-8"
        )
        self._stats_template = (PLUGIN_DIR / "templates" / "stats_card.html").read_text(
            encoding="utf-8"
        )
        web_api.register_routes(self)
        self._daily_task = asyncio.create_task(self._daily_loop())
        logger.info("[english_tutor] plugin loaded, data dir: %s", data_dir)

    async def terminate(self) -> None:
        if self._daily_task:
            self._daily_task.cancel()
            try:
                await self._daily_task
            except (asyncio.CancelledError, Exception):
                pass
            self._daily_task = None
        tasks = list(self._bg_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._bg_tasks.clear()
        if self.store:
            self.store.close()
            self.store = None
        self.audio_manager = None
        logger.info("[english_tutor] plugin unloaded")

    # ==================== config helpers ====================

    def _cfg(self, section: str, key: str, default: Any = None) -> Any:
        value = (self.config.get(section) or {}).get(key, default)
        return default if value is None else value

    def _today(self) -> str:
        return today_str()

    def _spawn(self, coro: Any) -> None:
        """Create a background task and keep a reference so it is not GC'd."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _schedule_audio(
        self,
        owner_type: str,
        owner_id: int,
        text: str,
        item_index: int = -1,
    ) -> None:
        manager = getattr(self, "audio_manager", None)
        if not manager or not owner_id or not str(text or "").strip():
            return
        target = {
            "owner_type": owner_type,
            "owner_id": int(owner_id),
            "item_index": int(item_index),
            "text": str(text).strip(),
        }
        if not manager.category_enabled(owner_type):
            return
        self._spawn(manager.ensure(target))

    async def _generate_daily_audio(self, practice: dict[str, Any]) -> None:
        manager = getattr(self, "audio_manager", None)
        if not manager or not manager.category_enabled("practice"):
            return
        daily_id = int(practice.get("id") or 0)
        for index, item in enumerate(practice.get("items") or []):
            text = str((item or {}).get("en") or "").strip()
            if not text or not daily_id:
                continue
            await manager.ensure(
                {
                    "owner_type": "practice",
                    "owner_id": daily_id,
                    "item_index": index,
                    "text": text,
                }
            )

    async def _send_daily_audio(self, destination: str, practice: dict[str, Any]) -> bool:
        manager = getattr(self, "audio_manager", None)
        if not manager or not destination:
            return False
        mode = str(self._cfg("audio", "daily_push_mode", "none") or "none").lower()
        if mode not in {"each", "combined"}:
            return False
        await self._generate_daily_audio(practice)
        paths: list[Path] = []
        daily_id = int(practice.get("id") or 0)
        for index, item in enumerate(practice.get("items") or []):
            text = str((item or {}).get("en") or "").strip()
            if not text or not daily_id:
                continue
            path = manager.asset_path_for_target(
                {
                    "owner_type": "practice",
                    "owner_id": daily_id,
                    "item_index": index,
                    "text": text,
                }
            )
            if path:
                paths.append(path)
        if not paths:
            return False
        if mode == "each":
            for index, path in enumerate(paths, 1):
                await self.context.send_message(
                    destination,
                    MessageChain(
                        chain=[Comp.File(name=f"english_practice_{index}.wav", file=str(path))]
                    ),
                )
            return True
        output = manager.audio_dir / f"daily_{practice.get('date', 'practice')}_combined.wav"
        merged = manager.merge_wav(paths, output)
        if not merged:
            return False
        await self.context.send_message(
            destination,
            MessageChain(
                chain=[Comp.File(name=merged.name, file=str(merged))]
            ),
        )
        return True

    def _build_tutor_block(self) -> str:
        difficulty = self._cfg("tutor", "difficulty", "B1 中级")
        style = self._cfg("tutor", "style", "自然日常")
        return (
            "[英语私教附加任务（不影响你现有人设和说话风格）]\n"
            f"- 全程用英语陪用户对话（难度 {difficulty}，风格：{style}）。\n"
            "- 发现用户英语错误：先用英语自然回应，再用一行中文纠错"
            "（📝 原句 → 正确 + 一句解释），并调用 english_log_error 工具记录。\n"
            "- 用户请你记句子/生词、自评“记住了/忘了”、或说“考我”时，"
            "按需调用 english_save_sentence、english_add_vocab、english_mark_review、"
            "english_quiz_material 或 english_lookup。"
        )

    @staticmethod
    def _remove_system_injection(prompt: str) -> str:
        """Strip this plugin's own previous system block, leaving others intact."""
        pattern = (
            rf"(?:\r?\n)*{re.escape(_SYSTEM_BLOCK_START)}"
            rf".*?{re.escape(_SYSTEM_BLOCK_END)}(?:\r?\n)*"
        )
        return re.sub(pattern, "\n\n", str(prompt or ""), flags=re.DOTALL).rstrip()

    # ==================== english mode detection ====================

    # Noise stripped before language detection: URLs (e.g. Douyin share links),
    # HTML comments and XML/HTML tags (quoted-message / image-context wrappers
    # injected by other plugins into the user message) would otherwise add
    # "English" words and dilute the CJK ratio, making system text pass as
    # English conversation.
    _NOISE_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
    _NOISE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
    _NOISE_TAG = re.compile(r"</?[A-Za-z][^>\n]{0,200}>")
    _ASCII_WORD = re.compile(r"[A-Za-z]{2,}")
    _CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")

    @classmethod
    def _is_english_message(cls, text: str) -> bool:
        """Rule-based check: at least 3 ASCII words and few CJK characters.

        URLs, markup comments/tags and other plugin-injected wrappers are
        stripped first so share links and system text are never mistaken for
        English conversation.

        Args:
            text: The raw message text to inspect.

        Returns:
            True when the text looks like genuine English conversation.
        """
        if not text:
            return False
        text = cls._NOISE_COMMENT.sub(" ", text)
        text = cls._NOISE_URL.sub(" ", text)
        text = cls._NOISE_TAG.sub(" ", text)
        if len(cls._ASCII_WORD.findall(text)) < 3:
            return False
        cjk = len(cls._CJK_CHAR.findall(text))
        return cjk / max(len(text), 1) < 0.3

    async def _get_mode(self, umo: str) -> str:
        if umo not in self._modes:
            self._modes[umo] = await self.get_kv_data(f"mode:{umo}", "auto") or "auto"
        return self._modes[umo]

    async def _set_mode(self, umo: str, mode: str) -> None:
        self._modes[umo] = mode
        self._active[umo] = mode == "on"
        self._windows.pop(umo, None)
        await self.put_kv_data(f"mode:{umo}", mode)

    def _refresh_active(self, umo: str, mode: str, is_english: bool) -> bool:
        """Update the sliding window and effective active state for a session."""
        threshold = float(self._cfg("tutor", "english_ratio_threshold", 0.3))
        window_size = max(3, int(self._cfg("tutor", "window_size", 10)))
        exit_rounds = min(
            max(2, int(self._cfg("tutor", "auto_exit_rounds", 10))), window_size
        )
        win = self._windows.setdefault(umo, deque(maxlen=window_size))
        win.append(is_english)
        active = self._active.get(umo, False)

        if mode == "on":
            active = True
        elif mode == "off":
            active = False
        else:  # auto
            if active:
                recent = list(win)[-exit_rounds:]
                if len(recent) >= exit_rounds and not any(recent):
                    active = False
            if not active and len(win) >= 2:
                if sum(win) / len(win) >= threshold:
                    active = True
        self._active[umo] = active
        return active

    # ==================== llm request/response hooks ====================

    @filter.on_llm_request(priority=_INJECTION_PRIORITY)
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        # Always strip our own stale system block first, even when disabled, so
        # toggling the injection or its position never leaves residue behind.
        req.system_prompt = self._remove_system_injection(req.system_prompt or "")

        umo = event.unified_msg_origin
        mode = await self._get_mode(umo)
        if mode == "off":
            return
        active = self._refresh_active(
            umo, mode, self._is_english_message((event.message_str or "").strip())
        )
        if not active:
            return

        if self._cfg("tutor", "inject_prompt_enabled", False):
            position = self._cfg("tutor", "inject_prompt_mode", "临时用户消息末尾")
            if str(position) == "系统提示词末尾":
                block = (
                    f"{_SYSTEM_BLOCK_START}\n{self._tutor_block}\n{_SYSTEM_BLOCK_END}"
                )
                req.system_prompt = (
                    f"{req.system_prompt}\n\n{block}" if req.system_prompt else block
                )
            else:
                req.extra_user_content_parts.append(
                    TextPart(text=self._tutor_block).mark_as_temp()
                )

        # Per-turn content must stay out of system_prompt or it breaks the
        # provider prefix cache; the temp user part is never persisted.
        if self._cfg("tutor", "inject_context_enabled", False):
            digest = self._build_digest(umo)
            if digest:
                req.extra_user_content_parts.append(
                    TextPart(text=digest).mark_as_temp()
                )

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        umo = event.unified_msg_origin
        if not self._active.get(umo) or not self.store:
            return
        # Skip tool-loop intermediate rounds; keep the final user-visible reply
        # even when it followed tool calls (the tutor calls notebook tools often).
        if resp.role != "assistant" or resp.tools_call_name:
            return
        reply = (resp.completion_text or "").strip()
        if not reply:
            return

        # Only archive genuine English-practice rounds: Chinese chatter and
        # system-injected prompts (scheduled auto-reply instructions, quoted
        # message / image-context wrappers) must stay out of the archive. When
        # only the reply is English, keep it but never the non-English text.
        user_text = (event.message_str or "").strip()
        user_en = bool(user_text) and self._is_english_message(user_text)
        if not user_en and not self._is_english_message(reply):
            return

        date = self._today()
        if user_en:
            self.store.add_archive(umo, date, "user", user_text)
        self.store.add_archive(umo, date, "assistant", reply)

        trigger = max(1, int(self._cfg("extraction", "trigger_rounds", 10)))
        rounds = self._rounds.get(umo, 0) + 1
        if self._cfg("extraction", "enabled", True) and rounds >= trigger:
            self._rounds[umo] = 0
            self._spawn(self._extract_learning_records(umo, trigger))
        else:
            self._rounds[umo] = rounds

    # ==================== context digest ====================

    def _build_digest(self, umo: str) -> str:
        assert self.store
        today = self._today()
        lines = [f"【英语学习档案】今天是 {fmt_date_cn(today)}。"]

        errors = self.store.top_open_errors(
            umo, int(self._cfg("tutor", "inject_errors_top_n", 5))
        )
        if errors:
            lines.append("[用户近期常犯的错误，请在对话中留意并适时提醒]")
            for e in errors:
                category = e.get("category") or "一般错误"
                lines.append(
                    f"- 「{e['original']}」→「{e['corrected']}」"
                    f"（{category}；最近 {e['last_date']}，共 {e['hit_count']} 次）"
                )

        sentences = self.store.recent_sentences(
            umo, int(self._cfg("tutor", "inject_sentences_top_m", 5))
        )
        if sentences:
            lines.append("[用户收藏的句子，可在合适时机带用户活用]")
            for s in sentences:
                note = f"—— {s['note']}" if s.get("note") else ""
                lines.append(f"- {s['sentence']}{note}")

        practice = self.store.get_daily(today)
        if practice:
            lines.append(
                f"[今日练习已生成] 共 {len(practice.get('items') or [])} 条"
                f"（{practice.get('type')}），可在对话中带用户练一练。"
            )

        due_v = len(self.store.due_vocab(umo, today))
        due_e = len(self.store.due_errors(umo, today))
        if due_v or due_e:
            lines.append(
                f"[到期复习] 有 {due_v} 个单词、{due_e} 条错误待复习；"
                "用户说“考我”时优先考察这些。"
            )
        lines.append("（完整记录可用 english_lookup 工具查询。）")
        return "\n".join(lines)

    # ==================== llm fallback chain ====================

    async def _llm_text(
        self, prompt: str, system_prompt: str, umo: str | None = None
    ) -> str | None:
        """Call the configured provider chain in order; return the first non-empty text."""
        ids = [
            str(x).strip()
            for x in (self._cfg("llm", "provider_ids", []) or [])
            if str(x).strip()
        ]
        timeout = max(10, float(self._cfg("llm", "timeout_seconds", 120)))
        retries = max(0, int(self._cfg("llm", "max_retries", 0)))

        providers = []
        for pid in ids:
            provider = self.context.get_provider_by_id(pid)
            if provider:
                providers.append(provider)
            else:
                logger.warning(f"[english_tutor] provider not found, skipped: {pid}")
        if not providers and umo:
            try:
                provider = await self.context.get_using_provider_async(umo)
                if provider:
                    providers.append(provider)
            except Exception as e:
                logger.warning(f"[english_tutor] resolve default provider failed: {e}")
        if not providers:
            logger.warning("[english_tutor] no available provider for LLM call")
            return None

        for provider in providers:
            for attempt in range(retries + 1):
                try:
                    resp = await asyncio.wait_for(
                        provider.text_chat(
                            prompt=prompt,
                            system_prompt=system_prompt or None,
                            request_max_retries=0,
                        ),
                        timeout=timeout,
                    )
                    text = (resp.completion_text or "").strip()
                    if text:
                        return text
                    logger.warning("[english_tutor] provider returned empty text")
                except Exception as e:
                    logger.warning(
                        f"[english_tutor] LLM call failed (attempt {attempt + 1}/{retries + 1}): {e}"
                    )
        return None

    async def ai_fill(self, fill_type: str, text: str) -> dict[str, str]:
        """Generate notebook fields for a sentence/word via the model chain.

        The content prompt is user-configurable (config ``ai_fill`` section);
        a fixed JSON envelope is always appended so parsing stays reliable.

        Args:
            fill_type: "sentence" or "vocab".
            text: The sentence or word/phrase to describe.

        Returns:
            Dict of suggested fields (note/context/dialog or meaning/example/
            context/dialog); missing keys mean the model output was unusable
            and the caller should keep the form as-is.
        """
        text = (text or "").strip()
        if not text:
            return {}
        if fill_type == "sentence":
            template = str(
                self._cfg("ai_fill", "sentence_prompt", DEFAULT_SENTENCE_FILL_PROMPT)
                or DEFAULT_SENTENCE_FILL_PROMPT
            )
            prompt = template.replace("{text}", text) + SENTENCE_FILL_SPEC
            fields = ("note", "context", "dialog")
        else:
            template = str(
                self._cfg("ai_fill", "vocab_prompt", DEFAULT_VOCAB_FILL_PROMPT)
                or DEFAULT_VOCAB_FILL_PROMPT
            )
            prompt = template.replace("{text}", text) + VOCAB_FILL_SPEC
            fields = ("meaning", "example", "context", "dialog")
        raw = await self._llm_text(prompt, "")
        if not raw:
            return {}
        data = extract_json(raw)
        if not data:
            return {}
        return {k: str(data.get(k) or "").strip() for k in fields if data.get(k)}

    # ==================== background extraction ====================

    async def _extract_learning_records(self, umo: str, rounds: int) -> None:
        assert self.store
        messages = self.store.recent_archive(umo, rounds * 2)
        if len(messages) < 2:
            return
        convo = "\n".join(
            f"{'用户' if m['role'] == 'user' else '教练'}: {m['content']}"
            for m in messages
        )
        existing = self.store.top_open_errors(umo, 20)
        existing_text = "\n".join(f"- {e['original']}" for e in existing) or "（无）"
        prompt = EXTRACTION_PROMPT.replace("{conversation}", convo).replace(
            "{existing}", existing_text
        )
        text = await self._llm_text(prompt, "", umo)
        if not text:
            logger.warning("[english_tutor] extraction LLM call failed, skipped")
            return
        data = extract_json(text)
        if not data:
            logger.warning(
                "[english_tutor] extraction returned unparsable content, skipped"
            )
            return

        today = self._today()
        new_errors = merged = new_sentences = new_vocab = 0
        for item in data.get("errors") or []:
            original = str(item.get("original") or "").strip()
            if not original:
                continue
            target = None
            repeat_of = str(item.get("repeat_of") or "").strip()
            if repeat_of:
                target = next(
                    (x for x in existing if x["original"].lower() == repeat_of.lower()),
                    None,
                )
            if target is None:
                target = self.store.find_open_error(umo, original)
            if target:
                self.store.touch_error(target["id"], today)
                merged += 1
            else:
                self.store.add_error(
                    umo,
                    today,
                    original,
                    str(item.get("corrected") or ""),
                    str(item.get("explanation") or ""),
                    str(item.get("category") or ""),
                )
                new_errors += 1
        for item in data.get("sentences") or []:
            sentence = str(item.get("sentence") or "").strip()
            if sentence:
                sentence_id = self.store.add_sentence(
                    umo,
                    today,
                    sentence,
                    str(item.get("note") or ""),
                    "ai",
                    dialog=str(item.get("dialog") or "").strip(),
                )
                self._schedule_audio("sentence", sentence_id, sentence)
                new_sentences += 1
        for item in data.get("vocab") or []:
            word = str(item.get("word") or "").strip()
            if word:
                vocab_id = self.store.add_vocab(
                    umo,
                    today,
                    word,
                    str(item.get("meaning") or ""),
                    str(item.get("example") or ""),
                )
                self._schedule_audio("vocab", vocab_id, word)
                new_vocab += 1
        logger.info(
            f"[english_tutor] extraction done: errors +{new_errors} (merged {merged}),"
            f" sentences +{new_sentences}, vocab +{new_vocab}"
        )

    # ==================== daily generation ====================

    def _build_daily_prompt(self, requirement: str = "") -> str:
        assert self.store
        cfg = self.config.get("daily_gen") or {}
        template = str(cfg.get("prompt_template") or DEFAULT_PROMPT_TEMPLATE)
        difficulty = self._cfg("tutor", "difficulty", "B1 中级")
        topic = str(cfg.get("topic") or "不限")
        count = max(1, int(cfg.get("count") or 5))
        filled = (
            template.replace("{difficulty}", difficulty)
            .replace("{topic}", topic)
            .replace("{count}", str(count))
        )
        extra = str(requirement or "").strip()
        if extra:
            filled += f"\n本次附加要求（优先满足）：{extra}"
        context_lines = []
        for e in self.store.top_open_errors(limit=5):
            corrected = f" → {e['corrected']}" if e.get("corrected") else ""
            context_lines.append(f"- 常犯错误：{e['original']}{corrected}")
        for s in self.store.recent_sentences(limit=5):
            context_lines.append(f"- 收藏句子：{s['sentence']}")
        context = "\n".join(context_lines) or "（暂无）"
        spec = OUTPUT_FORMAT_SPEC.replace("{count}", str(count))
        return f"{filled}\n\n【我的学习档案（供参考）】\n{context}\n{spec}"

    async def _generate_daily(
        self, force: bool = False, requirement: str = ""
    ) -> dict[str, Any] | None:
        assert self.store
        if not force:
            existing = self.store.get_daily(self._today())
            if existing:
                await self._generate_daily_audio(existing)
                return existing
        prompt = self._build_daily_prompt(requirement)
        bound = await self.get_kv_data("bound_umo", "")
        text = await self._llm_text(prompt, "", str(bound) if bound else None)
        if not text:
            return None
        data = extract_json(text)
        ptype = "sentences"
        items = []
        if data:
            if data.get("type") in ("sentences", "dialogue"):
                ptype = data["type"]
            for raw in (data.get("items") or [])[:20]:
                if not isinstance(raw, dict):
                    continue
                en = str(raw.get("en") or "").strip()
                if not en:
                    continue
                items.append(
                    {
                        "en": en,
                        "zh": str(raw.get("zh") or "").strip(),
                        "note": str(raw.get("note") or "").strip(),
                        "speaker": str(raw.get("speaker") or "").strip(),
                    }
                )
        if not items:
            logger.warning("[english_tutor] daily generation returned no valid items")
            return None
        previous = self.store.get_daily(self._today())
        if previous and self.audio_manager:
            self.audio_manager.delete_owner("practice", int(previous["id"]))
        self.store.save_daily(self._today(), str(bound or ""), ptype, items)
        practice = self.store.get_daily(self._today())
        if practice:
            await self._generate_daily_audio(practice)
        return practice

    def _due_line(self, umo: str | None) -> str:
        assert self.store
        today = self._today()
        due_v = len(self.store.due_vocab(umo, today))
        due_e = len(self.store.due_errors(umo, today))
        if not due_v and not due_e:
            return ""
        return f"今日到期复习：{due_v} 个单词、{due_e} 条错误"

    def _format_practice_text(
        self, practice: dict[str, Any], include_translation: bool = True
    ) -> str:
        difficulty = self._cfg("tutor", "difficulty", "B1 中级")
        topic = self._cfg("daily_gen", "topic", "不限")
        title = "情景对话" if practice.get("type") == "dialogue" else "句子练习"
        lines = [
            f"📚 今日英语练习 · {fmt_date_cn(practice['date'])} · {difficulty} · {topic}"
        ]
        lines.append(f"【{title}】")
        for i, item in enumerate(practice.get("items") or [], 1):
            speaker = f"{item['speaker']}：" if item.get("speaker") else ""
            lines.append(f"{i}. {speaker}{item['en']}")
            if include_translation and item.get("zh"):
                extra = f"（{item['note']}）" if item.get("note") else ""
                lines.append(f"   {item['zh']}{extra}")
        due = self._due_line(practice.get("umo") or None)
        if due:
            lines.append(f"💡 {due}")
        return "\n".join(lines)

    def _crop_render_image(self, path: str) -> str:
        """Trim the white margins the t2i renderer adds around the card.

        The renderer captures with a fixed viewport (width varies between
        runs), so the raw image can carry large white borders. Returns the
        cropped PNG path, or the original path when PIL is unavailable.
        """
        try:
            from PIL import Image, ImageChops

            img = Image.open(path).convert("RGB")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            diff = (
                ImageChops.difference(img, bg)
                .convert("L")
                .point(lambda p: 255 if p > 12 else 0)
            )
            bbox = diff.getbbox()
            if not bbox:
                return path
            pad = 2
            box = (
                max(0, bbox[0] - pad),
                max(0, bbox[1] - pad),
                min(img.width, bbox[2] + pad),
                min(img.height, bbox[3] + pad),
            )
            cropped_path = path + ".crop.png"
            img.crop(box).save(cropped_path, "PNG")
            return cropped_path
        except Exception as e:
            logger.warning(f"[english_tutor] crop rendered image failed: {e}")
            return path

    async def _render_practice_image(self, practice: dict[str, Any]) -> str | None:
        try:
            date = datetime.strptime(practice["date"], "%Y-%m-%d")
            data = {
                "date_cn": fmt_date_cn(practice["date"]),
                "weekday": WEEKDAY_CN[date.weekday()],
                "difficulty": self._cfg("tutor", "difficulty", "B1 中级"),
                "topic": self._cfg("daily_gen", "topic", "不限"),
                "practice_type": practice.get("type") or "sentences",
                "items": practice.get("items") or [],
                "include_translation": bool(
                    self._cfg("daily_gen", "image_include_translation", True)
                ),
                "due_line": self._due_line(practice.get("umo") or None),
            }
            path = await self.html_render(
                self._card_template,
                data,
                return_url=False,
                options={"type": "png", "full_page": True},
            )
            return self._crop_render_image(str(path))
        except Exception as e:
            logger.warning(f"[english_tutor] render practice card failed: {e}")
            return None

    def _build_stats_data(self) -> dict[str, Any]:
        """Assemble the learning-stats payload for the stats card image."""
        assert self.store
        overview = self.store.overview()
        today = self._today()
        tiles = [
            {"num": overview["sentences"], "label": "收藏句子"},
            {
                "num": overview["errors"],
                "label": f"错误（{overview['open_errors']} 未解决）",
            },
            {"num": overview["vocab"], "label": "单词"},
            {"num": overview["daily"], "label": "每日练习"},
        ]
        week = self.store.archive_day_counts(7)
        max_week = max((int(w["c"]) for w in week), default=0) or 1
        week_rows = [
            {
                "date": w["date"][5:],
                "count": int(w["c"]),
                "pct": int(int(w["c"]) / max_week * 100),
            }
            for w in week
        ]
        sentences = [
            {"en": s["sentence"], "note": s.get("note") or "", "date": s["date"][5:]}
            for s in self.store.recent_sentences(limit=5)
        ]
        vocab = [
            {
                "word": v["word"],
                "meaning": v.get("meaning") or "",
                "date": v["date"][5:],
            }
            for v in self.store.list_vocab(limit=5)
        ]
        return {
            "date_cn": fmt_date_cn(today),
            "weekday": WEEKDAY_CN[datetime.now().weekday()],
            "difficulty": self._cfg("tutor", "difficulty", "B1 中级"),
            "tiles": tiles,
            "week_rows": week_rows,
            "sentences": sentences,
            "vocab": vocab,
        }

    async def _render_stats_image(self) -> str | None:
        try:
            path = await self.html_render(
                self._stats_template,
                self._build_stats_data(),
                return_url=False,
                options={"type": "png", "full_page": True},
            )
            return self._crop_render_image(str(path))
        except Exception as e:
            logger.warning(f"[english_tutor] render stats card failed: {e}")
            return None

    async def _push_daily(self, practice: dict[str, Any]) -> bool:
        assert self.store
        bound = str(await self.get_kv_data("bound_umo", "") or "")
        if not bound:
            logger.warning(
                "[english_tutor] daily practice generated but no session bound"
            )
            return False
        push_mode = self._cfg("daily_gen", "push_mode", "image")
        sent = False
        if push_mode in ("image", "both"):
            image_path = await self._render_practice_image(practice)
            if image_path:
                await self.context.send_message(
                    bound,
                    MessageChain(chain=[Comp.Image.fromFileSystem(image_path)]),
                )
                sent = True
            elif push_mode == "image":
                logger.warning(
                    "[english_tutor] image render failed, falling back to text"
                )
        if not sent or push_mode == "both":
            text = self._format_practice_text(practice)
            await self.context.send_message(
                bound, MessageChain(chain=[Comp.Plain(text)])
            )
            sent = True
        await self._send_daily_audio(bound, practice)
        self.store.set_daily_status(practice["id"], "pushed")
        return sent

    async def _daily_loop(self) -> None:
        while True:
            try:
                await self._daily_tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[english_tutor] daily loop error: {e}")
            await asyncio.sleep(30)

    async def _daily_tick(self) -> None:
        assert self.store
        if not self._cfg("daily_gen", "enabled", False):
            return
        today = self._today()

        last_cleanup = await self.get_kv_data("last_cleanup_date", "")
        if last_cleanup != today:
            keep_days = int(self._cfg("storage", "archive_retention_days", 90))
            removed = self.store.cleanup_archive(keep_days)
            await self.put_kv_data("last_cleanup_date", today)
            if removed:
                logger.info(f"[english_tutor] cleaned {removed} archived messages")

        time_str = str(self._cfg("daily_gen", "time", "08:00") or "08:00")
        try:
            hh, mm = (int(x) for x in time_str.split(":")[:2])
        except ValueError:
            logger.warning(f"[english_tutor] invalid daily_gen.time: {time_str}")
            return
        now = datetime.now()
        target = now.replace(hour=hh % 24, minute=mm % 60, second=0, microsecond=0)
        if now < target:
            return
        minutes_late = (now - target).total_seconds() / 60
        last = await self.get_kv_data("last_daily_date", "")
        if last == today:
            return
        if minutes_late > 30 and not self._cfg("daily_gen", "catch_up", True):
            return
        fail_ts = float(await self.get_kv_data("last_daily_fail_ts", 0) or 0)
        if fail_ts and (now.timestamp() - fail_ts) < 1800:
            return

        practice = await self._generate_daily()
        if not practice:
            logger.warning(
                "[english_tutor] daily generation failed, will retry in 30 min"
            )
            await self.put_kv_data("last_daily_fail_ts", now.timestamp())
            return
        try:
            await self._push_daily(practice)
        except Exception as e:
            logger.error(f"[english_tutor] daily push failed: {e}")
            await self.put_kv_data("last_daily_fail_ts", now.timestamp())
            return
        await self.put_kv_data("last_daily_date", today)

    # ==================== notebook tool ====================

    def _resolve_date(self, date_str: str) -> str:
        value = (date_str or "").strip().lower()
        if value in ("", "today", "今天"):
            return self._today()
        if value in ("yesterday", "昨天"):
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return value

    def _capture_context(self, event: AstrMessageEvent, limit: int = 4) -> str:
        """Snapshot the recent conversation as the learning context of a record."""
        assert self.store
        umo = event.unified_msg_origin
        messages = self.store.recent_archive(umo, limit)
        lines = [
            f"{'用户' if m['role'] == 'user' else '教练'}: {m['content']}"
            for m in messages
        ]
        current = (event.message_str or "").strip()
        if current:
            lines.append(f"用户: {current}")
        context = "\n".join(lines)
        return context[:600]

    @filter.llm_tool(name="english_quiz_material")
    async def english_quiz_material(self, event: AstrMessageEvent) -> str:
        """用户想接受英语测验时，获取出题材料。"""
        return await self.english_notebook(event, action="quiz")

    @filter.llm_tool(name="english_save_sentence")
    async def english_save_sentence(
        self,
        event: AstrMessageEvent,
        sentence: str,
        note: str = "",
        dialog: str = "",
    ) -> str:
        """收藏一个值得学习的英文句子。

        Args:
            sentence(string): 必填；要收藏的完整英文原句。
            note(string): 可选；中文备注、解释或使用提示。
            dialog(string): 可选但强烈建议；提供 3 至 4 句包含该句的英文情景对话，每行使用“说话人: 台词”格式。
        """
        return await self.english_notebook(
            event,
            action="save_sentence",
            sentence=sentence,
            note=note,
            dialog=dialog,
        )

    @filter.llm_tool(name="english_log_error")
    async def english_log_error(
        self,
        event: AstrMessageEvent,
        sentence: str,
        correction: str,
        note: str = "",
    ) -> str:
        """记录用户的一条英语错误。

        Args:
            sentence(string): 必填；用户说出的错误英文原句。
            correction(string): 必填；修改后的正确英文句子。
            note(string): 可选；中文错误解释或语法备注。
        """
        return await self.english_notebook(
            event,
            action="log_error",
            sentence=sentence,
            correction=correction,
            note=note,
        )

    @filter.llm_tool(name="english_add_vocab")
    async def english_add_vocab(
        self,
        event: AstrMessageEvent,
        sentence: str,
        note: str = "",
    ) -> str:
        """把一个英文单词或短语加入单词本。

        Args:
            sentence(string): 必填；要记录的英文单词或短语。
            note(string): 可选；中文释义、解释或使用提示。
        """
        return await self.english_notebook(
            event,
            action="add_vocab",
            sentence=sentence,
            note=note,
        )

    @filter.llm_tool(name="english_mark_review")
    async def english_mark_review(
        self,
        event: AstrMessageEvent,
        sentence: str,
        kind: str,
        remembered: bool,
    ) -> str:
        """记录一次英语复习结果。

        Args:
            sentence(string): 必填；被考察的单词、短语或错误原句。
            kind(string): 必填；单词使用 vocab，错误原句使用 errors。
            remembered(boolean): 必填；用户记住了传 true，忘了传 false，以更新后续复习间隔。
        """
        return await self.english_notebook(
            event,
            action="mark_review",
            sentence=sentence,
            kind=kind,
            remembered=remembered,
        )

    @filter.llm_tool(name="english_lookup")
    async def english_lookup(
        self,
        event: AstrMessageEvent,
        kind: str,
        date: str = "",
    ) -> str:
        """查询用户的英语学习记录。

        Args:
            kind(string): 必填；使用 errors、sentences、vocab、archive 或 practice。practice 表示每日生成的练习内容。
            date(string): 可选；YYYY-MM-DD、today 或 yesterday。查询 archive 和 practice 时应提供；其他类别留空可查看最近记录。
        """
        return await self.english_notebook(
            event,
            action="lookup",
            kind=kind,
            date=date,
        )

    async def english_notebook(
        self,
        event: AstrMessageEvent,
        action: str = "",
        sentence: str = "",
        correction: str = "",
        note: str = "",
        kind: str = "",
        date: str = "",
        remembered: bool = True,
        dialog: str = "",
    ) -> str:
        """Internal dispatcher shared by the registered English learning tools."""
        assert self.store
        umo = event.unified_msg_origin
        today = self._today()
        action = (action or "").strip().lower()
        sentence = (sentence or "").strip()

        if action == "quiz":
            due_v = self.store.due_vocab(umo, today)[:5]
            due_e = self.store.due_errors(umo, today)[:5]
            recent_s = self.store.recent_sentences(umo, 5)
            lines = ["【出题材料】优先考察以下到期条目："]
            if due_v:
                lines.append("到期单词：")
                lines += [
                    f"- {v['word']}（{v['meaning'] or '无释义'}；例句：{v['example'] or '无'}）"
                    for v in due_v
                ]
            if due_e:
                lines.append("到期错句：")
                lines += [
                    f"- 「{e['original']}」应为「{e['corrected']}」（{e['category'] or '一般错误'}）"
                    for e in due_e
                ]
            if recent_s:
                lines.append("也可考察这些近期收藏的句子：")
                lines += [
                    f"- {s['sentence']}（{s['note'] or '无备注'}）" for s in recent_s
                ]
            if len(lines) == 1:
                lines.append("（暂无记录，可围绕日常话题自由出题。）")
            lines.append(
                "【出题要求】每轮只出一题，用英文出题（考用法、造句或情景应答），"
                "等用户作答后先点评再出下一题；结束后询问用户记住了还是忘了，并用 mark_review 记录。"
            )
            return "\n".join(lines)

        if action == "save_sentence":
            if not sentence:
                return "缺少 sentence 参数，未保存。"
            sentence_id = self.store.add_sentence(
                umo,
                today,
                sentence,
                note.strip(),
                "ai",
                context=self._capture_context(event),
                dialog=(dialog or "").strip(),
            )
            self._schedule_audio("sentence", sentence_id, sentence)
            return f"已收藏句子：{sentence}"

        if action == "log_error":
            if not sentence:
                return "缺少 sentence 参数，未记录。"
            existing = self.store.find_open_error(umo, sentence)
            if existing:
                self.store.touch_error(existing["id"], today)
                return f"该错误已记录过，本次为重复出现（第 {existing['hit_count'] + 1} 次）。"
            self.store.add_error(umo, today, sentence, correction.strip(), note.strip())
            return f"已记录错误：{sentence} → {correction.strip()}"

        if action == "add_vocab":
            if not sentence:
                return "缺少 sentence 参数，未记录。"
            vocab_id = self.store.add_vocab(
                umo,
                today,
                sentence,
                note.strip(),
                context=self._capture_context(event),
            )
            self._schedule_audio("vocab", vocab_id, sentence)
            return f"已收录单词/短语：{sentence}"

        if action == "mark_review":
            k = (kind or "").strip().lower()
            if k == "vocab":
                row = self.store.find_vocab(umo, sentence)
                if not row:
                    return "未找到该单词的记录。"
                self.store.mark_vocab_review(row["id"], remembered, today)
                return "已更新复习状态：" + (
                    "记住了" if remembered else "忘了，稍后会再安排复习"
                )
            if k == "errors":
                row = self.store.find_open_error(umo, sentence)
                if not row:
                    return "未找到该错误的记录。"
                self.store.mark_error_review(row["id"], remembered, today)
                return "已更新复习状态：" + (
                    "记住了" if remembered else "忘了，稍后会再安排复习"
                )
            return "mark_review 需要指定 kind 为 vocab 或 errors。"

        if action == "lookup":
            k = (kind or "sentences").strip().lower()
            if k == "archive":
                target_date = self._resolve_date(date)
                messages = self.store.list_archive(date=target_date, umo=umo, limit=60)
                if not messages:
                    return f"{target_date} 没有英语对话存档。"
                lines = [f"{target_date} 的英语对话存档："]
                lines += [
                    f"{'用户' if m['role'] == 'user' else '教练'}: {m['content']}"
                    for m in messages
                ]
                return "\n".join(lines)
            if k == "practice":
                target_date = self._resolve_date(date)
                practice = self.store.get_daily(target_date)
                if not practice:
                    return f"{target_date} 没有生成过每日练习。"
                return self._format_practice_text(practice)
            if k == "errors":
                rows = self.store.list_errors(umo=umo, limit=10)
                if not rows:
                    return "还没有错误记录。"
                lines = ["最近的错误记录："]
                lines += [
                    f"- 「{r['original']}」→「{r['corrected']}」（{r['category'] or '一般'}，"
                    f"{r['last_date']}，共 {r['hit_count']} 次）"
                    for r in rows
                ]
                return "\n".join(lines)
            if k == "vocab":
                rows = self.store.list_vocab(umo=umo, limit=10)
                if not rows:
                    return "单词本还是空的。"
                lines = ["最近的单词/短语："]
                for r in rows:
                    entry = f"- {r['word']}（{r['meaning'] or '无释义'}，记录于 {r['date']}）"
                    if r.get("example"):
                        entry += f"\n  例句：{r['example']}"
                    if r.get("dialog"):
                        entry += f"\n  示例对话：{r['dialog'][:200]}"
                    if r.get("context"):
                        entry += f"\n  情境：{r['context'][:150]}"
                    lines.append(entry)
                return "\n".join(lines)
            # sentences
            target_date = self._resolve_date(date) if date else None
            rows = self.store.list_sentences(umo=umo, date=target_date, limit=10)
            if not rows:
                return "没有找到收藏的句子。"
            lines = ["收藏的句子："]
            for r in rows:
                entry = f"- {r['sentence']}" + (
                    f"（{r['note']}，{r['date']}）"
                    if r.get("note")
                    else f"（{r['date']}）"
                )
                if r.get("dialog"):
                    entry += f"\n  示例对话：{r['dialog'][:200]}"
                if r.get("context"):
                    entry += f"\n  情境：{r['context'][:150]}"
                lines.append(entry)
            return "\n".join(lines)

        return f"未知操作：{action}。可用操作：save_sentence / log_error / add_vocab / mark_review / lookup。"

    # ==================== commands ====================

    @filter.command("英语开启", alias={"en_on"})
    async def cmd_en_on(self, event: AstrMessageEvent):
        """开启英语私教模式"""
        await self._set_mode(event.unified_msg_origin, "on")
        yield event.plain_result(
            "✅ 英语私教模式已开启：全程英文对话，我会帮你纠错并记录。"
        )

    @filter.command("英语关闭", alias={"en_off"})
    async def cmd_en_off(self, event: AstrMessageEvent):
        """关闭英语私教模式"""
        await self._set_mode(event.unified_msg_origin, "off")
        yield event.plain_result("⏸️ 英语私教模式已关闭，恢复日常聊天。")

    @filter.command("英语自动", alias={"en_auto"})
    async def cmd_en_auto(self, event: AstrMessageEvent):
        """恢复自动检测英文对话"""
        await self._set_mode(event.unified_msg_origin, "auto")
        yield event.plain_result(
            "🤖 已恢复自动检测：检测到英文对话时自动进入私教模式。"
        )

    @filter.command("英语状态", alias={"en_status"})
    async def cmd_en_status(self, event: AstrMessageEvent):
        """查看英语私教状态"""
        assert self.store
        umo = event.unified_msg_origin
        mode = await self._get_mode(umo)
        mode_name = {"auto": "自动检测", "on": "手动开启", "off": "关闭"}.get(
            mode, mode
        )
        active = self._active.get(umo, False)
        overview = self.store.overview()
        due_v = len(self.store.due_vocab(umo, self._today()))
        due_e = len(self.store.due_errors(umo, self._today()))
        bound = await self.get_kv_data("bound_umo", "")
        trigger = int(self._cfg("extraction", "trigger_rounds", 10))
        lines = [
            "📖 英语私教状态",
            f"- 模式：{mode_name}（{'当前对话中' if active else '当前未激活'}）",
            f"- 本轮已积累 {self._rounds.get(umo, 0)}/{trigger} 轮（满后自动提取学习记录）",
            f"- 笔记本：错误 {overview['open_errors']} 条未解决 / 句子 {overview['sentences']} 条 / 单词 {overview['vocab']} 个",
            f"- 到期复习：单词 {due_v} 个、错误 {due_e} 条",
            f"- 对话存档：{overview['archive']} 条",
            f"- 每日推送：{'已绑定' if bound else '未绑定（用 /英语绑定 绑定本会话）'}",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("英语注入", alias={"en_inject"})
    async def cmd_en_inject(self, event: AstrMessageEvent):
        """预览当前会注入的提示词内容"""
        assert self.store
        umo = event.unified_msg_origin
        mode = await self._get_mode(umo)
        mode_name = {"auto": "自动检测", "on": "手动开启", "off": "关闭"}.get(
            mode, mode
        )
        active = self._active.get(umo, False)
        prompt_on = bool(self._cfg("tutor", "inject_prompt_enabled", False))
        prompt_mode = str(self._cfg("tutor", "inject_prompt_mode", "临时用户消息末尾"))
        context_on = bool(self._cfg("tutor", "inject_context_enabled", False))

        lines = ["📥 注入预览"]
        if active:
            lines.append(
                f"- 当前：英语模式激活中（{mode_name}），以下内容正在注入本次对话"
            )
        else:
            lines.append(f"- 当前：{mode_name}，未处于英语模式，此刻不注入任何内容")
            lines.append("- 进入英语模式后才会注入以下内容 ↓")
        lines.append("")

        lines.append(
            f"【1】教练提示词：{'✅ 开 · 注入位置：' + prompt_mode if prompt_on else '❌ 关（不注入）'}"
        )
        lines.append("内容如下：")
        lines.append(self._tutor_block)
        lines.append("")

        lines.append(
            f"【2】学习档案摘要：{'✅ 开 · 注入位置：临时用户消息' if context_on else '❌ 关（不注入）'}"
        )
        lines.append("内容如下（按当前数据实时生成）：")
        lines.append(self._build_digest(umo))
        yield event.plain_result("\n".join(lines))

    @filter.command("英语绑定", alias={"en_bind"})
    async def cmd_en_bind(self, event: AstrMessageEvent):
        """绑定当前会话接收每日练习推送"""
        await self.put_kv_data("bound_umo", event.unified_msg_origin)
        yield event.plain_result("📌 已绑定当前会话，每日英语练习将推送到这里。")

    @filter.command("英语解绑", alias={"en_unbind"})
    async def cmd_en_unbind(self, event: AstrMessageEvent):
        """取消每日练习推送绑定"""
        await self.delete_kv_data("bound_umo")
        yield event.plain_result("✂️ 已取消每日练习推送。")

    @filter.command("英语统计", alias={"en_stats"})
    async def cmd_en_stats(self, event: AstrMessageEvent):
        """查看学习数据统计（图片卡片）"""
        assert self.store
        image_path = await self._render_stats_image()
        if image_path:
            yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
            return
        # 渲染失败时降级为文字版
        overview = self.store.overview()
        lines = [
            "📊 英语学习统计",
            f"- 收藏句子：{overview['sentences']} 条",
            f"- 错误日记：{overview['errors']} 条（未解决 {overview['open_errors']} 条）",
            f"- 单词本：{overview['vocab']} 个",
            f"- 每日练习：已生成 {overview['daily']} 期",
            f"- 对话存档：{overview['archive']} 条",
        ]
        day_counts = self.store.archive_day_counts(7)
        if day_counts:
            lines.append("- 近 7 天英文对话：")
            lines += [f"  · {d['date']}：{d['c']} 条" for d in day_counts]
        recent_s = self.store.recent_sentences(limit=5)
        if recent_s:
            lines.append("- 最近收藏的句子：")
            lines += [
                f"  · {s['sentence']}" + (f"（{s['note']}）" if s.get("note") else "")
                for s in recent_s
            ]
        recent_v = self.store.list_vocab(limit=5)
        if recent_v:
            lines.append("- 最近学的单词：")
            lines += [
                f"  · {v['word']}" + (f"（{v['meaning']}）" if v.get("meaning") else "")
                for v in recent_v
            ]
        yield event.plain_result("\n".join(lines))

    @filter.command("英语测验", alias={"en_quiz"})
    async def cmd_en_quiz(self, event: AstrMessageEvent):
        """让私教现在就考考你"""
        assert self.store
        umo = event.unified_msg_origin
        today = self._today()
        due_v = self.store.due_vocab(umo, today)
        due_e = self.store.due_errors(umo, today)
        parts = []
        if due_v:
            parts.append(
                "到期单词："
                + "；".join(f"{v['word']}（{v['meaning']}）" for v in due_v[:5])
            )
        if due_e:
            parts.append(
                "到期错误："
                + "；".join(f"{e['original']} → {e['corrected']}" for e in due_e[:5])
            )
        material = (
            "\n".join(parts)
            if parts
            else "（暂无到期条目，请从收藏句子和近期错误中挑选）"
        )
        prompt = (
            "（用户请求开始测验）请现在考我英语。可用材料：\n"
            f"{material}\n"
            "要求：每轮只出一题（考用法、造句或情景应答），用英文出题，等我的回答后再点评并出下一题。"
        )
        yield event.request_llm(prompt=prompt)

    @filter.command("英语总结", alias={"en_summarize"})
    async def cmd_en_summarize(self, event: AstrMessageEvent):
        """立即提取最近对话中的学习记录"""
        assert self.store
        umo = event.unified_msg_origin
        trigger = int(self._cfg("extraction", "trigger_rounds", 10))
        self._spawn(self._extract_learning_records(umo, trigger))
        yield event.plain_result(
            "🔄 正在整理最近的对话学习记录，稍后可在 WebUI 中查看。"
        )

    @filter.command("每日英语", alias={"en_daily"})
    async def cmd_en_daily(self, event: AstrMessageEvent, requirement: str = ""):
        """获取今日英语练习（图片卡片），可附带要求，如：/每日英语 商务对话"""
        assert self.store
        practice = self.store.get_daily(self._today())
        if not practice or str(requirement or "").strip():
            yield event.plain_result("✍️ 正在生成今日英语练习，请稍候……")
            practice = await self._generate_daily(
                force=bool(str(requirement or "").strip()), requirement=requirement
            )
            if not practice:
                yield event.plain_result(
                    "❌ 生成失败：所有模型都未能返回有效内容。请稍后重试，或在插件配置中检查模型回退链。"
                )
                return
        image_path = await self._render_practice_image(practice)
        if image_path:
            yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
        else:
            yield event.plain_result(self._format_practice_text(practice))
        await self._send_daily_audio(event.unified_msg_origin, practice)

    @filter.command("英语重写", alias={"en_rewrite"})
    async def cmd_en_rewrite(self, event: AstrMessageEvent, requirement: str = ""):
        """重新生成今日英语练习，可附带要求，如：/英语重写 校园场景 来几个句子"""
        yield event.plain_result(
            "🔁 正在重新生成今日英语练习，请稍候……"
            + (
                f"\n本次要求：{str(requirement).strip()}"
                if str(requirement).strip()
                else ""
            )
        )
        practice = await self._generate_daily(force=True, requirement=requirement)
        if not practice:
            yield event.plain_result(
                "❌ 重新生成失败：所有模型都未能返回有效内容，今日原内容已保留。请稍后重试。"
            )
            return
        image_path = await self._render_practice_image(practice)
        if image_path:
            yield event.chain_result([Comp.Image.fromFileSystem(image_path)])
        else:
            yield event.plain_result(self._format_practice_text(practice))
        await self._send_daily_audio(event.unified_msg_origin, practice)
