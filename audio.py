"""Audio asset management and GPT-SoVITS integration for English tutor."""

from __future__ import annotations

import asyncio
import io
import mimetypes
import secrets
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from astrbot.api import logger

from .storage import TutorStore

if TYPE_CHECKING:
    from .main import EnglishTutorPlugin


PLUGIN_ROUTE = "/api/plug/astrbot_plugin_english_tutor"
TTS_PLUGIN_NAMES = (
    "astrbot_plugin_lingxi_gpt_sovits",
    "astrbot_plugin_GPT_SoVITS",
)


@dataclass(frozen=True)
class _MediaTicket:
    path: Path
    mime_type: str
    expires_at: float


class _TutorMediaGateway:
    """Serve ticket-bound tutor audio outside Dashboard authentication."""

    def __init__(self, ttl_seconds: int = 30 * 60) -> None:
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._tickets: dict[str, _MediaTicket] = {}
        self._lock = asyncio.Lock()
        self._runner: web.AppRunner | None = None
        self._port: int | None = None
        self._closed = False

    async def issue_url(self, path: Path, mime_type: str = "") -> str:
        resource = path.resolve(strict=True)
        if not resource.is_file():
            raise FileNotFoundError(resource)
        async with self._lock:
            if self._closed:
                raise RuntimeError("Tutor media gateway is closed")
            await self._ensure_started_locked()
            self._remove_expired_locked()
            ticket = secrets.token_urlsafe(32)
            self._tickets[ticket] = _MediaTicket(
                resource,
                mime_type
                or mimetypes.guess_type(resource.name)[0]
                or "application/octet-stream",
                time.monotonic() + self._ttl_seconds,
            )
            port = self._port
        if port is None:
            raise RuntimeError("Tutor media gateway did not expose a port")
        return f"http://127.0.0.1:{port}/media/{ticket}"

    async def stop(self) -> None:
        async with self._lock:
            self._closed = True
            self._tickets.clear()
            runner, self._runner = self._runner, None
            self._port = None
        if runner is not None:
            await runner.cleanup()

    async def _ensure_started_locked(self) -> None:
        if self._runner is not None and self._port is not None:
            return
        app = web.Application()
        app.router.add_get("/media/{ticket}", self._serve_media)
        runner = web.AppRunner(app, access_log=None)
        try:
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            server = site._server
            sockets = server.sockets if server is not None else ()
            if not sockets:
                raise RuntimeError("Tutor media gateway did not expose a socket")
        except BaseException:
            await runner.cleanup()
            raise
        self._runner = runner
        self._port = int(sockets[0].getsockname()[1])

    async def _serve_media(self, request: web.Request) -> web.StreamResponse:
        async with self._lock:
            self._remove_expired_locked()
            ticket = self._tickets.get(request.match_info.get("ticket", ""))
        if ticket is None or not ticket.path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(
            ticket.path,
            headers={
                "Cache-Control": "no-store",
                "Content-Type": ticket.mime_type,
                "Cross-Origin-Resource-Policy": "cross-origin",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    def _remove_expired_locked(self) -> None:
        now = time.monotonic()
        for ticket_id, ticket in list(self._tickets.items()):
            if ticket.expires_at <= now:
                self._tickets.pop(ticket_id, None)


class TutorAudioManager:
    """Keep tutor-owned WAV files separate from the TTS plugin cache."""

    def __init__(
        self,
        plugin: EnglishTutorPlugin,
        store: TutorStore,
        audio_dir: Path,
    ) -> None:
        self.plugin = plugin
        self.store = store
        self.audio_dir = audio_dir.resolve()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[tuple[str, int, int], asyncio.Lock] = {}
        self.media_gateway = _TutorMediaGateway()

    async def close(self) -> None:
        await self.media_gateway.stop()

    async def media_url(self, asset_id: int) -> str | None:
        row = self.store.get_audio_asset_by_id(int(asset_id))
        path = self.path_for_asset(row)
        if path is None:
            return None
        return await self.media_gateway.issue_url(path, "audio/wav")

    def _lock_for(self, target: dict[str, Any]) -> asyncio.Lock:
        key = (
            str(target["owner_type"]),
            int(target["owner_id"]),
            int(target.get("item_index", -1)),
        )
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _cfg(self, key: str, default: Any = None) -> Any:
        return self.plugin._cfg("audio", key, default)

    def category_enabled(self, owner_type: str) -> bool:
        if not bool(self._cfg("enabled", True)):
            return False
        key = {
            "sentence": "sentences_enabled",
            "vocab": "vocab_enabled",
            "practice": "daily_enabled",
        }.get(owner_type)
        return bool(self._cfg(key, True)) if key else False

    def default_settings(self) -> dict[str, str]:
        mode = str(self._cfg("emotion_mode", "default") or "default").strip().lower()
        if mode not in {"default", "auto", "specified"}:
            mode = "default"
        return {
            "emotion_mode": mode,
            "emotion": str(self._cfg("emotion", "") or "").strip(),
            "role": str(self._cfg("role", "") or "").strip(),
        }

    def _tts_plugin(self) -> Any | None:
        context = getattr(self.plugin, "context", None)
        getter = getattr(context, "get_registered_star", None)
        candidates: list[Any] = []
        if callable(getter):
            for name in TTS_PLUGIN_NAMES:
                try:
                    metadata = getter(name)
                except Exception:
                    continue
                if metadata is not None:
                    candidates.append(metadata)

        # AstrBot's lookup is exact against metadata.name, while plugin
        # directories and legacy manifests commonly use a different name.
        # Fall back to the complete registry so integrations survive either
        # naming scheme.
        all_stars = getattr(context, "get_all_stars", None)
        if callable(all_stars):
            try:
                candidates.extend(all_stars() or [])
            except Exception:
                pass

        wanted = {name.casefold() for name in TTS_PLUGIN_NAMES}
        for metadata in candidates:
            values = {
                str(getattr(metadata, key, "") or "").strip().casefold()
                for key in ("name", "root_dir_name", "module_path", "plugin_id")
            }
            if not values.intersection(wanted):
                continue
            if getattr(metadata, "activated", True) is False:
                continue
            instance = getattr(metadata, "star_cls", None)
            if callable(getattr(instance, "synthesize_for_plugin", None)):
                return instance
        return None

    def options(self) -> dict[str, Any]:
        tts = self._tts_plugin()
        if tts is None:
            return {
                "available": False,
                "enabled": False,
                "roles": [],
                "emotions": [],
                "active_role": self.default_settings()["role"],
            }
        getter = getattr(tts, "get_integration_options", None)
        if callable(getter):
            try:
                data = dict(getter())
            except Exception as exc:
                logger.warning("读取语音插件选项失败: %s", exc)
                data = {}
        else:
            data = {}
        settings = self.default_settings()
        roles = data.get("roles") or []
        emotions = data.get("emotions") or []
        return {
            "available": callable(getattr(tts, "synthesize_for_plugin", None)),
            "enabled": bool(data.get("enabled", False)),
            "roles": [str(item) for item in roles],
            "emotions": [str(item) for item in emotions],
            "active_role": str(data.get("active_role") or settings["role"]),
            "settings": settings,
        }

    def _path_from_row(self, row: dict[str, Any]) -> Path | None:
        file_name = str(row.get("file_name") or "").strip()
        if not file_name:
            return None
        path = (self.audio_dir / file_name).resolve()
        try:
            path.relative_to(self.audio_dir)
        except ValueError:
            return None
        return path

    def path_for_asset(self, row: dict[str, Any] | None) -> Path | None:
        if not row:
            return None
        path = self._path_from_row(row)
        return path if path and path.is_file() else None

    def _payload(
        self,
        row: dict[str, Any] | None,
        *,
        expected_text: str = "",
    ) -> dict[str, Any] | None:
        if not row or (expected_text and str(row.get("text") or "") != expected_text):
            return None
        if self.path_for_asset(row) is None:
            return None
        return {
            "id": int(row["id"]),
            "status": str(row.get("status") or "current"),
            "text": str(row.get("text") or ""),
            "url": f"{PLUGIN_ROUTE}/audio/file/{int(row['id'])}",
            "emotion_mode": str(row.get("emotion_mode") or "default"),
            "emotion": str(row.get("emotion") or ""),
            "role": str(row.get("role") or ""),
            "created_at": str(row.get("created_at") or ""),
        }

    def attach(
        self,
        item: dict[str, Any],
        owner_type: str,
        owner_id: int,
        item_index: int = -1,
    ) -> dict[str, Any]:
        text = str(item.get("sentence" if owner_type == "sentence" else "word") or "")
        if owner_type == "practice":
            text = str(item.get("en") or "")
        current = self.store.get_audio_asset(
            owner_type, owner_id, item_index, "current"
        )
        candidate = self.store.get_audio_asset(
            owner_type, owner_id, item_index, "candidate"
        )
        item["audio"] = self._payload(current, expected_text=text)
        item["audio_candidate"] = self._payload(candidate, expected_text=text)
        return item

    def target(
        self,
        kind: str,
        item_id: int,
        item_index: int = -1,
    ) -> dict[str, Any] | None:
        kind = str(kind or "").strip().lower()
        if kind in {"sentences", "sentence"}:
            row = self.store.get_sentence(item_id)
            if not row:
                return None
            return {
                "owner_type": "sentence",
                "owner_id": int(row["id"]),
                "item_index": -1,
                "text": str(row.get("sentence") or "").strip(),
            }
        if kind in {"vocab", "word"}:
            row = self.store.get_vocab(item_id)
            if not row:
                return None
            return {
                "owner_type": "vocab",
                "owner_id": int(row["id"]),
                "item_index": -1,
                "text": str(row.get("word") or "").strip(),
            }
        if kind in {"practice", "daily"}:
            practice = self.store.get_daily_by_id(item_id)
            if not practice:
                return None
            items = practice.get("items") or []
            if item_index < 0 or item_index >= len(items):
                return None
            text = str((items[item_index] or {}).get("en") or "").strip()
            if not text:
                return None
            return {
                "owner_type": "practice",
                "owner_id": int(practice["id"]),
                "item_index": item_index,
                "text": text,
            }
        return None

    async def _call_tts(
        self,
        text: str,
        settings: dict[str, str],
    ) -> tuple[bytes | None, str]:
        tts = self._tts_plugin()
        method = getattr(tts, "synthesize_for_plugin", None) if tts else None
        if not callable(method):
            return None, "语音合成插件未安装或未提供集成接口"
        try:
            result = await method(
                text,
                emotion_mode=settings["emotion_mode"],
                emotion=settings["emotion"],
                role=settings["role"],
                umo="",
                force_wav=True,
            )
        except Exception as exc:
            return None, str(exc)
        data = getattr(result, "data", None)
        if not bool(getattr(result, "ok", False)) or not data:
            return None, str(getattr(result, "error", "语音合成失败") or "语音合成失败")
        return bytes(data), ""

    def _remove_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            path = self._path_from_row(row)
            if path:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("删除英语私教音频失败 %s: %s", path, exc)

    async def _synthesize(
        self,
        target: dict[str, Any],
        *,
        settings: dict[str, str],
        candidate: bool,
    ) -> tuple[dict[str, Any] | None, str]:
        owner_type = str(target["owner_type"])
        owner_id = int(target["owner_id"])
        item_index = int(target.get("item_index", -1))
        async with self._lock_for(target):
            if candidate:
                self._remove_rows(
                    self.store.delete_audio_assets(
                        owner_type, owner_id, item_index, status="candidate"
                    )
                )
            data, error = await self._call_tts(str(target["text"]), settings)
            if not data:
                return None, error
            file_name = f"english_tutor_{uuid.uuid4().hex}.wav"
            path = self.audio_dir / file_name
            try:
                path.write_bytes(data)
                if not candidate:
                    self._remove_rows(
                        self.store.delete_audio_assets(
                            owner_type, owner_id, item_index, status="current"
                        )
                    )
                    self._remove_rows(
                        self.store.delete_audio_assets(
                            owner_type, owner_id, item_index, status="candidate"
                        )
                    )
                self.store.add_audio_asset(
                    owner_type,
                    owner_id,
                    item_index,
                    str(target["text"]),
                    file_name,
                    "candidate" if candidate else "current",
                    settings["emotion_mode"],
                    settings["emotion"],
                    settings["role"],
                )
            except Exception as exc:
                path.unlink(missing_ok=True)
                return None, str(exc)
            row = self.store.get_audio_asset(
                owner_type,
                owner_id,
                item_index,
                "candidate" if candidate else "current",
            )
            return self._payload(row, expected_text=str(target["text"])), ""

    async def ensure(
        self,
        target: dict[str, Any],
        *,
        settings: dict[str, str] | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        if not force and not self.category_enabled(str(target["owner_type"])):
            return None
        current = self.store.get_audio_asset(
            str(target["owner_type"]),
            int(target["owner_id"]),
            int(target.get("item_index", -1)),
            "current",
        )
        if self._payload(current, expected_text=str(target["text"])):
            return self._payload(current, expected_text=str(target["text"]))
        payload, error = await self._synthesize(
            target,
            settings=settings or self.default_settings(),
            candidate=False,
        )
        if error:
            logger.warning("英语私教音频生成失败（%s）: %s", target["text"], error)
        return payload

    async def generate(
        self,
        target: dict[str, Any],
        settings: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Generate or reuse a current asset for an explicit user request."""
        current = self.store.get_audio_asset(
            str(target["owner_type"]),
            int(target["owner_id"]),
            int(target.get("item_index", -1)),
            "current",
        )
        payload = self._payload(current, expected_text=str(target["text"]))
        if payload:
            return payload, ""
        return await self._synthesize(
            target,
            settings=settings or self.default_settings(),
            candidate=False,
        )

    async def regenerate(
        self,
        target: dict[str, Any],
        settings: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        return await self._synthesize(
            target,
            settings=settings or self.default_settings(),
            candidate=True,
        )

    async def apply(self, target: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        owner_type = str(target["owner_type"])
        owner_id = int(target["owner_id"])
        item_index = int(target.get("item_index", -1))
        async with self._lock_for(target):
            candidate = self.store.get_audio_asset(
                owner_type, owner_id, item_index, "candidate"
            )
            if not self._payload(candidate, expected_text=str(target["text"])):
                return None, "没有可应用的候选音频"
            old_rows = self.store.delete_audio_assets(
                owner_type, owner_id, item_index, status="current"
            )
            self.store.set_audio_status(int(candidate["id"]), "current")
            self._remove_rows(old_rows)
            current = self.store.get_audio_asset(
                owner_type, owner_id, item_index, "current"
            )
            return self._payload(current, expected_text=str(target["text"])), ""

    def delete_owner(self, owner_type: str, owner_id: int) -> None:
        self._remove_rows(self.store.delete_audio_assets(owner_type, owner_id))

    def asset_path_for_target(self, target: dict[str, Any]) -> Path | None:
        row = self.store.get_audio_asset(
            str(target["owner_type"]),
            int(target["owner_id"]),
            int(target.get("item_index", -1)),
            "current",
        )
        if not row or str(row.get("text") or "") != str(target["text"]):
            return None
        return self.path_for_asset(row)

    def file_path(self, asset_id: int) -> Path | None:
        row = self.store.get_audio_asset_by_id(asset_id)
        return self.path_for_asset(row)

    async def batch(self, kind: str = "all", limit: int = 200) -> dict[str, int]:
        limit = max(1, min(200, int(limit)))
        normalized = str(kind or "all").strip().lower()
        targets: list[dict[str, Any]] = []
        if normalized in {"all", "sentences", "sentence"}:
            for row in self.store.list_sentences(limit=limit):
                targets.append(
                    {
                        "owner_type": "sentence",
                        "owner_id": int(row["id"]),
                        "item_index": -1,
                        "text": str(row.get("sentence") or "").strip(),
                    }
                )
        if normalized in {"all", "vocab", "word"}:
            for row in self.store.list_vocab(limit=limit):
                targets.append(
                    {
                        "owner_type": "vocab",
                        "owner_id": int(row["id"]),
                        "item_index": -1,
                        "text": str(row.get("word") or "").strip(),
                    }
                )
        if normalized in {"all", "practice", "daily"}:
            for practice in self.store.list_daily(limit=limit):
                for index, item in enumerate(practice.get("items") or []):
                    text = str((item or {}).get("en") or "").strip()
                    if text:
                        targets.append(
                            {
                                "owner_type": "practice",
                                "owner_id": int(practice["id"]),
                                "item_index": index,
                                "text": text,
                            }
                        )
        result = {"total": len(targets), "generated": 0, "failed": 0, "skipped": 0}
        for target in targets:
            before = self.store.get_audio_asset(
                target["owner_type"],
                target["owner_id"],
                target["item_index"],
                "current",
            )
            payload = await self.ensure(target, force=True)
            if payload:
                if before and before.get("text") == target["text"]:
                    result["skipped"] += 1
                else:
                    result["generated"] += 1
            elif before and before.get("text") == target["text"]:
                result["skipped"] += 1
            else:
                result["failed"] += 1
        return result

    async def combined(
        self, practice_id: int, *, generate_missing: bool = True
    ) -> tuple[dict[str, Any] | None, str]:
        """Return a merged WAV asset covering every item of one practice day.

        The combined file is cached as an audio asset with ``item_index = -1``
        whose ``text`` field stores the fingerprint (joined asset ids) of the
        per-item source audios. The daily push and the WebUI play-all player
        share this cache: either side builds it once and the other reuses it
        until one of the item audios changes.

        Args:
            practice_id: The daily_practice row id.
            generate_missing: When False this is a cheap probe that only
                returns an already cached combined asset and never synthesizes
                or merges.

        Returns:
            (asset payload, "") on success, (None, error) on failure. The probe
            mode returns (None, "") when no reusable combined asset exists.
        """
        practice = self.store.get_daily_by_id(int(practice_id))
        if not practice:
            return None, "练习记录不存在"
        fingerprint_parts: list[str] = []
        source_paths: list[Path] = []
        item_sources: list[tuple[int, str, float]] = []
        for index, item in enumerate(practice.get("items") or []):
            text = str((item or {}).get("en") or "").strip()
            if not text:
                continue
            row = self.store.get_audio_asset(
                "practice", int(practice_id), index, "current"
            )
            payload = self._payload(row, expected_text=text)
            if not payload and not generate_missing:
                return None, ""
            if not payload:
                payload, error = await self.generate(
                    {
                        "owner_type": "practice",
                        "owner_id": int(practice_id),
                        "item_index": index,
                        "text": text,
                    }
                )
                if not payload:
                    return None, error or f"第 {index + 1} 条音频不可用，请先补齐音频"
                row = self.store.get_audio_asset(
                    "practice", int(practice_id), index, "current"
                )
            path = self.path_for_asset(row)
            if row is None or not path:
                return None, f"第 {index + 1} 条音频文件缺失"
            try:
                with wave.open(str(path), "rb") as handle:
                    duration = handle.getnframes() / float(handle.getframerate())
            except (wave.Error, OSError):
                return None, f"第 {index + 1} 条音频文件读取失败"
            fingerprint_parts.append(str(row["id"]))
            source_paths.append(path)
            item_sources.append((index, text, duration))
        if not fingerprint_parts:
            return None, "这一天没有可合并的音频"

        # Per-item start offsets inside the merged file so the WebUI playlist
        # can jump straight to any sentence.
        items_info: list[dict[str, Any]] = []
        start = 0.0
        for index, text, duration in item_sources:
            items_info.append({"index": index, "start": round(start, 3), "text": text})
            start += duration

        fingerprint = "|".join(fingerprint_parts)
        cached = self.store.get_audio_asset("practice", int(practice_id), -1, "current")
        if cached and str(cached.get("text") or "") == fingerprint:
            payload = self._payload(cached)
            if payload:
                payload["items"] = items_info
                return payload, ""
        self._remove_rows(
            self.store.delete_audio_assets("practice", int(practice_id), -1)
        )
        merged = self.merge_wav(
            source_paths, self.audio_dir / f"english_tutor_{uuid.uuid4().hex}.wav"
        )
        if not merged:
            return None, "合并音频失败"
        settings = self.default_settings()
        self.store.add_audio_asset(
            "practice",
            int(practice_id),
            -1,
            fingerprint,
            merged.name,
            "current",
            settings["emotion_mode"],
            settings["emotion"],
            settings["role"],
        )
        row = self.store.get_audio_asset("practice", int(practice_id), -1, "current")
        payload = self._payload(row)
        if not payload:
            return None, "合并音频失败"
        payload["items"] = items_info
        return payload, ""

    @staticmethod
    def merge_wav(paths: list[Path], output: Path) -> Path | None:
        chunks: list[bytes] = []
        for path in paths:
            try:
                chunks.append(path.read_bytes())
            except OSError:
                continue
        if not chunks:
            return None
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as out_wav:
                params_set = False
                for chunk in chunks:
                    with wave.open(io.BytesIO(chunk), "rb") as in_wav:
                        if not params_set:
                            out_wav.setparams(in_wav.getparams())
                            params_set = True
                        out_wav.writeframes(in_wav.readframes(in_wav.getnframes()))
            output.write_bytes(buffer.getvalue())
            return output
        except Exception as exc:
            logger.warning("合并每日练习音频失败: %s", exc)
            return None
