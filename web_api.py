"""Web API routes for the English tutor plugin management page.

All handlers are zero-argument async functions (path/query access goes through
the ``astrbot.api.web`` request proxy) and return ``json_response`` payloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api.web import error_response, file_response, json_response, request

if TYPE_CHECKING:
    from .main import EnglishTutorPlugin

PAGE_PREFIX = "/astrbot_plugin_english_tutor"


def _paging(default_size: int = 20) -> tuple[int, int, int]:
    """Return (page, offset, page_size) from query parameters."""
    try:
        page = max(1, int(request.query.get("page", 1)))
        page_size = min(200, max(1, int(request.query.get("page_size", default_size))))
    except (TypeError, ValueError):
        page, page_size = 1, default_size
    return page, (page - 1) * page_size, page_size


def _int_field(payload: dict[str, Any], key: str) -> int | None:
    try:
        return int(payload.get(key))
    except (TypeError, ValueError):
        return None


def _audio_settings(plugin: EnglishTutorPlugin, payload: dict[str, Any]) -> dict[str, str]:
    manager = getattr(plugin, "audio_manager", None)
    defaults = manager.default_settings() if manager else {
        "emotion_mode": "default",
        "emotion": "",
        "role": "",
    }
    mode = str(payload.get("emotion_mode", defaults["emotion_mode"]) or "default")
    mode = mode.strip().lower()
    if mode not in {"default", "auto", "specified"}:
        mode = defaults["emotion_mode"]
    return {
        "emotion_mode": mode,
        "emotion": str(payload.get("emotion", defaults["emotion"]) or "").strip(),
        "role": str(payload.get("role", defaults["role"]) or "").strip(),
    }


def _audio_target(
    plugin: EnglishTutorPlugin, payload: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    manager = getattr(plugin, "audio_manager", None)
    if manager is None:
        return None, "音频功能未初始化"
    item_id = _int_field(payload, "id")
    if not item_id:
        return None, "缺少 id"
    item_index = _int_field(payload, "item_index")
    if item_index is None:
        item_index = -1
    target = manager.target(str(payload.get("kind") or ""), item_id, item_index)
    return (target, "") if target else (None, "记录不存在或练习条目无效")


def register_routes(plugin: EnglishTutorPlugin) -> None:
    """Register all management-page APIs on the plugin context."""
    store = plugin.store
    assert store is not None
    register = plugin.context.register_web_api

    # ---------- stats ----------

    async def stats():
        overview = store.overview()
        today = plugin._today()
        overview["due_vocab"] = len(store.due_vocab(today=today))
        overview["due_errors"] = len(store.due_errors(today=today))
        overview["bound_umo"] = await plugin.get_kv_data("bound_umo", "")
        return json_response(overview)

    # ---------- sentences ----------

    async def list_sentences():
        page, offset, page_size = _paging()
        keyword = str(request.query.get("keyword", ""))
        date = str(request.query.get("date", "")) or None
        items = store.list_sentences(
            date=date, keyword=keyword, limit=page_size, offset=offset
        )
        manager = getattr(plugin, "audio_manager", None)
        if manager:
            for item in items:
                manager.attach(item, "sentence", int(item["id"]))
        total = store.count_sentences(date=date, keyword=keyword)
        return json_response({"items": items, "total": total, "page": page})

    async def update_sentence():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        fields = {
            k: str(payload.get(k, "")).strip()
            for k in ("sentence", "note", "tags", "context", "dialog")
            if k in payload
        }
        store.update_sentence(item_id, fields)
        updated = store.get_sentence(item_id)
        if updated:
            plugin._schedule_audio("sentence", item_id, str(updated.get("sentence") or ""))
        return json_response({"updated": item_id})

    async def delete_sentence():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        manager = getattr(plugin, "audio_manager", None)
        if manager:
            manager.delete_owner("sentence", item_id)
        store.delete_sentence(item_id)
        return json_response({"deleted": item_id})

    async def add_sentence():
        payload = await request.json(default={})
        sentence = str(payload.get("sentence", "")).strip()
        if not sentence:
            return error_response("句子不能为空", status_code=400)
        sentence_id = store.add_sentence(
            "",
            plugin._today(),
            sentence,
            str(payload.get("note", "")).strip(),
            "user",
            str(payload.get("tags", "")).strip(),
            str(payload.get("context", "")).strip(),
            str(payload.get("dialog", "")).strip(),
        )
        plugin._schedule_audio("sentence", sentence_id, sentence)
        return json_response({"added": sentence, "id": sentence_id})

    # ---------- errors ----------

    async def list_errors():
        page, offset, page_size = _paging()
        keyword = str(request.query.get("keyword", ""))
        status = str(request.query.get("status", "all")) or "all"
        items = store.list_errors(
            status=status, keyword=keyword, limit=page_size, offset=offset
        )
        total = store.count_errors(status=status, keyword=keyword)
        return json_response({"items": items, "total": total, "page": page})

    async def update_error():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        fields = {
            k: str(payload.get(k, "")).strip()
            for k in ("original", "corrected", "explanation", "category", "status")
            if k in payload
        }
        store.update_error(item_id, fields)
        return json_response({"updated": item_id})

    async def delete_error():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        store.delete_error(item_id)
        return json_response({"deleted": item_id})

    async def add_error():
        payload = await request.json(default={})
        original = str(payload.get("original", "")).strip()
        if not original:
            return error_response("原句不能为空", status_code=400)
        today = plugin._today()
        store.add_error(
            "",
            today,
            original,
            str(payload.get("corrected", "")).strip(),
            str(payload.get("explanation", "")).strip(),
            str(payload.get("category", "")).strip(),
        )
        status = str(payload.get("status", "")).strip()
        if status in ("open", "resolved"):
            row = store.find_open_error("", original)
            if row and status == "resolved":
                store.update_error(row["id"], {"status": "resolved"})
        return json_response({"added": original})

    # ---------- vocab ----------

    async def list_vocab():
        page, offset, page_size = _paging()
        keyword = str(request.query.get("keyword", ""))
        items = store.list_vocab(keyword=keyword, limit=page_size, offset=offset)
        manager = getattr(plugin, "audio_manager", None)
        if manager:
            for item in items:
                manager.attach(item, "vocab", int(item["id"]))
        total = store.count_vocab(keyword=keyword)
        return json_response({"items": items, "total": total, "page": page})

    async def update_vocab():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        fields = {
            k: str(payload.get(k, "")).strip()
            for k in ("word", "meaning", "example", "context", "dialog")
            if k in payload
        }
        store.update_vocab(item_id, fields)
        updated = store.get_vocab(item_id)
        if updated:
            plugin._schedule_audio("vocab", item_id, str(updated.get("word") or ""))
        return json_response({"updated": item_id})

    async def delete_vocab():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        manager = getattr(plugin, "audio_manager", None)
        if manager:
            manager.delete_owner("vocab", item_id)
        store.delete_vocab(item_id)
        return json_response({"deleted": item_id})

    async def add_vocab():
        payload = await request.json(default={})
        word = str(payload.get("word", "")).strip()
        if not word:
            return error_response("单词不能为空", status_code=400)
        today = plugin._today()
        vocab_id = store.add_vocab(
            "",
            today,
            word,
            str(payload.get("meaning", "")).strip(),
            str(payload.get("example", "")).strip(),
            "user",
            str(payload.get("context", "")).strip(),
            str(payload.get("dialog", "")).strip(),
        )
        plugin._schedule_audio("vocab", vocab_id, word)
        return json_response({"added": word, "id": vocab_id})

    # ---------- ai fill ----------

    async def ai_fill():
        payload = await request.json(default={})
        fill_type = str(payload.get("type", "")).strip()
        text = str(payload.get("text", "")).strip()
        if fill_type not in ("sentence", "vocab") or not text:
            return error_response(
                "参数无效：需要 type(sentence/vocab) 和 text", status_code=400
            )
        fields = await plugin.ai_fill(fill_type, text)
        if not fields:
            return error_response(
                "模型未返回有效内容，请检查模型回退链配置后重试", status_code=502
            )
        return json_response({"fields": fields})

    # ---------- daily practice ----------

    async def list_practice():
        page, offset, page_size = _paging(30)
        items = store.list_daily(limit=page_size, offset=offset)
        manager = getattr(plugin, "audio_manager", None)
        if manager:
            for practice in items:
                for index, item in enumerate(practice.get("items") or []):
                    if isinstance(item, dict):
                        manager.attach(item, "practice", int(practice["id"]), index)
        total = store.count_daily()
        return json_response({"items": items, "total": total, "page": page})

    async def delete_practice():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        manager = getattr(plugin, "audio_manager", None)
        if manager:
            manager.delete_owner("practice", item_id)
        store.delete_daily(item_id)
        return json_response({"deleted": item_id})

    # ---------- archive ----------

    async def list_archive():
        page, offset, page_size = _paging(50)
        keyword = str(request.query.get("keyword", ""))
        date = str(request.query.get("date", "")) or None
        items = store.list_archive(
            date=date, keyword=keyword, limit=page_size, offset=offset
        )
        total = store.count_archive(date=date, keyword=keyword)
        return json_response({"items": items, "total": total, "page": page})

    async def delete_archive():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        store.delete_archive(item_id)
        return json_response({"deleted": item_id})

    # ---------- audio integration ----------

    async def audio_options():
        manager = getattr(plugin, "audio_manager", None)
        if manager is None:
            return json_response({"available": False, "enabled": False})
        return json_response(manager.options())

    async def audio_generate():
        payload = await request.json(default={})
        target, error = _audio_target(plugin, payload or {})
        if error:
            return error_response(error, status_code=400)
        manager = plugin.audio_manager
        assert manager is not None and target is not None
        asset, generation_error = await manager.generate(
            target, settings=_audio_settings(plugin, payload or {})
        )
        if not asset:
            return error_response(generation_error or "音频生成失败", status_code=502)
        return json_response({"audio": asset})

    async def audio_regenerate():
        payload = await request.json(default={})
        target, error = _audio_target(plugin, payload or {})
        if error:
            return error_response(error, status_code=400)
        manager = plugin.audio_manager
        assert manager is not None and target is not None
        asset, generation_error = await manager.regenerate(
            target, settings=_audio_settings(plugin, payload or {})
        )
        if not asset:
            return error_response(generation_error or "候选音频生成失败", status_code=502)
        return json_response({"audio": asset})

    async def audio_apply():
        payload = await request.json(default={})
        target, error = _audio_target(plugin, payload or {})
        if error:
            return error_response(error, status_code=400)
        manager = plugin.audio_manager
        assert manager is not None and target is not None
        asset, apply_error = await manager.apply(target)
        if not asset:
            return error_response(apply_error or "候选音频应用失败", status_code=409)
        return json_response({"audio": asset})

    async def audio_batch():
        payload = await request.json(default={})
        payload = payload or {}
        kind = str(payload.get("kind") or "all")
        try:
            limit = max(1, min(200, int(payload.get("limit", 200))))
        except (TypeError, ValueError):
            limit = 200
        manager = plugin.audio_manager
        if manager is None:
            return error_response("音频功能未初始化", status_code=503)
        result = await manager.batch(kind, limit)
        return json_response(result)

    async def audio_file(asset_id: str):
        try:
            parsed_id = int(asset_id)
        except (TypeError, ValueError):
            return error_response("音频不存在", status_code=404)
        manager = plugin.audio_manager
        path = manager.file_path(parsed_id) if manager else None
        if path is None:
            return error_response("音频不存在", status_code=404)
        return file_response(
            path,
            content_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )

    register(f"{PAGE_PREFIX}/stats", stats, ["GET"], "Tutor overview stats")

    register(f"{PAGE_PREFIX}/sentences", list_sentences, ["GET"], "List sentences")
    register(f"{PAGE_PREFIX}/sentences/add", add_sentence, ["POST"], "Add a sentence")
    register(
        f"{PAGE_PREFIX}/sentences/update",
        update_sentence,
        ["POST"],
        "Update a sentence",
    )
    register(
        f"{PAGE_PREFIX}/sentences/delete",
        delete_sentence,
        ["POST"],
        "Delete a sentence",
    )

    register(f"{PAGE_PREFIX}/errors", list_errors, ["GET"], "List errors")
    register(f"{PAGE_PREFIX}/errors/add", add_error, ["POST"], "Add an error")
    register(f"{PAGE_PREFIX}/errors/update", update_error, ["POST"], "Update an error")
    register(f"{PAGE_PREFIX}/errors/delete", delete_error, ["POST"], "Delete an error")

    register(f"{PAGE_PREFIX}/vocab", list_vocab, ["GET"], "List vocab")
    register(f"{PAGE_PREFIX}/vocab/add", add_vocab, ["POST"], "Add a vocab item")
    register(
        f"{PAGE_PREFIX}/vocab/update", update_vocab, ["POST"], "Update a vocab item"
    )
    register(
        f"{PAGE_PREFIX}/vocab/delete", delete_vocab, ["POST"], "Delete a vocab item"
    )

    register(f"{PAGE_PREFIX}/ai_fill", ai_fill, ["POST"], "AI-fill notebook fields")

    register(f"{PAGE_PREFIX}/practice", list_practice, ["GET"], "List daily practice")
    register(
        f"{PAGE_PREFIX}/practice/delete",
        delete_practice,
        ["POST"],
        "Delete a daily practice",
    )

    register(f"{PAGE_PREFIX}/archive", list_archive, ["GET"], "List archived messages")
    register(
        f"{PAGE_PREFIX}/archive/delete",
        delete_archive,
        ["POST"],
        "Delete an archived message",
    )

    register(f"{PAGE_PREFIX}/audio/options", audio_options, ["GET"], "Audio options")
    register(
        f"{PAGE_PREFIX}/audio/generate",
        audio_generate,
        ["POST"],
        "Generate current audio",
    )
    register(
        f"{PAGE_PREFIX}/audio/regenerate",
        audio_regenerate,
        ["POST"],
        "Generate candidate audio",
    )
    register(
        f"{PAGE_PREFIX}/audio/apply",
        audio_apply,
        ["POST"],
        "Apply candidate audio",
    )
    register(
        f"{PAGE_PREFIX}/audio/batch",
        audio_batch,
        ["POST"],
        "Batch generate missing audio",
    )
    register(
        f"{PAGE_PREFIX}/audio/file/<asset_id>",
        audio_file,
        ["GET"],
        "Serve tutor audio",
    )
