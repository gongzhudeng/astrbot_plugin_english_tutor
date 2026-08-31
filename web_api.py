"""Web API routes for the English tutor plugin management page.

All handlers are zero-argument async functions (path/query access goes through
the ``astrbot.api.web`` request proxy) and return ``json_response`` payloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astrbot.api.web import error_response, json_response, request

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
        return json_response({"updated": item_id})

    async def delete_sentence():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        store.delete_sentence(item_id)
        return json_response({"deleted": item_id})

    async def add_sentence():
        payload = await request.json(default={})
        sentence = str(payload.get("sentence", "")).strip()
        if not sentence:
            return error_response("句子不能为空", status_code=400)
        store.add_sentence(
            "",
            plugin._today(),
            sentence,
            str(payload.get("note", "")).strip(),
            "user",
            str(payload.get("tags", "")).strip(),
            str(payload.get("context", "")).strip(),
            str(payload.get("dialog", "")).strip(),
        )
        return json_response({"added": sentence})

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
        return json_response({"updated": item_id})

    async def delete_vocab():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
        store.delete_vocab(item_id)
        return json_response({"deleted": item_id})

    async def add_vocab():
        payload = await request.json(default={})
        word = str(payload.get("word", "")).strip()
        if not word:
            return error_response("单词不能为空", status_code=400)
        today = plugin._today()
        store.add_vocab(
            "",
            today,
            word,
            str(payload.get("meaning", "")).strip(),
            str(payload.get("example", "")).strip(),
            "user",
            str(payload.get("context", "")).strip(),
            str(payload.get("dialog", "")).strip(),
        )
        return json_response({"added": word})

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
        total = store.count_daily()
        return json_response({"items": items, "total": total, "page": page})

    async def delete_practice():
        payload = await request.json(default={})
        item_id = _int_field(payload, "id")
        if not item_id:
            return error_response("缺少 id", status_code=400)
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
