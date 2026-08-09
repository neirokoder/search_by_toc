import asyncio
import json
import logging
import re
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..clients import registry_client, rag_client, llm_client
from ..config import get_settings
from ..models import ChatMessage, ChatSession, ChatSource
from .prompts import _SUMMARY_PROMPT, _SYSTEM_PROMPT, _TOOLS, _REVIEW_PROMPT

logger = logging.getLogger(__name__)

_FINAL_ASSISTANT_STATUSES = ("answered", "not_found")
_SOURCE_REF_RE = re.compile(r"\[source:(\d+)\]")
_SOURCE_REF_COMBINED_RE = re.compile(r"\[source:\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_SOURCE_REF_RANGE_RE = re.compile(r"\[source:(\d+)-(\d+)\]")
_DOC_ID_REF_RE = re.compile(r"\[(?:document_id|doc_id):\s*(\d+)([^\]]*)\]")
_REF_SECTION_ID_RE = re.compile(r"(?:section_id|sec):\s*(\d+)")
_REF_PAGE_RE = re.compile(r"(?:стр\.?\s*|page:\s*)(\d+)")
_REF_FIGURE_RE = re.compile(r"рис\.?\s*([^,\]]+)")
_BARE_N_REF_RE = re.compile(r"\[(\d+)\]")
_OLD_MARKER_RE = re.compile(r"\s*%\[[^\]]*\]%")
_INLINE_LATEX_RE = re.compile(r"\\\((.+?)\\\)")
_DISPLAY_LATEX_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)


def _convert_latex_delimiters(text: str) -> str:
    """Convert LaTeX delimiters to widely-supported $$ and $ forms."""
    text = _DISPLAY_LATEX_RE.sub(r"$$\1$$", text)
    text = _INLINE_LATEX_RE.sub(r"$\1$", text)
    return text


def _parse_doc_id_refs(text: str) -> list[dict]:
    """Извлекает doc_refs из финального текста, поддерживая все форматы цитат."""
    matches = _DOC_ID_REF_RE.findall(text)
    refs = []
    for did, rest in matches:
        if not did:
            continue
        ref = {"document_id": int(did), "page_number": None, "section_id": None, "figure": None}
        rest = rest.strip()
        if rest.startswith(","):
            rest = rest[1:].strip()
        if rest:
            m_sid = _REF_SECTION_ID_RE.search(rest)
            if m_sid:
                ref["section_id"] = int(m_sid.group(1))
            m_page = _REF_PAGE_RE.search(rest)
            if m_page:
                ref["page_number"] = int(m_page.group(1))
            m_fig = _REF_FIGURE_RE.search(rest)
            if m_fig:
                ref["figure"] = m_fig.group(1).strip()
        refs.append(ref)
    return refs

# Tool definitions are in prompts.py — see _TOOLS

_MAX_TOOL_ITERS = 35
_MAX_CONSECUTIVE_EMPTY_SEARCHES = 3


class _DataExhausted(Exception):
    """Control-flow исключение: LLM повторно вызывает rag_search после блокировки.
    Перехватывается в run_pipeline для установки статуса not_found."""


_CONTEXT_TOKEN_BUDGET = 100000
_SUMMARY_MAX_TOKENS = 16196
_RECENT_KEEP_MESSAGES = 50


def _normalize_source_refs(text: str) -> str:
    text = _SOURCE_REF_RANGE_RE.sub(
        lambda m: "".join(
            f"[source:{i}]" for i in range(int(m.group(1)), int(m.group(2)) + 1)
        ),
        text,
    )
    text = _SOURCE_REF_COMBINED_RE.sub(
        lambda m: "".join(f"[source:{i.strip()}]" for i in m.group(1).split(",")),
        text,
    )
    return text


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


_GENERIC_WORDS = {
    "что", "как", "где", "когда", "зачем", "почему", "какие", "какой", "какая",
    "этот", "эта", "это", "эти", "его", "её", "их", "наш", "ваш",
    "должен", "должна", "должны", "нужно", "необходимо", "может", "могут",
    "является", "являются", "имеет", "имеют", "все", "всё", "также",
    "определение", "понятие", "сущность",
    "регламентировать", "регламентируют", "описать", "описывает", "описывают",
    "рассказать", "расскажи", "покажи", "показать", "показывает",
    "назвать", "назови", "перечислить", "перечисли",
}


def _extract_key_terms(query: str) -> str:
    terms = []
    for w in query.lower().split():
        w = w.strip(""".,!?;:()[]{}""-'\"«»""")
        if w not in _GENERIC_WORDS and len(w) > 2:
            terms.append(w)
    return " ".join(terms)


def _is_garbage(text: str) -> bool:
    if not text:
        return True
    text = text.strip()
    if len(text) < 10:
        return False
    low = text.lower()
    words = low.split()
    if len(words) < 5:
        return False
    for w in set(words):
        run = 0
        for i, w2 in enumerate(words):
            if w2 == w:
                run += 1
                if run >= 8:
                    return True
            else:
                run = 0
    unique_ratio = len(set(words)) / len(words)
    if len(words) > 200 and unique_ratio < 0.05:
        return True
    return False


def _clean_content(role: str, content: str) -> str:
    if role == "assistant":
        content = _normalize_source_refs(content)
        content = _SOURCE_REF_RE.sub("", content)
        content = _BARE_N_REF_RE.sub("", content)
        content = _OLD_MARKER_RE.sub("", content)
    return content


def _enrich_citations(
    llm_text: str, chunks: list[rag_client.Chunk]
) -> tuple[str, list[int]]:
    used_indices: list[int] = []
    seen: set[int] = set()

    def replace(m: re.Match) -> str:
        n = int(m.group(1))
        if n < 0 or n >= len(chunks):
            return ""
        chunk = chunks[n]
        if n not in seen:
            seen.add(n)
            used_indices.append(n)
        clause = f" §{chunk.clause}" if chunk.clause else ""
        page = f", стр. {chunk.page}" if chunk.page else ""
        return f"[document_id:{chunk.document_id}, section_id:{chunk.section_id}{clause}{page}]"

    normalized = _normalize_source_refs(llm_text)
    enriched = _SOURCE_REF_RE.sub(replace, normalized)
    return enriched, used_indices


def _format_chunks(chunks: list[rag_client.Chunk], start_index: int) -> str:
    parts = []
    for i, c in enumerate(chunks):
        clause = f" §{c.clause}" if c.clause else ""
        page = f", стр. {c.page}" if c.page else ""
        doc_ref = f" [doc_id:{c.document_id}]"
        sec_ref = f" [sec:{c.section_id}]" if c.section_id else ""
        parts.append(f"[{start_index + i}] «Документ №{c.document_id}»{clause}{page}{doc_ref}{sec_ref}:\n{c.content}")
    return "\n\n".join(parts)


async def _load_session_meta(
    session_factory: async_sessionmaker, session_id: str
) -> tuple[str | None, int | None]:
    async with session_factory() as db:
        session = await db.get(ChatSession, session_id)
        if session is None:
            return None, None
        return session.summary, session.summarized_until_message_id


async def _load_messages_after(
    session_factory: async_sessionmaker,
    session_id: str,
    exclude_message_id: str,
    after_id: int | None,
) -> list[dict]:
    async with session_factory() as db:
        q = select(ChatMessage).where(
            ChatMessage.session_id == session_id,
            ChatMessage.message_id != exclude_message_id,
            ChatMessage.content.is_not(None),
            ChatMessage.role.in_(("user", "assistant")),
            ChatMessage.status.in_(("pending",) + _FINAL_ASSISTANT_STATUSES),
        )
        if after_id is not None:
            q = q.where(ChatMessage.message_id > after_id)
        rows = (await db.execute(q.order_by(ChatMessage.message_id.asc()))).scalars().all()

    return [
        {"id": m.message_id, "role": m.role, "content": _clean_content(m.role, m.content)}
        for m in rows
    ]


async def _summarize(prev_summary: str | None, messages: list[dict], settings, session_id: str = "") -> str:
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    if settings.MOCK_LLM_ENABLED:
        base = f"{prev_summary} | " if prev_summary else ""
        return base + "; ".join(m["content"][:60] for m in messages)

    base = f"Текущее резюме:\n{prev_summary}\n\n" if prev_summary else ""
    prompt = [
        {"role": "system", "content": _SUMMARY_PROMPT},
        {"role": "user", "content": base + f"Добавь в резюме переписку:\n{convo}"},
    ]
    result = await llm_client.complete(prompt, max_tokens=_SUMMARY_MAX_TOKENS, cache_key=session_id or None)
    return result.content or ""


async def _prepare_context(
    session_factory: async_sessionmaker,
    session_id: str,
    exclude_message_id: str,
    settings,
) -> tuple[str | None, list[dict]]:
    summary, until = await _load_session_meta(session_factory, session_id)
    history = await _load_messages_after(session_factory, session_id, exclude_message_id, until)

    budget = _CONTEXT_TOKEN_BUDGET
    keep = _RECENT_KEEP_MESSAGES

    def total_tokens() -> int:
        t = _estimate_tokens(summary) if summary else 0
        return t + sum(_estimate_tokens(m["content"]) for m in history)

    if total_tokens() > budget and len(history) > keep:
        to_compress = history[:-keep]
        summary = await _summarize(summary, to_compress, settings, session_id)
        last_id = to_compress[-1]["id"]
        async with session_factory() as db:
            async with db.begin():
                await db.execute(
                    update(ChatSession)
                    .where(ChatSession.session_id == session_id)
                    .values(summary=summary, summarized_until_message_id=last_id)
                )
        history = history[-keep:]

    return summary, [{"role": m["role"], "content": m["content"]} for m in history]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _set_status(
    session_factory: async_sessionmaker,
    message_id: str,
    status: str,
    status_message: str | None = None,
    progress: int | None = None,
) -> None:
    async with session_factory() as db:
        async with db.begin():
            values: dict = {"status": status, "timestamp": _utcnow()}
            if status_message is not None:
                values["message"] = status_message
            if progress is not None:
                values["progress"] = progress
            await db.execute(
                update(ChatMessage)
                .where(ChatMessage.message_id == message_id)
                .values(**values)
            )


def _build_llm_mock(query: str, chunks: list[rag_client.Chunk]) -> str:
    if not chunks:
        return "По данному запросу релевантные фрагменты в базе знаний не найдены."

    parts = []
    for i, chunk in enumerate(chunks[:3]):
        parts.append(f"{chunk.excerpt} [source:{i}]")
    return " ".join(parts)


_ANALYZE_PROGRESS = [22, 30, 38, 46, 54, 62, 70, 75, 78, 78]
_SEARCH_PROGRESS = [26, 34, 42, 50, 58, 66, 73, 77, 78, 78]


async def _run_tool_loop(
    messages: list[dict],
    settings,
    valid_at: str,
    message_id: str,
    session_id: str,
    session_factory: async_sessionmaker,
) -> tuple[str, list[rag_client.Chunk], int, int]:
    all_chunks: list[rag_client.Chunk] = []
    total_prompt = 0
    total_completion = 0

    consecutive_empty_searches = 0

    for i in range(_MAX_TOOL_ITERS):
        p = _ANALYZE_PROGRESS[i] if i < len(_ANALYZE_PROGRESS) else 78
        has_tool_results = any(m.get("role") == "tool" for m in messages)
        msg = "Обработка результатов поиска..." if has_tool_results else "Анализ запроса..."
        await _set_status(session_factory, message_id, "analyzing", msg, progress=p)
        result = await llm_client.complete(messages, tools=_TOOLS, cache_key=session_id)
        total_prompt += result.prompt_tokens
        total_completion += result.completion_tokens

        if not result.tool_calls:
            content = result.content or ""
            if _is_garbage(content):
                logger.warning("llm_iter=%d garbage content, retrying", i, extra={"message_id": message_id})
                messages.append({"role": "user", "content": "Сгенерирован некорректный ответ. Повтори попытку."})
                continue
            if content.strip().startswith("План:"):
                logger.warning("llm_iter=%d plan-only without tool_calls, retrying", i, extra={"message_id": message_id})
                messages.append({"role": "user", "content": "Вызови rag_search сейчас."})
                continue
            return content, all_chunks, total_prompt, total_completion

        messages.append({
            "role": "assistant",
            "content": result.content,
            "tool_calls": result.tool_calls,
        })

        if result.content:
            calls = ", ".join(tc["function"]["name"] for tc in result.tool_calls)
            logger.info("llm_iter=%d plan=%.200s calls=[%s]", i, result.content.replace("\n", " "), calls, extra={"message_id": message_id})

        for tc in result.tool_calls:
            fn = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            sp = _SEARCH_PROGRESS[i] if i < len(_SEARCH_PROGRESS) else 78

            if fn == "rag_search":
                instruction = args.get("instruction", "")
                phrases = args.get("phrases")
                is_retry = args.get("is_retry", False)
                if is_retry and consecutive_empty_searches >= _MAX_CONSECUTIVE_EMPTY_SEARCHES:
                    logger.info(
                        "rag_search blocked after %d empty retries, telling LLM no data",
                        _MAX_CONSECUTIVE_EMPTY_SEARCHES,
                        extra={"message_id": message_id},
                    )
                    tool_content = "Поиск не дал результатов. Данные в БД не найдены. Используй другие инструменты или ответь на основе имеющейся информации."
                else:
                    if not is_retry:
                        consecutive_empty_searches = 0
                    search_type = args.get("search_type", "hybrid_rrf")
                    top_n = args.get("top_n", 15)
                    instruction = args.get("instruction")
                    logger.info(
                        "tool_call rag_search instruction=%r phrases=%r search_type=%s top_n=%d",
                        instruction, phrases, search_type, top_n,
                        extra={"message_id": message_id},
                    )
                    status_msg = f"Поиск: {instruction[:60]}..." if not phrases else f"Поиск по {len(phrases)} фразам..."
                    await _set_status(session_factory, message_id, "searching", status_msg, progress=sp)
                    try:
                        chunks = await asyncio.wait_for(
                            rag_client.search(
                                instruction=instruction, phrases=phrases,
                                top_n=top_n, search_type=search_type,
                                valid_at=valid_at,
                            ),
                            timeout=180.0,
                        )
                        if not chunks and not phrases:
                            if search_type != "bm25":
                                logger.info("rag_search fallback bm25 exact instruction=%r", instruction, extra={"message_id": message_id})
                                await _set_status(session_factory, message_id, "searching", f"Повторный поиск (bm25): {instruction[:60]}...", progress=sp)
                                chunks = await asyncio.wait_for(
                                    rag_client.search(instruction, top_n=top_n, search_type="bm25", valid_at=valid_at),
                                    timeout=180.0,
                                )
                            if not chunks:
                                simple_query = _extract_key_terms(instruction)
                                if simple_query != instruction:
                                    logger.info("rag_search fallback bm25 simplified instruction=%r", simple_query, extra={"message_id": message_id})
                                    await _set_status(session_factory, message_id, "searching", f"Повторный поиск (bm25): {simple_query[:60]}...", progress=sp)
                                    chunks = await asyncio.wait_for(
                                        rag_client.search(simple_query, top_n=top_n, search_type="bm25", valid_at=valid_at),
                                        timeout=180.0,
                                    )
                        start_idx = len(all_chunks)
                        all_chunks.extend(chunks)
                        tool_content = _format_chunks(chunks, start_idx) if chunks else "Релевантные фрагменты не найдены."
                    except Exception as exc:
                        logger.error("tool_call rag_search failed: %s", exc, extra={"message_id": message_id}, exc_info=True)
                        tool_content = f"Ошибка поиска: {exc}"

                    if is_retry:
                        consecutive_empty_searches += 1

            elif fn == "registry_get_page_markdown":
                sec_ids = args.get("section_ids") or (args.get("section_id") is not None and [args["section_id"]]) or []
                if sec_ids:
                    logger.info("tool_call registry_get_page_markdown section_ids=%s", sec_ids, extra={"message_id": message_id})
                    await _set_status(session_factory, message_id, "searching", f"Чтение страниц {len(sec_ids)} секций...", progress=sp)
                    try:
                        tool_content = await asyncio.wait_for(
                            registry_client.get_sections_markdown([int(s) for s in sec_ids]),
                            timeout=180.0,
                        )
                    except Exception as exc:
                        logger.warning("tool_call registry_get_page_markdown section_ids=%s failed: %s", sec_ids, exc, extra={"message_id": message_id}, exc_info=True)
                        tool_content = f"Ошибка получения страниц секций: {exc}"
                else:
                    doc_ids = args.get("document_ids", [])
                    pages = args.get("pages", [])
                    if doc_ids and pages:
                        if len(doc_ids) == 1 and len(pages) == 1:
                            did = int(doc_ids[0])
                            pn = int(pages[0])
                            logger.info("tool_call registry_get_page_markdown doc_id=%d page=%d", did, pn, extra={"message_id": message_id})
                            await _set_status(session_factory, message_id, "searching", f"Чтение страницы {pn} документа {did}...", progress=sp)
                            try:
                                tool_content = await asyncio.wait_for(
                                    registry_client.get_page_markdown(document_id=did, page=pn),
                                    timeout=180.0,
                                )
                            except Exception as exc:
                                logger.warning("tool_call registry_get_page_markdown doc_id=%d page=%d failed: %s", did, pn, exc, extra={"message_id": message_id}, exc_info=True)
                                tool_content = f"Ошибка получения страницы {pn} документа {did}: {exc}"
                        else:
                            pairs = list(zip(doc_ids, pages))
                            logger.info("tool_call registry_get_page_markdown pairs=%s", pairs, extra={"message_id": message_id})
                            await _set_status(session_factory, message_id, "searching", f"Чтение {len(pairs)} страниц...", progress=sp)
                            try:
                                tool_content = await asyncio.wait_for(
                                    registry_client.get_pages_markdown([(int(d), int(p)) for d, p in pairs]),
                                    timeout=180.0,
                                )
                            except Exception as exc:
                                logger.warning("tool_call registry_get_page_markdown batch failed: %s", exc, extra={"message_id": message_id}, exc_info=True)
                                tool_content = f"Ошибка получения страниц: {exc}"
                    else:
                        tool_content = "Не указаны ни section_ids, ни пары document_ids+pages."

            elif fn == "registry_list_documents":
                logger.info("tool_call registry_list_documents args=%s", args, extra={"message_id": message_id})
                await _set_status(session_factory, message_id, "searching", "Поиск документов в реестре...", progress=sp)
                try:
                    tool_content = await asyncio.wait_for(
                        registry_client.list_documents(**{k: v for k, v in args.items() if v is not None}),
                        timeout=180.0,
                    )
                except Exception as exc:
                    logger.warning("tool_call registry_list_documents failed: %s", exc, extra={"message_id": message_id}, exc_info=True)
                    tool_content = f"Ошибка поиска документов: {exc}"

            elif fn == "registry_get_document":
                doc_ids = args.get("document_ids", [])
                if not doc_ids:
                    tool_content = "Не указан document_ids."
                elif len(doc_ids) == 1:
                    logger.info("tool_call registry_get_document doc_id=%d", doc_ids[0], extra={"message_id": message_id})
                    await _set_status(session_factory, message_id, "searching", f"Получение метаданных документа {doc_ids[0]}...", progress=sp)
                    try:
                        tool_content = await asyncio.wait_for(
                            registry_client.get_document(int(doc_ids[0])),
                            timeout=180.0,
                        )
                    except Exception as exc:
                        logger.warning("tool_call registry_get_document doc_id=%d failed: %s", doc_ids[0], exc, extra={"message_id": message_id}, exc_info=True)
                        tool_content = f"Ошибка получения метаданных документа {doc_ids[0]}: {exc}"
                else:
                    logger.info("tool_call registry_get_document doc_ids=%s", doc_ids, extra={"message_id": message_id})
                    await _set_status(session_factory, message_id, "searching", f"Получение метаданных {len(doc_ids)} документов...", progress=sp)
                    try:
                        tool_content = await asyncio.wait_for(
                            registry_client.get_documents([int(i) for i in doc_ids]),
                            timeout=180.0,
                        )
                    except Exception as exc:
                        logger.warning("tool_call registry_get_document batch failed: %s", exc, extra={"message_id": message_id}, exc_info=True)
                        tool_content = f"Ошибка получения метаданных документов: {exc}"

            else:
                tool_content = f"Неизвестный инструмент: {fn}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_content,
            })

    _PROHIBITIONS = [
        "На основе всей полученной информации напиши ответ пользователю. Не вызывай инструменты.",
        "Ты исчерпал лимит вызовов инструментов. ОТВЕТЬ ТЕКСТОМ на основе собранных данных. Не вызывай инструменты.",
        "ЗАПРЕЩЕНО вызывать инструменты. Напиши ответ прямо сейчас.",
        "НЕМЕДЛЕННО ОТВЕТЬ ТЕКСТОМ. Любые вызовы инструментов будут проигнорированы.",
        "ФИНАЛЬНОЕ ПРЕДУПРЕЖДЕНИЕ: ответь текстом на основе того, что уже собрано.",
    ]
    await _set_status(session_factory, message_id, "analyzing", "Формирование ответа...", progress=80)
    for attempt in range(5):
        prog = min(80 + attempt * 2, 95)
        await _set_status(session_factory, message_id, "analyzing", "Формирование ответа...", progress=prog)
        result = await llm_client.complete(messages, tools=_TOOLS, tool_choice="none", cache_key=session_id)
        total_prompt += result.prompt_tokens
        total_completion += result.completion_tokens
        if result.tool_calls:
            logger.warning("pipeline: wind-down attempt=%d/5 still got tool_calls=%s", attempt + 1, [tc["function"]["name"] for tc in result.tool_calls], extra={"message_id": message_id})
        elif result.content and not _is_garbage(result.content) and not result.content.strip().startswith("План:"):
            return result.content, all_chunks, total_prompt, total_completion
        else:
            if result.content:
                if _is_garbage(result.content):
                    logger.warning("pipeline: wind-down attempt=%d/5 garbage content, retrying", attempt + 1, extra={"message_id": message_id})
                else:
                    logger.warning("pipeline: wind-down attempt=%d/5 plan-only content", attempt + 1, extra={"message_id": message_id})
        msg = _PROHIBITIONS[attempt]
        logger.warning("pipeline: wind-down attempt=%d/5 adding prohibition", attempt + 1, extra={"message_id": message_id})
        messages.append({"role": "user", "content": msg})
    return "", all_chunks, total_prompt, total_completion


async def run_pipeline(
    session_factory: async_sessionmaker,
    message_id: str,
    session_id: str,
    user_query: str,
) -> None:
    settings = get_settings()
    logger.info("pipeline started", extra={"message_id": message_id, "session_id": session_id})

    try:
        t_start = _utcnow()
        warnings: list[str] = []
        prompt_tokens: int = 0
        completion_tokens: int = 0

        await _set_status(session_factory, message_id, "enriching", "Нормализация терминов запроса...", progress=10)
        enrichment_skipped = False
        try:
            enriched_query, _synonyms = await asyncio.wait_for(
                registry_client.enrich_query(user_query), timeout=180.0
            )
        except Exception:
            enriched_query = user_query
            enrichment_skipped = True
            warnings.append("Обогащение терминов недоступно. Поиск выполнен без нормализации.")
            logger.warning("query enrichment skipped", extra={"message_id": message_id}, exc_info=True)

        await _set_status(session_factory, message_id, "generating", "Подготовка контекста диалога...", progress=20)
        summary, history = await _prepare_context(
            session_factory, session_id, message_id, settings
        )

        if settings.MOCK_LLM_ENABLED:
            await _set_status(session_factory, message_id, "searching", f"Поиск: {enriched_query[:60]}...", progress=30)
            try:
                chunks = await asyncio.wait_for(
                    rag_client.search(enriched_query, top_n=15, valid_at=_utcnow().strftime("%Y-%m-%d")),
                    timeout=180.0,
                )
            except Exception:
                logger.error("rag search failed", extra={"message_id": message_id}, exc_info=True)
                async with session_factory() as db:
                    async with db.begin():
                        await db.execute(
                            update(ChatMessage)
                            .where(ChatMessage.message_id == message_id)
                            .values(
                                content="Поиск временно недоступен. Попробуйте повторить запрос.",
                                status="failed",
                                processing_time_ms=0,
                            )
                        )
                return

            if not chunks:
                async with session_factory() as db:
                    async with db.begin():
                        await db.execute(
                            update(ChatMessage)
                            .where(ChatMessage.message_id == message_id)
                            .values(
                                content="В базе знаний не найдено подтверждённых фрагментов по данному запросу.",
                                status="not_found",
                                processing_time_ms=0,
                            )
                        )
                return

            await asyncio.sleep(0.3)
            llm_text = _build_llm_mock(enriched_query, chunks)
            all_chunks = chunks
        else:
            toc_text = await registry_client.get_toc()
            system_prompt = _SYSTEM_PROMPT.replace("[TABLE_OF_CONTENTS]", toc_text) if toc_text else _SYSTEM_PROMPT.replace("[TABLE_OF_CONTENTS]\n\n", "").replace("[TABLE_OF_CONTENTS]", "")
            messages = [{"role": "system", "content": system_prompt}]
            if summary:
                messages.append({"role": "system", "content": f"Резюме предыдущего диалога:\n{summary}"})
            messages.extend(history)
            reminder = "Помни про системный промпт и обязательный формат ссылок на цитирование."
            messages.append({"role": "user", "content": f"{reminder}\n\nСообщение пользователя:\n{enriched_query}"})

            try:
                llm_text, all_chunks, prompt_tokens, completion_tokens = await _run_tool_loop(
                    messages, settings, _utcnow().strftime("%Y-%m-%d"), message_id, session_id, session_factory,
                )

                # Если после первого прохода нет chunks, принудительно вызываем rag_search
                if not all_chunks:
                    logger.warning("pipeline: no chunks after tool_loop, forcing rag_search", extra={"message_id": message_id})
                    await _set_status(session_factory, message_id, "analyzing", "Принудительный поиск...", progress=75)
                    messages.append({
                        "role": "user",
                        "content": "Ты не выполнил поиск по БД знаний. НЕМЕДЛЕННО вызови инструмент rag_search для поиска информации по запросу пользователя."
                    })
                    result = await llm_client.complete(messages, tools=_TOOLS, cache_key=session_id)
                    prompt_tokens += result.prompt_tokens
                    completion_tokens += result.completion_tokens

                    if result.tool_calls:
                        messages.append({
                            "role": "assistant",
                            "content": result.content,
                            "tool_calls": result.tool_calls,
                        })
                        for tc in result.tool_calls:
                            fn = tc["function"]["name"]
                            args = json.loads(tc["function"]["arguments"])
                            if fn == "rag_search":
                                instruction = args.get("instruction", "")
                                phrases = args.get("phrases")
                                search_type = args.get("search_type", "hybrid_rrf")
                                top_n = args.get("top_n", 15)
                                logger.info(
                                    "forced_rag_search instruction=%r phrases=%r search_type=%s top_n=%d",
                                    instruction, phrases, search_type, top_n,
                                    extra={"message_id": message_id},
                                )
                                await _set_status(session_factory, message_id, "searching", f"Принудительный поиск: {instruction[:60]}...", progress=78)
                                try:
                                    chunks = await asyncio.wait_for(
                                        rag_client.search(
                                            instruction=instruction, phrases=phrases,
                                            top_n=top_n, search_type=search_type,
                                            valid_at=_utcnow().strftime("%Y-%m-%d"),
                                        ),
                                        timeout=180.0,
                                    )
                                    start_idx = len(all_chunks)
                                    all_chunks.extend(chunks)
                                    tool_content = _format_chunks(chunks, start_idx) if chunks else "Релевантные фрагменты не найдены."
                                except Exception as exc:
                                    logger.error("forced_rag_search failed: %s", exc, extra={"message_id": message_id}, exc_info=True)
                                    tool_content = f"Ошибка поиска: {exc}"
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": tool_content,
                                })

                        # После принудительного вызова инструмента, запускаем wind-down для получения ответа
                        await _set_status(session_factory, message_id, "analyzing", "Формирование ответа...", progress=80)
                        _PROHIBITIONS = [
                            "На основе всей полученной информации напиши ответ пользователю. Не вызывай инструменты.",
                            "Ты исчерпал лимит вызовов инструментов. ОТВЕТЬ ТЕКСТОМ на основе собранных данных. Не вызывай инструменты.",
                            "ЗАПРЕЩЕНО вызывать инструменты. Напиши ответ прямо сейчас.",
                            "НЕМЕДЛЕННО ОТВЕТЬ ТЕКСТОМ. Любые вызовы инструментов будут проигнорированы.",
                            "ФИНАЛЬНОЕ ПРЕДУПРЕЖДЕНИЕ: ответь текстом на основе того, что уже собрано.",
                        ]
                        for attempt in range(5):
                            prog = min(80 + attempt * 2, 95)
                            await _set_status(session_factory, message_id, "analyzing", "Формирование ответа...", progress=prog)
                            result = await llm_client.complete(messages, tools=_TOOLS, tool_choice="none", cache_key=session_id)
                            prompt_tokens += result.prompt_tokens
                            completion_tokens += result.completion_tokens
                            if result.tool_calls:
                                logger.warning("pipeline: wind-down attempt=%d/5 still got tool_calls=%s", attempt + 1, [tc["function"]["name"] for tc in result.tool_calls], extra={"message_id": message_id})
                            elif result.content and not _is_garbage(result.content) and not result.content.strip().startswith("План:"):
                                llm_text = result.content
                                break
                            else:
                                if result.content:
                                    if _is_garbage(result.content):
                                        logger.warning("pipeline: wind-down attempt=%d/5 garbage content, retrying", attempt + 1, extra={"message_id": message_id})
                                    else:
                                        logger.warning("pipeline: wind-down attempt=%d/5 plan-only content", attempt + 1, extra={"message_id": message_id})
                                msg = _PROHIBITIONS[attempt]
                                logger.warning("pipeline: wind-down attempt=%d/5 adding prohibition", attempt + 1, extra={"message_id": message_id})
                                messages.append({"role": "user", "content": msg})
                        else:
                            llm_text = ""
                    else:
                        logger.warning("pipeline: forced rag_search still no tool_calls", extra={"message_id": message_id})
                        llm_text = "Не удалось выполнить поиск по БД знаний. Попробуйте уточнить запрос."

            except _DataExhausted:
                logger.warning("pipeline: data exhausted after 3 consecutive empty rag_search", extra={"message_id": message_id})
                async with session_factory() as db:
                    async with db.begin():
                        await db.execute(
                            update(ChatMessage)
                            .where(ChatMessage.message_id == message_id)
                            .values(
                                content="В базе знаний не найдено подтверждённых фрагментов по данному запросу.",
                                status="not_found",
                                message=None,
                                processing_time_ms=0,
                            )
                        )
                return
            except Exception as exc:
                logger.error("tool loop failed", extra={"message_id": message_id}, exc_info=True)
                error_msg = str(exc)
                if "LLM completion failed" in error_msg or "Server disconnected" in error_msg:
                    user_msg = "Сервис LLM временно недоступен. Попробуйте повторить запрос позже."
                else:
                    user_msg = "Внутренняя ошибка при обработке запроса. Попробуйте повторить."
                async with session_factory() as db:
                    async with db.begin():
                        await db.execute(
                            update(ChatMessage)
                            .where(ChatMessage.message_id == message_id)
                            .values(
                                content=user_msg,
                                status="failed",
                                processing_time_ms=0,
                            )
                        )
                return

            if not all_chunks or not llm_text.strip() or llm_text.strip().startswith("План:"):
                llm_text = "Не удалось найти решение за отведённое время. Попробуйте уточнить запрос."

            # Review: добавить запрос проверки в конец сообщений
            if llm_text != "Не удалось найти решение за отведённое время. Попробуйте уточнить запрос.":
                try:
                    messages.append({"role": "user", "content": _REVIEW_PROMPT})
                    review_result = await llm_client.complete(messages, tools=_TOOLS, cache_key=session_id)
                    if review_result.content and not _is_garbage(review_result.content):
                        review_text = review_result.content.strip()
                        if "изменений не требуется" in review_text or ("[source:" not in review_text and "[document_id:" not in review_text):
                            logger.info("review: no changes needed, keeping original", extra={"message_id": message_id})
                        else:
                            logger.info("review: answer improved, len=%d preview=%.200s",
                                        len(review_text), review_text.replace("\n", " ")[:200],
                                        extra={"message_id": message_id})
                            llm_text = review_text
                    else:
                        logger.info("review: no improvement, keeping original", extra={"message_id": message_id})
                except Exception:
                    logger.warning("review failed, keeping original", extra={"message_id": message_id}, exc_info=True)

        await _set_status(session_factory, message_id, "enriching_citations", "Обогащение цитат...", progress=90)

        logger.info("raw LLM text (pre‑enrich) len=%d preview=%.300s",
                     len(llm_text), llm_text.replace("\n", " ")[:300],
                     extra={"message_id": message_id})
        logger.info("all_chunks for enrichment: %s", [
            {"idx": i, "doc_id": c.document_id, "sec_id": c.section_id, "page": c.page}
            for i, c in enumerate(all_chunks)
        ], extra={"message_id": message_id})

        try:
            final_text, used_indices = await asyncio.wait_for(
                asyncio.to_thread(_enrich_citations, llm_text, all_chunks),
                timeout=180.0,
            )
        except Exception:
            warnings.append("Обогащение цитат недоступно.")
            logger.warning("citation enrichment skipped", extra={"message_id": message_id}, exc_info=True)
            final_text = llm_text
            used_indices = sorted({
                int(m.group(1)) for m in _SOURCE_REF_RE.finditer(_normalize_source_refs(llm_text))
                if int(m.group(1)) < len(all_chunks)
            })

        used_chunks_list = [all_chunks[i] for i in used_indices if i < len(all_chunks)]

        logger.info("used_indices=%s enriched_text len=%d preview=%.300s",
                     used_indices, len(final_text), final_text.replace("\n", " ")[:300],
                     extra={"message_id": message_id})
        logger.info("used_chunks mapping: %s", [
            {"idx": i, "doc_id": c.document_id, "sec_id": c.section_id, "page": c.page}
            for i, c in enumerate(used_chunks_list)
        ], extra={"message_id": message_id})

        final_status = "not_found" if not final_text.strip() else "answered"

        doc_refs = _parse_doc_id_refs(final_text)

        doc_map: dict[int, dict] = {}
        if used_chunks_list or doc_refs:
            unique_ids = sorted({c.document_id for c in used_chunks_list} | {ref["document_id"] for ref in doc_refs})
            if unique_ids:
                resp = await registry_client.get_documents(unique_ids)
                data = json.loads(resp).get("data", [])
                doc_map = {d["id"]: d for d in data}
                for doc in data:
                    if not doc.get("doc_code"):
                        logger.warning("doc_code is empty", extra={
                            "message_id": message_id, "document_id": doc.get("id"),
                            "title": doc.get("title", "")[:80],
                        })

        chunk_by_key: dict[tuple[int, int], rag_client.Chunk] = {}
        for chunk in used_chunks_list:
            chunk_by_key[(chunk.document_id, chunk.section_id)] = chunk

        logger.info("pipeline writing final answer", extra={
            "message_id": message_id, "final_text_len": len(final_text),
            "doc_refs": len(doc_refs), "status": final_status,
        })
        async with session_factory() as db:
            async with db.begin():
                processing_time_ms = int((_utcnow() - t_start).total_seconds() * 1000)
                result = await db.execute(
                    update(ChatMessage)
                    .where(ChatMessage.message_id == message_id)
                    .values(
                        status=final_status,
                        progress=100,
                        message=None,
                        processing_time_ms=processing_time_ms,
                        prompt_tokens=prompt_tokens or None,
                        completion_tokens=completion_tokens or None,
                        model_used=settings.LLM_MODEL,
                        enrichment_skipped=False,
                        warnings=warnings or None,
                    )
                )
                if result.rowcount == 0:
                    logger.warning("pipeline: message deleted before finish, skipping sources", extra={"message_id": message_id})
                    return
                key_to_citation: dict[tuple[int, int | None, int | None], int] = {}
                match_to_idx: dict[int, int] = {}
                for m in _DOC_ID_REF_RE.finditer(final_text):
                    doc_id = int(m.group(1))
                    sec_id = _REF_SECTION_ID_RE.search(m.group(0))
                    section_id_val = int(sec_id.group(1)) if sec_id else None
                    page_m = _REF_PAGE_RE.search(m.group(0))
                    page_val = int(page_m.group(1)) if page_m else None
                    key = (doc_id, section_id_val, page_val)

                    if key not in key_to_citation:
                        citation_index = len(key_to_citation) + 1
                        key_to_citation[key] = citation_index
                        before = final_text[:m.start()].rstrip()
                        before = re.sub(r'\[[^\]]*\]', '', before).strip()
                        before = re.sub(r'[\s.,;:!?)\]]+$', '', before)
                        before = re.sub(r'^[\s.,;:!?(]+', '', before)
                        before = re.sub(r'^\d+\.\s*', '', before)
                        ctx_words = before.split()[-10:]
                        ctx = ' '.join(ctx_words) if ctx_words else None
                        doc_info = doc_map.get(doc_id, {})
                        chunk = chunk_by_key.get((doc_id, section_id_val))
                        short_name = (doc_info.get("short_name") or "")[:64] or None
                        db.add(ChatSource(
                            message_id=message_id,
                            citation_index=citation_index,
                            chunk_id=chunk.chunk_id if chunk else None,
                            document_id=doc_id,
                            document_title=(doc_info.get("title") or "")[:256],
                            short_name=short_name,
                            section=ctx,
                            section_id=section_id_val,
                            page_number=page_val or (chunk.page if chunk else None),
                            clause=chunk.clause if chunk else None,
                            section_title=chunk.section_title if chunk else None,
                            excerpt=chunk.excerpt if chunk else None,
                            text=chunk.content if chunk else None,
                            score=chunk.score if chunk else None,
                            confidence=chunk.confidence if chunk else None,
                        ))
                    else:
                        citation_index = key_to_citation[key]
                        logger.info(
                            "duplicate citation (doc_id=%s sec_id=%s page=%s) → reusing citation_index=%s",
                            doc_id, section_id_val, page_val, citation_index,
                            extra={"message_id": message_id},
                        )
                    match_to_idx[m.start()] = citation_index

                # Replace [document_id:X, ...] with [source:N] for frontend display
                final_display = final_text
                for m in reversed(list(_DOC_ID_REF_RE.finditer(final_text))):
                    idx = match_to_idx[m.start()]
                    final_display = final_display[:m.start()] + f"[source:{idx}]" + final_display[m.end():]
                final_display = _convert_latex_delimiters(final_display)
                await db.execute(
                    update(ChatMessage)
                    .where(ChatMessage.message_id == message_id)
                    .values(content=final_display)
                )

        logger.info("pipeline finished", extra={"message_id": message_id, "chunks": len(all_chunks)})

    except BaseException:
        logger.error("pipeline crashed", extra={"message_id": message_id}, exc_info=True)
        async with session_factory() as db:
            async with db.begin():
                await db.execute(
                    update(ChatMessage)
                    .where(ChatMessage.message_id == message_id)
                    .values(
                        content="Внутренняя ошибка при обработке запроса. Попробуйте повторить.",
                        status="failed",
                        processing_time_ms=0,
                    )
                )
