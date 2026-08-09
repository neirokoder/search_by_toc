"""Оркестрация мини-ассистента: цикл вызовов инструментов по образцу example/pipeline.py."""

import json
import logging
import re

from openai import OpenAI

from archive import Archive
from config import Settings
from prompts import _REVIEW_PROMPT, _SYSTEM_PROMPT, _TOOLS

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERS = 25
_MAX_PAGES_PER_CALL = 10

_WIND_DOWN_PROHIBITIONS = [
    "На основе всей полученной информации напиши ответ пользователю. Не вызывай инструменты.",
    "Ты исчерпал лимит вызовов инструментов. ОТВЕТЬ ТЕКСТОМ на основе собранных данных. Не вызывай инструменты.",
    "ЗАПРЕЩЕНО вызывать инструменты. Напиши ответ прямо сейчас.",
    "НЕМЕДЛЕННО ОТВЕТЬ ТЕКСТОМ. Любые вызовы инструментов будут проигнорированы.",
    "ФИНАЛЬНОЕ ПРЕДУПРЕЖДЕНИЕ: ответь текстом на основе того, что уже собрано.",
]

_GENERIC_WORDS = {
    "что", "как", "где", "когда", "зачем", "почему", "какие", "какой", "какая",
    "этот", "эта", "это", "эти", "его", "её", "их", "наш", "ваш",
    "должен", "должна", "должны", "нужно", "необходимо", "может", "могут",
    "является", "являются", "имеет", "имеют", "все", "всё", "также",
    "определение", "понятие", "сущность",
    "рассказать", "расскажи", "покажи", "показать", "показывает",
    "назвать", "назови", "перечислить", "перечисли",
}

_PLAN_ONLY_RE = re.compile(r"^\s*(план|шаг|этап)\b", re.IGNORECASE)


def _is_garbage(text: str) -> bool:
    if not text:
        return True
    text = text.strip()
    if len(text) < 10:
        return False
    words = text.lower().split()
    if len(words) < 5:
        return False
    for w in set(words):
        run = 0
        for w2 in words:
            if w2 == w:
                run += 1
                if run >= 8:
                    return True
            else:
                run = 0
    if len(words) > 200 and len(set(words)) / len(words) < 0.05:
        return True
    return False


class MiniAssistant:
    def __init__(self, archive: Archive, settings: Settings):
        self.archive = archive
        self.settings = settings
        kwargs = {"api_key": settings.api_key or ("sk-local" if settings.base_url else "")}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self.llm = OpenAI(**kwargs)
        self.model = settings.model
        self.history: list[dict] = []

    def _system_prompt(self) -> str:
        docs_text = self.archive.list_text()
        return _SYSTEM_PROMPT.replace("[DOCUMENTS]", docs_text)

    def _call_tool(self, name: str, args: dict) -> str:
        if name == "list_documents":
            logger.info("tool_call list_documents")
            return self.archive.list_text()

        if name == "toc_navigate":
            doc = self.archive.get(int(args.get("document_id")))
            if doc is None:
                return "Документ с таким номером не найден. Вызови list_documents для получения перечня."
            node_id = args.get("node_id")
            logger.info("tool_call toc_navigate document_id=%s node_id=%s", doc.number, node_id)
            return doc.navigate(int(node_id) if node_id is not None else None,
                                max_nodes=self.settings.max_toc_nodes)

        if name == "get_pages_by_clause":
            doc = self.archive.get(int(args.get("document_id")))
            if doc is None:
                return "Документ с таким номером не найден. Вызови list_documents для получения перечня."
            clauses = [str(c) for c in (args.get("clauses") or [])]
            logger.info("tool_call get_pages_by_clause document_id=%s clauses=%s", doc.number, clauses)
            return doc.pages_by_clause(clauses)

        if name == "get_page_markdown":
            doc = self.archive.get(int(args.get("document_id")))
            if doc is None:
                return "Документ с таким номером не найден. Вызови list_documents для получения перечня."
            pages = [int(p) for p in (args.get("pages") or [])]
            if not pages:
                return "Не указаны номера страниц (pages)."
            pages = pages[: _MAX_PAGES_PER_CALL]
            logger.info("tool_call get_page_markdown document_id=%s pages=%s", doc.number, pages)
            return doc.get_pages(pages)

        return f"Неизвестный инструмент: {name}"

    def _run_tool_loop(self, messages: list[dict]) -> str:
        for i in range(_MAX_TOOL_ITERS):
            print(f"  ▸ итерация {i + 1}: анализ...", flush=True)
            response = self.llm.chat.completions.create(
                model=self.model, messages=messages, tools=_TOOLS
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                content = msg.content or ""
                if _is_garbage(content) or _PLAN_ONLY_RE.match(content):
                    messages.append({"role": "user",
                                     "content": "Сгенерирован некорректный ответ без вызова инструментов. Продолжай: вызови нужный инструмент или дай ответ."})
                    continue
                return content
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                print(f"  ▸ вызов {fn}({json.dumps(args, ensure_ascii=False)[:120]})", flush=True)
                tool_content = self._call_tool(fn, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_content,
                })
        return ""

    def _wind_down(self, messages: list[dict]) -> str:
        for attempt, prohibition in enumerate(_WIND_DOWN_PROHIBITIONS):
            print(f"  ▸ формирование ответа (попытка {attempt + 1}/{len(_WIND_DOWN_PROHIBITIONS)})...", flush=True)
            response = self.llm.chat.completions.create(
                model=self.model, messages=messages, tools=_TOOLS, tool_choice="none"
            )
            content = (response.choices[0].message.content or "").strip()
            if content and not _is_garbage(content) and not _PLAN_ONLY_RE.match(content):
                return content
            messages.append({"role": "user", "content": prohibition})
        return ""

    def _review(self, messages: list[dict], answer: str) -> str:
        if "в доступных документах не найдена информация" in answer.lower():
            return answer
        try:
            review_messages = list(messages) + [{"role": "user", "content": _REVIEW_PROMPT}]
            response = self.llm.chat.completions.create(
                model=self.model, messages=review_messages, tools=_TOOLS
            )
            content = (response.choices[0].message.content or "").strip()
            if (content and not _is_garbage(content)
                    and "изменений не требуется" not in content.lower()
                    and "[doc_id:" in content):
                return content
        except Exception:
            logger.warning("review failed, keeping original", exc_info=True)
        return answer

    def run(self, user_query: str) -> str:
        messages = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self.history)
        messages.append({
            "role": "user",
            "content": f"Сообщение пользователя:\n{user_query}",
        })
        answer = self._run_tool_loop(messages)
        if not answer:
            answer = self._wind_down(messages)
        if not answer:
            answer = "Не удалось найти решение за отведённое время. Попробуйте уточнить запрос."
        answer = self._review(messages, answer)
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})
        return answer
