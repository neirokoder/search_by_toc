"""Доступ к архивам документов: метаданные, оглавление (toc), страницы.

Архив: zip с toc.json (дерево заголовков с уровнями и привязкой к страницам),
metadata.json (код, название) и pages/NNN.md (текст страниц).
"""

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

_NUMBER_RE = re.compile(r"-(\d+)\.zip$")
_PART_TITLE_RE = re.compile(r"«([^»]+)»")


@dataclass
class TocNode:
    title: str
    clause: str
    level: int
    page_start: int
    page_end: int
    node_id: int = 0
    children: list["TocNode"] = field(default_factory=list)


def _dedup_nodes(nodes: list[dict]) -> list[TocNode]:
    """Сворачивает дубликаты заголовков: соседние узлы с одинаковыми title+level
    сливаются в один (диапазон страниц расширяется, дети объединяются)."""
    out: list[TocNode] = []
    for n in nodes:
        title = (n.get("title") or "").strip()
        clause = (n.get("clause") or "").strip()
        level = int(n.get("level") or 1)
        start = int(n.get("page_start") or 0)
        end = int(n.get("page_end") or 0)
        children = _dedup_nodes(n.get("children") or [])
        if out and out[-1].title == title and out[-1].level == level:
            prev = out[-1]
            prev.page_start = min(prev.page_start, start)
            prev.page_end = max(prev.page_end, end)
            prev.children = _merge_children(prev.children, children)
        else:
            out.append(TocNode(title=title, clause=clause, level=level, page_start=start,
                               page_end=max(end, start), children=children))
    return out


def _merge_children(a: list[TocNode], b: list[TocNode]) -> list[TocNode]:
    if not b:
        return a
    if not a:
        return b
    if a[-1].title == b[0].title and a[-1].level == b[0].level:
        a[-1].page_start = min(a[-1].page_start, b[0].page_start)
        a[-1].page_end = max(a[-1].page_end, b[0].page_end)
        a[-1].children = _merge_children(a[-1].children, b[0].children)
        return a[:-1] + [a[-1]] + b[1:]
    return a + b


def _assign_ids(nodes: list[TocNode], counter: list[int]) -> None:
    for n in nodes:
        counter[0] += 1
        n.node_id = counter[0]
        _assign_ids(n.children, counter)


def _flatten(nodes: list[TocNode]) -> list[TocNode]:
    out: list[TocNode] = []
    for n in nodes:
        out.append(n)
        out.extend(_flatten(n.children))
    return out


def _walk_parents(nodes: list[TocNode], stack: list[TocNode | None], parents: dict[int, TocNode]) -> None:
    for n in nodes:
        stack.append(n)
        for c in n.children:
            parents[c.node_id] = n
            _walk_parents([c], stack, parents)
        stack.pop()


_PER_CLAUSE_MAX_PAGES = 3


class Document:
    def __init__(self, number: int, zip_path: Path):
        self.number = number
        self.zip_path = zip_path
        self._meta: dict | None = None
        self._tree: list[TocNode] | None = None
        self._flat: list[TocNode] | None = None
        self._pages_raw: int | None = None
        self._parent: dict[int, TocNode] | None = None

    def __repr__(self) -> str:
        return f"Document({self.number}, {self.doc_code})"

    @property
    def meta(self) -> dict:
        if self._meta is None:
            with zipfile.ZipFile(self.zip_path) as zf:
                self._meta = json.loads(zf.read("metadata.json"))
        return self._meta

    @property
    def doc_code(self) -> str:
        return self.meta.get("doc_code") or self.zip_path.stem

    @property
    def title(self) -> str:
        """Короткое название части (текст в кавычках из metadata.title)."""
        raw = self.meta.get("title") or ""
        m = _PART_TITLE_RE.search(raw)
        return m.group(1).strip() if m else raw[:120].strip()

    @property
    def pages_count(self) -> int:
        return self._pages_count_raw or max(p.page_end for p in self.flat_nodes())

    @property
    def _pages_count_raw(self) -> int:
        if self._pages_raw is None:
            with zipfile.ZipFile(self.zip_path) as zf:
                toc = json.loads(zf.read("toc.json"))
            self._pages_raw = int(toc.get("pages_count") or 0)
        return self._pages_raw

    def tree(self) -> list[TocNode]:
        if self._tree is None:
            with zipfile.ZipFile(self.zip_path) as zf:
                toc = json.loads(zf.read("toc.json"))
            self._tree = _dedup_nodes(toc.get("tree") or [])
            _assign_ids(self._tree, [0])
            parents: dict[int, TocNode] = {}
            stack = [None]
            _walk_parents(self._tree, stack, parents)
            self._parent = parents
        return self._tree

    def flat_nodes(self) -> list[TocNode]:
        if self._flat is None:
            self._flat = _flatten(self.tree())
        return self._flat

    def _section_path(self, node: TocNode) -> str:
        """Название раздела (уровень 1), в котором находится узел."""
        parts = []
        cur = node
        seen = 0
        while cur is not None and seen < 100:
            if cur.level == 1:
                parts.append(cur.title)
                break
            cur = self._parent.get(cur.node_id)
            seen += 1
        return parts[0] if parts else ""

    def _find_clause(self, clause: str) -> list[TocNode]:
        return [n for n in self.flat_nodes() if n.clause == clause]

    def get_page(self, page: int) -> str | None:
        name = f"pages/{page:03d}.md"
        with zipfile.ZipFile(self.zip_path) as zf:
            if name not in zf.namelist():
                return None
            return zf.read(name).decode("utf-8", errors="replace")

    def get_pages(self, pages: list[int]) -> str:
        """Форматированный вывод страниц: текст каждой страницы с пометкой."""
        parts = []
        for p in pages:
            content = self.get_page(p)
            if content is None:
                parts.append(f"===== Документ №{self.number}, стр. {p}: страница не найдена =====")
            else:
                parts.append(f"===== Документ №{self.number}, стр. {p} =====\n{content}")
        return "\n\n".join(parts)

    def navigate(self, node_id: int | None, max_nodes: int = 60) -> str:
        """Форматированное содержание: верхний уровень или дети узла node_id.
        Каждая строка содержит номер пункта (clause), уровень и страницы."""
        if node_id is None:
            nodes = self.tree()
            prefix = f"Содержание документа №{self.number} «{self.title}» ({self.pages_count} стр.).\n"
            return prefix + self._format_nodes(nodes, max_nodes)
        node = next((n for n in self.flat_nodes() if n.node_id == node_id), None)
        if node is None:
            return f"Узел id={node_id} не найден. Вызови навигацию без node_id для просмотра верхнего уровня."
        prefix = self._format_node(node) + "\n"
        if not node.children:
            return prefix + "Вложенных подразделов нет."
        return prefix + self._format_nodes(node.children, max_nodes, indent="  ")

    @staticmethod
    def _format_node(n: TocNode) -> str:
        rng = f"{n.page_start}–{n.page_end}" if n.page_end > n.page_start else str(n.page_start)
        clause = f"{n.clause} " if n.clause else ""
        marker = " ▸" if n.children else ""
        return f"id={n.node_id} {clause}{n.title} — стр. {rng}{marker}"

    @staticmethod
    def _format_nodes(nodes: list[TocNode], max_nodes: int, indent: str = "") -> str:
        lines = []
        for i, n in enumerate(nodes):
            if i >= max_nodes:
                lines.append(f"{indent}... и ещё {len(nodes) - max_nodes} узлов (укажи конкретный id для раскрытия)")
                break
            lines.append(indent + Document._format_node(n))
        return "\n".join(lines)

    def pages_by_clause(self, clauses: list[str]) -> str:
        """Страницы, содержащие пункты с указанными номерами (clause).
        Номера пунктов могут повторяться в разных разделах документа — каждое
        совпадение выводится отдельно с указанием раздела. На пункт выводятся
        не более _PER_CLAUSE_MAX_PAGES страниц. Возвращает описание найденных
        пунктов и текст их страниц."""
        clauses = [c.strip() for c in clauses if c and c.strip()]
        if not clauses:
            return "Не указаны номера пунктов (clauses)."
        header: list[str] = []
        pages: set[int] = set()
        for c in clauses:
            matches = self._find_clause(c)
            if not matches:
                header.append(f"Пункт {c} не найден в содержании документа №{self.number}.")
                continue
            for n in matches:
                span = n.page_end - n.page_start + 1
                shown_end = min(n.page_end, n.page_start + _PER_CLAUSE_MAX_PAGES - 1)
                rng = f"{n.page_start}–{shown_end}" if shown_end > n.page_start else str(n.page_start)
                section = self._section_path(n)
                where = f" (раздел «{section}»)" if section else ""
                note = f"; всего стр. {span}, показаны первые {_PER_CLAUSE_MAX_PAGES}" if span > _PER_CLAUSE_MAX_PAGES else ""
                header.append(f"п.{c} «{n.title}» — стр. {rng}{where}{note}")
                pages.update(range(n.page_start, shown_end + 1))
        if not pages:
            return "\n".join(header)
        body = self.get_pages(sorted(pages))
        return "\n".join(header) + "\n\n" + body


class Archive:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._docs: dict[int, Document] = {}

    def documents(self) -> list[Document]:
        if not self._docs:
            docs = []
            for p in sorted(self.data_dir.glob("*.zip")):
                m = _NUMBER_RE.search(p.name)
                if m:
                    docs.append(Document(int(m.group(1)), p))
            docs.sort(key=lambda d: d.number)
            self._docs = {d.number: d for d in docs}
        return list(self._docs.values())

    def get(self, number: int) -> Document | None:
        self.documents()
        return self._docs.get(number)

    def list_text(self) -> str:
        lines = []
        for d in self.documents():
            lines.append(f"№{d.number} — {d.doc_code} «{d.title}» ({d.pages_count} стр.)")
        return "\n".join(lines)
