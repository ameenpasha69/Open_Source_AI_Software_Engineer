import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RawChunk:
    symbol: str | None
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    content: str


class Chunker(ABC):
    """Strategy interface for splitting one file's source into RawChunks.

    Only Python gets true AST-based chunking today. Other languages use
    GenericChunker's fixed-window fallback. A tree-sitter based chunker for
    JS/TS/Go/etc. would slot in here as another Chunker implementation
    without changing anything upstream (file_walker, indexer, DB schema).
    """

    @abstractmethod
    def chunk(self, source: str) -> list[RawChunk]: ...


class GenericChunker(Chunker):
    """Fixed-size sliding window over lines, with overlap. Language-agnostic;
    used for every extension without a dedicated chunker."""

    def __init__(self, max_lines: int, overlap_lines: int):
        self._max_lines = max_lines
        self._overlap_lines = min(overlap_lines, max_lines - 1)

    def chunk(self, source: str) -> list[RawChunk]:
        lines = source.splitlines()
        if not lines:
            return []
        return _window_lines(lines, start_offset=0, max_lines=self._max_lines, overlap_lines=self._overlap_lines)


def _window_lines(
    lines: list[str], *, start_offset: int, max_lines: int, overlap_lines: int
) -> list[RawChunk]:
    """Slide a window over `lines` (already sliced to the range of interest).
    `start_offset` is the 0-indexed line number of lines[0] in the original file,
    so returned line numbers stay correct relative to the whole file."""
    step = max_lines - overlap_lines
    chunks: list[RawChunk] = []
    i = 0
    n = len(lines)
    while i < n:
        window = lines[i : i + max_lines]
        chunks.append(
            RawChunk(
                symbol=None,
                start_line=start_offset + i + 1,
                end_line=start_offset + i + len(window),
                content="\n".join(window),
            )
        )
        if i + max_lines >= n:
            break
        i += step
    return chunks


class PythonChunker(Chunker):
    """AST-based: one chunk per top-level function, one per class (whole class
    if compact, otherwise one chunk per method), and any remaining module-level
    code (imports, constants, script logic) windowed separately so nothing is
    dropped. Oversized functions/classes are further split by GenericChunker
    so no single chunk becomes unbounded.
    """

    def __init__(self, max_lines: int, overlap_lines: int):
        self._max_lines = max_lines
        self._fallback = GenericChunker(max_lines, overlap_lines)
        # A class is only kept as a single chunk if it's small — otherwise even a
        # class well under `max_lines` (e.g. 100 lines / 4 methods) would swallow
        # multiple unrelated methods into one chunk, hurting retrieval precision.
        # Above this, always split by method.
        self._small_class_max_lines = max(20, max_lines // 4)

    def chunk(self, source: str) -> list[RawChunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._fallback.chunk(source)

        lines = source.splitlines()
        chunks: list[RawChunk] = []
        covered = [False] * (len(lines) + 1)  # 1-indexed

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunks.extend(self._chunk_class(node, lines, covered))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.extend(
                    self._chunk_span(node.name, _decorated_start(node), node.end_lineno, lines, covered)
                )

        chunks.extend(self._chunk_leftover(lines, covered))
        return sorted(chunks, key=lambda c: c.start_line)

    def _chunk_class(
        self, node: ast.ClassDef, lines: list[str], covered: list[bool]
    ) -> list[RawChunk]:
        methods = [
            child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        class_start = _decorated_start(node)
        span = (node.end_lineno or node.lineno) - class_start + 1
        if not methods or span <= self._small_class_max_lines:
            return self._chunk_span(node.name, class_start, node.end_lineno, lines, covered)

        chunks: list[RawChunk] = []
        for method in methods:
            chunks.extend(
                self._chunk_span(
                    f"{node.name}.{method.name}",
                    _decorated_start(method),
                    method.end_lineno,
                    lines,
                    covered,
                )
            )
        # The `class Foo:` line, docstring, and any class-level attributes are
        # intentionally left uncovered here — the leftover pass below picks
        # them up as their own small chunk.
        return chunks

    def _chunk_span(
        self, symbol: str, start_line: int, end_line: int | None, lines: list[str], covered: list[bool]
    ) -> list[RawChunk]:
        end_line = end_line or start_line
        for ln in range(start_line, end_line + 1):
            covered[ln] = True

        span_lines = lines[start_line - 1 : end_line]
        if len(span_lines) <= self._max_lines:
            return [
                RawChunk(
                    symbol=symbol,
                    start_line=start_line,
                    end_line=end_line,
                    content="\n".join(span_lines),
                )
            ]

        parts = _window_lines(
            span_lines, start_offset=start_line - 1, max_lines=self._max_lines, overlap_lines=0
        )
        return [
            RawChunk(
                symbol=f"{symbol}#part{i + 1}" if len(parts) > 1 else symbol,
                start_line=p.start_line,
                end_line=p.end_line,
                content=p.content,
            )
            for i, p in enumerate(parts)
        ]

    def _chunk_leftover(self, lines: list[str], covered: list[bool]) -> list[RawChunk]:
        """Module-level code not inside any indexed function/class: imports,
        constants, `if __name__ == "__main__"` blocks, etc. Grouped into
        contiguous uncovered runs and windowed like any generic file."""
        chunks: list[RawChunk] = []
        run_start: int | None = None
        for ln in range(1, len(lines) + 1):
            is_blank = not lines[ln - 1].strip()
            if not covered[ln] and not is_blank:
                if run_start is None:
                    run_start = ln
            else:
                if run_start is not None:
                    chunks.extend(self._window_leftover(lines, run_start, ln - 1))
                    run_start = None
        if run_start is not None:
            chunks.extend(self._window_leftover(lines, run_start, len(lines)))
        return chunks

    def _window_leftover(self, lines: list[str], start_line: int, end_line: int) -> list[RawChunk]:
        span_lines = lines[start_line - 1 : end_line]
        return _window_lines(span_lines, start_offset=start_line - 1, max_lines=self._max_lines, overlap_lines=0)


def _decorated_start(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """A decorated function/class's real start is its first decorator line,
    not the `def`/`class` line — keeps e.g. `@app.get(...)` attached to its handler."""
    if node.decorator_list:
        return min(d.lineno for d in node.decorator_list)
    return node.lineno


def get_chunker(language: str, max_lines: int, overlap_lines: int) -> Chunker:
    if language == "python":
        return PythonChunker(max_lines, overlap_lines)
    return GenericChunker(max_lines, overlap_lines)
