import textwrap

from app.retrieval.chunker import GenericChunker, PythonChunker, get_chunker


def _dedent(source: str) -> str:
    return textwrap.dedent(source).strip("\n") + "\n"


def test_python_chunker_splits_top_level_functions():
    source = _dedent(
        """
        import os


        def add(a, b):
            return a + b


        def subtract(a, b):
            return a - b
        """
    )
    chunks = PythonChunker(max_lines=200, overlap_lines=20).chunk(source)

    symbols = {c.symbol for c in chunks}
    assert "add" in symbols
    assert "subtract" in symbols

    add_chunk = next(c for c in chunks if c.symbol == "add")
    assert "return a + b" in add_chunk.content
    assert "return a - b" not in add_chunk.content


def test_python_chunker_captures_module_level_leftover():
    source = _dedent(
        """
        import os

        API_VERSION = "v1"

        def handler():
            return API_VERSION
        """
    )
    chunks = PythonChunker(max_lines=200, overlap_lines=20).chunk(source)

    leftover = [c for c in chunks if c.symbol is None]
    assert any("API_VERSION" in c.content for c in leftover)
    assert any("import os" in c.content for c in leftover)


def test_python_chunker_keeps_decorator_with_function():
    source = _dedent(
        """
        @app.get("/health")
        def health():
            return {"status": "ok"}
        """
    )
    chunks = PythonChunker(max_lines=200, overlap_lines=20).chunk(source)

    health_chunk = next(c for c in chunks if c.symbol == "health")
    assert '@app.get("/health")' in health_chunk.content


def test_python_chunker_chunks_small_class_as_one_unit():
    source = _dedent(
        """
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        """
    )
    chunks = PythonChunker(max_lines=200, overlap_lines=20).chunk(source)

    assert any(c.symbol == "Point" for c in chunks)


def test_python_chunker_splits_large_class_into_methods():
    methods = "\n\n".join(
        f"    def method_{i}(self):\n" + "\n".join(f"        x{j} = {j}" for j in range(15))
        for i in range(10)
    )
    source = f"class Big:\n{methods}\n"

    chunks = PythonChunker(max_lines=50, overlap_lines=5).chunk(source)

    method_symbols = {c.symbol for c in chunks if c.symbol and c.symbol.startswith("Big.method_")}
    assert len(method_symbols) == 10


def test_python_chunker_falls_back_on_syntax_error():
    chunks = PythonChunker(max_lines=200, overlap_lines=20).chunk("def broken(:\n    pass\n")
    assert len(chunks) == 1
    assert chunks[0].symbol is None


def test_generic_chunker_windows_with_overlap():
    lines = [f"line {i}" for i in range(1, 251)]
    source = "\n".join(lines)

    chunks = GenericChunker(max_lines=100, overlap_lines=20).chunk(source)

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 100
    assert chunks[1].start_line == 81  # 100 - 20 overlap + 1
    assert chunks[-1].end_line == 250


def test_generic_chunker_handles_short_file():
    chunks = GenericChunker(max_lines=100, overlap_lines=20).chunk("a\nb\nc\n")
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3


def test_get_chunker_dispatches_by_language():
    assert isinstance(get_chunker("python", 200, 20), PythonChunker)
    assert isinstance(get_chunker("javascript", 200, 20), GenericChunker)
    assert isinstance(get_chunker("unknown-language", 200, 20), GenericChunker)
