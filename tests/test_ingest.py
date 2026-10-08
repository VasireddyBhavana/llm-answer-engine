import pytest

from grounded.ingest import chunk_document

DOC = """# Title

## Section A

First paragraph about apples.

Second paragraph about pears.

## Section B

""" + " ".join(f"word{i}" for i in range(250))


def test_chunks_never_span_sections():
    chunks = chunk_document("d", DOC, chunk_size=100, overlap=10)
    assert all(not ("apples" in c.text and "word1" in c.text) for c in chunks)
    assert {c.section for c in chunks} == {"Section A", "Section B"}


def test_long_paragraph_uses_sliding_window_with_overlap():
    chunks = [c for c in chunk_document("d", DOC, chunk_size=100, overlap=10)
              if c.section == "Section B"]
    assert len(chunks) == 3
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-10:] == second[-100:][:10]  # 10-word overlap between neighbours


def test_chunk_ids_and_headers():
    chunks = chunk_document("d", DOC, chunk_size=100, overlap=10)
    assert chunks[0].id == "d#0"
    assert chunks[0].text.startswith("[Title > Section A]")


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_document("d", DOC, chunk_size=10, overlap=10)
