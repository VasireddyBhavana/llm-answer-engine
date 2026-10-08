import numpy as np

from grounded.embeddings import HashingEmbedder
from grounded.ingest import Chunk
from grounded.retrieval import BM25Index, Hit, Retriever, reciprocal_rank_fusion


def mk(i: str, text: str, doc: str | None = None) -> Chunk:
    return Chunk(i, doc or i, i, "", text)


CHUNKS = [
    mk("a", "BM25 ranks documents by exact keyword overlap and term rarity."),
    mk("b", "Embeddings capture meaning so paraphrases land close together."),
    mk("c", "Error code E1234 appears when the disk is full."),
]


def test_bm25_prefers_exact_rare_terms():
    hits = BM25Index(CHUNKS).search("what does E1234 mean", k=3)
    assert hits[0].chunk.id == "c"


def test_bm25_ignores_unknown_terms():
    assert BM25Index(CHUNKS).search("zzzz qqqq", k=3) == []


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(dim=256)
    v1, v2 = e.embed(["hello world"]), e.embed(["hello world"])
    assert np.allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1), 1.0)


def test_rrf_rewards_agreement_between_lists():
    a, b, c = (mk(x, x) for x in "abc")
    fused = reciprocal_rank_fusion([[Hit(a, 1, "x"), Hit(b, 1, "x")],
                                    [Hit(b, 1, "y"), Hit(c, 1, "y")]])
    assert fused[0].chunk.id == "b"  # in both lists
    assert abs(fused[0].score - (1 / 62 + 1 / 61)) < 1e-9


def test_personalization_boost_reorders():
    r = Retriever([mk("x1", "vector search basics", "x"), mk("y1", "vector search basics", "y")],
                  HashingEmbedder())
    top = r.search("vector search", k=1, mode="bm25", boost_docs={"y": 2.0})
    assert top[0].chunk.doc_id == "y"


def test_mmr_avoids_duplicates():
    dup = "reciprocal rank fusion merges ranked lists"
    r = Retriever([mk("d1", dup, "d"), mk("d2", dup, "d"),
                   mk("e1", "reciprocal rank fusion uses k equals sixty", "e")],
                  HashingEmbedder())
    ids = [h.chunk.id for h in r.search("reciprocal rank fusion", k=2, use_mmr=True)]
    assert "e1" in ids


def test_save_and_load_roundtrip(tmp_path):
    e = HashingEmbedder()
    Retriever(CHUNKS, e).save(tmp_path)
    loaded = Retriever.load(tmp_path, e)
    assert [c.id for c in loaded.chunks] == ["a", "b", "c"]
    assert loaded.search("E1234", k=1, mode="bm25")[0].chunk.id == "c"
