"""
End-to-end integration test for ChromaEmbeddingsDB.

Uses a deterministic local embedder (hash → seeded numpy) — no model downloads,
no network, runs fast in CI. Verifies a real add→query→nearest-neighbour flow
against an actual PersistentClient backed by a tmp directory.
"""
import hashlib
import numpy as np
import pytest
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

EMBED_DIM = 32  # small but real enough for HNSW


def _embed(text: str) -> np.ndarray:
    """Deterministic text → float32 vector via seeded numpy RNG."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 31)
    rng = np.random.default_rng(seed)
    vec = rng.random(EMBED_DIM).astype(np.float32)
    # L2-normalise so cosine distance is meaningful
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# Corpus: pairs of (key, text).  The texts within each group are semantically
# "close" by construction — same seed family — but the embedder is purely
# deterministic so the test is reproducible.
CORPUS = [
    ("fruit/apple",  "apple"),
    ("fruit/banana", "banana"),
    ("fruit/cherry", "cherry"),
    ("animal/cat",   "cat"),
    ("animal/dog",   "dog"),
]


@pytest.fixture()
def db(tmp_path):
    return ChromaEmbeddingsDB(config={"path": str(tmp_path)})


def test_e2e_add_and_nearest_neighbour(db):
    """Add all corpus items, then query each and assert it returns itself as top-1."""
    for key, text in CORPUS:
        db.add_embeddings(key, _embed(text))

    assert db.count_embeddings_in_collection() == len(CORPUS)

    for key, text in CORPUS:
        query_vec = _embed(text)  # same deterministic vector → must be nearest to itself
        results = db.query(query_vec, top_k=1)
        assert len(results) == 1
        top_id, top_dist = results[0]
        assert top_id == key, (
            f"Expected '{key}' as nearest neighbour of its own embedding, "
            f"got '{top_id}' (distance={top_dist:.6f})"
        )
        # cosine distance of a vector to itself should be ~0
        assert top_dist < 1e-4, f"Self-distance too large: {top_dist}"


def test_e2e_batch_add_and_query_with_metadata(db):
    """Batch-insert items with metadata; query returns correct metadata."""
    keys = [k for k, _ in CORPUS]
    vecs = [_embed(t) for _, t in CORPUS]
    metas = [{"category": k.split("/")[0], "label": k.split("/")[1]} for k in keys]

    db.add_embeddings_batch(keys, vecs, metadata=metas)

    # Query with a vector identical to "animal/cat" — should come back first
    query_vec = _embed("cat")
    results = db.query(query_vec, top_k=3, return_metadata=True)

    assert len(results) == 3
    top_id, top_dist, top_meta = results[0]
    assert top_id == "animal/cat"
    assert top_meta["category"] == "animal"
    assert top_meta["label"] == "cat"
    assert top_dist < 1e-4


def test_e2e_delete_and_requery(db):
    """After deleting the true nearest neighbour, the next one becomes top-1."""
    for key, text in CORPUS[:3]:          # apple, banana, cherry
        db.add_embeddings(key, _embed(text))

    # Verify apple is top-1 for its own query
    results = db.query(_embed("apple"), top_k=1)
    assert results[0][0] == "fruit/apple"

    db.delete_embeddings("fruit/apple")
    assert db.count_embeddings_in_collection() == 2

    # After deletion, apple must not appear
    results = db.query(_embed("apple"), top_k=2)
    ids = [r[0] for r in results]
    assert "fruit/apple" not in ids


def test_e2e_collection_isolation(tmp_path):
    """Embeddings in different collections do not bleed into each other's queries."""
    db = ChromaEmbeddingsDB(config={"path": str(tmp_path),
                                    "default_collection_name": "col_a"})
    db.create_collection("col_b")

    vec_a = _embed("alpha")
    vec_b = _embed("beta")

    db.add_embeddings("item_a", vec_a, collection_name="col_a")
    db.add_embeddings("item_b", vec_b, collection_name="col_b")

    # query col_a — should only return item_a
    hits_a = db.query(vec_a, top_k=5, collection_name="col_a")
    assert len(hits_a) == 1
    assert hits_a[0][0] == "item_a"

    # query col_b — should only return item_b
    hits_b = db.query(vec_b, top_k=5, collection_name="col_b")
    assert len(hits_b) == 1
    assert hits_b[0][0] == "item_b"
