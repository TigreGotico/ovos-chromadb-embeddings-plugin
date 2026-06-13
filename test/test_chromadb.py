"""Tests for ChromaEmbeddingsDB — network-free, uses PersistentClient with tmp_path."""
import numpy as np
import pytest

from ovos_chromadb_embeddings import ChromaEmbeddingsDB


def make_db(tmp_path):
    """Helper: fresh DB at a tmp directory."""
    return ChromaEmbeddingsDB(config={"path": str(tmp_path)})


# ---------------------------------------------------------------------------
# Basic init
# ---------------------------------------------------------------------------

def test_init_creates_default_collection(tmp_path):
    db = make_db(tmp_path)
    names = [c.name for c in db.list_collections()]
    assert db.default_collection_name in names


# ---------------------------------------------------------------------------
# add / get single embedding
# ---------------------------------------------------------------------------

def test_add_get_embeddings_roundtrip(tmp_path):
    db = make_db(tmp_path)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    db.add_embeddings("k1", vec)
    result = db.get_embeddings("k1")
    assert result is not None
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, vec, atol=1e-5)


def test_get_embeddings_missing_key_returns_none(tmp_path):
    db = make_db(tmp_path)
    assert db.get_embeddings("nonexistent") is None


def test_get_embeddings_with_return_metadata(tmp_path):
    db = make_db(tmp_path)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    meta = {"label": "foo", "score": 42}
    db.add_embeddings("k1", vec, metadata=meta)
    emb, returned_meta = db.get_embeddings("k1", return_metadata=True)
    assert isinstance(emb, np.ndarray)
    np.testing.assert_allclose(emb, vec, atol=1e-5)
    assert returned_meta["label"] == "foo"
    assert returned_meta["score"] == 42


def test_get_embeddings_missing_key_with_metadata_returns_none_tuple(tmp_path):
    db = make_db(tmp_path)
    result = db.get_embeddings("missing", return_metadata=True)
    assert result == (None, None)


# ---------------------------------------------------------------------------
# batch add / get
# ---------------------------------------------------------------------------

def test_add_get_batch(tmp_path):
    db = make_db(tmp_path)
    keys = ["b1", "b2", "b3"]
    vecs = [np.array([float(i), float(i + 1), float(i + 2)]) for i in range(3)]
    db.add_embeddings_batch(keys, vecs)
    results = db.get_embeddings_batch(keys)
    result_keys = [r[0] for r in results]
    for k in keys:
        assert k in result_keys


def test_add_get_batch_with_metadata(tmp_path):
    db = make_db(tmp_path)
    keys = ["b1", "b2"]
    vecs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    metas = [{"tag": "alpha"}, {"tag": "beta"}]
    db.add_embeddings_batch(keys, vecs, metadata=metas)
    results = db.get_embeddings_batch(keys, return_metadata=True)
    assert len(results) == 2
    for key, emb, meta in results:
        assert isinstance(emb, np.ndarray)
        assert "tag" in meta


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def test_query_returns_top_k(tmp_path):
    db = make_db(tmp_path)
    for i in range(5):
        db.add_embeddings(f"item{i}", np.array([float(i), float(i), float(i)]))
    query_vec = np.array([0.0, 0.0, 0.0])
    results = db.query(query_vec, top_k=3)
    assert len(results) == 3
    for item in results:
        assert len(item) == 2  # (id, distance)


def test_query_with_metadata(tmp_path):
    db = make_db(tmp_path)
    for i in range(3):
        db.add_embeddings(f"item{i}", np.array([float(i), float(i), float(i)]),
                          metadata={"index": i})
    query_vec = np.array([1.0, 1.0, 1.0])
    results = db.query(query_vec, top_k=2, return_metadata=True)
    assert len(results) == 2
    for item in results:
        assert len(item) == 3  # (id, distance, metadata)
        assert "index" in item[2]


# ---------------------------------------------------------------------------
# collection lifecycle
# ---------------------------------------------------------------------------

def test_create_list_get_delete_collection(tmp_path):
    db = make_db(tmp_path)
    db.create_collection("mycol")
    names = [c.name for c in db.list_collections()]
    assert "mycol" in names

    col = db.get_collection("mycol")
    assert col.name == "mycol"

    db.delete_collection("mycol")
    names_after = [c.name for c in db.list_collections()]
    assert "mycol" not in names_after


def test_get_collection_missing_raises_value_error(tmp_path):
    db = make_db(tmp_path)
    with pytest.raises(ValueError):
        db.get_collection("does_not_exist")


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def test_count_embeddings_in_collection(tmp_path):
    db = make_db(tmp_path)
    assert db.count_embeddings_in_collection() == 0
    db.add_embeddings("x", np.array([1.0, 0.0, 0.0]))
    db.add_embeddings("y", np.array([0.0, 1.0, 0.0]))
    assert db.count_embeddings_in_collection() == 2


# ---------------------------------------------------------------------------
# delete single + batch
# ---------------------------------------------------------------------------

def test_delete_embeddings(tmp_path):
    db = make_db(tmp_path)
    db.add_embeddings("del1", np.array([1.0, 2.0, 3.0]))
    db.delete_embeddings("del1")
    assert db.get_embeddings("del1") is None


def test_delete_embeddings_batch(tmp_path):
    db = make_db(tmp_path)
    keys = ["d1", "d2", "d3"]
    for k in keys:
        db.add_embeddings(k, np.array([1.0, 2.0, 3.0]))
    db.delete_embeddings_batch(["d1", "d2"])
    assert db.get_embeddings("d1") is None
    assert db.get_embeddings("d2") is None
    assert db.get_embeddings("d3") is not None
