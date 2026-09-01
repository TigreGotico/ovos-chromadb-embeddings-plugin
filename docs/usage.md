# Usage

All examples assume a `ChromaEmbeddingsDB` instance called `db`.
Embeddings are `numpy.ndarray` of `float32`. All vectors in a collection must share the same dimensionality.

## Initialisation

```python
import tempfile
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

tmp = tempfile.mkdtemp()
db = ChromaEmbeddingsDB(config={"path": tmp})
```

On init the default collection (key: `default_collection_name`) is created if absent.

---

## Collections

Collections are independent namespaces for vectors. Think of them as tables.

```python
# Create
db.create_collection("faces")
db.create_collection("voices", metadata={"hnsw:space": "l2"})

# List
for col in db.list_collections():
    print(col.name)

# Get handle
col = db.get_collection("faces")  # raises ValueError if absent

# Delete
db.delete_collection("faces")
```

---

## Storing embeddings

### Single upsert

```python
import numpy as np

vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
db.add_embeddings("user:42", vec, metadata={"name": "Alice"})
```

- `key` must be unique within the collection. Calling `add_embeddings` with an existing key **updates** the vector.
- `metadata` values must be `str`, `int`, or `float` (ChromaDB limitation).

### Batch upsert

```python
keys = ["item:1", "item:2", "item:3"]
vecs = [np.random.rand(4).astype(np.float32) for _ in range(3)]
meta = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}]

db.add_embeddings_batch(keys, vecs, metadata=meta)
```

### Targeting a non-default collection

Pass `collection_name` to any method:

```python
db.add_embeddings("speaker:7", vec, collection_name="voices")
```

---

## Retrieving embeddings

```python
# Returns np.ndarray or None
emb = db.get_embeddings("user:42")

# Also return metadata
emb, meta = db.get_embeddings("user:42", return_metadata=True)

# Batch retrieval: returns list of (key, embedding) or (key, embedding, metadata)
results = db.get_embeddings_batch(["item:1", "item:2"], return_metadata=True)
for key, emb, meta in results:
    print(key, emb.shape, meta)
```

---

## Querying (nearest neighbour search)

```python
query_vec = np.array([0.12, 0.22, 0.31, 0.41], dtype=np.float32)

# Returns list of (id, distance)
hits = db.query(query_vec, top_k=5)
best_id, best_dist = hits[0]

# Include metadata
hits = db.query(query_vec, top_k=5, return_metadata=True)
for id_, dist, meta in hits:
    print(id_, dist, meta)
```

Distance values depend on `hnsw:space`:
- `cosine`: 0 = identical direction, 2 = opposite
- `l2`: squared Euclidean distance
- `ip`: negated inner product (lower = more similar)

---

## Deleting embeddings

```python
# Single
db.delete_embeddings("user:42")

# Batch
db.delete_embeddings_batch(["item:1", "item:2"])
```

---

## Counting

```python
n = db.count_embeddings_in_collection()               # default collection
n = db.count_embeddings_in_collection("voices")       # named collection
```

---

## Numpy in / out

All embedding inputs accept `np.ndarray` or plain Python `list[float]`.
All outputs are `np.ndarray` (converted from ChromaDB's internal list representation).

---
[← Configuration](configuration.md) · [Home](../README.md)
