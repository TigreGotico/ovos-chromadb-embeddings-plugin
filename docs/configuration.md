# Configuration

`ChromaEmbeddingsDB` accepts a `config` dict passed directly to the constructor,
or read from the OVOS configuration under its plugin key.

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `"./chromadb_storage"` | Filesystem path for local persistence (PersistentClient). Created if absent. |
| `host` | `str` | — | Hostname of a remote ChromaDB server. Presence of this key switches to HttpClient mode. |
| `port` | `int` | `8000` | TCP port for the remote server. Ignored in local mode. |
| `default_collection_name` | `str` | `"embeddings"` | The collection that is created on init and used when no `collection_name` argument is supplied. |
| `hnsw:space` | `str` | `"cosine"` | Distance metric for newly created collections. Passed as collection metadata. |

## Local (PersistentClient) mode

Used when `host` is **not** present in the config.
ChromaDB stores data on disk at the given `path`.

```python
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

db = ChromaEmbeddingsDB(config={
    "path": "/var/lib/ovos/chromadb",
    "default_collection_name": "memories",
})
```

The directory is created automatically. Data survives process restarts.

## Remote (HttpClient) mode

Used when `host` is present. The plugin connects to a running `chroma run` server.

```python
db = ChromaEmbeddingsDB(config={
    "host": "192.168.1.10",
    "port": 8000,
    "default_collection_name": "shared_embeddings",
})
```

Start a server with:

```bash
pip install chromadb
chroma run --host 0.0.0.0 --port 8000 --path /data/chroma
```

## Distance metric (`hnsw:space`)

ChromaDB supports three distance functions for the HNSW index:

| Value | Distance | Use case |
|---|---|---|
| `"cosine"` | Cosine distance (default) | Text/semantic similarity — direction matters, not magnitude |
| `"l2"` | Squared L2 (Euclidean) | Raw geometric distance |
| `"ip"` | Inner product | Maximum inner product search (requires normalised vectors) |

The metric is fixed at collection creation time. To change it, delete and recreate the collection.

Pass `hnsw:space` in the collection `metadata` dict when calling `create_collection`:

```python
db.create_collection("l2_collection", metadata={"hnsw:space": "l2"})
```

The default collection uses the value from the top-level config key (default: `"cosine"`).

---
[Home](../README.md) · [Usage →](usage.md)
