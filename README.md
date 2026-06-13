[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenVoiceOS/ovos-chromadb-embeddings-plugin)

# ovos-chromadb-embeddings-plugin

ChromaDB-backed `EmbeddingsDB` vector store plugin for [OpenVoiceOS](https://openvoiceos.org).

## Install

```bash
pip install ovos-chromadb-embeddings-plugin
```

## What is an EmbeddingsDB?

`EmbeddingsDB` is the abstract base class from `ovos-plugin-manager` for vector stores.
Plugins implementing it are discovered automatically by OPM under the entry-point group `opm.embeddings`.
This plugin registers as:

```
opm.embeddings → ovos-chromadb-embeddings-plugin → ChromaEmbeddingsDB
```

OVOS subsystems (e.g. memory, RAG pipelines, face/voice recognition) call
`OVOSPluginFactory.get_plugin("opm.embeddings")` to obtain a configured store without
coupling to a specific backend.

## Quickstart

```python
import tempfile, numpy as np
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

with tempfile.TemporaryDirectory() as tmp:
    db = ChromaEmbeddingsDB(config={"path": tmp})

    # Store a few 4-d vectors
    db.add_embeddings("apple",  np.array([0.9, 0.1, 0.0, 0.0]))
    db.add_embeddings("banana", np.array([0.0, 0.9, 0.1, 0.0]))
    db.add_embeddings("cherry", np.array([0.0, 0.0, 0.9, 0.1]))

    # Nearest-neighbour query
    query = np.array([0.85, 0.15, 0.0, 0.0])
    results = db.query(query, top_k=2)
    # → [("apple", 0.003...), ("banana", 0.45...)]
    print(results[0][0])  # "apple"
```

Pair with an embedding producer such as
[ovos-gguf-embeddings-plugin](https://github.com/OpenVoiceOS/ovos-gguf-embeddings-plugin)
(`ovos-gguf-plugin`) to convert text → vectors before storing them here.

## Configuration

Pass a `config` dict to `ChromaEmbeddingsDB(config=...)` or set it in your OVOS
configuration under the plugin key.

| Key | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `"./chromadb_storage"` | Local persistence directory (PersistentClient mode). |
| `host` | `str` | — | Remote ChromaDB server host. When set, uses HttpClient instead of PersistentClient. |
| `port` | `int` | `8000` | Port for the remote ChromaDB server (HttpClient mode only). |
| `default_collection_name` | `str` | `"embeddings"` | Name of the collection created/used on init. |
| `hnsw:space` | `str` | `"cosine"` | Distance metric for HNSW index. Accepted: `"cosine"`, `"l2"`, `"ip"`. Set via collection metadata. |

### Local (persistent) mode

```python
db = ChromaEmbeddingsDB(config={"path": "/var/lib/ovos/chromadb"})
```

### Remote server mode

```python
db = ChromaEmbeddingsDB(config={"host": "192.168.1.10", "port": 8000})
```

## API overview

| Method | Description |
|---|---|
| `add_embeddings(key, embedding, metadata, collection_name)` | Upsert a single vector. |
| `add_embeddings_batch(keys, embeddings, metadata, collection_name)` | Upsert a list of vectors. |
| `get_embeddings(key, collection_name, return_metadata)` | Retrieve a vector by key. |
| `get_embeddings_batch(keys, collection_name, return_metadata)` | Retrieve multiple vectors. |
| `delete_embeddings(key, collection_name)` | Delete a vector by key. |
| `delete_embeddings_batch(keys, collection_name)` | Delete multiple vectors. |
| `query(embedding, top_k, return_metadata, collection_name)` | ANN search; returns `[(id, distance)]`. |
| `create_collection(name, metadata)` | Create (or get) a named collection. |
| `get_collection(name)` | Retrieve a collection handle (raises `ValueError` if absent). |
| `delete_collection(name)` | Drop a collection. |
| `list_collections()` | List all collections. |
| `count_embeddings_in_collection(collection_name)` | Count stored vectors. |

## Documentation

- [`docs/configuration.md`](docs/configuration.md) — full config reference (local vs remote, distance metrics)
- [`docs/usage.md`](docs/usage.md) — collections, CRUD, batch ops, metadata, numpy in/out

## Examples

- [`examples/quickstart.py`](examples/quickstart.py) — add vectors + query
- [`examples/collections.py`](examples/collections.py) — multi-collection workflow
- [`examples/remote_server.py`](examples/remote_server.py) — HttpClient usage

## Testing

```bash
pip install -e ".[test]"
pytest test/ -v
```

The test suite uses a temporary `PersistentClient` with no network access.
`test/test_e2e.py` runs a real end-to-end flow (add → query → verify nearest neighbour)
using a small deterministic local embedder so it passes in CI without model downloads.

---

## Credits

Originally developed by [TigreGótico](https://tigregotico.pt) for [OpenVoiceOS](https://openvoiceos.org),
sponsored by VisioLab. Modernized under the [NGI0 Commons Fund](https://nlnet.nl/commonsfund) / [NLnet](https://nlnet.nl).

![image](https://github.com/user-attachments/assets/809588a2-32a2-406c-98c0-f88bf7753cb4)

> This work was sponsored by VisioLab, part of [Royal Dutch Visio](https://visio.org/), is the test, education, and research center in the field of (innovative) assistive technology for blind and visually impaired people and professionals. We explore (new) technological developments such as Voice, VR and AI and make the knowledge and expertise we gain available to everyone.

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
