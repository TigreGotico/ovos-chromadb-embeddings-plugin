"""
HttpClient (remote server) usage example.

Requires a running ChromaDB server:
    pip install chromadb
    chroma run --host 0.0.0.0 --port 8000 --path /tmp/chroma_server

This script is illustrative — it will fail if no server is reachable at CHROMA_HOST:CHROMA_PORT.
"""
import os
import numpy as np

CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

try:
    from ovos_chromadb_embeddings import ChromaEmbeddingsDB

    db = ChromaEmbeddingsDB(config={
        "host": CHROMA_HOST,
        "port": CHROMA_PORT,
        "default_collection_name": "remote_demo",
    })

    db.add_embeddings("vec_a", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    db.add_embeddings("vec_b", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    results = db.query(np.array([0.9, 0.1, 0.0], dtype=np.float32), top_k=2)
    print("Results:", results)

    db.delete_collection("remote_demo")
    print("Done — collection cleaned up.")

except Exception as exc:
    print(f"Could not connect to ChromaDB server at {CHROMA_HOST}:{CHROMA_PORT}: {exc}")
    print("Start a server with: chroma run --host 0.0.0.0 --port 8000 --path /tmp/chroma_server")
