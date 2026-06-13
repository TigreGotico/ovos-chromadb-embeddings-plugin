"""Quickstart: add a few vectors to a temporary ChromaDB store and query them."""
import tempfile
import numpy as np
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

with tempfile.TemporaryDirectory() as tmp:
    db = ChromaEmbeddingsDB(config={"path": tmp})

    # Store 4-dimensional vectors representing fruit "embeddings"
    db.add_embeddings("apple",  np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32))
    db.add_embeddings("banana", np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32))
    db.add_embeddings("cherry", np.array([0.0, 0.0, 0.9, 0.1], dtype=np.float32))

    print(f"Stored {db.count_embeddings_in_collection()} embeddings")

    # Query: find the two nearest neighbours to a vector close to "apple"
    query = np.array([0.85, 0.15, 0.0, 0.0], dtype=np.float32)
    results = db.query(query, top_k=2)

    print("Nearest neighbours:")
    for label, distance in results:
        print(f"  {label:10s}  distance={distance:.4f}")

    # Expected: apple is closest, banana second
    assert results[0][0] == "apple", f"Expected 'apple', got '{results[0][0]}'"
    print("OK")
