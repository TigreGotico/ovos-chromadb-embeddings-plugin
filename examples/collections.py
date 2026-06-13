"""Multi-collection workflow: separate namespaces for faces and voices."""
import tempfile
import numpy as np
from ovos_chromadb_embeddings import ChromaEmbeddingsDB

with tempfile.TemporaryDirectory() as tmp:
    db = ChromaEmbeddingsDB(config={"path": tmp, "default_collection_name": "faces"})

    # Create a second collection with L2 distance
    db.create_collection("voices", metadata={"hnsw:space": "l2"})

    # Add face embeddings (3-d)
    db.add_embeddings("alice_face", np.array([0.8, 0.1, 0.1], dtype=np.float32),
                      metadata={"person": "Alice"})
    db.add_embeddings("bob_face",   np.array([0.1, 0.8, 0.1], dtype=np.float32),
                      metadata={"person": "Bob"})

    # Add voice embeddings into a different collection (different dim OK per collection)
    db.add_embeddings("alice_voice", np.array([0.9, 0.05, 0.05], dtype=np.float32),
                      collection_name="voices", metadata={"person": "Alice"})
    db.add_embeddings("bob_voice",   np.array([0.05, 0.9, 0.05], dtype=np.float32),
                      collection_name="voices", metadata={"person": "Bob"})

    print("Collections:", [c.name for c in db.list_collections()])

    # Query faces
    face_query = np.array([0.75, 0.15, 0.10], dtype=np.float32)
    face_hits = db.query(face_query, top_k=1, return_metadata=True)
    print(f"Closest face: {face_hits[0][0]} ({face_hits[0][2]['person']})")

    # Query voices
    voice_query = np.array([0.08, 0.85, 0.07], dtype=np.float32)
    voice_hits = db.query(voice_query, top_k=1, collection_name="voices", return_metadata=True)
    print(f"Closest voice: {voice_hits[0][0]} ({voice_hits[0][2]['person']})")

    # Collections are fully independent — counts reflect only their own data
    print(f"Face embeddings: {db.count_embeddings_in_collection('faces')}")
    print(f"Voice embeddings: {db.count_embeddings_in_collection('voices')}")
