import chromadb
import numpy as np
from typing import List, Tuple
from ovos_plugin_manager.templates.embeddings import EmbeddingsDB


class ChromaEmbeddingsDB(EmbeddingsDB):
    def __init__(self, path: str):
        super().__init__()
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection("embeddings",
                                                               metadata={"hnsw:space": "cosine"}  # l2 is the default
                                                               )

    def add_embeddings(self, key: str, embedding: np.ndarray) -> None:
        self.collection.upsert(
            embeddings=[embedding.tolist()],
            ids=[key]
        )

    def delete_embedding(self, key: str) -> None:
        self.collection.delete(ids=[key])

    def get_embedding(self, key: str) -> np.ndarray:
        e = self.collection.get(ids=[key], include=["embeddings"])['embeddings'][0]
        return np.array(e)

    def query(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        ids = [i for i in res["ids"][0]]
        distances = [i for i in res["distances"][0]]
        return list(zip(ids, distances))
