from typing import List, Tuple, Optional, Dict, Union

import chromadb
import numpy as np

from ovos_plugin_manager.templates.embeddings import EmbeddingsDB


class ChromaEmbeddingsDB(EmbeddingsDB):
    def __init__(self, path: str):
        super().__init__()
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            "embeddings", metadata={"hnsw:space": "cosine"})

    def add_embeddings(self, key: str,
                       embedding: np.ndarray,
                       metadata: Optional[Dict] = None) -> None:
        self.collection.upsert(
            embeddings=[embedding.tolist()],
            ids=[key],
            metadatas=[metadata or {}]
        )

    def delete_embedding(self, key: str) -> None:
        self.collection.delete(ids=[key])

    def get_embedding(self, key: str) -> np.ndarray:
        e = self.collection.get(ids=[key], include=["embeddings"])['embeddings'][0]
        return np.array(e)

    def query(self, embedding: np.ndarray, top_k: int = 5,
              return_metadata: bool = False) -> List[Union[Tuple[str, float], Tuple[str, float, Dict]]]:
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        ids = [i for i in res["ids"][0]]
        distances = [i for i in res["distances"][0]]
        metadatas = [i for i in res["metadatas"][0]]
        if return_metadata:
            return list(zip(ids, distances, metadatas))
        return list(zip(ids, distances))


if __name__ == "__main__":
    import numpy as np

    # Initialize the database
    db = ChromaEmbeddingsDB(path="chromadb_storage")

    # Add embeddings
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.4, 0.5, 0.6])
    db.add_embeddings("user1", embedding1, metadata={"name": "Bob"})
    db.add_embeddings("user2", embedding2, metadata={"name": "Joe"})

    # Retrieve and print embeddings
    print(db.get_embedding("user1"))
    print(db.get_embedding("user2"))

    # Query embeddings
    query_embedding = np.array([0.2, 0.3, 0.4])
    results = db.query(query_embedding, top_k=2)
    print(results)
    # [('user2', 0.0053884541273605535),
    # ('user1', 0.007416666029069874)]
    results = db.query(query_embedding, top_k=2, return_metadata=True)
    print(results)
    # [('user2', 0.0053884541273605535, {'name': 'Joe'}),
    # ('user1', 0.007416666029069874, {'name': 'Bob'})]

    # Delete an embedding
    db.delete_embedding("user1")
