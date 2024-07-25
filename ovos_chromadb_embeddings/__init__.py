from typing import List, Optional, Dict

import chromadb
import numpy as np

from ovos_plugin_manager.templates.embeddings import EmbeddingsDB, EmbeddingsTuple


class ChromaEmbeddingsDB(EmbeddingsDB):
    """An implementation of EmbeddingsDB using ChromaDB for managing embeddings."""

    def __init__(self, path: str):
        """Initialize the ChromaEmbeddingsDB.

        Args:
            path (str): The path to the ChromaDB storage.
        """
        super().__init__()
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            "embeddings", metadata={"hnsw:space": "cosine"}
        )

    def add_embeddings(self, key: str, embedding: np.ndarray, metadata: Optional[Dict[str, any]] = None) -> None:
        """Add or update embeddings in the database.

        Args:
            key (str): The unique key for the embedding.
            embedding (np.ndarray): The embedding vector to store.
            metadata (Optional[Dict[str, any]]): Optional metadata associated with the embedding.
        """
        self.collection.upsert(
            embeddings=[embedding.tolist()],
            ids=[key],
            metadatas=[metadata or {}]
        )

    def delete_embedding(self, key: str) -> None:
        """Delete embeddings from the database.

        Args:
            key (str): The unique key for the embedding to delete.
        """
        self.collection.delete(ids=[key])

    def get_embedding(self, key: str) -> np.ndarray:
        """Retrieve embeddings from the database.

        Args:
            key (str): The unique key for the embedding to retrieve.

        Returns:
            np.ndarray: The retrieved embedding vector.
        """
        result = self.collection.get(ids=[key], include=["embeddings"])
        embedding_list = result['embeddings'][0]
        return np.array(embedding_list)

    def query(self, embedding: np.ndarray, top_k: int = 5, return_metadata: bool = False) -> List[EmbeddingsTuple]:
        """Query the database for the closest embeddings to the given query embedding.

        Args:
            embedding (np.ndarray): The embedding vector to query.
            top_k (int, optional): The number of top results to return. Defaults to 5.
            return_metadata (bool, optional): Whether to include metadata in the results. Defaults to False.

        Returns:
            List[EmbeddingsTuple]: A list of tuples containing the keys, distances, and optionally metadata.
        """
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        results = self.collection.query(
            query_embeddings=[embedding_list],
            n_results=top_k
        )
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

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
    #  ('user1', 0.007416666029069874)]
    results = db.query(query_embedding, top_k=2, return_metadata=True)
    print(results)
    # [('user2', 0.0053884541273605535, {'name': 'Joe'}),
    #  ('user1', 0.007416666029069874, {'name': 'Bob'})]

    # Delete an embedding
    db.delete_embedding("user1")
