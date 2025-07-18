from typing import List, Optional, Dict, Any, Tuple, Union

import chromadb
from chromadb.config import Settings

from ovos_plugin_manager.templates.embeddings import EmbeddingsDB, EmbeddingsArray, EmbeddingsTuple, \
    RetrievedEmbeddingResult


class ChromaEmbeddingsDB(EmbeddingsDB):
    """An implementation of EmbeddingsDB using ChromaDB for managing embeddings."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ChromaEmbeddingsDB instance with the specified configuration.
        
        Creates a ChromaDB client using either HTTP or persistent storage based on the provided configuration. Sets the default collection name and ensures the default collection exists upon initialization.
        """
        super().__init__(config)
        # Determine the default collection name from config, or use "embeddings"
        self.default_collection_name = self.config.get("default_collection_name", "embeddings")

        # Initialize ChromaDB client based on host (for HTTP client) or path (for Persistent client)
        if "host" in self.config:
            host = self.config["host"]
            port = self.config.get("port", 8000)
            self.client = chromadb.HttpClient(host=host, port=port, ssl=False, headers=None, settings=None,
                                              tenant=chromadb.DEFAULT_TENANT, database=chromadb.DEFAULT_DATABASE)
        else:
            # Default path for persistent client if not specified
            path = self.config.get("path", "./chromadb_storage")
            self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))

        # Ensure the default collection exists upon initialization
        # This calls the internal helper, not the abstract method, to avoid recursion
        self._get_or_create_collection(self.default_collection_name)

    def _get_collection_instance(self, collection_name: Optional[str]) -> chromadb.api.models.Collection.Collection:
        """
        Retrieve a ChromaDB collection instance by name or return the default collection.
        
        Raises:
            ValueError: If the specified collection does not exist or cannot be accessed.
        
        Returns:
            The ChromaDB collection instance corresponding to the given name or the default collection if no name is provided.
        """
        name = collection_name or self.default_collection_name
        try:
            return self.client.get_collection(name=name)
        except Exception as e:
            # ChromaDB raises an exception if collection not found
            raise ValueError(f"Collection '{name}' not found or could not be accessed: {e}")

    def _get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Get an existing ChromaDB collection by name or create it with specified metadata if it does not exist.
        
        If no `"hnsw:space"` is specified in the metadata, it defaults to `"cosine"`. Returns the collection instance.
        """
        collection_metadata = metadata or {}
        if "hnsw:space" not in collection_metadata:
            collection_metadata["hnsw:space"] = "cosine"
        return self.client.get_or_create_collection(name, metadata=collection_metadata)

    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a new embeddings collection with the specified name and optional metadata.
        
        If a collection with the given name already exists, returns the existing collection handle.
        
        Parameters:
            name (str): Name of the collection to create.
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with the collection.
        
        Returns:
            A handle to the created or existing collection.
        """
        return self._get_or_create_collection(name, metadata)

    def get_collection(self, name: str) -> Any:
        """
        Retrieve a ChromaDB collection by its name.
        
        Parameters:
            name (str): Name of the collection to retrieve.
        
        Returns:
            The collection object corresponding to the specified name.
        
        Raises:
            ValueError: If the collection does not exist.
        """
        return self._get_collection_instance(name)

    def delete_collection(self, name: str) -> None:
        """
        Deletes the specified collection from the database.
        
        Parameters:
            name (str): Name of the collection to delete.
        """
        try:
            self.client.delete_collection(name=name)
        except Exception as e:
            print(f"Error deleting collection '{name}': {e}")
            # Optionally raise a more specific error or handle silently

    def list_collections(self) -> List[Any]:
        """
        Return a list of all collections available in the database.
        
        Returns:
            List[Any]: Handles or objects representing each collection.
        """
        return self.client.list_collections()

    def add_embeddings(self, key: str, embedding: EmbeddingsArray,
                       metadata: Optional[Dict[str, Any]] = None,
                       collection_name: Optional[str] = None) -> EmbeddingsArray:
        """
                       Adds or updates an embedding vector under the specified key in a ChromaDB collection, optionally associating metadata.
                       
                       Parameters:
                           key (str): Unique identifier for the embedding.
                           embedding (EmbeddingsArray): The embedding vector to store.
                           metadata (Optional[Dict[str, Any]]): Additional metadata to associate with the embedding.
                           collection_name (Optional[str]): Name of the collection to use; defaults to the configured default collection if not provided.
                       
                       Returns:
                           EmbeddingsArray: The embedding that was stored.
                       """
        collection = self._get_collection_instance(collection_name)
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        collection.upsert(
            embeddings=[embedding_list],
            ids=[key],
            metadatas=[metadata or {}]
        )
        return embedding

    def add_embeddings_batch(self, keys: List[str], embeddings: List[EmbeddingsArray],
                             metadata: Optional[List[Dict[str, Any]]] = None,
                             collection_name: Optional[str] = None) -> None:
        """
                             Add or update multiple embeddings in a batch within the specified or default collection.
                             
                             Parameters:
                                 keys (List[str]): Unique identifiers for each embedding.
                                 embeddings (List[EmbeddingsArray]): Embedding vectors to store.
                                 metadata (Optional[List[Dict[str, Any]]]): Optional metadata for each embedding.
                                 collection_name (Optional[str]): Name of the collection to use; defaults to the configured default collection.
                             """
        collection = self._get_collection_instance(collection_name)
        # Ensure embeddings are lists for ChromaDB
        embeddings_list = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]
        collection.upsert(
            embeddings=embeddings_list,
            ids=keys,
            metadatas=metadata or ([{}] * len(keys))  # Provide empty dicts if metadata is None
        )

    def get_embeddings(self, key: str, collection_name: Optional[str] = None,
                       return_metadata: bool = False) -> Union[Optional[EmbeddingsArray],
    Tuple[Optional[EmbeddingsArray], Optional[Dict[str, Any]]]]:
        """
                       Retrieve the embedding associated with the given key from the specified or default collection.
                       
                       Parameters:
                           key (str): The unique identifier for the embedding.
                           collection_name (Optional[str]): The collection to search in. If not provided, uses the default collection.
                           return_metadata (bool): If True, also returns the embedding's metadata.
                       
                       Returns:
                           If `return_metadata` is False, returns the embedding as a numpy array, or None if not found.
                           If `return_metadata` is True, returns a tuple (embedding, metadata) or (None, None) if not found.
                       """
        collection = self._get_collection_instance(collection_name)

        # Include metadata only if requested
        include_fields = ["embeddings"]
        if return_metadata:
            include_fields.append("metadatas")

        result = collection.get(ids=[key], include=include_fields)

        if result and result['embeddings'] is not None:
            embedding_list = result['embeddings']
            embedding_array = np.array(embedding_list)
            if return_metadata:
                metadata = result['metadatas'][0] if result['metadatas'] else {}
                return embedding_array, metadata
            return embedding_array
        return None if not return_metadata else (None, None)

    def get_embeddings_batch(self, keys: List[str], collection_name: Optional[str] = None,
                             return_metadata: bool = False) -> List[RetrievedEmbeddingResult]:
        """
                             Retrieve multiple embeddings by keys from a specified collection, optionally including metadata.
                             
                             Parameters:
                                 keys (List[str]): The list of keys identifying the embeddings to retrieve.
                                 collection_name (Optional[str]): The name of the collection to query. If None, uses the default collection.
                                 return_metadata (bool): If True, includes metadata for each embedding in the results.
                             
                             Returns:
                                 List[RetrievedEmbeddingResult]: A list of tuples containing (key, embedding) or (key, embedding, metadata) for each found embedding.
                             """
        collection = self._get_collection_instance(collection_name)

        # Include metadata only if requested
        include_fields = ["embeddings"]
        if return_metadata:
            include_fields.append("metadatas")

        result = collection.get(ids=keys, include=include_fields)

        retrieved_embeddings = []
        if result and result['ids']:
            for i, key in enumerate(result['ids']):
                embedding = np.array(result['embeddings'][i]) if result['embeddings'] is not None else None
                if embedding is not None:  # Only add if embedding was found
                    if return_metadata:
                        metadata = result['metadatas'][i] if result['metadatas'] else {}
                        retrieved_embeddings.append((key, embedding, metadata))
                    else:
                        retrieved_embeddings.append((key, embedding))
        return retrieved_embeddings

    def delete_embeddings(self, key: str, collection_name: Optional[str] = None) -> None:
        """
        Delete the embedding associated with the given key from the specified or default collection.
        
        Parameters:
            key (str): The unique identifier of the embedding to delete.
            collection_name (Optional[str]): The name of the collection to delete from. If not provided, the default collection is used.
        """
        collection = self._get_collection_instance(collection_name)
        collection.delete(ids=[key])

    def delete_embeddings_batch(self, keys: List[str], collection_name: Optional[str] = None) -> None:
        """
        Delete multiple embeddings identified by their keys from the specified or default collection.
        
        Parameters:
            keys (List[str]): List of embedding keys to delete.
            collection_name (Optional[str]): Name of the collection to delete from. If not provided, uses the default collection.
        """
        collection = self._get_collection_instance(collection_name)
        collection.delete(ids=keys)

    def query(self, embeddings: EmbeddingsArray, top_k: int = 5,
              return_metadata: bool = False, collection_name: Optional[str] = None) -> List[EmbeddingsTuple]:
        """
              Finds the closest embeddings to a given query embedding in the specified or default collection.
              
              Parameters:
                  embeddings (np.ndarray): The embedding vector to use as the query.
                  top_k (int, optional): Maximum number of nearest results to return. Defaults to 5.
                  return_metadata (bool, optional): If True, includes metadata for each result. Defaults to False.
                  collection_name (Optional[str]): Name of the collection to query; uses the default if not specified.
              
              Returns:
                  List[EmbeddingsTuple]: A list of tuples containing (id, distance) or (id, distance, metadata) for each nearest embedding found.
              """
        collection = self._get_collection_instance(collection_name)
        embedding_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

        # ChromaDB's query method returns lists of lists for ids, distances, metadatas
        results = collection.query(
            query_embeddings=[embedding_list],
            n_results=top_k,
            include=["distances", "metadatas"]  # Always include distances and metadatas for consistent return
        )

        # Extract the first (and only) list from the results
        ids = results["ids"][0] if results and results["ids"] else []
        distances = results["distances"][0] if results and results["distances"] else []
        metadatas = results["metadatas"][0] if results and results["metadatas"] else []

        # Combine results into the expected EmbeddingsTuple format
        if return_metadata:
            return list(zip(ids, distances, metadatas))
        return list(zip(ids, distances))

    def count_embeddings_in_collection(self, collection_name: Optional[str] = None) -> int:
        """
        Return the number of embeddings stored in the specified collection.
        
        Parameters:
        	collection_name (Optional[str]): Name of the collection to count embeddings in. If None, uses the default collection.
        
        Returns:
        	int: The total number of embeddings in the collection.
        """
        collection = self._get_collection_instance(collection_name)
        return collection.count()


if __name__ == "__main__":
    import numpy as np
    import os
    import shutil

    # Define a storage path for testing
    TEST_DB_PATH = "chromadb_test_storage"

    # Clean up previous test data if it exists
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)
        print(f"Cleaned up existing test data at {TEST_DB_PATH}")

    # Initialize the database with a test path
    print("\n--- Initializing ChromaEmbeddingsDB ---")
    db = ChromaEmbeddingsDB(config=dict(path=TEST_DB_PATH, default_collection_name="my_test_embeddings"))
    print(f"Default collection name: {db.default_collection_name}")

    # Test collection management
    print("\n--- Testing Collection Management ---")
    new_collection_name = "my_new_collection"
    db.create_collection(new_collection_name, metadata={"purpose": "testing"})
    print(f"Created new collection: {new_collection_name}")

    collections = db.list_collections()
    print("Available collections:")
    for col in collections:
        print(f"  - {col.name}")

    # Get a specific collection
    retrieved_collection = db.get_collection(new_collection_name)
    print(f"Retrieved collection: {retrieved_collection.name}")

    # Add embeddings to the default collection
    print("\n--- Adding Embeddings to Default Collection ---")
    embedding1 = np.array([0.1, 0.2, 0.3])
    embedding2 = np.array([0.4, 0.5, 0.6])
    embedding3 = np.array([0.7, 0.8, 0.9])

    db.add_embeddings("user1", embedding1, metadata={"name": "Bob", "age": 30})
    db.add_embeddings("user2", embedding2, metadata={"name": "Joe", "city": "New York"})
    print("Added user1 and user2 embeddings to default collection.")

    # Add embeddings to the new collection
    print("\n--- Adding Embeddings to New Collection ---")
    db.add_embeddings("itemA", embedding1 * 0.5, metadata={"type": "product"}, collection_name=new_collection_name)
    db.add_embeddings("itemB", embedding2 * 0.5, metadata={"type": "service"}, collection_name=new_collection_name)
    print("Added itemA and itemB embeddings to new_collection.")

    # Test count_embeddings_in_collection
    print("\n--- Testing Embedding Count ---")
    print(f"Embeddings in default collection: {db.count_embeddings_in_collection()}")
    print(f"Embeddings in '{new_collection_name}' collection: {db.count_embeddings_in_collection(new_collection_name)}")

    # Retrieve and print embeddings from default collection
    print("\n--- Retrieving Embeddings from Default Collection ---")
    retrieved_emb1 = db.get_embeddings("user1")
    print(f"Retrieved embedding for user1 (no metadata): {retrieved_emb1}")
    retrieved_emb1_meta, retrieved_meta1 = db.get_embeddings("user1", return_metadata=True)
    print(f"Retrieved embedding and metadata for user1: {retrieved_emb1_meta}, {retrieved_meta1}")

    # Retrieve and print embeddings from new collection
    print("\n--- Retrieving Embeddings from New Collection ---")
    retrieved_itemA = db.get_embeddings("itemA", collection_name=new_collection_name)
    print(f"Retrieved embedding for itemA (no metadata): {retrieved_itemA}")
    retrieved_itemA_meta, retrieved_metaA = db.get_embeddings("itemA", collection_name=new_collection_name,
                                                              return_metadata=True)
    print(f"Retrieved embedding and metadata for itemA: {retrieved_itemA_meta}, {retrieved_metaA}")

    # Test batch add and get
    print("\n--- Testing Batch Operations ---")
    batch_keys = ["batch_user3", "batch_user4"]
    batch_embeddings = [np.array([0.9, 0.8, 0.7]), np.array([0.6, 0.5, 0.4])]
    batch_metadata = [{"source": "batch_upload"}, {"source": "batch_upload", "tag": "test"}]
    db.add_embeddings_batch(batch_keys, batch_embeddings, batch_metadata)
    print("Added batch_user3 and batch_user4 via batch operation.")

    retrieved_batch = db.get_embeddings_batch(batch_keys, return_metadata=True)
    print("Retrieved batch embeddings (with metadata):")
    for key, emb, meta in retrieved_batch:
        print(f"  Key: {key}, Embedding: {emb}, Metadata: {meta}")

    retrieved_batch_no_meta = db.get_embeddings_batch(batch_keys, return_metadata=False)
    print("Retrieved batch embeddings (no metadata):")
    for key, emb in retrieved_batch_no_meta:
        print(f"  Key: {key}, Embedding: {emb}")

    # Query embeddings in default collection
    print("\n--- Querying Embeddings in Default Collection ---")
    query_embedding = np.array([0.2, 0.3, 0.4])
    results = db.query(query_embedding, top_k=2)
    print(f"Query results (no metadata): {results}")
    results_with_meta = db.query(query_embedding, top_k=2, return_metadata=True)
    print(f"Query results (with metadata): {results_with_meta}")

    # Query embeddings in new collection
    print("\n--- Querying Embeddings in New Collection ---")
    query_item_embedding = np.array([0.3, 0.4, 0.5])  # A query closer to itemA/itemB
    item_results = db.query(query_item_embedding, top_k=2, collection_name=new_collection_name, return_metadata=True)
    print(f"Query results in '{new_collection_name}': {item_results}")

    # Test batch delete
    print("\n--- Testing Batch Delete ---")
    db.delete_embeddings_batch(["batch_user3", "user1"])
    print("Deleted batch_user3 and user1 via batch delete.")
    print(f"Embeddings in default collection after batch delete: {db.count_embeddings_in_collection()}")
    print(f"Retrieved embedding for user1 after delete: {db.get_embeddings('user1')}")  # Should be None

    # Delete an embedding from the new collection
    print("\n--- Deleting Embeddings from New Collection ---")
    db.delete_embeddings("itemA", collection_name=new_collection_name)
    print("Deleted itemA from new_collection.")
    print(
        f"Embeddings in '{new_collection_name}' collection after delete: {db.count_embeddings_in_collection(new_collection_name)}")

    # Delete the new collection
    print("\n--- Deleting New Collection ---")
    db.delete_collection(new_collection_name)
    collections_after_delete = db.list_collections()
    print("Available collections after deleting new_collection:")
    for col in collections_after_delete:
        print(f"  - {col.name}")
    if not any(col.name == new_collection_name for col in collections_after_delete):
        print(f"Collection '{new_collection_name}' successfully deleted.")
    else:
        print(f"Collection '{new_collection_name}' still exists (unexpected).")

    # Clean up test data
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)
        print(f"\nCleaned up test data at {TEST_DB_PATH}")
