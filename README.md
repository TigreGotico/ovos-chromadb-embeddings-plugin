[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ovos-chromadb-embeddings-plugin)

# ChromaEmbeddingsDB Plugin

## Overview

The `ChromaEmbeddingsDB` plugin integrates with the [ChromaDB](https://www.trychroma.com/) database to provide a robust solution for managing and querying embeddings. This plugin extends the abstract `EmbeddingsDB` class, allowing you to store, retrieve, and query embeddings efficiently using ChromaDB’s capabilities.

This plugin is meant to be used by other specialized plugins such as:
- https://github.com/TigreGotico/ovos-face-embeddings-plugin
- https://github.com/TigreGotico/ovos-voice-embeddings-plugin
- https://github.com/TigreGotico/ovos-gguf-embeddings-plugin

## Features

- **Add Embeddings**: Store embeddings with associated keys.
- **Retrieve Embeddings**: Fetch embeddings by their keys.
- **Delete Embeddings**: Remove embeddings by their keys.
- **Query Embeddings**: Find the closest embeddings to a given query, with support for cosine distance.

## Example

Here is a full example demonstrating the basic usage of `ChromaEmbeddingsDB`.

```python
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

```

> Ensure that the path provided to the `ChromaEmbeddingsDB` constructor is accessible and writable.


## Credits

![image](https://github.com/user-attachments/assets/809588a2-32a2-406c-98c0-f88bf7753cb4)

> This work was sponsored by VisioLab, part of [Royal Dutch Visio](https://visio.org/), is the test, education, and research center in the field of (innovative) assistive technology for blind and visually impaired people and professionals. We explore (new) technological developments such as Voice, VR and AI and make the knowledge and expertise we gain available to everyone.
