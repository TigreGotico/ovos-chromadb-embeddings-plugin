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
import numpy as np
from chroma_embeddings_db import ChromaEmbeddingsDB

# Initialize the database
db = ChromaEmbeddingsDB(path="path_to_chromadb_storage")

# Add embeddings
embedding1 = np.array([0.1, 0.2, 0.3])
embedding2 = np.array([0.4, 0.5, 0.6])
db.add_embeddings("user1", embedding1)
db.add_embeddings("user2", embedding2)

# Retrieve and print embeddings
print(db.get_embedding("user1"))
print(db.get_embedding("user2"))

# Query embeddings
query_embedding = np.array([0.2, 0.3, 0.4])
results = db.query(query_embedding, top_k=2)
print(results)

# Delete an embedding
db.delete_embedding("user1")
```

> Ensure that the path provided to the `ChromaEmbeddingsDB` constructor is accessible and writable.


## Credits

![image](https://github.com/user-attachments/assets/809588a2-32a2-406c-98c0-f88bf7753cb4)

> This work was sponsored by VisioLab, part of [Royal Dutch Visio](https://visio.org/), is the test, education, and research center in the field of (innovative) assistive technology for blind and visually impaired people and professionals. We explore (new) technological developments such as Voice, VR and AI and make the knowledge and expertise we gain available to everyone.
