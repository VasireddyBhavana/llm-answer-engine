# Embeddings and Vector Search

An embedding is a dense vector that represents the meaning of a piece of text. Texts with similar meaning have embeddings that point in similar directions, so similarity can be measured with cosine similarity.

## Vector databases

A vector database stores embeddings and supports nearest-neighbour search. Exact search compares the query with every vector, which is simple but slow at large scale. Approximate nearest neighbour indexes such as HNSW trade a little recall for much faster queries. Popular options include FAISS, pgvector, Qdrant, Weaviate and Pinecone.

## Dense versus sparse retrieval

Dense retrieval uses embeddings and is good at matching paraphrases and synonyms. Sparse retrieval such as BM25 scores documents by exact term overlap, weighted by how rare each term is. BM25 is strong for names, codes, error messages and other exact strings that embeddings can blur.

## Hybrid search

Hybrid search runs dense and sparse retrieval and merges the result lists. Reciprocal rank fusion is a simple merge method: each document gets the sum of 1 divided by (k plus its rank) across the lists, with k typically set to 60. It needs no score calibration because it only uses ranks.
