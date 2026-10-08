# Retrieval-Augmented Generation

Retrieval-augmented generation, or RAG, answers a question by first retrieving relevant passages from a knowledge base and then asking a language model to answer using only those passages. RAG lets a model use private or fresh data without retraining, and it makes answers auditable because each claim can cite a source.

## The pipeline

A typical RAG pipeline has two phases. Offline ingestion loads documents, splits them into chunks, embeds each chunk and stores the vectors in an index. Online, the system embeds the user query, retrieves the top-k most similar chunks, places them in the prompt with source labels, and generates a cited answer.

## Chunking

Chunk size is a trade-off. Small chunks are precise but can lose surrounding context. Large chunks keep context but dilute the embedding and waste prompt tokens. Chunks commonly range from 200 to 800 tokens with some overlap between neighbouring chunks so that sentences on a boundary are not lost. Splitting on document structure such as headings and paragraphs usually works better than fixed character windows.

## Failure modes

Most RAG failures are retrieval failures: the right passage was never retrieved, so the model could not use it. This is why retrieval quality should be measured separately from answer quality.
