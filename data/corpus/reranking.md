# Reranking and Context Selection

First-stage retrieval is optimized for recall: it should find the right passages somewhere in the top results, even if they are not ranked first.

## Cross-encoder rerankers

A reranker is a second, more expensive model that scores each query and passage pair together. Because a cross-encoder reads the query and passage jointly, it is more accurate than comparing two independent embeddings. A common pattern is to retrieve 50 candidates cheaply and rerank them down to the best 5.

## Maximal marginal relevance

Maximal marginal relevance, or MMR, selects passages that are relevant to the query but not redundant with passages already selected. A lambda parameter balances relevance against diversity. MMR helps when a corpus contains many near-duplicate chunks.

## Context budget

Even with long context windows, more context is not always better. Irrelevant passages distract the model and increase cost and latency. Placing the most relevant passages first and keeping the context tight usually improves answer quality.
