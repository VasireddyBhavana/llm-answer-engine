# Sampling and Decoding

When a language model generates text it produces a probability distribution over the next token. The decoding strategy decides which token is chosen.

## Temperature

Temperature rescales the logits before the softmax. A temperature of 0 makes decoding greedy, always choosing the most likely token, which gives the most deterministic output. Higher temperatures flatten the distribution and produce more varied, creative text. For factual question answering and evaluation runs, a low temperature is usually preferred.

## Top-p and top-k

Top-k sampling keeps only the k most likely tokens. Top-p, also called nucleus sampling, keeps the smallest set of tokens whose cumulative probability exceeds p. Both techniques cut off the long tail of unlikely tokens that can lead to incoherent output.

## Hallucination

A hallucination is a fluent but false statement produced by a model. Hallucinations happen because the model is optimized to produce plausible text, not verified text. Grounding answers in retrieved documents and requiring citations are the most common mitigations.
