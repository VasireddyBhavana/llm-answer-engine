# Transformers and Attention

The transformer is the neural network architecture behind modern large language models. It was introduced in 2017 and replaced recurrent networks for most language tasks because it processes all tokens in a sequence in parallel.

## Self-attention

Self-attention lets every token look at every other token in the context window. Each token is projected into a query, a key and a value vector. The attention weight between two tokens is the softmax of the scaled dot product of one token's query with the other token's key. The output for a token is the weighted sum of value vectors.

Because every token attends to every other token, the cost of attention grows quadratically with sequence length. This is the main reason long context windows are expensive.

## Tokens

Language models do not read characters or words directly. Text is split into tokens by a tokenizer, usually with byte-pair encoding. A token is roughly three quarters of an English word on average. Model pricing and context limits are measured in tokens.

## Decoder-only models

Most chat models are decoder-only transformers trained to predict the next token. Generation happens one token at a time: the model predicts a probability distribution over the vocabulary, a token is sampled, appended to the input, and the process repeats.
