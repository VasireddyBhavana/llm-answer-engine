# Evaluating LLM Systems

LLM systems are evaluated with a fixed dataset of inputs and expected outputs, often called a golden set. Every change to prompts, models or retrieval settings is run against the golden set so that improvements and regressions are measured rather than guessed.

## Retrieval metrics

Recall at k measures the fraction of questions for which a relevant passage appears in the top k results. Mean reciprocal rank, or MRR, averages 1 divided by the rank of the first relevant result, so it rewards putting the right passage near the top.

## Answer metrics

Faithfulness, also called groundedness, asks whether every claim in the answer is supported by the retrieved context. Answer relevance asks whether the answer addresses the question. Citation precision asks whether the cited sources actually support the answer.

## LLM as a judge

Many answer metrics are scored by a second language model acting as a judge with a rubric. Judges are cheap and scalable but can be biased, for example toward longer answers, so judge scores should be spot-checked against human labels.

## Online evaluation

Offline evals catch regressions before launch. After launch, teams track online signals such as thumbs up and down, follow-up question rate, latency and cost per query.
