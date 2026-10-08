# LLM Agents and Tool Use

An agent is a system where a language model decides which actions to take in a loop instead of following a fixed script. The model is given a set of tools, each described by a name, a description and a JSON schema for its arguments.

## The agent loop

The loop works like this: the model receives the task and the tool definitions; it either answers or emits a tool call; the application executes the tool and returns the result to the model; and the cycle repeats until the model produces a final answer or a step limit is reached. A maximum step count is an important safety limit that prevents runaway loops and cost.

## Designing tools

Good tools have clear names, narrow responsibilities and descriptive error messages, because the model reads the error to decide what to do next. Tools that take irreversible actions such as sending email or making payments should require human confirmation.

## Prompt injection

Text returned by tools, such as web pages or documents, can contain instructions written by an attacker. This is called prompt injection. Tool outputs should be treated as data rather than commands, and agents should be given the least privilege they need.
