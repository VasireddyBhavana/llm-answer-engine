# Personalization and Memory

Personalization adapts an AI system to an individual user. In LLM products this is usually done through memory rather than by training a separate model for each user.

## Kinds of memory

Short-term memory is the conversation history inside the context window. Long-term memory stores facts and preferences across sessions, such as a user's expertise level, preferred answer length or topics of interest. Long-term memories are typically stored in a database and the relevant ones are retrieved and inserted into the system prompt.

## Using memory well

Memory should change how an answer is delivered, such as depth, length and examples, without changing what is true. Users should be able to see and delete what the system remembers about them, and sensitive information should not be stored without consent.

## Personalized retrieval

Retrieval can also be personalized, for example by boosting documents on topics a user has engaged with before. This must be balanced against filter bubbles, where a user only ever sees one kind of source.
