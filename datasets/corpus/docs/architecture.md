# Architecture notes

Hands-on practice of "Microservices Patterns" decomposition: start from a
monolith, extract one bounded context at a time.

## Extracted so far
1. jarvis-conversation-service — owns conversations + subagent traces
2. jarvis-file-service — owns the folder/file tree, blob storage (MinIO),
   and the semantic search index (Qdrant embeddings)

## Data ownership
Three Postgres databases share one pod but stay logically isolated — no
service queries another's tables, always an internal HTTP call with a
shared X-Internal-Api-Key header. Keycloak has its own separate Postgres.

## Search
- grep_files — keyword match over file names and extracted text
- search_files — semantic vector search (bge-m3 embeddings via OpenRouter)
