# Embeddings & vector search

## Model choice
Picked baai/bge-m3 over OpenAI text-embedding-3-small:
- multilingual, handles Vietnamese + English in one space
- $0.01 per million tokens (half the price)
- 1024 dimensions (smaller vectors, less Qdrant storage)
Accessed through OpenRouter's OpenAI-compatible /embeddings endpoint.

## Pipeline
upload -> extract plaintext -> chunk (~300 words, 50 overlap) -> embed each
chunk -> upsert into one Qdrant collection `file_chunks` with a user_id
payload filter for isolation.

## Hybrid search
grep_files (keyword) and search_files (semantic) are complementary — exact
identifiers and error strings favour grep, "find my notes about X" favours
vectors.
