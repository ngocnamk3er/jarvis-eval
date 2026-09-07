# Jarvis roadmap

## Done
- Split conversation management into its own service
- Per-user file workspace (folders, upload, grep, preview, download)
- Semantic search (bge-m3 + Qdrant)

## Next
- Reliability patterns around inter-service calls (retry, circuit breaker,
  timeouts) — Chapter 3
- API composition for endpoints that read from more than one service
- Saga once a request needs to write across two services

## Later
- Presigned URLs so file downloads don't proxy every byte through backend
- Streaming upload/download instead of loading whole files in memory
