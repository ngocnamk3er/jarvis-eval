# Glossary

- **Bounded context** — a slice of the domain with its own model and data,
  the unit you extract into a service.
- **Database per service** — each service owns its schema; others reach it
  only through its API, never a shared table.
- **Saga** — a sequence of local transactions across services with
  compensating actions, used when a single request must write to more than
  one service without a distributed transaction.
- **Checkpoint** — LangGraph's saved graph state per conversation thread,
  in the `jarvis` database.
- **Chunk** — a ~300-word slice of a file's extracted text; the unit that
  gets embedded and stored in Qdrant.
