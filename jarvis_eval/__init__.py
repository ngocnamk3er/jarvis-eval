"""jarvis-eval — a black-box evaluation & benchmarking harness for the Jarvis agent.

Runs against the deployed test cluster: authenticates as a dedicated `eval`
Keycloak user, seeds a versioned corpus into that user's file workspace, then
scores three suites — `retrieval` (recall@k / MRR / nDCG straight against
file-service), `rag_qa` and `agent_tasks` (full agent runs through
/api/v1/chat/stream, graded by an LLM judge + deterministic checks).
"""

__version__ = "0.1.0"
