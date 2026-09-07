# Jarvis

Jarvis is a self-hosted, agentic AI assistant. A LangGraph agent with tool
use (bash, web search, web fetch, visualization, and a persistent file
workspace) behind a Next.js frontend and Keycloak login.

## Services
- jarvis-frontend — the chat UI
- jarvis-backend — agent orchestration
- jarvis-conversation-service — conversation history
- jarvis-file-service — this file workspace (tree + MinIO blobs + Qdrant embeddings)
- jarvis-keycloak — authentication

Everything runs on one machine: minikube + Jenkins + GitLab + ArgoCD.
