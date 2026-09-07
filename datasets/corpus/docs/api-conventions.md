# Internal API conventions

Every service in Jarvis follows the same rules for service-to-service calls.

## Authentication
Internal endpoints sit behind an `X-Internal-Api-Key` header check. The key
is a shared secret in the `jarvis-secrets` Kubernetes secret. Services are
ClusterIP-only — never exposed through the ingress — so a plain header
check is enough. The frontend never holds this key; it calls the backend
with a Keycloak bearer token, and the backend calls the other services.

## Shape
- JSON in, JSON out; `snake_case` fields.
- `user_id` is passed explicitly as a query or form parameter on
  file-service calls (that service does no token parsing of its own).
- Errors: 4xx with a `{"detail": "..."}` body, same as FastAPI's default.

## Base URLs
In-cluster DNS: `http://<service>.jarvis.svc.cluster.local:8000`. Locally,
each service has a fixed port (backend 8000, conversation 8001, file 8002,
sandbox 8003).
