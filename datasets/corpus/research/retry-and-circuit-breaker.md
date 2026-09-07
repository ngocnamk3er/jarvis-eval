# Reliability patterns for inter-service calls

Notes from "Microservices Patterns" chapter 3, to apply to the internal
HTTP calls jarvis-backend makes to conversation-service and file-service.

## Retry with backoff
Retry only *idempotent* reads (GET) and only on transient failures
(connection refused, 502/503, timeout). Exponential backoff with jitter,
a small cap on attempts (3). Never blind-retry a POST that may have
already committed.

## Circuit breaker
After N consecutive failures, open the circuit: fail fast for a cooldown
window instead of hammering a service that is down. Half-open after the
cooldown to probe recovery. Protects the caller's thread pool from being
exhausted waiting on a dead dependency.

## Timeouts
Every outbound call needs an explicit timeout shorter than the caller's
own request budget. A missing timeout is the most common cause of
cascading failure.

Current state: none of this is implemented yet — conversation_client.py
deliberately does a bare httpx call so failures are loud during the split.
