"""Central configuration.

Every knob lives here as a pydantic-settings field, so it can be set from
`.env`, an environment variable, or left at the default below. `settings`
(the singleton at the bottom) is imported everywhere else in the package.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- fixed paths, derived from where this file sits -----------------------
REPO_ROOT = Path(__file__).resolve().parent.parent   # .../jarvis-eval/
DATASETS = REPO_ROOT / "datasets"
RESULTS_DIR = REPO_ROOT / "results"                   # per-run outputs (git-ignored)
BASELINE_FILE = DATASETS / "baseline.json"            # the frozen reference scores


class Settings(BaseSettings):
    # env_file=".env": load overrides from a .env in the cwd.
    # case_sensitive: env var names must match field names exactly.
    # extra="ignore": tolerate unknown keys in .env instead of erroring.
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # --- how to reach the deployed test cluster ---
    # Everything is vhost-routed off one ingress IP; we send a `Host:` header
    # per service (the same trick as `curl --resolve`). Point INGRESS_IP at
    # `minikube ip -p minikube`.
    INGRESS_IP: str = "192.168.49.2"
    API_HOST: str = "api.jarvis.local"     # jarvis-backend
    AUTH_HOST: str = "auth.jarvis.local"   # keycloak

    # --- keycloak ---
    KC_REALM: str = "jarvis"
    KC_CLIENT_ID: str = "jarvis-frontend"          # the OIDC client we log in through
    KC_CLIENT_SECRET: str = "jarvis-dev-secret"
    EVAL_USERNAME: str = "eval"                    # base user; benchmarks use eval-<name>
    EVAL_PASSWORD: str = "eval123"                 # every eval user gets this password
    KC_ADMIN_USERNAME: str = "admin"              # master-realm admin, to create users
    KC_ADMIN_PASSWORD: str = "admin"              # from the jarvis-keycloak-secrets k8s secret

    # --- file-service (talked to directly, needs `make port-forward` first) ---
    # It is ClusterIP-only and the backend does not proxy its /search/* routes.
    FILE_SERVICE_URL: str = "http://localhost:18002"
    INTERNAL_API_KEY: str = ""                    # from the jarvis-secrets k8s secret

    # --- model the agent runs as for the hotpotqa / gaia suites ---
    # (BEIR is pure retrieval, no agent.) A ChatRequest.model id — see
    # jarvis-backend AVAILABLE_MODELS.
    RUNNER_MODEL: str = "deepseek/deepseek-v4-flash"
    RUNNER_THINKING_EFFORT: str = "high"

    # --- `jeval embcompare` only: an OpenAI-compatible /embeddings endpoint,
    # to score a candidate embedding model offline before deploying it ---
    EMBEDDING_BASE_URL: str = "https://openrouter.ai/api/v1"
    # Only used by the groundedness metrics — answer_relevancy compares the
    # answer to questions it generates, in vector space. Should match what
    # jarvis-file-service indexed with, so the comparison happens in the same
    # space as the retrieval being scored.
    EMBEDDING_MODEL: str = "openai/text-embedding-3-large"
    EMBEDDING_API_KEY: str = ""   # jarvis-secrets EMBEDDING_API_KEY (an OpenRouter key)

    # --- `jeval rerankcompare --backend openrouter` only: an OpenAI-compatible
    # /chat/completions endpoint for listwise (RankGPT-style) LLM reranking ---
    # LLM-as-judge for the groundedness metrics. Separate from RUNNER_MODEL
    # so judging the agent doesn't change when the agent's own model does —
    # a moving judge makes scores incomparable across runs.
    GROUNDEDNESS_MODEL: str = "deepseek/deepseek-v4-flash"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: str = ""   # a chat-capable OpenRouter key

    # --- agent run behaviour ---
    MAX_HITL_ROUNDS: int = 8       # auto-approve this many bash prompts before giving up
    RUN_TIMEOUT: float = 420.0     # wall-clock ceiling for one agent run (seconds)

    # --- benchmarks ---
    # HF token — only GAIA needs it (gated dataset). https://huggingface.co/settings/tokens
    HF_TOKEN: str = ""

    # Langfuse — the self-hosted trace/experiment store (jarvis-deploy/langfuse/).
    # Empty host turns the integration off, so `jeval run` still works on a
    # machine with no Langfuse. See jarvis_eval/langfuse_sync.py.
    LANGFUSE_HOST: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_ENVIRONMENT: str = "test"
    # Cap on docs loaded into a benchmark workspace (bounds embedding + Qdrant cost).
    BENCH_MAX_DOCS: int = 8000
    # HotpotQA: how many questions to sample. Their 10-paragraph pools, deduped,
    # become the corpus — 300 questions ≈ 3000 paragraphs.
    HOTPOTQA_SAMPLE: int = 300
    # Of the sample, how many to actually run through the agent (answer EM/F1).
    # Retrieval metrics are scored on all sampled questions regardless.
    BENCH_AGENT_SAMPLE: int = 50

    # --- convenience accessors (computed, not stored) ---
    @property
    def api_base(self) -> str:
        return f"http://{self.INGRESS_IP}"

    @property
    def api_headers(self) -> dict:
        """Attach to every request meant for jarvis-backend."""
        return {"Host": self.API_HOST}

    @property
    def auth_headers(self) -> dict:
        """Attach to every request meant for keycloak."""
        return {"Host": self.AUTH_HOST}

    @property
    def token_url(self) -> str:
        """Keycloak's OIDC token endpoint (password grant + refresh)."""
        return f"/realms/{self.KC_REALM}/protocol/openid-connect/token"


# Instantiated once at import time; reads .env here. Import this, not the class.
settings = Settings()
