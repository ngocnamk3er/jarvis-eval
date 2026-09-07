from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = REPO_ROOT / "datasets"
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_FILE = DATASETS / "baseline.json"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # --- cluster access ---
    INGRESS_IP: str = "192.168.49.2"
    API_HOST: str = "api.jarvis.local"
    AUTH_HOST: str = "auth.jarvis.local"

    # --- keycloak ---
    KC_REALM: str = "jarvis"
    KC_CLIENT_ID: str = "jarvis-frontend"
    KC_CLIENT_SECRET: str = "jarvis-dev-secret"
    EVAL_USERNAME: str = "eval"
    EVAL_PASSWORD: str = "eval123"
    KC_ADMIN_USERNAME: str = "admin"
    KC_ADMIN_PASSWORD: str = "admin"

    # --- file-service (direct, via `make port-forward`) ---
    FILE_SERVICE_URL: str = "http://localhost:18002"
    INTERNAL_API_KEY: str = ""

    # --- model the agent runs as for hotpotqa / gaia (a ChatRequest.model id) ---
    RUNNER_MODEL: str = "deepseek/deepseek-v4-flash"
    RUNNER_THINKING_EFFORT: str = "high"

    # --- run behaviour ---
    MAX_HITL_ROUNDS: int = 8
    RUN_TIMEOUT: float = 420.0

    # --- standard benchmarks (hotpotqa / beir_scifact / beir_nfcorpus / gaia) ---
    # HF token — only GAIA needs it (gated). https://huggingface.co/settings/tokens
    HF_TOKEN: str = ""
    # Cap on docs loaded into a benchmark's workspace (Qdrant + embedding cost).
    BENCH_MAX_DOCS: int = 8000
    # HotpotQA: how many questions to sample (their paragraph pools form the corpus).
    HOTPOTQA_SAMPLE: int = 300
    # Questions actually run through the full agent (retrieval is scored on all).
    BENCH_AGENT_SAMPLE: int = 50

    @property
    def api_base(self) -> str:
        return f"http://{self.INGRESS_IP}"

    @property
    def api_headers(self) -> dict:
        return {"Host": self.API_HOST}

    @property
    def auth_headers(self) -> dict:
        return {"Host": self.AUTH_HOST}

    @property
    def token_url(self) -> str:
        return f"/realms/{self.KC_REALM}/protocol/openid-connect/token"


settings = Settings()
