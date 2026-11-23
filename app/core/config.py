"""Application settings loaded from environment variables (and `.env` locally)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Every field maps to the upper-case environment variable.

    Secrets default to empty strings so endpoints that don't need them still start;
    the features that do need them fail with a clear 503 instead.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_name: str = "Portfolio AI Chatbot"
    api_prefix: str = "/api/v1"

    frontend_origins: str = "http://localhost:5173"

    supabase_url: str = ""
    supabase_secret_key: str = ""

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "portfolio_knowledge"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str | None = None  # None = the model's documented prefix
    sparse_model: str = "Qdrant/bm25"  # "" = dense-only retrieval
    # Off by default: benchmarked +~380 MB RSS and ~1.5 s/query on a desktop CPU,
    # which doesn't fit Render free (512 MB, 0.1 CPU). See docs/rag-improvement-roadmap.md.
    reranker_model: str = ""  # e.g. "Xenova/ms-marco-MiniLM-L-6-v2"

    retrieval_candidates: int = 20  # per-retriever candidates fed into fusion/reranking
    retrieval_top_k: int = 5  # sources given to the LLM for a normal question
    retrieval_list_top_k: int = 10  # ...for "list all X" questions
    rrf_k: int = 60
    # Dense cosine floor (reranking off). Benchmarked: the weakest correct answer scores
    # 0.477 and off-topic questions 0.467-0.664, so this trims noise without losing
    # answers; out-of-scope questions are refused by the answer prompt, not this gate.
    retrieval_score_threshold: float = 0.45
    # Cross-encoder logit floor (reranking on). -11 is the highest value that loses no
    # benchmark hit for ms-marco-MiniLM-L-6-v2; recalibrate for any other reranker.
    rerank_score_threshold: float = -11.0
    max_context_tokens: int = 3000

    llm_provider: str = ""  # "gemini" | "groq"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3

    history_max_turns: int = 6
    rag_debug: bool = False  # include the retrieval trace in /chat/ask responses (dev only)
    preload_models: bool = True

    admin_api_token: str = ""  # header x-admin-token, for manual/admin calls
    webhook_secret: str = ""  # header X-Webhook-Secret, for automated sync triggers
    sync_enabled: bool = True
    rate_limit_per_minute: int = 30  # /chat/ask requests per client IP

    @property
    def frontend_origins_list(self) -> list[str]:
        """Return the comma-separated `FRONTEND_ORIGINS` as a clean list.

        Returns:
            Allowed CORS origins, whitespace trimmed and empties removed.
        """
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]


settings = Settings()
