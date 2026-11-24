"""Application settings loaded from environment variables (and `.env` locally)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Every field maps to the upper-case environment variable.

    Secrets default to empty strings so endpoints that don't need them still start;
    the features that do need them fail with a clear 503 instead.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_title: str = "AI Knowledge Assistant"
    api_prefix: str = "/api/v1"
    allowed_origins: str = "*"  # comma-separated; "*" lets any website embed the chat

    # Who the assistant is. The grounding and safety rules are always appended to it.
    assistant_persona: str = (
        "You are a helpful assistant that answers questions about the topics covered by this knowledge base."
    )
    no_answer_message: str = "That information is not available in the knowledge base."

    admin_password: str = ""  # unlocks /dataentry and /api/v1/admin/*
    webhook_secret: str = ""  # header X-Webhook-Secret, for automated sync triggers
    sync_enabled: bool = True
    rate_limit_per_minute: int = 30  # chat requests per client IP

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_table_name: str = "portfolio_knowledge_base"

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "knowledge_base"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str | None = None  # None = the model's documented prefix
    sparse_model: str = "Qdrant/bm25"  # "" = dense-only retrieval
    # Off by default: benchmarked +~380 MB RSS and ~1.5 s/query on a desktop CPU,
    # which doesn't fit Render free (512 MB, 0.1 CPU). See docs/rag-improvement-roadmap.md.
    reranker_model: str = ""  # e.g. "Xenova/ms-marco-MiniLM-L-6-v2"
    # Entries longer than this (characters, ~4 per token) are split into chunks.
    # 1200 keeps every chunk well inside bge-small's 512-token window.
    chunk_size: int = 1200

    retrieval_candidates: int = 20  # per-retriever candidates fed into fusion/reranking
    retrieval_top_k: int = 5  # sources given to the LLM for a normal question
    retrieval_list_top_k: int = 10  # ...for "list all X" questions
    rrf_k: int = 60
    # Dense cosine floor (reranking off). Benchmarked: the weakest correct answer scored
    # 0.477, so this trims noise without losing answers; out-of-scope questions are
    # refused by the answer prompt, not this gate.
    retrieval_score_threshold: float = 0.45
    # Cross-encoder logit floor (reranking on). -11 is the highest value that loses no
    # benchmark hit for ms-marco-MiniLM-L-6-v2; recalibrate for any other reranker.
    rerank_score_threshold: float = -11.0
    max_context_tokens: int = 3000

    history_max_turns: int = 6
    rag_debug: bool = False  # include the retrieval trace in chat responses (dev only)
    preload_models: bool = True

    @property
    def allowed_origins_list(self) -> list[str]:
        """Return the comma-separated `ALLOWED_ORIGINS` as a clean list.

        Returns:
            Allowed CORS origins, whitespace trimmed and empties removed.
        """
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
