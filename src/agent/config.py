from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    llm_model: str = Field("models/gemini-2.5-flash", alias="LLM_MODEL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, alias="LLM_MAX_TOKENS")

    langsmith_api_key: str = Field("", alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(False, alias="LANGSMITH_TRACING_V2")
    langsmith_project: str = Field("shopwave-agent", alias="LANGSMITH_PROJECT")

    postgres_host: str = Field("localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("shopwave_agent", alias="POSTGRES_DB")
    postgres_user: str = Field("shopwave", alias="POSTGRES_USER")
    postgres_password: str = Field("change_me_in_production", alias="POSTGRES_PASSWORD")

    chroma_persist_dir: str = Field("./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("shopwave_kb", alias="CHROMA_COLLECTION_NAME")

    agent_max_iterations: int = Field(15, alias="AGENT_MAX_ITERATIONS")
    agent_confidence_threshold: float = Field(0.6, alias="AGENT_CONFIDENCE_THRESHOLD")
    agent_concurrency_limit: int = Field(5, alias="AGENT_CONCURRENCY_LIMIT")
    planner_strict_expected_action: bool = Field(True, alias="PLANNER_STRICT_EXPECTED_ACTION")

    tool_failure_rate: float = Field(0.15, alias="TOOL_FAILURE_RATE")
    tool_timeout_seconds: float = Field(3.0, alias="TOOL_TIMEOUT_SECONDS")
    tool_failure_seed: int = Field(7, alias="TOOL_FAILURE_SEED")
    tool_max_retries: int = Field(3, alias="TOOL_MAX_RETRIES")
    tool_retry_delays_raw: str = Field("1.0,2.0,4.0", alias="TOOL_RETRY_DELAYS")

    data_dir: str = Field("./data", alias="DATA_DIR")
    audit_log_path: str = Field("./audit_log.json", alias="AUDIT_LOG_PATH")
    run_report_path: str = Field("./output/run_report.json", alias="RUN_REPORT_PATH")
    policy_reference_date: str = Field("2024-03-15", alias="POLICY_REFERENCE_DATE")
    kb_top_k: int = Field(3, alias="KB_TOP_K")

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_format: str = Field("json", alias="LOG_FORMAT")

    agent_min_tool_calls: int = Field(3, alias="AGENT_MIN_TOOL_CALLS")
    refund_estimated_days: int = Field(7, alias="REFUND_ESTIMATED_DAYS")
    escalation_assigned_to: str = Field("human_support_queue", alias="ESCALATION_ASSIGNED_TO")

    @field_validator("llm_model", mode="before")
    @classmethod
    def _normalize_llm_model(cls, value: str) -> str:
        model = str(value).strip()
        if not model:
            return "models/gemini-2.5-flash"
        return model if model.startswith("models/") else f"models/{model}"

    @property
    def tool_retry_delays(self) -> list[float]:
        cleaned = self.tool_retry_delays_raw.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        if not cleaned:
            return [1.0, 2.0, 4.0]
        try:
            return [float(part.strip()) for part in cleaned.split(",") if part.strip()]
        except Exception:
            return [1.0, 2.0, 4.0]

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_sync_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()

