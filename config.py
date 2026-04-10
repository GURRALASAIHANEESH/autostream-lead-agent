from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT: Path = Path(__file__).resolve().parent


SUPPORTED_LLM_PROVIDERS: frozenset[str] = frozenset({"openai", "google", "anthropic"})


class Settings(BaseSettings):
    """Immutable, validated application settings.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",          
        case_sensitive=False,     
    )

    
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"

   
    OPENAI_API_KEY: Optional[SecretStr] = None
    GOOGLE_API_KEY: Optional[SecretStr] = None
    ANTHROPIC_API_KEY: Optional[SecretStr] = None

   
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    
    KB_PATH: Path = _PROJECT_ROOT / "knowledge" / "autostream_kb.md"
    FAISS_INDEX_PATH: Path = _PROJECT_ROOT / ".faiss_index"

    
    MAX_CONVERSATION_TURNS: int = 6  # 6 turns = 12 messages (human + AI)

    
    LOG_LEVEL: str = "INFO"

    
    @field_validator("LLM_PROVIDER")
    @classmethod
    def _normalise_and_validate_provider(cls, value: str) -> str:
        """Normalise to lowercase and reject unknown providers early.
        """
        normalised = value.strip().lower()
        if normalised not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {sorted(SUPPORTED_LLM_PROVIDERS)}, "
                f"got '{value}'"
            )
        return normalised

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Ensure the log level string maps to a real Python logging level."""
        normalised = value.strip().upper()
        if normalised not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                f"LOG_LEVEL must be a standard Python log level, got '{value}'"
            )
        return normalised

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application-wide Settings singleton.
    """
    settings = Settings()

    # Configure the root logger once, at first access.
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger(__name__).debug(
        "Settings loaded — provider=%s  model=%s",
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
    )
    return settings

def get_llm(*, temperature: float = 0.3):
    """Return the configured LangChain ChatModel. Imports are lazy so only
    the active provider's SDK needs to be installed."""
    settings = get_settings()
    provider = settings.LLM_PROVIDER          # already normalised by validator
    model = settings.LLM_MODEL

    if provider == "openai":
        if settings.OPENAI_API_KEY is None:
            raise ValueError(
                "OPENAI_API_KEY must be set in .env or the environment "
                "when LLM_PROVIDER='openai'."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for the OpenAI provider. "
                "Install with:  pip install langchain-openai"
            ) from exc

        return ChatOpenAI(
            model=model,
            api_key=settings.OPENAI_API_KEY,
            temperature=temperature,
            max_retries=2,
        )

    if provider == "google":
        if settings.GOOGLE_API_KEY is None:
            raise ValueError(
                "GOOGLE_API_KEY must be set in .env or the environment "
                "when LLM_PROVIDER='google'."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required for the Google provider. "
                "Install with:  pip install langchain-google-genai"
            ) from exc

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    if provider == "anthropic":
        if settings.ANTHROPIC_API_KEY is None:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set in .env or the environment "
                "when LLM_PROVIDER='anthropic'."
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for the Anthropic provider. "
                "Install with:  pip install langchain-anthropic"
            ) from exc

        return ChatAnthropic(
            model=model,
            anthropic_api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=1024,
        )

    raise ValueError(f"Unhandled LLM_PROVIDER: '{provider}'")  