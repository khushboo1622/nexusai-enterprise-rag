"""
backend/chat/llm_provider.py

Central LLM provider — swap between Groq and Gemini via .env
Set MODEL_PROVIDER=groq or MODEL_PROVIDER=gemini in .env

This is the ONLY place that knows which LLM is being used.
All other modules import get_llm() from here.
"""

import logging
from typing import Optional
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None


def get_llm():
    """
    Returns the configured LLM singleton.
    Provider is determined by MODEL_PROVIDER in .env.
    """
    global _llm
    if _llm is not None:
        return _llm

    provider = settings.MODEL_PROVIDER.lower()

    if provider == "groq":
        from llama_index.llms.groq import Groq
        _llm = Groq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
        )
        logger.info(f"LLM initialized: Groq / {settings.GROQ_MODEL}")

    elif provider == "gemini":
        from llama_index.llms.gemini import Gemini
        _llm = Gemini(
            model_name="models/gemini-1.5-flash",
            api_key=settings.GEMINI_API_KEY,
        )
        logger.info(f"LLM initialized: Gemini / gemini-1.5-flash")

    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER: '{provider}'. "
            f"Set MODEL_PROVIDER=groq or MODEL_PROVIDER=gemini in .env"
        )

    return _llm


def reset_llm():
    """Reset LLM singleton — useful for testing."""
    global _llm
    _llm = None