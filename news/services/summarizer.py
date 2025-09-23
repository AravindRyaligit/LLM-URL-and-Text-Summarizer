import logging
from typing import Optional
from services.llm_groq import get_summary

logger = logging.getLogger(__name__)

def summarize_text(text: str) -> Optional[str]:
    """Summarize text with error handling."""
    if not text.strip():
        logger.warning("Empty input text received.")
        return None
    
    try:
        return get_summary(text)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return None