# services/comparator.py
from services.llm_groq import get_summary


def compare_texts(text1: str, text2: str) -> str:
    prompt = f"""
Compare the following two texts. Highlight:
1. Key differences in content
2. Any evident bias or opinion
3. Which one is more factual (if possible)

Text 1:
{text1}

Text 2:
{text2}
"""
    return get_summary(prompt)
