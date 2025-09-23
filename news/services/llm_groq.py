import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer  # For smart chunking
import concurrent.futures  # For parallel URL processing
import openai

# Load Groq API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize tokenizer for chunking
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")

def get_summary(text: str) -> str:
    """Summarize text using Groq's LLM."""
    openai = openai(api_key=os.getenv("OPENAI_API_KEY"))

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or "mixtral-8x7b-32768"
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": f"Summarize this concisely:\n\n{text}"},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content
# --- Core Functions ---
def extract_text_from_url(url):
    """Fetch and clean text from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        if 'application/pdf' in response.headers.get('content-type', ''):
            return None  # Skip PDFs (or add PDF extraction later)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()
        return ' '.join(soup.stripped_strings)
    
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def chunk_text(text, max_tokens=2000):
    """Split text into token-sized chunks."""
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]


def summarize_with_groq(text, is_url=False):
    """Summarize text using Groq's API with proper chunking."""
    if not text:
        return "Error: No content to summarize."

    # Token-based chunking
    chunks = chunk_text(text, max_tokens=1500)  # 1500 input tokens
    full_summary = []

    for chunk in chunks:
        # Use a minimal prompt for each chunk
        messages = [
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": f"Summarize this concisely in 3-5 bullet points:\n\n{chunk}"},
        ]
        
        try:
            stream = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.3,
                max_tokens=512,  # Response tokens
                stream=True
            )

            chunk_summary = ""
            for response in stream:
                chunk_summary += response.choices[0].delta.content or ""
            full_summary.append(chunk_summary)
        
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            continue

    return "\n".join(full_summary)


# --- User-Facing Functions ---
def summarize_text(text):
    """Summarize raw text."""
    return summarize_with_groq(text, is_url=False)

def summarize_url(url):
    """Summarize a URL."""
    text = extract_text_from_url(url)
    return summarize_with_groq(text, is_url=True) if text else "Error: URL content unavailable."

def summarize_urls_parallel(urls, max_workers=5):
    """Process multiple URLs concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(summarize_url, urls))

# --- Example Usage ---
if __name__ == "__main__":
    # Summarize text
    text = "The quick brown fox jumps over the lazy dog..."
    print("Text Summary:", summarize_text(text))
    
    # Summarize URL
    print("URL Summary:", summarize_url("https://en.wikipedia.org/wiki/Large_language_model"))
    
    # Parallel URL processing
    urls = [
        "https://www.nytimes.com/2024/06/01/technology/ai-meta-llama-3.html",
        "https://www.wired.com/story/groq-ai-chips-fast-llm-inference/"
    ]
    print("Parallel Summaries:", summarize_urls_parallel(urls))