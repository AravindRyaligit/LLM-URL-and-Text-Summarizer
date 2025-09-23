from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from services.summarizer import summarize_text
import uvicorn
import logging

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, content: str = Form(...)):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    result = summarize_text(content)
    if not result:
        raise HTTPException(status_code=500, detail="Summarization failed.")
    
    return templates.TemplateResponse(
        "index.html", {"request": request, "result": result}
    )

@app.post("/compare")
async def compare(request: Request, text1: str = Form(...), text2: str = Form(...)):
    # Implement comparison logic here (e.g., using embeddings or LLM)
    compare_result = f"Comparison between: '{text1[:50]}...' and '{text2[:50]}...'"
    return templates.TemplateResponse(
        "index.html", {"request": request, "compare_result": compare_result}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)