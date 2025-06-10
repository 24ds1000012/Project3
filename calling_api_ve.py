from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import os, re, logging, json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import faiss

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY_NEW")
env_path = find_dotenv()
print("Using .env file at:", env_path)
DISCOURSE_EMAIL = os.getenv("DISCOURSE_EMAIL")
DISCOURSE_PASSWORD = os.getenv("DISCOURSE_PASSWORD")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Logging setup
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if not OPENAI_API_KEY:
    raise ValueError("API_KEY_NEW not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loadable global variables
raw_chunks = None
chunk_embeddings = None
index = None
embedder = None
embedding_dim = 384

def load_embeddings_and_index():
    global raw_chunks, chunk_embeddings, index, embedder

    if raw_chunks is not None:
        return  # Already loaded

    with open("extracted_contents_filtered_v1.doc", "r", encoding="utf-8") as f:
        raw_chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

    chunk_embeddings = np.load("chunk_embeddings.npy")
    if chunk_embeddings.shape[1] != embedding_dim:
        raise ValueError("Embedding shape mismatch")

    import faiss  # local import to save memory at startup
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(chunk_embeddings))

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

class QARequest(BaseModel):
    question: str
    image: Optional[str] = None
    link: Optional[str] = None

class QAResponse(BaseModel):
    answer: str
    links: List[dict]

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.get("/api/")
async def api_root():
    return {"message": "API endpoint - POST requests only"}

@app.post("/api/", response_model=QAResponse)
async def answer_question(request_raw: Request):
    try:
        body_text = await request_raw.body()
        data = json.loads(body_text.decode("utf-8"))
        if isinstance(data, str):
            data = json.loads(data)
        request = QARequest(**data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request format: {e}")

    if request.image and (not isinstance(request.image, str) or len(request.image) < 100):
        raise HTTPException(status_code=400, detail="Invalid image data")

    if request.link:
        context = await scrape_discourse_post(request.link)
    else:
        load_embeddings_and_index()
        question_embedding = embedder.encode([request.question])
        D, I = index.search(np.array(question_embedding), k=10)
        context_chunks = [raw_chunks[idx] for idx in I[0]]
        context = "\n---\n".join(context_chunks)

    if not context.strip():
        context = "No relevant content available."

    system_prompt = """
You are a helpful teaching assistant for the IIT Madras Online BSc program.
Answer student questions strictly based only on the course content and forum posts provided.
If an image is provided, use it only if relevant.
Do not use any information outside the given context.
Include relevant links from the content if possible.
"""

    user_message_str = f"Context:\n{context}\n\nQuestion: {request.question}"
    if request.image:
        user_message_str += "\n[Image data included but not shown here]"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_str},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        answer_text = response.choices[0].message.content
        markdown_link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')
        matches = markdown_link_pattern.findall(answer_text)
        links = [{"url": url, "text": text} for text, url in matches]
        return {"answer": answer_text, "links": links}

    except Exception as e:
        logging.exception("OpenAI API call failed")
        raise HTTPException(status_code=500, detail=str(e))

async def scrape_discourse_post(url: str) -> str:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not DEBUG_MODE)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto("https://discourse.onlinedegree.iitm.ac.in/login", timeout=60000)
            await page.fill('#login-account-name', DISCOURSE_EMAIL)
            await page.fill('#login-account-password', DISCOURSE_PASSWORD)
            await page.click("#login-button")
            await page.wait_for_url(lambda url: "login" not in url, timeout=15000)
            await page.goto(url, timeout=60000)
            await page.wait_for_selector(".topic-post .cooked", timeout=30000)
            post_elements = await page.locator(".topic-post .cooked").all()
            all_posts = [await post_el.inner_text() for post_el in post_elements]
            return "\n\n---\n\n".join(post.strip() for post in all_posts)
        except Exception as e:
            logging.exception("Scraping failed")
            raise HTTPException(status_code=500, detail=f"Scraping failed: {e}")
        finally:
            await browser.close()

"""
if __name__ == "__main__":
    uvicorn.run("calling_api_ve:app", host="0.0.0.0", port=8000, reload=DEBUG_MODE)
"""
