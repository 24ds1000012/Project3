from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("API_KEY_NEW")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

# Load filtered content
with open("extracted_contents_filtered_v1.doc", "r", encoding="utf-8") as f:
    raw_chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

# Create sentence embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384

try:
    # Try loading precomputed embeddings
    chunk_embeddings = np.load("chunk_embeddings.npy")
    assert chunk_embeddings.shape[1] == embedding_dim
except:
    chunk_embeddings = embedder.encode(raw_chunks)
    np.save("chunk_embeddings.npy", chunk_embeddings)

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(chunk_embeddings))

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ngrok URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QARequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image or local path starting with file://

class QAResponse(BaseModel):
    answer: str
    links: List[dict]

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.get("/api")
async def api_root():
    return {"message": "API endpoint - POST requests only"}


@app.post("/api/", response_model=QAResponse)
def answer_question(request: QARequest):
    # Step 0: If image path is a file, convert it to base64
    image_b64 = None
    if request.image:
        if request.image.startswith("file://"):
            try:
                file_path = request.image[7:]
                with open(file_path, "rb") as f:
                    image_b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read image file: {e}")
        else:
            image_b64 = request.image

    # Step 1: Semantic search to filter chunks
    question_embedding = embedder.encode([request.question])
    D, I = index.search(np.array(question_embedding), k=5)
    selected_chunks = [raw_chunks[i] for i in I[0]]
    context = "\n---\n".join(selected_chunks)

    system_prompt = """
You are a helpful teaching assistant for the IIT Madras Online BSc program.
Answer student questions based only on the course content and forum posts provided.
If unsure, say "I'm not sure" rather than making up an answer.
Include relevant links from the content if possible.
"""

    user_prompt = f"Context:\n{context}\n\nQuestion: {request.question}"

    # Step 2: Compose the message for the OpenAI API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if image_b64:
        messages.append({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{image_b64}"
                }
            }]
        })

    # Step 3: Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )

        answer_text = response.choices[0].message.content

        # Extract markdown-style links
        markdown_link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')
        matches = markdown_link_pattern.findall(answer_text)
        links = [{"url": url, "text": text} for text, url in matches]

        return {"answer": answer_text, "links": links}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("calling_api:app", host="0.0.0.0", port=8000, reload=True)
