from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn
import re

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("API_KEY_NEW")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

# Load filtered content
with open("extracted_contents_filtered.doc", "r", encoding="utf-8") as f:
    raw_chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

# FastAPI app
app = FastAPI()

class QARequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image, optional

class QAResponse(BaseModel):
    answer: str
    links: List[dict]

@app.post("/api/", response_model=QAResponse)
def answer_question(request: QARequest):
    context = "\n---\n".join(raw_chunks)  

    system_prompt = """
You are a helpful teaching assistant for the IIT Madras Online BSc program.
Answer student questions based only on the course content and forum posts provided.
If unsure, say "I'm not sure" rather than making up an answer.
Include relevant links from the content if possible.
"""

    user_prompt = f"Context:\n{context}\n\nQuestion: {request.question}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer_text = response.choices[0].message.content

        # Dummy links extraction (placeholder)
        links = []

        url_pattern = re.compile(r'(https?://\S+)')
        matches = url_pattern.findall(answer_text)
        links = [{"url": match, "text": "Relevant link"} for match in url_pattern.findall(answer_text)]
        """
        for line in answer_text.splitlines():
            if "http" in line:
                parts = line.split(" ")
                for part in parts:
                    if part.startswith("http"):
                        links.append({"url": part, "text": "Relevant link"})
        """
        """
            match = re.search(r'(https?://[^\s]+)', line)
            if match:
                url = match.group(0)
                links.append({"url": url, "text": "Relevant link"})
        """

        return {"answer": answer_text, "links": links}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("calling_api:app", host="0.0.0.0", port=8000, reload=True)


    """
    Endpoint to answer a question based on the provided text chunks and optional image.
 
    question = request.question
    image = request.image

    # Load the model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Encode the question
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Encode the raw chunks
    chunk_embeddings = model.encode(raw_chunks, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]

    # Get the top 5 most similar chunks
    top_results = torch.topk(similarities, k=5)

    # Prepare the response
    answer = "\n\n".join([raw_chunks[idx] for idx in top_results.indices])
    
    return QAResponse(answer=answer, links=[])

"""