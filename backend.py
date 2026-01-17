from fastapi.responses import HTMLResponse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ssl
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import openai

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()

# Set OpenAI API key
OPENAI_API_KEY = ""
if not OPENAI_API_KEY:
    raise Exception("Please set OPENAI_API_KEY environment variable")
openai.api_key = OPENAI_API_KEY

# Scraping function
def scrape_vit_website(sitemap_url="https://vit.ac.in/files/sitemap.html", max_pages=15):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(sitemap_url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Sitemap fetch error: {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    links = []
    for link in soup.find_all("a"):
        url = link.get("href")
        if url and not url.startswith(("javascript:", "mailto:", "#")):
            full_url = urljoin(sitemap_url, url)
            links.append(full_url)

    documents = []
    for i, url in enumerate(links[:max_pages]):
        try:
            page_resp = requests.get(url, headers=headers, verify=False, timeout=15)
            page_resp.raise_for_status()
            page_soup = BeautifulSoup(page_resp.content, "html.parser")
            for elem in page_soup(["script", "style", "nav", "footer", "header", "iframe", "img"]):
                elem.decompose()
            text = page_soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            if len(text) > 100:
                documents.append({
                    "url": url,
                    "title": page_soup.title.string if page_soup.title else url,
                    "content": text[:8000]
                })
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return documents

# Build vector index
class VectorIndex:
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def build_index(self, documents):
        self.documents = documents
        contents = [d["content"] for d in documents]
        embeddings = self.model.encode(contents, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))

    def search(self, query, k=3):
        query_emb = self.model.encode([query]).astype("float32")
        D, I = self.index.search(query_emb, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.documents):
                results.append({"document": self.documents[idx], "score": float(dist)})
        return results

# Initialize index on startup
documents = scrape_vit_website()
vector_index = VectorIndex()
vector_index.build_index(documents)

# Pydantic model for requests
class QueryRequest(BaseModel):
    question: str

def openai_chat_completion(messages, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response
        except Exception as e:
            print(f"OpenAI attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise e


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>VITChat</title></head>
        <body>
            <h1>✅ VITChat API is running!</h1>
            <p>Try POSTing to <code>/chat</code> with your message.</p>
        </body>
    </html>
    """

"""Hello there"""

@app.post("/ask")
def ask_question(req: QueryRequest):
    question = req.question
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    results = vector_index.search(question, k=3)
    context_text = "\n\n".join([f"Source: {r['document']['title']}\nContent: {r['document']['content']}" for r in results]) or "No relevant documents found."

    messages = [
        {"role": "system", "content": f"You are a helpful assistant for VIT Vellore. Use only the provided context to answer.\nContext:\n{context_text[:3000]}"},
        {"role": "user", "content": question}
    ]

    try:
        response = openai_chat_completion(messages)
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    return {
        "answer": answer,
        "sources": [{"title": r["document"]["title"], "url": r["document"]["url"]} for r in results]
    }
