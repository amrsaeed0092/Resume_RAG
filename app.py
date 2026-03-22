from src.data_loader import load_all_documents 
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from langchain_community.vectorstores import FAISS
from src.search import RAGSearch
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

from src.logger import prepare_logging
prepare_logging()



app = FastAPI(title="Resume RAG API")
templates = Jinja2Templates(directory="templates")

# --- Initialize System Globally (Load once, not per request) ---
# This ensures your vector store is ready before the first user asks a question
docs = load_all_documents(data_dir="./data")
embedding = EmbeddingPipeline()
chunks, embeded_data = embedding.chunk_documents(docs)

store = FaissVectorStore("faiss_store")
store.build_from_documents(embeded_data, chunks)

rag_system = RAGSearch(groq_api_key=groq_api_key, embeddings=embeded_data, chunks=chunks)

# Define the data structure for the user's request
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    # This serves the fancy HTML file when you visit http://127.0.0.1:8000
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(request: QueryRequest):
    # Use your existing logic
    answer = rag_system.search_and_summarize(request.question, top_k=request.top_k)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
# Run with: uvicorn main:app --reload --port 8000