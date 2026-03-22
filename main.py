from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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
import uvicorn
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

from src.logger import prepare_logging
prepare_logging()


app = FastAPI()
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
    query: str
    top_k: int = 3
# Mock RAG call - Replace with your actual RAGResume.search_and_summarize
def get_rag_response(query):
    return f"Based on the documents, here is the answer for: {query}. The candidate has extensive experience in Python and Machine Learning."
@app.api_route("/", methods=["GET", "POST"])
async def index(request: Request, query: str = Form(None)):
    answer = None
    
    # Check if the user submitted a question
    if request.method == "POST" and query:
        try:
            # IMPORTANT: Use the 'query' variable directly, NOT 'request.query'
            # We hardcode top_k=3 here for the UI
            answer = rag_system.search_and_summarize(query, top_k=3)
        except Exception as e:
            answer = f"Error processing request: {str(e)}"
       
    # Return the fancy HTML with the results
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "answer": answer, 
        "query": query  # This puts your question back in the textbox after refresh
    })

# --- 3. RUN ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
