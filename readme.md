## Resume RAG Chatbot
This project is a Retrieval-Augmented Generation (RAG) chat interface designed to answer specific questions about resumes. Instead of a general AI, this model specifically "learns" from the documents stored in the /data folder to provide accurate, context-aware answers about candidates.
## The Tech Stack

* Orchestration: LangChain for the RAG pipeline.
* LLM & Inference: Groq for lightning-fast processing.
* Embeddings: HuggingFace for document vectorization.
* API: FastAPI to power the backend.

## Getting Started

   1. Environment Setup
   Ensure your virtual environment is active (you should see (.venv) in your terminal).
   2. Install Dependencies
   Before running the app, pull in the necessary libraries:
   
   pip install -r requirements.txt
   
   3. Add Your Data
   Drop the resume files (PDFs/Docs) you want to query into the /data folder.
   4. Run the API
   Start the FastAPI server by running:
   
   python main.py# OR if using uvicorn directly:
   uvicorn main:app --reload
   
   
## Quick Tip
If you're testing the logic via a Jupyter Notebook (.ipynb), make sure your Kernel is set to the .venv interpreter so all the LangChain and Groq imports work correctly.


