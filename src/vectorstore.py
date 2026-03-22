import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline
from langchain_community.vectorstores import FAISS
import logging
logger = logging.getLogger(__name__) 

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir =  os.path.join(os.getcwd(),persist_dir)
        self.index = None
        self.metadata = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_from_documents(self,embeddings, chunks):
        
        if not os.path.exists(os.path.join(self.persist_dir, "index.faiss")):
            os.makedirs(self.persist_dir, exist_ok=True)
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            # Save the index to your D:\Master_RAG directory
            self.vectorstore.save_local(self.persist_dir)
            logger.info(f"[INFO] Vector store built and saved to {self.persist_dir}")
        else:
            self.vectorstore = FAISS.load_local(
                self.persist_dir, 
                embeddings, 
                allow_dangerous_deserialization=True #You must allow_dangerous_deserialization=True because FAISS uses 'pickle' to load the data locally.
            )
            logger.info(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

        return self.vectorstore

    def search(self, query: str, k: int = 3) -> List[Any]:
        """
        Perform a similarity search without an LLM.
        Returns a list of Document objects.
        """
        if not os.path.exists(self.persist_dir):
            
            logger.info(f"[INFO] Vector store not found in {self.persist_dir}, please create the vector index first. ")
            return
        
        # This uses the embedding model to vectorize the query and find matches
        results = self.vectorstore.similarity_search(query, k=k)
        # 2. Print the results
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Content: {doc.page_content[:200]}...") # Print first 200 chars
            print(f"Metadata: {doc.metadata}")
        return results

