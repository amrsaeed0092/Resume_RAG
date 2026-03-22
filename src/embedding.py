from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
from src.data_loader import load_all_documents
import logging
logger = logging.getLogger(__name__) 

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks, self.embeddings


