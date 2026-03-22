import os
from dotenv import load_dotenv

from src.data_loader import load_all_documents 
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
import logging
logger = logging.getLogger(__name__) 

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", llm_model: str = "llama-3.1-8b-instant", groq_api_key: str = "", embeddings =None, chunks = None):

        self.persist_dir = persist_dir
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "index.faiss")
        meta_path = os.path.join(persist_dir, "index.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            logger.info(f"[INFO] Vector store is not found in {self.persist_dir}, please create the vector store first")
        else:
             self.vectorstore = FAISS.load_local(
                self.persist_dir, 
                embeddings, 
                allow_dangerous_deserialization=True #You must allow_dangerous_deserialization=True because FAISS uses 'pickle' to load the data locally.
             )
             logger.info(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")
        
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        logger.info(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        # Define the RAG Prompt
        template = """You are an expert Technical Recruiter. 
        Use the provided Resume Context to answer the Hiring Manager's question.
        If the information isn't in the context, say you don't know—don't make up skills.

        Resume Context: {context}
        Hiring Manager Question: {query}
        Helpful Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the Chain to format and clean the output
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "query": RunnablePassthrough()}

            | prompt
            | self.llm
            | StrOutputParser()
        )
        result = rag_chain.invoke(query)

        return result

# Example usage
'''
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "Summarize this resume in bullet points?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    logger.info("Summary:", summary)
'''
