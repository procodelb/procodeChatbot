"""
RAG Engine Module
Handles the Retrieval Augmented Generation logic using LangChain and Google GenAI.
"""
import os
import logging
import time
from typing import List, Optional, Any
from dotenv import load_dotenv

from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=4, max=10),
    stop=stop_after_attempt(3),
)
def get_answer(chain: RetrievalQA, question: str) -> Any:
    return chain.invoke({"query": question})


class RAGChatBot:
    """
    The Brain of the application: Manages Embeddings, Vector Store, and QA Chain.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
        llm: Optional[ChatGoogleGenerativeAI] = None,
    ):
        """
        Initialize the RAG ChatBot with Google GenAI models.
        
        Args:
            api_key (Optional[str]): The Google API key. If not provided, tries to load from env.
            embeddings (Optional[GoogleGenerativeAIEmbeddings]): Pre-created embeddings instance.
            llm (Optional[ChatGoogleGenerativeAI]): Pre-created chat model instance.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables or arguments.")
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in the .env file or provide it via settings.")
            
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=self.api_key,
            )
        
        if llm is not None:
            self.llm = llm
        else:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=self.api_key,
                temperature=0.3,
            )
        
        self.vector_store = None

    def create_vector_store(self, chunks: List[Document]) -> None:
        """
        Creates a local FAISS vector store from document chunks.
        Handles rate limits by processing in batches.

        Args:
            chunks (List[Document]): List of document chunks to embed.
        """
        if not chunks:
            logger.warning("No chunks provided to create vector store.")
            return
            
        try:
            logger.info(f"Creating vector store from {len(chunks)} chunks...")
            
            # Batch processing settings to avoid Rate Limit Errors (429)
            # Free tier often has limits around requests per minute.
            batch_size = 10 
            total_chunks = len(chunks)
            
            # Create vector store with the first batch
            first_batch = chunks[:batch_size]
            self.vector_store = FAISS.from_documents(first_batch, embedding=self.embeddings)
            logger.info(f"Processed initial batch of {len(first_batch)} chunks.")
            
            # Process remaining batches
            if total_chunks > batch_size:
                for i in range(batch_size, total_chunks, batch_size):
                    batch = chunks[i : i + batch_size]
                    if not batch:
                        continue
                        
                    # Sleep to respect API rate limits (avoid 429)
                    time.sleep(2.0) 
                    
                    self.vector_store.add_documents(batch)
                    logger.info(f"Processed batch starting at index {i} ({len(batch)} chunks)")
            
            logger.info("Vector store created successfully.")
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def get_qa_chain(self) -> RetrievalQA:
        """
        Creates and returns a RetrievalQA chain with a custom prompt.

        Returns:
            RetrievalQA: The configured QA chain.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")

        # Custom Prompt Template
        prompt_template = """You are a helpful and intelligent University Tutor. Your goal is to help the student understand their course. Use the provided context pieces to answer the question. If the exact answer isn't there, try to infer it from the context or summarize relevant parts. Only say "I do not know" if the context is completely empty or unrelated.

Context:
{context}

Question:
{question}

Answer:
"""
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        # Create RetrievalQA Chain
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 6}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True 
        )
        
        return chain
