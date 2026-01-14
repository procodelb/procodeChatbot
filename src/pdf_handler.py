"""
PDF Handler Module
Responsible for loading and processing PDF files.
"""
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles loading and splitting of PDF documents.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the processor with splitting configuration.
        
        Args:
            chunk_size (int): Character limit for each text chunk.
            chunk_overlap (int): Overlap characters between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a PDF from a file path and splits it into chunks.
        
        Args:
            file_path (str): The absolute path to the PDF file.
            
        Returns:
            List[Document]: A list of chunked documents.
        """
        try:
            logger.info(f"Loading PDF from: {file_path}")
            # Use PyPDFLoader as requested
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                logger.warning(f"No content found in PDF: {file_path}")
                return []
            
            empty_pages = [d for d in documents if not d.page_content or not d.page_content.strip()]
            if len(empty_pages) == len(documents):
                print("Warning: PDF might be a scanned image")

            logger.info(f"Splitting {len(documents)} pages into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Successfully created {len(chunks)} chunks.")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
