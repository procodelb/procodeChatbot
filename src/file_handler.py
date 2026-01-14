"""
File Handler Module
Responsible for loading and processing PDF, Word, Excel, and image files.
"""
import logging
import os
from typing import List

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from rapidocr_onnxruntime import RapidOCR
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.ocr = RapidOCR()

    def process_pdf(self, file_path: str) -> List[Document]:
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if not documents:
                return []
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        except Exception:
            logger.exception("Error processing PDF file")
            return []

    def process_docx(self, file_path: str) -> List[Document]:
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            if not documents:
                return []
            chunks = self.text_splitter.split_documents(documents)
            return chunks
        except Exception:
            logger.exception("Error processing DOCX file")
            return []

    def process_excel(self, file_path: str) -> List[Document]:
        try:
            df = pd.read_excel(file_path)
            if df.empty:
                return []
            documents: List[Document] = []
            for index, row in df.iterrows():
                parts = []
                for column in df.columns:
                    value = row[column]
                    if pd.isna(value):
                        continue
                    parts.append(f"{column}: {value}")
                if not parts:
                    continue
                content = ", ".join(parts)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": file_path, "row_index": int(index)},
                    )
                )
            return documents
        except Exception:
            logger.exception("Error processing Excel file")
            return []

    def process_image(self, file_path: str) -> List[Document]:
        try:
            result, _ = self.ocr(file_path)
            texts: List[str] = []
            if result:
                for _, text, _ in result:
                    if text:
                        texts.append(text)
            full_text = "\n".join(texts).strip()
            if not full_text:
                full_text = "Warning: No text detected in image."
            return [Document(page_content=full_text, metadata={"source": file_path})]
        except Exception:
            logger.exception("Error processing image file")
            return []

    def process_file(self, file_path: str) -> List[Document]:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            return self.process_pdf(file_path)
        if extension == ".docx":
            return self.process_docx(file_path)
        if extension == ".xlsx":
            return self.process_excel(file_path)
        if extension in {".png", ".jpg", ".jpeg"}:
            return self.process_image(file_path)
        logger.error("Unsupported file type: %s", extension)
        return []

