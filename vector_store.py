import os
import torch
from typing import List, Dict, Any
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# For Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.schema import Document

# Constants
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
CHUNK_SIZE = 512

class VectorStoreManager:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.processed_files = set()  # Track which files have been processed
        self.initialize_embedding_model()
    
    def initialize_embedding_model(self):
        """Initialize the embedding model once"""
        if self.embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                multi_process=False,  # Set to False to avoid multiprocessing issues
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
            )
    
    def get_text_splitter(self):
        """Get the text splitter with proper tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=int(CHUNK_SIZE / 10),
            add_start_index=True,
            strip_whitespace=True
        )
    
    def process_pdf_file(self, file_path: str, file_name: str) -> List[Document]:
        """Process a single PDF file and return documents"""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            
            # Add metadata to identify source file
            for doc in raw_docs:
                doc.metadata['source_file'] = file_name
                doc.metadata['file_path'] = file_path
            
            # Split documents
            text_splitter = self.get_text_splitter()
            docs_processed = []
            for doc in raw_docs:
                docs_processed += text_splitter.split_documents([doc])
            
            return docs_processed
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            return []
    
    def update_vector_store(self, uploaded_files: List[Dict[str, Any]]):
        """Update vector store with new uploaded files"""
        new_files = []
        
        # Check for new files that haven't been processed
        for file_info in uploaded_files:
            if file_info['path'] not in self.processed_files:
                new_files.append(file_info)
        
        if not new_files:
            return  # No new files to process
        
        # Process new files
        all_docs = []
        for file_info in new_files:
            docs = self.process_pdf_file(file_info['path'], file_info['name'])
            all_docs.extend(docs)
            self.processed_files.add(file_info['path'])
        
        if not all_docs:
            return
        
        # Initialize or update vector store
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = FAISS.from_documents(
                all_docs, 
                self.embedding_model, 
                distance_strategy=DistanceStrategy.COSINE
            )
        else:
            # Add documents to existing vector store
            self.vector_store.add_documents(all_docs)
        
        print(f"Vector store updated with {len(all_docs)} new document chunks from {len(new_files)} files")
    
    def get_relevant_docs(self, query: str, k: int = 2) -> List[Document]:
        """Get relevant documents for a query"""
        if self.vector_store is None:
            print("No vector store available. Please upload some PDF files first.")
            return []
        
        try:
            retrieved_docs = self.vector_store.similarity_search(query=query, k=k)
            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the current vector store state"""
        if self.vector_store is None:
            return {
                "status": "not_initialized",
                "processed_files": 0,
                "total_documents": 0
            }
        
        return {
            "status": "active",
            "processed_files": len(self.processed_files),
            "total_documents": self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else "unknown"
        }
    
    def clear_vector_store(self):
        """Clear the vector store and reset state"""
        self.vector_store = None
        self.processed_files.clear()
        print("Vector store cleared")

# Global instance
vector_store_manager = VectorStoreManager() 