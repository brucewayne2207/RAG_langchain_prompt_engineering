# import os
# from langchain_community.document_loaders import PyMuPDFLoader,PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pathlib import Path
#
# # Creating a function to read all pdfs from a directory
# def process_all_pdf(pdf_directory):
#     all_docs=[]
#     pdf_dir = Path(pdf_directory)
#
#     pdf_files=list(pdf_dir.glob("**/*.pdf"))
#     print(f"Found {len(pdf_files)} PDF files to process")
#
#     for pdf_file in pdf_files:
#         print(f"\nProcessing: {pdf_file.name}" )
#         try:
#             loader=PyMuPDFLoader(str(pdf_file))
#             docs=loader.load()
#
#             for doc in docs:
#                 doc.metadata['source_file']=pdf_file.name
#                 doc.metadata['File_type']='pdf'
#
#             all_docs.extend(docs)
#             print(f"Loaded {len(docs)} pages")
#
#         except Exception as e:
#             print(f"Error:{e}")
#
#     print(f"Total documents loaded {len(all_docs)}")
#     return all_docs
#
# all_pdf_documents=process_all_pdf("C:/Users/ricky/PycharmProjects/DS/DS_6/data")
# # print(all_pdf_documents)
#
# # Chunking process
# def split_documents(document,chunk_size=1000,chunk_overlap=200):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,
#                                                  chunk_overlap=chunk_overlap,
#                                                  length_function=len,
#                                                  separators=["\n\n","\n"," ","''"])
#     split_docs=text_splitter.split_documents(document)
#     print(f'Split {len(document)} into {len(split_docs)} chunk')
#
#
#
#     if split_docs:
#         print(f"\nExample chunk:")
#         print(f"Content: {split_docs[0].page_content[:200]} ... ")
#         print(f"Metadata: {split_docs[0].metadata}")
#
#     return split_docs
#
# chunk=split_documents(all_pdf_documents)
# print(chunk)
#

# embedding ang vectorDB
import os
import numpy as np
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from typing import List,Tuple,Dict,Any
import chromadb
from chromadb.config import Settings
from langchain_text_splitters

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""

    def _init_(self, model_name: str = "all-MiniLM-L6-v2"):

          """Initialize the embedding manager
          Args:
          model_name: HuggingFace model name for sentence embeddings"""

    self.model_name = model_name
    self.model = None
    self._load_model()


def _load_model(self):
    """Load the SentenceTransformer model"""
    try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    except Exception as e:
     print(f"Error loading model {self.model_name}: {e}")
    raise

def generate_embeddings(self, texts: List[str]) -> np.ndarray:

            """Generate embeddings for a list of texts
            Args:
              texts: List of text strings to embed
            Returns:
              numpy array of embeddings with shape (len(texts), embedding_dim)"""

            if not self.model:
             raise ValueError("Model not loaded")

            print(f"Generating embeddings for {len(texts)} texts ... ")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings

emb_manager=EmbeddingManager()
print(emb_manager)