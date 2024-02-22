"""
Convenience wrapper for initializing/loading FAISS vector store.
2024
"""

from langchain.vectorstores import FAISS
from  langchain.schema import Document
from typing import Iterable
import os

class VectorDB:
    def __init__(self, embedder,
                 vector_store_location: str="./vector_db"):
        self.embedder = embedder
        self.persist = vector_store_location
        os.makedirs(self.persist, exist_ok=True)

    def create(self, documents: Iterable[Document]):
        self.db = FAISS.from_documents(documents, self.embedder)
        self.save()
    def load(self):
        self.db = FAISS.load_local(self.persist, self.embedder)

    def save(self):
        self.db.save_local(self.persist)

    def add_documents(self,documents: Iterable[Document]):
        self.db.add_documents(documents)
        self.save()