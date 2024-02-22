"""
A wrapper to make a cached embeddings model
2024
"""

from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from typing import Dict, Any

class Embedders:
    def __init__(self, model_id:str,
                 model_cache_dir:str=None,
                 model_kwargs: Dict[str, Any] = None,
                 show_progress:bool=False,
                 embed_cache:str=None
                 ):
        self.__model_id = model_id
        self.__model_cache_dir = model_cache_dir
        self.__model_kwargs = model_kwargs
        self.__show_progress = show_progress
        if embed_cache:
            self.__embed_cache = LocalFileStore(embed_cache)
        else:
            self.__embed_cache = None

    def load(self):
        self.__embeddings_model = HuggingFaceEmbeddings(
            model_name=self.__model_id, cache_folder=self.__model_cache_dir,
            model_kwargs=self.__model_kwargs,
            show_progress=self.__show_progress
        )
        if self.__embed_cache:
            self.embedder = CacheBackedEmbeddings.from_bytes_store(
                self.__embeddings_model,
                self.__embed_cache,
                namespace=self.__model_id)
        else:
            self.embedder = self.__embeddings_model

    def get_embedder(self):
        return self.embedder