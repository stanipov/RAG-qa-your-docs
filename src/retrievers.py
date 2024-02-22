"""
Convenience wrapper for retriever with long-context reordering

MMR search will be used by default
    https://python.langchain.com/docs/modules/model_io/prompts/example_selector_types/mmr
    https://medium.com/tech-that-works/maximal-marginal-relevance-to-rerank-results-in-unsupervised-keyphrase-extraction-22d95015c7c5

See:
    https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
2024
"""
from langchain_community.document_transformers import LongContextReorder

class ComposedRetriever:
    # KWs for MMR search
    def_kw = {
        # Amount of documents to return (Default: 4)
        "k": 40,
        # Amount of documents to pass to MMR algorithm (Default: 20)
        "fetch_k": 40,
        # 1 for minimum diversity and 0 for maximum. (Default: 0.5)
        "lambda_mult": 0.5
    }

    def __init__(self, vector_store, search_type:str="mmr", **search_kwargs):
        """
        Uses the vector store as a retriever and adds long contex reorderer
        search_type: Can be "similarity", "mmr" (default), or "similarity_score_threshold"
        search_kwargs: kwargs for the search, see LangChain documentation
        """

        self.retriever = vector_store.as_retriever(search_type=search_type,
                                                   search_kwargs=search_kwargs)
        self.reordering = LongContextReorder()

    def get_docs(self, query, *,
                 callbacks: 'Callbacks' = None,
                 tags: 'Optional[List[str]]' = None,
                 metadata: 'Optional[Dict[str, Any]]' = None,
                 run_name: 'Optional[str]' = None,
                 **kwargs: 'Any', ):

        """
        Returns reordered documents.
        """
        docs = self.retriever.get_relevant_documents(query, callbacks=callbacks,
                                                     tags=tags,
                                                     metadata=metadata,
                                                     run_name=run_name,
                                                     **kwargs)
        return self.reordering.transform_documents(docs)