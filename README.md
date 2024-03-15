# RAG-qa-your-docs
RAG Q&amp;A tool that helped me with regulatory and compliance project. <br>
This is a small project which I ran on a desktop with a 24Gb VRAM GPU. Probably, 16 Gb VRAM will be enough (yet in this 
case I recommend to change batch size for inference).
# Highlights
The code is a build on top of [LangChain](https://python.langchain.com/docs/get_started/introduction) and [HuggingFace](https://huggingface.co/)
models.The actual inference is organized as [Chains](https://python.langchain.com/docs/modules/chains). I wrote a simple 
class that creates a chain using provided embeddings and LLM models. This allows to decouple actual generation from the 
models allowing to use any other model (e.g. locally served like OLLAMA or a paid alternative via API). 
The vector store is FAISS. <br>
The best way to understand it is to inspect examples. They are very simple.
This code was intended for usage with other tasks. 
The RAG part expects prepared LangChain Document data structure. Therefore, you shall prepare&split it yourself. 
I used a combination of [Semantic Chunker](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker)
and a Recursive Splitter. 
# Requirements
YOu need an environment with:
- transformers >= 4.38
- bitsandbytes >= 0.4
- LangChain >= 0.1.7
- LangChain community >= 0.0.20
- LangChain experimental >= 0.0.51
- Python >=3.10
