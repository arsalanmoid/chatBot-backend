import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

_embeddings = None
_vectorstore = None

def get_embeddings():
    # lazy-load: create once and reuse — avoids reconnecting on every request
    global _embeddings
    if _embeddings is None:
        # converts text into 384-dimensional vectors via HuggingFace API
        _embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
        )
    return _embeddings

def _ensure_index():
    # create the Pinecone index if it doesn't exist yet
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "study-materials")
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,   # must match the embedding model output size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

def get_vectorstore():
    # returns a LangChain wrapper around Pinecone for storing and searching chunks
    global _vectorstore
    if _vectorstore is None:
        _ensure_index()
        _vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME", "study-materials"),
            embedding=get_embeddings(),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )
    return _vectorstore

def count() -> int:
    # returns total number of chunks stored in Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "study-materials"))
    return index.describe_index_stats().get("total_vector_count", 0)
