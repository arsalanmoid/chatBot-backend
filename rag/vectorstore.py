import os
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

EMBEDDING_DIM = 384
_embeddings = None
_index = None

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

def get_index():
    # lazy-load: connect to Pinecone and create index if it doesn't exist
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "study-materials")
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,  # must match the embedding model output size
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        _index = pc.Index(index_name)
    return _index

def add_documents(documents: list, ids: list = None):
    # embed all chunks and upsert into Pinecone with metadata
    texts = [d.page_content for d in documents]
    embs = get_embeddings().embed_documents(texts)
    vectors = [
        {
            "id": ids[i] if ids else f"doc_{i}",
            "values": embs[i],
            "metadata": {**documents[i].metadata, "text": documents[i].page_content},
        }
        for i in range(len(documents))
    ]
    index = get_index()
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])

def similarity_search(query: str, k: int = 5) -> list:
    # embed the query and find top-k most similar chunks in Pinecone
    emb = get_embeddings().embed_query(query)
    results = get_index().query(vector=emb, top_k=k, include_metadata=True)
    docs = []
    for match in results.get("matches", []):
        text = match["metadata"].get("text", "")
        meta = {key: val for key, val in match["metadata"].items() if key != "text"}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

def count() -> int:
    # returns total number of chunks stored in Pinecone
    return get_index().describe_index_stats().get("total_vector_count", 0)
