import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

_llm = None

def get_llm():
    # lazy-load: connect to HuggingFace API once and reuse
    global _llm
    if _llm is None:
        # HuggingFaceEndpoint calls the HuggingFace Inference API (no local model download)
        endpoint = HuggingFaceEndpoint(
            repo_id=os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
            max_new_tokens=512,
            temperature=0.7,
            task="conversational",
        )
        # ChatHuggingFace wraps the endpoint to support system/user message format
        _llm = ChatHuggingFace(llm=endpoint)
    return _llm
