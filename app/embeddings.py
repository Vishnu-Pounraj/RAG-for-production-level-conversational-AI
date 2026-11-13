from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

@lru_cache(maxsize=1)
def get_embeddings():
    model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return model