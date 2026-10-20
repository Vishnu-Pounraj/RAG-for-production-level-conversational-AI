import chromadb
import logging
import os
from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from . import constant as CONSTANT
from dotenv import load_dotenv

load_dotenv()

logger=logging.getLogger(__name__)


class HybridRetrieval:
    def __init__(self,collection_name=CONSTANT.CHROMA_COLLECTIONS , model_name="all-MiniLM-L6-v2"):
        """Initializes the dense embedding and BM25 index."""
        logger.info(f"Loading QueryEmbedding: {model_name}...")
        self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Connecting to ChromaDB...")
        self.client = chromadb.CloudClient(api_key=os.getenv("CHROMA_API_KEY"),tenant=os.getenv("TENANT"),database=os.getenv("DATABASE"))
        self.collection_name = CONSTANT.CHROMA_COLLECTIONS
        # self.client=chromadb.PersistentClient(path=CONSTANT.CHROMA_DIRS)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self.corpus_texts=[]
        self.corpus_ids=[]
        self.corpus_metadatas = []
        self.bm25_index=None
        self._bm25_search_index()

    def _bm25_search_index(self):
        """Sync chroma documents to in memory of bm25"""
        data = self.collection.get(include=["documents", "metadatas"])
        if data and data["documents"]:
            self.corpus_texts = data["documents"]
            self.corpus_ids = data["ids"]
            self.corpus_metadatas = data["metadatas"] if data["metadatas"] else [{}] * len(data["documents"])

            tokenized_corpus=[doc.lower().split() for doc in self.corpus_texts] 
            self.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self.bm25_index = None

    def add_documents(self, documents: list[str], ids: list[str], metadatas: list[dict] = None):
        """Adds documents to both ChromaDB and the BM25 index."""
        doc_embeddings = self.embedding.embed_documents(documents)
        self.collection.add(
                documents=documents,
                embeddings=doc_embeddings,
                ids=ids,
                metadatas=metadatas
            )
        self._bm25_search_index()

    def search_dense(self, query: str, top_k: int = 3) -> list[dict]:
        """Runs semantic vector search."""
        query_embedding = self.embedding.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        dense_result = []
        if results["ids"] and results["ids"][0]:
            for doc_id, text, dist, meta in zip(results["ids"][0], results["documents"][0], results["distances"][0], results["metadatas"][0]):
                dense_result.append({
                    "id": doc_id,
                    "text": text,
                    "source": "dense",
                    "score": float(dist)
                })
        return dense_result

    def search_sparse(self, query: str, top_k: int = 3) -> list[dict]:
        """Runs lexical keyword search via BM25."""
        if not self.bm25_index:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        sparse_result = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                sparse_result.append({
                    "id": self.corpus_ids[idx],
                    "text": self.corpus_texts[idx],
                    "source": "sparse",
                    "score": score
                })
        return sparse_result

    def hybrid_search(self, query: str , top_k:int=3) -> list[dict]:
        """
        Runs both searches independently, combines results, 
        and removes duplicates if a document was found by both methods.
        """
        dense_results = self.search_dense(query, top_k=top_k)
        sparse_results = self.search_sparse(query, top_k=top_k)
        
        seen_ids = set()
        combined_result = []
        
        for hit in dense_results + sparse_results:
            if hit["id"] not in seen_ids:
                seen_ids.add(hit["id"])
                combined_result.append(hit)

        return combined_result

if __name__ == "__main__":
    retriever = HybridRetrieval()
    query = "vishnu as backend developer what skills he have?"
    results = retriever.hybrid_search(query, top_k=3)

    for result in results:
        print(result)