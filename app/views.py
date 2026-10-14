import chromadb
import logging
from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import constant as CONSTANT

logger=logging.getLogger(__name__)


class HybridRetrieval:
    def __init__(self,collection_name="" , model_name="all-MiniLM-L6-v2"):
        """Initializes the dense embedding encoder and BM25 index."""
        logger.info(f"Loading QueryEncoder: {model_name}...")
        self.encoder = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Connecting to ChromaDB...")
        self.client=chromadb.PersistentClient(path=CONSTANT.CHROMA_DIRS)
        self.collection = self.client.get_or_create_collection(name=CONSTANT.CHROMA_COLLECTIONS)

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

            tokenized_corpus=[doc.lower.split() for doc in self.corpus_texts] 
            self.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self.bm25_index = None

    def add_documents(self, documents: list[str], ids: list[str], metadatas: list[dict] = None):
        """Adds documents to both ChromaDB and the BM25 index."""
        embeddings = self.encoder.encode(documents).tolist()
        self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
        self._bm25_search_index()

    def search_dense(self, query: str, top_k: int = 3) -> list[dict]:
        """Runs semantic vector search."""
        query_embedding = self.encoder.encode(query).tolist()
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

    def hybrid_search(self, query: str, top_k: int = 3) -> list[dict]:
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

        return combined_result[:top_k]