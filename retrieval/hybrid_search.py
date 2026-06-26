"""
Hybrid Search — Dense + Sparse Retrieval
Author: Adham Aboulkheir
"""
import numpy as np
from typing import List
from retrieval.vector_store import TFIDFVectorStore, SearchResult, Document


class HybridSearchEngine:
    """
    Combines dense (semantic) and sparse (keyword) retrieval.
    
    In production: dense = sentence-transformers + FAISS
                   sparse = BM25
    """
    
    def __init__(self, dense_weight: float = 0.6, sparse_weight: float = 0.4):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.dense_store = TFIDFVectorStore(max_features=512)
        self.sparse_store = TFIDFVectorStore(max_features=256)
    
    def add_documents(self, documents: List[Document]) -> None:
        self.dense_store.add_documents(documents)
        self.sparse_store.add_documents(documents)
    
    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Hybrid search combining dense and sparse scores."""
        dense_results  = {r.document.content: r.score for r in self.dense_store.search(query, k=k*2)}
        sparse_results = {r.document.content: r.score for r in self.sparse_store.search(query, k=k*2)}
        
        # Combine scores
        all_docs = set(dense_results.keys()) | set(sparse_results.keys())
        combined = {}
        for doc_content in all_docs:
            d_score = dense_results.get(doc_content, 0.0)
            s_score = sparse_results.get(doc_content, 0.0)
            combined[doc_content] = (self.dense_weight * d_score + 
                                      self.sparse_weight * s_score)
        
        # Sort and return top-k
        sorted_docs = sorted(combined.items(), key=lambda x: -x[1])[:k]
        
        results = []
        for rank, (content, score) in enumerate(sorted_docs, 1):
            # Find original document
            for doc in self.dense_store.documents:
                if doc.content == content:
                    results.append(SearchResult(document=doc, score=score, rank=rank))
                    break
        
        return results


if __name__ == "__main__":
    from retrieval.vector_store import build_default_knowledge_base
    
    print("Hybrid Search Demo")
    kb = build_default_knowledge_base()
    
    engine = HybridSearchEngine(dense_weight=0.6, sparse_weight=0.4)
    engine.add_documents(kb.documents)
    
    query = "power supply problem error code"
    results = engine.search(query, k=3)
    print(f"Query: {query}")
    for r in results:
        print(f"  [{r.rank}] Score={r.score:.3f} | {r.document.content[:80]}...")
