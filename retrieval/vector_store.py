"""
Vector Store for RAG System
Author: Adham Aboulkheir
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    content: str
    source: str
    metadata: dict = None
    embedding: np.ndarray = None


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


class TFIDFVectorStore:
    """
    TF-IDF based vector store for document retrieval.
    In production: replace with FAISS + sentence-transformers.
    """
    
    def __init__(self, max_features: int = 1024):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store and build index."""
        self.documents.extend(documents)
        texts = [d.content for d in self.documents]
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
    
    def add_document(self, document: Document) -> None:
        """Add a single document (requires re-indexing)."""
        self.documents.append(document)
        texts = [d.content for d in self.documents]
        self.embeddings = self.vectorizer.fit_transform(texts).toarray()
    
    def search(self, query: str, k: int = 5,
               min_score: float = 0.01) -> List[SearchResult]:
        """Search for most relevant documents."""
        if not self.documents:
            return []
        
        q_vec = self.vectorizer.transform([query]).toarray()
        scores = cosine_similarity(q_vec, self.embeddings)[0]
        
        top_indices = np.argsort(scores)[::-1]
        results = []
        for rank, idx in enumerate(top_indices[:k], 1):
            if scores[idx] >= min_score:
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    rank=rank
                ))
        
        return results
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __repr__(self) -> str:
        return f"TFIDFVectorStore(n_docs={len(self)}, vocab_size={self.max_features})"


def build_default_knowledge_base() -> TFIDFVectorStore:
    """Build a sample knowledge base for telecom troubleshooting."""
    store = TFIDFVectorStore()
    
    documents = [
        Document("Error E-47 indicates a power supply fault. Check the main fuse (5A) and power cable connections. Replace fuse if blown. Test power supply output: should be 12V ± 0.5V.", "maintenance_manual_v3.pdf", {"error_code": "E-47", "category": "power"}),
        Document("LED flashing red (3 blinks) indicates overheating. Ensure ventilation slots are clear. Check cooling fan operation. Allow 30 minutes to cool before restarting.", "troubleshooting_guide.pdf", {"symptom": "LED_RED", "category": "thermal"}),
        Document("Connection timeout errors often caused by firewall rules blocking port 8443. Check network configuration and whitelist the device IP address in your firewall settings.", "network_guide.pdf", {"category": "network"}),
        Document("Firmware update failure: ensure stable power during update, use wired connection, disable antivirus temporarily. If update fails, use recovery mode by holding reset for 10 seconds.", "firmware_guide.pdf", {"category": "firmware"}),
        Document("Calibration drift detected: recalibrate using factory reset procedure. Hold reset button for 10 seconds until LED turns blue. Then run auto-calibration from settings menu.", "calibration_manual.pdf", {"category": "calibration"}),
        Document("Signal loss on channel 3: check antenna connections and cable integrity. Verify signal strength is above -70 dBm. Replace damaged cables with shielded coaxial cable.", "signal_guide.pdf", {"category": "signal"}),
        Document("Device not responding to remote commands: check network connectivity, verify device IP has not changed, restart the management service on the control server.", "remote_guide.pdf", {"category": "remote"}),
        Document("Battery backup failure: test battery voltage (should be 12V). Replace battery every 3 years. Check charging circuit if battery drains faster than expected.", "battery_guide.pdf", {"category": "battery"}),
        Document("High error rate on data transmission: check for electromagnetic interference, verify cable shielding, reduce transmission speed if errors persist above 0.1%.", "transmission_guide.pdf", {"category": "transmission"}),
        Document("Sensor reading anomaly: clean sensor surface with isopropyl alcohol, check for physical damage, verify sensor calibration date. Replace sensor if readings are inconsistent.", "sensor_guide.pdf", {"category": "sensor"}),
    ]
    
    store.add_documents(documents)
    return store


if __name__ == "__main__":
    print("Vector Store Demo")
    store = build_default_knowledge_base()
    print(f"Knowledge base: {store}")
    
    queries = [
        "Device shows error E-47 and power light is off",
        "LED is blinking red and device is hot",
        "Cannot connect to device over network",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = store.search(query, k=2)
        for r in results:
            print(f"  [{r.rank}] Score={r.score:.3f} | {r.document.source}")
            print(f"      {r.document.content[:80]}...")
