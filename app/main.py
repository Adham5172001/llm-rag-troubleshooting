"""
LLM + RAG Troubleshooting System — FastAPI Application
Author: Adham Aboulkheir
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import time

from retrieval.vector_store import build_default_knowledge_base
from retrieval.hybrid_search import HybridSearchEngine
from llm.generator import MockLLMGenerator
from llm.prompts import TROUBLESHOOTING_PROMPT


app = FastAPI(
    title="RAG Troubleshooting System",
    description="LLM + RAG system for intelligent technical issue resolution",
    version="1.0.0"
)

# Initialise components
kb = build_default_knowledge_base()
search_engine = HybridSearchEngine()
search_engine.add_documents(kb.documents)
llm = MockLLMGenerator()


class DiagnoseRequest(BaseModel):
    query: str
    k: int = 5
    include_sources: bool = True


class DiagnoseResponse(BaseModel):
    query: str
    root_cause: str
    resolution_steps: List[str]
    confidence: str
    sources: Optional[List[str]] = None
    processing_time_ms: float


@app.get("/health")
def health_check():
    return {"status": "healthy", "knowledge_base_size": len(kb)}


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(request: DiagnoseRequest):
    """Diagnose a technical issue using RAG."""
    start = time.time()
    
    # Retrieve relevant documents
    results = search_engine.search(request.query, k=request.k)
    
    if not results:
        raise HTTPException(status_code=404, detail="No relevant documentation found")
    
    # Build context
    context = "\n\n".join([
        f"[{r.document.source}] {r.document.content}"
        for r in results
    ])
    
    # Generate response
    messages = TROUBLESHOOTING_PROMPT.format(context=context, query=request.query)
    raw_response = llm.generate(messages)
    parsed = llm.parse_response(raw_response)
    
    elapsed = (time.time() - start) * 1000
    
    return DiagnoseResponse(
        query=request.query,
        root_cause=parsed.root_cause,
        resolution_steps=parsed.resolution_steps,
        confidence=parsed.confidence,
        sources=[r.document.source for r in results] if request.include_sources else None,
        processing_time_ms=elapsed
    )


@app.get("/knowledge-base/stats")
def kb_stats():
    """Get knowledge base statistics."""
    categories = {}
    for doc in kb.documents:
        cat = doc.metadata.get("category", "unknown") if doc.metadata else "unknown"
        categories[cat] = categories.get(cat, 0) + 1
    return {"total_documents": len(kb), "categories": categories}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
