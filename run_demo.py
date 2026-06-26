"""
LLM + RAG Troubleshooting System — Demo
Author: Adham Aboulkheir
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from retrieval.vector_store import build_default_knowledge_base
from retrieval.hybrid_search import HybridSearchEngine
from llm.generator import MockLLMGenerator
from llm.prompts import TROUBLESHOOTING_PROMPT


def main():
    print("=" * 60)
    print("LLM + RAG TROUBLESHOOTING SYSTEM DEMO")
    print("Author: Adham Aboulkheir")
    print("=" * 60)
    
    print("\n[1/3] Building knowledge base...")
    kb = build_default_knowledge_base()
    print(f"  Loaded {len(kb)} documents")
    
    print("\n[2/3] Initialising hybrid search engine...")
    engine = HybridSearchEngine(dense_weight=0.6, sparse_weight=0.4)
    engine.add_documents(kb.documents)
    llm = MockLLMGenerator()
    print("  Search engine ready")
    
    print("\n[3/3] Running diagnostic queries...")
    queries = [
        "Device shows error code E-47 and the power LED is off",
        "The device LED is flashing red 3 times and it feels very hot",
        "Cannot connect to device — connection keeps timing out",
        "Firmware update failed halfway through the process",
        "Sensor readings are inconsistent and jumping around",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*60}")
        print(f"Query {i}: {query}")
        print("─" * 60)
        
        # Retrieve
        results = engine.search(query, k=3)
        context = "\n\n".join([f"[{r.document.source}] {r.document.content}" for r in results])
        
        # Generate
        messages = TROUBLESHOOTING_PROMPT.format(context=context, query=query)
        raw = llm.generate(messages)
        response = llm.parse_response(raw)
        
        print(f"Root Cause:  {response.root_cause}")
        print(f"Confidence:  {response.confidence}")
        print("Resolution:")
        for j, step in enumerate(response.resolution_steps[:4], 1):
            print(f"  {j}. {step}")
        print(f"Sources: {[r.document.source for r in results[:2]]}")
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
