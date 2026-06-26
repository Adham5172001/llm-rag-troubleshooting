"""
Prompt Templates for RAG Troubleshooting System
Author: Adham Aboulkheir
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PromptTemplate:
    system: str
    user_template: str
    few_shot_examples: List[dict] = None
    
    def format(self, **kwargs) -> List[dict]:
        messages = [{"role": "system", "content": self.system}]
        if self.few_shot_examples:
            for ex in self.few_shot_examples:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append({"role": "user", "content": self.user_template.format(**kwargs)})
        return messages


TROUBLESHOOTING_PROMPT = PromptTemplate(
    system="""You are an expert technical support engineer for telecom equipment.
Use ONLY the provided documentation to answer questions.
Always structure your response as:
1. Root Cause: [one sentence]
2. Resolution Steps: [numbered list]
3. Confidence: [High/Medium/Low]
If the documentation does not contain relevant information, say so explicitly.""",
    
    user_template="""Documentation:
{context}

Issue reported: {query}

Provide a structured diagnosis and resolution.""",
    
    few_shot_examples=[
        {
            "user": "Device shows error E-47",
            "assistant": """1. Root Cause: Power supply fault (Error E-47)
2. Resolution Steps:
   1. Check main fuse (5A) — replace if blown
   2. Inspect power cable for damage
   3. Test power supply output (should be 12V ± 0.5V)
   4. If voltage is out of range, replace power supply unit
3. Confidence: High"""
        }
    ]
)

STRUCTURED_EXTRACTION_PROMPT = PromptTemplate(
    system="Extract structured information from the text. Return valid JSON only.",
    user_template="Text: {text}\n\nExtract: {fields}\n\nJSON output:"
)

CHAIN_OF_THOUGHT_PROMPT = PromptTemplate(
    system="""You are a technical expert. Think step by step before answering.
Show your reasoning process, then provide the final answer.""",
    user_template="Problem: {problem}\n\nThink through this step by step:"
)


if __name__ == "__main__":
    print("Prompt Templates Demo")
    messages = TROUBLESHOOTING_PROMPT.format(
        context="Error E-47 indicates a power supply fault.",
        query="Device shows error E-47 and won't turn on"
    )
    print(f"Generated {len(messages)} messages:")
    for m in messages:
        print(f"  [{m['role']}]: {m['content'][:80]}...")
