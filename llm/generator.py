"""
LLM Response Generator for RAG System
Author: Adham Aboulkheir
"""
import re
from dataclasses import dataclass
from typing import List, Optional
from llm.prompts import TROUBLESHOOTING_PROMPT


@dataclass
class LLMResponse:
    root_cause: str
    resolution_steps: List[str]
    confidence: str
    raw_response: str
    tokens_used: int = 0


class MockLLMGenerator:
    """
    Mock LLM generator for demonstration.
    In production: replace with OpenAI GPT-4 or Anthropic Claude.
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    def generate(self, messages: List[dict]) -> str:
        """Generate a response (mock implementation)."""
        # Extract the user query from messages
        user_msg = ""
        context = ""
        for m in messages:
            if m["role"] == "user" and "Issue reported:" in m["content"]:
                parts = m["content"].split("Issue reported:")
                context = parts[0].replace("Documentation:", "").strip()
                user_msg = parts[1].strip() if len(parts) > 1 else ""
        
        # Generate rule-based response based on keywords
        query_lower = user_msg.lower()
        
        if "e-47" in query_lower or "power" in query_lower:
            return """1. Root Cause: Power supply fault detected (Error E-47)
2. Resolution Steps:
   1. Check main fuse (5A) — replace if blown
   2. Inspect power cable for damage or loose connections
   3. Test power supply output voltage (should be 12V ± 0.5V)
   4. If voltage is out of range, replace the power supply unit
   5. After repair, run self-test sequence to verify
3. Confidence: High"""
        
        elif "led" in query_lower or "red" in query_lower or "heat" in query_lower or "overheat" in query_lower:
            return """1. Root Cause: Device overheating (indicated by red LED flashing)
2. Resolution Steps:
   1. Power off the device immediately
   2. Clear all ventilation slots of dust and obstructions
   3. Check cooling fan operation — replace if not spinning
   4. Allow 30 minutes for device to cool
   5. Restart and monitor temperature via management interface
3. Confidence: High"""
        
        elif "connect" in query_lower or "network" in query_lower or "timeout" in query_lower:
            return """1. Root Cause: Network connectivity issue (possible firewall blocking)
2. Resolution Steps:
   1. Verify device IP address has not changed (check DHCP lease)
   2. Check firewall rules — whitelist device IP on port 8443
   3. Test connectivity: ping device_ip from management PC
   4. Restart network interface on device (Settings > Network > Restart)
   5. If issue persists, check switch port configuration
3. Confidence: Medium"""
        
        elif "firmware" in query_lower or "update" in query_lower:
            return """1. Root Cause: Firmware update failure
2. Resolution Steps:
   1. Ensure stable power supply during update (use UPS if available)
   2. Switch to wired connection — do not use WiFi for firmware updates
   3. Temporarily disable antivirus software
   4. If update fails, enter recovery mode: hold reset for 10 seconds
   5. Re-attempt update from recovery mode
3. Confidence: High"""
        
        else:
            # Generic response using context
            context_snippet = context[:200] if context else "No specific documentation found"
            return f"""1. Root Cause: Issue requires investigation based on reported symptoms
2. Resolution Steps:
   1. Review the relevant documentation: {context_snippet[:100]}...
   2. Check device logs for error codes
   3. Perform basic hardware inspection
   4. Contact technical support if issue persists
3. Confidence: Low"""
    
    def parse_response(self, raw: str) -> LLMResponse:
        """Parse structured response into components."""
        lines = raw.strip().split("\n")
        root_cause = ""
        steps = []
        confidence = "Medium"
        
        for line in lines:
            line = line.strip()
            if line.startswith("1. Root Cause:"):
                root_cause = line.replace("1. Root Cause:", "").strip()
            elif re.match(r"^\s*\d+\.\s", line) and steps is not None:
                step = re.sub(r"^\s*\d+\.\s", "", line).strip()
                if step and "Resolution Steps" not in step:
                    steps.append(step)
            elif line.startswith("3. Confidence:"):
                confidence = line.replace("3. Confidence:", "").strip()
        
        return LLMResponse(
            root_cause=root_cause or "Unknown issue",
            resolution_steps=steps or ["Check device documentation"],
            confidence=confidence,
            raw_response=raw,
            tokens_used=len(raw.split())
        )


if __name__ == "__main__":
    print("LLM Generator Demo")
    gen = MockLLMGenerator()
    
    test_queries = [
        "Device shows error E-47 and power light is off",
        "LED is blinking red and device feels hot",
        "Cannot connect to device over the network",
    ]
    
    for query in test_queries:
        messages = TROUBLESHOOTING_PROMPT.format(
            context="Refer to maintenance manual for error codes.",
            query=query
        )
        raw = gen.generate(messages)
        response = gen.parse_response(raw)
        print(f"\nQuery: {query}")
        print(f"  Root Cause: {response.root_cause}")
        print(f"  Confidence: {response.confidence}")
        print(f"  Steps: {len(response.resolution_steps)}")
