import requests

class QueryRewriter:
    def __init__(self, ollama_url: str, ollama_model: str):
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model

    def rewrite(self, query: str) -> str:
        """
        Rewrite the user query using the LLM for clarity and retrieval optimization.
        Raises an exception if rewriting fails.
        """
        
        prompt = f"""
        You are a constitutional-search optimizer.  
        Rewrite the user’s question into **one clear, unambiguous query** that:

        1. Keeps the original legal intent.  
        2. Targets **retrievable clauses** only; **no commentary, no assumed facts, no article numbers unless present in the original**.  
        3. Is **concise** (≤25 words).  
        4. Provide only rewritten query, no additional text or explanation.
        5. Assume that every question is about Pakistan's 1973 Constitution.

        Original: {query}  

        Rewritten (≤25 words):
        """
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()
        rewritten = data.get("response", "").strip()
        if not rewritten:
            raise ValueError("Query rewriting failed: No response from LLM.")
        return rewritten