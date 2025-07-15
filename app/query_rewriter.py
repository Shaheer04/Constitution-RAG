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
        prompt = (
            "Rewrite the following question to be as clear, specific, and concise as possible, "
            "optimizing it for retrieval from a constitutional database. "
            "Do not change the meaning, but resolve ambiguity and add missing context if needed."
            "Assume the question is about Pakistan's 1973 Constitution unless it is clearly not."
            "If the question mentions an article number (e.g., Article 63), treat it as referring to Pakistan's Constitution.\n\n"
            f"Original question: {query}\n\nRewritten:"
        )
        
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