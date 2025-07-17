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
        prompt = f"""You are an expert at rewriting questions for optimal search of Pakistan's 1973 Constitution. Your goal is to make the question as clear, specific, and concise as possible for retrieving relevant information from a constitutional database. Do not change the core intent of the question.

        Here are some examples:

        Original question: What is the eligibility criteria to be president?
        Rewritten question: Eligibility criteria for President of Pakistan (1973 Constitution)

        Original question: How is the speaker of the National Assembly elected?
        Rewritten question: Election process for Speaker of the National Assembly (1973 Constitution)

        Original question: Compare the powers of the President and the Prime Minister in dissolving the National Assembly.
        Rewritten question: Comparison of Presidential and Prime Ministerial powers regarding dissolution of the National Assembly (1973 Constitution)

        Original question: How does the Constitution address conflicts between Federal and Provincial laws?
        Rewritten question: Resolution of conflicts between Federal and Provincial laws according to the 1973 Constitution of Pakistan.

        Now, rewrite the question below.

        ---
        Original question: {query}
        ---

        Focus on:
        *   Removing ambiguity.
        *   Using terminology found in the 1973 Constitution.
        *   Ensuring the question can be answered directly from the constitutional text.
        *   Do not add or assume specific Article numbers unless they are already in the original question.
        *   Do not add extraneous information not found in the original question.
        *   Do not include any conversational elements or explanations of your reasoning.

        Rewritten question:
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