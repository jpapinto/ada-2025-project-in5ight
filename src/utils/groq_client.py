import os
from dotenv import load_dotenv
from groq import Groq
import time

# Load env variables once when the module is imported
load_dotenv()

def get_groq_completion(prompt: str, model: str = "llama-3.3-70b-versatile", temperature: float = 0.1, max_tokens: int = 100) -> str:
    """
    Generic wrapper for Groq API.
    Sends a prompt and returns the clean string response.
    """
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq API Error: {e}")
        return ""