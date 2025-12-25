import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def test_keys():
    keys_str = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    
    if not keys:
        print("No API keys found in .env")
        return

    for i, key in enumerate(keys):
        print(f"\nTesting Key {i+1}: {key[:8]}...")
        genai.configure(api_key=key)
        try:
            models = genai.list_models()
            print("Available models:")
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    print(f" - {m.name}")
        except Exception as e:
            print(f"Error listing models: {e}")

if __name__ == "__main__":
    test_keys()
