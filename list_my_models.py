
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEYS").split(",")[0]

print(f"--- Checking Models for Key: {api_key[:8]}... ---")
genai.configure(api_key=api_key)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Available: {m.name}")
except Exception as e:
    print(f"❌ Error fetching models: {e}")
