import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def debug_keys():
    keys_str = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    
    if not keys:
        print("❌ No API keys found in .env")
        return

    model_names = [
        'gemini-2.0-flash-exp', 
        'gemini-2.0-flash', 
        'gemini-1.5-flash', 
        'gemini-1.5-pro', 
        'gemini-pro'
    ]

    for i, key in enumerate(keys):
        print(f"\n🔑 Testing Key {i+1}: {key[:8]}...")
        genai.configure(api_key=key, transport='rest')
        
        for m_name in model_names:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content("Hi", generation_config={"max_output_tokens": 5})
                print(f" ✅ {m_name}: SUCCESS")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f" ⚠️ {m_name}: QUOTA EXCEEDED (429)")
                elif "404" in error_msg:
                    print(f" ❌ {m_name}: NOT FOUND (404)")
                elif "400" in error_msg:
                    print(f" ❌ {m_name}: INVALID REQUEST (400) - {error_msg[:50]}...")
                else:
                    print(f" ❌ {m_name}: ERROR - {error_msg[:100]}...")

if __name__ == "__main__":
    debug_keys()
