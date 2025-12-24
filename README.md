# TranslatorGO: English ↔ Urdu Translator

A high-performance, context-aware translation app powered by **Google Gemini** with automatic rate-limit handling and local fallback.

## 🌟 Key Features

- ✅ **Context-Aware Translation** - Maintains conversation history for natural, human-like translations.
- ✅ **API Key Rotation** - Automatically rotates through multiple Gemini API keys to bypass rate limits and "quota exceeded" errors.
- ✅ **Offline Fallback** - Seamlessly switches to local **MarianMT (Helsinki-NLP)** models if the API is unavailable or limits are reached.
- ✅ **Bidirectional Support** - High-quality translation for both English to Urdu and Urdu to English.
- ✅ **Modern Streamlit UI** - Clean, responsive chat-based interface.

## 🛠️ Technical Setup

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration (.env)
Create a `.env` file in the root directory and add your Gemini API keys:
```env
# Add multiple keys separated by commas for automatic rotation
GEMINI_API_KEYS=key1,key2,key3,key4

# Default Settings
DEFAULT_SOURCE_LANG=en
DEFAULT_TARGET_LANG=ur
HISTORY_LIMIT=20
```

### 3. Usage
Run the app locally:
```bash
python -m streamlit run app.py
```

## ☁️ Hosted Deployment (Streamlit Cloud)

To use the multi-key system on Streamlit Cloud, add your keys to the **Secrets** menu in TOML format:

```toml
GEMINI_API_KEYS = "key1,key2,key3,key4"
DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "ur"
HISTORY_LIMIT = 20
```

## 🧠 How It Works

1. **Primary Engine (Gemini)**: The app attempts to translate using the first key in your `GEMINI_API_KEYS` list.
2. **Auto-Rotation**: If a key hits a rate limit (Error 429), the app instantly moves to the next key and retries the request.
3. **Safety Net (Local Models)**: If all keys are exhausted or there is no internet connection, the system falls back to **Helsinki-NLP/opus-mt** models stored locally/on the server.

## 🛡️ License
This project is open-source.
