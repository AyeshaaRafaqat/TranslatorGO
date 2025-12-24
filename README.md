# English ↔ Urdu Translator

A context-aware chat application for translating between English and Urdu using **Google Gemini** - a powerful AI-context aware translation.

## Features

- ✅ **Context-Aware** - Maintains conversation history for better accuracy
- ✅ **Google Gemini Powered** - Uses advanced AI capability
- ✅ **Backup API Keys** - Automatically rotates through multiple keys to bypass rate limits
- ✅ **Offline Fallback** - Switches to local MarianMT models if API keys are unavailable or exhausted
- ✅ **Bidirectional Translation** - English ↔ Urdu
- ✅ **Modern UI** - Built with Streamlit

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create a `.env` file (or rename `env.example` to `.env`).
3. Add one or more keys (separated by commas):
   ```
   GEMINI_API_KEYS=key1,key2,key3
   ```
   *Note: Single key as `GEMINI_API_KEY` also works.*

### 3. Run the App

```bash
streamlit run app.py
```

## Configuration

Create a `.env` file in the project root to customize settings:

```env
# Google Gemini API Key
GEMINI_API_KEY=your_api_key_here

# Default translation languages
DEFAULT_SOURCE_LANG=en
DEFAULT_TARGET_LANG=ur

# Conversation history limit
HISTORY_LIMIT=20
```

## How It Works

- **Google Gemini**: Primary translation engine. It rotates through your list of keys if one hits a rate limit.
- **Local Fallback**: If Gemini is unavailable, uses **Helsinki-NLP/opus-mt** models via Hugging Face Transformers.
- **Streamlit**: Web UI framework.
- **Session Memory**: JSON-based storage for conversation history.

## Known Limitations

- Gemini Free Tier has rate limits (approx 15 RPM). Using multiple keys helps split the load!
- Local models require ~2GB of RAM.
