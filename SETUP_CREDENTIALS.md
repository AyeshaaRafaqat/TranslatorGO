# Setting Up Google Gemini Credentials

## Step 1: Get a Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Click **Create API key**.
3. Copy the key.

## Step 2: Update Your .env File

1. Open the `.env` file in your project directory.
2. You can add multiple keys to bypass rate limits by using `GEMINI_API_KEYS` separated by commas:
   
```env
GEMINI_API_KEYS=key1,key2,key3
```

Alternatively, a single key still works with `GEMINI_API_KEY=your_key`.

## Step 3: Verify the App

1. If the app is running, it should pick up the key (or you might need to restart it).
2. The warning about the missing key should disappear.
