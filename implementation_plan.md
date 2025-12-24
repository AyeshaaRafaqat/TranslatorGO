The goal is to create a robust, locally running English <-> Urdu translation app using Hugging Face's `MarianMT` models, ensuring it works completely offline after model download.

## User Requirements
- **Local Execution**: No API keys, no internet dependency for translation (after initial download).
- **Models**: Use `Helsinki-NLP/opus-mt-en-ur` and `Helsinki-NLP/opus-mt-ur-en`.
- **Functionality**: English to Urdu and Urdu to English translation.
- **Context**: User requested context awareness. (Note: Standard NMT models are sentence-based. We will implement basic context management in the UI, but model input will focus on the latest message to ensure accuracy).

## Proposed Changes

### 1. Environment & Dependencies
- Ensure `transformers`, `torch`, `sentencepiece`, `sacremoses` are correctly installed and compatible.
- Create a test script to verify model loading and translation without Streamlit first.

### 2. Backend (Translator Service)
- **Model Loading**: Implement robust caching. These models are ~300MB each.
- **Translation Logic**: 
    - Implement `translate_en_to_ur` and `translate_ur_to_en` using `MarianMTModel`.
    - Handle "Context": For strictly local NMT, context concatenation often hurts performance. I will focus on high-quality single-message translation, but I will ensure the UI displays the conversation history correctly.

### 3. Frontend (Streamlit App)
- Add a clear "Loading Models..." indicator.
- Fix any "ModuleNotFoundError" or runtime errors.
- Ensure the selected direction (En->Ur or Ur->En) matches the loaded model.

## Verification Plan
1. Run a standalone Python script to prove models work.
2. Run the Streamlit app.
3. Verify via CLI output or screenshots (if allowed).
