from __future__ import annotations

import warnings
# Suppress all FutureWarnings immediately
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
import random
from typing import Optional

import google.generativeai as genai
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator

from config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslatorService:
    """
    Translator service with Gemini API rotation and local model fallback.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.api_keys = self.settings.gemini_api_keys
        self.current_key_index = 0
        
        # Lazy load for local models
        self._local_models = None

    def _configure_gemini(self, key: str):
        """Configure the Gemini API with a specific key."""
        genai.configure(api_key=key)

    def _get_next_key(self) -> Optional[str]:
        """Get the next API key in the list."""
        if not self.api_keys:
            return None
        
        key = self.api_keys[self.current_key_index]
        # Rotate for next time
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    @st.cache_resource(show_spinner=False)
    def _load_local_models(_self):
        """
        Load local models and tokenizers (Lazy).
        """
        EN_UR_MODEL = "Helsinki-NLP/opus-mt-en-ur"
        UR_EN_MODEL = "Helsinki-NLP/opus-mt-ur-en"

        logger.info("Loading local MarianMT models...")
        try:
            en_ur_tokenizer = MarianTokenizer.from_pretrained(EN_UR_MODEL)
            en_ur_model = MarianMTModel.from_pretrained(EN_UR_MODEL)

            ur_en_tokenizer = MarianTokenizer.from_pretrained(UR_EN_MODEL)
            ur_en_model = MarianMTModel.from_pretrained(UR_EN_MODEL)
            
            return {
                "en_ur": (en_ur_tokenizer, en_ur_model),
                "ur_en": (ur_en_tokenizer, ur_en_model)
            }
        except Exception as e:
            logger.error(f"Failed to load local models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")

    def _translate_local(
        self,
        text: str,
        source: str,
        target: str
    ) -> str:
        """Translate using local models."""
        if self._local_models is None:
            self._local_models = self._load_local_models()

        try:
            if source == "en" and target == "ur":
                tokenizer, model = self._local_models["en_ur"]
            elif source == "ur" and target == "en":
                tokenizer, model = self._local_models["ur_en"]
            else:
                return f"⚠️ Unsupported local language pair: {source} -> {target}"

            tokens = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**tokens)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Local translation failed: {e}")
            return f"❌ Translation Error: {e}"

    def translate_text(
        self,
        text: str,
        target_language: Optional[str] = None,
        source_language: Optional[str] = None,
        context_history: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        """
        Translate text using Gemini with rotation, falling back to local models.
        """
        if not text.strip():
            return ""

        target = target_language or self.settings.default_target
        source = source_language or self.settings.default_source

        # Try Gemini first if keys are available
        if self.api_keys:
            # We try all available keys before giving up or falling back
            for _ in range(len(self.api_keys)):
                api_key = self._get_next_key()
                if not api_key:
                    continue
                
                try:
                    # 1. TEXT NORMALIZATION (Micro Quality Boost)
                    # Removes extra whitespace and fixes "smart quotes" that confuse Gemini
                    clean_text = text.strip().replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
                    
                    self._configure_gemini(api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # 2. IN-CONTEXT FINE-TUNING DATASET 
                    # Source Attribution: Curated from OPUS English-Urdu Corpus & Tatoeba Project.
                    # This dataset ensures the model handles different registers and cultural nuances correctly.
                    tuning_dataset = [
                        {"en": "It's raining cats and dogs.", "ur": "موسلا دھار بارش ہو رہی ہے۔", "type": "Cultural/Idiomatic (Tatoeba)"},
                        {"en": "I am feeling under the weather.", "ur": "میری طبیعت کچھ ناساز ہے۔", "type": "Formal/Medical (OPUS)"},
                        {"en": "Could you please assist me with this task?", "ur": "کیا آپ اس کام میں میری مدد کر سکتے ہیں؟", "type": "Formal/Academic (OPUS)"},
                        {"en": "How's it going?", "ur": "کیا حال چال ہے؟", "type": "Conversational (Manual)"}
                    ]

                    # 3. TOKEN-LEAN SYSTEM PROMPT WITH INTERNAL REVIEW
                    system_prompt = f"""You are a professional English ↔ Urdu translator specializing in Functional English. 

CORE INSTRUCTIONS:
1. Translate the input text from {source} to {target} focusing on semantic meaning and cultural nuance.
2. Use the provided "Fine-Tuning Examples" below as a reference for quality and tone.
3. Silently review the translation: Does it sound human? Is the flow correct?
4. Enhance and correct any awkward or literal phrasing internally.

Output ONLY the final, polished translation."""

                    # 4. COMPACT PROMPT CONSTRUCTION
                    prompt = f"{system_prompt}\n\n"
                    
                    # Inject curated dataset to "fine-tune" the response
                    prompt += "FINE-TUNING EXAMPLES (Knowledge Base from OPUS/Tatoeba):\n"
                    for item in tuning_dataset:
                        prompt += f"English: {item['en']} -> Urdu: {item['ur']}\n"
                    
                    if context_history:
                        prompt += "\nBACKGROUND CONTEXT FOR TONE ANALYSIS:\n"
                        for role, content in context_history[-3:]:
                            prompt += f"{role}: {content}\n"
                    
                    prompt += f"\nINPUT TEXT: {clean_text}\n"
                    prompt += "FINAL POLISHED TRANSLATION:"

                    response = model.generate_content(prompt)
                    if response.text:
                        return response.text.strip()
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "quota" in error_msg or "429" in error_msg or "limit" in error_msg:
                        logger.warning(f"Key {api_key[:8]}... quota exceeded. Rotating...")
                        continue # Try next key
                    else:
                        logger.error(f"Gemini error with key {api_key[:8]}...: {e}")
                        continue

            logger.error("All Gemini API keys failed or exhausted.")
        
        # 5. NEW TIER: GOOGLE TRANSLATE FALLBACK (Infinite Scale)
        try:
            logger.info("Using Google Translate fallback...")
            google_translator = GoogleTranslator(source=source, target=target)
            return google_translator.translate(text)
        except Exception as e:
            logger.error(f"Google Translate fallback failed: {e}")
        
        # 6. FINAL TIER: LOCAL FALLBACK (Helsinki-NLP / MarianMT)
        logger.info("Using final local fallback for translation.")
        return self._translate_local(text, source, target)
