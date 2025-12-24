from __future__ import annotations

import logging
import random
from typing import Optional

import google.generativeai as genai
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

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
                    
                    # 2. TOKEN-LEAN SYSTEM PROMPT WITH INTERNAL REVIEW
                    # This instructs Gemini to perform a 3-step quality check internally before outputting.
                    system_prompt = """You are a professional English ↔ Urdu translator. 

1. Translate the input text focusing on semantic meaning and cultural nuance.
2. Silently review the translation: Does it sound human? Is the flow correct? Does it match the context?
3. Enhance and correct any awkward or literal phrasing internally.

Output ONLY the final, polished translation. No explanations or extra sentences."""

                    # 3. COMPACT PROMPT CONSTRUCTION
                    prompt = f"{system_prompt}\n\nTranslate this text from {source} to {target}.\n"
                    
                    if context_history:
                        prompt += "Background Context:\n"
                        # Only take last 3 messages to keep prompt small and relevant
                        for role, content in context_history[-3:]:
                            prompt += f"{role}: {content}\n"
                    
                    prompt += f"\nInput Text: {clean_text}\n"
                    prompt += "Final Translation:"

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
                        # If it's not a quota error, it might be a general error. 
                        # We still rotate to see if another key/project works.
                        continue

            logger.error("All Gemini API keys failed or exhausted.")
        
        # Fallback to local models
        logger.info("Using local fallback for translation.")
        return self._translate_local(text, source, target)
