from __future__ import annotations

import warnings
# Suppress all FutureWarnings immediately
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
import random
import time
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
                    # 1. TEXT NORMALIZATION
                    clean_text = text.strip().replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
                    
                    # THE ELITE MULTI-PROVIDER LOGIC 
                    # 1. TRY GEMINI (STABLE v1beta)
                    genai.configure(api_key=api_key)
                    
                    # We use the most specific model names for maximum connection probability
                    model_names = [
                        'models/gemini-1.5-flash', 
                        'gemini-1.5-flash',
                        'models/gemini-pro',
                        'gemini-pro'
                    ]
                    
                    for model_name in model_names:
                        try:
                            # 2. ELITE KNOWLEDGE BASE (Bilingual Nuance Dataset)
                            tuning_dataset = [
                                {"en": "It's a piece of cake for me.", "ur": "یہ میرے لیے بائیں ہاتھ کا کھیل ہے۔"},
                                {"en": "I'm feeling under the weather.", "ur": "میری طبیعت کچھ ناساز ہے۔"},
                                {"en": "Don't beat around the bush.", "ur": "ادھر ادھر کی باتیں مت کرو، اصل بات پر آؤ۔"}
                            ]

                            system_prompt = f"""You are an Elite Linguistic Expert. 
Translate the INPUT exactly to {target}. 
NEVER translate literally. Use equivalent Urdu Muhaawras (idioms). 
Maintain the perfect respect level ('Aap' for second person).

EXAMPLES:
{chr(10).join([f"- {i['en']} -> {i['ur']}" for i in tuning_dataset])}

OUTPUT ONLY THE TRANSLATION."""

                            model = genai.GenerativeModel(
                                model_name=model_name,
                                system_instruction=system_prompt if "1.5" in model_name else None
                            )

                            # 4. TASK EXECUTION
                            prompt_content = f"{system_prompt}\n\nTranslate: {clean_text}" if "1.5" not in model_name else clean_text
                            
                            response = model.generate_content(prompt_content)
                            if response.text:
                                return "✨ " + response.text.strip() # Indicates Elite AI
                                
                        except Exception as inner_e:
                            logger.error(f"Try {model_name} failed: {inner_e}")
                            continue 
                            
                    # 2. TRY GROQ (SECONDARY ELITE)
                    groq_key = os.getenv("GROQ_API_KEY")
                    if groq_key:
                        try:
                            import requests
                            headers = {
                                "Authorization": f"Bearer {groq_key}",
                                "Content-Type": "application/json"
                            }
                            data = {
                                "model": "llama3-70b-8192", # Industry standard for accuracy
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": clean_text}
                                ],
                                "temperature": 0.5
                            }
                            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
                            if resp.status_code == 200:
                                return "⚡ " + resp.json()['choices'][0]['message']['content'].strip() # Groq result
                        except Exception as ge:
                            logger.error(f"Groq fallback failed: {ge}")

                    raise Exception("All cloud providers failed.")
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "quota" in error_msg or "429" in error_msg or "limit" in error_msg:
                        logger.warning(f"Key {api_key[:8]}... quota exceeded. Rotating...")
                        continue # Try next key
                    else:
                        logger.error(f"Gemini error with key {api_key[:8]}...: {e}")
                        continue

            logger.error("All Gemini API keys failed or exhausted.")
        
        # 5. FINAL TIER: LOCAL FALLBACK
        logger.info("Using final local fallback for translation.")
        return self._translate_local(text, source, target)
