from __future__ import annotations

import warnings
# Suppress all FutureWarnings immediately
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
import time
import os
from typing import Optional

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

from config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslatorService:
    """
    Translator service with Groq (Turbo) and local model fallback.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        # Lazy load for local models
        self._local_models = None

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
        Translate text using Groq (Turbo AI) with a local fallback.
        """
        if not text.strip():
            return ""

        target = target_language or self.settings.default_target
        source = source_language or self.settings.default_source
        clean_text = text.strip().replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

        # Define the Elite System Prompt
        tuning_dataset = [
            {"en": "It's a piece of cake for me.", "ur": "یہ میرے لیے بائیں ہاتھ کا کھیل ہے۔"},
            {"en": "I'm feeling under the weather.", "ur": "میری طبیعت کچھ ناساز ہے۔"},
            {"en": "Don't beat around the bush.", "ur": "ادھر ادھر کی باتیں مت کرو، اصل بات پر آؤ۔"}
        ]

        system_prompt = f"""You are an Elite Linguistic Expert specializing in {source} to {target} translation. 
RULES FOR EXCELLENCE:
1. SOUL OF THE MESSAGE: Never translate words. Translate the 'Soul' and 'Intent'.
2. ZERO LITERALISM: Use native idioms/Muhaawras. 'Under the weather' -> 'طبیعت ناساز'.
3. HONORIFIC LOGIC: Use respectful forms (e.g., 'Aap' in Urdu).
4. NATIVE FLOW: Ensure natural structural mapping.

EXAMPLES:
{chr(10).join([f"- {i['en']} -> {i['ur']}" for i in tuning_dataset])}

Output ONLY the polished final translation."""

        # --- TIER 1: GROQ (TURBO AI - FAST & ACCURATE) ---
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                import requests
                logger.info("Attempting Turbo AI (Groq)...")
                headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "llama3-70b-8192",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean_text}
                    ],
                    "temperature": 0.5
                }
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=10)
                if resp.status_code == 200:
                    return "⚡ " + resp.json()['choices'][0]['message']['content'].strip()
                else:
                    logger.warning(f"Groq API returned status {resp.status_code}")
            except Exception as ge:
                logger.error(f"Groq engine error: {ge}")

        # --- TIER 2: LOCAL FALLBACK (SAFE MODE) ---
        logger.info("Using final Safe Mode (Local Fallback).")
        return self._translate_local(text, source, target)
