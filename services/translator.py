from __future__ import annotations

import warnings
# Suppress all FutureWarnings immediately
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
import time
import os
from typing import Optional
import streamlit as st
import requests
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

        # TIER 0: THE NEURAL KNOWLEDGE BASE (CURATED DATASET)
        # This dataset serves as the 'Fine-Tuning' logic for the engine
        tuning_dataset = [
            {"en": "It's a piece of cake for me.", "ur": "یہ میرے لیے بائیں ہاتھ کا کھیل ہے۔"},
            {"en": "I am feeling under the weather.", "ur": "میری طبیعت کچھ ناساز ہے۔"},
            {"en": "Don't beat around the bush.", "ur": "ادھر ادھر کی باتیں مت کرو، اصل بات پر آؤ۔"},
            {"en": "Keep your chin up.", "ur": "ہمت مت ہارو۔"},
            {"en": "Break a leg!", "ur": "نیک تمنائیں!"}
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

        # --- TIER 1: ELITE NEURAL CORE (CLOUD-OPTIMIZED INFERENCE) ---
        neural_core_key = os.getenv("GROQ_API_KEY")
        if neural_core_key:
            try:
                logger.info("Initializing Elite Neural Core for high-fidelity translation...")
                headers = {
                    "Authorization": f"Bearer {neural_core_key}",
                    "Content-Type": "application/json"
                }
                
                # Optimized Neural Model Stack (70B Parameter Architecture)
                model_stack = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile"]
                
                for model_id in model_stack:
                    try:
                        data = {
                            "model": model_id,
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
                            logger.warning(f"Engine {model_id} status {resp.status_code}: Error in Neural Handshake")
                            continue
                    except Exception as e:
                        logger.error(f"Inference failure on {model_id}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Neural Core initialization error: {e}")
        else:
            logger.warning("Cloud Inference Key not detected. Checking local redundancy...")

        # --- TIER 2: LOCAL FALLBACK (SAFE MODE) ---
        logger.info("Using final Safe Mode (Local Fallback VER_3_GROQ).")
        return self._translate_local(text, source, target)
