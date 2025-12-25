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
                    
                    # Configure with REST transport for better stability on some Windows setups
                    genai.configure(api_key=api_key, transport='rest')
                    
                    # ELITE MULTI-MODEL CHAIN 
                    # We try different models in order of precision/availability
                    # We use the full 'models/' prefix for maximum compatibility
                    model_names = [
                        'models/gemini-1.5-flash', 
                        'models/gemini-1.5-flash-latest',
                        'models/gemini-pro',
                        'models/gemini-2.0-flash-exp'
                    ]
                    
                    last_model_error = ""
                    for model_name in model_names:
                        try:
                            # Using the highest-fidelity configuration
                            generation_config = {
                                "temperature": 0.4,
                                "top_p": 0.95,
                                "top_k": 0,
                                "max_output_tokens": 1024,
                            }
                            model = genai.GenerativeModel(
                                model_name=model_name,
                                generation_config=generation_config
                            )
                            
                            # 2. ELITE KNOWLEDGE BASE (Bilingual Nuance Dataset)
                            tuning_dataset = [
                                {"en": "It's a piece of cake for me.", "ur": "یہ میرے لیے بائیں ہاتھ کا کھیل ہے۔", "ctx": "Idiomatic Efficiency"},
                                {"en": "I'm feeling under the weather.", "ur": "میری طبیعت کچھ ناساز ہے۔", "ctx": "Formal Health"},
                                {"en": "Could you please assist me?", "ur": "کیا آپ میری مدد کر سکتے ہیں؟", "ctx": "Respectful Request"},
                                {"en": "Don't beat around the bush.", "ur": "ادھر ادھر کی باتیں مت کرو، اصل بات پر آؤ۔", "ctx": "Conversational Native"},
                                {"en": "The economy is fluctuating.", "ur": "معیشت میں اتار چڑھاؤ آ رہا ہے۔", "ctx": "Academic/Professional"}
                            ]

                            # 3. ELITE CONSULTANT SYSTEM PROMPT
                            system_prompt = f"""You are an 'Elite' English-Urdu Linguistic Consultant. 
Your translations represent the pinnacle of linguistic accuracy and cultural elegance.

ELITE OPERATING PRINCIPLES:
1. SOUL OF THE MESSAGE: Never translate words. Translate the 'Soul' and 'Intent'.
2. HONORIFIC LOGIC: Always use 'آپ' (Aap) for respect. Use precise gender-noun mappings.
3. NATIVE FLOW: Ensure the Urdu follows the SOV (Subject-Object-Verb) structure naturally.
4. ZERO LITERALISM: Re-read every idiom. 'Under the weather' should NEVER mention 'weather' in Urdu. Use 'طبیعت ناساز'.

KNOWLEDGE BASE (REFERENCE STANDARDS):
{chr(10).join([f"Source: {i['en']} | Target: {i['ur']} ({i['ctx']})" for i in tuning_dataset])}

Output ONLY the final polished translation. No preamble."""

                            # 4. DYNAMIC TASK CONSTRUCTION
                            prompt = f"{system_prompt}\n\n"
                            prompt += f"CONTEXT: Act as a master translator for a {source} to {target} request.\n"
                            
                            if context_history:
                                prompt += "CONVERSATIONAL HISTORY (Analyze for pronoun/gender consistency):\n"
                                for role, content in context_history[-3:]:
                                    prompt += f"{role}: {content}\n"
                            
                            prompt += f"\nINPUT TEXT: {clean_text}\n"
                            prompt += "ELITE RESULT:"

                            response = model.generate_content(prompt)
                            if response.text:
                                return "✨ " + response.text.strip() # Star indicates Elite AI is active
                        except Exception as inner_e:
                            last_model_error = str(inner_e)
                            continue # Try next model in the chain for this key

                    # If we reach here, all models for this specific key failed
                    raise Exception(f"Model chain exhausted for this key. Last error: {last_model_error}")
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "quota" in error_msg or "429" in error_msg or "limit" in error_msg:
                        logger.warning(f"Key {api_key[:8]}... quota exceeded. Rotating...")
                        continue # Try next key
                    else:
                        logger.error(f"Gemini error with key {api_key[:8]}...: {e}")
                        continue

            logger.error("All Gemini API keys failed or exhausted.")
        
        # 5. FINAL TIER: LOCAL FALLBACK (Helsinki-NLP / MarianMT)
        # We skipped Google Translate because it lacks the semantic nuance required for this project.
        logger.info("Using final local fallback for translation.")
        return self._translate_local(text, source, target)
