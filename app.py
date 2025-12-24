import uuid

import streamlit as st

from config import get_settings
from services.memory import MemoryService
from services.translator import TranslatorService


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def main() -> None:
    st.set_page_config(page_title="English <-> Urdu Translator")
    st.title("English <-> Urdu Translator")
    st.caption("Context-aware translation")  
    # Initialize services
    try:
        settings = get_settings()
        memory = MemoryService()
        
        # TranslatorService now handles both Gemini (preferred) and Local (fallback)
        translator = TranslatorService()

            
    except Exception as exc:
        st.error(f"⚠️ Initialization error: {exc}")
        st.info("The app will continue to run, but translation may not work. Check your internet connection and try again.")
        # Use defaults if settings fail
        from dataclasses import dataclass
        @dataclass
        class DefaultSettings:
            default_source: str = "en"
            default_target: str = "ur"
        settings = DefaultSettings()
        memory = MemoryService()
        translator = None
    
    session_id = get_session_id()

    direction = st.radio(
        "Translation direction",
        options=[
            ("en", "ur", "English to Urdu"),
            ("ur", "en", "Urdu to English"),
        ],
        format_func=lambda x: x[2],
        horizontal=True,
        index=0 if settings.default_source == "en" else 1,
    )
    source_lang, target_lang, _ = direction

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Clear conversation", type="secondary"):
            memory.clear_history(session_id)
            st.session_state.pop("history_cache", None)
            st.success("Conversation cleared.")
            st.stop()
    with col2:
        st.write(f"Source: `{source_lang}` -> Target: `{target_lang}`")

    history = memory.get_history(session_id)

    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Add a separator and note about the input box
    if not history:
        st.info("💬 **Type your message in the input box below** (scroll down if you don't see it)")
    
    st.divider()
    
    # Chat input - this should always be visible at the bottom
    user_input = st.chat_input("Type a message to translate")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        if translator is None:
            with st.chat_message("assistant"):
                st.error("Translation service is not available. Please check your connection and try again.")
        else:
            try:
                # Build context from conversation history for context-aware translation
                context_history = []
                for msg in history:
                    context_history.append((msg["role"], msg["content"]))
                
                # Translate with context
                translated = translator.translate_text(
                    user_input,
                    target_language=target_lang,
                    source_language=source_lang,
                    context_history=context_history,
                )
                with st.chat_message("assistant"):
                    st.write(translated)
                memory.append_message(session_id, "user", user_input)
                memory.append_message(session_id, "assistant", translated)
            except Exception as exc:  # broad catch to surface errors to UI
                with st.chat_message("assistant"):
                    st.error(f"Translation failed: {exc}")


if __name__ == "__main__":
    main()
