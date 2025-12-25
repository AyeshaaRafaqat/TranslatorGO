import streamlit as st
from config import get_settings
from services.translator import TranslatorService
from services.memory import MemoryService
import uuid

def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]

def main() -> None:
    st.set_page_config(page_title="TranslatorGO", layout="wide")
    
    # Precise CSS to match the reference image theme
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .main { background-color: #000000; }
        .label-text {
            color: #FFFFFF;
            font-weight: bold;
            font-size: 24px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .output-box {
            padding: 15px;
            border-radius: 5px;
            background-color: #121212;
            border: 1px solid #333333;
            min-height: 250px;
            color: #AAAAAA;
            font-size: 20px;
            word-wrap: break-word;
        }
        .rtl { direction: rtl; text-align: right; }
        div[role="radiogroup"] > label { color: white !important; }
        </style>
    """, unsafe_allow_html=True)

    # Initialize services
    try:
        settings = get_settings()
        translator = TranslatorService()
        memory = MemoryService()
    except Exception as e:
        st.error(f"Error: {e}")
        return

    session_id = get_session_id()

    # History cleanup for memory
    if st.sidebar.button("Clear History"):
        memory.clear_history(session_id)
        st.rerun()

    direction = st.radio(
        "Translation Direction",
        options=["English → Urdu", "Urdu → English"],
        label_visibility="collapsed",
        index=0
    )

    is_ur_input = "Urdu → English" in direction
    source_lang = "ur" if is_ur_input else "en"
    target_lang = "en" if is_ur_input else "ur"

    st.write("") # Spacing

    # Parallel UI within a Form for "Translate Now" button
    with st.form("translation_panel"):
        col_in, col_out = st.columns(2)
        
        with col_in:
            st.markdown('<p class="label-text">INPUT</p>', unsafe_allow_html=True)
            user_text = st.text_area(
                "input",
                placeholder="Enter text to translate...",
                height=250,
                label_visibility="collapsed"
            )
        
        with col_out:
            st.markdown('<p class="label-text">OUTPUT</p>', unsafe_allow_html=True)
            output_placeholder = st.empty()
            output_placeholder.markdown('<div class="output-box">Translation will appear here...</div>', unsafe_allow_html=True)

        # The "Translate Now" Button
        submit = st.form_submit_button("Translate Now", use_container_width=True, type="primary")

    if submit:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Processing context-aware translation..."):
                try:
                    # Get history for context awareness
                    history = memory.get_history(session_id)
                    context_history = [(msg["role"], msg["content"]) for msg in history]

                    # Translate using rotated keys and the "Heavy" Prompt
                    translated_text = translator.translate_text(
                        user_text,
                        target_language=target_lang,
                        source_language=source_lang,
                        context_history=context_history
                    )

                    # Update UI
                    alignment_class = "rtl" if target_lang == "ur" else "ltr"
                    
                    # Detect engine based on the ✨ marker
                    is_elite = translated_text.startswith("✨")
                    display_text = translated_text.replace("✨ ", "").strip()
                    engine_label = "Elite AI (Gemini)" if is_elite else "Safe Mode (Offline Fallback)"
                    engine_color = "#00FF00" if is_elite else "#FFA500"

                    output_placeholder.markdown(f"""
                        <div style="color: {engine_color}; font-size: 14px; margin-bottom: 5px;">
                            Engine: {engine_label}
                        </div>
                        <div class="output-box {alignment_class}">
                            {display_text}
                        </div>
                    """, unsafe_allow_html=True)

                    # Save to memory for future context
                    memory.append_message(session_id, "user", user_text)
                    memory.append_message(session_id, "assistant", translated_text)

                except Exception as e:
                    st.error(f"Translation failed: {str(e)}")

if __name__ == "__main__":
    main()
