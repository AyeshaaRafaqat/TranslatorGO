import streamlit as st
from config import get_settings
from services.translator import TranslatorService

def main() -> None:
    st.set_page_config(page_title="TranslatorGO", layout="wide")
    
    # Clean CSS for Parallel UI
    st.markdown("""
        <style>
        .stTextArea textarea {
            font-size: 18px !important;
        }
        .rtl {
            direction: rtl;
            text-align: right;
            font-family: inherit;
        }
        .output-box {
            padding: 20px;
            border-radius: 8px;
            background-color: #1a1c24;
            border: 1px solid #30363d;
            min-height: 250px;
            font-size: 18px;
            white-space: pre-wrap;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("English ↔ Urdu Translator")
    st.caption("Context-aware semantic translation with Gemini & Local Fallback")

    # Initialize service
    try:
        settings = get_settings()
        translator = TranslatorService()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return

    # Language Direction
    direction = st.radio(
        "Translation Direction",
        options=["English → Urdu", "Urdu → English"],
        horizontal=True
    )

    is_ur_input = "Urdu → English" in direction
    source_lang = "ur" if is_ur_input else "en"
    target_lang = "en" if is_ur_input else "ur"

    # Parallel Interface
    with st.form("translation_form", clear_on_submit=False):
        c_in, c_out = st.columns(2)
        
        with c_in:
            st.subheader("Input")
            user_text = st.text_area(
                "Enter text to translate",
                placeholder="Type here...",
                height=300,
                label_visibility="collapsed"
            )
        
        with c_out:
            st.subheader("Translation")
            output_placeholder = st.empty()
            # Default state
            output_placeholder.markdown('<div class="output-box">Result will appear here...</div>', unsafe_allow_html=True)

        st.form_submit_button("Translate Now", use_container_width=True, type="primary")

    if user_text:
        with st.spinner("Processing..."):
            try:
                # We use the standard translate_text call
                translated = translator.translate_text(
                    user_text,
                    target_language=target_lang,
                    source_language=source_lang
                )
                
                # Update output box with alignment
                alignment_class = "rtl" if target_lang == "ur" else "ltr"
                output_placeholder.markdown(f"""
                    <div class="output-box {alignment_class}">{translated}</div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Oops! Something went wrong: {e}")

    st.divider()
    st.info("💡 Tip: Use your multiple Gemini keys for uninterrupted service!")

if __name__ == "__main__":
    main()
