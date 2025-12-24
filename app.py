import streamlit as st
from config import get_settings
from services.translator import TranslatorService

def main() -> None:
    st.set_page_config(page_title="TranslatorGO | English ↔ Urdu", layout="wide")
    
    # Premium Styling
    st.markdown("""
        <style>
        .main {
            background-color: #0e1117;
        }
        .stTextArea textarea {
            font-size: 18px !important;
        }
        .rtl {
            direction: rtl;
            text-align: right;
            font-family: 'Urdu Typesetting', 'Jameel Noori Nastaleeq', sans-serif;
        }
        .ltr {
            direction: ltr;
            text-align: left;
        }
        .output-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #1a1c24;
            border: 1px solid #30363d;
            min-height: 200px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚀 TranslatorGO")
    st.caption("AI-Powered Semantic English ↔ Urdu Translator")

    # Initialize service
    try:
        settings = get_settings()
        translator = TranslatorService()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return

    # Sidebar / Top Controls
    with st.expander("⚙️ Settings & Tone", expanded=True):
        col_rot, col_tone = st.columns(2)
        with col_rot:
            direction = st.radio(
                "Translation Direction",
                options=["English → Urdu", "Urdu → English"],
                horizontal=True
            )
        with col_tone:
            tone = st.select_slider(
                "Select Voice Tone",
                options=["Casual", "Formal", "Literary"],
                value="Formal"
            )

    is_ur_input = "Urdu → English" in direction
    source_lang = "ur" if is_ur_input else "en"
    target_lang = "en" if is_ur_input else "ur"

    # Main Translation Form (Parallel Boxes)
    with st.form("translation_form", clear_on_submit=False):
        c_in, c_out = st.columns(2)
        
        with c_in:
            st.subheader("📝 Input")
            user_text = st.text_area(
                "Enter text to translate",
                placeholder="Type something here...",
                height=250,
                label_visibility="collapsed"
            )
        
        with c_out:
            st.subheader("✨ Result")
            output_placeholder = st.empty()
            # Initial placeholder content
            output_placeholder.markdown('<div class="output-box">Translation will appear here...</div>', unsafe_allow_html=True)

        submit = st.form_submit_button("Translate Now", use_container_width=True, type="primary")

    if submit:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Translating..."):
                try:
                    result = translator.translate_text(
                        user_text,
                        target_language=target_lang,
                        source_language=source_lang,
                        tone=tone
                    )
                    
                    if isinstance(result, dict):
                        translated = result["translation"]
                        insight = result["insight"]
                    else:
                        translated = result
                        insight = None

                    # Update output box
                    alignment_class = "ltr" if target_lang == "en" else "rtl"
                    output_placeholder.markdown(f"""
                        <div class="output-box {alignment_class}">
                            {translated}
                        </div>
                    """, unsafe_allow_html=True)

                    # Show Insight if available
                    if insight:
                        st.info(f"💡 **Meaning Insight:** {insight}")

                except Exception as e:
                    st.error(f"Translation failed: {e}")

    st.divider()
    st.markdown("Developed for Hackathon | Powered by Google Gemini & Key Rotation")

if __name__ == "__main__":
    main()
