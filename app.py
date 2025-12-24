import streamlit as st
from config import get_settings
from services.translator import TranslatorService

def main() -> None:
    st.set_page_config(page_title="TranslatorGO", layout="wide")
    
    # Precise CSS to match the reference image theme
    st.markdown("""
        <style>
        /* Hide deploy button and other elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Set background and text colors */
        .main {
            background-color: #000000;
        }
        
        /* Font styles for Input/Output labels */
        .label-text {
            color: #FFFFFF;
            font-weight: bold;
            font-size: 24px;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        
        /* Styling for the output container to look like the image */
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
        
        .rtl {
            direction: rtl;
            text-align: right;
            font-family: inherit;
        }

        /* Adjust radio button spacing */
        div[role="radiogroup"] > label {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize service
    try:
        settings = get_settings()
        translator = TranslatorService()
    except Exception as e:
        st.error(f"Error: {e}")
        return

    # Language Selection (Top Left)
    direction = st.radio(
        "",
        options=["English → Urdu", "Urdu → English"],
        label_visibility="collapsed",
        index=0
    )

    is_ur_input = "Urdu → English" in direction
    source_lang = "ur" if is_ur_input else "en"
    target_lang = "en" if is_ur_input else "ur"

    st.write("") # Spacing

    # Parallel UI structure
    col_in, col_out = st.columns(2)
    
    with col_in:
        st.markdown('<p class="label-text">INPUT</p>', unsafe_allow_html=True)
        user_text = st.text_area(
            "input",
            placeholder="Enter text...",
            height=250,
            label_visibility="collapsed",
            key="user_input_area"
        )
    
    with col_out:
        st.markdown('<p class="label-text">OUTPUT</p>', unsafe_allow_html=True)
        
        # Translation Logic
        translated_text = ""
        if user_text.strip():
            try:
                # We use the key rotation and fallback logic from the backend
                translated_text = translator.translate_text(
                    user_text,
                    target_language=target_lang,
                    source_language=source_lang
                )
            except Exception as e:
                translated_text = f"Error: {str(e)}"

        # Output Box
        alignment_class = "rtl" if target_lang == "ur" else "ltr"
        st.markdown(f"""
            <div class="output-box {alignment_class}">
                {translated_text if translated_text else ""}
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
