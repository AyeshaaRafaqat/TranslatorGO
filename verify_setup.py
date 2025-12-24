from transformers import MarianMTModel, MarianTokenizer
import sys

def verify_models():
    print("Verifying environment and downloading models if needed...")
    try:
        import torch
        print(f"Torch version: {torch.__version__}")
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"Import Error: {e}")
        return

    models = ["Helsinki-NLP/opus-mt-en-ur", "Helsinki-NLP/opus-mt-ur-en"]
    
    for model_name in models:
        print(f"\nChecking {model_name}...")
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            print("  -> Loaded successfully")
            
            # Test translation
            text = "Hello" if "en-ur" in model_name else "شکریہ"
            inputs = tokenizer(text, return_tensors="pt")
            translated = model.generate(**inputs)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            print(f"  -> Test translation ('{text}'): {result}")
            
        except Exception as e:
            print(f"  -> Failed to load/translate: {e}")

if __name__ == "__main__":
    verify_models()
