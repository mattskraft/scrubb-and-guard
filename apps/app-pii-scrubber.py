import streamlit as st
import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path to import scrubb_guard package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrubb_guard import GermanPIIScrubber, REGEX_PATTERNS

# Page Config
st.set_page_config(page_title="Scrubb & Guard", layout="centered")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def get_available_models() -> list[str]:
    """Scan models directory for available trained models."""
    models = []
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                onnx_path = model_dir / "onnx" / "model_quantized.onnx"
                if onnx_path.exists():
                    models.append(model_dir.name)
    return sorted(models)


# Initialize Scrubber (Cached)
@st.cache_resource
def get_scrubber():
    return GermanPIIScrubber()


# Initialize Safety Classifier (Cached per model)
@st.cache_resource
def get_classifier(model_id: str):
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
    
    model_dir = MODELS_DIR / model_id / "onnx"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir), file_name="model_quantized.onnx"
    )
    return tokenizer, model


scrubber = get_scrubber()

# Header
st.title("🛡️ Scrubb & Guard")
st.markdown("**PII Scrubbing** and **Safety Classification** for German text")

# Model Selection
available_models = get_available_models()

if not available_models:
    st.warning("No trained models found in `models/`. Train a model first with `train_safety_model.py`.")
    selected_model = None
else:
    selected_model = st.selectbox(
        "Safety Model",
        options=available_models,
        help="Choose which trained model to use for safety classification"
    )

st.divider()

# Input Area
user_text = st.text_area(
    "Enter text and press Ctrl+Enter:",
    height=150,
    placeholder="Ich heiße Thomas Müller, wohne in Berlin...",
)

# Process when text is entered
if user_text:
    st.divider()
    
    # Two columns for results
    col1, col2 = st.columns(2)
    
    # Left column: Scrubbed output
    with col1:
        st.subheader("🧹 Scrubbed")
        clean_text = scrubber.scrub(user_text)
        st.text_area(
            "Scrubbed text:",
            value=clean_text,
            height=150,
            label_visibility="collapsed",
            disabled=True
        )
    
    # Right column: Safety classification
    with col2:
        st.subheader("🛡️ Safety")
        
        if selected_model:
            tokenizer, model = get_classifier(selected_model)
            
            # Run inference
            inputs = tokenizer(user_text, return_tensors="np", truncation=True, max_length=128)
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            pred_class = np.argmax(logits, axis=1)[0]
            
            safe_prob = probs[0][0]
            unsafe_prob = probs[0][1]
            
            label = "SAFE" if pred_class == 0 else "UNSAFE"
            color = "green" if pred_class == 0 else "red"
            
            # Display result
            st.markdown(f"### :{color}[{label}]")
            
            # Confidence bars
            st.caption("Safe")
            st.progress(float(safe_prob), text=f"{safe_prob:.1%}")
            st.caption("Unsafe")
            st.progress(float(unsafe_prob), text=f"{unsafe_prob:.1%}")
        else:
            st.info("No model selected")
    
    # Expandable details
    with st.expander("Details"):
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.write("**Scrubber Stats**")
            st.write(f"- Names: {len(scrubber.deny_list_names):,}")
            st.write(f"- Cities: {len(scrubber.deny_list_cities):,}")
            st.write(f"- Safe words: {len(scrubber.safe_words):,}")
        
        with detail_col2:
            if selected_model:
                st.write("**Model Info**")
                st.write(f"- Model ID: `{selected_model}`")
                st.write(f"- Type: Quantized ONNX")
                st.write(f"- Max tokens: 128")
