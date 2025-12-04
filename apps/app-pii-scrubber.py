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
MODEL_DIR = PROJECT_ROOT / "models" / "safety_classifier_onnx"

# Initialize Scrubber (Cached)
@st.cache_resource
def get_scrubber():
    return GermanPIIScrubber()

# Initialize Safety Classifier (Cached)
@st.cache_resource
def get_classifier():
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
    
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = ORTModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), file_name="model_quantized.onnx"
    )
    return tokenizer, model

scrubber = get_scrubber()

# Header
st.title("🛡️ Scrubb & Guard")
st.markdown("**PII Scrubbing** and **Safety Classification** for German text")

st.divider()

# Input Area
st.subheader("Input Text")
user_text = st.text_area(
    "Enter text to process:",
    height=200,
    placeholder="Ich heiße Thomas Müller, wohne in Berlin...",
    label_visibility="collapsed"
)

# Action Buttons
col1, col2 = st.columns(2)

with col1:
    scrubb_clicked = st.button("🧹 Scrubb", use_container_width=True, type="primary")

with col2:
    guard_clicked = st.button("🛡️ Guard", use_container_width=True, type="secondary")

st.divider()

# Results Area
if scrubb_clicked and user_text:
    st.subheader("Scrubbed Output")
    clean_text = scrubber.scrub(user_text)
    st.text_area(
        "Result:",
        value=clean_text,
        height=200,
        label_visibility="collapsed"
    )
    
    # Stats
    with st.expander("Scrubber Stats"):
        st.write(f"**Names in deny list:** {len(scrubber.deny_list_names):,}")
        st.write(f"**Cities in deny list:** {len(scrubber.deny_list_cities):,}")
        st.write(f"**Safe words:** {len(scrubber.safe_words):,}")

elif guard_clicked and user_text:
    st.subheader("Safety Classification")
    
    with st.spinner("Classifying..."):
        tokenizer, model = get_classifier()
        
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
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Safe", f"{safe_prob:.1%}")
        st.progress(float(safe_prob))
    with col2:
        st.metric("Unsafe", f"{unsafe_prob:.1%}")
        st.progress(float(unsafe_prob))
    
    # Model info
    with st.expander("Model Info"):
        st.write(f"**Model:** Quantized ONNX (TinyBERT)")
        st.write(f"**Path:** `{MODEL_DIR}`")
        st.write(f"**Max length:** 128 tokens")

elif (scrubb_clicked or guard_clicked) and not user_text:
    st.warning("Please enter some text first.")
