"""
Streamlit app for SetFit Safety Classification.

Interactive testing of fine-tuned SetFit models for
SAFE/UNSAFE binary classification of German text.
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path

# Add parent directory to path to import scrubb_guard package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page Config
st.set_page_config(
    page_title="SetFit Safety Classifier", 
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for distinctive styling (matching project aesthetic)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1628 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #e0e0ff !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #00ff88, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #8892b0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .safe-result {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%);
        border: 2px solid #00ff88;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    
    .unsafe-result {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 60, 60, 0.05) 100%);
        border: 2px solid #ff6b6b;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    
    .result-label {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .safe-label {
        color: #00ff88;
    }
    
    .unsafe-label {
        color: #ff6b6b;
    }
    
    .confidence-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        color: #8892b0;
    }
    
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
        color: #e0e0ff !important;
        font-size: 1rem !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00d4ff !important;
    }
    
    .example-chip {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px;
        font-size: 0.85rem;
        color: #e0e0ff;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-chip:hover {
        background: rgba(0, 212, 255, 0.2);
        border-color: #00d4ff;
    }
    
    .model-info {
        background: rgba(123, 44, 191, 0.1);
        border: 1px solid rgba(123, 44, 191, 0.3);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Model Loading ---
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

ID2LABEL = {0: "SAFE", 1: "UNSAFE"}


def get_available_models() -> list[str]:
    """Scan models directory for SetFit models."""
    models = []
    if MODELS_DIR.exists():
        for model_dir in MODELS_DIR.iterdir():
            setfit_path = model_dir / "setfit" / "best_model"
            if setfit_path.exists():
                models.append(model_dir.name)
    return sorted(models)


@st.cache_resource
def load_model(model_id: str):
    """Load SetFit model with caching."""
    from setfit import SetFitModel
    
    model_path = MODELS_DIR / model_id / "setfit" / "best_model"
    if not model_path.exists():
        return None
    
    return SetFitModel.from_pretrained(str(model_path))


def classify_text(model, text: str) -> tuple[int, list[float]]:
    """Classify text and return label + probabilities."""
    # Get prediction
    prediction = model.predict([text])[0]
    label = int(prediction)
    
    # Try to get probabilities if available
    try:
        probs = model.predict_proba([text])[0]
        probabilities = probs.tolist() if hasattr(probs, 'tolist') else list(probs)
    except Exception:
        # Fallback: use binary confidence
        probabilities = [0.0, 1.0] if label == 1 else [1.0, 0.0]
    
    return label, probabilities


# --- UI ---

# Header
st.markdown('<p class="main-title">🛡️ SetFit Safety Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fine-tuned German safety classification</p>', unsafe_allow_html=True)

# Sidebar: Model Selection
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    available_models = get_available_models()
    
    if not available_models:
        st.warning("⚠️ No trained models found!")
        st.markdown("""
        Train a model first:
        ```bash
        python scrubb_guard/train_setfit_model.py train \\
            -m setfit_v1 \\
            -d data/synthetic/final_setfit_train.csv
        ```
        """)
        selected_model = None
    else:
        selected_model = st.selectbox(
            "Select Model:",
            available_models,
            help="Choose a trained SetFit model"
        )
        
        if selected_model:
            st.markdown(f"""
            <div class="model-info">
                <strong>📁 Model Path:</strong><br>
                <code>models/{selected_model}/setfit/</code>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Info section
    st.markdown("""
    ### ℹ️ About SetFit
    
    SetFit uses **contrastive learning** to create powerful classifiers from minimal data.
    
    **Labels:**
    - 🟢 **SAFE** — Normal conversation
    - 🔴 **UNSAFE** — Concerning content
    
    **Best for:**
    - German mental health text
    - Crisis detection
    - Content moderation
    """)

# Main content
if selected_model:
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(selected_model)
    
    if model is None:
        st.error(f"Failed to load model: {selected_model}")
    else:
        st.success(f"✅ Model loaded: **{selected_model}**", icon="🤖")
        
        # Example sentences section
        st.divider()
        st.subheader("📝 Test Examples")
        st.caption("Click to load an example, or enter your own text below")
        
        # Example buttons organized by category
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Expected SAFE:**")
            safe_examples = [
                "Ich esse heute Pizza.",
                "Das Wetter ist heute schön.",
                "Ich sterbe vor Lachen! 😂",
                "Die Stimmen sind da, aber ich tue nichts.",
                "Früher hatte ich solche Gedanken, aber das ist vorbei.",
            ]
            for ex in safe_examples:
                if st.button(f"'{ex[:35]}..'" if len(ex) > 35 else f"'{ex}'", key=f"safe_{ex[:20]}", use_container_width=True):
                    st.session_state.test_input = ex
        
        with col2:
            st.markdown("**🔴 Expected UNSAFE:**")
            unsafe_examples = [
                "Ich will nicht mehr leben.",
                "Keiner würde mich vermissen.",
                "Ich nehme das Messer und mache Schluss.",
                "Heute nehme ich alle Tabletten.",
                "Ich weiß genau wie ich es machen werde.",
            ]
            for ex in unsafe_examples:
                if st.button(f"'{ex[:35]}..'" if len(ex) > 35 else f"'{ex}'", key=f"unsafe_{ex[:20]}", use_container_width=True):
                    st.session_state.test_input = ex
        
        st.divider()
        
        # Text input
        default_text = st.session_state.get("test_input", "")
        user_text = st.text_area(
            "Enter text to classify:",
            value=default_text,
            height=120,
            placeholder="Gib hier einen Text ein, um ihn zu klassifizieren...",
            key="classify_input"
        )
        
        # Classify on input
        if user_text.strip():
            st.divider()
            
            with st.spinner("Classifying..."):
                start_time = time.perf_counter()
                label, probabilities = classify_text(model, user_text)
                elapsed_time = time.perf_counter() - start_time
            
            # Result display
            label_str = ID2LABEL[label]
            confidence = probabilities[label] if len(probabilities) > label else 1.0
            
            if label == 0:  # SAFE
                st.markdown(f"""
                <div class="safe-result">
                    <div class="result-label safe-label">✅ SAFE</div>
                    <div class="confidence-text">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:  # UNSAFE
                st.markdown(f"""
                <div class="unsafe-result">
                    <div class="result-label unsafe-label">🚨 UNSAFE</div>
                    <div class="confidence-text">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Warning for unsafe content
                st.warning(
                    "⚠️ **Alert**: This content has been flagged as potentially concerning. "
                    "If this is real user input, consider appropriate escalation procedures."
                )
            
            # Detailed metrics
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", label_str)
            
            with col2:
                st.metric("Inference Time", f"{elapsed_time*1000:.0f} ms")
            
            with col3:
                st.metric("Model", selected_model)
            
            # Probability distribution
            st.subheader("📊 Score Distribution")
            
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                safe_prob = probabilities[0] if len(probabilities) > 0 else (1.0 - confidence if label == 1 else confidence)
                st.markdown("**SAFE Score:**")
                st.progress(safe_prob)
                st.caption(f"{safe_prob:.1%}")
            
            with prob_col2:
                unsafe_prob = probabilities[1] if len(probabilities) > 1 else (confidence if label == 1 else 1.0 - confidence)
                st.markdown("**UNSAFE Score:**")
                st.progress(unsafe_prob)
                st.caption(f"{unsafe_prob:.1%}")
            
            # Raw output
            with st.expander("📋 Raw Output"):
                st.json({
                    "input_text": user_text,
                    "prediction": label,
                    "label": label_str,
                    "probabilities": {
                        "SAFE": probabilities[0] if len(probabilities) > 0 else None,
                        "UNSAFE": probabilities[1] if len(probabilities) > 1 else None,
                    },
                    "confidence": confidence,
                    "inference_time_ms": elapsed_time * 1000,
                    "model_id": selected_model,
                })
        
        else:
            st.info("✍️ Enter text above or click an example to classify")

else:
    # No model available
    st.markdown("""
    <div class="result-card">
        <h3 style="color: #ffd93d; margin-top: 0;">⚠️ No Model Available</h3>
        <p style="color: #8892b0;">
            Train a SetFit model first to use this classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
# Train a new model:
python scrubb_guard/train_setfit_model.py train \\
    --model-id setfit_v1 \\
    --data data/synthetic/final_setfit_train.csv

# Or run full pipeline (train + test):
python scrubb_guard/train_setfit_model.py all \\
    --model-id setfit_v1 \\
    --data data/synthetic/final_setfit_train.csv
    """, language="bash")

# Footer
st.divider()
st.caption(
    "Base Model: `paraphrase-multilingual-mpnet-base-v2` · "
    "SetFit Contrastive Learning · Binary Safety Classification"
)

