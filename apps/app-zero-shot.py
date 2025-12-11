"""
Streamlit app for Zero-Shot Text Classification.

Allows dynamic label definition and interactive classification
using multilingual DeBERTa model.
"""

import streamlit as st
import sys
import os
import time

# Add parent directory to path to import scrubb_guard package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrubb_guard.zero_shot_classifier import (
    ZeroShotClassifier, DEFAULT_LABELS, DEFAULT_LABELS_EN,
    DEFAULT_LABEL_MAP, DEFAULT_TEMPLATE_EN, DEFAULT_TEMPLATE_DE
)

# Page Config
st.set_page_config(
    page_title="Zero-Shot Classifier", 
    page_icon="🎯",
    layout="centered"
)

# Custom CSS for distinctive styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #e0e0ff !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff6b6b);
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
    
    .score-bar {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        border-radius: 4px;
        height: 24px;
        transition: width 0.5s ease;
    }
    
    .label-chip {
        display: inline-block;
        background: rgba(123, 44, 191, 0.2);
        border: 1px solid #7b2cbf;
        border-radius: 16px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.9rem;
        color: #e0e0ff;
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(123, 44, 191, 0.4) !important;
        color: #e0e0ff !important;
    }
    
    .stTextInput input {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(123, 44, 191, 0.4) !important;
        color: #e0e0ff !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    """Load classifier once and cache."""
    return ZeroShotClassifier()


def render_score_bar(label: str, score: float, rank: int):
    """Render a styled score bar for a label."""
    colors = ["#00d4ff", "#7b2cbf", "#ff6b6b", "#ffd93d", "#6bcb77"]
    color = colors[rank % len(colors)]
    
    col1, col2, col3 = st.columns([3, 5, 2])
    
    with col1:
        st.markdown(f"**{label}**")
    
    with col2:
        st.progress(score)
    
    with col3:
        st.markdown(f"**{score:.1%}**")


# Header
st.markdown('<p class="main-title">🎯 Zero-Shot Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Multilingual text classification without training</p>', unsafe_allow_html=True)

# Initialize session state for labels and settings
if "labels" not in st.session_state:
    st.session_state.labels = list(DEFAULT_LABELS_EN)  # Start with English for best results
if "use_english" not in st.session_state:
    st.session_state.use_english = True
if "label_map" not in st.session_state:
    st.session_state.label_map = dict(DEFAULT_LABEL_MAP)

# Sidebar for label management
with st.sidebar:
    st.header("📝 Classification Labels")
    st.caption("Define the categories for classification")
    
    # Show current labels with display names
    st.markdown("**Active Labels:**")
    
    labels_to_remove = []
    for i, label in enumerate(st.session_state.labels):
        col1, col2 = st.columns([4, 1])
        display_name = st.session_state.label_map.get(label, label)
        with col1:
            if display_name != label:
                st.markdown(f"`{label}` → {display_name}")
            else:
                st.markdown(f"`{label}`")
        with col2:
            if st.button("×", key=f"remove_{i}", help=f"Remove {label}"):
                labels_to_remove.append(label)
    
    # Remove labels after iteration
    for label in labels_to_remove:
        st.session_state.labels.remove(label)
        st.rerun()
    
    st.divider()
    
    # Add new label
    new_label = st.text_input(
        "Add new label:",
        placeholder="e.g., suicide, anxiety, safe...",
        key="new_label_input"
    )
    
    if st.button("➕ Add Label", use_container_width=True):
        if new_label and new_label.strip():
            clean_label = new_label.strip()
            if clean_label not in st.session_state.labels:
                st.session_state.labels.append(clean_label)
                st.rerun()
            else:
                st.warning("Label already exists!")
        else:
            st.warning("Please enter a label name")
    
    st.divider()
    
    # Reset to defaults
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        st.session_state.labels = list(DEFAULT_LABELS_EN)
        st.session_state.use_english = True
        st.rerun()
    
    # Quick presets (English labels for better accuracy)
    st.divider()
    st.markdown("**Quick Presets** *(English labels)*:")
    
    if st.button("🏥 Mental Health", use_container_width=True):
        st.session_state.labels = ["suicide", "self-harm", "depression", "anxiety", "neutral"]
        st.session_state.use_english = True
        st.rerun()
    
    if st.button("💬 Sentiment", use_container_width=True):
        st.session_state.labels = ["positive", "negative", "neutral"]
        st.session_state.use_english = True
        st.rerun()
    
    if st.button("⚠️ Content Safety", use_container_width=True):
        st.session_state.labels = ["safe", "hate speech", "insult", "violence", "sexual content"]
        st.session_state.use_english = True
        st.rerun()

# Main content area
st.divider()

# Settings row
settings_col1, settings_col2 = st.columns(2)

with settings_col1:
    multi_label = st.toggle(
        "Multi-Label Mode",
        value=False,  # Single-label often works better
        help="When enabled, each label is scored independently. When disabled, scores sum to 100%."
    )

with settings_col2:
    use_english_template = st.toggle(
        "🚀 English Template",
        value=True,
        help="Use English hypothesis template for better accuracy (recommended even for German text)"
    )

# Show template being used
hypothesis_template = DEFAULT_TEMPLATE_EN if use_english_template else DEFAULT_TEMPLATE_DE
st.caption(f"Template: *\"{hypothesis_template}\"*")

# Text input
user_text = st.text_area(
    "Enter text to classify:",
    height=150,
    placeholder="Es hat alles keinen Sinn mehr, ich will einfach nur noch schlafen...",
    key="input_text"
)

# Show current labels as chips (with display names)
if st.session_state.labels:
    st.markdown("**Classifying against:**")
    display_labels = [st.session_state.label_map.get(l, l) for l in st.session_state.labels]
    labels_html = " ".join([f'<span class="label-chip">{label}</span>' for label in display_labels])
    st.markdown(f'<div>{labels_html}</div>', unsafe_allow_html=True)
else:
    st.warning("Please add at least one label in the sidebar")

# Classification
if user_text and st.session_state.labels:
    st.divider()
    
    with st.spinner("Classifying..."):
        classifier = get_classifier()
        start_time = time.perf_counter()
        result = classifier.classify(
            user_text, 
            labels=st.session_state.labels,
            multi_label=multi_label,
            hypothesis_template=hypothesis_template,
            label_mapping=st.session_state.label_map
        )
        elapsed_time = time.perf_counter() - start_time
    
    # Results header
    st.subheader("📊 Classification Results")
    
    # Top prediction highlight
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Top Prediction",
            value=result.top_label,
            delta=f"{result.top_score:.1%} confidence"
        )
    with col2:
        mode = "Multi-Label" if multi_label else "Single-Label"
        st.metric(
            label="Mode",
            value=mode,
            delta=f"{len(st.session_state.labels)} labels"
        )
    with col3:
        st.metric(
            label="Inference Time",
            value=f"{elapsed_time:.2f}s",
            delta=f"{elapsed_time*1000:.0f} ms"
        )
    
    st.divider()
    
    # All scores (using display labels)
    st.markdown("**All Scores (ranked):**")
    
    display_labels = result.get_labels_for_display()
    for rank, (label, score) in enumerate(zip(display_labels, result.scores)):
        render_score_bar(label, score, rank)
    
    # Details expander
    with st.expander("📋 Raw Output"):
        st.json(result.to_dict())
        
    # Warning for concerning content (check both English and German labels)
    concerning_labels = {
        "Suizidalität", "Selbstverletzung", "Gewaltandrohung",
        "suicide", "self-harm", "violence"
    }
    detected_concerning = [
        (label, score) 
        for label, score in zip(result.labels, result.scores) 
        if label in concerning_labels and score > 0.5
    ]
    
    if detected_concerning:
        st.divider()
        st.warning(
            "⚠️ **Content Alert**: High scores detected for concerning categories. "
            "If this is real user content, consider appropriate escalation."
        )

elif not st.session_state.labels:
    st.info("👈 Add labels in the sidebar to get started")
else:
    st.info("✍️ Enter text above and press Ctrl+Enter to classify")

# Footer
st.divider()
st.caption(
    "Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` · "
    "Multilingual NLI · Zero-Shot Classification"
)

