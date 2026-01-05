"""
Simplified Zero-Shot Suicide Intent Classifier.

Single-hypothesis classification with clean, minimal UI.
"""

import streamlit as st
import sys
import os
import time
import random
import pandas as pd

# Add parent directory to path to import scrubb_guard package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrubb_guard.zero_shot_classifier import ZeroShotClassifier

# Page Config
st.set_page_config(
    page_title="Zero-shot NLI-based Classifier",
    page_icon="🔬",
    layout="centered"
)

# Custom CSS - dark theme with cyan accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap');
    
    .stApp {
        background: #0a0a0f;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #f0f0f5 !important;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00e5cc;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #6b7280;
        text-align: center;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .score-display {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 4rem;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }
    
    .score-low { color: #22c55e; }
    .score-medium { color: #eab308; }
    .score-high { color: #ef4444; }
    
    .hypothesis-box {
        background: rgba(0, 229, 204, 0.08);
        border: 1px solid rgba(0, 229, 204, 0.25);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 1rem 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.9rem;
        color: #a0a0b0;
    }
    
    .stTextArea textarea {
        font-family: 'IBM Plex Mono', monospace !important;
        background: #12121a !important;
        border: 1px solid #2a2a3a !important;
        color: #e0e0e8 !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input {
        font-family: 'IBM Plex Mono', monospace !important;
        background: #12121a !important;
        border: 1px solid #2a2a3a !important;
        color: #e0e0e8 !important;
        border-radius: 8px !important;
    }
    
    .inference-time {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #4a4a5a;
        text-align: center;
        margin-top: 2rem;
    }
    
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #00e5cc 0%, #00b8a9 100%);
        color: #0a0a0f;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 6px;
    }
    
    div[data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #00f5dc 0%, #00c8b9 100%);
    }
    
    /* Sidebar example buttons */
    .example-safe {
        background: rgba(34, 197, 94, 0.15) !important;
        border: 1px solid rgba(34, 197, 94, 0.4) !important;
        color: #22c55e !important;
    }
    
    .example-unsafe {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
        color: #ef4444 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0d0d14;
    }
    
    section[data-testid="stSidebar"] .stButton button {
        background: rgba(255, 255, 255, 0.05);
        color: #c0c0d0 !important;
        text-align: left !important;
        font-size: 0.8rem;
        padding: 0.4rem 0.6rem;
        margin: 2px 0;
        white-space: normal;
        height: auto;
        line-height: 1.3;
        border: 1px solid rgba(255, 255, 255, 0.1);
        justify-content: flex-start !important;
    }
    
    section[data-testid="stSidebar"] .stButton button p {
        text-align: left !important;
        width: 100%;
    }
    
    section[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255, 255, 255, 0.1);
        color: #ffffff !important;
    }
    
    .example-container {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    """Load classifier once and cache."""
    return ZeroShotClassifier()


@st.cache_data
def load_sample_data():
    """Load sample texts and labels from CSV."""
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/synthetic/final_setfit_train.csv"
    )
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Strip quotation marks from text entries
        texts = [text.strip('"').strip("'") for text in df["text"].tolist()]
        labels = df["label"].tolist()
        return list(zip(texts, labels))
    return []


def get_score_class(score: float) -> str:
    """Return CSS class based on score."""
    if score < 0.4:
        return "score-low"
    elif score < 0.7:
        return "score-medium"
    else:
        return "score-high"


# Load sample data (text, label pairs)
sample_data = load_sample_data()

def load_example(text: str):
    """Callback to load an example into text input."""
    st.session_state.text_input = text

def load_random_sample():
    """Callback to load a random sample into text input."""
    if sample_data:
        text, _ = random.choice(sample_data)
        st.session_state.text_input = text

# Sidebar with examples
with st.sidebar:
    st.markdown("### 📚 Examples")
    
    # Random button at top
    if sample_data:
        st.button("🎲 Random Sample", use_container_width=True, 
                  on_click=load_random_sample, key="random_btn")
    
    st.markdown("---")
    
    # Count examples by label
    safe_count = sum(1 for _, label in sample_data if label == 0)
    unsafe_count = sum(1 for _, label in sample_data if label == 1)
    st.caption(f"🟢 Safe: {safe_count} · 🔴 Unsafe: {unsafe_count}")
    
    # Scrollable example list
    st.markdown('<div class="example-container">', unsafe_allow_html=True)
    
    for i, (text, label) in enumerate(sample_data):
        # Truncate long text for button display
        display_text = text[:60] + "..." if len(text) > 60 else text
        
        # Color indicator
        color_indicator = "🟢" if label == 0 else "🔴"
        
        st.button(
            f"{color_indicator} {display_text}",
            key=f"example_{i}",
            use_container_width=True,
            on_click=load_example,
            args=(text,)
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<p class="main-title">Zero-shot NLI-based Classifier</p>', unsafe_allow_html=True)

# Hypothesis templates for each language
HYPOTHESES = {
    "English": {
        "primary": "This text is about suicidal intent",
        "alt": "This text is about something else."
    },
    "Deutsch": {
        "primary": "Dieser Text drückt Suizidgedanken aus.",
        "alt": "Dieser Text handelt von etwas anderem."
    }
}

# Language toggle and contrasting mode in a row
col_lang, col_contrast = st.columns(2)

with col_lang:
    use_german = st.toggle(
        "🇩🇪 German hypotheses",
        value=False,
        help="Switch between English and German hypothesis templates."
    )

with col_contrast:
    contrasting_mode = st.toggle(
        "🔀 Contrasting mode",
        value=True,
        help="When ON, compares both hypotheses (more accurate). When OFF, scores primary hypothesis independently."
    )

# Get default values based on language
lang = "Deutsch" if use_german else "English"
default_primary = HYPOTHESES[lang]["primary"]
default_alt = HYPOTHESES[lang]["alt"]

# Hypothesis inputs
col_hyp1, col_hyp2 = st.columns(2)

with col_hyp1:
    hypothesis = st.text_input(
        "Primary hypothesis:",
        value=default_primary,
        key=f"primary_hyp_{lang}",
        help="The hypothesis to test for."
    )

with col_hyp2:
    alt_hypothesis = st.text_input(
        "Alternative hypothesis:",
        value=default_alt,
        key=f"alt_hyp_{lang}",
        help="Contrasting hypothesis (used when 'Contrasting mode' is enabled)."
    )

# Text input
user_text = st.text_area(
    "Text to classify:",
    height=150,
    placeholder="Select an example from the sidebar or enter text here...",
    key="text_input"
)

# Classification
if user_text.strip():
    classifier = get_classifier()
    
    start_time = time.perf_counter()
    
    if contrasting_mode:
        # Compare both hypotheses - scores sum to 100%
        result = classifier.classify(
            user_text,
            labels=[hypothesis, alt_hypothesis],
            multi_label=False,  # Contrasting: scores sum to 1
            hypothesis_template="{}"
        )
        # Get score for primary hypothesis
        score = result.scores[0] if result.scores else 0.0
    else:
        # Independent scoring of primary hypothesis only
        result = classifier.classify(
            user_text,
            labels=[hypothesis],
            multi_label=True,  # Independent scoring
            hypothesis_template="{}"
        )
        score = result.scores[0] if result.scores else 0.0
    
    elapsed_time = time.perf_counter() - start_time
    
    score_class = get_score_class(score)
    
    # Display score only
    st.markdown("---")
    st.markdown(
        f'<p class="score-display {score_class}">{score:.1%}</p>',
        unsafe_allow_html=True
    )
    
    # Inference time at bottom
    st.markdown(
        f'<p class="inference-time">Inference: {elapsed_time*1000:.0f}ms</p>',
        unsafe_allow_html=True
    )

else:
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #4a4a5a; font-family: IBM Plex Mono, monospace;">'
        '↑ Enter text above to classify</p>',
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.caption(
    "Model: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`"
)

