"""
Streamlit app for testing the PII Anonymization Pipeline.
Uses Presidio + SpaCy for German PII detection and anonymization.
"""
import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path to import scrubb_guard package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrubb_guard.anonymization_pipeline import PiiPipeline

# Page Config
st.set_page_config(
    page_title="PII Anonymizer",
    page_icon="🔒",
    layout="centered"
)

# Custom CSS for a distinctive look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'Outfit', sans-serif;
        color: #888;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    
    .entity-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .entity-person { background: #7b2cbf33; border: 1px solid #7b2cbf; color: #c77dff; }
    .entity-location { background: #00d4ff33; border: 1px solid #00d4ff; color: #00d4ff; }
    .entity-org { background: #ff8c0033; border: 1px solid #ff8c00; color: #ffaa44; }
    .entity-plz { background: #ff006e33; border: 1px solid #ff006e; color: #ff006e; }
    .entity-tel { background: #ffc30033; border: 1px solid #ffc300; color: #ffc300; }
    .entity-email { background: #00ff8533; border: 1px solid #00ff85; color: #00ff85; }
    .entity-intern { background: #ff595933; border: 1px solid #ff5959; color: #ff5959; }
    .entity-misc { background: #9966ff33; border: 1px solid #9966ff; color: #bb99ff; }
    .entity-date { background: #66ccff33; border: 1px solid #66ccff; color: #99ddff; }
    .entity-default { background: #88888833; border: 1px solid #888888; color: #aaa; }
    
    .output-box {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1rem;
        line-height: 1.8;
    }
    
    div[data-testid="stTextArea"] textarea {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
    }
    
    .stButton > button {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        background: linear-gradient(90deg, #7b2cbf, #00d4ff);
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(123, 44, 191, 0.4);
    }
    
    .example-chip {
        display: inline-block;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 0.4rem 1rem;
        margin: 0.25rem;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .example-chip:hover {
        background: rgba(123, 44, 191, 0.2);
        border-color: #7b2cbf;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Pipeline (Cached - loads only once)
@st.cache_resource(show_spinner="🔄 Loading SpaCy model and initializing pipeline...")
def get_pipeline():
    """Load the PII pipeline once and cache it."""
    return PiiPipeline()


# Load the pipeline
pipeline = get_pipeline()


# Header
st.markdown('<h1 class="main-title">🔒 PII Anonymizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Presidio + SpaCy German NER • Real-time PII Detection & Masking</p>', unsafe_allow_html=True)

# Entity legend
st.markdown("##### Detected Entity Types")
entity_legend = """
<div style="margin-bottom: 1.5rem;">
    <span class="entity-tag entity-person">&lt;PERSON&gt;</span>
    <span class="entity-tag entity-location">&lt;LOCATION&gt;</span>
    <span class="entity-tag entity-org">&lt;ORG&gt;</span>
    <span class="entity-tag entity-plz">&lt;PLZ&gt;</span>
    <span class="entity-tag entity-tel">&lt;TEL&gt;</span>
    <span class="entity-tag entity-email">&lt;EMAIL&gt;</span>
    <span class="entity-tag entity-intern">&lt;INTERN&gt;</span>
</div>
"""
st.markdown(entity_legend, unsafe_allow_html=True)

st.divider()

# Example texts for quick testing
EXAMPLE_TEXTS = [
    "Ich heiße Peter Müller und wohne in 12345 Berlin.",
    "Mein Arzt ist Dr. Müller in der Klinik am See.",
    "Ruf mich an unter 0176-12345678 oder mail mir: peter@example.com",
    "Wir treffen uns in Essen zum Essen.",
    "Ich komme aus Oer-Erkenschwick.",
]

# Initialize session state for form input
if "form_input" not in st.session_state:
    st.session_state.form_input = ""


def set_example(example_text: str):
    """Callback to set example text in the form."""
    st.session_state.form_input = example_text


# Quick examples section
st.markdown("##### Quick Examples")
cols = st.columns(len(EXAMPLE_TEXTS))
for i, (col, example) in enumerate(zip(cols, EXAMPLE_TEXTS)):
    with col:
        st.button(
            f"📝 {i+1}",
            key=f"example_{i}",
            help=example,
            use_container_width=True,
            on_click=set_example,
            args=(example,)
        )

# Input Area - using a form to capture Ctrl+Enter
with st.form(key="anonymize_form"):
    user_text = st.text_area(
        "Enter German text to anonymize (Ctrl+Enter to submit):",
        height=120,
        placeholder="Geben Sie deutschen Text ein, der personenbezogene Daten enthält...",
        key="form_input",  # Direct binding to session state
    )
    
    # Center the submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        process_btn = st.form_submit_button("🚀 Anonymize", type="primary", use_container_width=True)

# Process when form submitted (button click or Ctrl+Enter)
# Use user_text (return value) which has current value, not session_state which updates after
if process_btn and user_text:
    st.divider()
    
    # Run the pipeline
    with st.spinner("Analyzing..."):
        result = pipeline.process(user_text)
    
    # Compact stats line
    st.markdown(
        f'<p style="color: #888; font-size: 0.9rem; margin-bottom: 1rem;">'
        f'<strong>{result["original_length"]}</strong> characters · '
        f'<strong style="color: #00d4ff;">{result["items_changed"]}</strong> PIIs found</p>',
        unsafe_allow_html=True
    )
    
    # Side by side comparison
    col_orig, col_anon = st.columns(2)
    
    with col_orig:
        st.markdown("**Original Text**")
        st.text_area(
            "Original:",
            value=user_text,
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )
    
    with col_anon:
        st.markdown("**Anonymized Text**")
        # Highlight the anonymized tokens with colors
        anonymized = result['anonymized_text']
        
        # Apply color coding to output
        highlighted = anonymized
        replacements = [
            ("<PERSON>", '<span class="entity-tag entity-person">&lt;PERSON&gt;</span>'),
            ("<LOCATION>", '<span class="entity-tag entity-location">&lt;LOCATION&gt;</span>'),
            ("<ORG>", '<span class="entity-tag entity-org">&lt;ORG&gt;</span>'),
            ("<PLZ>", '<span class="entity-tag entity-plz">&lt;PLZ&gt;</span>'),
            ("<TEL>", '<span class="entity-tag entity-tel">&lt;TEL&gt;</span>'),
            ("<EMAIL>", '<span class="entity-tag entity-email">&lt;EMAIL&gt;</span>'),
            ("<INTERN>", '<span class="entity-tag entity-intern">&lt;INTERN&gt;</span>'),
            ("<MISC>", '<span class="entity-tag entity-misc">&lt;MISC&gt;</span>'),
            ("<DATE>", '<span class="entity-tag entity-date">&lt;DATE&gt;</span>'),
            ("<NRP>", '<span class="entity-tag entity-default">&lt;NRP&gt;</span>'),
            ("<PII>", '<span class="entity-tag entity-default">&lt;PII&gt;</span>'),
        ]
        
        for old, new in replacements:
            highlighted = highlighted.replace(old, new)
        
        st.markdown(f'<div class="output-box">{highlighted}</div>', unsafe_allow_html=True)
    
    # Copy button for anonymized text
    st.markdown("<br>", unsafe_allow_html=True)
    st.code(result['anonymized_text'], language=None)
    
    # Show detected entities details
    if result.get('entities'):
        with st.expander("🔍 Detected Entities (Debug Info)"):
            for ent in result['entities']:
                entity_class = ent['entity_type'].lower().replace('_', '-')
                st.markdown(
                    f"**{ent['text']}** → `{ent['entity_type']}` (score: {ent['score']:.2f})",
                    unsafe_allow_html=True
                )

elif process_btn and not user_text:
    st.warning("Please enter some text to anonymize.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    Powered by <strong>Presidio</strong> + <strong>SpaCy de_core_news_lg</strong><br>
    🔒 All processing happens locally • No data leaves your machine
</div>
""", unsafe_allow_html=True)

