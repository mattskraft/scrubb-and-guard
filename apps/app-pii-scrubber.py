import streamlit as st
import sys
import os
# Add parent directory to path to import scrubb_guard package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrubb_guard import GermanPIIScrubber, REGEX_PATTERNS

# Page Config
st.set_page_config(page_title="PII Scrubber Prototype", layout="wide")

# Initialize Scrubber (Cached so it doesn't reload data on every click)
@st.cache_resource
def get_scrubber():
    return GermanPIIScrubber()

scrubber = get_scrubber()

st.title("🇩🇪 German PII Scrubber (Mobile Prototype)")
st.markdown("""
This tool simulates the **On-Device PII Removal** pipeline. 
It uses **Regex + Deny Lists** (no heavy AI) to prove mobile feasibility.
""")

# Input Area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    user_text = st.text_area("Enter patient text here:", height=300, 
                             value="Ich heiße Thomas Müller, komme aus 10115 Berlin und nehme 2 Tabletten. Meine KV-Nr ist A123456789.")

with col2:
    st.subheader("Scrubbed Output")
    if user_text:
        # Run the scrub
        clean_text = scrubber.scrub(user_text)
        
        # Highlight changes (Simple diff visualization)
        st.text_area("Result for LLM:", value=clean_text, height=300)

# Debug / Info Section
with st.expander("Show Configuration & Stats"):
    st.write(f"**Loaded Names:** {len(scrubber.deny_list_names)}")
    st.write(f"**Loaded Cities:** {len(scrubber.deny_list_cities)}")
    st.write("**Active Regex Patterns:**")
    st.json(REGEX_PATTERNS)
