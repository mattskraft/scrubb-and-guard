import re
import json
import os
from typing import List, Set, Dict

# --- STAGE 1: REGEX PATTERNS (The "Sledgehammer") ---
# These run first to catch structured identifiers
REGEX_PATTERNS = {
    "MEDICAL_ID": r"\b[A-Z]\d{9}\b",  # German KV-Nummer (e.g., A123456789)
    "IBAN": r"\bDE\d{2}\s?(?:\d{4}\s?){4}\d{2}\b",  # German IBAN with optional spaces
    "PHONE": r"(?:\+49|0)(?:\s*\d+){1,4}[-/\s]*\d{3,}", # Loose German Phone (DIN 5008ish)
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "DATE_YEAR": r"\b(19|20)\d{2}\b", # Catches years 1900-2099 (Optional: decide if you want to scrub these)
    "STREET": r"\b[A-ZÄÖÜ][a-zäöüß]*(?:straße|strasse|str\.|weg|platz|allee|gasse|ring|damm|ufer)\s*\d+[a-zA-Z]?\b",  # German street addresses
}

class GermanPIIScrubber:
    def __init__(self, data_dir=None):
        # If data_dir is not provided, resolve it relative to the package location
        if data_dir is None:
            # Get the directory where this package is located
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(package_dir, "data")
        
        self.data_dir = data_dir
        self.deny_list_names: Set[str] = set()
        self.deny_list_cities: Set[str] = set()
        self.deny_list_common: Set[str] = set() # Words that are names but also common verbs/nouns (e.g., "Essen")
        
        # Safe words to exclude from scrubbing even if they are in lists
        # Loaded from safe_words.txt - words that are place/person names but also common German words
        self.safe_words: Set[str] = set()

        # Load data from local files
        self._load_data()

    def _load_data(self):
        """Loads data from local files in data directory."""
        print(f"Loading PII data from {self.data_dir}...")
        
        # 0. Load safe words first (must be loaded before names/cities)
        safe_words_file = os.path.join(self.data_dir, "safe_words.txt")
        if os.path.exists(safe_words_file):
            with open(safe_words_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        self.safe_words.add(line)
            print(f"Loaded {len(self.safe_words)} safe words.")
        else:
            print(f"Warning: {safe_words_file} not found, using empty safe words list")
        
        # 1. Load names from namen.txt
        names_file = os.path.join(self.data_dir, "namen.txt")
        if os.path.exists(names_file):
            with open(names_file, 'r', encoding='utf-8') as f:
                for line in f:
                    clean_word = line.strip()
                    # Remove quotes if present
                    clean_word = clean_word.strip('"')
                    if len(clean_word) > 2 and clean_word not in self.safe_words:
                        self.deny_list_names.add(clean_word)
        else:
            print(f"Warning: {names_file} not found")
        
        # 2. Load cities from orte.txt
        cities_file = os.path.join(self.data_dir, "orte.txt")
        if os.path.exists(cities_file):
            with open(cities_file, 'r', encoding='utf-8') as f:
                for line in f:
                    clean_word = line.strip()
                    # Remove quotes if present
                    clean_word = clean_word.strip('"')
                    if len(clean_word) > 3 and clean_word not in self.safe_words:
                        self.deny_list_cities.add(clean_word)
        else:
            print(f"Warning: {cities_file} not found")
        
        print(f"Loaded {len(self.deny_list_names)} names and {len(self.deny_list_cities)} locations.")

    def scrub(self, text: str) -> str:
        """The Main Pipeline: Input Text -> Anonymized Text"""
        
        # Stage 1: Regex (Structured Data)
        for label, pattern in REGEX_PATTERNS.items():
            text = re.sub(pattern, f"[{label}]", text)

        # Stage 2: Deny Lists (Exact Match / "Dictionary" Attack)
        # We split by whitespace to keep it fast (Basic Tokenization)
        words = text.split()
        scrubbed_words = []
        
        for word in words:
            # Strip punctuation for checking (keep original for replacement)
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check Names (Case Sensitive usually better for German Nouns)
            if clean_word in self.deny_list_names:
                scrubbed_words.append("[PERSON]")
            
            # Check Cities
            elif clean_word in self.deny_list_cities:
                scrubbed_words.append("[LOCATION]")
            
            else:
                scrubbed_words.append(word)
        
        text = " ".join(scrubbed_words)

        # Stage 3: Brute Force Numbers (The Safety Net)
        # Remove any remaining sequence of 3 or more digits
        text = re.sub(r'\d{3,}', '[NUM]', text)

        return text

    def export_mobile_config(self, output_path="mobile_assets"):
        """Exports the logic to JSON + Raw Text for the Mobile App"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # 1. Export Rules JSON
        config = {
            "regex_rules": [{"name": k, "pattern": v} for k, v in REGEX_PATTERNS.items()],
            "deny_list_files": {
                "PERSON": "deny_names.txt",
                "LOCATION": "deny_cities.txt"
            }
        }
        with open(f"{output_path}/pii_config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        # 2. Export Raw Lists (Cleaned)
        with open(f"{output_path}/deny_names.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(sorted(self.deny_list_names)))
            
        with open(f"{output_path}/deny_cities.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(sorted(self.deny_list_cities)))
            
        print(f"✅ Mobile assets exported to /{output_path}")

# CLI Usage
if __name__ == "__main__":
    scrubber = GermanPIIScrubber()
    sample = "Ich heiße Thomas, wohne in Berlin und meine KV-Nummer ist A123456789. Ruf mich an unter 0176 12345678."
    print("\nOriginal:", sample)
    print("Scrubbed:", scrubber.scrub(sample))
    
    # Export for mobile
    scrubber.export_mobile_config()
