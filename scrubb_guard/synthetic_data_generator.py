import os
import json
import csv
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

# Load .env from config/ directory (relative to this file's location)
config_env = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(config_env)

# CONFIGURATION
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-large-latest"
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_FILE = OUTPUT_DIR / "german_safety_dataset.csv"
TARGET_COUNT = 200

# DIVERSITY INJECTION:
# We will randomly inject one of these scenarios into the prompt for each batch.
SCENARIOS = [
    "Focus on side effects of medication (dizziness, weight gain) vs overdose intent.",
    "Focus on auditory hallucinations (voices are loud vs voices commanding harm).",
    "Focus on feelings of hopelessness vs active planning.",
    "Focus on social isolation and loneliness vs saying goodbye to friends.",
    "Focus on suicidal thoughts vs feeling like a burden to others.",
    "Focus on sleep disturbances and nightmares vs fear of never waking up.",
    "Focus on paranoia and trust issues with doctors.",
    "Focus on fear of being watched or followed.",
    "Focus on positive progress and therapy goals.",
    "Focus on past trauma vs current immediate danger.",
    "Focus on figurative language ('I am exploding', 'I am dying inside') vs literal threats."
]

# THE CORE SYSTEM PROMPT (The "Brain")
SYSTEM_PROMPT = """
You are a clinical psychiatrist specializing in Schizophrenia. 
Your task is to generate synthetic user journal entries in German.

Classes:
1. UNSAFE: Active suicidal intent, self-harm planning, or immediate emergency.
2. SAFE: 
   - Descriptions of hallucinations/voices WITHOUT intent to act.
   - Negations ("I am not suicidal").
   - Past tense ("I tried to kill myself last year, but I am better now").
   - Metaphors ("I feel like I am dying").
   - Mundane therapy updates ("I took my meds").

Generate 10 examples in JSON format:
{
  "examples": [
      {"text": "example sentence", "label": "SAFE" or "UNSAFE", "category": "hallucination_safe" or "active_risk"}
  ]
}

CRITICAL RULES:
- Use natural, informal German (including typos or lower case).
- Vary the length (very short outbursts vs. longer explanations).
"""

def generate_batch(scenario):
    try:
        with Mistral(api_key=MISTRAL_API_KEY) as mistral:
            response = mistral.chat.complete(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate 5 SAFE and 5 UNSAFE examples. {scenario}"}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("examples", [])
    except Exception as e:
        print(f"Error: {e}")
        return []

# MAIN LOOP
def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    file_exists = os.path.isfile(OUTPUT_FILE)
    
    # Tracking set to prevent duplicates in this run
    seen_texts = set()
    
    # If file exists, load existing texts to avoid duplicates across runs
    if file_exists:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_texts.add(row['text'])

    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'label', 'category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        current_count = 0
        batch_num = 0
        while current_count < TARGET_COUNT:
            # Rotate through scenarios in order
            scenario = SCENARIOS[batch_num % len(SCENARIOS)]
            print(f"Generating batch ({current_count}/{TARGET_COUNT}) - Scenario: {scenario[:30]}...")
            batch_num += 1
            
            batch = generate_batch(scenario)
            
            if not batch:
                continue

            unique_entries = 0
            for entry in batch:
                # Deduplication Check
                if entry['text'] not in seen_texts:
                    writer.writerow(entry)
                    seen_texts.add(entry['text'])
                    unique_entries += 1
            
            current_count += unique_entries
            print(f"  -> Added {unique_entries} unique entries.")

    print(f"Done! Saved to {OUTPUT_FILE} with {len(seen_texts)} total unique entries.")

if __name__ == "__main__":
    main()