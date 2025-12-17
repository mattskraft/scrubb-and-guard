"""
Unified Data Generator for Safety Classification

Generate training data for German safety classification using multiple strategies:
- synthetic: Generate completely synthetic German examples from scratch
- translate: Translate English suicide detection dataset to German
- negations: Generate hard negatives (safe sentences with dangerous keywords)
- twins: Generate anti-evil twins (flip UNSAFE to SAFE keeping style)
- clinical: Generate clinical scenarios with ternary labels (UNSAFE, RELEVANT_BUT_SAFE, SAFE_CONTEXT)
- answers: Generate free text answers from kiso_app_merged_structured.json

Usage:
    python data_generator.py synthetic --id v1 --count 200
    python data_generator.py translate --id v2 --count 500
    python data_generator.py negations --id v2 --count 100
    python data_generator.py twins --id v3 --input data/synthetic/german_safety_v2_translate.csv --count 500
    python data_generator.py clinical --id v5
    python data_generator.py answers --id v6
    python data_generator.py all --id v4 --synthetic 100 --translate 500 --negations 100 --twins 200

Output files are named: german_safety_{id}_{mode}.csv
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
from tqdm import tqdm

# --- CONFIGURATION ---
config_env = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(config_env)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PROJECT_ROOT = Path(__file__).parent.parent
KAGGLE_CSV = PROJECT_ROOT / "data" / "datasets" / "Suicide_Detection.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"

# Model choices
FAST_MODEL = "mistral-small-latest"    # Translation, simple tasks
SMART_MODEL = "mistral-large-latest"   # Complex generation (synthetic, twins, negations)

# Rate limiting
API_DELAY_SECONDS = 0.5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- PROMPTS ---

TRANSLATE_PROMPT = """You are a professional translator. Translate the following text into natural, informal German.
Preserve the emotional tone (sadness, anger, hopelessness). 
If the text is slang, translate to German slang.
Output ONLY the German translation, nothing else."""

NEGATION_PROMPT = """Generate 10 German sentences that contain suicidal keywords (sterben, tod, umbringen, suizid, Tabletten, Medikamente) 
BUT are semantically SAFE because they use strong negation, past tense, or positive context.

Examples:
- "Ich will nicht sterben." (SAFE - negation)
- "Ich habe keine Gedanken daran, mich umzubringen." (SAFE - negation)
- "Früher wollte ich sterben, aber jetzt liebe ich das Leben." (SAFE - past tense + positive)
- "Ich habe Angst vor dem Tod." (SAFE - fear, not intent)
- "Ich habe meine Medikamente genommen und fühle mich besser." (SAFE - positive medication context)
- "Die Therapie hilft mir, ich nehme meine Tabletten regelmäßig." (SAFE - positive therapy context)

Return strictly a JSON object: {"sentences": ["sentence 1", "sentence 2", ...]}"""

TWIN_PROMPT = """You are a linguist specializing in German internet slang and psychology.
Your task: Rewrite the Input Text to create an "Anti-Evil Twin".

Rules:
1. MEANING: Flip the meaning from "Suicidal/Hopeless" to "Resilient/Hopeful/Safe". 
   - Instead of wanting to die, the user wants to live/fight/sleep well.
   - Instead of "I have a gun", "I threw away the gun".
2. STYLE: You MUST copy the exact style of the input.
   - If input has typos, keep typos.
   - If input is short/angry, keep it short/angry (but angry at the world, not life).
3. VOCABULARY: Reuse as many words from the input as possible.

Examples:
Input: "ich kann nicht mehr ich will sterben"
Output: "ich kann nicht mehr leiden aber ich will leben"

Input: "Habe Tabletten genommen, ciao"
Output: "Habe Tabletten weggeworfen, hallo Leben"

Input Text: "{text}"

Output strictly JSON: {{"twin_text": "..."}}"""

SYNTHETIC_SYSTEM_PROMPT = """You are a clinical psychiatrist specializing in Schizophrenia. 
Your task is to generate synthetic user journal entries in German.

Classes:
1. UNSAFE: Active suicidal intent, self-harm planning, or immediate emergency.
2. SAFE: 
   - Descriptions of hallucinations/voices WITHOUT intent to act.
   - Negations ("I am not suicidal").
   - Past tense ("I tried to kill myself last year, but I am better now").
   - Metaphors ("I feel like I am dying").
   - Mundane therapy updates ("I took my meds").

Generate examples in JSON format:
{
  "examples": [
      {"text": "example sentence", "label": "SAFE" or "UNSAFE"}
  ]
}

CRITICAL RULES:
- Use natural, informal German (including typos or lower case).
- Vary the length (very short outbursts of only a few words vs. longer explanations)."""

SYNTHETIC_SCENARIOS = [
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
    "Focus on figurative language ('I am exploding', 'I am dying inside') vs literal threats.",
]

# --- CLINICAL PROMPTS (German, ternary labels) ---

CLINICAL_SYSTEM_PROMPT = """Du bist ein Psychiater. Deine Aufgabe ist es Tagebucheinträge eines Patienten zu generieren.
Schreibe aus der Ich-Perspektive in natürlicher, informeller deutscher Sprache."""

CLINICAL_PROMPT = """Generiere Einträge aus der Ich-Perspektive eines Patienten.
Unterscheide scharf zwischen:
1. UNSAFE: Akute Eigengefährdung, konkrete Suizidpläne, Befehls-Halluzinationen.
2. RELEVANT_BUT_SAFE: Beschreibung von Leid/Symptomen (Stimmen, Schmerz, Angst), 
   ABER ohne akute Absicht, sich etwas anzutun. Distanzierung oder reine Zustandsbeschreibung."""

TRICKY_PROMPT = """Generiere Einträge, die gefährlich klingen KÖNNTEN (Triggerwörter enthalten), 
es aber im Kontext NICHT sind.
Label ist immer: SAFE_CONTEXT"""

CLINICAL_OUTPUT_FORMAT = """Generiere Beispiele in JSON-Format:
{
  "examples": [
      {"text": "Tagebucheintrag", "label": "LABEL"}
  ]
}"""

CLINICAL_SCENARIOS = [
    "Thema: Stimmenhören. UNSAFE = Befehl zum Springen. RELEVANT_BUT_SAFE = Stimmen kommentieren nur.",
    "Thema: Medikamente. UNSAFE = Überdosis sammeln. RELEVANT_BUT_SAFE = Über Nebenwirkung klagen.",
    "Thema: Hoffnungslosigkeit. UNSAFE = Abschiedsbrief schreiben. RELEVANT_BUT_SAFE = Über Traurigkeit reden.",
    "Thema: Selbsttötung. UNSAFE = Selbsttötung planen. RELEVANT_BUT_SAFE = Über Selbsttötung reden.",
    "Thema: Selbstverletzung. UNSAFE = Selbstverletzung planen. RELEVANT_BUT_SAFE = Über Selbstverletzung reden.",
]

TRICKY_SCENARIOS = [
    "Nutze Metaphern wie 'Ich sterbe vor Lachen', 'Das bringt mich um (Job)'.",
    "Nutze Verneinungen: 'Ich will NICHT sterben', 'Ich tue mir NICHTS an'.",
    "Nutze Vergangenheitsform: 'Früher wollte ich tot sein, heute nicht mehr'.",
]

OUTPUT_LENGTHS = [
    "MAXIMAL 5 Wörter.",
    "GENAU 1 Satz mit 6-12 Wörtern.",
    "2-3 Sätze mit insgesamt 20-40 Wörtern.",
]

ANSWER_PROMPT = """Formuliere eine realistische Antwort für die Frage. Formuliere NUR die Antwort, keine weiteren Texte oder Struktur. Sprache: Deutsch, natürlich klingend.

KONTEXT:
{exercise}
{text}
Frage: {question}

Länge: {length}"""

# Default path for kiso_app_merged_structured.json
DEFAULT_EXERCISES_JSON = Path("/home/matthias/Kiso/code/projects/summary-prompt-lab/data/processed/kiso_app_merged_structured.json")


# --- GENERATION FUNCTIONS ---

def generate_synthetic(client: Mistral, count: int, output_path: Path) -> int:
    """Generate completely synthetic German safety examples."""
    logger.info(f"Generating {count} synthetic examples...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_texts: set[str] = set()
    
    # Load existing if file exists
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_texts.add(row['text'])
    
    file_exists = output_path.exists()
    generated = 0
    batch_num = 0
    
    with open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'label', 'source']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        pbar = tqdm(total=count, desc="Generating synthetic")
        while generated < count:
            scenario = SYNTHETIC_SCENARIOS[batch_num % len(SYNTHETIC_SCENARIOS)]
            batch_num += 1
            
            try:
                response = client.chat.complete(
                    model=SMART_MODEL,
                    messages=[
                        {"role": "system", "content": SYNTHETIC_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Generate 5 SAFE and 5 UNSAFE examples. {scenario}"}
                    ],
                    response_format={"type": "json_object"}
                )
                data = json.loads(response.choices[0].message.content)
                examples = data.get("examples", [])
                
                for entry in examples:
                    if entry['text'] not in seen_texts and generated < count:
                        writer.writerow({
                            'text': entry['text'],
                            'label': entry['label'],
                            'source': 'synthetic'
                        })
                        seen_texts.add(entry['text'])
                        generated += 1
                        pbar.update(1)
                        f.flush()
                
                time.sleep(API_DELAY_SECONDS)
                
            except Exception as e:
                logger.error(f"Synthetic generation error: {e}")
        
        pbar.close()
    
    return generated


def translate_dataset(client: Mistral, count: int, output_path: Path, resume: bool = True) -> int:
    """Translate English suicide dataset to German."""
    logger.info(f"Translating {count} samples per class from Kaggle dataset...")
    
    if not KAGGLE_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {KAGGLE_CSV}. "
            "Download from: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch"
        )
    
    df = pd.read_csv(KAGGLE_CSV)
    df = df[df['text'].str.len() < 350]  # Filter for token limit
    
    # Sample balanced classes
    df_suicide = df[df['class'] == 'suicide'].sample(
        min(count, len(df[df['class'] == 'suicide'])), random_state=42
    )
    df_safe = df[df['class'] == 'non-suicide'].sample(
        min(count, len(df[df['class'] == 'non-suicide'])), random_state=42
    )
    combined_df = pd.concat([df_suicide, df_safe])
    logger.info(f"Selected {len(combined_df)} samples")
    
    # Load existing for resume
    already_done: set[str] = set()
    if resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('source_text'):
                    already_done.add(row['source_text'])
        logger.info(f"Resuming: {len(already_done)} already translated")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and resume
    translated = 0
    
    with open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'label', 'source', 'source_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Translating"):
            if row['text'] in already_done:
                continue
            
            try:
                response = client.chat.complete(
                    model=FAST_MODEL,
                    messages=[
                        {"role": "system", "content": TRANSLATE_PROMPT},
                        {"role": "user", "content": row['text']}
                    ],
                )
                german_text = response.choices[0].message.content.strip()
                
                writer.writerow({
                    'text': german_text,
                    'label': "UNSAFE" if row['class'] == 'suicide' else "SAFE",
                    'source': 'translated',
                    'source_text': row['text']
                })
                translated += 1
                f.flush()
                
            except Exception as e:
                logger.error(f"Translation error: {e}")
            
            time.sleep(API_DELAY_SECONDS)
    
    return translated


def generate_negations(client: Mistral, count: int, output_path: Path) -> int:
    """Generate hard negative examples (safe sentences with dangerous keywords)."""
    logger.info(f"Generating {count} hard negatives...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    negations: list[str] = []
    
    batches = (count + 9) // 10
    for _ in tqdm(range(batches), desc="Generating negations"):
        try:
            response = client.chat.complete(
                model=SMART_MODEL,
                messages=[{"role": "user", "content": NEGATION_PROMPT}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            negations.extend(data.get("sentences", []))
            time.sleep(API_DELAY_SECONDS)
        except Exception as e:
            logger.error(f"Negation generation error: {e}")
    
    negations = negations[:count]
    
    with open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'label', 'source', 'source_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for text in negations:
            writer.writerow({
                'text': text,
                'label': 'SAFE',
                'source': 'hard_negative',
                'source_text': ''
            })
    
    return len(negations)


def generate_twins(client: Mistral, count: int, input_path: Path, output_path: Path) -> int:
    """Generate anti-evil twins from UNSAFE examples."""
    logger.info(f"Generating {count} anti-evil twins from {input_path}...")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    unsafe_df = df[df['label'] == 'UNSAFE']
    
    if len(unsafe_df) == 0:
        logger.warning("No UNSAFE examples found in input file")
        return 0
    
    # Sample up to count
    sample_size = min(count, len(unsafe_df))
    unsafe_sample = unsafe_df.sample(sample_size, random_state=42)
    logger.info(f"Processing {sample_size} UNSAFE examples")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    generated = 0
    
    with open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'label', 'source', 'source_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for _, row in tqdm(unsafe_sample.iterrows(), total=len(unsafe_sample), desc="Generating twins"):
            original_text = row['text']
            
            try:
                prompt = TWIN_PROMPT.format(text=original_text)
                response = client.chat.complete(
                    model=SMART_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                data = json.loads(response.choices[0].message.content)
                twin_text = data.get("twin_text")
                
                if twin_text:
                    # Write original UNSAFE
                    writer.writerow({
                        'text': original_text,
                        'label': 'UNSAFE',
                        'source': 'twin_original',
                        'source_text': ''
                    })
                    # Write twin SAFE
                    writer.writerow({
                        'text': twin_text,
                        'label': 'SAFE',
                        'source': 'twin_generated',
                        'source_text': original_text
                    })
                    generated += 1
                    f.flush()
                
            except Exception as e:
                logger.error(f"Twin generation error: {e}")
            
            time.sleep(API_DELAY_SECONDS)
    
    return generated


def generate_answers(client: Mistral, input_path: Path, output_path: Path) -> int:
    """Generate free text answers from kiso_app_merged_structured.json.
    
    Extracts patterns where a Text block is followed by a Question with 
    AnswerOptions="free_text". For each pattern, generates a realistic 
    German answer using Mistral.
    
    JSON structure: Category -> Subcategory -> Exercise Name -> [blocks]
    Each block is either {"Text": "..."} or {"Question": "...", "AnswerOptions": ...}
    
    Args:
        client: Mistral API client
        input_path: Path to kiso_app_merged_structured.json
        output_path: Path for output CSV file
    
    Returns:
        Number of answers generated
    """
    logger.info(f"Generating answers from {input_path}...")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    generated = 0
    length_idx = 0
    
    # Collect all patterns first for progress bar
    # Structure: Category -> Subcategory -> Exercise -> [blocks]
    patterns = []
    for category, subcategories in data.items():
        if not isinstance(subcategories, dict):
            continue
        for subcategory, exercises in subcategories.items():
            if not isinstance(exercises, dict):
                continue
            for exercise_name, blocks in exercises.items():
                if not isinstance(blocks, list):
                    continue
                
                last_text = None
                for block in blocks:
                    # Track Text blocks
                    if "Text" in block:
                        text_content = block.get("Text", "")
                        if isinstance(text_content, str) and text_content.strip():
                            last_text = text_content.strip()
                        continue
                    
                    # Check for Question with free_text AnswerOptions
                    if "Question" in block:
                        answer_options = block.get("AnswerOptions")
                        if answer_options == "free_text":
                            question_text = block.get("Question", "")
                            # We have a valid pattern: Text -> Question with free_text
                            if last_text and question_text:
                                patterns.append({
                                    "exercise": exercise_name,
                                    "text": last_text,
                                    "question": question_text
                                })
    
    logger.info(f"Found {len(patterns)} text+question+free_text patterns")
    
    with open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['exercise', 'text', 'question', 'answer', 'length_instruction']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for pattern in tqdm(patterns, desc="Generating answers"):
            length_instruction = OUTPUT_LENGTHS[length_idx % len(OUTPUT_LENGTHS)]
            length_idx += 1
            
            prompt = ANSWER_PROMPT.format(
                exercise=pattern["exercise"],
                text=pattern["text"],
                question=pattern["question"],
                length=length_instruction
            )
            
            try:
                response = client.chat.complete(
                    model=FAST_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=150
                )
                answer = response.choices[0].message.content.strip()
                
                writer.writerow({
                    'exercise': pattern["exercise"],
                    'text': pattern["text"],
                    'question': pattern["question"],
                    'answer': answer,
                    'length_instruction': length_instruction
                })
                generated += 1
                f.flush()
                
            except Exception as e:
                logger.error(f"Answer generation error: {e}")
            
            time.sleep(API_DELAY_SECONDS)
    
    return generated


def generate_clinical(client: Mistral, output_path: Path) -> int:
    """Generate clinical training data with ternary labels using German prompts.
    
    Uses structured scenarios and length variations:
    - Clinical scenarios: UNSAFE vs RELEVANT_BUT_SAFE
    - Tricky scenarios: SAFE_CONTEXT (false positive prevention)
    """
    logger.info("Generating clinical training data with ternary labels...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_texts: set[str] = set()
    
    # Load existing if file exists
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_texts.add(row['text'])
    
    file_exists = output_path.exists()
    generated = 0
    
    # Calculate total API calls for progress bar
    total_calls = len(CLINICAL_SCENARIOS) * len(OUTPUT_LENGTHS) + len(TRICKY_SCENARIOS) * len(OUTPUT_LENGTHS)
    
    with open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'label', 'source', 'source_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        pbar = tqdm(total=total_calls, desc="Generating clinical data")
        
        # Loop 1: Clinical scenarios (UNSAFE + RELEVANT_BUT_SAFE)
        for scenario in CLINICAL_SCENARIOS:
            for length in OUTPUT_LENGTHS:
                try:
                    system_content = CLINICAL_SYSTEM_PROMPT + "\n" + CLINICAL_PROMPT + "\n" + CLINICAL_OUTPUT_FORMAT
                    user_content = f"{scenario} Generiere 3 Beispiele für jedes Label mit folgender Länge: {length}"
                    
                    response = client.chat.complete(
                        model=SMART_MODEL,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ],
                        response_format={"type": "json_object"}
                    )
                    data = json.loads(response.choices[0].message.content)
                    examples = data.get("examples", [])
                    
                    for entry in examples:
                        text = entry.get('text', '')
                        label = entry.get('label', '')
                        if text and text not in seen_texts and label in ('UNSAFE', 'RELEVANT_BUT_SAFE'):
                            writer.writerow({
                                'text': text,
                                'label': label,
                                'source': 'clinical',
                                'source_text': ''
                            })
                            seen_texts.add(text)
                            generated += 1
                            f.flush()
                    
                    time.sleep(API_DELAY_SECONDS)
                    
                except Exception as e:
                    logger.error(f"Clinical generation error: {e}")
                
                pbar.update(1)
        
        # Loop 2: Tricky scenarios (SAFE_CONTEXT only)
        for scenario in TRICKY_SCENARIOS:
            for length in OUTPUT_LENGTHS:
                try:
                    system_content = CLINICAL_SYSTEM_PROMPT + "\n" + TRICKY_PROMPT + "\n" + CLINICAL_OUTPUT_FORMAT
                    user_content = f"{scenario} Generiere 3 Beispiele mit folgender Länge: {length}"
                    
                    response = client.chat.complete(
                        model=SMART_MODEL,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content}
                        ],
                        response_format={"type": "json_object"}
                    )
                    data = json.loads(response.choices[0].message.content)
                    examples = data.get("examples", [])
                    
                    for entry in examples:
                        text = entry.get('text', '')
                        if text and text not in seen_texts:
                            writer.writerow({
                                'text': text,
                                'label': 'SAFE_CONTEXT',
                                'source': 'tricky',
                                'source_text': ''
                            })
                            seen_texts.add(text)
                            generated += 1
                            f.flush()
                    
                    time.sleep(API_DELAY_SECONDS)
                    
                except Exception as e:
                    logger.error(f"Tricky generation error: {e}")
                
                pbar.update(1)
        
        pbar.close()
    
    return generated


# --- MAIN ---

def get_output_path(dataset_id: str, mode: str) -> Path:
    """Generate output filename with dataset ID and mode."""
    return OUTPUT_DIR / f"german_safety_{dataset_id}_{mode}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Generate German safety classification training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Generation mode")
    
    # Synthetic mode
    syn_parser = subparsers.add_parser("synthetic", help="Generate synthetic examples from scratch")
    syn_parser.add_argument("--id", required=True, help="Dataset identifier (e.g., v1)")
    syn_parser.add_argument("--count", "-c", type=int, default=200, help="Number of examples")
    
    # Translate mode
    trans_parser = subparsers.add_parser("translate", help="Translate Kaggle suicide dataset")
    trans_parser.add_argument("--id", required=True, help="Dataset identifier")
    trans_parser.add_argument("--count", "-c", type=int, default=500, help="Samples per class")
    trans_parser.add_argument("--resume", "-r", action="store_true", help="Resume previous run")
    
    # Negations mode
    neg_parser = subparsers.add_parser("negations", help="Generate hard negatives")
    neg_parser.add_argument("--id", required=True, help="Dataset identifier")
    neg_parser.add_argument("--count", "-c", type=int, default=100, help="Number of negations")
    
    # Twins mode
    twin_parser = subparsers.add_parser("twins", help="Generate anti-evil twins")
    twin_parser.add_argument("--id", required=True, help="Dataset identifier")
    twin_parser.add_argument("--input", "-i", type=Path, required=True, help="Input CSV with UNSAFE examples")
    twin_parser.add_argument("--count", "-c", type=int, default=500, help="Number of twins")
    
    # Clinical mode (German prompts, ternary labels)
    clinical_parser = subparsers.add_parser("clinical", help="Generate clinical scenarios with ternary labels (German)")
    clinical_parser.add_argument("--id", required=True, help="Dataset identifier")
    
    # Answers mode (generate free text answers from exercises JSON)
    answers_parser = subparsers.add_parser("answers", help="Generate free text answers from kiso_app_merged_structured.json")
    answers_parser.add_argument("--id", required=True, help="Dataset identifier")
    answers_parser.add_argument("--input", "-i", type=Path, default=DEFAULT_EXERCISES_JSON,
                                help="Path to kiso_app_merged_structured.json (default: summary-prompt-lab)")
    
    # All-in-one mode
    all_parser = subparsers.add_parser("all", help="Run all generation modes")
    all_parser.add_argument("--id", required=True, help="Dataset identifier")
    all_parser.add_argument("--synthetic", type=int, default=100, help="Synthetic count")
    all_parser.add_argument("--translate", type=int, default=500, help="Translate count per class")
    all_parser.add_argument("--negations", type=int, default=100, help="Negations count")
    all_parser.add_argument("--twins", type=int, default=200, help="Twins count")
    all_parser.add_argument("--resume", "-r", action="store_true", help="Resume translation")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY not found. Set it in config/.env")
        return
    
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        
        if args.command == "synthetic":
            output = get_output_path(args.id, "synthetic")
            count = generate_synthetic(client, args.count, output)
            logger.info(f"Generated {count} synthetic examples -> {output}")
        
        elif args.command == "translate":
            output = get_output_path(args.id, "translate")
            count = translate_dataset(client, args.count, output, args.resume)
            logger.info(f"Translated {count} examples -> {output}")
        
        elif args.command == "negations":
            output = get_output_path(args.id, "negations")
            count = generate_negations(client, args.count, output)
            logger.info(f"Generated {count} negations -> {output}")
        
        elif args.command == "twins":
            output = get_output_path(args.id, "twins")
            count = generate_twins(client, args.count, args.input, output)
            logger.info(f"Generated {count} twin pairs -> {output}")
        
        elif args.command == "clinical":
            output = get_output_path(args.id, "clinical")
            count = generate_clinical(client, output)
            logger.info(f"Generated {count} clinical examples -> {output}")
        
        elif args.command == "answers":
            output = get_output_path(args.id, "answers")
            count = generate_answers(client, args.input, output)
            logger.info(f"Generated {count} answers -> {output}")
        
        elif args.command == "all":
            output = get_output_path(args.id, "combined")
            total = 0
            
            if args.synthetic > 0:
                total += generate_synthetic(client, args.synthetic, output)
            
            if args.translate > 0:
                total += translate_dataset(client, args.translate, output, args.resume)
            
            if args.negations > 0:
                total += generate_negations(client, args.negations, output)
            
            if args.twins > 0:
                # For twins, we need existing UNSAFE data - use the same output file
                total += generate_twins(client, args.twins, output, output)
            
            logger.info(f"Generated {total} total entries -> {output}")
    
    logger.info("Done! Use 'python train_safety_model.py train -d <output_file>' to train.")


if __name__ == "__main__":
    main()

