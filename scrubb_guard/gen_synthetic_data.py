"""
Synthetic Data Generator for Safety Classification

Translates English suicide detection dataset to German and generates
hard negative examples (safe sentences with dangerous keywords).

Usage:
    python gen_synthetic_data.py --translate 500 --negations 200
    python gen_synthetic_data.py --translate 0 --negations 100  # Only negations
    python gen_synthetic_data.py --resume  # Continue from previous run
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
INPUT_CSV = PROJECT_ROOT / "data" / "datasets" / "Suicide_Detection.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "synthetic" / "german_safety_augmented.csv"

# Model choices
TRANSLATION_MODEL = "mistral-small-latest"  # Cheaper/faster for translation
GENERATION_MODEL = "mistral-large-latest"   # Better logic for hard negatives

# Rate limiting
API_DELAY_SECONDS = 0.5  # Delay between API calls

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


def translate_text(client: Mistral, text: str) -> str | None:
    """Translate a single text from English to German."""
    try:
        response = client.chat.complete(
            model=TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_PROMPT},
                {"role": "user", "content": text}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return None


def generate_negations(client: Mistral, count: int) -> list[str]:
    """Generate hard negative examples (safe sentences with dangerous keywords)."""
    logger.info(f"Generating {count} hard negative examples...")
    negations = []
    
    batches = count // 10
    if count % 10 > 0:
        batches += 1
    
    for i in tqdm(range(batches), desc="Generating negations"):
        try:
            response = client.chat.complete(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": NEGATION_PROMPT}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            batch_sentences = data.get("sentences", [])
            negations.extend(batch_sentences)
            
            time.sleep(API_DELAY_SECONDS)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
    
    # Trim to exact count
    return negations[:count]


def load_existing_translations(output_path: Path) -> set[str]:
    """Load already translated texts to enable resume."""
    seen = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('source_text'):
                    seen.add(row['source_text'])
    return seen


def translate_dataset(
    client: Mistral,
    input_path: Path,
    output_path: Path,
    samples_per_class: int,
    resume: bool = True
) -> int:
    """Translate English dataset to German."""
    logger.info(f"Loading dataset from {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {input_path}. "
            "Download from: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch"
        )
    
    df = pd.read_csv(input_path)
    
    # Filter to shorter texts (mobile-friendly)
    df = df[df['text'].str.len() < 500]
    
    # Sample balanced classes
    df_suicide = df[df['class'] == 'suicide'].sample(
        min(samples_per_class, len(df[df['class'] == 'suicide'])),
        random_state=42
    )
    df_safe = df[df['class'] == 'non-suicide'].sample(
        min(samples_per_class, len(df[df['class'] == 'non-suicide'])),
        random_state=42
    )
    
    combined_df = pd.concat([df_suicide, df_safe])
    logger.info(f"Selected {len(combined_df)} samples ({len(df_suicide)} suicide, {len(df_safe)} non-suicide)")
    
    # Load existing translations for resume
    already_translated = set()
    if resume:
        already_translated = load_existing_translations(output_path)
        logger.info(f"Found {len(already_translated)} existing translations (resume mode)")
    
    # Prepare output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and resume
    
    translated_count = 0
    
    with open(output_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'label', 'source', 'source_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Translating"):
            source_text = row['text']
            
            # Skip if already translated
            if source_text in already_translated:
                continue
            
            # Translate
            german_text = translate_text(client, source_text)
            
            if german_text:
                label = "UNSAFE" if row['class'] == 'suicide' else "SAFE"
                writer.writerow({
                    'text': german_text,
                    'label': label,
                    'source': 'kaggle_translated',
                    'source_text': source_text
                })
                translated_count += 1
                f.flush()  # Flush after each write for resume safety
            
            time.sleep(API_DELAY_SECONDS)
    
    return translated_count


def add_negations(output_path: Path, negations: list[str]) -> int:
    """Append hard negation examples to the output file."""
    file_exists = output_path.exists()
    
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['text', 'label', 'source', 'source_text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for text in negations:
            writer.writerow({
                'text': text,
                'label': 'SAFE',
                'source': 'hard_negative_gen',
                'source_text': ''
            })
    
    return len(negations)


def main():
    parser = argparse.ArgumentParser(
        description="Generate German safety classification training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--translate", "-t",
        type=int,
        default=500,
        help="Number of samples per class to translate (default: 500)"
    )
    parser.add_argument(
        "--negations", "-n",
        type=int,
        default=200,
        help="Number of hard negative examples to generate (default: 200)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from previous run (skip already translated texts)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})"
    )
    
    args = parser.parse_args()
    
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY not found. Set it in config/.env")
        return
    
    logger.info(f"Starting data generation: translate={args.translate}, negations={args.negations}")
    
    total_added = 0
    
    with Mistral(api_key=MISTRAL_API_KEY) as client:
        # 1. Translate dataset
        if args.translate > 0:
            translated = translate_dataset(
                client,
                INPUT_CSV,
                args.output,
                samples_per_class=args.translate,
                resume=args.resume
            )
            total_added += translated
            logger.info(f"Translated {translated} new samples")
        
        # 2. Generate hard negatives
        if args.negations > 0:
            negations = generate_negations(client, args.negations)
            added = add_negations(args.output, negations)
            total_added += added
            logger.info(f"Added {added} hard negative examples")
    
    logger.info(f"Done! Added {total_added} rows to {args.output}")
    logger.info("Next: Run 'python train_safety_model.py train' with the new dataset")


if __name__ == "__main__":
    main()
