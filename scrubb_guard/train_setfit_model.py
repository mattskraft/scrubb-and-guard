"""
SetFit Safety Model Training Pipeline

Train a multilingual SetFit model for safety classification (SAFE vs UNSAFE).
SetFit uses contrastive learning and works exceptionally well with few-shot data.

Usage:
    python train_setfit_model.py train --model-id setfit_v1 --data data/synthetic/final_setfit_train.csv
    python train_setfit_model.py test --model-id setfit_v1
    python train_setfit_model.py all --model-id setfit_v1 --data data/synthetic/final_setfit_train.csv

All outputs are saved to: models/{model_id}/
"""

import argparse
import logging
from pathlib import Path
from typing import List

from datasets import load_dataset, Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sentence_transformers.losses import CosineSimilarityLoss

# --- CONFIGURATION ---
BASE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Global variables set by CLI
DATA_FILE: Path = None  # type: ignore
MODEL_ID: str = None    # type: ignore
OUTPUT_DIR: Path = None # type: ignore

LABEL_MAP = {"SAFE": 0, "UNSAFE": 1}
ID2LABEL = {0: "SAFE", 1: "UNSAFE"}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_training_data() -> Dataset:
    """Load CSV data and prepare for SetFit training."""
    logger.info(f"Loading data from {DATA_FILE}")
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. "
            "Please run the data generator first."
        )
    
    # Load dataset from CSV
    dataset = load_dataset("csv", data_files=str(DATA_FILE))
    train_dataset = dataset["train"]
    
    # Log dataset distribution
    labels = train_dataset["label"]
    safe_count = labels.count(0)
    unsafe_count = labels.count(1)
    
    logger.info("Dataset Distribution:")
    logger.info(f"  SAFE (0):   {safe_count}")
    logger.info(f"  UNSAFE (1): {unsafe_count}")
    logger.info(f"  Total:      {len(train_dataset)}")
    logger.info(f"  Ratio:      1:{safe_count/unsafe_count:.2f}" if unsafe_count > 0 else "")
    
    return train_dataset


def train_model() -> Path:
    """Fine-tune the SetFit model for safety classification."""
    logger.info(f"Starting SetFit training with base model: {BASE_MODEL}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    train_dataset = load_training_data()
    
    logger.info(f"Training mit {len(train_dataset)} Beispielen...")
    
    # Load base model
    logger.info("Loading SetFit model...")
    model = SetFitModel.from_pretrained(BASE_MODEL)
    
    # Training arguments (SetFit v1.0+ API)
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        batch_size=16,
        num_iterations=20,  # Generates 20 pairs per sentence (data multiplication!)
        num_epochs=1,       # One epoch is usually enough for SetFit
        logging_steps=10,
        loss=CosineSimilarityLoss,
    )
    
    # Initialize trainer (SetFit v1.0+ uses Trainer instead of SetFitTrainer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        metric="accuracy",
        column_mapping={"text": "text", "label": "label"},
    )
    
    # Train
    logger.info("Starting fine-tuning (Contrastive Learning)...")
    trainer.train()
    
    # Save model
    model_path = OUTPUT_DIR / "best_model"
    model.save_pretrained(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def test_model(model_path: Path = None) -> None:
    """Test the trained model with example sentences."""
    if model_path is None:
        model_path = OUTPUT_DIR / "best_model"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please run training first: python train_setfit_model.py train -m {MODEL_ID} -d <data.csv>"
        )
    
    logger.info(f"Loading model from {model_path}")
    model = SetFitModel.from_pretrained(str(model_path))
    
    # Test sentences covering different cases
    test_sentences: List[str] = [
        # Should be SAFE (0)
        "Ich esse heute Pizza.",
        "Das Wetter ist heute schön.",
        "Ich sterbe vor Lachen über diesen Witz.",  # Metaphor
        "Die Stimmen sind da, aber ich tue nichts.",  # Clinical Safe
        "Ich habe früher auch mal daran gedacht, aber das ist vorbei.",  # Past tense
        
        # Should be UNSAFE (1)
        "Ich will nicht mehr leben, es ist vorbei.",
        "Ich nehme das Messer und mache Schluss.",
        "Keiner würde mich vermissen wenn ich weg bin.",
        "Heute nehme ich alle Tabletten auf einmal.",
        "Ich weiß genau wie ich es machen werde.",
    ]
    
    logger.info("\n" + "=" * 60)
    logger.info("FUNCTIONAL TEST")
    logger.info("=" * 60)
    
    predictions = model.predict(test_sentences)
    
    for text, label in zip(test_sentences, predictions):
        label_int = int(label)
        label_str = ID2LABEL[label_int]
        indicator = "🚨 ALARM" if label_int == 1 else "✅ OK"
        
        logger.info(f"\nText: '{text}'")
        logger.info(f"  → {label_str} {indicator}")
    
    logger.info("\n" + "=" * 60)


def run_full_pipeline() -> None:
    """Run the complete pipeline: train and test."""
    logger.info("Running full pipeline...")
    
    model_path = train_model()
    test_model(model_path)
    
    logger.info(f"\nPipeline complete! Model saved at: {model_path}")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    global DATA_FILE, MODEL_ID, OUTPUT_DIR
    
    parser = argparse.ArgumentParser(
        description="Train and test a SetFit safety classification model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["train", "test", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--model-id", "-m",
        type=str,
        required=True,
        help="Model identifier (outputs saved to models/{model_id}/)",
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        help="Path to training data CSV (required for train/all)",
    )
    parser.add_argument(
        "--base-model", "-b",
        type=str,
        default=BASE_MODEL,
        help=f"Base sentence-transformer model (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--num-iterations", "-n",
        type=int,
        default=20,
        help="Number of contrastive pairs per sentence (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    
    args = parser.parse_args()
    
    # Set global model ID and paths
    MODEL_ID = args.model_id
    OUTPUT_DIR = MODELS_DIR / MODEL_ID / "setfit"
    
    # Validate data argument for commands that need it
    if args.command in ["train", "all"]:
        if args.data is None:
            parser.error(f"--data is required for '{args.command}' command")
        DATA_FILE = args.data
        if not DATA_FILE.exists():
            parser.error(f"Data file not found: {DATA_FILE}")
    
    logger.info(f"Model ID: {MODEL_ID}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    
    if args.command == "train":
        train_model()
    elif args.command == "test":
        test_model()
    elif args.command == "all":
        run_full_pipeline()


if __name__ == "__main__":
    main()

