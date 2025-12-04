"""
Safety Model Training Pipeline

Train a German TinyBERT model for safety classification (SAFE vs UNSAFE).
Supports training, ONNX export, and quantization as separate stages.

Usage:
    python train_safety_model.py train      # Fine-tune the model
    python train_safety_model.py export     # Export to ONNX
    python train_safety_model.py quantize   # Quantize ONNX model
    python train_safety_model.py all        # Run full pipeline
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
)
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- CONFIGURATION ---
MODEL_ID = "dvm1983/TinyBERT_General_4L_312D_de"  # Pre-trained German TinyBERT

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "synthetic" / "german_safety_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "safety_classifier"
ONNX_DIR = PROJECT_ROOT / "models" / "safety_classifier_onnx"

LABEL_MAP = {"SAFE": 0, "UNSAFE": 1}
ID2LABEL = {0: "SAFE", 1: "UNSAFE"}
LABEL2ID = {"SAFE": 0, "UNSAFE": 1}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 score."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_and_prepare_data(tokenizer: PreTrainedTokenizer) -> tuple[Dataset, Dataset, Any]:
    """Load CSV data, tokenize, and prepare for training."""
    logger.info(f"Loading data from {DATA_FILE}")
    
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_FILE}. "
            "Please run synthetic_data_generator.py first."
        )
    
    df = pd.read_csv(DATA_FILE)
    
    logger.info("Dataset Distribution:")
    logger.info(f"\n{df['label'].value_counts()}")
    
    # Map text labels to integers
    df["label"] = df["label"].map(LABEL_MAP)
    
    # Convert to HuggingFace Dataset and split
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return tokenized_datasets["train"], tokenized_datasets["test"], data_collator


def train_model() -> Path:
    """Fine-tune the German TinyBERT model for safety classification."""
    logger.info(f"Starting training with model: {MODEL_ID}")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Load and prepare data
    train_dataset, eval_dataset, data_collator = load_and_prepare_data(tokenizer)
    
    # Load model
    logger.info("Loading model for fine-tuning...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,  # Increased, early stopping will prevent overfitting
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=10,
    )
    
    # Initialize trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    logger.info("Training started...")
    trainer.train()
    
    # Evaluate
    logger.info("Training complete. Evaluating...")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")
    
    # Save the best model
    model_path = OUTPUT_DIR / "best_model"
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    
    logger.info(f"Model saved to {model_path}")
    return model_path


def export_to_onnx(model_path: Path | None = None) -> Path:
    """Export the trained PyTorch model to ONNX format."""
    if model_path is None:
        model_path = OUTPUT_DIR / "best_model"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run training first: python train_safety_model.py train"
        )
    
    logger.info(f"Exporting model from {model_path} to ONNX...")
    
    # Ensure output directory exists
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and export to ONNX
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path),
        export=True,
    )
    ort_model.save_pretrained(str(ONNX_DIR))
    
    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.save_pretrained(str(ONNX_DIR))
    
    logger.info(f"ONNX model saved to {ONNX_DIR}")
    return ONNX_DIR


def quantize_model(onnx_path: Path | None = None) -> Path:
    """Quantize the ONNX model to int8 for smaller size and faster inference."""
    if onnx_path is None:
        onnx_path = ONNX_DIR
    
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. "
            "Please run export first: python train_safety_model.py export"
        )
    
    logger.info(f"Quantizing model at {onnx_path}...")
    
    quantizer = ORTQuantizer.from_pretrained(str(onnx_path))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    
    quantizer.quantize(
        save_dir=str(onnx_path),
        quantization_config=qconfig,
    )
    
    quantized_model_path = onnx_path / "model_quantized.onnx"
    logger.info(f"Quantized model saved to {quantized_model_path}")
    
    return quantized_model_path


def run_full_pipeline() -> None:
    """Run the complete pipeline: train, export, and quantize."""
    logger.info("Running full pipeline...")
    
    model_path = train_model()
    onnx_path = export_to_onnx(model_path)
    quantized_path = quantize_model(onnx_path)
    
    logger.info(f"Pipeline complete! Mobile-ready model at: {quantized_path}")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train, export, and quantize a safety classification model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["train", "export", "quantize", "all"],
        default="all",
        nargs="?",
        help="Pipeline stage to run (default: all)",
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model()
    elif args.command == "export":
        export_to_onnx()
    elif args.command == "quantize":
        quantize_model()
    else:  # "all" or default
        run_full_pipeline()


if __name__ == "__main__":
    main()
