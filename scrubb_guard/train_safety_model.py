"""
Safety Model Training Pipeline

Train a German TinyBERT model for safety classification (SAFE vs UNSAFE).
Supports training, ONNX export, and quantization as separate stages.

Usage:
    python train_safety_model.py train      # Fine-tune the model
    python train_safety_model.py export     # Export to ONNX
    python train_safety_model.py quantize   # Quantize ONNX model
    python train_safety_model.py evaluate   # Compare original vs quantized
    python train_safety_model.py all        # Run full pipeline
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import torch
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


def preprocess_logits_for_metrics(
    logits: Union[torch.Tensor, tuple], labels: torch.Tensor
) -> torch.Tensor:
    """Extract only the classification logits, ignore any other model outputs."""
    # If logits is a tuple (e.g., (logits, hidden_states)), take just the first element
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


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
    
    # Remove columns not needed for training (keep only input_ids, attention_mask, label)
    columns_to_remove = [col for col in tokenized_datasets["train"].column_names 
                         if col not in ["input_ids", "attention_mask", "label"]]
    tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
    
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
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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


def evaluate_quantization() -> None:
    """Compare original vs quantized model performance on the test split."""
    logger.info("Evaluating quantization impact...")
    
    # Check paths
    model_path = OUTPUT_DIR / "best_model"
    quantized_model_path = ONNX_DIR / "model_quantized.onnx"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Original model not found at {model_path}")
    if not quantized_model_path.exists():
        raise FileNotFoundError(f"Quantized model not found at {quantized_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Reconstruct the SAME test split (seed=42)
    df = pd.read_csv(DATA_FILE)
    df["label"] = df["label"].map(LABEL_MAP)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_data = dataset["test"]
    
    logger.info(f"Test set size: {len(test_data)} samples")
    
    # Prepare test texts and labels
    texts = test_data["text"]
    labels = np.array(test_data["label"])
    
    def get_predictions(model, is_onnx: bool = False) -> np.ndarray:
        """Run inference and return predicted class indices."""
        predictions = []
        for text in texts:
            inputs = tokenizer(
                text, 
                return_tensors="np" if is_onnx else "pt",
                truncation=True, 
                max_length=128
            )
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if is_onnx:
                pred = np.argmax(logits, axis=1)[0]
            else:
                pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)
        return np.array(predictions)
    
    def calc_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate metrics from predictions."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", pos_label=1
        )
        accuracy = accuracy_score(labels, preds)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    # Evaluate original PyTorch model
    logger.info("Evaluating original PyTorch model...")
    original_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    original_model.eval()
    with torch.no_grad():
        original_preds = get_predictions(original_model, is_onnx=False)
    original_metrics = calc_metrics(original_preds, labels)
    
    # Evaluate quantized ONNX model
    logger.info("Evaluating quantized ONNX model...")
    quantized_model = ORTModelForSequenceClassification.from_pretrained(
        str(ONNX_DIR), file_name="model_quantized.onnx"
    )
    quantized_preds = get_predictions(quantized_model, is_onnx=True)
    quantized_metrics = calc_metrics(quantized_preds, labels)
    
    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("QUANTIZATION IMPACT COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Metric':<12} {'Original':<12} {'Quantized':<12} {'Delta':<12}")
    logger.info("-" * 60)
    
    for metric in ["accuracy", "precision", "recall", "f1"]:
        orig = original_metrics[metric]
        quant = quantized_metrics[metric]
        delta = quant - orig
        delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
        logger.info(f"{metric:<12} {orig:<12.4f} {quant:<12.4f} {delta_str:<12}")
    
    logger.info("=" * 60)
    
    # Check prediction agreement
    agreement = np.mean(original_preds == quantized_preds) * 100
    logger.info(f"Prediction agreement: {agreement:.1f}%")


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
        choices=["train", "export", "quantize", "evaluate", "all"],
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
    elif args.command == "evaluate":
        evaluate_quantization()
    else:  # "all" or default
        run_full_pipeline()


if __name__ == "__main__":
    main()
