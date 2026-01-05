"""
Evaluate zero-shot classifier performance using ROC curve analysis.

Tests the ZeroShotClassifier against a dataset using a custom hypothesis.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from scrubb_guard.zero_shot_classifier import ZeroShotClassifier


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    if 'category' in df.columns:
        print(f"Category distribution:\n{df['category'].value_counts()}")
    return df


def run_classification(
    classifier: ZeroShotClassifier, 
    texts: list[str],
    hypothesis: str
) -> list[float]:
    """
    Run zero-shot classification on all texts using a single hypothesis.
    
    Args:
        classifier: ZeroShotClassifier instance
        texts: List of texts to classify
        hypothesis: Full hypothesis string (e.g., "This text is about suicidal intent")
    
    Returns:
        Scores for the hypothesis.
    """
    # Use only the provided hypothesis as a single label
    labels = [hypothesis]
    scores = []
    
    print(f"\nClassifying {len(texts)} samples...")
    print(f"Hypothesis: {hypothesis}")
    
    for text in tqdm(texts, desc="Classifying"):
        result = classifier.classify(
            text,
            labels=labels,
            multi_label=True,  # Use multi_label=True to get independent scores
            hypothesis_template="{}"
        )
        
        # Get score for the hypothesis (should be the only label)
        scores.append(result.scores[0])
    
    return scores


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: str, hypothesis: str, dataset_name: str):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    
    # ROC curve
    plt.plot(fpr, tpr, color='#2563eb', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=1, linestyle='--', 
             label='Random classifier')
    
    # Mark optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], 
                color='#dc2626', s=100, zorder=5,
                label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve: Zero-Shot Classification\n'
              f'Hypothesis: {hypothesis[:50]}{"..." if len(hypothesis) > 50 else ""}\n'
              f'Dataset: {dataset_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nROC curve saved to: {output_path}")
    
    return roc_auc, optimal_threshold


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: str, hypothesis: str, dataset_name: str):
    """Plot and save Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(recall, precision, color='#059669', lw=2,
             label=f'PR curve (AP = {ap:.3f})')
    
    # Baseline (proportion of positives)
    baseline = y_true.mean()
    plt.axhline(y=baseline, color='#94a3b8', lw=1, linestyle='--',
                label=f'Baseline (prevalence = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve: Zero-Shot Classification\n'
              f'Hypothesis: {hypothesis[:50]}{"..." if len(hypothesis) > 50 else ""}\n'
              f'Dataset: {dataset_name}', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"PR curve saved to: {output_path}")
    
    return ap


def plot_score_distribution(y_true: np.ndarray, y_scores: np.ndarray, output_path: str, positive_label: str, negative_label: str):
    """Plot score distribution by class."""
    plt.figure(figsize=(10, 6))
    
    negative_scores = y_scores[y_true == 0]
    positive_scores = y_scores[y_true == 1]
    
    plt.hist(negative_scores, bins=30, alpha=0.7, label=negative_label, color='#22c55e', edgecolor='white')
    plt.hist(positive_scores, bins=30, alpha=0.7, label=positive_label, color='#ef4444', edgecolor='white')
    
    plt.xlabel('Hypothesis Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Score Distribution by Ground Truth Label', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Score distribution saved to: {output_path}")


def print_classification_report(y_true: np.ndarray, y_scores: np.ndarray, threshold: float, positive_label: str, negative_label: str):
    """Print classification metrics at given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    
    print(f"\n{'='*60}")
    print(f"Classification Report (threshold = {threshold:.3f})")
    print('='*60)
    print(classification_report(
        y_true, y_pred, 
        target_names=[negative_label, positive_label],
        digits=3
    ))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              {negative_label:10s}  {positive_label:10s}")
    print(f"Actual {negative_label:6s}  {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       {positive_label:6s}  {cm[1,0]:4d}   {cm[1,1]:4d}")


def analyze_errors(
    df: pd.DataFrame, 
    y_scores: np.ndarray, 
    threshold: float,
    positive_label,
    negative_label,
    n_examples: int = 5
):
    """Show example misclassifications."""
    df = df.copy()
    df['score'] = y_scores
    df['y_pred'] = (df['score'] >= threshold).astype(int)
    
    # Get the actual y_true from the dataframe using the matched label value
    if 'label' not in df.columns:
        print("Warning: Could not determine ground truth labels for error analysis")
        return
    
    y_true = (df['label'] == positive_label).astype(int).values
    df['y_true'] = y_true
    
    # False positives (negative classified as positive)
    fp = df[(df['y_true'] == 0) & (df['y_pred'] == 1)].sort_values('score', ascending=False)
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVES ({str(negative_label)} texts classified as {str(positive_label)}): {len(fp)}")
    print('='*60)
    for i, (_, row) in enumerate(fp.head(n_examples).iterrows()):
        category = row.get('category', 'N/A')
        print(f"\n[{i+1}] Score: {row['score']:.3f} | Category: {category}")
        print(f"    {row['text'][:200]}...")
    
    # False negatives (positive classified as negative)
    fn = df[(df['y_true'] == 1) & (df['y_pred'] == 0)].sort_values('score', ascending=True)
    print(f"\n{'='*60}")
    print(f"FALSE NEGATIVES ({str(positive_label)} texts classified as {str(negative_label)}): {len(fn)}")
    print('='*60)
    for i, (_, row) in enumerate(fn.head(n_examples).iterrows()):
        category = row.get('category', 'N/A')
        print(f"\n[{i+1}] Score: {row['score']:.3f} | Category: {category}")
        print(f"    {row['text'][:200]}...")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot classifier performance using ROC curve analysis"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the input dataset CSV file"
    )
    parser.add_argument(
        "positive_label",
        type=str,
        help="Label of the positive class in the dataset (e.g., '1' or 'UNSAFE')"
    )
    parser.add_argument(
        "hypothesis",
        type=str,
        help="Full hypothesis string (e.g., 'This text is about suicidal intent')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/ in project root)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_path = Path(args.dataset).resolve()
    if not data_path.exists():
        print(f"Error: Dataset file not found: {data_path}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("="*60)
    print("Zero-Shot Classifier ROC Analysis")
    print("="*60)
    df = load_dataset(str(data_path))
    
    # Check required columns
    if 'label' not in df.columns:
        print("Error: Dataset must have a 'label' column")
        sys.exit(1)
    if 'text' not in df.columns:
        print("Error: Dataset must have a 'text' column")
        sys.exit(1)
    
    # Handle label type conversion (string vs numeric)
    # Convert positive_label to match the type in the dataset
    unique_labels = df['label'].unique()
    label_types = set(type(l).__name__ for l in unique_labels)
    
    # Try to match the positive_label with the dataset labels
    # First try exact match, then try type conversion
    positive_label_matched = None
    for label in unique_labels:
        if str(label) == str(args.positive_label) or label == args.positive_label:
            positive_label_matched = label
            break
    
    if positive_label_matched is None:
        print(f"Error: Positive label '{args.positive_label}' not found in dataset.")
        print(f"Available labels: {list(unique_labels)}")
        sys.exit(1)
    
    # Use the matched label from the dataset (preserves original type)
    positive_label_value = positive_label_matched
    
    # Find negative label (the one that's not the positive label)
    negative_labels = [l for l in unique_labels if l != positive_label_value]
    if len(negative_labels) == 0:
        print("Error: Could not determine negative label")
        sys.exit(1)
    negative_label = negative_labels[0]  # Use first non-positive label
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = ZeroShotClassifier()
    
    # Run classification
    scores = run_classification(classifier, df['text'].tolist(), args.hypothesis)
    
    # Convert to numpy arrays (use the matched label value for comparison)
    y_true = (df['label'] == positive_label_value).astype(int).values
    y_scores = np.array(scores)
    
    # Save scores to CSV for later analysis
    results_df = df.copy()
    results_df['hypothesis_score'] = y_scores
    dataset_name = data_path.stem
    results_csv_path = output_dir / f"{dataset_name}_classification_scores.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nScores saved to: {results_csv_path}")
    
    # Plot ROC curve
    roc_auc, optimal_threshold = plot_roc_curve(
        y_true, y_scores, 
        str(output_dir / f"{dataset_name}_roc_curve.png"),
        args.hypothesis,
        dataset_name
    )
    
    # Plot PR curve
    ap = plot_precision_recall_curve(
        y_true, y_scores,
        str(output_dir / f"{dataset_name}_pr_curve.png"),
        args.hypothesis,
        dataset_name
    )
    
    # Plot score distribution
    plot_score_distribution(
        y_true, y_scores,
        str(output_dir / f"{dataset_name}_score_distribution.png"),
        str(positive_label_value),
        str(negative_label)
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Dataset:           {dataset_name}")
    print(f"Hypothesis:        {args.hypothesis}")
    print(f"Positive Label:    {positive_label_value}")
    print(f"Negative Label:    {negative_label}")
    print(f"ROC AUC:           {roc_auc:.3f}")
    print(f"Average Precision: {ap:.3f}")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    
    # Classification report at optimal threshold
    print_classification_report(y_true, y_scores, optimal_threshold, str(positive_label_value), str(negative_label))
    
    # Also show at 0.5 threshold
    print_classification_report(y_true, y_scores, 0.5, str(positive_label_value), str(negative_label))
    
    # Error analysis
    analyze_errors(df, y_scores, optimal_threshold, positive_label_value, negative_label)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Check '{output_dir}' for plots.")
    print('='*60)


if __name__ == "__main__":
    main()

