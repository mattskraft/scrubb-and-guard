"""
Evaluate zero-shot classifier performance using ROC curve analysis.

Tests the ZeroShotClassifier against the german_safety_dataset.csv
using the hypothesis "The author expresses suicidal intent".
"""

import sys
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
    """Load the German safety dataset."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    print(f"Category distribution:\n{df['category'].value_counts()}")
    return df


def run_classification(
    classifier: ZeroShotClassifier, 
    texts: list[str],
    hypothesis: str = "The author expresses suicidal intent."
) -> list[float]:
    """
    Run zero-shot classification on all texts.
    
    Returns scores for the positive class (suicidal intent).
    """
    labels = ["suicidal intent", "safe content"]
    scores = []
    
    print(f"\nClassifying {len(texts)} samples...")
    print(f"Hypothesis template: 'This text is about {{}}.'")
    print(f"Labels: {labels}")
    
    for text in tqdm(texts, desc="Classifying"):
        result = classifier.classify(
            text,
            labels=labels,
            multi_label=False,
            hypothesis_template="This text is about {}."
        )
        
        # Get score for "suicidal intent" label
        suicidal_idx = result.labels.index("suicidal intent")
        scores.append(result.scores[suicidal_idx])
    
    return scores


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: str):
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
    plt.title('ROC Curve: Zero-Shot Suicidal Intent Detection\n'
              '(german_safety_dataset.csv)', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nROC curve saved to: {output_path}")
    
    return roc_auc, optimal_threshold


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, output_path: str):
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
    plt.title('Precision-Recall Curve: Zero-Shot Suicidal Intent Detection\n'
              '(german_safety_dataset.csv)', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"PR curve saved to: {output_path}")
    
    return ap


def plot_score_distribution(y_true: np.ndarray, y_scores: np.ndarray, output_path: str):
    """Plot score distribution by class."""
    plt.figure(figsize=(10, 6))
    
    safe_scores = y_scores[y_true == 0]
    unsafe_scores = y_scores[y_true == 1]
    
    plt.hist(safe_scores, bins=30, alpha=0.7, label='SAFE', color='#22c55e', edgecolor='white')
    plt.hist(unsafe_scores, bins=30, alpha=0.7, label='UNSAFE', color='#ef4444', edgecolor='white')
    
    plt.xlabel('Suicidal Intent Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Score Distribution by Ground Truth Label', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Score distribution saved to: {output_path}")


def print_classification_report(y_true: np.ndarray, y_scores: np.ndarray, threshold: float):
    """Print classification metrics at given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    
    print(f"\n{'='*60}")
    print(f"Classification Report (threshold = {threshold:.3f})")
    print('='*60)
    print(classification_report(
        y_true, y_pred, 
        target_names=['SAFE', 'UNSAFE'],
        digits=3
    ))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              SAFE  UNSAFE")
    print(f"Actual SAFE    {cm[0,0]:4d}   {cm[0,1]:4d}")
    print(f"       UNSAFE  {cm[1,0]:4d}   {cm[1,1]:4d}")


def analyze_errors(
    df: pd.DataFrame, 
    y_scores: np.ndarray, 
    threshold: float,
    n_examples: int = 5
):
    """Show example misclassifications."""
    df = df.copy()
    df['score'] = y_scores
    df['y_true'] = (df['label'] == 'UNSAFE').astype(int)
    df['y_pred'] = (df['score'] >= threshold).astype(int)
    
    # False positives (SAFE classified as UNSAFE)
    fp = df[(df['y_true'] == 0) & (df['y_pred'] == 1)].sort_values('score', ascending=False)
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVES (SAFE texts classified as UNSAFE): {len(fp)}")
    print('='*60)
    for i, (_, row) in enumerate(fp.head(n_examples).iterrows()):
        print(f"\n[{i+1}] Score: {row['score']:.3f} | Category: {row['category']}")
        print(f"    {row['text'][:200]}...")
    
    # False negatives (UNSAFE classified as SAFE)
    fn = df[(df['y_true'] == 1) & (df['y_pred'] == 0)].sort_values('score', ascending=True)
    print(f"\n{'='*60}")
    print(f"FALSE NEGATIVES (UNSAFE texts classified as SAFE): {len(fn)}")
    print('='*60)
    for i, (_, row) in enumerate(fn.head(n_examples).iterrows()):
        print(f"\n[{i+1}] Score: {row['score']:.3f} | Category: {row['category']}")
        print(f"    {row['text'][:200]}...")


def main():
    # Paths
    data_path = project_root / "data" / "synthetic" / "german_safety_dataset.csv"
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("="*60)
    print("Zero-Shot Classifier ROC Analysis")
    print("="*60)
    df = load_dataset(data_path)
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = ZeroShotClassifier()
    
    # Run classification
    scores = run_classification(classifier, df['text'].tolist())
    
    # Convert to numpy arrays
    y_true = (df['label'] == 'UNSAFE').astype(int).values
    y_scores = np.array(scores)
    
    # Save scores to CSV for later analysis
    results_df = df.copy()
    results_df['suicidal_intent_score'] = y_scores
    results_csv_path = output_dir / "classification_scores.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nScores saved to: {results_csv_path}")
    
    # Plot ROC curve
    roc_auc, optimal_threshold = plot_roc_curve(
        y_true, y_scores, 
        str(output_dir / "roc_curve.png")
    )
    
    # Plot PR curve
    ap = plot_precision_recall_curve(
        y_true, y_scores,
        str(output_dir / "pr_curve.png")
    )
    
    # Plot score distribution
    plot_score_distribution(
        y_true, y_scores,
        str(output_dir / "score_distribution.png")
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"ROC AUC:           {roc_auc:.3f}")
    print(f"Average Precision: {ap:.3f}")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    
    # Classification report at optimal threshold
    print_classification_report(y_true, y_scores, optimal_threshold)
    
    # Also show at 0.5 threshold
    print_classification_report(y_true, y_scores, 0.5)
    
    # Error analysis
    analyze_errors(df, y_scores, optimal_threshold)
    
    print(f"\n{'='*60}")
    print("Analysis complete! Check the 'results' folder for plots.")
    print('='*60)


if __name__ == "__main__":
    main()

