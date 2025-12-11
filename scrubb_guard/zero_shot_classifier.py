"""
Zero-Shot Text Classification using multilingual DeBERTa model.

Uses MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 for 
German-language zero-shot classification without fine-tuning.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Default model for German zero-shot classification
DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Default labels for mental health safety classification
DEFAULT_LABELS = ["Suizidalität", "Selbstverletzung", "Neutral"]


@dataclass
class ClassificationResult:
    """Result of zero-shot classification."""
    text: str
    labels: List[str]
    scores: List[float]
    multi_label: bool
    
    @property
    def top_label(self) -> str:
        """Return the highest-scoring label."""
        return self.labels[0] if self.labels else ""
    
    @property
    def top_score(self) -> float:
        """Return the highest score."""
        return self.scores[0] if self.scores else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "labels": self.labels,
            "scores": self.scores,
            "top_label": self.top_label,
            "top_score": self.top_score,
            "multi_label": self.multi_label,
        }


class ZeroShotClassifier:
    """
    Zero-shot text classifier using multilingual NLI model.
    
    Supports arbitrary label sets without fine-tuning. Particularly suited
    for German-language mental health and safety classification.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        """
        Initialize the zero-shot classifier.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self._pipeline = None
        
    @property
    def pipeline(self):
        """Lazy-load the classification pipeline."""
        if self._pipeline is None:
            from transformers import pipeline
            logger.info(f"Loading zero-shot model: {self.model_name}")
            
            kwargs = {"model": self.model_name}
            if self.device:
                kwargs["device"] = self.device
                
            self._pipeline = pipeline("zero-shot-classification", **kwargs)
            logger.info("Model loaded successfully")
        return self._pipeline
    
    def classify(
        self, 
        text: str, 
        labels: Optional[List[str]] = None,
        multi_label: bool = True
    ) -> ClassificationResult:
        """
        Classify text against given labels.
        
        Args:
            text: Input text to classify
            labels: List of candidate labels (uses defaults if None)
            multi_label: If True, each label is scored independently.
                        If False, scores sum to 1 (single-label mode).
        
        Returns:
            ClassificationResult with labels sorted by score (highest first)
        """
        if labels is None:
            labels = DEFAULT_LABELS
            
        if not text.strip():
            return ClassificationResult(
                text=text,
                labels=labels,
                scores=[0.0] * len(labels),
                multi_label=multi_label
            )
        
        output = self.pipeline(text, labels, multi_label=multi_label)
        
        return ClassificationResult(
            text=text,
            labels=output["labels"],
            scores=output["scores"],
            multi_label=multi_label
        )
    
    def classify_batch(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        multi_label: bool = True
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of input texts
            labels: List of candidate labels
            multi_label: Independent label scoring mode
            
        Returns:
            List of ClassificationResult objects
        """
        if labels is None:
            labels = DEFAULT_LABELS
            
        results = []
        for text in texts:
            results.append(self.classify(text, labels, multi_label))
        return results


# --- ENTRY POINT ---
if __name__ == "__main__":
    import sys
    
    # Demo usage
    classifier = ZeroShotClassifier()
    
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Es hat alles keinen Sinn mehr, ich will einfach nur noch schlafen und nie wieder aufwachen."
    
    labels = ["Suizidalität", "Selbstverletzung", "Neutral", "Depressive Stimmung"]
    
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    print(f"Labels: {labels}")
    print(f"{'='*60}\n")
    
    result = classifier.classify(text, labels)
    
    print("Results (sorted by score):")
    for label, score in zip(result.labels, result.scores):
        bar = "█" * int(score * 30)
        print(f"  {label:20s} {score:6.1%} {bar}")
    
    print(f"\n→ Top prediction: {result.top_label} ({result.top_score:.1%})")

