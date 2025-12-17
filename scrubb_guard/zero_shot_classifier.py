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

# Default labels for mental health safety classification (German)
DEFAULT_LABELS = ["Suizidalität", "Selbstverletzung", "Neutral"]

# English labels often work better, even for German text (NLI models trained mostly on English)
DEFAULT_LABELS_EN = ["suicide", "self-harm", "neutral"]

# Mapping from English to German labels for display
DEFAULT_LABEL_MAP = {
    "suicide": "Suizidalität",
    "self-harm": "Selbstverletzung",
    "neutral": "Neutral",
    "depression": "Depressive Stimmung",
    "anxiety": "Angst",
    "safe": "Sicher",
    "hate speech": "Hassrede",
    "insult": "Beleidigung",
    "violence": "Gewaltandrohung",
    "sexual content": "Sexueller Inhalt",
    "positive": "Positiv",
    "negative": "Negativ",
}

# Default hypothesis templates
DEFAULT_TEMPLATE_EN = "This text is about {}."
DEFAULT_TEMPLATE_DE = "In diesem Text geht es um {}."


@dataclass
class ClassificationResult:
    """Result of zero-shot classification."""
    text: str
    labels: List[str]
    scores: List[float]
    multi_label: bool
    display_labels: Optional[List[str]] = None  # Mapped labels for display
    
    @property
    def top_label(self) -> str:
        """Return the highest-scoring label (display version if available)."""
        labels = self.display_labels or self.labels
        return labels[0] if labels else ""
    
    @property
    def top_score(self) -> float:
        """Return the highest score."""
        return self.scores[0] if self.scores else 0.0
    
    def get_labels_for_display(self) -> List[str]:
        """Return display labels if available, otherwise raw labels."""
        return self.display_labels or self.labels
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "labels": self.labels,
            "display_labels": self.display_labels,
            "scores": self.scores,
            "top_label": self.top_label,
            "top_score": self.top_score,
            "multi_label": self.multi_label,
        }


@dataclass
class RelevanceResult:
    """Result of question-answer relevance check."""
    question: str
    answer: str
    is_relevant: bool
    relevance_score: float
    irrelevance_score: float
    verdict: str  # "relevant" or "off-topic"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "question": self.question,
            "answer": self.answer,
            "combined_input": f"Frage: {self.question} Antwort: {self.answer}",
            "is_relevant": self.is_relevant,
            "relevance_score": self.relevance_score,
            "irrelevance_score": self.irrelevance_score,
            "verdict": self.verdict,
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
        multi_label: bool = True,
        hypothesis_template: Optional[str] = None,
        label_mapping: Optional[Dict[str, str]] = None
    ) -> ClassificationResult:
        """
        Classify text against given labels.
        
        Args:
            text: Input text to classify
            labels: List of candidate labels (uses defaults if None)
            multi_label: If True, each label is scored independently.
                        If False, scores sum to 1 (single-label mode).
            hypothesis_template: Custom template for NLI hypothesis, e.g. "This text is about {}."
                                Using English templates with English labels often improves results.
            label_mapping: Dict mapping labels to display names, e.g. {"suicide": "Suizidalität"}
        
        Returns:
            ClassificationResult with labels sorted by score (highest first)
        """
        if labels is None:
            labels = DEFAULT_LABELS
            
        if not text.strip():
            display_labels = [label_mapping.get(l, l) for l in labels] if label_mapping else None
            return ClassificationResult(
                text=text,
                labels=labels,
                scores=[0.0] * len(labels),
                multi_label=multi_label,
                display_labels=display_labels
            )
        
        # Build pipeline kwargs
        kwargs = {"multi_label": multi_label}
        if hypothesis_template:
            kwargs["hypothesis_template"] = hypothesis_template
        
        output = self.pipeline(text, labels, **kwargs)
        
        # Map labels to display names if mapping provided
        display_labels = None
        if label_mapping:
            display_labels = [label_mapping.get(l, l) for l in output["labels"]]
        
        return ClassificationResult(
            text=text,
            labels=output["labels"],
            scores=output["scores"],
            multi_label=multi_label,
            display_labels=display_labels
        )
    
    def classify_batch(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        multi_label: bool = True,
        hypothesis_template: Optional[str] = None,
        label_mapping: Optional[Dict[str, str]] = None
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of input texts
            labels: List of candidate labels
            multi_label: Independent label scoring mode
            hypothesis_template: Custom NLI hypothesis template
            label_mapping: Dict mapping labels to display names
            
        Returns:
            List of ClassificationResult objects
        """
        if labels is None:
            labels = DEFAULT_LABELS
            
        results = []
        for text in texts:
            results.append(self.classify(
                text, labels, multi_label, 
                hypothesis_template, label_mapping
            ))
        return results
    
    def check_relevance(
        self,
        question: str,
        answer: str,
        relevance_threshold: float = 0.5,
        labels: Optional[List[str]] = None,
        hypothesis_template: str = "Diese Antwort ist {}."
    ) -> RelevanceResult:
        """
        Check if an answer is relevant to a question.
        
        Uses zero-shot classification to detect off-topic responses,
        trolling attempts, or nonsensical "word salad" answers.
        
        Args:
            question: The question that was asked
            answer: The answer to evaluate
            relevance_threshold: Score threshold for considering answer relevant (default 0.5)
            labels: Custom labels (default: German relevance labels)
            hypothesis_template: NLI hypothesis template
        
        Returns:
            RelevanceResult with relevance verdict and scores
        """
        if labels is None:
            labels = [
                "eine sinnvolle, relevante Antwort",
                "Themenverfehlung oder Unsinn"
            ]
        
        # Combine question and answer for context
        input_text = f"Frage: {question} Antwort: {answer}"
        
        if not answer.strip():
            return RelevanceResult(
                question=question,
                answer=answer,
                is_relevant=False,
                relevance_score=0.0,
                irrelevance_score=1.0,
                verdict="off-topic"
            )
        
        output = self.pipeline(
            input_text, 
            labels, 
            multi_label=False,
            hypothesis_template=hypothesis_template
        )
        
        # Find scores for each category
        relevance_score = 0.0
        irrelevance_score = 0.0
        
        for label, score in zip(output["labels"], output["scores"]):
            if "sinnvoll" in label.lower() or "relevant" in label.lower():
                relevance_score = score
            else:
                irrelevance_score = score
        
        is_relevant = relevance_score >= relevance_threshold
        
        return RelevanceResult(
            question=question,
            answer=answer,
            is_relevant=is_relevant,
            relevance_score=relevance_score,
            irrelevance_score=irrelevance_score,
            verdict="relevant" if is_relevant else "off-topic"
        )


# --- ENTRY POINT ---
if __name__ == "__main__":
    import sys
    
    # Demo usage
    classifier = ZeroShotClassifier()
    
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Ich will sterben"
    
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    print(f"{'='*60}\n")
    
    # Compare German vs English labels
    print("--- German Labels (no template) ---")
    labels_de = ["Suizidalität", "Selbstverletzung", "Neutral"]
    result_de = classifier.classify(text, labels_de, multi_label=False)
    for label, score in zip(result_de.labels, result_de.scores):
        bar = "█" * int(score * 30)
        print(f"  {label:20s} {score:6.1%} {bar}")
    
    print("\n--- English Labels + Template (recommended) ---")
    labels_en = ["suicide", "self-harm", "neutral"]
    result_en = classifier.classify(
        text, 
        labels_en, 
        multi_label=False,
        hypothesis_template=DEFAULT_TEMPLATE_EN,
        label_mapping=DEFAULT_LABEL_MAP
    )
    for label, score in zip(result_en.get_labels_for_display(), result_en.scores):
        bar = "█" * int(score * 30)
        print(f"  {label:20s} {score:6.1%} {bar}")
    
    print(f"\n→ Top prediction: {result_en.top_label} ({result_en.top_score:.1%})")

