"""
Scrubb and Guard - German PII Scrubber

A lightweight, on-device PII (Personally Identifiable Information) scrubbing tool
for German text. Designed for mobile deployment with regex patterns and deny lists.
"""

from scrubb_guard.pii_scrubber import GermanPIIScrubber, REGEX_PATTERNS
from scrubb_guard.zero_shot_classifier import ZeroShotClassifier, DEFAULT_LABELS

__all__ = ["GermanPIIScrubber", "REGEX_PATTERNS", "ZeroShotClassifier", "DEFAULT_LABELS"]
__version__ = "0.1.0"

