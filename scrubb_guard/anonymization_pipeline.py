import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Presidio Imports
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerResult, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# --- KONFIGURATION & LOGGING ---
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PII_Pipeline")

# SpaCy model name (installed via requirements.txt)
SPACY_MODEL = "de_core_news_lg"

# Path to deny list config file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DENY_LIST_PATH = PROJECT_ROOT / "config" / "deny_list.txt"


def load_deny_list(path: Optional[Path] = None) -> List[str]:
    """Load deny list from config file. Returns empty list if file doesn't exist."""
    file_path = path or DENY_LIST_PATH
    if not file_path.exists():
        logger.warning(f"Deny list file not found: {file_path}")
        return []
    try:
        content = file_path.read_text(encoding="utf-8")
        entries = [line.strip() for line in content.strip().split("\n") if line.strip()]
        logger.info(f"Loaded {len(entries)} entries from deny list")
        return entries
    except Exception as e:
        logger.error(f"Error loading deny list: {e}")
        return []


def save_deny_list(entries: List[str], path: Optional[Path] = None) -> bool:
    """Save deny list to config file. Returns True on success."""
    file_path = path or DENY_LIST_PATH
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        logger.info(f"Saved {len(entries)} entries to deny list")
        return True
    except Exception as e:
        logger.error(f"Error saving deny list: {e}")
        return False

class PiiPipeline:
    def __init__(self, deny_list: Optional[List[str]] = None):
        """
        Initialisiert die Pipeline.
        Dies sollte beim Start des Cloud Run Containers 1x passieren (Cold Start).
        
        Args:
            deny_list: Optional custom deny list. If None, loads from config file.
        """
        logger.info("Initialisiere PII Pipeline und lade Modelle...")
        
        # Load deny list from file or use provided list
        self.deny_list = deny_list if deny_list is not None else load_deny_list()
        
        # 1. SpaCy German Model Configuration
        # Wir zwingen Presidio, das 'de_core_news_lg' Modell zu nutzen.
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "de", "model_name": SPACY_MODEL}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        self.nlp_engine = provider.create_engine()
        
        # 2. Analyzer Engine instanziieren
        self.analyzer = AnalyzerEngine(nlp_engine=self.nlp_engine, supported_languages=["de"])
        
        # 3. Custom Recognizers hinzufügen (Die "Ingenieurs-Schicht")
        self._add_custom_recognizers()
        
        # 4. Anonymizer Engine instanziieren
        self.anonymizer = AnonymizerEngine()
        
        logger.info(f"Pipeline bereit. Deny list: {len(self.deny_list)} entries.")

    def _add_custom_recognizers(self):
        """
        Fügt deterministische Regeln (Regex & Deny Lists) hinzu.
        """
        # A. Deutsche Postleitzahlen (PLZ) - 5 Ziffern
        # SpaCy erkennt Orte gut, aber PLZ sind kritische Identifier.
        plz_pattern = Pattern(name="plz_pattern", regex=r"\b\d{5}\b", score=1.0)
        plz_recognizer = PatternRecognizer(
            supported_entity="GERMAN_ZIP",
            supported_language="de",
            patterns=[plz_pattern]
        )
        self.analyzer.registry.add_recognizer(plz_recognizer)

        # B. German titles followed by names (Dr., Prof., Herr, Frau, etc.)
        # Catches names that SpaCy NER might miss
        title_patterns = [
            # Dr./Prof. + Name (e.g., "Dr. Müller", "Prof. Schmidt")
            Pattern(name="dr_title", regex=r"\b(?:Dr\.|Prof\.)\s+[A-ZÄÖÜ][a-zäöüß]+", score=0.9),
            # Herr/Frau + Name (e.g., "Herr Meier", "Frau Weber")
            Pattern(name="herr_frau", regex=r"\b(?:Herr|Frau)\s+[A-ZÄÖÜ][a-zäöüß]+", score=0.85),
        ]
        title_recognizer = PatternRecognizer(
            supported_entity="PERSON",
            supported_language="de",
            patterns=title_patterns
        )
        self.analyzer.registry.add_recognizer(title_recognizer)

        # C. Interne Deny-List (Kontext-Blacklist)
        # Erkennt spezifische interne Namen, egal was der Kontext sagt.
        self._update_deny_list_recognizer()

    def _update_deny_list_recognizer(self):
        """Update or create the deny list recognizer with current deny_list."""
        # Remove existing deny list recognizer if present
        existing = [r for r in self.analyzer.registry.recognizers 
                   if getattr(r, 'supported_entities', None) == ["INTERNAL_SENSITIVE"]]
        for r in existing:
            self.analyzer.registry.remove_recognizer(r)
        
        # Add new recognizer if deny list has entries
        if self.deny_list:
            deny_list_recognizer = PatternRecognizer(
                supported_entity="INTERNAL_SENSITIVE",
                supported_language="de",
                deny_list=self.deny_list
            )
            self.analyzer.registry.add_recognizer(deny_list_recognizer)

    def reload_deny_list(self, entries: Optional[List[str]] = None) -> List[str]:
        """Reload deny list and update the recognizer.
        
        Args:
            entries: If provided, use these entries directly. 
                     If None, reload from config file.
        """
        if entries is not None:
            self.deny_list = entries
        else:
            self.deny_list = load_deny_list()
        self._update_deny_list_recognizer()
        return self.deny_list

    def process(self, text: str) -> Dict:
        """
        Hauptfunktion: Nimmt Text, gibt anonymisiertes Resultat zurück.
        """
        if not text:
            return {"anonymized_text": "", "meta": []}

        # Schritt 1: Analyse (Wo ist PII?)
        # Entities: PERSON, LOCATION, ORGANIZATION, PHONE_NUMBER, EMAIL, etc. werden automatisch erkannt.
        # + unsere Custom Entities: GERMAN_ZIP, INTERNAL_SENSITIVE
        results = self.analyzer.analyze(
            text=text,
            language="de",
            score_threshold=0.4 # Konservativ: Lieber zu viel erkennen als zu wenig
        )

        # Audit-Log (Nur Typen, keine Inhalte!)
        self._log_findings(results)

        # Schritt 2: Anonymisierung (Ersetzen durch Placeholder)
        # Wir nutzen "Replace", um saubere Tokens für das LLM zu generieren: <PERSON>, <LOCATION>
        # SpaCy German model uses: PER (person), LOC (location), GPE (geo-political), ORG, MISC
        # Presidio maps some of these but we need to ensure all location types are covered
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"}),
                # Person entities
                "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                "PER": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                # Location entities (SpaCy uses LOC for non-GPE locations, GPE for cities/countries)
                "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
                "LOC": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
                "GPE": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
                # Organization
                "ORGANIZATION": OperatorConfig("replace", {"new_value": "<ORG>"}),
                "ORG": OperatorConfig("replace", {"new_value": "<ORG>"}),
                # Miscellaneous (often places, events, etc.)
                "MISC": OperatorConfig("replace", {"new_value": "<MISC>"}),
                # Custom entities
                "GERMAN_ZIP": OperatorConfig("replace", {"new_value": "<PLZ>"}),
                "INTERNAL_SENSITIVE": OperatorConfig("replace", {"new_value": "<INTERN>"}),
                # Contact info
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TEL>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                # Additional common entity types
                "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE>"}),
                "NRP": OperatorConfig("replace", {"new_value": "<NRP>"}),  # Nationalities, religious, political groups
            }
        )

        # Collect entity details for debugging/display
        entities_found = []
        for r in results:
            entities_found.append({
                "entity_type": r.entity_type,
                "text": text[r.start:r.end],
                "score": r.score
            })
        
        return {
            "original_length": len(text),
            "anonymized_text": anonymized_result.text,
            "items_changed": len(results),
            "entities": entities_found
        }

    def _log_findings(self, results: List[RecognizerResult]):
        """
        Schreibt in das Log, WAS gefunden wurde (nicht WER).
        Wichtig für regulatorische Audits.
        """
        counts = {}
        for r in results:
            counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
        if counts:
            logger.info(f"PII gefunden: {counts}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Instanziierung der Pipeline
    pipeline = PiiPipeline()
    
    # Check if text was provided as command line argument
    if len(sys.argv) > 1:
        # Join all arguments as input text
        input_text = " ".join(sys.argv[1:])
        result = pipeline.process(input_text)
        print(f"\nInput:  {input_text}")
        print(f"Output: {result['anonymized_text']}")
        print(f"Stats:  {result['items_changed']} PIIs maskiert")
    else:
        # Fallback: Test-Szenarien
        test_inputs = [
            "Ich heiße Peter Müller und wohne in 12345 Berlin.",
            "Mein Arzt ist Dr. Müller in der Klinik am See.",
            "Ruf mich an unter 0176-12345678 oder mail mir: peter@example.com",
            "Wir treffen uns in Essen zum Essen.",  # Der Härtetest (Ort vs Verb)
            "Ich komme aus Oer-Erkenschwick."       # Kleinstadt Test
        ]

        print(f"{' BEISPIEL-TESTS ':*^40}")
        for txt in test_inputs:
            result = pipeline.process(txt)
            print(f"\nInput:  {txt}")
            print(f"Output: {result['anonymized_text']}")
            print(f"Stats:  {result['items_changed']} PIIs maskiert")