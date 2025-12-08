import logging
import sys
from typing import List, Dict

# Presidio Imports
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerResult, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# --- KONFIGURATION & LOGGING ---
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PII_Pipeline")

# Konfiguration für interne "Deny List" (Mitarbeiter, Kliniknamen, etc.)
INTERNAL_DENY_LIST = ["Dr. Müller", "Klinik am See", "Projekt Phoenix"]

class PiiPipeline:
    def __init__(self):
        """
        Initialisiert die Pipeline.
        Dies sollte beim Start des Cloud Run Containers 1x passieren (Cold Start).
        """
        logger.info("Initialisiere PII Pipeline und lade Modelle...")
        
        # 1. SpaCy German Model Configuration
        # Wir zwingen Presidio, das 'de_core_news_lg' Modell zu nutzen.
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "de", "model_name": "de_core_news_lg"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        self.nlp_engine = provider.create_engine()
        
        # 2. Analyzer Engine instanziieren
        self.analyzer = AnalyzerEngine(nlp_engine=self.nlp_engine, supported_languages=["de"])
        
        # 3. Custom Recognizers hinzufügen (Die "Ingenieurs-Schicht")
        self._add_custom_recognizers()
        
        # 4. Anonymizer Engine instanziieren
        self.anonymizer = AnonymizerEngine()
        
        logger.info("Pipeline bereit.")

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

        # B. Interne Deny-List (Kontext-Blacklist)
        # Erkennt spezifische interne Namen, egal was der Kontext sagt.
        deny_list_recognizer = PatternRecognizer(
            supported_entity="INTERNAL_SENSITIVE",
            supported_language="de",
            deny_list=INTERNAL_DENY_LIST
        )
        self.analyzer.registry.add_recognizer(deny_list_recognizer)

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
        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"}),
                "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
                "GERMAN_ZIP": OperatorConfig("replace", {"new_value": "<PLZ>"}),
                "INTERNAL_SENSITIVE": OperatorConfig("replace", {"new_value": "<INTERN>"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<TEL>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            }
        )

        return {
            "original_length": len(text),
            "anonymized_text": anonymized_result.text,
            "items_changed": len(results)
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