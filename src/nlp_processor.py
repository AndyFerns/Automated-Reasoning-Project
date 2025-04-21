# nlp_processor.py
import spacy

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def process_text(self, text: str):
        """Process text and return a spaCy Doc object."""
        return self.nlp(text)
