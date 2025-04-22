import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from typing import Dict, List, Tuple, Any

# Download required NLTK resources
nltk.download('punkt', quiet=True)

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class PIIMasker:
    """Class to handle PII masking in emails"""

    def __init__(self):
        # Regular expressions for different PII types
        self.regex_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_number": r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "dob": r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
            "aadhar_num": r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            "credit_debit_no": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "cvv_no": r'\b(?:CVV|cvv|Cvv)?\s*:?\s*\d{3,4}\b',
            "expiry_no": r'\b(?:exp|EXP|Exp|Expiry|expiry|expiration)\s*:?\s*\d{1,2}[/\-]\d{2,4}\b'
        }

    def _find_full_names(self, text: str) -> List[Dict[str, Any]]:
        """Find full names using SpaCy NER"""
        doc = nlp(text)
        names = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.append({
                    "position": [ent.start_char, ent.end_char],
                    "classification": "full_name",
                    "entity": ent.text
                })

        return names

    def _find_regex_entities(self, text: str) -> List[Dict[str, Any]]:
        """Find entities using regex patterns"""
        entities = []

        for ent_type, pattern in self.regex_patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    "position": [match.start(), match.end()],
                    "classification": ent_type,
                    "entity": match.group()
                })

        return entities

    def mask_pii(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Mask PII in text and return masked text and list of entities

        Args:
            text: Input email text

        Returns:
            tuple: (masked_text, entities_list)
        """
        # Find all entities
        entities = []
        entities.extend(self._find_full_names(text))
        entities.extend(self._find_regex_entities(text))

        # Sort entities by position to handle overlapping entities
        entities = sorted(entities, key=lambda x: x["position"][0], reverse=True)

        # Create a copy of the text for masking
        masked_text = text

        # Replace entities with mask
        for entity in entities:
            start, end = entity["position"]
            entity_type = entity["classification"]
            masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]

        return masked_text, entities

    def unmask_pii(self, masked_text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Restore original text from masked text using entities list

        Args:
            masked_text: Masked email text
            entities: List of entities

        Returns:
            str: Original unmasked text
        """
        # This function is included for completeness but not used in the API
        restored_text = masked_text

        # Sort entities by position in ascending order to restore correctly
        entities = sorted(entities, key=lambda x: x["position"][0])

        # Adjustments needed for position shifts after restoration
        offset = 0

        for entity in entities:
            start, end = entity["position"]
            entity_type = entity["classification"]
            entity_value = entity["entity"]

            # Adjust positions based on previous replacements
            start += offset
            end += offset

            # Find the mask marker and replace it with original value
            mask = f"[{entity_type}]"
            mask_pos = restored_text.find(mask, max(0, start - 20))

            if mask_pos != -1:
                # Replace mask with original value
                restored_text = restored_text[:mask_pos] + entity_value + restored_text[mask_pos + len(mask):]

                # Update offset
                offset += len(entity_value) - len(mask)

        return restored_text


# Utility function for text preprocessing
def preprocess_text(text: str) -> str:
    """Basic text preprocessing for classification"""
    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text