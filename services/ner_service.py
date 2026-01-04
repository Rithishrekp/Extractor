import re, spacy
from utils.cleaners import clean_entities

nlp = spacy.load("en_core_web_sm")

PHONE_REGEX = r"(?:\+91[\s-]?)?[6-9]\d{9}\b"
EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PAN_REGEX = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
AADHAAR_REGEX = r"\b\d{4}\s\d{4}\s\d{4}\b"

def extract_keywords(text: str) -> dict:
    out = {}

    # 1️⃣ spaCy NER (MAIN & ONLY)
    doc = nlp(text)
    for ent in doc.ents:
        out.setdefault(ent.label_, []).append(ent.text)

    # 2️⃣ Regex (highest priority)
    if phones := re.findall(PHONE_REGEX, text):
        out["PHONE"] = phones

    if emails := re.findall(EMAIL_REGEX, text):
        out["EMAIL"] = emails

    if pan := re.findall(PAN_REGEX, text):
        out["PAN"] = pan

    if aadhaar := re.findall(AADHAAR_REGEX, text):
        out["AADHAAR"] = aadhaar

    # 3️⃣ Final cleanup
    return clean_entities(out)
