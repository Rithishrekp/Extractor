import re
import spacy
from utils.cleaners import clean_entities

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# ---------------- REGEX PATTERNS ----------------
PHONE_REGEX   = r"(?:\+91[\s-]?)?[6-9]\d{9}\b"
EMAIL_REGEX   = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
PAN_REGEX     = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
AADHAAR_REGEX = r"\b\d{4}\s\d{4}\s\d{4}\b"
PINCODE_REGEX = r"\b[1-9][0-9]{5}\b"
DEGREE_REGEX  = r"\b(B\.?E\.?|B\.?Tech|M\.?E\.?|M\.?Tech|Degree)\b"
AMOUNT_REGEX  = r"rs\.?\s?\d{1,3}(?:,\d{3})*"


def extract_keywords(text: str) -> dict:
    """
    Extract entities from ENGLISH text only
    Uses spaCy + regex + strong cleanup
    """
    out = {}

    # ---------------- 1️⃣ spaCy NER ----------------
    doc = nlp(text)
    for ent in doc.ents:
        out.setdefault(ent.label_, []).append(ent.text)

    # ---------------- 2️⃣ REGEX (HIGH PRIORITY) ----------------
    if phones := re.findall(PHONE_REGEX, text):
        out["PHONE"] = phones

    if emails := re.findall(EMAIL_REGEX, text):
        out["EMAIL"] = emails

    if pan := re.findall(PAN_REGEX, text):
        out["PAN"] = pan

    if aadhaar := re.findall(AADHAAR_REGEX, text):
        out["AADHAAR"] = aadhaar

    # ---------------- 3️⃣ DEGREE (IMPORTANT FIX) ----------------
    if degrees := re.findall(DEGREE_REGEX, text, re.IGNORECASE):
        out["DEGREE"] = list(set(degrees))

        # ❌ Remove degree words from PERSON if wrongly added
        if "PERSON" in out:
            out["PERSON"] = [
                p for p in out["PERSON"]
                if not any(d.lower() in p.lower() for d in degrees)
            ]
            if not out["PERSON"]:
                out.pop("PERSON")

    # ---------------- 4️⃣ PINCODE ----------------
    if pincodes := re.findall(PINCODE_REGEX, text):
        out["PINCODE"] = pincodes

        # ❌ Remove PINCODE wrongly classified as DATE
        if "DATE" in out:
            out["DATE"] = [
                d for d in out["DATE"]
                if d not in pincodes
            ]
            if not out["DATE"]:
                out.pop("DATE")

    # ---------------- 5️⃣ AMOUNT (OPTIONAL / RECEIPTS) ----------------
    if amounts := re.findall(AMOUNT_REGEX, text.lower()):
        out["AMOUNT"] = list(set(amounts))

    # ---------------- 6️⃣ FINAL CLEANUP ----------------
    return clean_entities(out)
