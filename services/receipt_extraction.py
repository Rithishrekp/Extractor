import re

def extract_receipt_fields(text: str) -> dict:
    out = {}

    # Amount (Rs / ₹ / decimal)
    if amt := re.findall(r"(rs\.?\s?\d+|\₹\s?\d+|\d+\.\d{2})", text.lower()):
        out["AMOUNT"] = amt

    # Phone number
    if phone := re.findall(r"\b[6-9]\d{9}\b", text):
        out["PHONE"] = phone

    # PIN code (India)
    if pin := re.findall(r"\b\d{6}\b", text):
        out["PINCODE"] = pin

    # Reference / Receipt numbers
    if ref := re.findall(r"\b\d{10,16}\b", text):
        out["REFERENCE_ID"] = ref

    return out
