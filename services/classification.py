import re

def detect_document_field(text: str, keywords: dict) -> str:
    blob_parts = []
    for k, vals in keywords.items():
        if isinstance(vals, list):
            blob_parts.extend(vals)

    blob = (text + " " + " ".join(blob_parts)).lower()

    financial_patterns = [
        r"\bsalary\b", r"\baccount\b", r"\bifsc\b", r"\bgst\b",
        r"\bamount\b", r"\bfunding\b", r"\binvoice\b",
        r"\btransaction\b", r"\bloan\b", r"\bbank\b"
    ]

    education_patterns = [
        r"\bdegree\b", r"\buniversity\b", r"\binstitute\b",
        r"\bschool\b", r"\bstudent\b", r"\bexam\b",
        r"\bcertificate\b", r"\bmarks\b", r"\bsemester\b"
    ]

    health_patterns = [
        r"\bhospital\b", r"\bpatient\b", r"\bdoctor\b",
        r"\bprescription\b", r"\bmedical\b", r"\bhealth\b",
        r"\btreatment\b", r"\bdiagnosis\b"
    ]

    identity_patterns = [
        r"\baadhaar\b", r"\bpassport\b", r"\bvoter\b",
        r"\blicense\b", r"\blicence\b", r"\bid card\b",
        r"\bpan\b", r"\bdriving\b"
    ]

    govt_patterns = [
        r"\bministry\b", r"\bgovernment\b", r"\bdepartment\b",
        r"\bsecretary\b", r"\bgazette\b", r"\border\b",
        r"\bnotification\b", r"\bscheme\b",
        r"\boffice memorandum\b", r"\bcircular\b"
    ]

    def has_any(patterns):
        return any(re.search(p, blob) for p in patterns)

    # Priority order (same as your code)
    if has_any(education_patterns):
        return "Education"
    if has_any(govt_patterns):
        return "Government"
    if has_any(financial_patterns):
        return "Financial"
    if has_any(health_patterns):
        return "Health"
    if has_any(identity_patterns):
        return "Identity"

    return "General"
