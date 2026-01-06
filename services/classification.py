import re

def detect_document_field(text: str, keywords: dict) -> str:
    """
    Decide document type using BOTH raw text + extracted keywords
    Uses strict priority-based rules to avoid misclassification
    """

    # -------------------------------
    # Build searchable blob
    # -------------------------------
    blob_parts = []
    if keywords:
        for _, vals in keywords.items():
            if isinstance(vals, list):
                blob_parts.extend(vals)

    blob = (text + " " + " ".join(blob_parts)).lower()

    # -------------------------------
    # BLOCKLIST (prevents receipt misfire)
    # -------------------------------
    certificate_blocklist = [
        "certificate",
        "certified that",
        "appointment",
        "employed as",
        "salary certificate",
        "authorized signatory",
        "office order",
        "government of",
        "ministry of",
        "district collector",
        "education certificate"
    ]

    # -------------------------------
    # PATTERN DEFINITIONS
    # -------------------------------

    # 🔴 RECEIPT (STRICT – lowest priority)
    receipt_patterns = [
        r"\breceipt\b",
        r"\bfee receipt\b",
        r"\btax invoice\b",
        r"\bcash receipt\b",
        r"\bpayment received\b",
        r"\bchallan\b",
        r"\btransaction id\b"
    ]

    # 🟣 FINANCIAL / SALARY
    financial_patterns = [
        r"\bsalary\b",
        r"\bnet salary\b",
        r"\bgross salary\b",
        r"\bpayslip\b",
        r"\baccount\b",
        r"\bifsc\b",
        r"\bbank\b",
        r"\binvoice\b",
        r"\bloan\b"
    ]

    # 🟢 EDUCATION
    education_patterns = [
        r"\bdegree\b",
        r"\buniversity\b",
        r"\binstitute\b",
        r"\bcollege\b",
        r"\bschool\b",
        r"\bstudent\b",
        r"\bexam\b",
        r"\bmarks\b",
        r"\bsemester\b",
        r"\bpassed\b",
        r"\bcompleted\b",
        r"\beducation\b"
    ]

    # 🔵 GOVERNMENT
    govt_patterns = [
        r"\bgovernment\b",
        r"\bministry\b",
        r"\bdepartment\b",
        r"\bcollector\b",
        r"\bsecretary\b",
        r"\bgazette\b",
        r"\boffice order\b",
        r"\bnotification\b",
        r"\bcircular\b"
    ]

    # 🟠 HEALTH
    health_patterns = [
        r"\bhospital\b",
        r"\bpatient\b",
        r"\bdoctor\b",
        r"\bprescription\b",
        r"\bmedical\b",
        r"\btreatment\b",
        r"\bdiagnosis\b"
    ]

    # 🟡 IDENTITY
    identity_patterns = [
        r"\baadhaar\b",
        r"\bpassport\b",
        r"\bvoter\b",
        r"\blicense\b",
        r"\bid card\b",
        r"\bpan\b",
        r"\bdriving\b"
    ]

    # -------------------------------
    # Helper functions
    # -------------------------------
    def has_any(patterns):
        return any(re.search(p, blob) for p in patterns)

    def has_blocklist():
        return any(b in blob for b in certificate_blocklist)

    # -------------------------------
    # PRIORITY ORDER (VERY IMPORTANT)
    # -------------------------------

    # 1️⃣ Education documents
    if has_any(education_patterns):
        return "Education"

    # 2️⃣ Government documents
    if has_any(govt_patterns):
        return "Government"

    # 3️⃣ Financial / Salary documents
    if has_any(financial_patterns):
        return "Financial"

    # 4️⃣ Receipt (ONLY if not certificate/salary)
    if has_any(receipt_patterns) and not has_blocklist():
        return "Receipt"

    # 5️⃣ Health
    if has_any(health_patterns):
        return "Health"

    # 6️⃣ Identity
    if has_any(identity_patterns):
        return "Identity"

    # 7️⃣ Default
    return "General"
