def clean_list(values):
    seen = set()
    out = []
    for v in values:
        if not v:
            continue
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def clean_entities(out: dict) -> dict:
    # ==================================================
    # 1️⃣ REMOVE JUNK / UNWANTED LABELS
    # ==================================================
    REMOVE_LABELS = {"MISC", "WORK_OF_ART", "CARDINAL", "ORDINAL", "PERCENT"}
    for lbl in REMOVE_LABELS:
        out.pop(lbl, None)

    # ==================================================
    # 2️⃣ PERSON CLEANUP (STRICT & FINAL)
    # ==================================================
    PERSON_BLOCKLIST = {
        "tamil nadu", "government", "govt", "district",
        "collector", "office", "certificate",
        "discipline", "competence", "integrity",
        "department", "education", "india", "chennai"
    }

    DEGREE_WORDS = {
        "b.tech", "btech", "b.e", "be",
        "m.tech", "mtech", "m.e", "me",
        "degree", "engineering"
    }

    if "PERSON" in out:
        cleaned_persons = []

        for p in out["PERSON"]:
            if not isinstance(p, str):
                continue

            p_low = p.lower()

            # remove govt words & honorifics
            for bad in ["tamil nadu", "government", "govt", "mr.", "mrs.", "mr", "mrs"]:
                p_low = p_low.replace(bad, "")

            # remove degree words
            for deg in DEGREE_WORDS:
                p_low = p_low.replace(deg, "")

            p_clean = " ".join(p_low.split()).title()

            # keep realistic human names only
            if (
                p_clean
                and 1 <= len(p_clean.split()) <= 3
                and not any(ch.isdigit() for ch in p_clean)
                and p_clean.lower() not in PERSON_BLOCKLIST
            ):
                cleaned_persons.append(p_clean)

        cleaned_persons = list(dict.fromkeys(cleaned_persons))

        if cleaned_persons:
            out["PERSON"] = cleaned_persons
        else:
            out.pop("PERSON")

    # ==================================================
    # 3️⃣ ORG CLEANUP
    # ==================================================
    ORG_BLOCKLIST = {
        "pan", "tds", "hra", "account", "bank account",
        "salary", "slip", "statement", "form",
        "ika s", "aplp"
    }

    if "ORG" in out:
        cleaned_orgs = []

        # Prefer longer org names first
        out["ORG"] = sorted(set(out["ORG"]), key=len, reverse=True)

        for o in out["ORG"]:
            if not isinstance(o, str):
                continue

            o_clean = o.strip()

            # basic validity
            if len(o_clean) < 4 or o_clean.isdigit():
                continue

            # blocklist
            if o_clean.lower() in ORG_BLOCKLIST:
                continue

            # accept only real org keywords
            if not any(
                k in o_clean.lower()
                for k in [
                    "pvt", "ltd", "limited", "technologies",
                    "company", "corp", "bank",
                    "institute", "college", "university"
                ]
            ):
                continue

            cleaned_orgs.append(o_clean)

        cleaned_orgs = list(dict.fromkeys(cleaned_orgs))

        if cleaned_orgs:
            out["ORG"] = cleaned_orgs
        else:
            out.pop("ORG")

    # ==================================================
    # 4️⃣ PRIORITY ENFORCEMENT (REMOVE OVERLAPS)
    # ==================================================
    PRIORITY = ["EMAIL", "PHONE", "PAN", "AADHAAR", "MONEY", "DATE", "PINCODE"]
    for high in PRIORITY:
        if high in out:
            for low in ["PERSON", "ORG", "LOC", "GPE"]:
                if low in out:
                    out[low] = [x for x in out[low] if x not in out[high]]
                    if not out[low]:
                        out.pop(low)

    # ==================================================
    # 5️⃣ FINAL DEDUPLICATION
    # ==================================================
    for k in list(out.keys()):
        out[k] = clean_list(out[k])
        if not out[k]:
            out.pop(k)

    return out
