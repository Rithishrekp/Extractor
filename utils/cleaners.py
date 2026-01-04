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
    # ---------- 1. Remove junk labels ----------
    REMOVE_LABELS = {"MISC", "WORK_OF_ART", "CARDINAL", "ORDINAL", "PERCENT"}
    for lbl in REMOVE_LABELS:
        out.pop(lbl, None)

    # ---------- 2. PERSON cleanup ----------
    PERSON_BLOCKLIST = {
        "tamil nadu", "chennai", "salary slip", "invoice",
        "statement", "account", "bank", "department",
        "india", "government", "java development"
    }

    if "PERSON" in out:
        out["PERSON"] = [
            p for p in out["PERSON"]
            if isinstance(p, str)
            and len(p.split()) <= 3
            and not any(char.isdigit() for char in p)
            and p.lower() not in PERSON_BLOCKLIST
        ]
        if not out["PERSON"]:
            out.pop("PERSON")

    # ---------- 3. ORG cleanup ----------
    ORG_BLOCKLIST = {
        "pan", "tds", "hra", "account", "bank account",
        "salary", "slip", "statement", "form",
        "ika s", "aplp"
    }

    if "ORG" in out:
        cleaned_orgs = []
        out["ORG"] = sorted(
            set(out["ORG"]),
            key=len,
            reverse=True
        )
        for o in out["ORG"]:
            if not isinstance(o, str):
                continue

            o_clean = o.strip()

            if len(o_clean) < 4 or o_clean.isdigit():
                continue

            if o_clean.lower() in ORG_BLOCKLIST:
                continue

            if not any(
                k in o_clean.lower()
                for k in ["pvt", "ltd", "limited", "technologies", "company", "corp", "bank", "institute"]
            ):
            
                continue

            cleaned_orgs.append(o_clean)

        cleaned_orgs = list(dict.fromkeys(cleaned_orgs))
        if cleaned_orgs:
            out["ORG"] = cleaned_orgs
        else:
            out.pop("ORG")

    # ---------- 4. Priority enforcement ----------
    PRIORITY = ["EMAIL", "PHONE", "PAN", "AADHAAR", "MONEY", "DATE"]
    for high in PRIORITY:
        if high in out:
            for low in ["PERSON", "ORG", "LOC", "GPE"]:
                if low in out:
                    out[low] = [x for x in out[low] if x not in out[high]]
                    if not out[low]:
                        out.pop(low)

    # ---------- 5. Deduplicate ----------
    for k in list(out.keys()):
        out[k] = clean_list(out[k])
        if not out[k]:
            out.pop(k)

    return out
