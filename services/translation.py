from deep_translator import GoogleTranslator
from googletrans import Translator

# -------------------------------------------------
# Normalize language codes (IMPORTANT FIX)
# -------------------------------------------------
LANG_MAP = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW"
}

# -------------------------------------------------
# Cache to avoid repeated API calls
# -------------------------------------------------
_translation_cache = {}

# -------------------------------------------------
# Translate English → Target Language
# (Used at FINAL OUTPUT stage only)
# -------------------------------------------------
def translate(text: str, target_lang: str) -> str:
    if not text or target_lang == "en":
        return text

    # Normalize language code
    target_lang = LANG_MAP.get(target_lang.lower(), target_lang)

    key = (text, target_lang)
    if key in _translation_cache:
        return _translation_cache[key]

    try:
        result = GoogleTranslator(
            source="auto",
            target=target_lang
        ).translate(text)

        _translation_cache[key] = result
        return result

    except Exception as e:
        print("Translation failed:", e)
        return text


# -------------------------------------------------
# Translate ANY language → English (MAIN PIPELINE)
# -------------------------------------------------
translator = Translator()

def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        return translator.translate(text, dest="en").text
    except Exception as e:
        print("Translation to English failed:", e)
        return text


# -------------------------------------------------
# Explicit source language → English (OPTIONAL)
# -------------------------------------------------
def to_english(text: str, src_lang: str) -> str:
    if not text or src_lang == "en":
        return text
    try:
        return translator.translate(
            text, src=src_lang, dest="en"
        ).text
    except Exception as e:
        print("Explicit translation failed:", e)
        return text
