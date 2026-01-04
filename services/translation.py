from deep_translator import GoogleTranslator
from googletrans import Translator

_translation_cache = {}

def translate(text, target_lang):
    if target_lang == "en" or not text:
        return text

    key = (text, target_lang)
    if key in _translation_cache:
        return _translation_cache[key]

    result = GoogleTranslator(source="auto", target=target_lang).translate(text)
    _translation_cache[key] = result
    return result


translator = Translator()
def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        return translator.translate(text, dest="en").text
    except Exception as e:
        print("Translation to English failed:", e)
        return text

def to_english(text: str, src_lang: str) -> str:
    if src_lang == "en":
        return text
    return translator.translate(text, src=src_lang, dest="en").text

