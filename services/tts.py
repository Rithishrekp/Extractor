import os, uuid
from gtts import gTTS
from config import STATIC_AUDIO

def speak(text, lang="en"):
    if not text.strip():
        return None

    os.makedirs(STATIC_AUDIO, exist_ok=True)
    fname = f"{uuid.uuid4().hex}.mp3"
    path = os.path.join(STATIC_AUDIO, fname)

    gTTS(text=text, lang=lang).save(path)
    return f"audio/{fname}"
