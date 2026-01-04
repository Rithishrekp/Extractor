from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from services.text_extraction import extract_text
from services.ner_service import extract_keywords
from services.summarization import extractive_summary
from services.translation import translate
from services.classification import detect_document_field
from services.language_detection import detect_language
from services.translation import translate_to_english


from services.tts import speak

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():

    # Defaults for GET
    detected_field = None
    extracted = None
    translated_keywords = None
    ex_summary = None
    translated_ex_summary = None
    full_text = None
    translated_text = None
    audio_combined = None
    lang = "en"

    if request.method == "POST":
        print("POST request received")

        file = request.files.get("document")
        lang = request.form.get("lang", "en")

        if file and allowed_file(file.filename):
            name = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, name)
            file.save(path)

            # 1️⃣ Text extraction
            text = extract_text(path)
            print("TEXT LENGTH:", len(text))

            full_text = text
            detected_lang = detect_language(text)
            if detected_lang != "en":
                english_text = translate_to_english(text)
            else:
                english_text = text
            extracted = extract_keywords(english_text)

            # 3️⃣ Document classification
            detected_field = detect_document_field(english_text, extracted)
            ex_summary = extractive_summary(english_text)


            # 5️⃣ Translation (ONLY once at end)
            if lang != "en":
                translated_text = translate(text, lang)
                translated_ex_summary = translate(ex_summary, lang)

                translated_keywords = {
                    k: [translate(word, lang) for word in v]
                    for k, v in extracted.items()
                }
            else:
                translated_keywords = None

            # 6️⃣ Audio (final output only)
            audio_combined = speak(
                translated_ex_summary if translated_ex_summary else ex_summary,
                lang
            )

            print("Processing completed")

    return render_template(
        "index.html",
        detected_field=detected_field,
        extracted=extracted,
        translated_keywords=translated_keywords,
        ex_summary=ex_summary,
        translated_ex_summary=translated_ex_summary,
        full_text=full_text,
        translated_text=translated_text,
        audio_combined=audio_combined,
        target_lang=lang
    )


if __name__ == "__main__":
    app.run(debug=True)
