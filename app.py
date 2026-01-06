from flask import (
    Flask, request, render_template,
    redirect, url_for, session, send_from_directory
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from services.text_extraction import extract_text
from services.ner_service import extract_keywords
from services.summarization import extractive_summary
from services.translation import translate, translate_to_english
from services.classification import detect_document_field
from services.language_detection import detect_language
from services.receipt_extraction import extract_receipt_fields
from services.tts import speak

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "super_secret_key"

# ================= DATABASE =================
DB_PATH = "instance/users.db"

def init_db():
    os.makedirs("instance", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= HELPERS =================
def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ================= AUTH =================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (email, password)
            )
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except:
            return "User already exists"

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            session["user"] = email
            return redirect(url_for("index"))
        return "Invalid email or password"

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ================= FILE SERVE =================
@app.route("/myfile/<filename>")
def myfile(filename):
    if "user" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(UPLOAD_FOLDER, session["user"])
    return send_from_directory(user_folder, filename)

# ================= MAIN APP =================
@app.route("/", methods=["GET", "POST"])
def index():
    # 🔐 Login required
    if "user" not in session:
        return redirect(url_for("login"))

    user_email = session["user"]
    user_folder = os.path.join(UPLOAD_FOLDER, user_email)
    os.makedirs(user_folder, exist_ok=True)

    uploaded_files = os.listdir(user_folder)

    # Defaults
    detected_field = None
    extracted = None
    receipt_fields = None
    translated_keywords = None
    ex_summary = None
    translated_ex_summary = None
    full_text = None
    translated_text = None
    audio_combined = None
    lang = "en"

    if request.method == "POST":
        file = request.files.get("document")
        lang = request.form.get("lang", "en")

        if file and allowed_file(file.filename):
            name = secure_filename(file.filename)
            path = os.path.join(user_folder, name)
            file.save(path)

            # 1️⃣ Text Extraction
            text = extract_text(path)
            full_text = text

            # 2️⃣ Language detection → English
            detected_lang = detect_language(text)
            english_text = (
                translate_to_english(text)
                if detected_lang != "en"
                else text
            )

            # 3️⃣ Document classification (TEXT ONLY)
            detected_field = detect_document_field(english_text, {})

            # ================= RECEIPT =================
            if detected_field == "Receipt":
                receipt_fields = extract_receipt_fields(english_text)

                # Do NOT run NER / Summary / Audio
                extracted = None
                translated_keywords = None
                ex_summary = None
                translated_ex_summary = None
                translated_text = None
                audio_combined = None

            # ================= NORMAL DOCUMENT =================
            else:
                extracted = extract_keywords(english_text)
                ex_summary = extractive_summary(english_text)

                if lang != "en":
                    translated_text = translate(english_text, lang)
                    translated_ex_summary = translate(ex_summary, lang)
                    translated_keywords = {
                        k: [translate(w, lang) for w in v]
                        for k, v in extracted.items()
                    }
                else:
                    translated_keywords = {}

                audio_combined = speak(
                    translated_ex_summary if translated_ex_summary else ex_summary,
                    lang
                )

    return render_template(
        "index.html",
        detected_field=detected_field,
        extracted=extracted,
        receipt_fields=receipt_fields,
        translated_keywords=translated_keywords,
        ex_summary=ex_summary,
        translated_ex_summary=translated_ex_summary,
        full_text=full_text,
        translated_text=translated_text,
        audio_combined=audio_combined,
        target_lang=lang,
        uploaded_files=uploaded_files
    )

if __name__ == "__main__":
    app.run(debug=True)
