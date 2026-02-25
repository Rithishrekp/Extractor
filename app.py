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
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "super_secret_key"

# ================= DATABASE =================
# We read the connection string from Environment Variables (set in Kubernetes)
# If it's not set (like when you are testing locally), it will fall back to using SQLite!
# A PostgreSQL URI looks like: postgresql://username:password@localhost:5432/my_database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.Text, nullable=False)

def init_db():
    with app.app_context():
        if app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite'):
            os.makedirs("instance", exist_ok=True)
        db.create_all()

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

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
             return "User already exists"

        try:
            new_user = User(email=email, password=password)
            db.session.add(new_user)
            db.session.commit()
            return redirect(url_for("login"))
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
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
