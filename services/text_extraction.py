import os, cv2, fitz, docx, pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    # ---------- PDF ----------
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text")
        doc.close()

        # 🔴 If PDF has NO embedded text → OCR fallback
        if not text.strip():
            print("PDF has no embedded text → using OCR")
            for page in fitz.open(file_path):
                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                text += pytesseract.image_to_string(
                    gray,
                    lang="eng+tam+hin+tel+kan+mal+mar"
                )

    # ---------- DOCX ----------
    elif ext == ".docx":
        d = docx.Document(file_path)
        parts = []

        # paragraphs
        for p in d.paragraphs:
            if p.text.strip():
                parts.append(p.text)

        # tables (IMPORTANT)
        for table in d.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text)

        text = "\n".join(parts)

    # ---------- TXT ----------
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            text = f.read()

    # ---------- IMAGE ----------
    elif ext in [".png", ".jpg", ".jpeg"]:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(
            gray,
            lang="eng+tam+hin+tel+kan+mal+mar"
        )

    return text.strip()
