# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - tesseract-ocr: For extracting text from images/PDFs
# - libgl1 & libglib2.0-0: Required for OpenCV (headless version)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-tam \
    tesseract-ocr-hin \
    tesseract-ocr-tel \
    tesseract-ocr-kan \
    tesseract-ocr-mal \
    tesseract-ocr-mar \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies and gunicorn 
RUN pip install --no-cache-dir -r requirements.txt

# Download required AI models (spacy and NLTK)
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader punkt punkt_tab

# Copy the entire project code into the container
COPY . .

# Create the instance directory for SQLite database
RUN mkdir -p instance

# Expose port (default for web hosting platforms like Render)
EXPOSE 5000

# Start Gunicorn server (2 workers, binding to port 5000)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
