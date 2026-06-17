import io
import requests
import pdfplumber

def load_pdf_from_url(url: str) -> str:
    # download the PDF from Cloudinary into memory (no disk write)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    text = ""
    # read each page and collect all text
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()
