"""Reading the text of uploaded clinical documents."""

from io import BytesIO

import ollama
import pypdfium2
from pypdf import PdfReader

from coda.config import settings

# File name endings of images read with OCR
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
# Scale of scanned PDF pages rendered for OCR, relative to 72 dpi
RENDER_SCALE = 2
OCR_PROMPT = "Text Recognition:"
# Context the OCR model needs for a page image and its text
OCR_CONTEXT = 16384


def ocr_model():
    """Return the configured OCR model, empty if OCR is off."""
    return settings.app.get("ocr_model") or ""


class OcrUnavailable(Exception):
    """The OCR model could not be reached or run."""


def ocr_image(data):
    """Return the text the OCR model reads in an image."""
    client = ollama.Client(host=settings.llm.ollama.base_url)
    try:
        response = client.generate(
            model=ocr_model(), prompt=OCR_PROMPT, images=[data],
            options={"num_ctx": OCR_CONTEXT, "temperature": 0})
    except (ollama.ResponseError, ConnectionError) as e:
        raise OcrUnavailable(str(e)) from e
    return response.response.strip()


def pdf_text(data):
    """Return the text of all pages of a PDF, reading pages that have no
    text layer, such as scans, with OCR, and whether any page was."""
    texts = []
    pdf = None
    for i, page in enumerate(PdfReader(BytesIO(data)).pages):
        text = page.extract_text() or ""
        if not text.strip() and ocr_model():
            pdf = pdf or pypdfium2.PdfDocument(data)
            image = BytesIO()
            pdf[i].render(scale=RENDER_SCALE).to_pil().save(image, "PNG")
            text = ocr_image(image.getvalue())
        texts.append(text)
    return "\n\n".join(texts), pdf is not None


def document_text(data, filename):
    """Return the text of a PDF or an image of a document, and whether it
    was read with OCR."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return pdf_text(data)
    if name.endswith(IMAGE_SUFFIXES) and ocr_model():
        return ocr_image(data), True
    raise ValueError(f"Unsupported document {filename}")
