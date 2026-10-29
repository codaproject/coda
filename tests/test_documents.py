from io import BytesIO

import pytest
from pypdf import PdfWriter

from coda.app import documents


def blank_pdf(pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    data = BytesIO()
    writer.write(data)
    return data.getvalue()


@pytest.fixture
def ocr(monkeypatch):
    read = []
    monkeypatch.setattr(documents, "ocr_image",
                        lambda data: read.append(data) or "Hb 4.8 g/dL")
    return read


def test_scanned_pdf_pages_are_read_with_ocr(ocr):
    text, read_with_ocr = documents.document_text(blank_pdf(2), "scan.PDF")
    assert text == "Hb 4.8 g/dL\n\nHb 4.8 g/dL" and read_with_ocr
    assert all(page.startswith(b"\x89PNG") for page in ocr)


def test_images_are_read_with_ocr(ocr):
    assert documents.document_text(b"jpeg", "photo.jpg") == \
        ("Hb 4.8 g/dL", True)
    assert ocr == [b"jpeg"]


def test_no_ocr_without_a_model(ocr, monkeypatch):
    monkeypatch.setattr(documents, "ocr_model", lambda: "")
    assert documents.document_text(blank_pdf(), "scan.pdf") == ("", False)
    with pytest.raises(ValueError):
        documents.document_text(b"jpeg", "photo.jpg")
    assert ocr == []


def test_other_files_are_unsupported(ocr):
    with pytest.raises(ValueError):
        documents.document_text(b"data", "notes.docx")
