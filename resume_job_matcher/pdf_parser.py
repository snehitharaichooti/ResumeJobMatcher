"""PDF -> clean text using PyMuPDF (fitz)."""
from __future__ import annotations

import re

import fitz  # PyMuPDF


def clean_text(text: str) -> str:
    t = text.replace("\u00a0", " ").replace("\ufb01", "fi").replace("\ufb02", "fl")
    t = re.sub(r"[•●▪■◦▶►‣⁃·]", "-", t)
    t = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", t)  # re-join words hyphenated across lines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_text_from_pdf(source) -> str:
    """Accepts a path, raw bytes, or a file-like object (e.g. Streamlit UploadedFile)."""
    if isinstance(source, (bytes, bytearray)):
        doc = fitz.open(stream=bytes(source), filetype="pdf")
    elif hasattr(source, "getvalue"):
        doc = fitz.open(stream=source.getvalue(), filetype="pdf")
    elif hasattr(source, "read"):
        doc = fitz.open(stream=source.read(), filetype="pdf")
    else:
        doc = fitz.open(str(source))
    try:
        pages = [page.get_text("text", sort=True) for page in doc]
    finally:
        doc.close()
    return clean_text("\n".join(pages))
