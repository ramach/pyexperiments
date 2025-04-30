```text
collection of utilities for document processing
For example pdf_retriever_util.py  utility:

Reads every page’s selectable text with PyMuPDF.

Extracts each embedded image’s bytes.

Runs your Tesseract OCR (or any OCR function) on those bytes.

Optionally saves extracted images to disk for preview.

Returns three parallel outputs: overall text, image metadata, and per‑image OCR results—ready to feed into your LLM mapping.


Extracts business rules from .docx

Maps them with an LLM

Displays results with confidence

Allows exporting all mapped rules as a single JSON file
```

