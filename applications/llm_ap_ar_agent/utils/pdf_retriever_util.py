from __future__ import annotations

from typing import Tuple, Dict, List


def retrieve_pdf_with_image_ocr(
        pdf_input,
        save_img_dir: str | None = None,
        img_format: str = "png"
) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Extracts text + OCR from embedded images in a PDF.
    Accepts either a file path (str) or file-like object (e.g., BytesIO).
    """
    import fitz
    import os
    from PIL import Image
    import io
    import pytesseract

    def extract_text_from_image_bytes(img_bytes: bytes) -> str:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            return f"[OCR‑ERROR] {e}"

    if isinstance(pdf_input, str):
        doc = fitz.open(pdf_input)
    else:
        # Handle file-like object (e.g., Streamlit upload)
        doc = fitz.open(pdf_input.read(), filetype="pdf")

    all_text = []
    images_meta, images_ocr = [], []

    if save_img_dir:
        os.makedirs(save_img_dir, exist_ok=True)

    for page_idx, page in enumerate(doc):
        all_text.append(page.get_text() or "")

        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base["image"]
            img_ext = base["ext"]

            filepath = None
            if save_img_dir:
                fname = f"page{page_idx+1}_img{img_idx+1}.{img_format}"
                filepath = os.path.join(save_img_dir, fname)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)

            ocr_text = extract_text_from_image_bytes(img_bytes)

            images_meta.append({
                "page": page_idx,
                "xref": xref,
                "width": base["width"],
                "height": base["height"],
                "ext": img_ext,
                "filepath": filepath,
            })
            images_ocr.append({
                "page": page_idx,
                "xref": xref,
                "ocr_text": ocr_text
            })

    doc.close()
    return "\n".join(all_text).strip(), images_meta, images_ocr

