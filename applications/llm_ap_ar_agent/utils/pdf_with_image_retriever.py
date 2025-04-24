# Required installations:
# pip install pymupdf pypdf pdfminer.six pytesseract Pillow

# --- IMPORTANT FOR OCR ---
# You MUST install the Tesseract OCR engine separately on your system.
# It's NOT installed via pip.
# - Windows: Download from official Tesseract GitHub (look for ub-mannheim builds) & add to PATH.
# - macOS: brew install tesseract (requires a workaround to be not sudo)
# - Linux (Debian/Ubuntu): sudo apt update && sudo apt install tesseract-ocr
# - Linux (Fedora): sudo dnf install tesseract
# You might need to configure the path to tesseract executable below.
'''
This document provides several functions:

extract_text_with_pymupdf: Uses the fast PyMuPDF library to get standard text.
extract_text_with_pypdf: Uses the pypdf library for standard text extraction.
extract_text_with_pdfminer: Uses pdfminer.six, which is good for analyzing text layout.
extract_text_with_ocr: Uses PyMuPDF to find and extract images and then uses pytesseract to perform Optical Character Recognition (OCR) on those images, combining the results with any standard text found.

'''
import fitz  # PyMuPDF library is imported as 'fitz'
from pypdf import PdfReader
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import pytesseract
from PIL import Image
import io
import os # For checking file existence

# --- Configuration ---
# Uncomment and set the path if pytesseract can't find Tesseract automatically
# (Especially common on Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# === Scenario 1: Extracting Standard Text (Ignoring Images) ===

def extract_text_with_pymupdf(pdf_path):
    """
    Extracts standard text content from a PDF file using PyMuPDF (fitz).
    Fast and generally accurate for standard text.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: The extracted text, or None if an error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return None

    text = ""
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        # Iterate through each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Extract text from the page
            text += page.get_text("text") # Use "text" for plain text extraction
        doc.close()
    except Exception as e:
        print(f"Error processing {pdf_path} with PyMuPDF: {e}")
        return None
    return text

def extract_text_with_pypdf(pdf_path):
    """
    Extracts standard text content from a PDF file using pypdf.
    A modern library for general PDF tasks.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: The extracted text, or None if an error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return None

    text = ""
    try:
        # Create a PDF reader object
        reader = PdfReader(pdf_path)
        # Iterate through each page
        for page in reader.pages:
            # Extract text from the page
            extracted = page.extract_text()
            if extracted: # Add text only if extraction was successful for the page
                text += extracted
    except Exception as e:
        print(f"Error processing {pdf_path} with pypdf: {e}")
        return None
    return text

def extract_text_with_pdfminer(pdf_path):
    """
    Extracts standard text content from a PDF file using pdfminer.six.
    Good for analyzing layout and reconstructing text order.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: The extracted text, or None if an error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return None

    output_string = StringIO()
    try:
        with open(pdf_path, 'rb') as fin:
            # LAParams helps analyze layout; adjust parameters if needed
            # Defaults are often fine, but can be tuned for complex layouts
            laparams = LAParams(line_margin=0.5, word_margin=0.1, boxes_flow=0.5)
            extract_text_to_fp(fin, output_string, laparams=laparams)
        text = output_string.getvalue()
        output_string.close()
    except Exception as e:
        print(f"Error processing {pdf_path} with pdfminer.six: {e}")
        return None
    return text

# === Scenario 2: Extracting Text Including from Images (OCR) ===

def extract_text_with_ocr(pdf_path, language='eng'):
    """
    Extracts both standard text and text from embedded images using OCR
    (PyMuPDF for PDF handling/image extraction + Pytesseract for OCR).

    Args:
        pdf_path (str): The file path to the PDF document.
        language (str): The language code for Tesseract OCR (e.g., 'eng', 'fra', 'spa').
                        Ensure the corresponding language data is installed for Tesseract.

    Returns:
        str: The combined extracted text (standard + OCR), or None if a critical error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return None

    full_text = ""
    processed_images_count = 0

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path} with PyMuPDF: {e}")
        return None

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = "" # Text for the current page

        # 1. Extract standard text from the page first
        try:
            standard_text = page.get_text("text")
            if standard_text:
                page_text += f"--- Page {page_num + 1} Standard Text ---\n"
                page_text += standard_text.strip() + "\n"
        except Exception as e:
            print(f"Warning: Could not extract standard text from page {page_num + 1}: {e}")

        # 2. Extract images and perform OCR
        image_list = []
        try:
            image_list = page.get_images(full=True)
        except Exception as e:
            print(f"Warning: Could not get images from page {page_num + 1}: {e}")

        if image_list:
            page_text += f"\n--- Page {page_num + 1} OCR Results ---\n"
            # print(f"Found {len(image_list)} image candidates on page {page_num + 1}")

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # Get the cross-reference number of the image

            try:
                # Extract the image bytes
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Load image bytes into Pillow (PIL fork)
                image = Image.open(io.BytesIO(image_bytes))

                # Perform OCR using pytesseract
                # You can add Tesseract config options if needed, e.g., psm for page segmentation mode
                # ocr_text = pytesseract.image_to_string(image, lang=language, config='--psm 6')
                ocr_text = pytesseract.image_to_string(image, lang=language)

                if ocr_text and ocr_text.strip(): # Add only if OCR finds non-whitespace text
                    page_text += f"[Image {img_index+1} OCR]:\n{ocr_text.strip()}\n"
                    processed_images_count += 1
                # else:
                #      print(f"  - No text found by OCR in image {img_index+1} (xref {xref}) on page {page_num+1}")

            except pytesseract.TesseractNotFoundError:
                print("\nCRITICAL ERROR: Tesseract executable not found.")
                print("Please install Tesseract OCR engine on your system")
                print("AND ensure it's in your PATH or configure 'pytesseract.pytesseract.tesseract_cmd'.")
                doc.close()
                return None # Stop processing if Tesseract is missing
            except Exception as e:
                print(f"Warning: Error processing image {img_index+1} (xref {xref}) on page {page_num+1}: {e}")
                # Continue to next image/page even if one image fails

        full_text += page_text # Append this page's text to the total

    doc.close()
    print(f"\nFinished processing. Found and attempted OCR on {processed_images_count} image(s).")
    return full_text.strip()


# --- Example Usage ---

# Create a dummy PDF for testing if you don't have one
def create_dummy_pdf(filename="dummy_test.pdf"):
    try:
        doc = fitz.open() # New empty PDF
        page = doc.new_page()
        # Add some standard text
        page.insert_text((50, 100), "This is standard text on the page.")
        page.insert_text((50, 120), "Here is another line of text.")

        # Try to add a simple image (a red square) - Note: Image embedding can be complex
        # For real testing, use a PDF that already contains images.
        # This simple shape might not be extracted as an 'image' by get_images in all cases.
        rect = fitz.Rect(50, 150, 150, 250)
        page.draw_rect(rect, color=(1, 0, 0), fill=(1, 0, 0)) # Red square
        page.insert_text((55, 170), "Text near shape") # Text near the shape

        doc.save(filename)
        doc.close()
        print(f"Created dummy PDF: {filename}")
        return True
    except Exception as e:
        print(f"Error creating dummy PDF: {e}")
        return False

# --- Main execution block ---
if __name__ == "__main__":
    # Create the dummy PDF or specify path to your existing PDF
    # pdf_file_path = 'your_actual_document.pdf' # <-- Use your PDF here
    pdf_file_path = 'dummy_test.pdf'
    if not os.path.exists(pdf_file_path):
        if not create_dummy_pdf(pdf_file_path):
            print("Failed to create dummy PDF. Exiting.")
            exit()


    print(f"\n--- Testing with: {pdf_file_path} ---")

    # Test Scenario 1: Standard Text Extraction
    print("\n1. Extracting text with PyMuPDF...")
    text_pymupdf = extract_text_with_pymupdf(pdf_file_path)
    if text_pymupdf:
        print("Success (PyMuPDF):\n", text_pymupdf[:300], "...\n") # Show preview
    else:
        print("Failed (PyMuPDF).\n")

    print("2. Extracting text with pypdf...")
    text_pypdf = extract_text_with_pypdf(pdf_file_path)
    if text_pypdf:
        print("Success (pypdf):\n", text_pypdf[:300], "...\n")
    else:
        print("Failed (pypdf).\n")

    print("3. Extracting text with pdfminer.six...")
    text_pdfminer = extract_text_with_pdfminer(pdf_file_path)
    if text_pdfminer:
        print("Success (pdfminer.six):\n", text_pdfminer[:300], "...\n")
    else:
        print("Failed (pdfminer.six).\n")

    # Test Scenario 2: OCR Extraction
    print("\n4. Extracting text with OCR (PyMuPDF + Pytesseract)...")
    print("(Requires Tesseract OCR engine to be installed and configured)")
    text_ocr = extract_text_with_ocr(pdf_file_path)
    if text_ocr is not None: # Check for None explicitly due to potential Tesseract error
        print("\n--- Full Text Extracted (including OCR attempts) ---")
        print(text_ocr)
        print("--- End of Full Text ---")
    else:
        print("\nOCR extraction failed or Tesseract not found.")