from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages)

def extract_text_from_word(file_path):
    """Extract text from a Word document."""
    doc = Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_image(file_path):
    """Extract text from an image file."""
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)