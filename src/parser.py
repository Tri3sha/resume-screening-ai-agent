import os
import pymupdf
from docx import Document


def extract_text_from_pdf(file_path):
    """Extract text from a PDF resume."""
    text = ""

    try:
        document = pymupdf.open(file_path)

        for page in document:
            text += page.get_text()

        document.close()

    except Exception as error:
        print(f"Error reading PDF {file_path}: {error}")

    return text


def extract_text_from_docx(file_path):
    """Extract text from a DOCX resume."""
    text = ""

    try:
        document = Document(file_path)

        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"

    except Exception as error:
        print(f"Error reading DOCX {file_path}: {error}")

    return text


def extract_text_from_txt(file_path):
    """Extract text from a TXT resume."""
    text = ""

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

    except Exception as error:
        print(f"Error reading TXT {file_path}: {error}")

    return text


def load_resumes(folder):
    """
    Read all PDF, DOCX and TXT resumes from a folder.

    Returns:
        Dictionary containing filename and extracted text.
    """

    resumes = {}

    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return resumes

    for file_name in os.listdir(folder):

        file_path = os.path.join(folder, file_name)

        if not os.path.isfile(file_path):
            continue

        extension = os.path.splitext(file_name)[1].lower()

        try:

            if extension == ".pdf":
                text = extract_text_from_pdf(file_path)

            elif extension == ".docx":
                text = extract_text_from_docx(file_path)

            elif extension == ".txt":
                text = extract_text_from_txt(file_path)

            else:
                continue

            resumes[file_name] = text

            print(f"Successfully read: {file_name}")

        except Exception as error:
            print(f"Error reading {file_name}: {error}")

    return resumes


if __name__ == "__main__":

    # Folder containing resumes
    resumes_folder = "data/resumes"

    print("\n===================================")
    print("      RESUME SCREENING AGENT")
    print("===================================\n")

    resumes = load_resumes(resumes_folder)

    print(f"\nTotal resumes found: {len(resumes)}\n")

    for file_name, text in resumes.items():

        print("-----------------------------------")
        print(f"Resume: {file_name}")
        print("-----------------------------------")

        # Display only the first 500 characters
        print(text[:500])

        print("\n")