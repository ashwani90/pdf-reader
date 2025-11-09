from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import requests
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import os
import uvicorn
from fastapi.responses import StreamingResponse
import shutil
import json
from PIL import Image
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re

app = FastAPI(title="PDF Reader API", version="1.1")

OUTPUT_DIR = "split_pdfs"

@app.get("/")
def root():
    return {"message": "Welcome to the PDF Reader API"}


def read_pdf_from_file(file_path: str) -> list[str]:
    """
    Read PDF file and return a list of strings, one per page.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    pages = []
    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
    
    return pages

def read_pdf_from_url(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Unable to download PDF from {url}")
    
    pdf_file = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text
    
@app.get("/company-report/")
def get_company_report(
    company_name: str = Query(..., description="Exact company name as used in stored JSON")
):
    """
    Example usage:
    http://localhost:8000/company-report/?company_name=Varun%20Beverages%20Limited
    """
    folder_path = os.path.join("output", "reports", company_name)
    file_path = os.path.join(folder_path, f"{company_name}.json")

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"No folder found for company '{company_name}'")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Merged JSON file not found for this company")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return JSONResponse(
            content=data,
            media_type="application/json",
            status_code=200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading JSON: {e}")

@app.get("/read-pdf/")
def read_pdf_api(source: str = Query(..., description="PDF local path or URL")):
    """
    Example usage:
    /read-pdf/?source=/absolute/path/to/file.pdf
    /read-pdf/?source=docs/sample.pdf
    /read-pdf/?source=https://arxiv.org/pdf/2408.09869
    """
    # try:
    #     if source.startswith("http://") or source.startswith("https://"):
    #         pages = read_pdf_from_url(source)
    #     else:
    #         # Ensure local path is absolute
    #         if not source.startswith("/"):
    #             # Convert relative path to absolute
    #             import os
    #             source = os.path.abspath(source)
    #         pages = read_pdf_from_file(source)

    #     formatted_content = {}
    #     for i, page_content in enumerate(pages):
    #         cleaned_text = page_content.strip().replace("\n\n", "\n")
    #         print(f"\n--- Page {i+1} ---\n{cleaned_text}\n")  # print as you go
    #         formatted_content[f"Page {i+1}"] = cleaned_text

    #     return JSONResponse(content={"source": source, "pages": formatted_content})

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    
    def page_generator(source):
        if source.startswith("http://") or source.startswith("https://"):
            pages = read_pdf_from_url(source)
        else:
            import os
            if not source.startswith("/"):
                source = os.path.abspath(source)
            pages = read_pdf_from_file(source)

        for i, page_content in enumerate(pages):
            cleaned_text = page_content.strip().replace("\n\n", "\n")
            yield f"\n--- Page {i+1} ---\n{cleaned_text}\n"

    return StreamingResponse(page_generator(source), media_type="text/plain")

@app.get("/split-pdf/")
def split_pdf_api(file_path: str = Query(..., description="Local PDF file path"),
                  pages_per_file: int = Query(10, description="Number of pages per split PDF")):
    """
    Split a PDF at a given local file path into smaller PDFs of 'pages_per_file' pages each.
    """
    output_dir = OUTPUT_DIR
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    if not file_path.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    folder = file_path.rsplit("/")[0].split(".")[0]
    output_dir = os.path.join(output_dir, folder)
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read PDF
    pdf_reader = PdfReader(file_path)
    total_pages = len(pdf_reader.pages)
    split_files = []

    # Split PDF
    for start in range(0, total_pages, pages_per_file):
        pdf_writer = PdfWriter()
        end = min(start + pages_per_file, total_pages)

        for i in range(start, end):
            pdf_writer.add_page(pdf_reader.pages[i])

        output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_part_{start+1}-{end}.pdf"
        print(output_file_name)
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0])
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, output_file_name)

        with open(output_path, "wb") as out_file:
            pdf_writer.write(out_file)

        split_files.append(output_file_name)

    return JSONResponse(content={
        "original_file": os.path.basename(file_path),
        "total_pages": total_pages,
        "pages_per_file": pages_per_file,
        "split_files": split_files
    })
    
@app.get("/scan-pdfs/")
def scan_pdfs(directory_path: str = Query(..., description="Path to the directory containing PDFs"), orig_text_file: str = Query(..., description="Original text file name"), small: bool = Query(False, description="If true, do not save to file, just return content")):
    """
    Scan a directory for PDF files and append their content page by page to a text file.
    """
    direct = "output/content/"
    # TODO:: Should come from the params
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        return JSONResponse(content={"message": "No PDF files found in the directory."})
    j = 1
    content = ''

    for pdf_file in pdf_files:
        full_pdf_path = os.path.join(directory_path, pdf_file)
        pages = read_pdf_from_file(full_pdf_path)  # use full path
        os.makedirs(direct + orig_text_file, exist_ok=True)
        text_file = direct + orig_text_file + "/" + orig_text_file + str(j) + ".txt"
        j += 1
        with open(text_file, "a", encoding="utf-8") as f:
            for i, page_content in enumerate(pages):
                cleaned_text = page_content.strip().replace("\n\n", "\n")
                if small:
                    content += f"\n---||---\n{cleaned_text}\n"
                else:
                    f.write(f"\n---||---\n{cleaned_text}\n")
    
    return JSONResponse(content={
        "directory_scanned": directory_path,
        "pdf_files_found": len(pdf_files),
        "txt_file_updated": text_file,
        "files_added": pdf_files,
        "content": content
    })
    
@app.get("/scan-images/")
def scan_images(
    directory_path: str = Query(..., description="Path to the directory containing images"),
    orig_text_file: str = Query(..., description="Base name for output text files"),
    small: bool = Query(False, description="If true, return text instead of saving")
):
    """
    Scan a directory for image files (JPG/PNG) and extract text using OCR.
    - Saves text in 'output/content/<orig_text_file>/' folder by default
    - If small=True, returns extracted text directly
    """
    output_dir = os.path.join("output", "content")
    os.makedirs(output_dir, exist_ok=True)

    # Validate directory
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")

    # Find all image files
    image_files = [
        f for f in os.listdir(directory_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp"))
    ]

    if not image_files:
        return JSONResponse(content={"message": "No image files found in the directory."})

    j = 1
    combined_content = ""

    for image_file in image_files:
        full_image_path = os.path.join(directory_path, image_file)

        try:
            # OCR extraction
            img = Image.open(full_image_path)
            extracted_text = pytesseract.image_to_string(img)
        except Exception as e:
            print(f"⚠️ Error reading {image_file}: {e}")
            continue

        # Prepare text folder and file path
        company_folder = os.path.join(output_dir, orig_text_file)
        os.makedirs(company_folder, exist_ok=True)
        text_file_path = os.path.join(company_folder, f"{orig_text_file}{j}.txt")
        j += 1

        # Clean text
        cleaned_text = extracted_text.strip().replace("\n\n", "\n")

        # Write or return
        if small:
            combined_content += f"\n---||---\n{cleaned_text}\n"
        else:
            with open(text_file_path, "a", encoding="utf-8") as f:
                f.write(f"\n---||---\n{cleaned_text}\n")

    return JSONResponse(content={
        "directory_scanned": directory_path,
        "image_files_found": len(image_files),
        "output_folder": os.path.join(output_dir, orig_text_file),
        "content_returned": small,
        "content": combined_content if small else "Saved to files",
        "files_added": image_files
    })


def preprocess_image(image_path):
    """
    Preprocess image for OCR:
    - Convert to grayscale
    - Enhance contrast
    - Binarize
    - Denoise (optional)
    """
    img = Image.open(image_path)

    # Convert to grayscale
    img = img.convert("L")

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Apply binary threshold
    img = img.point(lambda x: 0 if x < 140 else 255, '1')

    # Slight blur to remove small noise
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


def clean_extracted_text(text):
    """
    Clean OCR text:
    - Normalize spacing
    - Remove broken hyphenations
    - Replace multiple newlines
    - Remove junk symbols
    """
    text = text.replace("­", "")                # soft hyphens
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text) # join hyphenated words
    text = re.sub(r"\n{2,}", "\n", text)        # remove excessive newlines
    text = re.sub(r"[ \t]+", " ", text)         # normalize spaces
    text = text.strip()
    return text


@app.get("/scan-images2/")
def scan_images2(
    directory_path: str = Query(..., description="Path to the directory containing images"),
    orig_text_file: str = Query(..., description="Base name for output text files"),
    small: bool = Query(False, description="If true, return text instead of saving")
):
    """
    Scan a directory for image files (JPG/PNG) and extract text using improved OCR.
    - Saves text in 'output/content/<orig_text_file>/' folder
    - If small=True, returns combined text
    """
    output_dir = os.path.join("output", "content")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")

    image_files = [
        f for f in os.listdir(directory_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp"))
    ]

    if not image_files:
        return JSONResponse(content={"message": "No image files found in the directory."})

    combined_content = ""
    company_folder = os.path.join(output_dir, orig_text_file)
    os.makedirs(company_folder, exist_ok=True)

    for idx, image_file in enumerate(sorted(image_files), start=1):
        full_image_path = os.path.join(directory_path, image_file)

        try:
            img = preprocess_image(full_image_path)

            # layout-aware mode (detects paragraphs better)
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(img, config=custom_config)

            cleaned_text = clean_extracted_text(extracted_text)

            text_file_path = os.path.join(company_folder, f"{orig_text_file}{idx}.txt")

            if small:
                combined_content += f"\n---||---\n{cleaned_text}\n"
            else:
                with open(text_file_path, "a", encoding="utf-8") as f:
                    f.write(f"\n---||---\n{cleaned_text}\n")

        except Exception as e:
            print(f"⚠️ Error processing {image_file}: {e}")
            continue

    return JSONResponse(content={
        "directory_scanned": directory_path,
        "image_files_found": len(image_files),
        "output_folder": company_folder,
        "content_returned": small,
        "content": combined_content if small else "Saved to files",
        "files_processed": image_files
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)