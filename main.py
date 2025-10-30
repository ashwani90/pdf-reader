from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import requests
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import os
import uvicorn
from fastapi.responses import StreamingResponse
import shutil

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
def scan_pdfs(directory_path: str = Query(..., description="Path to the directory containing PDFs")):
    """
    Scan a directory for PDF files and append their content page by page to a text file.
    """
    direct = "output/content/"
    # TODO:: Should come from the params
    orig_text_file = "VBL-2023"
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory_path}")

    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        return JSONResponse(content={"message": "No PDF files found in the directory."})
    j = 1

    for pdf_file in pdf_files:
        full_pdf_path = os.path.join(directory_path, pdf_file)
        pages = read_pdf_from_file(full_pdf_path)  # use full path
        os.makedirs(direct + orig_text_file, exist_ok=True)
        text_file = direct + orig_text_file + "/" + orig_text_file + str(j) + ".txt"
        j += 1
        with open(text_file, "a", encoding="utf-8") as f:
            for i, page_content in enumerate(pages):
                cleaned_text = page_content.strip().replace("\n\n", "\n")
                f.write(f"\n---||---\n{cleaned_text}\n")
    
    return JSONResponse(content={
        "directory_scanned": directory_path,
        "pdf_files_found": len(pdf_files),
        "txt_file_updated": text_file,
        "files_added": pdf_files
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)