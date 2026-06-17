import os
import hashlib
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.vectorstore import add_documents
from ingestion.pdf_loader import load_pdf_from_url

# split large PDF text into 500-word chunks with 50-word overlap between chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def _index_document(url: str, title: str, subject: str, doc_id: str):
    # download and extract text from the PDF
    text = load_pdf_from_url(url)
    if not text:
        raise ValueError(f"No text could be extracted from {url}")

    # wrap text in a LangChain Document so the splitter preserves metadata on each chunk
    doc = Document(
        page_content=text,
        metadata={"title": title, "subject": subject, "material_id": doc_id, "url": url},
    )
    chunks = splitter.split_documents([doc])

    # use stable IDs so re-indexing the same doc overwrites instead of duplicating
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    add_documents(chunks, ids=ids)
    return len(chunks)

def ingest_all():
    # fetch all uploaded materials from MongoDB
    with MongoClient(os.getenv("MONGODB_URI")) as client:
        db = client[os.getenv("DB_NAME", "test")]
        materials = list(db.materials.find({}))

    print(f"Found {len(materials)} material(s) in MongoDB")
    total_chunks = 0

    for mat in materials:
        url = mat.get("file", {}).get("url", "")
        title = mat.get("title", "Untitled")
        subject = str(mat.get("subject", "General"))
        mat_id = str(mat["_id"])

        if not url:
            print(f"  Skipping '{title}' — no file URL")
            continue

        print(f"Processing: {title}")
        try:
            n = _index_document(url, title, subject, mat_id)
            total_chunks += n
            print(f"  Indexed {n} chunks")
        except Exception as e:
            print(f"  Error processing '{title}': {e}")

    print(f"Ingestion complete. Total chunks indexed: {total_chunks}")

def ingest_single(url: str, title: str, subject: str = "General"):
    # md5 of URL used as doc_id — same URL always produces same ID
    print(f"Ingesting: {title}")
    doc_id = hashlib.md5(url.encode()).hexdigest()
    n = _index_document(url, title, subject, doc_id)
    print(f"Indexed {n} chunks for '{title}'")
