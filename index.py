"""
SOP Document Ingestion Pipeline
Loads PDFs, cleans text, chunks them, and builds a FAISS vector index.
"""
import argparse
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_PDF_DIR = "SOP"
DEFAULT_INDEX_PATH = "db/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """Normalize whitespace and strip common PDF artifacts."""
    # Remove "Page X of Y" / "Page X" footers
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPage\s+\d+\b", " ", text, flags=re.IGNORECASE)
    # Strip non-printable control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ingest_docs(pdf_dir: str, index_path: str) -> None:
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists() or not any(pdf_path.glob("*.pdf")):
        log.error("No PDFs found in '%s'. Add SOP files and retry.", pdf_dir)
        sys.exit(1)

    # 1. Load PDFs
    log.info("Loading PDFs from '%s'...", pdf_dir)
    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    n_files = len({d.metadata.get("source") for d in documents})
    log.info("Loaded %d pages from %d file(s).", len(documents), n_files)

    # 2. Clean text + enrich metadata
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        src = doc.metadata.get("source", "unknown")
        doc.metadata["source_file"] = Path(src).name

    # 3. Chunk
    log.info("Splitting into chunks (size=%d, overlap=%d)...", CHUNK_SIZE, CHUNK_OVERLAP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info("Produced %d chunks.", len(chunks))

    # 4. Embed
    log.info("Loading embedding model '%s'...", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 5. Build & persist FAISS
    log.info("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_path)

    log.info("✅ FAISS index saved to '%s' (%d chunks).", index_path, len(chunks))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest SOP PDFs into a FAISS index.")
    p.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR,
                   help=f"Folder of PDFs (default: {DEFAULT_PDF_DIR})")
    p.add_argument("--index-path", default=DEFAULT_INDEX_PATH,
                   help=f"Output FAISS path (default: {DEFAULT_INDEX_PATH})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_docs(args.pdf_dir, args.index_path)