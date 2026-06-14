import os
import re
import time
from typing import Dict

# Core LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_qdrant import QdrantVectorStore
# OCR Fallback Imports
from pdf2image import convert_from_path
import pytesseract

# --- CONFIGURATION ---
DATA_PATH = "./Dataset_govt"   # Folder where your PDFs are stored
DB_PATH = "./qdrant_storage"     # Folder where the vector database will be saved
COLLECTION_NAME = "gov_docs"

def extract_metadata(text: str, filename: str) -> Dict:
    """Extracts structural legal metadata and document type classifications."""
    metadata = {
        "source_file": filename,
        "document_type": "Unknown",
        "circular_no": "Unknown",
        "date": "Unknown",
        "supersedes": None
    }
    
    # Simple heuristic classification
    lower_text = text.lower()[:1000]
    if "notification" in lower_text: metadata["document_type"] = "Notification"
    elif "circular" in lower_text: metadata["document_type"] = "Circular"
    elif "press release" in lower_text: metadata["document_type"] = "Press Release"

    # Regex Extraction Rules
    circ_match = re.search(r'(Circular No\.|Notification No\.|Circular No:|Notification No:)\s*([A-Z0-9/_.-]+)', text, re.IGNORECASE)
    date_match = re.search(r'(Dated|Date):?\s*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})', text, re.IGNORECASE)
    super_match = re.search(r'in supersession of.*?(Circular No\.\s*[A-Z0-9/_.-]+|Notification No\.\s*[A-Z0-9/_.-]+|\d+/\d+)', text, re.IGNORECASE)
    
    if circ_match: metadata["circular_no"] = circ_match.group(2).strip()
    if date_match: metadata["date"] = date_match.group(2).strip()
    if super_match: metadata["supersedes"] = super_match.group(1).strip()
        
    return metadata

def run_ingestion():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Directory '{DATA_PATH}' not found. Please create it and add PDFs.")
        return

    pdf_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in {DATA_PATH}.")
        return

    print(f"Found {len(pdf_files)} PDF files to ingest.")

    # 1. Initialize Text Splitter (500-1000 tokens ≈ 3000 chars)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    all_chunks = []

    # 2. Process Each PDF
    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        first_page_text = pages[0].page_content.strip() if pages else ""
        
        # Dual-Path: Native Text vs Scanned OCR
        if len(first_page_text) < 40:
            print("   -> Detected scanned PDF. Running OCR...")
            full_text = ""
            try:
                images = convert_from_path(file_path)
                for img in images:
                    full_text += pytesseract.image_to_string(img) + "\n"
            except Exception as e:
                print(f"   -> OCR Failed. Skipping file. Error: {e}")
                continue
        else:
            print("   -> Native text PDF detected.")
            full_text = "\n".join([p.page_content for p in pages])

        # Extract Metadata
        metadata_payload = extract_metadata(full_text, filename)
        print(f"   -> Metadata extracted: ID={metadata_payload['circular_no']}")

        # Create Chunks
        file_chunks = text_splitter.create_documents([full_text], metadatas=[metadata_payload])
        all_chunks.extend(file_chunks)

    if all_chunks:
        print("\nLoading HuggingFace Embedding Model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        print(f"Storing {len(all_chunks)} chunks into persistent Qdrant database...")
        start_time = time.time()
        
        # IMPORTANT: This saves the vectors to the physical folder specified in DB_PATH
        QdrantVectorStore.from_documents(
            all_chunks,
            embedding_model,
            path=DB_PATH,
            collection_name=COLLECTION_NAME
        )
        
        print(f"✅ Ingestion complete! Database saved to '{DB_PATH}' in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_ingestion()