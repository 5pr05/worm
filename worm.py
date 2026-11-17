import sys
import os
import pymupdf
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

DB_PATH = os.path.expanduser("~/.local/share/worm_db")
BOOKS_DIR = os.path.expanduser("~/G3N/bookz")

def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
    return client.get_or_create_collection(name="library", embedding_function=emb_fn)

def index_pdf(file_path):
    if not os.path.exists(file_path):
        return
        
    book_name = os.path.basename(file_path)
    collection = get_collection()
    
    existing = collection.get(where={"book": book_name})
    if len(existing["ids"]) > 0:
        print(f"Skipping: '{book_name}' (already indexed)")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    print(f"Indexing '{book_name}'...")
    doc = pymupdf.open(file_path)
    
    docs_batch, metas_batch, ids_batch = [], [], []
    chunk_id = 0
    BATCH_SIZE = 150
    
    for page_num, page in enumerate(doc, start=1):
        if page_num % 50 == 0:
            print(f"  ...read {page_num} of {len(doc)} pages")
            
        text = page.get_text().strip()
        if len(text) < 50: 
            continue
        
        for chunk in splitter.split_text(text):
            docs_batch.append(chunk)
            metas_batch.append({"book": book_name, "page": page_num})
            ids_batch.append(f"{book_name}_p{page_num}_{chunk_id}")
            chunk_id += 1
            
            if len(docs_batch) >= BATCH_SIZE:
                print(f"  Vectorizing batch of {BATCH_SIZE} chunks...")
                collection.add(documents=docs_batch, metadatas=metas_batch, ids=ids_batch)
                docs_batch, metas_batch, ids_batch = [], [], []
                
    if docs_batch:
        collection.add(documents=docs_batch, metadatas=metas_batch, ids=ids_batch)
        
    print(f"Vectors added (total {chunk_id} chunks)")