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