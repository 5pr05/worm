# worm

A local semantic search CLI tool for your PDF library. It operates as a RAG (Retrieval-Augmented Generation) system without the "G" (Generation) — providing exact sources instead of generated text.

Unlike LLM wrappers that hallucinate facts or code, `worm` strictly retrieves the semantic meaning of your query and returns the exact absolute page numbers and filenames where the information is located. It relies on the actual authors of your books, not on a generative neural network.

## Features

* **Zero Hallucinations:** Does not generate text. It only points to actual pages in your local PDF files.
* **100% Offline & Private:** No telemetry, no API keys, no internet connection required after the initial model download.
* **Unix-Way:** Consumes 0 MB of RAM when idle. The script initializes, loads weights into memory, processes the query, and exits immediately.
* **OSC 8 Terminal Links:** Search results are output as clickable terminal hyperlinks. Clicking the page number directly opens your PDF reader at the exact page (requires a modern terminal emulator and a compatible PDF reader).
* **Hardware Friendly:** Optimized for CPU inference without requiring a discrete GPU. Chunk processing is batched to prevent CPU throttling.

## Tech Stack

* **Embedding Model:** BAAI/bge-m3
* **Vector Database:** ChromaDB
* **PDF Parsing:** PyMuPDF
* **Text Chunking:** LangChain Text Splitters

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/5pr05/worm.git
cd worm

```

2. **Set up an isolated virtual environment:**
```bash
python -m venv ~/.local/share/worm-env
source ~/.local/share/worm-env/bin/activate
pip install -r requirements.txt
deactivate

```

3. **Configure the book directory:**
Open `worm.py` in your text editor and change the `BOOKS_DIR` variable to the path where your PDF files are stored.

```python
BOOKS_DIR = os.path.expanduser("~/Your/PDF/Directory")

```

4. **Install the script locally:**
Copy the python script into the environment folder and create a global bash wrapper. Ensure `~/.local/bin` is in your system `$PATH`.

```bash
cp worm.py ~/.local/share/worm-env/
echo '#!/bin/bash
~/.local/share/worm-env/bin/python ~/.local/share/worm-env/worm.py "$@"' > ~/.local/bin/worm
chmod +x ~/.local/bin/worm

```

## Usage

**Indexing PDFs**
To index a single file:

```bash
worm --index ~/Your/PDF/Directory/book.pdf

```

To batch index an entire directory (the script automatically skips already indexed files to prevent duplicates):

```bash
for f in ~/Your/PDF/Directory/*.pdf; do worm --index "$f"; done

```

*Note: On the very first run, the system will download the `bge-m3` to your local cache.*

**Searching**
Pass your query directly as arguments (quotes are optional):

```bash
worm how to extract a variable

```

**Example Output:**

```text
Query: 'how to extract a variable'

File: Fowler_Refactoring.pdf
   Pages: 168, 172, 289

File: Professional_TypeScript.pdf
   Pages: 73

```

If your terminal supports OSC 8, the page numbers will be clickable hyperlinks that execute `xdg-open file://path/to/book.pdf#page=N`.

## Uninstallation

To completely remove the tool and its dependencies, delete the following directories and files:

```bash
rm -rf ~/.local/share/worm-env     # Removes the python environment and script
rm ~/.local/bin/worm               # Removes the bash wrapper
rm -rf ~/.local/share/worm_db      # Removes the vector database

```