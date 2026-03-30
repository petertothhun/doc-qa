import os
import docx2txt
import re

from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from docx2txt import process

# =========================
# CONFIG
# =========================

DATA_DIR = "./data"
PROCESSED_DIR = "./data_processed"
PERSIST_DIR = "./storage"
DEBUG = True   # set to False to disable chunk inspection

# =========================
# INIT
# =========================

load_dotenv()

Settings.llm = Ollama(
    model="qwen2.5:7b",
    request_timeout=120,
    context_window=4096
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

Settings.text_splitter = SentenceSplitter(
    chunk_size=400,
    chunk_overlap=100
)

# =========================
# BUILD / LOAD INDEX
# =========================

SECTION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)")

def split_into_sections(text):
    if not isinstance(text, str) or not text.strip():
        return []

    lines = text.splitlines()

    sections = []
    current_section = None

    for line in lines:
        if not line:
            continue

        line = line.strip()
        if not line:
            continue

        match = SECTION_PATTERN.match(line)

        if match:
            # start new section
            if current_section:
                sections.append(current_section.strip())

            current_section = line
        else:
            # append to current section
            if current_section:
                current_section += " " + line
            else:
                # text before first numbered section
                current_section = line

    if current_section:
        sections.append(current_section.strip())

    return sections


def prepare_documents():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for file in os.listdir(DATA_DIR):
        input_path = os.path.join(DATA_DIR, file)

        if file.endswith(".docx"):
            output_file = file.replace(".docx", ".txt")
            output_path = os.path.join(PROCESSED_DIR, output_file)

            # only convert if not already done
            if not os.path.exists(output_path):
                print(f"Converting {file} → {output_file}")
                text = docx2txt.process(input_path)

                with open(output_path, "w") as f:
                    f.write(text)

        elif file.endswith(".txt"):
            # copy txt files directly
            output_path = os.path.join(PROCESSED_DIR, file)
            if not os.path.exists(output_path):
                with open(input_path, "r") as src, open(output_path, "w") as dst:
                    dst.write(src.read())

def build_index():
    print("Preparing documents...")
    prepare_documents()

    print("Loading processed documents...")
    documents_raw = SimpleDirectoryReader(PROCESSED_DIR).load_data()

    documents = []

    for doc in documents_raw:
        sections = split_into_sections(doc.text)

        for sec in sections:
            documents.append(Document(text=sec))

    print("\n--- Sample document content ---")
    print(documents[0].text[:1000])

    print("\nBuilding index...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

    return index


def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    return load_index_from_storage(storage_context)


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    if not os.path.exists(f"{PERSIST_DIR}/docstore.json"):
        print("No valid index found. Building new one...")
        index = build_index()
    else:
        print("Loading existing index...")
        index = load_index()

    query_engine = index.as_query_engine(similarity_top_k=5)

    while True:
        q = input("\nQuestion: ")

        if q.lower() in ["exit", "quit"]:
            break

        # =========================
        # DEBUG: inspect retrieval
        # =========================
        if DEBUG:
            retriever = index.as_retriever(similarity_top_k=5)
            nodes = retriever.retrieve(q)

            print("\n--- Retrieved chunks ---")
            for i, n in enumerate(nodes):
                print(f"\nChunk {i+1}:\n{n.text[:500]}")

        # =========================
        # QUERY
        # =========================
        response = query_engine.query(
            f"Answer ONLY based on the provided context. "
            f"If unsure, say you do not know.\n\nQuestion: {q}"
        )

        print("\n--- Answer ---")
        print(response)