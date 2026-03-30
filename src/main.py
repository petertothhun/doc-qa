import os
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

# load env
load_dotenv()

# models
Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# chunking
Settings.text_splitter = SentenceSplitter(
    chunk_size=600,
    chunk_overlap=80
)

PERSIST_DIR = "./storage"

def build_index():
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def load_index():
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    return load_index_from_storage(storage_context)

if __name__ == "__main__":
    if not os.path.exists(PERSIST_DIR):
        print("Building index...")
        index = build_index()
    else:
        print("Loading existing index...")
        index = load_index()

    query_engine = index.as_query_engine(similarity_top_k=3)

    while True:
        q = input("\nQuestion: ")
        if q.lower() in ["exit", "quit"]:
            break

        response = query_engine.query(
            f"Answer only based on the provided context. If unsure, say you do not know.\n\n{q}"
        )
        print("\nAnswer:", response)