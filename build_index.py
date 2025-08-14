import json, os
import chromadb
from chromadb.utils import embedding_functions

DATA_PATH = "data/sample_docs.json"
DB_DIR = "chroma_db"

def main():
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(
        name="fin_consumer_corpus",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    ids = [d["id"] for d in docs]
    texts = [d["body"] for d in docs]
    metas = [{"title": d["title"], "url": d["url"], "date": d["date"]} for d in docs]
    collection.upsert(ids=ids, documents=texts, metadatas=metas)
    print(f"Indexed {len(ids)} docs into {DB_DIR}")

if __name__ == "__main__":
    main()