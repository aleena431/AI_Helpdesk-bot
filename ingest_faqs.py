"""Build a persistent local ChromaDB index from faqs.json.

Embeddings use a small deterministic feature-hashing function, so ingestion and
search work offline without downloading a model.
"""
import hashlib
import json
import re
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent
FAQ_PATH = ROOT / "faqs.json"
DB_PATH = ROOT / ".chroma"
COLLECTION = "it_helpdesk_faqs"
DIMENSIONS = 256


def embed(text: str) -> list[float]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    vector = [0.0] * DIMENSIONS
    for feature in features:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        vector[number % DIMENSIONS] += 1.0 if number & (1 << 8) else -1.0
    norm = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / norm for value in vector]


def ingest() -> int:
    faqs = json.loads(FAQ_PATH.read_text(encoding="utf-8"))
    if not isinstance(faqs, list) or not faqs:
        raise ValueError("faqs.json must contain a non-empty list")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[str(item.get("id", index)) for index, item in enumerate(faqs)],
        documents=[f"{item['question']}\n{item['answer']}" for item in faqs],
        metadatas=[{"question": item["question"], "answer": item["answer"]} for item in faqs],
        embeddings=[embed(f"{item['question']} {item['answer']}") for item in faqs],
    )
    return collection.count()


if __name__ == "__main__":
    print(f"Indexed {ingest()} FAQs in {DB_PATH}")
