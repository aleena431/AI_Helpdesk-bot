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
    try:
        faqs = json.loads(FAQ_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"FAQ file not found: {FAQ_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {FAQ_PATH}: {exc.msg}") from exc

    if not isinstance(faqs, list) or not faqs:
        raise ValueError("faqs.json must contain a non-empty list")

    valid_faqs: list[dict[str, str | int]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(faqs):
        if not isinstance(item, dict):
            raise ValueError(f"FAQ entry {index + 1} must be a JSON object")

        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"FAQ entry {index + 1} must have a non-empty question")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"FAQ entry {index + 1} must have a non-empty answer")

        faq_id = str(item.get("id", index))
        if not faq_id or faq_id in used_ids:
            faq_id = str(index)
        if faq_id in used_ids:
            faq_id = f"{faq_id}-{index}"
            while faq_id in used_ids:
                faq_id = f"{faq_id}-{index}"
        used_ids.add(faq_id)
        valid_faqs.append({
            "id": faq_id,
            "question": question.strip(),
            "answer": answer.strip(),
        })

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)
    except ValueError:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[str(item["id"]) for item in valid_faqs],
        documents=[f"{item['question']}\n{item['answer']}" for item in valid_faqs],
        metadatas=[{"question": item["question"], "answer": item["answer"]} for item in valid_faqs],
        embeddings=[embed(f"{item['question']} {item['answer']}") for item in valid_faqs],
    )
    return collection.count()


if __name__ == "__main__":
    print(f"Indexed {ingest()} FAQs in {DB_PATH}")
