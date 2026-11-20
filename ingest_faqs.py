import hashlib, json, re
from pathlib import Path
import chromadb

ROOT = Path(__file__).resolve().parent
FAQ_PATH = ROOT / "faqs.json"
DB_PATH = ROOT / ".chroma"
COLLECTION = "it_helpdesk_faqs"
DIMS = 256

def embed(text: str) -> list[float]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    vec = [0.0] * DIMS
    for f in feats:
        n = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
        vec[n % DIMS] += 1.0 if n & 256 else -1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]

def ingest() -> int:
    faqs = json.loads(FAQ_PATH.read_text(encoding="utf-8"))
    if not isinstance(faqs, list) or not faqs:
        raise ValueError("faqs.json must be a non-empty list")
    valid, used = [], set()
    for i, item in enumerate(faqs):
        q, a = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
        if not q or not a:
            raise ValueError(f"Invalid FAQ at index {i}")
        fid = str(item.get("id", i))
        while fid in used:
            fid = f"{fid}-{i}"
        used.add(fid)
        valid.append({"id": fid, "question": q, "answer": a})

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)
    except ValueError:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[x["id"] for x in valid],
        documents=[f"{x['question']}\n{x['answer']}" for x in valid],
        metadatas=[{"question": x["question"], "answer": x["answer"]} for x in valid],
        embeddings=[embed(f"{x['question']} {x['answer']}") for x in valid],
    )
    return col.count()

if __name__ == "__main__":
    print(f"Indexed {ingest()} FAQs in {DB_PATH}")