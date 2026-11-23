import json, re
from pathlib import Path

SYSTEM_PROMPT = (
    "You are Alish, an IT Helpdesk Assistant for VPN, password, and software support. "
    "Answer concisely using only approved FAQs; if none match, direct the user to IT via approved channels. "
    "Never ask for passwords/MFA codes, bypass security/admin controls, invent URLs, or claim a real ticket was created."
)

STOP = {"a","an","and","are","can","do","for","how","i","in","is","it","me","my","of","on","the","to","what","when","where","why","with","you"}

def _words(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if w not in STOP and len(w) > 2}

def load_faqs() -> list[dict]:
    p = Path(__file__).with_name("faqs.json")
    try:
        items = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        return [{"question": x["question"].strip(), "answer": x["answer"].strip()} for x in items if x.get("question") and x.get("answer")]
    except Exception:
        return []

def find_local_faq(question: str) -> dict | None:
    faqs, qw = load_faqs(), _words(question)
    if not faqs or not qw:
        return None
    best = max(faqs, key=lambda f: len(qw & _words(f["question"])))
    if len(qw & _words(best["question"])) >= 2:
        return best
    try:
        import chromadb
        from ingest_faqs import DB_PATH, COLLECTION, embed
        res = chromadb.PersistentClient(path=str(DB_PATH)).get_collection(COLLECTION).query(query_embeddings=[embed(question)], n_results=1)
        if res["ids"][0] and (not res["distances"][0] or res["distances"][0][0] <= 0.72):
            m = res["metadatas"][0][0]
            return {"question": m["question"], "answer": m["answer"]}
    except Exception:
        pass
    return None