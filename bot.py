import json, os, re, sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    Groq = None

SYSTEM_PROMPT = (
    "You are Alish, an IT Helpdesk Assistant for VPN, password, and software support. "
    "Answer concisely using only supplied approved FAQs; if none match, direct the user to IT via approved channels. "
    "Never ask for passwords/MFA codes, recommend disabling security/admin controls, clear saved credentials, "
    "invent internal URLs/tools, or claim a real ticket was created unless confirmed by the app."
)

STOP = {"a","an","and","are","can","do","for","how","i","in","is","it","me","my","of","on","the","to","what","when","where","why","with","you"}

def _words(text: str) -> set[str]:
    out = set()
    for w in re.sub(r"[^a-z0-9]+", " ", text.lower()).split():
        if w in STOP: continue
        if w.endswith("ing") and len(w) > 4: w = w[:-3]
        elif w.endswith("ies") and len(w) > 4: w = w[:-3] + "y"
        elif w.endswith("s") and len(w) > 3: w = w[:-1]
        out.add(w)
    return out

def load_faqs() -> list[dict]:
    p = Path(__file__).with_name("faqs.json")
    try:
        items = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        return [{"question": x["question"].strip(), "answer": x["answer"].strip()} for x in items if isinstance(x, dict) and x.get("question") and x.get("answer")]
    except Exception:
        return []

def find_local_faq(question: str) -> dict | None:
    faqs = load_faqs()
    nq, qw = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip(), _words(question)
    if not faqs or not qw: return None
    best, score = None, -1
    for f in faqs:
        s = len(qw & _words(f["question"])) + (5 if nq in re.sub(r"[^a-z0-9]+", " ", f["question"].lower()) else 0)
        if s > score: best, score = f, s
    if score >= 2: return best
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

def fallback_answer(q: str = "") -> str:
    return "No matching approved FAQ found. Please contact IT through your organization's approved channel."

def answer_with_faq(q: str) -> str:
    f = find_local_faq(q)
    return f["answer"] if f else fallback_answer(q)

def answer_with_groq(q: str) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "put_your_groq_api_key_here" or Groq is None:
        return answer_with_faq(q)
    f = find_local_faq(q)
    ctx = f"FAQ: {f['question']}\nAnswer: {f['answer']}" if f else "No matching FAQ."
    try:
        c = Groq(api_key=key).chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Question: {q}\n\n{ctx}"}],
            temperature=0.2, max_tokens=250,
        )
        txt = (c.choices[0].message.content or "").strip()
        return txt or answer_with_faq(q)
    except Exception:
        return answer_with_faq(q)

def handle_question(q: str) -> str:
    f = find_local_faq(q)
    return f["answer"] if f else answer_with_groq(q)

def main() -> None:
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"You: {q}\nAlish: {handle_question(q)}")
        return
    print("Alish: Ask about VPN, passwords, or software ('exit' to quit).")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q: continue
        if q.lower() in {"exit", "quit"}: break
        print(f"Alish: {handle_question(q)}")

if __name__ == "__main__":
    main()