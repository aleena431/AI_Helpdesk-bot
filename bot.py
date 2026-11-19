"""Interactive IT helpdesk assistant using the local FAQ as the source of truth."""

import json
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv() -> bool:
        return False

try:
    from groq import Groq
except ImportError:  # pragma: no cover - optional dependency
    Groq = None

load_dotenv()

SYSTEM_PROMPT = """You are Alish, an IT Helpdesk Assistant.
Help employees with VPN access, password resets, and software setup.
Use the supplied approved FAQ as your source for support procedures. Keep the
reply concise and do not add steps that are absent from the FAQ. If no FAQ is
provided or it does not answer the question, say so and direct the user to IT
through their organization's approved channel.
Never ask users to share passwords, multi-factor codes, or recovery codes.
Never recommend disabling or changing firewalls, antivirus, endpoint
protection, or other security controls; bypassing admin controls; clearing,
deleting, or modifying saved credentials; or installing unapproved software.
Do not invent company-specific URLs, server addresses, tools, or procedures.
Do not claim that a real ticket was created or that you contacted IT. Only say a ticket was created after the running application confirms that action.
"""

STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "for", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "the", "to", "what", "when",
    "where", "why", "with", "you",
}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def useful_words(text: str) -> set[str]:
    return {
        word
        for word in normalize_text(text).split()
        if word and word not in STOP_WORDS
    }


def load_faqs() -> list[dict]:
    faq_path = Path(__file__).with_name("faqs.json")
    if not faq_path.exists():
        return []

    try:
        with faq_path.open("r", encoding="utf-8") as faq_file:
            items = json.load(faq_file)
    except (OSError, json.JSONDecodeError, TypeError):
        return []

    if not isinstance(items, list):
        return []

    valid_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            valid_items.append({"question": question, "answer": answer})
    return valid_items


def find_local_faq(question: str) -> dict | None:
    """Retrieve the closest approved FAQ from the persistent local vector index."""
    try:
        import chromadb
        from ingest_faqs import DB_PATH, COLLECTION, embed

        client = chromadb.PersistentClient(path=str(DB_PATH))
        collection = client.get_collection(COLLECTION)
        result = collection.query(query_embeddings=[embed(question)], n_results=1)
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        if ids and metadatas and (not distances or distances[0] <= 0.72):
            item = metadatas[0]
            return {"question": item["question"], "answer": item["answer"]}
    except Exception:
        # Keep the bot usable before `python ingest_faqs.py` has been run.
        pass

    faqs = load_faqs()
    if not faqs:
        return None

    normalized_question = normalize_text(question)
    question_words = useful_words(normalized_question)
    if not question_words:
        return None

    best_faq = None
    best_score = -1

    for faq in faqs:
        faq_question = str(faq.get("question", ""))
        faq_words = useful_words(faq_question)
        score = len(question_words & faq_words)

        if normalized_question and normalized_question in normalize_text(faq_question):
            score += 5

        if score > best_score:
            best_faq = faq
            best_score = score

    return best_faq if best_score > 0 else None


def fallback_answer(question: str) -> str:
    return (
        "I could not find a matching approved FAQ for your question. "
        "Please contact IT through your organization's approved channel."
    )


def answer_with_faq(question: str) -> str:
    faq = find_local_faq(question)
    if faq:
        return str(faq.get("answer", fallback_answer(question)))
    return fallback_answer(question)


def answer_with_groq(question: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "put_your_groq_api_key_here":
        return answer_with_faq(question)

    if Groq is None:
        return answer_with_faq(question)

    faq = find_local_faq(question)
    faq_context = (
        f"Approved FAQ question: {faq['question']}\nApproved answer: {faq['answer']}"
        if faq
        else "No matching approved FAQ was found."
    )

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Employee question: {question}\n\n{faq_context}"},
            ],
            temperature=0.2,
            max_tokens=250,
        )
    except Exception:
        return answer_with_faq(question)

    if not completion or not getattr(completion, "choices", None):
        return answer_with_faq(question)

    message = completion.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    return answer_with_faq(question)


def handle_question(question: str) -> str:
    faq = find_local_faq(question)
    if faq:
        return str(faq["answer"])
    return answer_with_groq(question)


def interactive_chat() -> None:
    print("Alish: Hello! I can help with VPN, password, and software issues.")
    print("Type 'exit' or 'quit' to end the conversation.")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAlish: Goodbye.")
            break

        if not question:
            print("Alish: Please enter your question.")
            continue

        if question.lower() in {"exit", "quit"}:
            print("Alish: Goodbye.")
            break

        answer = handle_question(question)
        print(f"Alish: {answer}")


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"You: {question}")
        print(f"Alish: {handle_question(question)}")
        return

    interactive_chat()


if __name__ == "__main__":
    main()
