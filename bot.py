"""A small IT helpdesk assistant with a local FAQ mode."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = """You are Alina Khalid, an IT Helpdesk Assistant.
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
Do not claim to create a ticket or contact IT.
"""

STOP_WORDS = {
    "a", "an", "and", "are", "can", "do", "for", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "the", "to", "what", "when",
    "where", "why", "with", "you",
}


def useful_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOP_WORDS
    }


def find_local_faq(question: str) -> dict | None:
    faq_path = Path(__file__).with_name("faqs.json")
    with faq_path.open(encoding="utf-8") as faq_file:
        faqs = json.load(faq_file)

    question_words = useful_words(question)
    best_faq = None
    best_score = 0
    for faq in faqs:
        faq_words = useful_words(faq["question"])
        score = len(question_words & faq_words)
        if score > best_score:
            best_faq = faq
            best_score = score
    return best_faq


def main() -> None:
    question = "How do I troubleshoot a VPN that will not connect?"
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or api_key == "put_your_groq_api_key_here":
        print("Alina (local FAQ mode; no Groq API key configured)")
        faq = find_local_faq(question)
        if faq:
            print(f"You: {question}")
            print(f"Alina: {faq['answer']}")
        else:
            print("Alina: I could not find a matching FAQ. Please contact IT.")
        return

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    client = Groq(api_key=api_key)
    faq = find_local_faq(question)
    if faq:
        faq_context = f"Approved FAQ question: {faq['question']}\nApproved answer: {faq['answer']}"
    else:
        faq_context = "No matching approved FAQ was found."

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Employee question: {question}\n\n{faq_context}",
            },
        ],
    )

    answer = completion.choices[0].message.content or "I could not generate a reply."
    print(f"You: {question}")
    print(f"Alina: {answer}")


if __name__ == "__main__":
    main()
