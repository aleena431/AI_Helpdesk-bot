# Alish IT Helpdesk

A small FastAPI and WebSocket helpdesk demo. The browser chat streams replies, keeps a short memory for the active connection, searches approved IT FAQs, and can create an in-memory mock ticket. Tickets are not sent to a real IT system.

## Requirements

- Python 3.10 or newer
- A Groq API key for the WebSocket chat and model tool calling

## Setup (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and set `GROQ_API_KEY` to your key. `GROQ_MODEL` is optional; the default is `openai/gpt-oss-20b`.

Build or refresh the local ChromaDB index, then start the app:

```powershell
python ingest_faqs.py
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The app serves `index.html` and the WebSocket endpoint at `/ws/chat` on the same origin. Keep one browser connection open while demonstrating memory; choose **New chat** to reset it.

For a short FAQ-only WebSocket check, run this in another terminal while the server is running:

```powershell
python test_client.py --faq-only
```

## Three-minute live demo

**0:00–0:30 — Open the app.** Show the connected status and explain that the chat uses one WebSocket connection, so recent turns stay in context.

**0:30–1:15 — FAQ search.** Send: `Search the approved FAQ: Can I share my password with the helpdesk?` Show the `search_faq` tool call, the retrieved FAQ result, and Alish's policy-based answer.

**1:15–2:15 — Tool calling.** Send: `My VPN keeps disconnecting. Please create a high-priority ticket.` Show `create_ticket` and its `MOCK-...` result. Point out the UI disclaimer: the mock ticket is only in memory.

**2:15–3:00 — Conversation memory.** Send: `What was the issue in the ticket I just created?` The assistant should refer to the VPN disconnection from the previous turn. Finish by clicking **New chat** to show that a new connection resets that memory.

## FAQ data and retrieval

The 15 approved questions and answers are in `faqs.json`. `python ingest_faqs.py` stores local vectors in `.chroma/` using deterministic feature-hash embeddings; it does not download an embedding model. The database is local and ignored by Git. The bot also has a lexical FAQ fallback if ChromaDB is unavailable.

## Request size and privacy

The WebSocket limits each message to 2,000 characters, bounds recent history to six messages, trims message content, and drops older turns when the serialized prompt is larger than the 5,600-byte budget. Groq output is capped at 600 tokens, leaving headroom under a 7,000-token target for request framing. This is a conservative per-call cap; Groq TPM is shared across requests in a rolling minute, so rapid repeated calls can still exceed the organization limit. Avoid putting secrets or personal credentials into chat.

## Troubleshooting

- **Disconnected / missing API key:** check that `.env` has `GROQ_API_KEY=...`, then restart Uvicorn.
- **FAQ not found:** run `python ingest_faqs.py` and confirm `.chroma/` was created.
- **Browser won't connect:** open the app at `http://127.0.0.1:8000` rather than opening `index.html` directly; the UI needs the same-origin WebSocket endpoint.
- **Groq request fails:** check the selected model, API key, and your organization's current Groq limits.
