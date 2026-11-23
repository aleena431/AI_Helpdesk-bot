# Alish — IT Helpdesk Assistant

An IT Helpdesk Assistant built with [FastAPI](https://fastapi.tiangolo.com/) and WebSockets for VPN, password, and software support. It uses local FAQ lookup (`faqs.json`) and [Chroma](https://www.trychroma.com/) vector search (`ingest_faqs.py`), maintains per-session conversation memory, and supports a mock `create_ticket(issue, priority)` tool.

## Project Structure

- `main.py` — [FastAPI](https://fastapi.tiangolo.com/) application, WebSocket chat endpoint (`/ws/chat`), session memory, and tool calling (`search_faq`, `create_ticket`).
- `bot.py` — System prompt and FAQ matching logic.
- `ingest_faqs.py` — Loads `faqs.json` and indexes embeddings into `.chroma/`.
- `faqs.json` — Local IT helpdesk FAQ entries.
- `index.html` — Browser chat interface served at `/`.
- `.env.example` — Environment variable template.
- `requirements.txt` — Python dependencies.

## Setup & Installation

1. Create and activate a virtual environment (Windows PowerShell):

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Environment Configuration

1. Copy `.env.example` to `.env`:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` and set your [Groq](https://console.groq.com/) API key (and optional model name):

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```

Do not commit `.env` to Git.

## Ingest FAQs & Run the Server

1. Ingest the FAQs into [Chroma](https://www.trychroma.com/):

   ```bash
   python ingest_faqs.py
   ```

2. Start the [FastAPI](https://fastapi.tiangolo.com/) server with [Uvicorn](https://www.uvicorn.org/):

   ```bash
   uvicorn main:app --reload
   ```

3. Open the web UI in your browser:

   ```text
   http://localhost:8000
   ```

## Features & Notes

- **Streaming & Memory:** Streams responses over WebSocket (`/ws/chat`) while keeping recent messages in the active session.
- **FAQ Retrieval:** Searches `faqs.json` and falls back to the `.chroma/` collection when needed.
- **Mock Ticketing:** `create_ticket(issue, priority)` creates in-memory mock tickets (`MOCK-1001`, etc.) for demonstration only and does not submit tickets to a real IT system.