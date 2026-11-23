# Alish — IT Helpdesk Assistant

FastAPI + WebSocket IT helpdesk assistant with ChromaDB FAQ retrieval, session memory, and a mock `create_ticket(issue, priority)` tool.

## Setup & Run

1. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set `GROQ_API_KEY` (plus optional `GROQ_MODEL`):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
3. Ingest FAQs into ChromaDB:
   ```bash
   python ingest_faqs.py
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
5. Open the web UI in your browser at `http://localhost:8000`.