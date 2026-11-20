# Alish IT Helpdesk

FastAPI + WebSocket IT helpdesk assistant featuring streaming replies, per-connection memory, local ChromaDB/lexical FAQ search (`faqs.json`), and mock ticket creation (`MOCK-...`).

## Setup (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env