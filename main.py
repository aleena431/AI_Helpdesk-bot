"""FastAPI WebSocket chat endpoint with per-connection memory and safe streaming."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from groq import AsyncGroq

from bot import SYSTEM_PROMPT, find_local_faq

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = AsyncGroq(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="Alish IT Helpdesk Assistant")


def fallback_answer(question: str) -> str:
    faq = find_local_faq(question)
    if faq:
        return str(faq.get("answer", "I could not find a matching FAQ answer."))
    return (
        "I could not find a matching approved FAQ for your question. "
        "Please contact IT through your organization's approved channel."
    )


async def send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "content": message})
    except RuntimeError:
        logger.warning("Failed to send error message after WebSocket disconnect.")


async def safe_stream_answer(websocket: WebSocket, model: str, messages: list[dict[str, str]]) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    answer_parts: list[str] = []
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if websocket.application_state.name != "CONNECTED":
                raise WebSocketDisconnect()

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            token = getattr(delta, "content", None)
            if not isinstance(token, str) or not token:
                continue

            answer_parts.append(token)
            try:
                await websocket.send_json({"type": "token", "content": token})
            except RuntimeError:
                raise WebSocketDisconnect()

    except (RuntimeError, WebSocketDisconnect):
        raise
    except Exception:
        logger.exception("Groq streaming request failed")
        raise

    return "".join(answer_parts)


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    if client is None:
        await send_error(websocket, "Set GROQ_API_KEY in the local .env file.")
        await websocket.close(code=1011)
        return

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info("Helpdesk chat disconnected; in-memory history discarded.")
                return
            except Exception:
                await send_error(websocket, "Send valid JSON in the format {\"message\": \"your question\"}.")
                continue

            if not isinstance(payload, dict):
                await send_error(websocket, "Send a JSON object with a non-empty 'message' field.")
                continue

            question = payload.get("message")
            if not isinstance(question, str) or not question.strip():
                await send_error(websocket, "Send a non-empty message field.")
                continue

            question = question.strip()
            if len(question) > 2000:
                await send_error(websocket, "Keep each message under 2,000 characters.")
                continue

            faq = find_local_faq(question)
            if faq:
                user_content = (
                    f"Employee question: {question}\n\n"
                    f"Approved FAQ question: {faq['question']}\n"
                    f"Approved answer: {faq['answer']}"
                )
            else:
                user_content = f"Employee question: {question}\n\nNo matching approved FAQ was found."

            messages.append({"role": "user", "content": user_content})

            try:
                response_text = await safe_stream_answer(websocket, MODEL, messages)
            except WebSocketDisconnect:
                logger.info("Client disconnected while receiving response.")
                return
            except Exception:
                messages.pop()
                await send_error(websocket, "The assistant could not complete the request.")
                continue

            if response_text:
                messages.append({"role": "assistant", "content": response_text})
            else:
                fallback = fallback_answer(question)
                messages.append({"role": "assistant", "content": fallback})
                await websocket.send_json({"type": "token", "content": fallback})

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        logger.info("Helpdesk chat disconnected; in-memory history discarded.")

