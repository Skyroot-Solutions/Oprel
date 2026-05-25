from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from oprel.server import db
from oprel.server.services.context import logger
from oprel.server.services.generation import GenerateResult, StreamResult


def _estimate_tokens(text: str, model: str | None = None) -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _truncate_text_to_tokens(text: str, token_budget: int, model: str | None = None) -> str:
    if token_budget <= 0:
        return ""

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")

        token_ids = encoding.encode(text)
        if len(token_ids) <= token_budget:
            return text

        if token_budget < 50:
            return encoding.decode(token_ids[:token_budget])

        keep_head = token_budget // 2
        keep_tail = token_budget - keep_head
        head = encoding.decode(token_ids[:keep_head])
        tail = encoding.decode(token_ids[-keep_tail:])
        return head + "\n\n... (truncated to fit provider budget) ...\n\n" + tail
    except Exception:
        approx_chars = token_budget * 4
        if len(text) <= approx_chars:
            return text
        if approx_chars < 200:
            return text[:max(0, approx_chars)]
        half = approx_chars // 2
        return text[: half - 40] + "\n\n... (truncated to fit provider budget) ...\n\n" + text[-(half - 40) :]


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
        return "\n".join(parts)
    return str(content)


def list_providers() -> list[dict[str, Any]]:
    return db.list_providers()


def get_provider(provider_id: str) -> dict[str, Any] | None:
    return db.get_provider(provider_id)


def upsert_provider(data: dict[str, Any]) -> dict[str, Any]:
    return db.upsert_provider(data)


def delete_provider(provider_id: str) -> dict[str, Any]:
    db.delete_provider(provider_id)
    return {"success": True, "id": provider_id}


async def fetch_provider_models(provider_id: str) -> list[str]:
    p = db.get_provider(provider_id)
    if not p:
        raise KeyError("Provider not found")

    api_key = p.get("api_key")
    base_url = p.get("base_url")
    p_type = p.get("type", "openai")

    presets = {
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "nvidia": "https://integrate.api.nvidia.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }

    url = base_url or presets.get(p_type, "")
    if not url and p_type != "gemini":
        raise ValueError("Base URL is missing")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if p_type == "gemini":
                res = await client.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}")
                res.raise_for_status()
                data = res.json()
                models = [
                    m["name"].replace("models/", "")
                    for m in data.get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                return sorted(models)

            res = await client.get(
                f"{url}/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://oprel.ai",
                    "X-Title": "OPREL",
                },
            )
            res.raise_for_status()
            data = res.json()
            return sorted([m["id"] for m in data.get("data", [])])
        except Exception as exc:
            logger.error(f"Failed to fetch models for {provider_id}: {str(exc)}")
            raise


async def provider_chat_proxy(provider_id: str, body: Any) -> GenerateResult | StreamResult:
    p = db.get_provider(provider_id)
    if not p:
        raise KeyError("Provider not found")

    api_key = p.get("api_key")
    base_url = p.get("base_url")
    p_type = p.get("type", "openai")

    effective_conv_id = body.conversation_id
    if not effective_conv_id:
        title = "New Chat"
        if body.messages:
            first_msg = body.messages[0].get("content", "")
            if isinstance(first_msg, str) and first_msg:
                title = first_msg[:60] + ("..." if len(first_msg) > 60 else "")
        effective_conv_id = db.create_conversation(model_id=body.model, title=title)

    presets = {
        "openai": "https://api.openai.com/v1",
        "nvidia": "https://integrate.api.nvidia.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    url = base_url or presets.get(p_type, "")

    messages = [dict(message) for message in body.messages]

    if body.rag and messages:
        last_user_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None)

        if last_user_index is not None:
            query = str(messages[last_user_index].get("content", ""))
            messages[last_user_index]["_original_content"] = query

            try:
                from oprel.knowledge.knowledge_store import KnowledgeStore
                from oprel.downloader.aliases import resolve_model_id
                from oprel.server.services.generation import get_embeddings, EmbeddingParams

                async def internal_embed(text: str, model: str | None = None) -> list[float]:
                    res = await get_embeddings(EmbeddingParams(input=text, model=resolve_model_id(model or "nomic-embed-text")))
                    return res.embedding or []

                try:
                    from oprel.knowledge.config import TOP_K
                except ImportError:
                    TOP_K = 5

                store = KnowledgeStore(embed_func=internal_embed)
                search_results = await store.search(query, top_k=TOP_K)

                if search_results:
                    provider_model = str(body.model or "")
                    reply_reserve = max(body.max_tokens or 1024, 1024)
                    request_token_budget = 11000 if p_type == "groq" else 12000

                    existing_tokens = sum(
                        _estimate_tokens(_message_content_to_text(message.get("content", "")), provider_model)
                        for message in messages
                    )
                    wrapper_overhead_tokens = 120
                    available_tokens = max(0, request_token_budget - reply_reserve - existing_tokens - wrapper_overhead_tokens)

                    context_parts: list[str] = []
                    used_tokens = 0

                    for i, result in enumerate(search_results):
                        source = result.get("metadata", {}).get("filename", "Unknown source")
                        chunk = f"Source [{i+1}] ({source}):\n{result['text']}"
                        chunk_tokens = _estimate_tokens(chunk, provider_model)

                        if used_tokens + chunk_tokens > available_tokens:
                            remaining = available_tokens - used_tokens
                            if remaining > 50:
                                context_parts.append(_truncate_text_to_tokens(chunk, remaining, provider_model))
                            break

                        context_parts.append(chunk)
                        used_tokens += chunk_tokens + 8

                    if context_parts:
                        context_text = "\n\n".join(context_parts)
                        messages[last_user_index]["content"] = (
                            "CONTEXT FROM LOCAL KNOWLEDGE BASE:\n"
                            "----------------------------------------\n"
                            f"{context_text}\n"
                            "----------------------------------------\n\n"
                            f"QUESTION: {query}\n\n"
                            "INSTRUCTION: Use ONLY the provided context above to answer. "
                            "Cite source labels [1], [2], etc. If the answer isn't firmly supported by the context, "
                            "state that you don't have enough information."
                        )
                        logger.info(f"Provider RAG: Injected {len(context_parts)} chunks, ~{used_tokens} tokens")
            except Exception as exc:
                logger.error(f"Provider RAG search failed: {exc}")

    if p_type == "groq" and messages:
        provider_model = str(body.model or "")
        request_token_budget = 11000
        reply_reserve = max(body.max_tokens or 1024, 1024)
        wrapper_overhead_tokens = 120

        def compute_prompt_tokens(items: list[dict[str, Any]]) -> int:
            return sum(_estimate_tokens(_message_content_to_text(item.get("content", "")), provider_model) for item in items)

        total_tokens = compute_prompt_tokens(messages)
        allowed_prompt_tokens = max(0, request_token_budget - reply_reserve - wrapper_overhead_tokens)

        if total_tokens > allowed_prompt_tokens:
            trimmed_messages: list[dict[str, Any]] = []
            running_tokens = 0

            for message in messages[:-1]:
                message_text = _message_content_to_text(message.get("content", ""))
                message_tokens = _estimate_tokens(message_text, provider_model)

                if running_tokens + message_tokens > allowed_prompt_tokens:
                    remaining = allowed_prompt_tokens - running_tokens
                    if remaining <= 0:
                        continue
                    trimmed_messages.append({**message, "content": _truncate_text_to_tokens(message_text, remaining, provider_model)})
                    running_tokens = allowed_prompt_tokens
                    break

                trimmed_messages.append(message)
                running_tokens += message_tokens

            last_message = messages[-1]
            last_text = _message_content_to_text(last_message.get("content", ""))
            last_budget = max(50, allowed_prompt_tokens - running_tokens)
            trimmed_messages.append({**last_message, "content": _truncate_text_to_tokens(last_text, last_budget, provider_model)})

            messages = trimmed_messages
            logger.warning(
                f"Provider {p_type}: trimmed prompt from ~{total_tokens} to ~{compute_prompt_tokens(messages)} tokens to fit request budget"
            )

    user_msg = messages[-1] if messages else None
    if user_msg:
        db.add_message(effective_conv_id, user_msg["role"], user_msg.get("_original_content", user_msg["content"]))

    if not body.stream:
        async with httpx.AsyncClient(timeout=60.0) as client:
            full_response = ""

            if p_type == "gemini":
                model_name = body.model if body.model.startswith("models/") else f"models/{body.model}"
                system_msg = next((message for message in messages if message["role"] == "system"), None)
                contents = []
                use_system_instruction = "gemma" not in body.model.lower()

                for index, message in enumerate(messages):
                    if message["role"] == "system":
                        continue
                    role = "model" if message["role"] == "assistant" else "user"
                    content_text = str(message["content"])
                    if not use_system_instruction and system_msg and index == 1:
                        content_text = f"{system_msg['content']}\n\n{content_text}"
                    contents.append({"role": role, "parts": [{"text": content_text}]})

                gemini_body = {
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": body.max_tokens or 4096,
                        "temperature": body.temperature if body.temperature is not None else 0.7,
                    },
                }
                if use_system_instruction and system_msg:
                    gemini_body["systemInstruction"] = {"parts": [{"text": str(system_msg["content"])}]}

                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}",
                    json=gemini_body,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Gemini API Error: {resp.text}")

                data = resp.json()
                try:
                    full_response = data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    full_response = "Error parsing Gemini response"
            else:
                clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
                resp = await client.post(
                    f"{url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": body.model,
                        "messages": clean_messages,
                        "stream": False,
                        "max_tokens": body.max_tokens,
                        "temperature": body.temperature,
                    },
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Provider {p_type} Error: {resp.text}")

                data = resp.json()
                full_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if full_response.strip():
                db.add_message(effective_conv_id, "assistant", full_response)
                db.add_inference_log(
                    model_id=body.model,
                    prompt_tokens=_estimate_tokens(_message_content_to_text(messages[-1]["content"] if messages else ""), body.model),
                    completion_tokens=_estimate_tokens(full_response, body.model),
                    latency_ms=100.0,
                    tps=0.0,
                )

            return GenerateResult(
                text=full_response,
                model_id=body.model,
                conversation_id=effective_conv_id,
                message_count=len(messages) + 1,
            )

    async def stream_generator(conv_id: str) -> AsyncIterator[str]:
        full_response = ""

        async with httpx.AsyncClient(timeout=60.0) as client:
            if p_type == "gemini":
                model_name = body.model if body.model.startswith("models/") else f"models/{body.model}"
                system_msg = next((message for message in messages if message["role"] == "system"), None)
                contents = []
                use_system_instruction = "gemma" not in body.model.lower()

                for index, message in enumerate(messages):
                    if message["role"] == "system":
                        continue
                    role = "model" if message["role"] == "assistant" else "user"
                    content_text = str(message["content"])
                    if not use_system_instruction and system_msg and index == 1:
                        content_text = f"{system_msg['content']}\n\n{content_text}"
                    contents.append({"role": role, "parts": [{"text": content_text}]})

                gemini_body = {
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": body.max_tokens or 4096,
                        "temperature": body.temperature if body.temperature is not None else 0.7,
                    },
                }
                if use_system_instruction and system_msg:
                    gemini_body["systemInstruction"] = {"parts": [{"text": str(system_msg["content"])}]}

                try:
                    async with client.stream(
                        "POST",
                        f"https://generativelanguage.googleapis.com/v1beta/{model_name}:streamGenerateContent?key={api_key}&alt=sse",
                        json=gemini_body,
                    ) as resp:
                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            error_msg = f"Gemini API Error {resp.status_code}: {err_body.decode()}"
                            yield f"data: {json.dumps({'error': error_msg})}\n\n"
                            return

                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                try:
                                    json_data = json.loads(line[6:])
                                    token = (
                                        json_data.get("candidates", [{}])[0]
                                        .get("content", {})
                                        .get("parts", [{}])[0]
                                        .get("text", "")
                                    )
                                    if token:
                                        full_response += token
                                except Exception:
                                    pass
                            yield line + "\n"
                except Exception as exc:
                    yield f"data: {json.dumps({'error': f'Streaming error: {str(exc)}'})}\n\n"
            else:
                clean_messages = [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]
                async with client.stream(
                    "POST",
                    f"{url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": body.model,
                        "messages": clean_messages,
                        "stream": True,
                        "max_tokens": body.max_tokens,
                        "temperature": body.temperature,
                    },
                ) as resp:
                    if resp.status_code not in (200, 206):
                        err_body = await resp.aread()
                        error_msg = f"Provider {p_type} error {resp.status_code}: {err_body.decode()}"
                        logger.error(f"Streaming provider error: {error_msg}")
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                chunk = json.loads(line[6:])
                                token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if token:
                                    full_response += token
                            except Exception:
                                pass
                        yield line + "\n"

        if full_response.strip():
            db.add_message(conv_id, "assistant", full_response)
            db.add_inference_log(
                model_id=body.model,
                prompt_tokens=_estimate_tokens(_message_content_to_text(messages[-1]["content"] if messages else ""), body.model),
                completion_tokens=_estimate_tokens(full_response, body.model),
                latency_ms=100.0,
                tps=0.0,
            )

    return StreamResult(iterator=stream_generator(effective_conv_id), conversation_id=effective_conv_id)