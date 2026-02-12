"""Provider adapter interface and concrete adapters."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import AsyncIterator

import httpx

from attractor.llm.errors import (
    AuthenticationError,
    ConfigurationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    NetworkError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from attractor.llm.types import (
    ContentKind,
    ContentPart,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCallData,
    ToolChoice,
    Usage,
)


class ProviderAdapter(ABC):
    """Contract every provider must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def complete(self, request: Request) -> Response:
        ...

    @abstractmethod
    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        ...
        yield  # type: ignore[misc]

    async def close(self) -> None:
        pass

    async def initialize(self) -> None:
        pass

    def supports_tool_choice(self, mode: str) -> bool:
        return True


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(ProviderAdapter):
    """Adapter for the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/")
        self.default_max_tokens = default_max_tokens
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "anthropic"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and self._client.is_closed:
            self._client = None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        return self._client

    async def complete(self, request: Request) -> Response:
        client = await self._get_client()
        body = self._translate_request(request)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        resp = await client.post(f"{self.base_url}/v1/messages", json=body, headers=headers)
        if resp.status_code >= 400:
            self._raise_error(resp)
        return self._translate_response(resp.json(), request.model)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        client = await self._get_client()
        body = self._translate_request(request)
        body["stream"] = True
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with client.stream("POST", f"{self.base_url}/v1/messages", json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                data = await resp.aread()
                self._raise_error_from_bytes(resp.status_code, data)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                event = self._translate_stream_event(payload)
                if event:
                    yield event

    # -- translation helpers ---------------------------------------------------

    def _translate_request(self, req: Request) -> dict:
        system_parts = [m.text for m in req.messages if m.role == Role.SYSTEM]
        system_text = "\n".join(system_parts) if system_parts else None

        messages = []
        for m in req.messages:
            if m.role in (Role.SYSTEM, Role.DEVELOPER):
                continue
            messages.append(self._msg_to_api(m))

        body: dict = {"model": req.model, "messages": messages, "max_tokens": req.max_tokens or self.default_max_tokens}
        if system_text:
            body["system"] = system_text
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop_sequences:
            body["stop_sequences"] = req.stop_sequences
        if req.tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in req.tools
            ]
        if req.tool_choice:
            body["tool_choice"] = self._map_tool_choice(req.tool_choice)
        return body

    def _msg_to_api(self, m: Message) -> dict:
        role = "user" if m.role == Role.USER else "assistant" if m.role == Role.ASSISTANT else "user"
        content: list[dict] = []
        for part in m.content:
            if part.kind == ContentKind.TEXT and part.text:
                content.append({"type": "text", "text": part.text})
            elif part.kind == ContentKind.TOOL_CALL and part.tool_call:
                tc = part.tool_call
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments),
                })
            elif part.kind == ContentKind.TOOL_RESULT and part.tool_result:
                tr = part.tool_result
                return {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tr.tool_call_id,
                                 "content": str(tr.content), "is_error": tr.is_error}],
                }
            elif part.kind == ContentKind.IMAGE and part.image:
                img = part.image
                if img.data:
                    import base64
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": img.media_type or "image/png",
                                   "data": base64.b64encode(img.data).decode()},
                    })
                elif img.url:
                    content.append({"type": "image", "source": {"type": "url", "url": img.url}})
        if not content:
            content = [{"type": "text", "text": ""}]
        return {"role": role, "content": content}

    @staticmethod
    def _map_tool_choice(tc: ToolChoice) -> dict:
        if tc.mode == "auto":
            return {"type": "auto"}
        if tc.mode == "none":
            return {"type": "none"}
        if tc.mode == "required":
            return {"type": "any"}
        if tc.mode == "named" and tc.tool_name:
            return {"type": "tool", "name": tc.tool_name}
        return {"type": "auto"}

    def _translate_response(self, data: dict, model: str) -> Response:
        content_parts: list[ContentPart] = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content_parts.append(ContentPart(kind=ContentKind.TEXT, text=block["text"]))
            elif block["type"] == "tool_use":
                content_parts.append(ContentPart(
                    kind=ContentKind.TOOL_CALL,
                    tool_call=ToolCallData(id=block["id"], name=block["name"], arguments=block.get("input", {})),
                ))
            elif block["type"] == "thinking":
                from attractor.llm.types import ThinkingData
                content_parts.append(ContentPart(
                    kind=ContentKind.THINKING,
                    thinking=ThinkingData(text=block.get("thinking", ""), signature=block.get("signature")),
                ))

        message = Message(role=Role.ASSISTANT, content=content_parts)
        stop = data.get("stop_reason", "end_turn")
        reason_map = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "stop_sequence": "stop"}
        usage_data = data.get("usage", {})

        return Response(
            id=data.get("id", ""),
            model=model,
            provider="anthropic",
            message=message,
            finish_reason=FinishReason(reason=reason_map.get(stop, "other"), raw=stop),
            usage=Usage(
                input_tokens=usage_data.get("input_tokens", 0),
                output_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
                cache_read_tokens=usage_data.get("cache_read_input_tokens"),
                cache_write_tokens=usage_data.get("cache_creation_input_tokens"),
            ),
            raw=data,
        )

    def _translate_stream_event(self, payload: dict) -> StreamEvent | None:
        etype = payload.get("type", "")
        if etype == "content_block_delta":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                return StreamEvent(type=StreamEventType.TEXT_DELTA, delta=delta.get("text", ""))
            if delta.get("type") == "input_json_delta":
                return StreamEvent(type=StreamEventType.TOOL_CALL_DELTA, delta=delta.get("partial_json", ""))
        if etype == "message_stop":
            return StreamEvent(type=StreamEventType.FINISH)
        return None

    def _raise_error(self, resp: httpx.Response) -> None:
        try:
            data = resp.json()
        except Exception:
            data = {"error": {"message": resp.text}}
        msg = data.get("error", {}).get("message", resp.text)
        self._raise_for_status(resp.status_code, msg, data)

    def _raise_error_from_bytes(self, status: int, raw: bytes) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        msg = data.get("error", {}).get("message", raw.decode(errors="replace"))
        self._raise_for_status(status, msg, data)

    def _raise_for_status(self, status: int, msg: str, raw: dict) -> None:
        kwargs = {"message": msg, "provider": "anthropic", "status_code": status, "raw": raw}
        if status == 401:
            raise AuthenticationError(**kwargs)
        if status == 403:
            from attractor.llm.errors import AccessDeniedError
            raise AccessDeniedError(**kwargs)
        if status == 404:
            raise NotFoundError(**kwargs)
        if status in (400, 422):
            raise InvalidRequestError(**kwargs)
        if status == 429:
            raise RateLimitError(**kwargs, retryable=True)
        if 500 <= status < 600:
            raise ServerError(**kwargs, retryable=True)
        raise ProviderError(**kwargs)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter(ProviderAdapter):
    """Adapter for the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        org_id: str | None = None,
        default_max_tokens: int = 4096,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com")).rstrip("/")
        self.org_id = org_id or os.getenv("OPENAI_ORG_ID")
        self.default_max_tokens = default_max_tokens
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "openai"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and self._client.is_closed:
            self._client = None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        return self._client

    async def complete(self, request: Request) -> Response:
        client = await self._get_client()
        body = self._translate_request(request)
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id
        resp = await client.post(f"{self.base_url}/v1/chat/completions", json=body, headers=headers)
        if resp.status_code >= 400:
            self._raise_error(resp)
        return self._translate_response(resp.json(), request.model)

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        client = await self._get_client()
        body = self._translate_request(request)
        body["stream"] = True
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id
        async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                data = await resp.aread()
                self._raise_error_from_bytes(resp.status_code, data)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    yield StreamEvent(type=StreamEventType.FINISH)
                    break
                payload = json.loads(raw)
                event = self._translate_stream_event(payload)
                if event:
                    yield event

    def _translate_request(self, req: Request) -> dict:
        messages = []
        for m in req.messages:
            if m.role == Role.SYSTEM:
                messages.append({"role": "system", "content": m.text})
            elif m.role == Role.USER:
                messages.append({"role": "user", "content": m.text})
            elif m.role == Role.ASSISTANT:
                msg: dict = {"role": "assistant"}
                text_parts = [p.text for p in m.content if p.kind == ContentKind.TEXT and p.text]
                tool_calls_api = []
                for p in m.content:
                    if p.kind == ContentKind.TOOL_CALL and p.tool_call:
                        tc = p.tool_call
                        tool_calls_api.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                            },
                        })
                if text_parts:
                    msg["content"] = "\n".join(text_parts)
                if tool_calls_api:
                    msg["tool_calls"] = tool_calls_api
                messages.append(msg)
            elif m.role == Role.TOOL:
                for p in m.content:
                    if p.kind == ContentKind.TOOL_RESULT and p.tool_result:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": p.tool_result.tool_call_id,
                            "content": str(p.tool_result.content),
                        })

        body: dict = {"model": req.model, "messages": messages}
        if req.max_tokens:
            body["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop_sequences:
            body["stop"] = req.stop_sequences
        if req.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in req.tools
            ]
        if req.tool_choice:
            tc = req.tool_choice
            if tc.mode == "auto":
                body["tool_choice"] = "auto"
            elif tc.mode == "none":
                body["tool_choice"] = "none"
            elif tc.mode == "required":
                body["tool_choice"] = "required"
            elif tc.mode == "named" and tc.tool_name:
                body["tool_choice"] = {"type": "function", "function": {"name": tc.tool_name}}
        return body

    def _translate_response(self, data: dict, model: str) -> Response:
        choice = data["choices"][0]
        msg = choice["message"]

        content_parts: list[ContentPart] = []
        if msg.get("content"):
            content_parts.append(ContentPart(kind=ContentKind.TEXT, text=msg["content"]))
        for tc in msg.get("tool_calls", []):
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = fn["arguments"]
            content_parts.append(ContentPart(
                kind=ContentKind.TOOL_CALL,
                tool_call=ToolCallData(id=tc["id"], name=fn["name"], arguments=args),
            ))

        message = Message(role=Role.ASSISTANT, content=content_parts)
        reason_raw = choice.get("finish_reason", "stop")
        reason_map = {"stop": "stop", "length": "length", "tool_calls": "tool_calls", "content_filter": "content_filter"}
        usage_data = data.get("usage", {})

        return Response(
            id=data.get("id", ""),
            model=model,
            provider="openai",
            message=message,
            finish_reason=FinishReason(reason=reason_map.get(reason_raw, "other"), raw=reason_raw),
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            raw=data,
        )

    def _translate_stream_event(self, payload: dict) -> StreamEvent | None:
        choices = payload.get("choices", [])
        if not choices:
            return None
        delta = choices[0].get("delta", {})
        if delta.get("content"):
            return StreamEvent(type=StreamEventType.TEXT_DELTA, delta=delta["content"])
        if delta.get("tool_calls"):
            tc = delta["tool_calls"][0]
            return StreamEvent(
                type=StreamEventType.TOOL_CALL_DELTA,
                delta=tc.get("function", {}).get("arguments", ""),
            )
        return None

    def _raise_error(self, resp: httpx.Response) -> None:
        try:
            data = resp.json()
        except Exception:
            data = {}
        msg = data.get("error", {}).get("message", resp.text)
        self._raise_for_status(resp.status_code, msg, data)

    def _raise_error_from_bytes(self, status: int, raw: bytes) -> None:
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        msg = data.get("error", {}).get("message", raw.decode(errors="replace"))
        self._raise_for_status(status, msg, data)

    def _raise_for_status(self, status: int, msg: str, raw: dict) -> None:
        kwargs = {"message": msg, "provider": "openai", "status_code": status, "raw": raw}
        if status == 401:
            raise AuthenticationError(**kwargs)
        if status == 404:
            raise NotFoundError(**kwargs)
        if status in (400, 422):
            raise InvalidRequestError(**kwargs)
        if status == 429:
            raise RateLimitError(**kwargs, retryable=True)
        if 500 <= status < 600:
            raise ServerError(**kwargs, retryable=True)
        raise ProviderError(**kwargs)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
