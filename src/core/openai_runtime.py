"""
OpenAI SDK runtime — dùng cho:

  Blue Team → OpenRouter liquid/lfm-2.5-2.6b (create_blue_pair)
  Red Team  → OpenAI gpt-4o-mini (create_openai_pair) khi RED_TEAM_PROVIDER=openai

Gemini Red Team dùng Google ADK trong agents/*.py — không đi qua file này.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from core.tracing import record_decision, trace_span

from core.config import (
    get_red_model,
    get_red_provider,
    get_blue_model,
    get_blue_provider,
    blue_client_kwargs,
    red_openai_client_kwargs,
)


@dataclass
class OpenAIAgent:
    name: str
    instruction: str
    provider: str = "openai"


@dataclass
class _MockInvocationContext:
    user_id: str = "student"


@dataclass
class OpenAIRunner:
    """Optional ADK-style plugins + Chat Completions."""

    app_name: str
    model: str
    plugins: list = field(default_factory=list)
    provider: str = "openai"
    temperature: float = 0.4
    client_kwargs: dict = field(default_factory=dict)
    input_hooks: list[Callable[[str], str | None]] = field(default_factory=list)
    output_hooks: list[Callable[[str], str]] = field(default_factory=list)

    def _client(self):
        from openai import OpenAI

        kwargs = dict(self.client_kwargs or {})
        if self.provider == "openrouter":
            kwargs["max_retries"] = 0  # Honor provider Retry-After in one place.
        return OpenAI(**kwargs)

    async def _create_completion(self, client, request: dict):
        from openai import NotFoundError, RateLimitError

        retries = 0
        while True:
            try:
                return client.chat.completions.create(**request)
            except NotFoundError:
                if not (self.provider == "openrouter"
                        and self.model == "liquid/lfm-2.5-2.6b"):
                    raise
                self.model = "liquid/lfm-2.5-2.6b:free"
                request["model"] = self.model
            except RateLimitError as exc:
                if self.provider != "openrouter" or retries >= 3:
                    raise
                body = exc.body if isinstance(exc.body, dict) else {}
                error = body.get("error") or body
                if not isinstance(error, dict):
                    error = body
                metadata = error.get("metadata") or {}
                retry_after = (metadata.get("retry_after_seconds")
                               or exc.response.headers.get("Retry-After"))
                try:
                    delay = min(90.0, max(1.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = min(60.0, 5.0 * 2 ** retries)
                retries += 1
                print(f"OpenRouter rate limited; retrying in {delay:g}s "
                      f"({retries}/3)...", flush=True)
                await asyncio.sleep(delay)

    async def chat(self, agent: OpenAIAgent, user_message: str, *,
                   user_id: str = "student", with_decision: bool = False):
        def finish(response: str, *, decision: str, layer: str | None = None):
            result = {"response": response, "decision": decision, "layer": layer,
                      "blocked": decision == "BLOCK"}
            return result if with_decision else response

        target = ("blue" if self.provider == "openrouter" else
                  "red_advance" if "advance" in self.app_name else "red_default")
        with trace_span(f"{target}_request", input_chars=len(user_message),
                        input_text=user_message,
                        metadata={"agent": target, "model": self.model}) as run:
            for hook in self.input_hooks:
                hook_layer = f"{target}_input" if target == "red_advance" else "input_hook"
                with trace_span(hook_layer, input_chars=len(user_message),
                                input_text=user_message) as hook_run:
                    blocked = hook(user_message)
                    record_decision(hook_run, decision="BLOCK" if blocked else "ALLOW",
                                    layer=hook_layer, output_text=blocked)
                if blocked:
                    record_decision(run, decision="BLOCK", layer=hook_layer,
                                    output_text=blocked)
                    return finish(blocked, decision="BLOCK", layer=hook_layer)

            block_msg, blocked_at = await self._run_input_plugins(user_message, user_id=user_id)
            if block_msg is not None:
                record_decision(run, decision="BLOCK", layer=blocked_at,
                                output_text=block_msg)
                return finish(block_msg, decision="BLOCK", layer=blocked_at)

            with trace_span("model_call", input_chars=len(user_message),
                            input_text=user_message,
                            metadata={"agent": target, "model": self.model}) as model_run:
                client = self._client()
                request = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": agent.instruction},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": self.temperature,
                }
                completion = await self._create_completion(client, request)
                response = (completion.choices[0].message.content or "").strip()
                record_decision(model_run, decision="GENERATED", layer="model",
                                output_text=response)

            hook_layer = None
            for hook in self.output_hooks:
                layer = f"{target}_output" if target == "red_advance" else "output_hook"
                with trace_span(layer, input_chars=len(response),
                                input_text=response) as hook_run:
                    updated = hook(response)
                    changed = updated != response
                    record_decision(hook_run, decision="BLOCK" if changed else "ALLOW",
                                    layer=layer, output_text=updated)
                response = updated
                if changed:
                    hook_layer = layer

            response, output_layer, output_decision = await self._run_output_plugins(response)
            final_decision = output_decision or ("BLOCK" if hook_layer else "ALLOW")
            final_layer = output_layer or hook_layer
            record_decision(run, decision=final_decision, layer=final_layer,
                            output_text=response)
            return finish(response, decision=final_decision, layer=final_layer)

    async def _run_input_plugins(self, user_message: str, *, user_id: str = "student") -> tuple[str | None, str | None]:
        if not self.plugins:
            return None, None
        try:
            from google.genai import types
        except ImportError:
            return None, None

        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)],
        )
        ctx = _MockInvocationContext(user_id=user_id)
        for plugin in self.plugins:
            cb = getattr(plugin, "on_user_message_callback", None)
            if cb is None:
                continue
            try:
                result = await cb(
                    invocation_context=ctx, user_message=user_content
                )
            except TypeError:
                result = cb(invocation_context=ctx, user_message=user_content)
            if result is None:
                continue
            return _content_to_text(result), plugin.name
        return None, None

    async def _run_output_plugins(self, text: str) -> tuple[str, str | None, str | None]:
        if not self.plugins or not text:
            return text, None, None
        try:
            from google.genai import types
        except ImportError:
            return text, None, None

        content = types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        )

        class _Resp:
            pass

        llm_response = _Resp()
        llm_response.content = content

        class _Ctx:
            pass

        changed_at = None
        decision = None
        for plugin in self.plugins:
            cb = getattr(plugin, "after_model_callback", None)
            if cb is None:
                continue
            before = _content_to_text(llm_response.content)
            blocked_before = getattr(plugin, "blocked_count", 0)
            try:
                out = await cb(callback_context=_Ctx(), llm_response=llm_response)
            except TypeError:
                out = cb(callback_context=_Ctx(), llm_response=llm_response)
            if out is not None and getattr(out, "content", None) is not None:
                llm_response = out
            if _content_to_text(llm_response.content) != before:
                changed_at = plugin.name
                decision = ("BLOCK" if getattr(plugin, "blocked_count", 0) > blocked_before
                            else "REDACT")
        return _content_to_text(llm_response.content) or text, changed_at, decision


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks = []
    for part in parts:
        t = getattr(part, "text", None)
        if t:
            chunks.append(t)
    return "".join(chunks)


def _make_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    model: str,
    provider: str,
    client_kwargs: dict,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    agent = OpenAIAgent(name=name, instruction=instruction, provider=provider)
    runner = OpenAIRunner(
        app_name=app_name,
        model=model,
        provider=provider,
        client_kwargs=client_kwargs,
        plugins=list(plugins or []),
        input_hooks=list(input_hooks or []),
        output_hooks=list(output_hooks or []),
        temperature=temperature,
    )
    return agent, runner


def create_blue_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Blue Team — always OpenRouter liquid/lfm-2.5-2.6b."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=get_blue_model(),
        provider=get_blue_provider(),
        client_kwargs=blue_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )


def create_openai_pair(
    *,
    name: str,
    instruction: str,
    app_name: str,
    plugins: list | None = None,
    input_hooks: list | None = None,
    output_hooks: list | None = None,
    temperature: float = 0.4,
    model: str | None = None,
) -> tuple[OpenAIAgent, OpenAIRunner]:
    """Red Team OpenAI path (default = soft model; advance may pass harder)."""
    return _make_pair(
        name=name,
        instruction=instruction,
        app_name=app_name,
        model=model or get_red_model(),
        provider=get_red_provider(),
        client_kwargs=red_openai_client_kwargs(),
        plugins=plugins,
        input_hooks=input_hooks,
        output_hooks=output_hooks,
        temperature=temperature,
    )
