"""Readable LangSmith spans with sensitive values removed."""
from __future__ import annotations

import os
import re
from contextlib import nullcontext


_SENSITIVE_PATTERNS = (
    r"\bsk-[A-Za-z0-9-]+\b",
    r"\bdb\.vinbank\.internal(?::\d+)?\b",
    r"\badmin123\b",
    r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w-])",
    r"(?<!\d)0\d{9,10}(?!\d)",
    r"(?<!\d)(?:\d{12}|\d{9})(?!\d)",
)
_ENCODED_TOKEN = re.compile(r"(?<![A-Za-z0-9_+/=-])[A-Za-z0-9_+/-]{32,}={0,2}(?![A-Za-z0-9_+/=-])")


def trace_preview(text: str, *, limit: int = 500) -> str:
    """Show enough context to debug a decision without uploading lab secrets."""
    preview = text[:limit + 1000]
    preview = re.sub(
        r"(\bpassword\s*(?::|=|is\b)\s*)(?!\[blank\]|\[redacted\])\S+",
        r"\1[REDACTED]", preview, flags=re.IGNORECASE,
    )
    for pattern in _SENSITIVE_PATTERNS:
        preview = re.sub(pattern, "[REDACTED]", preview, flags=re.IGNORECASE)
    preview = _ENCODED_TOKEN.sub("[ENCODED]", preview)
    return preview[:limit] + ("..." if len(text) > limit else "")


def trace_span(name: str, *, input_chars: int = 0, input_text: str | None = None,
               metadata: dict | None = None):
    """Return a trace context; disabled tracing does not affect the pipeline."""
    enabled = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    if not enabled or not os.getenv("LANGSMITH_API_KEY"):
        return nullcontext(None)

    from langsmith import trace

    inputs = {"input_chars": input_chars}
    if input_text is not None:
        inputs["input_preview"] = trace_preview(input_text)
    options = {"metadata": metadata} if metadata else {}
    return trace(name, run_type="chain", inputs=inputs, **options)


def record_decision(run, *, decision: str, layer: str | None = None, reason: str | None = None,
                    issues: list[str] | None = None, output_text: str | None = None,
                    details: dict | None = None):
    if run is None:
        return
    result = {"decision": decision, "layer": layer}
    if reason:
        result["reason"] = reason
    if issues:
        result["issues"] = issues
    if output_text is not None:
        result["output_preview"] = trace_preview(output_text)
    if details:
        result.update(details)
    run.outputs = result
