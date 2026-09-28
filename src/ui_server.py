"""Local web dashboard for testing the lab agents and checkpoints.

Run from the repository root: python src/ui_server.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.config import (  # noqa: E402
    get_blue_model, get_openrouter_api_key, get_red_model, get_red_provider,
    get_openai_api_key,
)


ARTIFACTS = {
    "results": "results.json",
    "audit": "audit_log.json",
    "metrics": "metrics.json",
    "attacks": "attack_results.json",
    "unsafe": "unsafe_attack_result.json",
    "guards": "guards_attack_result.json",
}


class DashboardState:
    def __init__(self):
        self.operation_lock = threading.Lock()
        self.job_lock = threading.Lock()
        self.job: dict | None = None
        self.process: subprocess.Popen | None = None
        self.agents: dict = {}

    def key_error(self, target: str) -> str | None:
        if target == "blue" and not get_openrouter_api_key():
            return "Thiếu OPENROUTER_API_KEY trong .env."
        if target in {"red", "red_advance"}:
            if get_red_provider() == "openai" and not get_openai_api_key():
                return "Thiếu OPENAI_API_KEY trong .env."
            if get_red_provider() == "gemini":
                import os
                if not os.getenv("GOOGLE_API_KEY", "").strip():
                    return "Thiếu GOOGLE_API_KEY trong .env."
        return None

    def get_agent(self, target: str):
        if target not in self.agents:
            if target == "blue":
                from agents.agent import create_blue_agent
                from assignment.pipeline import build_production_plugins
                pair = create_blue_agent(build_production_plugins(use_llm_judge=False))
            elif target == "red":
                from agents.agent import create_red_agent_default
                pair = create_red_agent_default()
            elif target == "red_advance":
                from agents.guards_agent import create_red_agent_advance
                pair = create_red_agent_advance()
            else:
                raise ValueError("Bot không hợp lệ.")
            self.agents[target] = pair
        return self.agents[target]

    def test_prompt(self, target: str, prompt: str) -> dict:
        from core.openai_runtime import OpenAIRunner
        from agents.guards_agent import check_secret_leak
        from core.utils import chat_with_agent

        agent, runner = self.get_agent(target)
        started = time.monotonic()
        exact = isinstance(runner, OpenAIRunner)
        if exact:
            result = asyncio.run(runner.chat(agent, prompt, user_id="dashboard",
                                             with_decision=True))
            response = result["response"]
            decision = result["decision"]
            layer = result["layer"]
        else:
            response, _ = asyncio.run(chat_with_agent(agent, runner, prompt))
            from attacks.attacks import classify_attack_outcome
            outcome = classify_attack_outcome(prompt, response,
                target_name="red_advance" if target == "red_advance" else "red_default")
            decision = "BLOCK" if outcome["blocked"] else "ALLOW"
            layer = outcome["layer"]
        leaked = check_secret_leak(response)
        if leaked:
            decision = "LEAK"
        stages = describe_stages(target, prompt, decision, layer, exact)
        return {
            "target": target,
            "model": get_blue_model() if target == "blue" else get_red_model(),
            "provider": "openrouter" if target == "blue" else get_red_provider(),
            "decision": decision,
            "layer": layer,
            "blocked": decision == "BLOCK",
            "leaked": leaked,
            "response": response,
            "stages": stages,
            "source": "runtime" if exact else "inferred from reply",
            "duration_ms": round((time.monotonic() - started) * 1000),
        }

    def start_job(self, part: int) -> dict:
        if not self.operation_lock.acquire(blocking=False):
            raise RuntimeError("Đang có một tác vụ khác chạy. Hãy chờ nó xong.")
        with self.job_lock:
            self.job = {"part": part, "status": "running", "lines": [],
                        "started_at": time.time(), "exit_code": None}
        thread = threading.Thread(target=self._run_job, args=(part,), daemon=True)
        thread.start()
        return self.job_snapshot()

    def _run_job(self, part: int) -> None:
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", str(SRC / "main.py"), "--part", str(part)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                bufsize=1,
            )
            with self.job_lock:
                self.process = process
            assert process.stdout is not None
            for line in process.stdout:
                with self.job_lock:
                    if self.job is not None:
                        self.job["lines"].append(line.rstrip("\r\n"))
                        self.job["lines"] = self.job["lines"][-500:]
            exit_code = process.wait()
            with self.job_lock:
                if self.job is not None:
                    self.job["exit_code"] = exit_code
                    self.job["status"] = "done" if exit_code == 0 else "failed"
                    self.job["ended_at"] = time.time()
        except Exception as exc:
            with self.job_lock:
                if self.job is not None:
                    self.job["status"] = "failed"
                    self.job["exit_code"] = -1
                    self.job["lines"].append(f"Dashboard error: {exc}")
        finally:
            with self.job_lock:
                self.process = None
            self.operation_lock.release()

    def stop_job(self) -> dict:
        with self.job_lock:
            process = self.process
            if process is None or process.poll() is not None:
                raise RuntimeError("Không có checkpoint nào đang chạy.")
            process.terminate()
        return self.job_snapshot()

    def job_snapshot(self) -> dict:
        with self.job_lock:
            return dict(self.job) if self.job is not None else {"status": "idle", "lines": []}


def describe_stages(target: str, prompt: str, decision: str,
                    layer: str | None, exact: bool) -> list[dict]:
    """Summarize runtime checkpoints; never claim unobserved stages ran."""
    if target == "red":
        return [
            {"name": "Input guardrail", "status": "off", "detail": "Red thường không gắn guardrail."},
            {"name": "Model", "status": "done", "detail": "Model đã tạo phản hồi."},
            {"name": "Output guardrail", "status": "off", "detail": "Không có bộ lọc đầu ra."},
        ]

    input_layer = layer in {"rate_limiter", "input_guardrail", "input_hook",
                            "red_advance_input", "input_injection", "input_topic"}
    output_layer = layer in {"output_guardrail", "output_hook", "red_advance_output",
                             "output_filter"}
    if target == "blue":
        from guardrails.input_guardrails import detect_injection, topic_filter
        reason = ("Prompt injection" if detect_injection(prompt) == "BLOCK" else
                  "Ngoài chủ đề ngân hàng" if topic_filter(prompt) == "BLOCK" else "Đã qua")
        rate = {"name": "Rate limiter", "status": "block" if layer == "rate_limiter" else "pass",
                "detail": "Vượt giới hạn của ứng dụng." if layer == "rate_limiter" else "Chưa vượt giới hạn."}
    else:
        from agents.guards_agent import detect_injection_strong, topic_filter_strong
        reason = ("Prompt injection" if detect_injection_strong(prompt) else
                  "Ngoài chủ đề ngân hàng" if topic_filter_strong(prompt) else "Đã qua")
        rate = None

    input_stage = {"name": "Input guardrail", "status": "block" if input_layer and layer != "rate_limiter" else
                   "skip" if layer == "rate_limiter" else "pass",
                   "detail": reason if input_layer and layer != "rate_limiter" else
                   "Chưa chạy." if layer == "rate_limiter" else "Prompt được cho qua."}
    model_stage = {"name": "Model", "status": "skip" if input_layer else "done",
                   "detail": "Không gọi model vì đã chặn đầu vào." if input_layer else "Model đã tạo phản hồi."}
    output_stage = {"name": "Output guardrail",
                    "status": "skip" if input_layer else
                              "redact" if decision == "REDACT" else
                              "block" if output_layer and decision == "BLOCK" else "pass",
                    "detail": "Chưa chạy." if input_layer else
                              "Đã sửa nội dung nhạy cảm." if decision == "REDACT" else
                              "Đã chặn phản hồi." if output_layer and decision == "BLOCK" else
                              "Không phát hiện vấn đề ở đầu ra."}
    stages = [rate, input_stage, model_stage, output_stage] if rate else [input_stage, model_stage, output_stage]
    if not exact:
        for stage in stages:
            stage["detail"] += " (ước lượng từ phản hồi)"
    return stages


def artifact_summary() -> dict:
    output = ROOT / "outputs"
    files = {}
    for key, name in ARTIFACTS.items():
        path = output / name
        files[key] = {"name": name, "exists": path.is_file(),
                      "bytes": path.stat().st_size if path.is_file() else 0}
    result = {"files": files}
    try:
        data = json.loads((output / "results.json").read_text(encoding="utf-8"))
        result["part3"] = {
            "safe": len(data.get("safe_queries", [])),
            "safe_blocked": sum(bool(x.get("blocked")) for x in data.get("safe_queries", [])),
            "attacks": len(data.get("attack_queries", [])),
            "attacks_blocked": sum(bool(x.get("blocked")) for x in data.get("attack_queries", [])),
            "edges": len(data.get("edge_cases", [])),
            "rate_limit": data.get("rate_limit"),
            "rows": {group: data.get(group, []) for group in
                     ("safe_queries", "attack_queries", "edge_cases")},
        }
    except (OSError, ValueError, TypeError):
        result["part3"] = None
    try:
        data = json.loads((output / "attack_results.json").read_text(encoding="utf-8"))
        result["part4"] = {
            "provider": data.get("llm_provider"), "model": data.get("llm_model"),
            "summary": data.get("summary", {}),
            "unsafe": data.get("unsafe_attacks", []),
            "guards": data.get("guards_attacks", []),
        }
    except (OSError, ValueError, TypeError):
        result["part4"] = None
    progress = output / ".part3-progress.json"
    if progress.is_file():
        try:
            saved = json.loads(progress.read_text(encoding="utf-8"))
            result["part3_progress"] = len(saved.get("completed", {}))
        except (OSError, ValueError, TypeError):
            result["part3_progress"] = 0
    else:
        result["part3_progress"] = 0
    return result


STATE = DashboardState()
PAGE = (SRC / "ui" / "index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 64_000:
            raise ValueError("Yêu cầu quá lớn hoặc rỗng.")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("JSON phải là object.")
        return data

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif path == "/api/status":
            self.respond(200, {"blue_model": get_blue_model(),
                               "red_model": get_red_model(),
                               "red_provider": get_red_provider(),
                               "blue_key": bool(get_openrouter_api_key()),
                               "red_key": STATE.key_error("red") is None,
                               "busy": STATE.operation_lock.locked(),
                               "job": STATE.job_snapshot()})
        elif path == "/api/artifacts":
            self.respond(200, artifact_summary())
        elif path == "/api/job":
            self.respond(200, STATE.job_snapshot())
        else:
            self.respond(404, {"error": "Không tìm thấy."})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            data = self.read_json()
            if path == "/api/prompt":
                target = data.get("target")
                prompt = data.get("prompt")
                if target not in {"blue", "red", "red_advance"}:
                    raise ValueError("Chọn Blue, Red hoặc Red Advance.")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Nhập prompt trước khi thử.")
                if len(prompt) > 10_000:
                    raise ValueError("Prompt tối đa 10.000 ký tự.")
                key_error = STATE.key_error(target)
                if key_error:
                    raise ValueError(key_error)
                if not STATE.operation_lock.acquire(blocking=False):
                    raise RuntimeError("Đang có một tác vụ khác chạy. Hãy chờ nó xong.")
                try:
                    self.respond(200, STATE.test_prompt(target, prompt.strip()))
                finally:
                    STATE.operation_lock.release()
            elif path == "/api/run":
                part = data.get("part")
                if type(part) is not int or part not in {2, 3, 4}:
                    raise ValueError("Chọn Part 2, 3 hoặc 4.")
                key_error = STATE.key_error("blue") if part in {3, 4} else None
                if part == 4:
                    key_error = key_error or STATE.key_error("red")
                if key_error:
                    raise ValueError(key_error)
                self.respond(202, STATE.start_job(part))
            elif path == "/api/stop":
                self.respond(200, STATE.stop_job())
            else:
                self.respond(404, {"error": "Không tìm thấy."})
        except (ValueError, json.JSONDecodeError) as exc:
            self.respond(400, {"error": str(exc)})
        except RuntimeError as exc:
            self.respond(409, {"error": str(exc)})
        except Exception as exc:
            self.respond(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="VinBank lab dashboard")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
