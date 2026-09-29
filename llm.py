"""Provider-agnostic LLM layer: Groq (primary) -> Gemini (fallback) -> models discovered at runtime.

Lessons from the PDF RAG project: model names get retired and free tiers get overloaded,
so we retry transient errors with backoff, skip retired models, and discover live models.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable

GROQ_URL = "https://api.groq.com/openai/v1"


def _env_list(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


GROQ_MODELS = _env_list("GROQ_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant")
GEMINI_MODELS = _env_list("GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash-lite")
TRANSIENT = ("429", "500", "502", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "timed out")


class LLMError(RuntimeError):
    pass


def _is_transient(err: Exception) -> bool:
    return any(t in str(err) for t in TRANSIENT)


def _groq_call(key: str, model: str, prompt: str, json_mode: bool) -> str:
    import requests

    body = {"model": model, "temperature": 0.2, "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{GROQ_URL}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
    if r.status_code != 200:
        raise LLMError(f"Groq {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _gemini_call(key: str, model: str, prompt: str, json_mode: bool) -> str:
    from google import genai
    from google.genai import types

    cfg = types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json" if json_mode else None)
    res = genai.Client(api_key=key).models.generate_content(model=model, contents=prompt, config=cfg)
    return res.text or ""


def _discover_groq(key: str) -> list[str]:
    import requests

    try:
        ids = [m["id"] for m in requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=15).json()["data"]]
    except Exception:  # noqa: BLE001
        return []
    bad = ("whisper", "guard", "tts", "playai", "orpheus", "distil")
    return sorted((i for i in ids if not any(b in i for b in bad)), reverse=True)[:3]


def _discover_gemini(key: str) -> list[str]:
    try:
        from google import genai

        names = []
        for m in genai.Client(api_key=key).models.list():
            n = (m.name or "").removeprefix("models/")
            if "gemini" in n and "generateContent" in (getattr(m, "supported_actions", None) or []) and not any(
                x in n for x in ("embedding", "image", "tts", "live", "audio", "vision", "robotics")
            ):
                names.append(n)
        return sorted(names, key=lambda n: ("flash" in n, n), reverse=True)[:3]
    except Exception:  # noqa: BLE001
        return []


class LLM:
    """complete(prompt, json_mode) with retries and fallbacks. `tried` records the last call's attempts."""

    def __init__(self, groq_key: str | None = None, gemini_key: str | None = None, sleep: Callable = time.sleep):
        if not (groq_key or gemini_key):
            raise LLMError("Set GROQ_API_KEY and/or GEMINI_API_KEY")
        self.groq_key, self.gemini_key, self.sleep = groq_key, gemini_key, sleep
        self.tried: list[str] = []
        self.used: str | None = None  # remember a working model to go straight there next time

    def _plan(self):
        plan = []
        if self.groq_key:
            k = self.groq_key
            plan += [("groq", lambda m, p, j: _groq_call(k, m, p, j), lambda: GROQ_MODELS),
                     ("groq", lambda m, p, j: _groq_call(k, m, p, j), lambda: _discover_groq(k))]
        if self.gemini_key:
            k2 = self.gemini_key
            plan += [("gemini", lambda m, p, j: _gemini_call(k2, m, p, j), lambda: GEMINI_MODELS),
                     ("gemini", lambda m, p, j: _gemini_call(k2, m, p, j), lambda: _discover_gemini(k2))]
        return plan

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        self.tried = []
        first: Exception | None = None
        plan = self._plan()
        if self.used:  # try the last working model first
            label, model = self.used.split(":", 1)
            call = next(c for lab, c, _ in plan if lab == label)
            plan.insert(0, (label, call, lambda m=model: [m]))
        for label, call, models in plan:
            for model in models():
                tag = f"{label}:{model}"
                if tag in self.tried:
                    continue
                self.tried.append(tag)
                for attempt in range(3):
                    try:
                        out = call(model, prompt, json_mode)
                        self.used = tag
                        return out
                    except Exception as e:  # noqa: BLE001
                        first = first or e
                        if not _is_transient(e):
                            break  # e.g. 404 retired model -> next model
                        self.sleep(2**attempt)
        raise LLMError(f"All models failed (tried {self.tried}): {first}")

    def complete_json(self, prompt: str) -> dict:
        return parse_json(self.complete(prompt + "\n\nRespond with a single valid JSON object only.", json_mode=True))


def parse_json(text: str) -> dict:
    """Tolerant JSON parsing: strips code fences and extra prose around the object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))
