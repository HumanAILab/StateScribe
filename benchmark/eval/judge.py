from __future__ import annotations

import json
import logging
import re
from time import sleep
import threading
from typing import Any, Dict, List, Sequence

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from benchmark.eval.adapters import GroundTruthEvent
from config import GEMINI_API_KEY


DEFAULT_JUDGE_MODEL = "gemini-3.1-pro-preview"
DEFAULT_JUDGE_TIMEOUT_MS = 60000
DEFAULT_JUDGE_MAX_ATTEMPTS = 10
DEFAULT_JUDGE_RETRY_SLEEP_SEC = 3.0
LOCATION_TOKEN_PATTERN = re.compile(r"\[\[(DIR|DIS):([A-Za-z0-9_:\-]+)\]\]")
LOGGER = logging.getLogger("benchmark.eval.judge")
RETRYABLE_CLIENT_ERROR_CODES = {408, 429}


class GeminiChangeJudge:
    def __init__(self, model: str = DEFAULT_JUDGE_MODEL) -> None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.model = str(model or DEFAULT_JUDGE_MODEL)
        self._thread_local = threading.local()

    def _client(self) -> genai.Client:
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={"timeout": DEFAULT_JUDGE_TIMEOUT_MS},
            )
            self._thread_local.client = client
        return client

    def match_prediction(
        self,
        *,
        prediction_text: str,
        candidates: Sequence[GroundTruthEvent],
    ) -> Dict[str, Any]:
        if not candidates:
            return {
                "matched_event_ids": [],
                "reason": "No candidate ground-truth events for this prediction.",
            }

        clean_text = _normalize_prediction_text(prediction_text)
        prompt = _build_prompt(clean_text, candidates)
        response = None
        for attempt in range(1, DEFAULT_JUDGE_MAX_ATTEMPTS + 1):
            try:
                response = self._client().models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        system_instruction=_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=_RESPONSE_SCHEMA,
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                    ),
                )
                break
            except (genai_errors.ServerError, genai_errors.ClientError, genai_errors.APIError) as exc:
                if attempt >= DEFAULT_JUDGE_MAX_ATTEMPTS or not _is_retryable_judge_error(exc):
                    raise
                retry_sleep_sec = DEFAULT_JUDGE_RETRY_SLEEP_SEC * float(attempt)
                LOGGER.warning(
                    "Judge request failed on attempt %s/%s (%s); retrying after %.1fs.",
                    attempt,
                    DEFAULT_JUDGE_MAX_ATTEMPTS,
                    _judge_error_summary(exc),
                    retry_sleep_sec,
                )
                sleep(retry_sleep_sec)
        if response is None:
            raise RuntimeError("Judge response is unexpectedly empty after retries.")
        payload = json.loads(response.text or "{}")
        matched = payload.get("matched_event_ids")
        if not isinstance(matched, list):
            matched = []
        valid_ids = {item.event_id for item in candidates}
        matched_ids = [
            str(item)
            for item in matched
            if isinstance(item, str) and str(item) in valid_ids
        ]
        return {
            "matched_event_ids": matched_ids,
            "reason": str(payload.get("reason", "") or ""),
        }


def _is_retryable_judge_error(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return int(getattr(exc, "code", 0) or 0) in RETRYABLE_CLIENT_ERROR_CODES
    if isinstance(exc, genai_errors.APIError):
        code = int(getattr(exc, "code", 0) or 0)
        return code in RETRYABLE_CLIENT_ERROR_CODES or 500 <= code < 600
    return False


def _judge_error_summary(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = getattr(exc, "message", None)
    parts = [str(part) for part in (code, status, message) if part not in {None, ""}]
    return " | ".join(parts) if parts else exc.__class__.__name__


def _normalize_prediction_text(text: str) -> str:
    stripped = LOCATION_TOKEN_PATTERN.sub("", text or "")
    return " ".join(stripped.split()).strip()


def _build_prompt(prediction_text: str, candidates: Sequence[GroundTruthEvent]) -> str:
    gt_lines = []
    for event in candidates:
        gt_lines.append(
            json.dumps(
                {
                    "event_id": event.event_id,
                    "change_type": event.change_type,
                    "object_description": event.object_description,
                    "change_description": event.change_description,
                    "segment_id": event.segment_id,
                },
                ensure_ascii=True,
            )
        )
    return (
        "Prediction text:\n"
        f"{prediction_text}\n\n"
        "Candidate ground-truth events:\n"
        + "\n".join(gt_lines)
    )


_SYSTEM_PROMPT = """You match one model-predicted scene-change description to zero or more ground-truth change events.

Be lenient and pragmatic:
- If the prediction clearly refers to the same real-world change, match it.
- Allow wording differences such as replace vs appear/disappear when they describe the same replacement event.
- Allow more detailed prediction text than the ground truth.
- Allow object aliases, paraphrases, and brand/detail differences if the core changed object is the same.
- Screen/display content changes can be paraphrased and still match.
- Ground-truth evidence can be conservative, so a prediction may still match even if it seems to describe the same change slightly earlier or more decisively than the annotation.
- Be cautious but allow these earlier-than-GT-style matches when the semantic identity of the change is still clear.

Be strict about identity:
- Do not match if it is a different object or a different change.
- Do not guess. If unsupported, return no matches.

Return every GT event id that the prediction clearly matches. If none match, return an empty list."""


_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "matched_event_ids": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "reason": {"type": "STRING"},
    },
    "required": ["matched_event_ids", "reason"],
}
