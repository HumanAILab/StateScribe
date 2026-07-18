# components/vlm/change_analyzer.py
import cv2
import json
import logging
from typing import Dict, Any, List

import numpy as np
from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_API_TIMEOUT_MS,
    DIFF_VLM_MODEL,
    DIFF_VLM_TEMPERATURE,
    DIFF_VLM_MAX_TOKENS,
    DIFF_VLM_HALLUCINATION_FILTER_ENABLED,
)
from prompts import (
    DIFF_SYSTEM_PROMPT,
    DIFF_USER_PROMPT,
    DIFF_OUTPUT_SCHEMA,
    DIFF_TEXT_FILTER_SYSTEM_PROMPT,
    DIFF_TEXT_FILTER_USER_PROMPT_TEMPLATE,
    DIFF_TEXT_FILTER_OUTPUT_SCHEMA,
)

logger = logging.getLogger(__name__)

_DIFF_TEXT_FILTER_MODEL = "gemini-3.1-flash-lite-preview"
_DIFF_TEXT_FILTER_TEMPERATURE = 0.0
_DIFF_TEXT_FILTER_MAX_TOKENS = 128

class ChangeAnalyzer:
    """
    Handles VLM-based change analysis:
    1. Full-frame change detection with bboxes
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": GEMINI_API_TIMEOUT_MS})
        logger.debug("ChangeAnalyzer initialized")

    def detect_changes(self, img_t0: np.ndarray, img_t1: np.ndarray) -> Dict[str, Any]:
        part_t0 = types.Part.from_bytes(
            data=self._encode_image_bytes(img_t0),
            mime_type="image/jpeg",
        )
        part_t1 = types.Part.from_bytes(
            data=self._encode_image_bytes(img_t1),
            mime_type="image/jpeg",
        )
        try:
            response = self.client.models.generate_content(
                model=DIFF_VLM_MODEL,
                contents=[part_t0, part_t1, DIFF_USER_PROMPT],
                config=types.GenerateContentConfig(
                    temperature=DIFF_VLM_TEMPERATURE,
                    max_output_tokens=DIFF_VLM_MAX_TOKENS,
                    system_instruction=DIFF_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=DIFF_OUTPUT_SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            result = json.loads(response.text)
        except Exception:
            logger.warning("Change detection API call failed.", exc_info=True)
            return {"changes": []}
        result = self._apply_text_hallucination_filter(result)
        logger.debug(f"Diff VLM returned {len(result.get('changes', []))} changes")
        return result

    @staticmethod
    def _encode_image_bytes(rgb: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr)
        if not ok:
            raise RuntimeError("Failed to encode image")
        return enc.tobytes()

    def _apply_text_hallucination_filter(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not DIFF_VLM_HALLUCINATION_FILTER_ENABLED:
            return result
        if not isinstance(result, dict):
            return {"changes": []}

        changes = result.get("changes")
        if not isinstance(changes, list) or not changes:
            return result

        candidates: List[Dict[str, str]] = []
        for idx, change in enumerate(changes):
            if not isinstance(change, dict):
                continue
            candidates.append(
                {
                    "candidate_id": f"c{idx}",
                    "change_type": str(change.get("change_type", "") or "").strip(),
                    "object_description": str(change.get("object_description", "") or "").strip(),
                    "change_description": str(change.get("change_description", "") or "").strip(),
                    "context_description": str(change.get("context_description", "") or "").strip(),
                }
            )

        if not candidates:
            return result

        prompt = DIFF_TEXT_FILTER_USER_PROMPT_TEMPLATE.format(
            candidates_json=json.dumps({"changes": candidates}, ensure_ascii=True)
        )
        try:
            response = self.client.models.generate_content(
                model=_DIFF_TEXT_FILTER_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=_DIFF_TEXT_FILTER_TEMPERATURE,
                    max_output_tokens=_DIFF_TEXT_FILTER_MAX_TOKENS,
                    system_instruction=DIFF_TEXT_FILTER_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=DIFF_TEXT_FILTER_OUTPUT_SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            parsed = json.loads(response.text or "{}")
        except Exception:
            return result

        reject_id = str(parsed.get("reject_candidate_id", "") or "").strip()
        if not reject_id:
            return result

        kept_changes = [
            change
            for idx, change in enumerate(changes)
            if f"c{idx}" != reject_id
        ]
        if len(kept_changes) == len(changes):
            return result

        filtered = dict(result)
        filtered["changes"] = kept_changes
        return filtered
