import json
import logging
import time
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from google import genai
from google.genai import types
from google.genai import _extra_utils

from components.memory.change_memory import ChangeMemory
from components.memory.memory_manager import MemoryManager
from config import (
    AGENT_ENABLED,
    AGENT_MAX_TOKENS,
    AGENT_MODEL,
    AGENT_TEMPERATURE,
    AGENT_THINKING_BUDGET,
    GEMINI_API_KEY,
    GEMINI_API_TIMEOUT_MS,
)
from prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT_TEMPLATE
from .tools import AgentTools

logger = logging.getLogger(__name__)
_image_return_patch_applied = False


def apply_image_return_patch() -> None:
    global _image_return_patch_applied
    if _image_return_patch_applied:
        return

    def patched_get_function_response_parts(
        response: types.GenerateContentResponse,
        function_map: Dict[str, Union[Callable[..., Any], Any]],
    ) -> List[types.Part]:
        final_parts: List[types.Part] = []

        if (
            response.candidates is None
            or not isinstance(response.candidates[0].content, types.Content)
            or response.candidates[0].content.parts is None
        ):
            return []

        for part in response.candidates[0].content.parts:
            if not part.function_call:
                continue

            func_name = part.function_call.name
            if func_name is None or part.function_call.args is None or func_name not in function_map:
                continue

            func = function_map[func_name]
            args = _extra_utils.convert_number_values_for_dict_function_call_args(part.function_call.args)

            try:
                func_result = _extra_utils.invoke_function_from_dict_args(args, func)

                if (
                    isinstance(func_result, tuple)
                    and len(func_result) == 2
                    and isinstance(func_result[0], list)
                    and isinstance(func_result[1], str)
                    and all(isinstance(media_part, types.Part) for media_part in func_result[0])
                ):
                    media_parts, response_text = func_result
                    for media_part in media_parts:
                        final_parts.append(media_part)
                    final_parts.append(
                        types.Part.from_function_response(
                            name=func_name,
                            response={"result": response_text},
                        )
                    )
                elif isinstance(func_result, types.Part):
                    final_parts.append(
                        types.Part.from_function_response(
                            name=func_name,
                            response={"result": f"Function {func_name} executed and returned a media Part."},
                        )
                    )
                    final_parts.append(func_result)
                else:
                    final_parts.append(
                        types.Part.from_function_response(
                            name=func_name,
                            response={"result": func_result},
                        )
                    )
            except Exception as exc:
                final_parts.append(
                    types.Part.from_function_response(
                        name=func_name,
                        response={"error": str(exc)},
                    )
                )

        return final_parts

    _extra_utils.get_function_response_parts = patched_get_function_response_parts
    _image_return_patch_applied = True


class GeminiAgent:
    """Handles Firebase-triggered QA with spatial tools and full change-memory context."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        change_memory: ChangeMemory,
        agent_tools: AgentTools,
        on_answer: Optional[Callable[[str], bool]] = None,
        on_answer_start: Optional[Callable[[], None]] = None,
        on_answer_end: Optional[Callable[[], None]] = None,
    ):
        self.memory_manager = memory_manager
        self.change_memory = change_memory
        self.agent_tools = agent_tools
        self.on_answer = on_answer
        self.on_answer_start = on_answer_start
        self.on_answer_end = on_answer_end

        self.enabled = bool(AGENT_ENABLED and GEMINI_API_KEY)
        if self.enabled:
            apply_image_return_patch()
        self.client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": GEMINI_API_TIMEOUT_MS}) if self.enabled else None

        self._queue: Queue[Optional[str]] = Queue(maxsize=1)
        self._stop_event = Event()
        self._worker: Optional[Thread] = None

        if self.enabled:
            self._worker = Thread(target=self._worker_loop, daemon=True, name="gemini-agent")
            self._worker.start()
            logger.debug("GeminiAgent initialized with model=%s", AGENT_MODEL)
        else:
            logger.debug("GeminiAgent disabled: AGENT_ENABLED or GEMINI_API_KEY is not available.")

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(None)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3.0)

    def answer_question(self, question: str) -> bool:
        """Queue a question asynchronously; keeps only the latest pending question."""
        question = (question or "").strip()
        if not question:
            return False

        if not self.enabled:
            if self.on_answer is None:
                return False
            self._emit_answer("QA agent is disabled.")
            return True

        try:
            self._queue.put_nowait(question)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(question)
        return True

    def answer_question_sync(self, question: str) -> str:
        """Generate an answer synchronously (used by dataset test)."""
        question = (question or "").strip()
        if not question:
            return ""
        if not self.enabled:
            return "QA agent is disabled."
        return self._generate_answer(question)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._queue.empty():
                time.sleep(0.1)
                continue

            question = self._queue.get_nowait()

            if question is None:
                break

            self._safe_callback(self.on_answer_start)
            answer = self._generate_answer(question)

            self._emit_answer(answer)
            self._safe_callback(self.on_answer_end)

    def _generate_answer(self, question: str) -> str:
        if self.client is None:
            return "QA agent is not initialized."

        change_memory_json = self._build_change_memory_context()

        user_pose_text = self._latest_user_pose_text()
        prompt = AGENT_USER_PROMPT_TEMPLATE.format(
            question=question,
            user_pose=user_pose_text,
            change_memory_json=change_memory_json,
        )

        try:
            chat = self.client.chats.create(
                model=AGENT_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM_PROMPT,
                    tools=[
                        self.agent_tools.get_object_distance_and_direction,
                        self.agent_tools.get_recent_change_snapshots,
                        self.agent_tools.retrieve_recent_images,
                    ],
                    temperature=AGENT_TEMPERATURE,
                    max_output_tokens=AGENT_MAX_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=AGENT_THINKING_BUDGET),
                ),
            )
            response = chat.send_message(prompt)
        except Exception:
            logger.warning("Agent API call failed.", exc_info=True)
            return "I could not produce an answer."
        text = (response.text or "").strip()
        if not text:
            return "I could not produce an answer."
        return " ".join(text.split())

    def _build_change_memory_context(self) -> str:
        objects = sorted(self.change_memory.get_all_objects(), key=lambda obj: obj.object_id)
        memory_objects: List[Dict[str, object]] = []

        for obj in objects:
            latest = obj.get_latest_snapshot()

            snapshots = []
            for snap in obj.snapshots:
                bbox_values: List[float] = []
                if snap.bbox_3d is not None:
                    bbox = np.asarray(snap.bbox_3d, dtype=np.float64).reshape(-1)
                    if bbox.size == 6:
                        bbox_values = [round(float(v), 3) for v in bbox.tolist()]
                snapshots.append(
                    {
                        "change_type": snap.change_type,
                        "timestamp": snap.timestamp.isoformat(),
                        "description": snap.description,
                        "bbox_3d": bbox_values,
                    }
                )

            latest_center = None
            if latest is not None and latest.bbox_3d is not None:
                arr = np.asarray(latest.bbox_3d, dtype=np.float64).reshape(-1)
                if arr.size == 6:
                    center = (arr[:3] + arr[3:]) / 2.0
                    latest_center = [round(float(v), 3) for v in center.tolist()]

            memory_objects.append(
                {
                    "object_id": obj.object_id,
                    "status": "disappeared" if obj.is_disappeared() else "active",
                    "latest_change_type": latest.change_type if latest else "",
                    "latest_description": latest.description if latest else "",
                    "latest_center_world_xyz": latest_center,
                    "snapshots": snapshots,
                }
            )

        payload = {
            "object_count": len(memory_objects),
            "objects": memory_objects,
            "reference_note": "Use object_id or numeric id when calling tools (for example: obj_0003 or 3).",
        }
        return json.dumps(payload, ensure_ascii=True)

    def _latest_user_pose_text(self) -> str:
        frames = self.memory_manager.get_memory()
        if not frames:
            return "No frame available."

        latest = frames[-1]
        pose = getattr(latest, "pose_matrix", None)
        if pose is None:
            return "Latest frame has no pose matrix."

        pose_arr = np.asarray(pose, dtype=np.float64)
        if pose_arr.shape != (4, 4):
            return "Latest frame has invalid pose matrix."

        position = pose_arr[:3, 3]
        forward = pose_arr[:3, 2]
        return json.dumps(
            {
                "frame_timestamp": latest.timestamp.isoformat() if latest.timestamp else "",
                "position_xyz": [round(float(v), 3) for v in position.tolist()],
                "forward_vector_xyz": [round(float(v), 3) for v in forward.tolist()],
            },
            ensure_ascii=True,
        )

    def _emit_answer(self, text: str) -> None:
        if not self.on_answer:
            return
        self.on_answer(text)

    @staticmethod
    def _safe_callback(callback: Optional[Callable[[], None]]) -> None:
        if callback is None:
            return
        callback()
