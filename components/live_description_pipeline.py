import logging
import time
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from queue import Queue
from time import perf_counter
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from google import genai
from google.genai import types

from components.feature_factory import FeatureFactory
from components.memory.frame import Frame
from components.speech_timing import estimate_tts_duration_sec
from config import (
    GEMINI_API_KEY,
    GEMINI_API_TIMEOUT_MS,
    LIVE_DESCRIPTION_ENABLED,
    LIVE_DESCRIPTION_MODEL,
    LIVE_DESCRIPTION_TEMPERATURE,
    LIVE_DESCRIPTION_MAX_TOKENS,
    LIVE_DESCRIPTION_INTERVAL_SECONDS,
    LIVE_DESCRIPTION_IMAGE_SIMILARITY_THRESHOLD,
    LIVE_DESCRIPTION_TEXT_SIMILARITY_THRESHOLD,
    LIVE_DESCRIPTION_FRAME_QUEUE_MAXSIZE,
    LIVE_DESCRIPTION_DISCARD_IF_SLOWER_THAN_SECONDS,
    DEFAULT_TTS_RATE,
    SENTENCE_INTERVAL_SECONDS,
    MAX_CONCURRENT_LIVE_DESCRIPTION,
)
from prompts import LIVE_SCENE_SYSTEM_PROMPT, LIVE_SCENE_USER_PROMPT

logger = logging.getLogger(__name__)
speech_logger = logging.getLogger("statescribe.speech")


class LiveDescriptionPipeline(threading.Thread):
    def __init__(
        self,
        on_description: Optional[Callable[[str], bool]] = None,
        feature_factory: Optional[FeatureFactory] = None,
        on_emit_success: Optional[Callable[[Frame], None]] = None,
        downstream_manages_final_emit: bool = False,
    ):
        super().__init__(daemon=True)
        self.on_description = on_description
        self.feature_factory = feature_factory if feature_factory is not None else FeatureFactory()
        self._owns_feature_factory = feature_factory is None
        self.on_emit_success = on_emit_success
        self.downstream_manages_final_emit = bool(downstream_manages_final_emit)
        self.frame_queue: Queue[Frame] = Queue(maxsize=LIVE_DESCRIPTION_FRAME_QUEUE_MAXSIZE)
        self.running = False
        self._paused = False
        self.enabled = bool(LIVE_DESCRIPTION_ENABLED and GEMINI_API_KEY)
        self.latest_frame: Optional[Frame] = None
        self._last_submitted_frame_id: Optional[int] = None
        self._submit_seq = 0
        self._latest_completed_seq = 0
        self.next_emit_at = perf_counter() + SENTENCE_INTERVAL_SECONDS
        self.force_emit_interval_seconds = max(0.0, float(LIVE_DESCRIPTION_INTERVAL_SECONDS))
        self.next_force_emit_at = perf_counter() + self.force_emit_interval_seconds
        self.last_image_feature: Optional[np.ndarray] = None
        self.last_text_embedding: Optional[np.ndarray] = None
        self.client = None
        self._live_executor: Optional[ThreadPoolExecutor] = None
        self._max_in_flight_jobs = max(1, int(MAX_CONCURRENT_LIVE_DESCRIPTION))

        if not LIVE_DESCRIPTION_ENABLED:
            logger.debug("LiveDescriptionPipeline disabled by config.")
            return
        if not GEMINI_API_KEY:
            logger.debug("LiveDescriptionPipeline disabled: GEMINI_API_KEY not set.")
            return

        self.client = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": GEMINI_API_TIMEOUT_MS})
        self._live_executor = ThreadPoolExecutor(max_workers=self._max_in_flight_jobs)
        logger.debug("LiveDescriptionPipeline initialized.")

    def set_paused(self, paused: bool) -> None:
        next_state = bool(paused)
        if self._paused == next_state:
            return
        self._paused = next_state
        logger.debug("LiveDescriptionPipeline %s for VQA", "paused" if next_state else "resumed")

    def add_frame(self, frame: Frame) -> None:
        if not self.enabled or self._paused:
            return
        if self.frame_queue.full():
            # Drop oldest frame to make room
            if not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        self.frame_queue.put_nowait(frame)

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        if not self.enabled:
            return
        self.running = True
        logger.debug("LiveDescriptionPipeline started.")
        pending_jobs: Dict[Future, Dict[str, Any]] = {}

        try:
            while self.running:
                self._drain_latest_frame()
                self._collect_live_results(pending_jobs)

                frame = self.latest_frame
                if frame is None:
                    time.sleep(0.05)
                    continue
                if self._last_submitted_frame_id == id(frame):
                    time.sleep(0.01)
                    continue

                now = perf_counter()
                force_emit_due = self._is_force_emit_due(now)
                if now < self.next_emit_at and not force_emit_due:
                    time.sleep(0.01)
                    continue

                image_feature = getattr(frame, "clip_embedding", None)
                dino_sec = 0.0
                if image_feature is None:
                    self._last_submitted_frame_id = id(frame)
                    speech_logger.debug("[live_timing] status=skip_missing_frame_feature")
                    continue

                sim_image = self._cosine_similarity(image_feature, self.last_image_feature)
                if not force_emit_due and sim_image is not None and sim_image >= LIVE_DESCRIPTION_IMAGE_SIMILARITY_THRESHOLD:
                    self._last_submitted_frame_id = id(frame)
                    speech_logger.debug(f"[live_timing] status=skip_image_sim dino={dino_sec:.3f}s sim_image={sim_image:.3f}")
                    continue

                if self._live_executor is None:
                    time.sleep(0.05)
                    continue
                if len(pending_jobs) >= self._max_in_flight_jobs:
                    time.sleep(0.01)
                    continue

                self._submit_seq += 1
                seq = self._submit_seq
                future = self._live_executor.submit(
                    self._build_live_candidate,
                    frame.rgb_image,
                )
                self.last_image_feature = image_feature
                if force_emit_due:
                    self._advance_force_emit_deadline(now)
                pending_jobs[future] = {
                    "sequence": seq,
                    "frame": frame,
                    "force_emit": force_emit_due,
                    "image_feature": image_feature,
                    "emit_clock": now,
                    "dino_sec": dino_sec,
                }
                self._last_submitted_frame_id = id(frame)
                self._collect_live_results(pending_jobs)
        finally:
            if self._live_executor is not None:
                self._live_executor.shutdown(wait=False)
                self._live_executor = None
            if self._owns_feature_factory:
                self.feature_factory.close()
            logger.debug("LiveDescriptionPipeline stopped.")

    def _drain_latest_frame(self) -> None:
        if self.frame_queue.empty():
            return
        frame = self.frame_queue.get_nowait()
        self.latest_frame = frame
        while not self.frame_queue.empty():
            frame = self.frame_queue.get_nowait()
            self.latest_frame = frame

    def _is_force_emit_due(self, now: float) -> bool:
        return self.force_emit_interval_seconds > 0.0 and now >= self.next_force_emit_at

    def _advance_force_emit_deadline(self, reference_now: float) -> None:
        if self.force_emit_interval_seconds <= 0.0:
            return
        self.next_force_emit_at = reference_now + self.force_emit_interval_seconds

    def _build_live_candidate(self, image: Optional[np.ndarray]) -> Dict[str, Any]:
        if image is None:
            return {
                "text": "",
                "text_embedding": None,
                "llm_sec": 0.0,
                "text_emb_sec": 0.0,
            }
        llm_started = perf_counter()
        text = self._generate_live_description(image)
        llm_sec = perf_counter() - llm_started
        if not text:
            return {
                "text": "",
                "text_embedding": None,
                "llm_sec": llm_sec,
                "text_emb_sec": 0.0,
            }

        text_emb_started = perf_counter()
        text_embedding = self.feature_factory.compute_text_feature(text)
        text_emb_sec = perf_counter() - text_emb_started
        return {
            "text": text,
            "text_embedding": text_embedding,
            "llm_sec": llm_sec,
            "text_emb_sec": text_emb_sec,
        }

    def _collect_live_results(self, pending_jobs: Dict[Future, Dict[str, Any]]) -> None:
        if not pending_jobs:
            return
        done, _ = wait(tuple(pending_jobs.keys()), timeout=0.0, return_when=FIRST_COMPLETED)
        if not done:
            return

        completed: list[tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        for future in done:
            meta = pending_jobs.pop(future, None)
            if meta is None:
                continue
            try:
                candidate = future.result()
            except Exception:
                logger.exception("Failed to build live description candidate.")
                continue

            sequence = int(meta.get("sequence", -1))
            completed.append((sequence, candidate, meta))

        completed.sort(key=lambda item: item[0], reverse=True)
        for sequence, candidate, meta in completed:
            if sequence < self._latest_completed_seq:
                speech_logger.debug(
                    f"[live_timing] status=drop_stale sequence={sequence} latest_completed={self._latest_completed_seq}"
                )
                continue

            self._latest_completed_seq = max(self._latest_completed_seq, sequence)
            self._apply_live_candidate(candidate, meta)

    def _apply_live_candidate(self, candidate: Dict[str, Any], meta: Dict[str, Any]) -> None:
        text = str(candidate.get("text", "")).strip()
        llm_sec = float(candidate.get("llm_sec", 0.0) or 0.0)
        text_emb_sec = float(candidate.get("text_emb_sec", 0.0) or 0.0)
        dino_sec = float(meta.get("dino_sec", 0.0) or 0.0)
        slow_llm_threshold_s = float(LIVE_DESCRIPTION_DISCARD_IF_SLOWER_THAN_SECONDS)

        if slow_llm_threshold_s > 0.0 and llm_sec > slow_llm_threshold_s:
            speech_logger.debug(
                f"[live_timing] status=drop_slow dino={dino_sec:.3f}s llm={llm_sec:.3f}s text_emb={text_emb_sec:.3f}s threshold={slow_llm_threshold_s:.3f}s"
            )
            return

        if not text:
            speech_logger.debug(f"[live_timing] status=skip_empty_text dino={dino_sec:.3f}s llm={llm_sec:.3f}s")
            return

        force_emit = bool(meta.get("force_emit", False))
        text_embedding = candidate.get("text_embedding")
        sim_text = self._cosine_similarity(text_embedding, self.last_text_embedding)

        if not force_emit and sim_text is not None and sim_text >= LIVE_DESCRIPTION_TEXT_SIMILARITY_THRESHOLD:
            speech_logger.debug(
                f"[live_timing] status=skip_text_sim dino={dino_sec:.3f}s llm={llm_sec:.3f}s text_emb={text_emb_sec:.3f}s sim_text={sim_text:.3f}"
            )
            return

        emit_started = perf_counter()
        emitted = self._emit_description(text)
        emit_sec = perf_counter() - emit_started
        tts_sec = estimate_tts_duration_sec(text, rate=DEFAULT_TTS_RATE)

        speech_logger.debug(
            f"[live_timing] status={'emitted' if emitted else 'broker_reject'} dino={dino_sec:.3f}s llm={llm_sec:.3f}s text_emb={text_emb_sec:.3f}s broker={emit_sec:.3f}s tts_est={tts_sec:.2f}s text={text}"
        )

        if emitted:
            self.last_text_embedding = text_embedding
            prev_time_at = self.next_emit_at
            delay = SENTENCE_INTERVAL_SECONDS if self.downstream_manages_final_emit else tts_sec + SENTENCE_INTERVAL_SECONDS
            self.next_emit_at = perf_counter() + delay
            logger.debug(f"Next emit at: {self.next_emit_at}, prev time at: {prev_time_at}")
            emit_clock = meta.get("emit_clock")
            emit_ref = float(emit_clock) if isinstance(emit_clock, (int, float)) else perf_counter()
            self._advance_force_emit_deadline(emit_ref)
            frame = meta.get("frame")
            if isinstance(frame, Frame):
                self._notify_emit_success(frame)

    def _emit_description(self, description: str) -> bool:
        if not self.on_description:
            return False
        result = self.on_description(description)
        if isinstance(result, bool):
            return result
        return True

    def _notify_emit_success(self, frame: Frame) -> None:
        if self.on_emit_success is None:
            return
        self.on_emit_success(frame)

    def _generate_live_description(self, image: np.ndarray) -> str:
        if self.client is None:
            return ""
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr)
        if not ok:
            return ""

        part = types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg")
        try:
            response = self.client.models.generate_content(
                model=LIVE_DESCRIPTION_MODEL,
                contents=[part, LIVE_SCENE_USER_PROMPT],
                config=types.GenerateContentConfig(
                    temperature=LIVE_DESCRIPTION_TEMPERATURE,
                    max_output_tokens=LIVE_DESCRIPTION_MAX_TOKENS,
                    system_instruction=LIVE_SCENE_SYSTEM_PROMPT,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception:
            logger.warning("Live description API call failed.", exc_info=True)
            return ""
        text = (response.text or "").strip()
        if not text:
            return ""
        return " ".join(text.split())

    @staticmethod
    def _cosine_similarity(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> Optional[float]:
        if vec_a is None or vec_b is None:
            return None
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
        if a.shape != b.shape:
            return None
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-9 or nb <= 1e-9:
            return None
        return float(np.dot(a, b) / (na * nb))
