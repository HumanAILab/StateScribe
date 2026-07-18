# main.py
import logging
import threading
import time
from queue import Queue
import signal

from logging_config import setup_logging
from config import (
    DESCRIPTION_OUTPUT_MODE,
    VISUALIZATION_ENABLED,
    VISUALIZATION_BACKEND,
)
from components.tcp_server import TCPServer
from components.data_processor import DataProcessor
from components.memory.memory_manager import MemoryManager
from components.change_detection_pipeline import ChangeDetectionPipeline
from components.live_description_pipeline import LiveDescriptionPipeline
from components.ai_description_pipeline import AIParaphraseDescriptionPipeline
from components.feature_factory import FeatureFactory
from components.description_broker import DescriptionBroker
from components.logging.world_log_writer import WorldStructuredLogWriter
from components.visualization.simple_visualizer import Open3DVisualizer
from components.visualization.display_manager import DisplayManager
from components.visualization.web_display_manager import WebDisplayManager
from components.visualization.web_visualizer import WebVisualizer
from components.memory.frame import Frame
from components.vlm.tools import AgentTools
from components.vlm.gemini_agent import GeminiAgent
from components.firebase_listener import FirebaseListener

# Setup logging first
setup_logging()

# Suppress noisy third-party loggers
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('google_genai.models').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('firebase').setLevel(logging.WARNING)
logging.getLogger('firebase_admin').setLevel(logging.WARNING)
logging.getLogger('grpc').setLevel(logging.WARNING)
logging.getLogger('proto').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('h11').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
_APP = None

class AppController:
    """
    Main application controller for StateScribe.
    Coordinates change detection pipeline and visualization.
    """
    def __init__(self):
        self.running = True
        self._shutdown_called = False
        self.raw_data_queue = Queue()
        
        # Core components
        self.memory_manager = MemoryManager()
        self.visualization_backend = (VISUALIZATION_BACKEND or "desktop").strip().lower()
        if self.visualization_backend not in {"desktop", "web"}:
            logger.warning("Unknown VISUALIZATION_BACKEND=%s. Falling back to desktop.", VISUALIZATION_BACKEND)
            self.visualization_backend = "desktop"

        # Display manager for 2D visualization
        if self.visualization_backend == "web":
            self.display_manager = WebDisplayManager(self.memory_manager)
        else:
            self.display_manager = DisplayManager()

        self.feature_factory = FeatureFactory()
        self.description_broker = DescriptionBroker(
            emit_func=self._emit_description,
        )
        self.world_log_writer = WorldStructuredLogWriter(
            current_world_name_getter=lambda: self.memory_manager.current_world_name,
        )

        self.ai_description_pipeline = None
        self.description_mode = "legacy"
        
        # Change detection pipeline
        self.change_pipeline = ChangeDetectionPipeline(
            self.memory_manager,
            self.display_manager,
            on_description=self.description_broker.publish_change,
            feature_factory=self.feature_factory,
        )

        live_description_callback = self.description_broker.publish_live
        if DESCRIPTION_OUTPUT_MODE == "ai_paraphrase":
            candidate = AIParaphraseDescriptionPipeline(
                memory_manager=self.memory_manager,
                change_memory=self.change_pipeline.change_memory,
                on_description=self.description_broker.publish_change,
                speech_busy_until_getter=self.description_broker.get_busy_until,
            )
            if candidate.enabled:
                self.ai_description_pipeline = candidate
                self.description_mode = "ai_paraphrase"
                live_description_callback = self.ai_description_pipeline.add_live_description
                self.change_pipeline.on_description = None
                self.change_pipeline.on_change_snapshot = self.ai_description_pipeline.add_change_snapshot
                self.change_pipeline.use_ai_paraphrase = True
                logger.debug("Description mode: ai_paraphrase")
            else:
                logger.warning("Description mode ai_paraphrase requested but unavailable; falling back to legacy.")
        else:
            logger.debug("Description mode: legacy")

        self.live_pipeline = LiveDescriptionPipeline(
            on_description=live_description_callback,
            feature_factory=self.feature_factory,
            on_emit_success=self._on_live_description_emitted,
            downstream_manages_final_emit=self.description_mode == "ai_paraphrase",
        )
        self._pipelines_paused_for_vqa = False
        self._resume_timer_lock = threading.Lock()
        self._resume_timer = None

        self.agent_tools = AgentTools(self.memory_manager, self.change_pipeline.change_memory)
        self.agent = GeminiAgent(
            memory_manager=self.memory_manager,
            change_memory=self.change_pipeline.change_memory,
            agent_tools=self.agent_tools,
            on_answer=self._emit_agent_answer,
            on_answer_start=self._on_agent_answer_start,
            on_answer_end=self._on_agent_answer_end,
        )

        # Firebase (optional) for publishing descriptions and receiving questions
        self.firebase_client = None
        if FirebaseListener:
            try:
                self.firebase_client = FirebaseListener(self._on_question_received)
                logger.debug("Firebase client ready for description updates and questions.")
            except Exception:
                self.firebase_client = None
                logger.warning("Firebase client unavailable; continuing without remote Q&A.", exc_info=True)
        else:
            logger.debug("FirebaseListener unavailable; descriptions/questions are unavailable.")
        
        # Visualizer for 3D display
        if VISUALIZATION_ENABLED:
            if self.visualization_backend == "web":
                state_store = getattr(self.display_manager, "state_store", None)
                self.visualizer = WebVisualizer(
                    self.memory_manager,
                    self.change_pipeline.change_memory,
                    state_store=state_store,
                )
            else:
                self.visualizer = Open3DVisualizer(self.memory_manager, self.change_pipeline.change_memory)
        else:
            self.visualizer = None
        
        # Data ingestion
        self.data_processor = DataProcessor(
            self.raw_data_queue, 
            self.memory_manager,
            self.change_pipeline,
            self.live_pipeline,
            feature_factory=self.feature_factory,
        )
        
        self.tcp_server = TCPServer(self.raw_data_queue)

    def _emit_description(self, description: str):
        if self.firebase_client:
            self.firebase_client.update_answer(description)
        else:
            logger.debug("[description] %s", description)

    def _on_live_description_emitted(self, frame: Frame):
        self.change_pipeline.mark_live_described_frame(frame.timestamp)
        set_live_frame = getattr(self.display_manager, "set_live_describing_frame", None)
        if callable(set_live_frame):
            try:
                set_live_frame(frame)
            except Exception:
                logger.exception("Failed to update live describing frame for visualizer.")

    def _emit_agent_answer(self, answer: str) -> bool:
        accepted = self.description_broker.publish_agent(answer)
        if accepted:
            self._schedule_pipeline_resume(self.description_broker.project_release_time(answer))
        return accepted

    def _set_vqa_pipeline_paused(self, paused: bool) -> None:
        next_state = bool(paused)
        if self._pipelines_paused_for_vqa == next_state:
            return
        self._pipelines_paused_for_vqa = next_state
        self.change_pipeline.set_paused(next_state)
        self.live_pipeline.set_paused(next_state)
        if self.ai_description_pipeline is not None:
            self.ai_description_pipeline.set_paused(next_state)
        logger.debug("Scene pipelines %s for VQA", "paused" if next_state else "resumed")

    def _on_agent_answer_start(self) -> None:
        self._set_vqa_pipeline_paused(True)
        self.description_broker.begin_agent_response()

    def _on_agent_answer_end(self) -> None:
        self.description_broker.end_agent_response()

    def _on_question_received(self, question: str):
        clean_question = (question or "").strip()
        if not clean_question:
            return
        self._cancel_pipeline_resume_timer()
        self._set_vqa_pipeline_paused(True)
        self.description_broker.begin_agent_response()
        accepted = self.agent.answer_question(clean_question)
        if not accepted:
            self.description_broker.cancel_agent_response()
            self._set_vqa_pipeline_paused(False)

    def _schedule_pipeline_resume(self, release_at: float) -> None:
        delay = max(0.0, float(release_at) - time.perf_counter())

        def _resume_scene() -> None:
            with self._resume_timer_lock:
                self._resume_timer = None
            if self._shutdown_called:
                return
            self._set_vqa_pipeline_paused(False)

        timer = threading.Timer(delay, _resume_scene)
        timer.daemon = True
        with self._resume_timer_lock:
            if self._resume_timer is not None:
                self._resume_timer.cancel()
            self._resume_timer = timer
        timer.start()

    def _cancel_pipeline_resume_timer(self) -> None:
        with self._resume_timer_lock:
            if self._resume_timer is None:
                return
            self._resume_timer.cancel()
            self._resume_timer = None

    def run(self):
        """Starts all components and enters the main application loop."""
        logger.debug("Starting StateScribe...")
        self.world_log_writer.start()

        # Initialize 2D windows early
        self.display_manager.initialize()

        # Initialize visualizer BEFORE starting background threads.
        # Open3D's GLFW event loop must be fully ready before other threads
        # (especially sklearn/OpenMP) start initializing their own threading,
        # which can interfere with the rendering pipeline on WSL2.
        if self.visualizer:
            self.visualizer.initialize()

        # Start background components AFTER visualizer is ready
        self.tcp_server.start()
        self.data_processor.start()
        if self.ai_description_pipeline:
            self.ai_description_pipeline.start()
        self.change_pipeline.start()
        self.live_pipeline.start()
        if self.firebase_client:
            self.firebase_client.start()

        # Main Loop
        while self.running:
            # Main thread handles visualizer updates
            if self.visualizer:
                if not self.visualizer.update():
                    logger.debug("Visualizer window closed. Shutting down.")
                    self.running = False

            self.display_manager.pump()
            
            time.sleep(0.01)

        self.shutdown()

    def shutdown(self):
        """Gracefully shut down all components."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        logger.debug("Initiating shutdown...")
        self.running = False
        self._cancel_pipeline_resume_timer()
        
        # Stop components
        self.change_pipeline.stop()
        self.live_pipeline.stop()
        if self.ai_description_pipeline:
            self.ai_description_pipeline.stop()
        self.agent.stop()
        self.description_broker.stop()
        if self.firebase_client:
            self.firebase_client.stop()
        if self.visualizer:
            self.visualizer.destroy()
        self.display_manager.close()
        self.data_processor.stop()
        self.raw_data_queue.put_nowait(None)
        self.tcp_server.stop()
        
        # Wait for threads
        self.data_processor.join(timeout=5)
        self.change_pipeline.join(timeout=10)
        self.live_pipeline.join(timeout=5)
        if self.ai_description_pipeline:
            self.ai_description_pipeline.join(timeout=5)
        self.feature_factory.close()
        
        logger.debug("StateScribe shut down complete.")
        self.world_log_writer.stop()

def signal_handler(sig, frame):
    """Handle signals for graceful shutdown."""
    logger.debug(f"Signal {sig} received. Shutting down.")
    if _APP:
        _APP.shutdown()
        return
    raise KeyboardInterrupt

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = AppController()
    _APP = app
    app.run()
