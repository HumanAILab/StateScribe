# config.py
import os

# ============================================================================
# Server Configuration
# ============================================================================
HOST = '0.0.0.0'
PORT = 1234
MESSAGE_DELIMITER = b"_TAIL"
ACK_MESSAGE = b"ACK"

# ============================================================================
# Google Gemini API Configuration
# ============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
GEMINI_API_TIMEOUT_MS = 30000  # per-request HTTP timeout in milliseconds

# ============================================================================
# Firebase Configuration
# ============================================================================
FIREBASE_CREDENTIALS_PATH = os.environ.get(
    "FIREBASE_CREDENTIALS_PATH",
    "path/to/your-firebase-adminsdk.json",
)
FIREBASE_COLLECTION = "test"
FIREBASE_DOCUMENT = "test_document"
FIREBASE_QUESTION_FIELD = "question"
FIREBASE_ANSWER_FIELD = "response"

# ============================================================================
# Memory Configuration
# ============================================================================
LONG_TERM_MEMORY_BASE_PATH = "benchmark_data"
LTM_LOAD_LATEST_CAPTURE_ONLY = True
LTM_CAPTURE_GAP_SECONDS = 10.0

# ============================================================================
# Data Processing Configuration
# ============================================================================
ORIGINAL_RESOLUTION = (1440, 1920)
TARGET_RESOLUTION = (720, 960)

# ============================================================================
# TTS default Configuration
# ============================================================================
DEFAULT_TTS_RATE = 1.0
DEFAULT_TTS_WPM_AT_HALF = 250  # Too slow when 150 WPM, and not really half.
SENTENCE_INTERVAL_SECONDS = 0.5

# Global pause inserted between two emitted messages by DescriptionBroker.
BROKER_MESSAGE_GAP_SECONDS = 0.5

# ============================================================================
# Visualization Configuration
# ============================================================================
VISUALIZATION_ENABLED = True
VISUALIZATION_HEADLESS = False
VISUALIZATION_BACKEND = "web"  # "desktop" (OpenCV + Open3D) or "web" (browser UI)
VISUALIZATION_ENABLE_3D_RECONSTRUCTION = False
WEB_VIS_HOST = "127.0.0.1"
WEB_VIS_PORT = 8765
WEB_VIS_MESH_MAX_TRIANGLES = 120000

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_FILE = "statescribe.log"
LOG_LEVEL = "WARNING"
CONSOLE_LOG_LEVEL = "WARNING"
FRAME_LOG_FILE = "statescribe_frames.log"
# Per-frame reports are extremely verbose; keep them disabled by default.
# Set back to DEBUG/INFO only when actively debugging frame-level behavior.
FRAME_LOG_LEVEL = "WARNING"
SPEECH_LOG_FILE = "statescribe_speech.log"
SPEECH_LOG_LEVEL = "DEBUG"

# ============================================================================
# StateScribe Pipeline Configuration
# ============================================================================

# Field-of-view-overlap Frame Matching
FRAME_MATCH_PREFILTER_TRANSLATION_M = 1.5  # meters
FRAME_MATCH_PREFILTER_ROTATION_DEG = 40.0  # degrees
FRAME_MATCH_OVERLAP_MIN_MEAN_RATIO = 0.74
FRAME_MATCH_OVERLAP_MIN_DIRECTIONAL_RATIO = 0.65
FRAME_MATCH_OVERLAP_DOWNSAMPLE_RATIO = 8
FRAME_MATCH_OVERLAP_MIN_DEPTH_M = 0.02

# Temporal Clustering
TEMPORAL_CLUSTERING_EPS_SECONDS = 8  # DBSCAN epsilon in seconds
TEMPORAL_CLUSTERING_MIN_SAMPLES = 2  # DBSCAN min_samples
FRAME_CLUSTER_REQUIRE_LIVE_DESCRIBED_TAG = False
FRAME_CLUSTER_LARGE_SIZE_THRESHOLD = 30
FRAME_CLUSTER_FORCE_COMPARE_INTERVAL = 30

FRAME_MATCHING_BRISQUE_MAX_SCORE = 45.0
FRAME_MATCHING_BRISQUE_MODEL_PATH = "components/model/brisque_model_live.yml"
FRAME_MATCHING_BRISQUE_RANGE_PATH = "components/model/brisque_range_live.yml"

# ============================================================================
# VLM Diff Configuration
# ============================================================================
DIFF_VLM_MODEL = "gemini-3-flash-preview"
DIFF_VLM_TEMPERATURE = 1.0
DIFF_VLM_MAX_TOKENS = 1024
DIFF_VLM_DISCARD_IF_SLOWER_THAN_SECONDS = 8.0
DIFF_VLM_HALLUCINATION_FILTER_ENABLED = True

# ============================================================================
# Live Scene Description Configuration
# ============================================================================
LIVE_DESCRIPTION_ENABLED = True
LIVE_DESCRIPTION_MODEL = "gemini-3.1-flash-lite-preview"
LIVE_DESCRIPTION_TEMPERATURE = 1.0
LIVE_DESCRIPTION_MAX_TOKENS = 196
LIVE_DESCRIPTION_INTERVAL_SECONDS = 15.0
LIVE_DESCRIPTION_IMAGE_SIMILARITY_THRESHOLD = 0.8
LIVE_DESCRIPTION_TEXT_SIMILARITY_THRESHOLD = 0.75
LIVE_DESCRIPTION_FRAME_QUEUE_MAXSIZE = 8
LIVE_DESCRIPTION_DISCARD_IF_SLOWER_THAN_SECONDS = 6.0

# Description output mode
# - "legacy": keep current deterministic composer/broker behavior
# - "ai_paraphrase": aggregate live/change updates and paraphrase via LLM
DESCRIPTION_OUTPUT_MODE = "ai_paraphrase"

# AI paraphrase description configuration
AI_PARAPHRASE_DESCRIPTION_ENABLED = True
AI_PARAPHRASE_DESCRIPTION_MODEL = "gemini-3-flash-preview"
AI_PARAPHRASE_DESCRIPTION_TEMPERATURE = 1.0
AI_PARAPHRASE_DESCRIPTION_MAX_TOKENS = 256
AI_PARAPHRASE_PREVIOUS_OUTPUT_COUNT = 1
AI_PARAPHRASE_MAX_CHANGES_PER_SUMMARY = 1
AI_PARAPHRASE_API_LEAD_SECONDS = 2

# ============================================================================
# QA Agent Configuration
# ============================================================================
AGENT_ENABLED = True
AGENT_MODEL = "gemini-3-flash-preview"
AGENT_TEMPERATURE = 1.0
AGENT_MAX_TOKENS = 256
AGENT_THINKING_BUDGET = 1024

# ============================================================================
# Segmentation Configuration
# ============================================================================
FASTSAM_MODEL_PATH = "FastSAM-s.pt"
FASTSAM_BBOX_SHRINK_RATIO = 0.2

# ============================================================================
# Change Memory Configuration
# ============================================================================
# Outlier removal for 3D bounding boxes
BBOX_3D_OUTLIER_STD_THRESHOLD = 2.0  # Standard deviations

# Change memory fusion (geometry-only)
CHANGE_MEMORY_IOU_MODE = "3d"  # "3d" or "xz"
CHANGE_MEMORY_IOU_THRESHOLD = 0.08
CHANGE_MEMORY_IOU_BBOX_EXPANSION_RATIO = 0.20
CHANGE_MEMORY_BBOX_WINDOW = 3
CHANGE_MEMORY_DINO_SIM_THRESHOLD = 0.7
CHANGE_MEMORY_TEXT_SIM_THRESHOLD = 0.65

# Reference-frame tagging
CHANGE_DETECTION_ALWAYS_TAG_REFERENCES = False
CHANGE_DETECTION_REPLAY_RECOVERED_HISTORY = False

# Snapshot feature models
DINO_FEATURE_MODEL_NAME = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
TEXT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

# 2D bbox filtering and mask processing
BBOX_MIN_AREA_RATIO = 0.001
VISIBILITY_MASK_KERNEL_SIZE = 3
VISIBILITY_MASK_DOWNSAMPLE_RATIO = 4
VISIBILITY_MASK_EDGE_TRIM_RATIO = 0.08  # trim this ratio from each image edge in visibility masks
VISIBILITY_MASK_MAX_OUTSIDE_AREA_RATIO = 0.55 # indoor: 0.55, outdoor: 0.65
OCCLUSION_RELIEF_DEPTH_TOLERANCE = 1.00  # DISABLED: meters; ref depth must be this much closer to count as occluder
OCCLUSION_RELIEF_MIN_OCCLUDED_RATIO = 0.30  # filter "appear" when ≥30% of bbox was occluded in ref
MASK_ERODE_KERNEL_SIZE = 3
BBOX_OUTLIER_TRIM_RATIO = 0.1

# ============================================================================
# Description Composer Configuration
# ============================================================================
DESCRIPTION_REPLACEMENT_IOU_THRESHOLD = 0.12

# Options: "metric" or "imperial"
LOCATION_PHRASE_UNIT_SYSTEM = "imperial"

# ============================================================================
# Pipeline Queue Configuration
# ============================================================================
# Maximum concurrent VLM processing
MAX_CONCURRENT_VLM_DIFF = 4
MAX_CONCURRENT_LIVE_DESCRIPTION = 4
MAX_CONCURRENT_AI_SUMMARY = 4

# Queue sizes
FRAME_QUEUE_MAXSIZE = 100
VLM_RESULT_QUEUE_MAXSIZE = 50

# ============================================================================
# Debug Configuration
# ============================================================================
FRAME_CLUSTER_DEBUG_ENABLED = False
FRAME_CLUSTER_DEBUG_DIR = "debug/frame_matching"

VLM_RAW_BBOX_DEBUG_ENABLED = False
VLM_RAW_BBOX_DEBUG_DIR = "debug/vlm_raw_bboxes"

REJECTED_BBOX_DEBUG_ENABLED = False
REJECTED_BBOX_DEBUG_DIR = "debug/rejected_bboxes"
