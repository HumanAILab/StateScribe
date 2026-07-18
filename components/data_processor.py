# components/data_processor.py
import json
import base64
import numpy as np
import cv2
import logging
from queue import Queue
from datetime import datetime
from threading import Thread
from time import perf_counter
from typing import Any, Optional

from components.memory.frame import Frame
from components.memory.memory_manager import MemoryManager
from components.feature_factory import FeatureFactory
from config import TARGET_RESOLUTION, ORIGINAL_RESOLUTION

logger = logging.getLogger(__name__)

class DataProcessor(Thread):
    """
    Processes raw data from the TCP server, creates Frame objects,
    and passes them to the MemoryManager.
    """
    def __init__(
        self,
        input_queue: Queue,
        memory_manager: MemoryManager,
        change_pipeline=None,
        live_pipeline=None,
        feature_factory: Optional[FeatureFactory] = None,
    ):
        super().__init__(daemon=True)
        self.input_queue = input_queue
        self.memory_manager = memory_manager
        self.change_pipeline = change_pipeline
        self.live_pipeline = live_pipeline
        self.feature_factory = feature_factory
        self.running = False

    def run(self):
        """The main loop for the data processing thread."""
        self.running = True
        logger.debug("DataProcessor started.")
        while self.running:
            raw_item = self.input_queue.get()
            if raw_item is None: # Sentinel for session end
                self.memory_manager.end_session()
                logger.debug("Client session ended. Processor is resetting.")
                continue

            ingress_wall: Optional[datetime] = None
            ingress_perf: Optional[float] = None
            raw_data: Any = raw_item
            if isinstance(raw_item, dict):
                raw_data = raw_item.get("payload")
                candidate_wall = raw_item.get("ingress_wall")
                candidate_perf = raw_item.get("ingress_perf")
                if isinstance(candidate_wall, datetime):
                    ingress_wall = candidate_wall
                if isinstance(candidate_perf, (int, float)):
                    ingress_perf = float(candidate_perf)

            if not isinstance(raw_data, (bytes, bytearray)):
                logger.debug("Unsupported payload type in DataProcessor: %s", type(raw_data).__name__)
                continue

            self.process_payload(
                bytes(raw_data),
                ingress_wall=ingress_wall,
                ingress_perf=ingress_perf,
            )
        logger.debug("DataProcessor stopped.")

    def stop(self):
        self.running = False

    def process_payload(
        self,
        message_data: bytes,
        ingress_wall: Optional[datetime] = None,
        ingress_perf: Optional[float] = None,
    ):
        """Decodes and processes a single message payload."""
        payload_str = message_data.decode('utf-8')
        payload = json.loads(payload_str)

        # Extract metadata
        timestamp = datetime.now()
        world_name = payload.get("worldName", "default_world")
        camera_data = payload.get("cameraPose")

        # Decode data
        rgb_image = self._decode_rgb(payload.get("rgbImage"))
        depth_map = self._decode_depth(payload.get("depthMap"))
        confidence_map = self._decode_confidence(payload.get("confidenceMap"))
        
        # Extract camera parameters
        intrinsics = self._extract_intrinsics(camera_data)
        pose_matrix = self._extract_pose(camera_data)

        if rgb_image is None or depth_map is None or intrinsics is None:
            logger.debug("Incomplete data received, skipping frame.")
            return

        # Process images to match intrinsics resolution
        processed_rgb = cv2.resize(rgb_image, TARGET_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        processed_depth = cv2.resize(depth_map, TARGET_RESOLUTION, interpolation=cv2.INTER_NEAREST)

        # Filter depth using confidence map
        if confidence_map is not None:
            resized_confidence = cv2.resize(confidence_map, TARGET_RESOLUTION, interpolation=cv2.INTER_NEAREST)
            processed_depth[resized_confidence == 0] = 0 # Keep only high-confidence points

        dino_feature = None
        if self.feature_factory is not None:
            try:
                dino_feature = self.feature_factory.compute_dino_feature(processed_rgb)
            except Exception:
                logger.exception("Failed to compute full-frame DINO feature.")

        # Create a Frame object
        frame = Frame(
            timestamp=timestamp,
            world_name=world_name,
            rgb_image=processed_rgb,
            depth_map=processed_depth,
            depth_map_original=cv2.resize(depth_map, TARGET_RESOLUTION, interpolation=cv2.INTER_NEAREST),
            confidence_map=confidence_map,
            pose_matrix=pose_matrix,
            intrinsics=intrinsics,
            # Historical field name: this now stores full-frame DINO features.
            clip_embedding=dino_feature,
        )

        # Add to memory
        self.memory_manager.add_frame(frame)

        # Send to change detection pipeline
        if self.change_pipeline:
            self.change_pipeline.add_frame(
                frame,
                ingress_wall=ingress_wall or datetime.now(),
                ingress_perf=float(ingress_perf) if isinstance(ingress_perf, (int, float)) else perf_counter(),
            )
        if self.live_pipeline:
            self.live_pipeline.add_frame(frame)

    def _decode_rgb(self, rgb_payload):
        if not rgb_payload or "data" not in rgb_payload: return None
        img_data = base64.b64decode(rgb_payload["data"])
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _decode_depth(self, depth_payload):
        if not depth_payload or "data" not in depth_payload: return None
        depth_binary = base64.b64decode(depth_payload["data"])
        depth_array = np.frombuffer(depth_binary, dtype=np.float32)
        w, h = depth_payload["width"], depth_payload["height"]
        
        expected = w * h
        if depth_array.size != expected:
            logger.debug(f"Depth data size mismatch. Expected {expected}, got {depth_array.size}.")
            depth_array = depth_array[:expected] # Attempt to recover
        
        depth_image = depth_array.reshape((h, w))
        return np.rot90(depth_image, k=-1)

    def _decode_confidence(self, conf_payload):
        if not conf_payload or "data" not in conf_payload: return None
        conf_binary = base64.b64decode(conf_payload["data"])
        conf_array = np.frombuffer(conf_binary, dtype=np.uint8)
        w, h = conf_payload["width"], conf_payload["height"]
        
        expected = w * h
        if conf_array.size != expected:
            logger.debug(f"Confidence map size mismatch. Expected {expected}, got {conf_array.size}.")
            conf_array = conf_array[:expected]

        conf_map = conf_array.reshape((h, w))
        return np.rot90(conf_map, k=-1)

    def _extract_intrinsics(self, camera_data):
        if not camera_data or 'intrinsics' not in camera_data: return None
        intrinsics = camera_data['intrinsics']
        fx, fy = intrinsics[0], intrinsics[4] # should swap the order, but fx and fy are the same. ¯\_(ツ)_/¯
        cx, cy = intrinsics[7], intrinsics[6] # the swapped order is intentional, do not change this order
        
        scale_x = TARGET_RESOLUTION[0] / ORIGINAL_RESOLUTION[0]
        scale_y = TARGET_RESOLUTION[1] / ORIGINAL_RESOLUTION[1]
        
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
        
        return np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])

    def _extract_pose(self, camera_data):
        if not camera_data or 'transform' not in camera_data: return None
        transform = np.array(camera_data['transform']).reshape(4, 4, order='F')
        rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        invert = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return transform @ rotation @ invert
