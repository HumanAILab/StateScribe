# components/visualization/display_manager.py
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from queue import Queue

from components.memory.change_memory import ChangeMemory

logger = logging.getLogger(__name__)

class DisplayManager:
    """
    Manages 2D visualization windows for StateScribe.
    - Image window: current/ref, VLM annotated views, and masks
    - Info window: Change memory and descriptions
    """
    
    def __init__(self):
        self.image_window_name = "StateScribe - Change Detection"
        self.info_window_name = "StateScribe - Memory & Descriptions"
        
        # Display state
        self.current_display_image: Optional[np.ndarray] = None
        self.current_reference_image: Optional[np.ndarray] = None
        self.current_mask_t0: Optional[np.ndarray] = None
        self.current_mask_t1: Optional[np.ndarray] = None
        self.vlm_annotated_t0: Optional[np.ndarray] = None
        self.vlm_annotated_t1: Optional[np.ndarray] = None
        self.descriptions: List[Dict[str, Any]] = []  # List of {timestamp, description}
        self.history: List[Dict[str, Any]] = []
        self.history_index = -1
        # Trackbar disabled temporarily.
        # self._trackbar_ready = False
        self._dirty = False
        self._last_change_memory: Optional[ChangeMemory] = None
        self._pending = Queue()
        
        # Window dimensions
        self.image_width = 480
        self.image_height = 360
        self.info_window_width = 800
        self.info_window_height = 600
        
        logger.debug("DisplayManager initialized")

    def add_description(self, description: str, timestamp: Optional[datetime] = None):
        """Queue a description-only update (for live narration without image update)."""
        if not description:
            return
        self._pending.put({
            "description_only": True,
            "description": description,
            "timestamp": timestamp or datetime.now(),
            "change_memory": None,
        })

    def initialize(self):
        """Create windows early with blank content."""
        blank = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.current_display_image = blank.copy()
        self.current_reference_image = blank.copy()
        self.current_mask_t1 = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        self.current_mask_t0 = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        self.vlm_annotated_t0 = None
        self.vlm_annotated_t1 = None
        self._update_image_window()
        self._update_info_window(self._last_change_memory)
    
    def update_detection_result(
        self,
        current_image: np.ndarray,
        reference_image: np.ndarray,
        mask_t1: np.ndarray,
        mask_t0: np.ndarray,
        description: str,
        timestamp: datetime,
        change_memory: Optional[ChangeMemory] = None,
        vlm_annotated_t0: Optional[np.ndarray] = None,
        vlm_annotated_t1: Optional[np.ndarray] = None
    ):
        self._pending.put({
            "current_image": current_image.copy(),
            "reference_image": reference_image.copy(),
            "mask_t1": mask_t1.copy() if mask_t1 is not None else None,
            "mask_t0": mask_t0.copy() if mask_t0 is not None else None,
            "vlm_annotated_t0": vlm_annotated_t0.copy() if vlm_annotated_t0 is not None else None,
            "vlm_annotated_t1": vlm_annotated_t1.copy() if vlm_annotated_t1 is not None else None,
            "description": description,
            "timestamp": timestamp,
            "change_memory": change_memory,
        })

    def pump(self):
        drained = False
        while True:
            if self._pending.empty():
                break
            item = self._pending.get_nowait()
            drained = True
            if item.get("description_only"):
                self.descriptions.append({
                    "timestamp": item["timestamp"],
                    "description": item["description"],
                })
                maybe_change_memory = item.get("change_memory")
                if maybe_change_memory is not None:
                    self._last_change_memory = maybe_change_memory
                continue

            self.descriptions.append({
                "timestamp": item["timestamp"],
                "description": item["description"],
            })
            self.history.append({
                "current_image": item["current_image"],
                "reference_image": item["reference_image"],
                "mask_t1": item["mask_t1"],
                "mask_t0": item["mask_t0"],
                "vlm_annotated_t0": item["vlm_annotated_t0"],
                "vlm_annotated_t1": item["vlm_annotated_t1"],
                "description": item["description"],
                "timestamp": item["timestamp"],
            })
            self.history_index = len(self.history) - 1
            self._last_change_memory = item.get("change_memory")
        if drained:
            self._dirty = True

        if not self.history:
            if self._dirty:
                self._update_info_window(self._last_change_memory)
                self._dirty = False
            return
        if self.current_display_image is None or self._dirty:
            self._render_history_item(self._last_change_memory)
            self._dirty = False
            return
        self._update_image_window()

    def _render_history_item(self, change_memory: Optional[ChangeMemory] = None):
        if not self.history:
            return
        idx = max(0, min(self.history_index, len(self.history) - 1))
        item = self.history[idx]
        self.current_display_image = item["current_image"]
        self.current_reference_image = item["reference_image"]
        self.current_mask_t1 = item["mask_t1"]
        self.current_mask_t0 = item["mask_t0"]
        self.vlm_annotated_t0 = item["vlm_annotated_t0"]
        self.vlm_annotated_t1 = item["vlm_annotated_t1"]
        self._update_image_window()
        self._update_info_window(change_memory, selected=item)
    
    def _update_image_window(self):
        """Update the image window with current, reference, and mask."""
        if self.current_display_image is None:
            return
        
        current_bgr = cv2.cvtColor(self.current_display_image, cv2.COLOR_RGB2BGR)
        reference_bgr = cv2.cvtColor(self.current_reference_image, cv2.COLOR_RGB2BGR)
        
        current_resized = cv2.resize(current_bgr, (self.image_width, self.image_height))
        reference_resized = cv2.resize(reference_bgr, (self.image_width, self.image_height))
        
        if self.current_mask_t1 is not None:
            mask_t1 = self.current_mask_t1
        else:
            mask_t1 = np.zeros(self.current_display_image.shape[:2], dtype=np.uint8)

        if self.current_mask_t0 is not None:
            mask_t0 = self.current_mask_t0
        else:
            mask_t0 = np.zeros(self.current_reference_image.shape[:2], dtype=np.uint8)

        mask_t1_resized = cv2.resize(mask_t1, (self.image_width, self.image_height))
        mask_t0_resized = cv2.resize(mask_t0, (self.image_width, self.image_height))

        mask_t1_colored = np.zeros_like(current_resized)
        mask_t1_colored[mask_t1_resized > 0] = [0, 255, 0]
        mask_t0_colored = np.zeros_like(reference_resized)
        mask_t0_colored[mask_t0_resized > 0] = [0, 255, 0]

        mask_t1_overlay = cv2.addWeighted(current_resized, 0.7, mask_t1_colored, 0.3, 0)
        mask_t0_overlay = cv2.addWeighted(reference_resized, 0.7, mask_t0_colored, 0.3, 0)
        
        current_labeled = self._add_label(current_resized, "Current Frame (t1)", (255, 255, 255))
        reference_labeled = self._add_label(reference_resized, "Reference Frame (t0)", (255, 255, 255))

        if self.vlm_annotated_t1 is not None:
            vlm_t1_bgr = cv2.cvtColor(self.vlm_annotated_t1, cv2.COLOR_RGB2BGR)
            vlm_t1_resized = cv2.resize(vlm_t1_bgr, (self.image_width, self.image_height))
        else:
            vlm_t1_resized = np.zeros_like(current_resized)

        if self.vlm_annotated_t0 is not None:
            vlm_t0_bgr = cv2.cvtColor(self.vlm_annotated_t0, cv2.COLOR_RGB2BGR)
            vlm_t0_resized = cv2.resize(vlm_t0_bgr, (self.image_width, self.image_height))
        else:
            vlm_t0_resized = np.zeros_like(reference_resized)

        vlm_t1_labeled = self._add_label(vlm_t1_resized, "Current Gemini", (0, 255, 255))
        vlm_t0_labeled = self._add_label(vlm_t0_resized, "Reference Gemini", (0, 255, 255))

        mask_t1_labeled = self._add_label(mask_t1_overlay, "Current FastSAM Mask", (0, 255, 0))
        mask_t0_labeled = self._add_label(mask_t0_overlay, "Reference FastSAM Mask", (0, 255, 0))

        row1 = np.hstack([current_labeled, reference_labeled])
        row2 = np.hstack([vlm_t1_labeled, vlm_t0_labeled])
        row3 = np.hstack([mask_t1_labeled, mask_t0_labeled])

        combined = np.vstack([row1, row2, row3])
        
        cv2.imshow(self.image_window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("a"), ord("j")):
            self.history_index = max(0, self.history_index - 1)
            self._render_history_item(self._last_change_memory)
        elif key in (ord("d"), ord("l")):
            self.history_index = min(len(self.history) - 1, self.history_index + 1)
            self._render_history_item(self._last_change_memory)
    
    def _update_info_window(self, change_memory: Optional[ChangeMemory] = None, selected: Optional[Dict[str, Any]] = None):
        """Update the info window with change memory and descriptions."""
        canvas = np.ones((self.info_window_height, self.info_window_width, 3), dtype=np.uint8) * 255
        
        y_offset = 20
        line_height = 22

        if selected:
            ts = selected.get("timestamp")
            desc = selected.get("description", "")
            ts_str = ts.strftime("%H:%M:%S") if ts else "unknown"
            cv2.putText(canvas, f"Selected: {ts_str} - {desc[:60]}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            y_offset += line_height + 5
        
        if change_memory:
            cv2.putText(canvas, "Change Memory", (10, y_offset), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
            y_offset += line_height + 5
            
            all_objects = change_memory.get_all_objects()
            active_objects = change_memory.get_active_objects()
            
            cv2.putText(canvas, f"Total: {len(all_objects)} objects ({len(active_objects)} active)", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
            y_offset += line_height
            
            for obj in all_objects[-5:]:
                latest = obj.get_latest_snapshot()
                if latest:
                    desc_short = latest.description[:45] + "..." if len(latest.description) > 45 else latest.description
                    status = "✓" if not obj.is_disappeared() else "✗"
                    cv2.putText(canvas, f"  {status} {obj.object_id}: {desc_short}", (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
                    y_offset += line_height - 5
            
            y_offset += 10
        
        cv2.line(canvas, (10, y_offset), (self.info_window_width - 10, y_offset), (200, 200, 200), 1)
        y_offset += 15
        
        cv2.putText(canvas, "Change Descriptions", (10, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        y_offset += line_height + 10
        
        recent_descriptions = self.descriptions[-10:]
        
        for desc_info in recent_descriptions:
            timestamp = desc_info['timestamp']
            description = desc_info['description']
            
            time_str = timestamp.strftime("%H:%M:%S")
            
            cv2.putText(canvas, f"[{time_str}]", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)
            
            wrapped_lines = self._wrap_text(description, 85)
            for line in wrapped_lines:
                y_offset += line_height
                if y_offset > self.info_window_height - 30:
                    break
                cv2.putText(canvas, line, (100, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            
            y_offset += line_height - 2
            
            if y_offset > self.info_window_height - 30:
                break
        
        cv2.imshow(self.info_window_name, canvas)
        cv2.waitKey(1)
    
    def update_memory_display(self, change_memory: ChangeMemory):
        self._update_info_window(change_memory)
    
    def _add_label(self, image: np.ndarray, label: str, color: tuple) -> np.ndarray:
        labeled = image.copy()
        cv2.rectangle(labeled, (0, 0), (self.image_width, 40), (0, 0, 0), -1)
        cv2.putText(labeled, label, (10, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return labeled
    
    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        return lines if lines else [text[:max_chars]]
    
    def clear(self):
        self.current_display_image = None
        self.current_reference_image = None
        self.current_mask_t0 = None
        self.current_mask_t1 = None
        self.vlm_annotated_t0 = None
        self.vlm_annotated_t1 = None
        self.descriptions.clear()
        self.history.clear()
        self.history_index = -1
        self._dirty = False
    
    def close(self):
        cv2.destroyAllWindows()
        logger.debug("DisplayManager closed")
