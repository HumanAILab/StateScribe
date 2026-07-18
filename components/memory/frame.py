# components/memory/frame.py
from dataclasses import dataclass
from typing import Optional
import numpy as np
from datetime import datetime

@dataclass
class Frame:
    """
    Represents a single frame of data captured from the client.
    """
    timestamp: datetime
    world_name: str
    rgb_image: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    depth_map_original: Optional[np.ndarray] = None
    confidence_map: Optional[np.ndarray] = None
    pose_matrix: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    clip_embedding: Optional[np.ndarray] = None  # Stores full-frame DINO features (legacy field name).
