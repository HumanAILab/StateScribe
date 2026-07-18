# utils/common.py
import zlib
import pickle

def compress_object(obj: object) -> bytes:
    """
    Serializes and compresses a Python object using pickle and zlib.
    """
    serialized_data = pickle.dumps(obj)
    return zlib.compress(serialized_data)

def decompress_object(data: bytes) -> object:
    """
    Decompresses and deserializes a Python object from bytes.
    """
    return pickle.loads(zlib.decompress(data))

from typing import Dict, Any

def convert_bbox_to_original_size(bbox: Dict[str, Any], frame) -> Dict[str, Any]:
    """Convert bounding box coordinates to original frame size"""
    if not hasattr(frame, 'rgb_image') or frame.rgb_image is None:
        return bbox
    height, width = frame.rgb_image.shape[:2]
    # Assuming bounding box is in normalized coordinates [y1, x1, y2, x2] * 1000
    box_2d = bbox.get('box_2d', [])
    if len(box_2d) >= 4:
        y1, x1, y2, x2 = box_2d[:4]
        # Convert to absolute coordinates
        abs_y1 = int(y1 / 1000 * height)
        abs_x1 = int(x1 / 1000 * width)
        abs_y2 = int(y2 / 1000 * height)
        abs_x2 = int(x2 / 1000 * width)
        # Ensure proper ordering
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        converted_bbox = bbox.copy()
        converted_bbox['box_2d_absolute'] = [abs_y1, abs_x1, abs_y2, abs_x2]
        return converted_bbox
    return bbox