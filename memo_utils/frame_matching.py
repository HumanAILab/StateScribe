# utils/frame_matching.py
import os
import numpy as np
import math
from typing import List, Dict, Optional, Any
from datetime import datetime
from sklearn.cluster import DBSCAN
import logging

try:
    import cv2
except Exception:
    cv2 = None

from components.memory.frame import Frame
from config import (
    FRAME_MATCH_PREFILTER_TRANSLATION_M,
    FRAME_MATCH_PREFILTER_ROTATION_DEG,
    FRAME_MATCH_OVERLAP_MIN_MEAN_RATIO,
    FRAME_MATCH_OVERLAP_MIN_DIRECTIONAL_RATIO,
    FRAME_MATCH_OVERLAP_DOWNSAMPLE_RATIO,
    FRAME_MATCH_OVERLAP_MIN_DEPTH_M,
    TEMPORAL_CLUSTERING_EPS_SECONDS,
    TEMPORAL_CLUSTERING_MIN_SAMPLES,
    FRAME_CLUSTER_LARGE_SIZE_THRESHOLD,
    FRAME_CLUSTER_FORCE_COMPARE_INTERVAL,
    FRAME_CLUSTER_DEBUG_ENABLED,
    FRAME_CLUSTER_DEBUG_DIR,
    FRAME_MATCHING_BRISQUE_MODEL_PATH,
    FRAME_MATCHING_BRISQUE_RANGE_PATH,
)

logger = logging.getLogger(__name__)

# Warm up sklearn / OpenMP threading at import time so that the first real
# DBSCAN call doesn't trigger OpenMP thread-pool initialisation while the
# Open3D visualizer event loop is running.  On WSL2 the late initialisation
# can interfere with GLFW / OpenGL and freeze the 3D window.
DBSCAN(eps=1, min_samples=1).fit(np.array([[0.0]]))

_BRISQUE_BACKEND_READY: Optional[bool] = None
_BRISQUE_BACKEND_MODE: Optional[str] = None
_BRISQUE_ESTIMATOR: Optional[Any] = None


def _extract_brisque_score(raw_score: Any) -> float:
    if isinstance(raw_score, (int, float)):
        return float(raw_score)
    if isinstance(raw_score, tuple) and raw_score:
        return _extract_brisque_score(raw_score[0])
    if isinstance(raw_score, list) and raw_score:
        return _extract_brisque_score(raw_score[0])
    if isinstance(raw_score, np.ndarray):
        if raw_score.size == 0:
            raise RuntimeError("Empty BRISQUE score array")
        return float(raw_score.reshape(-1)[0])
    raise RuntimeError(f"Unsupported BRISQUE score type: {type(raw_score).__name__}")


def _ensure_brisque_backend() -> bool:
    global _BRISQUE_BACKEND_READY, _BRISQUE_BACKEND_MODE, _BRISQUE_ESTIMATOR
    if _BRISQUE_BACKEND_READY is not None:
        return _BRISQUE_BACKEND_READY

    if cv2 is None:
        logger.warning("BRISQUE filter disabled: cv2 import failed.")
        _BRISQUE_BACKEND_READY = False
        return False

    quality_mod = getattr(cv2, "quality", None)
    if quality_mod is None:
        logger.warning("BRISQUE filter disabled: cv2.quality module is unavailable.")
        _BRISQUE_BACKEND_READY = False
        return False

    if not os.path.exists(FRAME_MATCHING_BRISQUE_MODEL_PATH):
        logger.warning(
            "BRISQUE filter disabled: model file not found at %s.",
            FRAME_MATCHING_BRISQUE_MODEL_PATH,
        )
        _BRISQUE_BACKEND_READY = False
        return False

    if not os.path.exists(FRAME_MATCHING_BRISQUE_RANGE_PATH):
        logger.warning(
            "BRISQUE filter disabled: range file not found at %s.",
            FRAME_MATCHING_BRISQUE_RANGE_PATH,
        )
        _BRISQUE_BACKEND_READY = False
        return False

    if hasattr(quality_mod, "QualityBRISQUE_create"):
        try:
            _BRISQUE_ESTIMATOR = quality_mod.QualityBRISQUE_create(
                FRAME_MATCHING_BRISQUE_MODEL_PATH,
                FRAME_MATCHING_BRISQUE_RANGE_PATH,
            )
            _BRISQUE_BACKEND_MODE = "estimator"
            _BRISQUE_BACKEND_READY = True
            return True
        except Exception as exc:
            logger.warning(
                "BRISQUE estimator init failed, trying static API fallback: %s",
                exc,
            )

    if hasattr(quality_mod, "QualityBRISQUE_compute"):
        _BRISQUE_BACKEND_MODE = "static"
        _BRISQUE_ESTIMATOR = None
        _BRISQUE_BACKEND_READY = True
        return True

    logger.warning("BRISQUE filter disabled: no usable QualityBRISQUE API in cv2 build.")
    _BRISQUE_BACKEND_READY = False
    return False


def _compute_frame_brisque_score(frame: Frame, cache: Dict[Any, Optional[float]]) -> Optional[float]:
    frame_ts = getattr(frame, "timestamp", None)
    cache_key: Any = frame_ts if frame_ts is not None else id(frame)
    if cache_key in cache:
        return cache[cache_key]

    rgb = getattr(frame, "rgb_image", None)
    if rgb is None or cv2 is None:
        cache[cache_key] = None
        return None

    if not _ensure_brisque_backend():
        cache[cache_key] = None
        return None

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        cache[cache_key] = None
        return None

    rgb_u8 = rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    quality_mod = cv2.quality
    try:
        if _BRISQUE_BACKEND_MODE == "estimator" and _BRISQUE_ESTIMATOR is not None:
            raw_score = _BRISQUE_ESTIMATOR.compute(bgr)
        elif _BRISQUE_BACKEND_MODE == "static" and hasattr(quality_mod, "QualityBRISQUE_compute"):
            raw_score = quality_mod.QualityBRISQUE_compute(
                bgr,
                FRAME_MATCHING_BRISQUE_MODEL_PATH,
                FRAME_MATCHING_BRISQUE_RANGE_PATH,
            )
        else:
            cache[cache_key] = None
            return None
        score = _extract_brisque_score(raw_score)
        cache[cache_key] = score
        return score
    except Exception as exc:
        logger.debug(
            "BRISQUE scoring failed for frame %s: %s",
            frame_ts,
            exc,
        )
        cache[cache_key] = None
        return None


def _filter_cluster_frames_by_brisque(
    frames_data: List[Dict[str, Any]],
    cluster_id: int,
    brisque_max_score: Optional[float],
    brisque_cache: Dict[Any, Optional[float]],
) -> List[Dict[str, Any]]:
    if brisque_max_score is None:
        return frames_data

    max_score = float(brisque_max_score)
    filtered: List[Dict[str, Any]] = []
    dropped = 0
    for data in frames_data:
        frame = data.get("frame")
        if frame is None:
            continue
        score = _compute_frame_brisque_score(frame, brisque_cache)
        if score is None or score <= max_score:
            filtered.append(data)
            continue
        dropped += 1

    if dropped > 0:
        logger.debug(
            "Cluster %s BRISQUE filter removed %d/%d frames (threshold=%.2f).",
            cluster_id,
            dropped,
            len(frames_data),
            max_score,
        )
    return filtered

def compute_rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute the rotation angle in degrees between two rotation matrices.
    
    Args:
        R1, R2: 3x3 rotation matrices
    
    Returns:
        Rotation angle in degrees
    """
    R_delta = R1.T @ R2
    trace = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = math.acos(trace)
    return math.degrees(angle_rad)

def compute_translation_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two translation vectors.
    
    Args:
        t1, t2: 3D translation vectors
    
    Returns:
        Distance in meters
    """
    return float(np.linalg.norm(t1 - t2))

def _compute_directional_visibility_overlap(
    source_frame: Frame,
    target_frame: Frame,
    downsample_ratio: int,
    min_depth_m: float,
) -> float:
    src_depth = getattr(source_frame, "depth_map", None)
    tgt_depth = getattr(target_frame, "depth_map", None)
    src_intr = getattr(source_frame, "intrinsics", None)
    tgt_intr = getattr(target_frame, "intrinsics", None)
    src_pose = getattr(source_frame, "pose_matrix", None)
    tgt_pose = getattr(target_frame, "pose_matrix", None)

    if src_depth is None or tgt_depth is None or src_intr is None or tgt_intr is None:
        return 0.0
    if src_pose is None or tgt_pose is None:
        return 0.0

    src_depth_arr = np.asarray(src_depth)
    tgt_depth_arr = np.asarray(tgt_depth)
    src_intr_arr = np.asarray(src_intr, dtype=np.float64)
    tgt_intr_arr = np.asarray(tgt_intr, dtype=np.float64)
    src_pose_arr = np.asarray(src_pose, dtype=np.float64)
    tgt_pose_arr = np.asarray(tgt_pose, dtype=np.float64)

    if src_depth_arr.ndim != 2 or tgt_depth_arr.ndim != 2:
        return 0.0
    if src_intr_arr.shape != (3, 3) or tgt_intr_arr.shape != (3, 3):
        return 0.0
    if src_pose_arr.shape != (4, 4) or tgt_pose_arr.shape != (4, 4):
        return 0.0

    step = max(1, int(downsample_ratio))
    if step > 1:
        src_depth_arr = src_depth_arr[::step, ::step]
        tgt_depth_arr = tgt_depth_arr[::step, ::step]
        src_intr_arr = src_intr_arr.copy()
        tgt_intr_arr = tgt_intr_arr.copy()
        src_intr_arr[:2, :] /= float(step)
        tgt_intr_arr[:2, :] /= float(step)

    src_depth_arr = src_depth_arr.astype(np.float64, copy=False)
    valid = np.isfinite(src_depth_arr) & (src_depth_arr > float(min_depth_m))
    if not np.any(valid):
        return 0.0

    ys, xs = np.nonzero(valid)
    z_vals = src_depth_arr[ys, xs]
    total_points = z_vals.size
    if total_points == 0:
        return 0.0

    fx_s, fy_s = float(src_intr_arr[0, 0]), float(src_intr_arr[1, 1])
    cx_s, cy_s = float(src_intr_arr[0, 2]), float(src_intr_arr[1, 2])
    fx_t, fy_t = float(tgt_intr_arr[0, 0]), float(tgt_intr_arr[1, 1])
    cx_t, cy_t = float(tgt_intr_arr[0, 2]), float(tgt_intr_arr[1, 2])

    if min(abs(fx_s), abs(fy_s), abs(fx_t), abs(fy_t)) < 1e-6:
        return 0.0

    x_src = xs.astype(np.float64)
    y_src = ys.astype(np.float64)
    X_src = (x_src - cx_s) * z_vals / fx_s
    Y_src = (y_src - cy_s) * z_vals / fy_s
    points_src = np.vstack([
        X_src,
        Y_src,
        z_vals,
        np.ones_like(z_vals),
    ])

    try:
        src_to_tgt = np.linalg.inv(tgt_pose_arr) @ src_pose_arr
    except np.linalg.LinAlgError:
        return 0.0

    points_tgt = src_to_tgt @ points_src
    z_tgt = points_tgt[2]
    in_front = np.isfinite(z_tgt) & (z_tgt > float(min_depth_m))
    if not np.any(in_front):
        return 0.0

    x_tgt = points_tgt[0, in_front]
    y_tgt = points_tgt[1, in_front]
    z_tgt = z_tgt[in_front]

    u = fx_t * x_tgt / z_tgt + cx_t
    v = fy_t * y_tgt / z_tgt + cy_t

    h_tgt, w_tgt = tgt_depth_arr.shape[:2]
    in_bounds = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < float(w_tgt))
        & (v >= 0.0)
        & (v < float(h_tgt))
    )
    return float(np.count_nonzero(in_bounds) / total_points)


def find_overlap_matched_frames(
    current_frame: Frame,
    candidate_frames: List[Frame],
    max_rot_prefilter_deg: float = FRAME_MATCH_PREFILTER_ROTATION_DEG,
    max_trans_prefilter_m: float = FRAME_MATCH_PREFILTER_TRANSLATION_M,
    min_mean_overlap_ratio: float = FRAME_MATCH_OVERLAP_MIN_MEAN_RATIO,
    min_directional_overlap_ratio: float = FRAME_MATCH_OVERLAP_MIN_DIRECTIONAL_RATIO,
    overlap_downsample_ratio: int = FRAME_MATCH_OVERLAP_DOWNSAMPLE_RATIO,
    min_depth_m: float = FRAME_MATCH_OVERLAP_MIN_DEPTH_M,
) -> List[Dict[str, Any]]:
    """
    Match frames using bidirectional field-of-view overlap after a light pose prefilter.
    """
    if current_frame.pose_matrix is None:
        logger.debug("Current frame has no pose information")
        return []

    current_ts = getattr(current_frame, "timestamp", None)
    R_cur = current_frame.pose_matrix[:3, :3]
    t_cur = current_frame.pose_matrix[:3, 3]
    matched: List[Dict[str, Any]] = []

    for frame in candidate_frames:
        frame_ts = getattr(frame, "timestamp", None)
        if current_ts is not None and frame_ts is not None and frame_ts > current_ts:
            continue
        if frame.pose_matrix is None:
            continue

        R_ref = frame.pose_matrix[:3, :3]
        t_ref = frame.pose_matrix[:3, 3]
        rot_deg = compute_rotation_angle(R_cur, R_ref)
        trans_m = compute_translation_distance(t_cur, t_ref)

        if rot_deg > float(max_rot_prefilter_deg) or trans_m > float(max_trans_prefilter_m):
            continue

        overlap_ref_to_cur = _compute_directional_visibility_overlap(
            source_frame=frame,
            target_frame=current_frame,
            downsample_ratio=overlap_downsample_ratio,
            min_depth_m=min_depth_m,
        )
        overlap_cur_to_ref = _compute_directional_visibility_overlap(
            source_frame=current_frame,
            target_frame=frame,
            downsample_ratio=overlap_downsample_ratio,
            min_depth_m=min_depth_m,
        )
        mean_overlap = 0.5 * (overlap_ref_to_cur + overlap_cur_to_ref)
        if mean_overlap < float(min_mean_overlap_ratio):
            continue
        if min(overlap_ref_to_cur, overlap_cur_to_ref) < float(min_directional_overlap_ratio):
            continue

        denom = overlap_ref_to_cur + overlap_cur_to_ref
        overlap_score = (2.0 * overlap_ref_to_cur * overlap_cur_to_ref / denom) if denom > 1e-8 else 0.0

        matched.append({
            "frame": frame,
            "score": float(overlap_score),
            "rot_deg": float(rot_deg),
            "trans_m": float(trans_m),
            "overlap_ref_to_cur": float(overlap_ref_to_cur),
            "overlap_cur_to_ref": float(overlap_cur_to_ref),
            "mean_overlap": float(mean_overlap),
            "timestamp": frame_ts,
        })

    matched.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("Found %d overlap-matched frames", len(matched))
    return matched

def temporal_clustering(
    frames_data: List[Dict[str, Any]],
    eps_seconds: float = TEMPORAL_CLUSTERING_EPS_SECONDS,
    min_samples: int = TEMPORAL_CLUSTERING_MIN_SAMPLES
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Cluster frames by temporal proximity using DBSCAN.
    
    Args:
        frames_data: List of frame data dicts, each must have 'timestamp' key
        eps_seconds: DBSCAN epsilon in seconds
        min_samples: DBSCAN min_samples parameter
    
    Returns:
        Dictionary mapping cluster_id to list of frame data in that cluster
        Cluster ID -1 represents noise/outliers
    """
    if not frames_data:
        return {}
    
    # Extract timestamps and convert to seconds since first frame
    timestamps = [data['timestamp'] for data in frames_data]
    t0 = min(timestamps)
    time_features = np.array([[(ts - t0).total_seconds()] for ts in timestamps])
    
    # Run DBSCAN
    clustering = DBSCAN(eps=eps_seconds, min_samples=min_samples).fit(time_features)
    labels = clustering.labels_
    
    # Group by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(frames_data[idx])
    
    logger.debug(f"Temporal clustering found {len([k for k in clusters.keys() if k != -1])} clusters")
    
    return clusters

def select_large_cluster_periodic_reference(
    clusters: Dict[int, List[Dict[str, Any]]],
    current_frame: Frame,
    size_threshold: int = FRAME_CLUSTER_LARGE_SIZE_THRESHOLD,
    force_interval: int = FRAME_CLUSTER_FORCE_COMPARE_INTERVAL,
    brisque_max_score: Optional[float] = None,
    brisque_cache: Optional[Dict[Any, Optional[float]]] = None,
) -> tuple[bool, Optional[Frame]]:
    current_ts = getattr(current_frame, "timestamp", None)
    if current_ts is None:
        return False, None
    if brisque_cache is None:
        brisque_cache = {}

    threshold = max(1, int(size_threshold))
    interval = max(1, int(force_interval))
    valid_clusters = {k: v for k, v in clusters.items() if k != -1}

    for cluster_id, frames_data in valid_clusters.items():
        ordered = sorted(frames_data, key=lambda data: data.get("timestamp") or datetime.min)
        current_idx = next(
            (
                idx
                for idx, data in enumerate(ordered)
                if getattr(data.get("frame"), "timestamp", None) == current_ts
            ),
            None,
        )
        if current_idx is None:
            continue

        if len(ordered) <= threshold:
            return False, None

        if current_idx < interval:
            logger.debug(
                "Large cluster %s has current frame but not enough history: size=%d current_idx=%d interval=%d",
                cluster_id,
                len(ordered),
                current_idx,
                interval,
            )
            return True, None

        if current_idx % interval != 0:
            logger.debug(
                "Large cluster %s has current frame but current_idx is not on interval boundary: size=%d current_idx=%d interval=%d",
                cluster_id,
                len(ordered),
                current_idx,
                interval,
            )
            return True, None

        ref_idx = current_idx - interval
        reference_frame = ordered[ref_idx].get("frame")
        if reference_frame is None:
            logger.debug(
                "Large cluster %s periodic reference is missing: current_idx=%d ref_idx=%d",
                cluster_id,
                current_idx,
                ref_idx,
            )
            return True, None

        if brisque_max_score is not None:
            score = _compute_frame_brisque_score(reference_frame, brisque_cache)
            if score is not None and score > float(brisque_max_score):
                logger.debug(
                    "Large cluster %s periodic reference filtered by BRISQUE: ref_ts=%s score=%.3f threshold=%.2f",
                    cluster_id,
                    getattr(reference_frame, "timestamp", None),
                    score,
                    float(brisque_max_score),
                )
                return True, None

        logger.debug(
            "✓ Selected periodic reference from large cluster %s: current_idx=%d ref_idx=%d current_ts=%s ref_ts=%s",
            cluster_id,
            current_idx,
            ref_idx,
            current_ts,
            getattr(reference_frame, "timestamp", None),
        )
        return True, reference_frame

    return False, None

def select_reference_frame(
    clusters: Dict[int, List[Dict[str, Any]]],
    described_frames: set,
    current_frame: Frame,
    live_described_frames: Optional[set] = None,
    require_live_described_cluster: bool = False,
    brisque_max_score: Optional[float] = None,
    brisque_cache: Optional[Dict[Any, Optional[float]]] = None,
) -> Optional[Frame]:
    """
    Select the best reference frame from clusters based on StateScribe rules.
    
    Rules (default):
    1. If a later cluster contains described frames, all earlier clusters are unusable
    2. Skip clusters containing already-described frames
    3. Skip clusters containing the current frame itself
    4. Select the newest available cluster
    5. Within that cluster, select the frame with best overlap score

    Rules (require_live_described_cluster=True):
    1. Cluster is valid only if it contains at least one live-described frame
    2. Skip clusters containing the current frame itself
    3. Select the newest available cluster
    4. Within that cluster, select the frame with best overlap score
    
    Args:
        clusters: Output from temporal_clustering()
        described_frames: Set of frame timestamps used by legacy described-frame logic
        current_frame: The current frame being analyzed
        live_described_frames: Set of frame timestamps that were emitted by live narration
        require_live_described_cluster: Whether to require live-described tags for valid clusters
    
    Returns:
        Selected reference Frame, or None if no valid frame found
    """
    # Remove noise cluster (-1)
    valid_clusters = {k: v for k, v in clusters.items() if k != -1}
    
    if not valid_clusters:
        logger.debug("No valid clusters found")
        return None

    if live_described_frames is None:
        live_described_frames = set()
    if brisque_cache is None:
        brisque_cache = {}

    if require_live_described_cluster:
        cluster_ages = {
            cluster_id: min(data["timestamp"] for data in frames_data)
            for cluster_id, frames_data in valid_clusters.items()
        }
        sorted_clusters = sorted(cluster_ages.items(), key=lambda x: x[1], reverse=True)
        logger.debug(
            "Checking %d clusters with live-described gating",
            len(sorted_clusters),
        )
        for cluster_id, oldest_ts in sorted_clusters:
            frames_data = valid_clusters[cluster_id]
            logger.debug(
                "Cluster %s: %d frames, oldest=%s",
                cluster_id,
                len(frames_data),
                oldest_ts,
            )

            has_current = any(
                data["frame"].timestamp == current_frame.timestamp
                for data in frames_data
            )
            if has_current:
                logger.debug("Cluster %s SKIP: contains current frame", cluster_id)
                continue

            has_live_described = any(
                data["frame"].timestamp in live_described_frames
                for data in frames_data
            )
            if not has_live_described:
                logger.debug("Cluster %s SKIP: no live-described frame", cluster_id)
                continue

            candidate_frames = _filter_cluster_frames_by_brisque(
                frames_data,
                cluster_id=cluster_id,
                brisque_max_score=brisque_max_score,
                brisque_cache=brisque_cache,
            )
            if not candidate_frames:
                logger.debug("Cluster %s SKIP: no frame passed BRISQUE filter", cluster_id)
                continue

            best_frame = max(candidate_frames, key=lambda x: x["score"])
            logger.debug(
                "✓ Selected reference frame from cluster %s (live-gated): timestamp=%s, score=%.3f",
                cluster_id,
                best_frame["timestamp"],
                best_frame["score"],
            )
            return best_frame["frame"]

        logger.debug("No valid reference frame found after checking live-gated clusters")
        return None
    
    # Find oldest timestamp in each cluster
    cluster_ages = {}
    for cluster_id, frames_data in valid_clusters.items():
        oldest_ts = min(data['timestamp'] for data in frames_data)
        cluster_ages[cluster_id] = oldest_ts
    
    # Compute described cutoff (latest cluster time that contains described frames)
    described_cutoff_ts = None
    for cluster_id, frames_data in valid_clusters.items():
        has_described = any(
            data['frame'].timestamp in described_frames
            for data in frames_data
        )
        if not has_described:
            continue
        oldest_ts = min(data['timestamp'] for data in frames_data)
        if described_cutoff_ts is None or oldest_ts > described_cutoff_ts:
            described_cutoff_ts = oldest_ts

    # Sort clusters by age (newest first)
    sorted_clusters = sorted(cluster_ages.items(), key=lambda x: x[1], reverse=True)
    
    # Iterate through clusters from oldest to newest
    logger.debug(f"Checking {len(sorted_clusters)} clusters for valid reference frame")
    
    for cluster_id, oldest_ts in sorted_clusters:
        frames_data = valid_clusters[cluster_id]
        
        logger.debug(f"Cluster {cluster_id}: {len(frames_data)} frames, oldest={oldest_ts}")
        
        # Skip clusters at or before described cutoff
        if described_cutoff_ts is not None and oldest_ts <= described_cutoff_ts:
            logger.debug(
                f"Cluster {cluster_id} SKIP: older than described cutoff ({described_cutoff_ts})"
            )
            continue

        # Check if any frame in this cluster is already described
        has_described = any(
            data['frame'].timestamp in described_frames 
            for data in frames_data
        )
        
        if has_described:
            logger.debug(f"Cluster {cluster_id} SKIP: contains described frame")
            continue
        
        # Check if current frame is in this cluster
        has_current = any(
            data['frame'].timestamp == current_frame.timestamp
            for data in frames_data
        )
        
        if has_current:
            logger.debug(f"Cluster {cluster_id} SKIP: contains current frame")
            continue

        candidate_frames = _filter_cluster_frames_by_brisque(
            frames_data,
            cluster_id=cluster_id,
            brisque_max_score=brisque_max_score,
            brisque_cache=brisque_cache,
        )
        if not candidate_frames:
            logger.debug(f"Cluster {cluster_id} SKIP: no frame passed BRISQUE filter")
            continue

        # This cluster is valid - select best frame by overlap score
        best_frame = max(candidate_frames, key=lambda x: x['score'])
        
        logger.debug(
            f"✓ Selected reference frame from cluster {cluster_id}: "
            f"timestamp={best_frame['timestamp']}, score={best_frame['score']:.3f}"
        )
        
        return best_frame['frame']
    
    logger.debug("No valid reference frame found after checking all clusters")
    return None

def find_and_select_reference_frame(
    current_frame: Frame,
    ltm_frames: List[Frame],
    described_frames: set,
    live_described_frames: Optional[set] = None,
    require_live_described_cluster: bool = False,
    brisque_max_score: Optional[float] = None,
) -> Optional[Frame]:
    """
    Complete pipeline to find and select a reference frame for change detection.
    
    Args:
        current_frame: The current frame to find a reference for
        ltm_frames: All frames in long-term memory
        described_frames: Set of timestamps used by legacy described-frame logic
        live_described_frames: Set of timestamps emitted by live narration
        require_live_described_cluster: Whether to require live-described tags for valid clusters
    
    Returns:
        Selected reference frame, or None if no suitable frame found
    """
    # Step 1: Find overlap-matched frames
    matched_frames = find_overlap_matched_frames(current_frame, ltm_frames)
    tagged_frames = live_described_frames if require_live_described_cluster else described_frames
    if tagged_frames is None:
        tagged_frames = set()
    
    if not matched_frames:
        logger.debug("No overlap-matched frames found")
        _save_frame_match_debug(
            current_frame,
            matched_frames,
            {},
            tagged_frames,
            None,
            require_live_described_cluster=require_live_described_cluster,
        )
        _save_frame_match_debug_text(
            current_frame, ltm_frames, matched_frames, {},
            tagged_frames, None,
            require_live_described_cluster=require_live_described_cluster,
        )
        return None
    
    logger.debug(f"Found {len(matched_frames)} overlap-matched frames")

    if len(matched_frames) <= 1:
        logger.debug("Skipping reference selection: only one overlap-matched frame is available.")
        _save_frame_match_debug(
            current_frame,
            matched_frames,
            {},
            tagged_frames,
            None,
            require_live_described_cluster=require_live_described_cluster,
        )
        _save_frame_match_debug_text(
            current_frame, ltm_frames, matched_frames, {},
            tagged_frames, None,
            require_live_described_cluster=require_live_described_cluster,
        )
        return None
    
    # Step 2: Temporal clustering
    clusters = temporal_clustering(matched_frames)
    logger.debug(f"Temporal clustering created {len([k for k in clusters.keys() if k != -1])} clusters")
    brisque_cache: Dict[Any, Optional[float]] = {}

    # Step 3: Select reference frame
    large_cluster_applied, reference_frame = select_large_cluster_periodic_reference(
        clusters,
        current_frame,
        brisque_max_score=brisque_max_score,
        brisque_cache=brisque_cache,
    )
    if not large_cluster_applied:
        reference_frame = select_reference_frame(
            clusters,
            described_frames,
            current_frame,
            live_described_frames=live_described_frames,
            require_live_described_cluster=require_live_described_cluster,
            brisque_max_score=brisque_max_score,
            brisque_cache=brisque_cache,
        )
    elif reference_frame is None:
        logger.debug("Large-cluster periodic rule skipped reference selection for this frame.")

    _save_frame_match_debug(
        current_frame,
        matched_frames,
        clusters,
        tagged_frames,
        reference_frame,
        require_live_described_cluster=require_live_described_cluster,
    )
    _save_frame_match_debug_text(
        current_frame, ltm_frames, matched_frames, clusters,
        tagged_frames, reference_frame,
        large_cluster_applied=large_cluster_applied,
        require_live_described_cluster=require_live_described_cluster,
    )

    if reference_frame:
        logger.debug(f"Selected reference frame: {reference_frame.timestamp}")
    
    return reference_frame


def _save_frame_match_debug(
    current_frame: Frame,
    matched_frames: List[Dict[str, Any]],
    clusters: Dict[int, List[Dict[str, Any]]],
    tagged_frames: set,
    reference_frame: Optional[Frame],
    require_live_described_cluster: bool = False,
):
    if not FRAME_CLUSTER_DEBUG_ENABLED:
        return
    import cv2

    os.makedirs(FRAME_CLUSTER_DEBUG_DIR, exist_ok=True)
    ts_name = current_frame.timestamp.strftime("%Y%m%d_%H%M%S_%f")
    out_path = os.path.join(FRAME_CLUSTER_DEBUG_DIR, f"frame_match_{ts_name}.png")

    if not matched_frames:
        img = np.zeros((160, 800, 3), dtype=np.uint8)
        cv2.putText(img, "no overlap-matched frames", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(out_path, img)
        return

    all_ts = [d.get("timestamp") for d in matched_frames if d.get("timestamp") is not None]
    if current_frame.timestamp is not None:
        all_ts.append(current_frame.timestamp)
    min_ts = min(all_ts)
    max_ts = max(all_ts)
    span = (max_ts - min_ts).total_seconds()
    if span <= 0:
        span = 1.0

    cluster_ids = sorted(clusters.keys(), key=lambda x: (x == -1, x))
    described_cutoff_ts = None
    if not require_live_described_cluster:
        for cid in cluster_ids:
            frames_data = clusters.get(cid, [])
            has_described = any(
                d.get("timestamp") in tagged_frames for d in frames_data
            )
            if not has_described:
                continue
            oldest_ts = min(d.get("timestamp") for d in frames_data if d.get("timestamp") is not None)
            if described_cutoff_ts is None or oldest_ts > described_cutoff_ts:
                described_cutoff_ts = oldest_ts
    rows = max(1, len(cluster_ids))

    width = 1200
    left = 220
    right = 20
    top = 70
    row_h = 38
    height = top + rows * row_h + 60
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    header = f"current={ts_name} world={getattr(current_frame, 'world_name', '')} matches={len(matched_frames)}"
    cv2.putText(img, header, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    tag_label = "live-tagged" if require_live_described_cluster else "described"
    legend = f"legend: current(red) reference(green) {tag_label}(orange) normal(gray)"
    if described_cutoff_ts is not None:
        legend += f" | cutoff={described_cutoff_ts.strftime('%H:%M:%S')}"
    cv2.putText(img, legend, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

    def to_x(ts: datetime) -> int:
        dt = (ts - min_ts).total_seconds()
        return left + int((dt / span) * (width - left - right))

    def point_color(ts: datetime) -> tuple[int, int, int]:
        if reference_frame is not None and ts == reference_frame.timestamp:
            return (0, 180, 0)
        if ts == current_frame.timestamp:
            return (0, 0, 255)
        if ts in tagged_frames:
            return (0, 165, 255)
        return (120, 120, 120)

    if not cluster_ids:
        cv2.putText(img, "no clusters", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for idx, cid in enumerate(cluster_ids):
            frames_data = clusters.get(cid, [])
            y = top + idx * row_h
            cv2.line(img, (left, y), (width - right, y), (220, 220, 220), 1)
            oldest_ts = min(
                (d.get("timestamp") for d in frames_data if d.get("timestamp") is not None),
                default=None,
            )
            has_tagged = any(
                d.get("timestamp") in tagged_frames for d in frames_data
            )
            has_current = any(
                d.get("timestamp") == current_frame.timestamp for d in frames_data
            )
            blocked_by_cutoff = (
                described_cutoff_ts is not None and oldest_ts is not None and oldest_ts <= described_cutoff_ts
            )
            if require_live_described_cluster:
                usable = has_tagged and not has_current
            else:
                usable = not (has_tagged or has_current or blocked_by_cutoff)
            reason = "ok"
            if require_live_described_cluster:
                if not has_tagged:
                    reason = "no_live_tag"
                elif has_current:
                    reason = "current"
            else:
                if blocked_by_cutoff:
                    reason = "cutoff"
                elif has_tagged:
                    reason = "described"
                elif has_current:
                    reason = "current"
            label = f"cluster {cid} | n={len(frames_data)} | usable={'yes' if usable else 'no'} | reason={reason}"
            label_color = (0, 160, 0) if usable else (0, 0, 255)
            cv2.putText(img, label, (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

            for d in frames_data:
                ts = d.get("timestamp")
                if ts is None:
                    continue
                x = to_x(ts)
                cv2.circle(img, (x, y), 4, point_color(ts), -1)

    min_label = min_ts.strftime("%H:%M:%S")
    max_label = max_ts.strftime("%H:%M:%S")
    cv2.putText(img, min_label, (left, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    cv2.putText(img, max_label, (width - right - 80, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    cv2.imwrite(out_path, img)


def _save_frame_match_debug_text(
    current_frame: Frame,
    ltm_frames: List[Frame],
    matched_frames: List[Dict[str, Any]],
    clusters: Dict[int, List[Dict[str, Any]]],
    tagged_frames: set,
    reference_frame: Optional[Frame],
    large_cluster_applied: bool = False,
    require_live_described_cluster: bool = False,
):
    if not FRAME_CLUSTER_DEBUG_ENABLED:
        return

    out_path = os.path.join(FRAME_CLUSTER_DEBUG_DIR, "frame_match_log.txt")
    os.makedirs(FRAME_CLUSTER_DEBUG_DIR, exist_ok=True)

    ts = current_frame.timestamp
    fmt = lambda dt: dt.strftime("%H:%M:%S.%f")[:12] if dt else "?"
    mode = "live_gated" if require_live_described_cluster else "legacy"

    L = [f"=== FM {ts.isoformat() if ts else '?'} ==="]
    pos = "None"
    if current_frame.pose_matrix is not None:
        p = current_frame.pose_matrix[:3, 3]
        pos = f"[{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}]"
    L.append(f"pos={pos} ltm={len(ltm_frames)} match={len(matched_frames)} mode={mode}")
    L.append(
        "prefilter: "
        f"t={FRAME_MATCH_PREFILTER_TRANSLATION_M}m "
        f"r={FRAME_MATCH_PREFILTER_ROTATION_DEG}deg"
    )
    L.append(
        "overlap_thresh: "
        f"mean>={FRAME_MATCH_OVERLAP_MIN_MEAN_RATIO:.2f} "
        f"dir>={FRAME_MATCH_OVERLAP_MIN_DIRECTIONAL_RATIO:.2f}"
    )

    tagged_sorted = sorted(tagged_frames)
    L.append(f"tagged({len(tagged_sorted)}): {' '.join(fmt(x) for x in tagged_sorted)}")

    if not matched_frames:
        L.append("RESULT: None (no_match)\n")
        _write_debug_text(out_path, L)
        return

    scores = [m["score"] for m in matched_frames]
    all_mts = [m["timestamp"] for m in matched_frames]
    span = (max(all_mts) - min(all_mts)).total_seconds()
    n_tagged = sum(1 for m in matched_frames if m["timestamp"] in tagged_frames)
    L.append(
        f"match_span={span:.1f}s scores=[{min(scores):.3f},{max(scores):.3f}] "
        f"tagged_in_match={n_tagged}/{len(matched_frames)}"
    )

    for m in matched_frames:
        fl = ("T" if m["timestamp"] in tagged_frames else ".") + \
             ("C" if m["timestamp"] == ts else ".")
        o_ref_cur = float(m.get("overlap_ref_to_cur", 0.0))
        o_cur_ref = float(m.get("overlap_cur_to_ref", 0.0))
        L.append(
            f"  {fmt(m['timestamp'])} s={m['score']:.3f} "
            f"r={m['rot_deg']:.1f} d={m['trans_m']:.3f} "
            f"o12={o_ref_cur:.3f} o21={o_cur_ref:.3f} {fl}"
        )

    valid_cl = {k: v for k, v in clusters.items() if k != -1}
    noise = clusters.get(-1, [])

    if not valid_cl and not noise:
        L.append("RESULT: None (no_clusters)\n")
        _write_debug_text(out_path, L)
        return

    cutoff = None
    if not require_live_described_cluster:
        for fdata in valid_cl.values():
            if any(d["timestamp"] in tagged_frames for d in fdata):
                o = min(d["timestamp"] for d in fdata)
                if cutoff is None or o > cutoff:
                    cutoff = o

    hdr = f"CLUSTERS({len(valid_cl)}+{len(noise)}n)"
    if cutoff:
        hdr += f" cutoff={fmt(cutoff)}"
    L.append(hdr)

    cl_ages = {cid: min(d["timestamp"] for d in v) for cid, v in valid_cl.items()}
    for cid, oldest in sorted(cl_ages.items(), key=lambda x: x[1], reverse=True):
        fdata = valid_cl[cid]
        newest = max(d["timestamp"] for d in fdata)
        span_s = (newest - oldest).total_seconds()
        tagged_in = [d["timestamp"] for d in fdata if d["timestamp"] in tagged_frames]
        has_cur = any(d["timestamp"] == ts for d in fdata)

        if require_live_described_cluster:
            reason = "no_live_tag" if not tagged_in else ("current" if has_cur else "usable")
        else:
            if cutoff and oldest <= cutoff:
                reason = "cutoff"
            elif tagged_in:
                reason = "described"
            elif has_cur:
                reason = "current"
            else:
                reason = "usable"

        if reason == "usable" and reference_frame:
            if any(getattr(d.get("frame"), "timestamp", None) == reference_frame.timestamp for d in fdata):
                best = max(fdata, key=lambda x: x["score"])
                reason = f"SELECTED(best={fmt(best['timestamp'])} s={best['score']:.3f})"

        tag_str = ",".join(fmt(x) for x in tagged_in)
        L.append(
            f"  C{cid}: n={len(fdata)} [{fmt(oldest)}..{fmt(newest)}] {span_s:.1f}s "
            f"tag=[{tag_str}] cur={'Y' if has_cur else 'N'} {reason}"
        )
        for d in sorted(fdata, key=lambda x: x["timestamp"]):
            ft = " T" if d["timestamp"] in tagged_frames else ""
            o_ref_cur = float(d.get("overlap_ref_to_cur", 0.0))
            o_cur_ref = float(d.get("overlap_cur_to_ref", 0.0))
            L.append(
                f"    {fmt(d['timestamp'])} s={d['score']:.3f} "
                f"r={d['rot_deg']:.1f} d={d['trans_m']:.3f} "
                f"o12={o_ref_cur:.3f} o21={o_cur_ref:.3f}{ft}"
            )

    if noise:
        L.append(f"  noise({len(noise)}): {' '.join(fmt(d['timestamp']) for d in noise)}")

    L.append(f"LARGE_CL={'applied' if large_cluster_applied else 'no'}")
    ref_ts = reference_frame.timestamp if reference_frame else None
    L.append(f"RESULT: {'ref=' + fmt(ref_ts) if ref_ts else 'None'}\n")

    _write_debug_text(out_path, L)


def _write_debug_text(path: str, lines: List[str]):
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
