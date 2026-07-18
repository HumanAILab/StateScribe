import cv2
import numpy as np
import torch
import torch.nn.functional as F

from components.memory.frame import Frame


def _to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _trim_mask_edges(mask: np.ndarray, edge_trim_ratio: float) -> np.ndarray:
    ratio = float(edge_trim_ratio)
    if ratio <= 0.0:
        return mask
    ratio = min(ratio, 0.49)

    h, w = mask.shape[:2]
    trim_y = min(int(round(h * ratio)), max(0, (h - 1) // 2))
    trim_x = min(int(round(w * ratio)), max(0, (w - 1) // 2))
    if trim_y <= 0 and trim_x <= 0:
        return mask

    trimmed = mask.copy()
    if trim_y > 0:
        trimmed[:trim_y, :] = 0
        trimmed[h - trim_y:, :] = 0
    if trim_x > 0:
        trimmed[:, :trim_x] = 0
        trimmed[:, w - trim_x:] = 0
    return trimmed


def compute_visibility_mask(
    ref_frame: Frame,
    cur_frame: Frame,
    device: str,
    dilation_kernel_size: int,
    downsample_ratio: int,
    min_depth: float = 0.02,
    edge_trim_ratio: float = 0.0,
) -> np.ndarray:
    h, w = ref_frame.depth_map.shape[:2]

    K_ref = _to_tensor(ref_frame.intrinsics, device)
    K_cur = _to_tensor(cur_frame.intrinsics, device)

    if downsample_ratio > 1:
        K_ref = K_ref.clone()
        K_cur = K_cur.clone()
        K_ref[:2] /= downsample_ratio
        K_cur[:2] /= downsample_ratio
        h //= downsample_ratio
        w //= downsample_ratio

    T_ref_c2w = _to_tensor(ref_frame.pose_matrix, device)
    T_cur_c2w = _to_tensor(cur_frame.pose_matrix, device)
    T_cur_w2c = torch.linalg.inv(T_cur_c2w)
    T_ref_to_cur = T_cur_w2c @ T_ref_c2w

    depth_ref_np = ref_frame.depth_map
    if downsample_ratio > 1:
        depth_ref_np = depth_ref_np[::downsample_ratio, ::downsample_ratio]

    depth_ref = _to_tensor(depth_ref_np, device)

    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    z_flat = depth_ref.flatten()

    valid_mask = z_flat > 0
    if not valid_mask.any():
        return np.zeros((h, w), dtype=np.uint8)

    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]

    fx = K_ref[0, 0]
    fy = K_ref[1, 1]
    cx = K_ref[0, 2]
    cy = K_ref[1, 2]

    X_ref = (x_valid - cx) * z_valid / fx
    Y_ref = (y_valid - cy) * z_valid / fy
    Z_ref = z_valid

    ones = torch.ones_like(Z_ref)
    P_ref_homo = torch.stack([X_ref, Y_ref, Z_ref, ones], dim=1)

    P_cur_homo = (T_ref_to_cur @ P_ref_homo.T).T
    X_cur = P_cur_homo[:, 0]
    Y_cur = P_cur_homo[:, 1]
    Z_cur = P_cur_homo[:, 2]

    valid_z_mask = Z_cur > min_depth
    X_cur = X_cur[valid_z_mask]
    Y_cur = Y_cur[valid_z_mask]
    Z_cur = Z_cur[valid_z_mask]

    if X_cur.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    fx_cur = K_cur[0, 0]
    fy_cur = K_cur[1, 1]
    cx_cur = K_cur[0, 2]
    cy_cur = K_cur[1, 2]

    u_proj = (fx_cur * X_cur / Z_cur) + cx_cur
    v_proj = (fy_cur * Y_cur / Z_cur) + cy_cur
    u_proj = torch.round(u_proj).long()
    v_proj = torch.round(v_proj).long()

    h_cur, w_cur = h, w
    in_bounds = (u_proj >= 0) & (u_proj < w_cur) & (v_proj >= 0) & (v_proj < h_cur)
    u_proj = u_proj[in_bounds]
    v_proj = v_proj[in_bounds]

    mask = torch.zeros((h_cur, w_cur), device=device, dtype=torch.float32)
    mask.index_put_((v_proj, u_proj), torch.tensor(1.0, device=device))

    if dilation_kernel_size > 1:
        mask = mask.unsqueeze(0).unsqueeze(0)
        padding = dilation_kernel_size // 2
        mask = F.max_pool2d(mask, kernel_size=dilation_kernel_size, stride=1, padding=padding)
        mask = mask.squeeze()

    mask_np = mask.cpu().numpy()

    if downsample_ratio > 1:
        orig_h, orig_w = cur_frame.depth_map.shape[:2]
        mask_np = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return _trim_mask_edges(mask_np, edge_trim_ratio)


def compute_occlusion_mask(
    cur_frame: Frame,
    ref_frame: Frame,
    device: str,
    downsample_ratio: int,
    depth_tolerance: float = 0.15,
    min_depth: float = 0.02,
) -> np.ndarray:
    """Mask on *cur_frame* where 1 = surface was occluded in *ref_frame*.

    Back-projects each cur_frame pixel into ref_frame and checks whether
    ref_frame's depth is strictly closer (by more than *depth_tolerance*).
    If so, an occluder was present in the reference view.
    """
    h, w = cur_frame.depth_map.shape[:2]

    K_cur = _to_tensor(cur_frame.intrinsics, device)
    K_ref = _to_tensor(ref_frame.intrinsics, device)

    if downsample_ratio > 1:
        K_cur = K_cur.clone()
        K_ref = K_ref.clone()
        K_cur[:2] /= downsample_ratio
        K_ref[:2] /= downsample_ratio
        h //= downsample_ratio
        w //= downsample_ratio

    T_cur_c2w = _to_tensor(cur_frame.pose_matrix, device)
    T_ref_c2w = _to_tensor(ref_frame.pose_matrix, device)
    T_cur_to_ref = torch.linalg.inv(T_ref_c2w) @ T_cur_c2w

    depth_cur_np = cur_frame.depth_map
    if downsample_ratio > 1:
        depth_cur_np = depth_cur_np[::downsample_ratio, ::downsample_ratio]
    depth_cur = _to_tensor(depth_cur_np, device)

    depth_ref_np = ref_frame.depth_map
    if downsample_ratio > 1:
        depth_ref_np = depth_ref_np[::downsample_ratio, ::downsample_ratio]
    depth_ref = _to_tensor(depth_ref_np, device)
    h_ref, w_ref = depth_ref.shape[:2]

    y, x = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    z_flat = depth_cur.flatten()

    valid = z_flat > min_depth
    if not valid.any():
        orig_h, orig_w = cur_frame.depth_map.shape[:2]
        return np.zeros((orig_h, orig_w), dtype=np.float32)

    x_v, y_v, z_v = x_flat[valid], y_flat[valid], z_flat[valid]
    idx_valid = torch.where(valid)[0]

    fx_c, fy_c = K_cur[0, 0], K_cur[1, 1]
    cx_c, cy_c = K_cur[0, 2], K_cur[1, 2]
    X_c = (x_v - cx_c) * z_v / fx_c
    Y_c = (y_v - cy_c) * z_v / fy_c
    P = torch.stack([X_c, Y_c, z_v, torch.ones_like(z_v)], dim=1)

    P_ref = (T_cur_to_ref @ P.T).T
    X_r, Y_r, Z_r = P_ref[:, 0], P_ref[:, 1], P_ref[:, 2]

    fx_r, fy_r = K_ref[0, 0], K_ref[1, 1]
    cx_r, cy_r = K_ref[0, 2], K_ref[1, 2]
    in_front = Z_r > min_depth
    u = torch.round(fx_r * X_r / Z_r + cx_r).long()
    v = torch.round(fy_r * Y_r / Z_r + cy_r).long()
    in_bounds = in_front & (u >= 0) & (u < w_ref) & (v >= 0) & (v < h_ref)

    occluded = torch.zeros(x_v.shape[0], device=device, dtype=torch.bool)
    if in_bounds.any():
        z_back = Z_r[in_bounds]
        z_ref_at = depth_ref[v[in_bounds], u[in_bounds]]
        occluded[in_bounds] = (z_ref_at > min_depth) & (z_ref_at < z_back - depth_tolerance)

    mask_flat = torch.zeros(h * w, device=device, dtype=torch.float32)
    mask_flat[idx_valid[occluded]] = 1.0
    mask_np = mask_flat.view(h, w).cpu().numpy()

    if downsample_ratio > 1:
        orig_h, orig_w = cur_frame.depth_map.shape[:2]
        mask_np = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return mask_np
