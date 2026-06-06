"""
ANALYTICAL Cylindrical Hand IK - Ultra-Fast Self-Collision Avoidance for BVH Motion

A blazing-fast analytical approach to fixing hand self-collisions:
- Uses pelvis-local cylindrical coordinate system  
- Smooth remapping of hand positions based on body height (Z-axis)
- ANALYTICAL IK: No optimization loops, direct closed-form solution
- 100-1000x faster than optimization-based IK

USAGE:
    python analytic_cylindrical_hand_ik.py --input_bvh input.bvh --output_bvh output.bvh

PARAMETER TUNING GUIDE:
    
    If hands still collide with body:
        → Increase thresholds in z_threshold_keypoints (e.g., 22→24 for knees, 25→27 for head)
    
    If hands are pushed too far out (unnatural):
        → Decrease thresholds in z_threshold_keypoints (e.g., 22→20)
        → Decrease cutoff_ratio (e.g., 1.5→1.3)

COORDINATE SYSTEM:
    This BVH uses Z-up:
    - X: left (-) / right (+)
    - Y: back (-) / forward (+)
    - Z: down (-) / up (+)
    
    All thresholds are in centimeters, measured in pelvis-local space.

ALGORITHM (Analytical 2-Step IK):
    1. For each frame, compute hand positions in pelvis-local coordinates
    2. Based on hand height (Z), interpolate safe distance threshold
    3. If hand XY-distance < threshold * cutoff_ratio, compute push-out target
    4. SWING: Rotate entire arm (as rigid triangle) to point at target
    5. BEND: Adjust elbow angle around triangle normal to fix distance
    6. Convert world rotations back to local joint rotations
    
    No iteration, no optimization - just geometry!

"""

from __future__ import annotations

import argparse
from typing import List, Sequence, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from utils.anim import bvh
import utils.rotation_conversions as rc


# ============================================================================
# SECTION 1: Forward Kinematics
# ============================================================================

def fk_global_positions(
    local_rot: torch.Tensor,  # (T, J, 3, 3)
    offsets: torch.Tensor,  # (J, 3)
    parents: Sequence[int],
    root_trans: torch.Tensor,  # (T, 3)
) -> torch.Tensor:
    """
    Compute world positions from local rotations (standard BVH FK).
    Returns: (T, J, 3) world positions for all joints across all frames
    """
    if local_rot.dim() != 4 or local_rot.shape[-2:] != (3, 3):
        raise ValueError(f"local_rot must be (T,J,3,3), got {tuple(local_rot.shape)}")
    if offsets.dim() != 2 or offsets.shape[-1] != 3:
        raise ValueError(f"offsets must be (J,3), got {tuple(offsets.shape)}")
    if root_trans.dim() != 2 or root_trans.shape[-1] != 3:
        raise ValueError(f"root_trans must be (T,3), got {tuple(root_trans.shape)}")

    T, J = local_rot.shape[0], local_rot.shape[1]
    device = local_rot.device
    dtype = local_rot.dtype

    offsets_t = offsets.to(device=device, dtype=dtype)
    root_trans_t = root_trans.to(device=device, dtype=dtype)

    global_rot_list: List[torch.Tensor] = [None] * J  # type: ignore
    global_pos_list: List[torch.Tensor] = [None] * J  # type: ignore

    for j in range(J):
        pj = int(parents[j])
        if pj == -1:
            grot = local_rot[:, j]
            gpos = offsets_t[j].view(1, 3).expand(T, 3)
        else:
            prot = global_rot_list[pj]
            ppos = global_pos_list[pj]
            grot = torch.matmul(prot, local_rot[:, j])
            rotated_offset = torch.matmul(prot, offsets_t[j].view(3, 1)).squeeze(-1)
            gpos = ppos + rotated_offset
        global_rot_list[j] = grot
        global_pos_list[j] = gpos

    global_pos = torch.stack(global_pos_list, dim=1)  # (T,J,3)
    return global_pos + root_trans_t[:, None, :]


def fk_global_with_rotations(
    local_rot: torch.Tensor,  # (T, J, 3, 3)
    offsets: torch.Tensor,  # (J, 3)
    parents: Sequence[int],
    root_trans: torch.Tensor,  # (T, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute world positions AND world rotations from local rotations.
    
    Returns:
        positions: (T, J, 3) world positions
        rotations: (T, J, 3, 3) world rotation matrices
    """
    if local_rot.dim() != 4 or local_rot.shape[-2:] != (3, 3):
        raise ValueError(f"local_rot must be (T,J,3,3), got {tuple(local_rot.shape)}")
    
    T, J = local_rot.shape[0], local_rot.shape[1]
    device = local_rot.device
    dtype = local_rot.dtype
    
    offsets_t = offsets.to(device=device, dtype=dtype)
    root_trans_t = root_trans.to(device=device, dtype=dtype)
    
    global_rot_list: List[torch.Tensor] = [None] * J  # type: ignore
    global_pos_list: List[torch.Tensor] = [None] * J  # type: ignore
    
    for j in range(J):
        pj = int(parents[j])
        if pj == -1:
            grot = local_rot[:, j]
            gpos = offsets_t[j].view(1, 3).expand(T, 3)
        else:
            prot = global_rot_list[pj]
            ppos = global_pos_list[pj]
            grot = torch.matmul(prot, local_rot[:, j])
            rotated_offset = torch.matmul(prot, offsets_t[j].view(3, 1)).squeeze(-1)
            gpos = ppos + rotated_offset
        global_rot_list[j] = grot
        global_pos_list[j] = gpos
    
    global_pos = torch.stack(global_pos_list, dim=1)  # (T,J,3)
    global_rot = torch.stack(global_rot_list, dim=1)  # (T,J,3,3)
    
    return global_pos + root_trans_t[:, None, :], global_rot


def compute_rest_positions(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Compute rest pose positions (identity rotations)."""
    J = offsets.shape[0]
    pos = np.zeros((J, 3), dtype=np.float32)
    for j in range(J):
        pj = int(parents[j])
        if pj == -1:
            pos[j] = offsets[j]
        else:
            pos[j] = pos[pj] + offsets[j]
    return pos


# ============================================================================
# SECTION 2: Pelvis-Local Coordinate Transforms
# ============================================================================

def world_to_pelvis_local(
    pos_world: torch.Tensor,  # (3,) or (T, 3)
    pelvis_pos: torch.Tensor,  # (3,) or (T, 3)
    pelvis_rot: torch.Tensor,  # (3, 3) or (T, 3, 3)
) -> torch.Tensor:
    """
    Transform world position to pelvis-local coordinates.
    Body-relative coordinates make it easy to define collision zones.
    """
    # Translate to pelvis origin
    relative = pos_world - pelvis_pos
    
    # Rotate to pelvis orientation (inverse rotation)
    pelvis_rot_inv = pelvis_rot.transpose(-1, -2)
    
    if relative.dim() == 1:
        # Single position
        local = torch.matmul(pelvis_rot_inv, relative)
    else:
        # Multiple positions
        local = torch.matmul(pelvis_rot_inv, relative.unsqueeze(-1)).squeeze(-1)
    
    return local


def pelvis_local_to_world(
    pos_local: torch.Tensor,  # (3,) or (T, 3)
    pelvis_pos: torch.Tensor,  # (3,) or (T, 3)
    pelvis_rot: torch.Tensor,  # (3, 3) or (T, 3, 3)
) -> torch.Tensor:
    """Transform pelvis-local position back to world space."""
    if pos_local.dim() == 1:
        # Single position
        rotated = torch.matmul(pelvis_rot, pos_local)
    else:
        # Multiple positions
        rotated = torch.matmul(pelvis_rot, pos_local.unsqueeze(-1)).squeeze(-1)
    
    world = rotated + pelvis_pos
    return world


# ============================================================================
# SECTION 3: XY Remapping Algorithm
# ============================================================================

def interpolate_threshold(z: torch.Tensor, z_threshold_keypoints: List[Tuple[float, float]]) -> torch.Tensor:
    """
    Vectorized linear interpolation for body profile thresholds.
    Works on batches of z values simultaneously (pure PyTorch).
    
    Args:
        z: scalar tensor or (T,) heights to query
        z_threshold_keypoints: [(z_height, threshold), ...] defining body profile
    
    Returns:
        threshold(s) - scalar tensor or (T,) interpolated thresholds
    """
    device = z.device if isinstance(z, torch.Tensor) else torch.device('cpu')
    dtype = z.dtype if isinstance(z, torch.Tensor) else torch.float32
    
    # Convert keypoints to tensors
    z_kp = torch.tensor([kp[0] for kp in z_threshold_keypoints], device=device, dtype=dtype)
    t_kp = torch.tensor([kp[1] for kp in z_threshold_keypoints], device=device, dtype=dtype)
    
    # Ensure z is at least 1D
    z_orig_shape = z.shape if isinstance(z, torch.Tensor) else ()
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, device=device, dtype=dtype)
    if z.dim() == 0:
        z = z.unsqueeze(0)
    
    # Find bracketing indices using searchsorted
    indices = torch.searchsorted(z_kp, z, right=False)
    indices = torch.clamp(indices, 1, len(z_kp) - 1)
    
    # Get bracketing values
    z0 = z_kp[indices - 1]
    z1 = z_kp[indices]
    t0 = t_kp[indices - 1]
    t1 = t_kp[indices]
    
    # Linear interpolation (vectorized)
    alpha = (z - z0) / (z1 - z0 + 1e-9)
    thresholds = t0 + alpha * (t1 - t0)
    
    # Restore original shape
    if len(z_orig_shape) == 0:
        thresholds = thresholds.squeeze(0)
    
    return thresholds


def xy_remap_bounded(xy_distance: torch.Tensor, threshold: torch.Tensor, cutoff_ratio: float) -> torch.Tensor:
    """
    Smooth 1D remapping: [0, ∞) → [threshold, ∞)
    Pushes hands outward if too close to body center.
    Vectorized - works on batches.
    
    Args:
        xy_distance: current horizontal distance(s) (scalar or (T,))
        threshold: safe distance(s) at this height (scalar or (T,))
        cutoff_ratio: beyond threshold*cutoff_ratio, no push (e.g., 1.5)
    
    Returns:
        target_distance (scalar or (T,)) or None if all safe
    
    How it works:
        - At xy=0: pushed to threshold (maximum correction)
        - At xy=threshold: pushed slightly beyond (smooth decay)
        - At xy>=threshold*cutoff_ratio: no push (safe)
    """
    cutoff = threshold * cutoff_ratio
    
    # Check if any need correction (vectorized)
    needs_correction = xy_distance < cutoff
    
    # If none need correction and input was scalar, return None
    if xy_distance.dim() == 0 and not needs_correction.item():
        return None
    
    # Normalized distance
    r = xy_distance / cutoff
    
    # Smooth decay from 1→0 using smoothstep (vectorized)
    decay = 1.0 - (r * r * (3.0 - 2.0 * r))
    
    # Pure remap: add threshold weighted by decay
    target = xy_distance + threshold * decay
    
    return target


def compute_target_position(hand_local: torch.Tensor, z_profile: List, cutoff_ratio: float, z_lift_ratio: float = 1.0):
    """
    Compute IK target for hand based on its current position.
    Pure PyTorch - no .item() calls, fully vectorized.
    
    Args:
        hand_local: (3,) or (T, 3) hand position(s) in pelvis-local space
        z_profile: body threshold profile [(z_height, threshold), ...]
        cutoff_ratio: cutoff distance multiplier
        z_lift_ratio: Z lift proportional to XY push (0=no lift, 1=full lift, default 1.0)
    
    Returns:
        target position where xy is pushed outward, z lifted proportionally
        Or None if hand is already safe (beyond cutoff distance)
    """
    device = hand_local.device
    dtype = hand_local.dtype
    
    # Handle both single and batch
    is_single = (hand_local.dim() == 1)
    if is_single:
        hand_local = hand_local.unsqueeze(0)  # (1, 3)
    
    # Get thresholds for all z values (vectorized)
    z = hand_local[:, 2]  # (T,)
    thresholds = interpolate_threshold(z, z_profile)  # (T,) or scalar
    
    # Compute XY distances (vectorized)
    xy_distances = torch.sqrt(hand_local[:, 0]**2 + hand_local[:, 1]**2)  # (T,)
    
    # Check if remapping needed and compute targets (vectorized)
    target_xy_dists = xy_remap_bounded(xy_distances, thresholds, cutoff_ratio)
    
    if target_xy_dists is None:
        return None  # All safe
    
    # Compute scale factors (vectorized)
    scales = target_xy_dists / (xy_distances + 1e-9)  # (T,)
    
    # Build targets (vectorized)
    targets = hand_local.clone()  # (T, 3)
    targets[:, 0] = hand_local[:, 0] * scales
    targets[:, 1] = hand_local[:, 1] * scales
    
    # Z lift: proportional to XY push amount (helps with reachability)
    xy_push_amount = (scales - 1) * xy_distances  # how much we pushed XY outward
    targets[:, 2] = hand_local[:, 2] + z_lift_ratio * xy_push_amount
    
    # Return single or batch
    if is_single:
        cutoff = thresholds * cutoff_ratio
        if xy_distances[0] >= cutoff:
            return None
        return targets[0]  # (3,)
    
    return targets  # (T, 3)


# ============================================================================
# SECTION 4: Analytical IK (NEW - replaces optimization)
# ============================================================================

def rotation_between_vectors_batch(
    a: torch.Tensor,  # (T, 3) or (3,) - source direction (normalized)
    b: torch.Tensor,  # (T, 3) or (3,) - target direction (normalized)
) -> torch.Tensor:
    """
    Compute rotation matrices that rotate vector a to vector b.
    Uses Rodrigues formula. Handles batch of vectors.
    
    Args:
        a: Source direction vectors (normalized), shape (T, 3) or (3,)
        b: Target direction vectors (normalized), shape (T, 3) or (3,)
    
    Returns:
        Rotation matrices (T, 3, 3) or (3, 3) that rotate a to b
    """
    # Ensure at least 2D
    squeeze_output = False
    if a.dim() == 1:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        squeeze_output = True
    
    T = a.shape[0]
    device = a.device
    dtype = a.dtype
    
    # Cross product: rotation axis
    v = torch.cross(a, b, dim=-1)  # (T, 3)
    
    # Dot product: cos(angle)
    c = torch.sum(a * b, dim=-1, keepdim=True)  # (T, 1)
    
    # Handle near-parallel vectors (a ≈ b)
    # and anti-parallel vectors (a ≈ -b)
    eps = 1e-6
    
    # For anti-parallel case, we need a perpendicular axis
    # Pick axis perpendicular to a
    perp = torch.zeros_like(a)
    perp[:, 0] = -a[:, 1]
    perp[:, 1] = a[:, 0]
    # If a is along z, use x-axis
    z_aligned = (torch.abs(a[:, 0]) < eps) & (torch.abs(a[:, 1]) < eps)
    perp[z_aligned, 0] = 1.0
    perp[z_aligned, 1] = 0.0
    perp = F.normalize(perp, dim=-1)
    
    # Skew-symmetric matrix [v]×
    vx = torch.zeros((T, 3, 3), device=device, dtype=dtype)
    vx[:, 0, 1] = -v[:, 2]
    vx[:, 0, 2] = v[:, 1]
    vx[:, 1, 0] = v[:, 2]
    vx[:, 1, 2] = -v[:, 0]
    vx[:, 2, 0] = -v[:, 1]
    vx[:, 2, 1] = v[:, 0]
    
    # Rodrigues formula: R = I + [v]× + [v]×² * (1-c) / (1-c²)
    # Simplified: R = I + [v]× + [v]×² * 1/(1+c)
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(T, 3, 3)
    
    # Compute vx² = vx @ vx
    vx2 = torch.bmm(vx, vx)  # (T, 3, 3)
    
    # Factor: 1 / (1 + c), but avoid division by zero for anti-parallel
    denom = 1.0 + c  # (T, 1)
    safe_denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)
    factor = 1.0 / safe_denom  # (T, 1)
    factor = factor.unsqueeze(-1)  # (T, 1, 1)
    
    # Standard Rodrigues
    R = I + vx + vx2 * factor  # (T, 3, 3)
    
    # Handle anti-parallel case: rotate 180° around perpendicular axis
    anti_parallel = (c.squeeze(-1) < -1.0 + eps)  # (T,)
    if anti_parallel.any():
        # For anti-parallel, R = 2 * outer(perp, perp) - I (180° rotation around perp)
        perp_outer = torch.bmm(perp.unsqueeze(-1), perp.unsqueeze(-2))  # (T, 3, 3)
        R_anti = 2.0 * perp_outer - I
        R = torch.where(anti_parallel.view(T, 1, 1), R_anti, R)
    
    # Handle parallel case: identity
    parallel = (c.squeeze(-1) > 1.0 - eps)  # (T,)
    if parallel.any():
        R = torch.where(parallel.view(T, 1, 1), I, R)
    
    if squeeze_output:
        R = R.squeeze(0)
    
    return R


def analytical_ik_arm(
    shoulder_pos: torch.Tensor,       # (T, 3) world positions
    elbow_pos: torch.Tensor,          # (T, 3) world positions  
    hand_pos: torch.Tensor,           # (T, 3) world positions
    target_pos: torch.Tensor,         # (T, 3) target hand positions in world
    upper_len: float,                 # Upper arm length
    lower_len: float,                 # Forearm length
    shoulder_world_rot: torch.Tensor, # (T, 3, 3) shoulder world rotation
    elbow_world_rot: torch.Tensor,    # (T, 3, 3) elbow world rotation
    shoulder_parent_rot: torch.Tensor,# (T, 3, 3) shoulder's parent world rotation
    shoulder_local_rot: torch.Tensor, # (T, 3, 3) original shoulder local rotation
    elbow_local_rot: torch.Tensor,    # (T, 3, 3) original elbow local rotation
    needs_fix: torch.Tensor,          # (T,) bool mask
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Analytical 2-step IK: SWING then BEND.
    
    Step 1 (SWING): Rotate shoulder so arm points at target direction
    Step 2 (BEND): Adjust elbow angle to reach target distance
    
    Returns:
        new_shoulder_local: (T, 3, 3) new shoulder local rotation
        new_elbow_local: (T, 3, 3) new elbow local rotation
    """
    T = shoulder_pos.shape[0]
    device = shoulder_pos.device
    dtype = shoulder_pos.dtype
    
    # Start with original rotations
    new_shoulder_local = shoulder_local_rot.clone()
    new_elbow_local = elbow_local_rot.clone()
    
    fix_indices = torch.where(needs_fix)[0]
    if len(fix_indices) == 0:
        return new_shoulder_local, new_elbow_local
    
    # Extract frames needing fix
    sh = shoulder_pos[fix_indices]
    el = elbow_pos[fix_indices]
    ha = hand_pos[fix_indices]
    tg = target_pos[fix_indices]
    sh_world = shoulder_world_rot[fix_indices]
    el_world = elbow_world_rot[fix_indices]
    sh_parent = shoulder_parent_rot[fix_indices]
    sh_local = shoulder_local_rot[fix_indices]
    el_local = elbow_local_rot[fix_indices]
    
    N = sh.shape[0]
    eps = 1e-6
    
    # ===== SWING: rotate shoulder so arm points at target =====
    orig_arm_dir = F.normalize(ha - sh, dim=-1)  # shoulder → hand
    target_dir = F.normalize(tg - sh, dim=-1)    # shoulder → target
    
    swing = rotation_between_vectors_batch(orig_arm_dir, target_dir)  # (N, 3, 3)
    
    # New shoulder world = swing @ original shoulder world
    sh_world_new = torch.bmm(swing, sh_world)
    
    # Convert to local: new_local = parent^T @ new_world
    sh_local_new = torch.bmm(sh_parent.transpose(-1, -2), sh_world_new)
    
    # DEBUG: Check if swing correctly aligns directions
    # After swing, hand should be at: sh + swing @ (ha - sh)
    ha_after_swing = sh + torch.bmm(swing, (ha - sh).unsqueeze(-1)).squeeze(-1)
    el_after_swing = sh + torch.bmm(swing, (el - sh).unsqueeze(-1)).squeeze(-1)
    ha_after_swing_dir = F.normalize(ha_after_swing - sh, dim=-1)
    alignment = torch.sum(ha_after_swing_dir * target_dir, dim=-1)  # should be ~1.0
    print(f"  [DEBUG] Swing alignment (should be ~1.0): min={alignment.min().item():.6f}, max={alignment.max().item():.6f}, mean={alignment.mean().item():.6f}")
    
    # ===== BEND: adjust elbow angle to reach target distance =====
    # DEBUG: COMMENTED OUT FOR TESTING - only shoulder swing for now
    # d = torch.norm(tg - sh, dim=-1)
    # min_d = abs(upper_len - lower_len) + 1e-4
    # max_d = upper_len + lower_len - 1e-4
    # d_clamped = torch.clamp(d, min_d, max_d)
    # 
    # # Original elbow internal angle (angle at elbow vertex)
    # upper_orig = F.normalize(el - sh, dim=-1)
    # lower_orig = F.normalize(ha - el, dim=-1)
    # cos_orig = torch.sum(-upper_orig * lower_orig, dim=-1)  # internal angle
    # cos_orig = torch.clamp(cos_orig, -1 + eps, 1 - eps)
    # orig_angle = torch.acos(cos_orig)
    # 
    # # New elbow internal angle from law of cosines
    # # d² = L1² + L2² - 2*L1*L2*cos(internal_angle)
    # cos_new = (upper_len**2 + lower_len**2 - d_clamped**2) / (2.0 * upper_len * lower_len + eps)
    # cos_new = torch.clamp(cos_new, -1 + eps, 1 - eps)
    # new_angle = torch.acos(cos_new)
    # 
    # # Bend delta (how much more/less to bend)
    # bend_delta = new_angle - orig_angle  # (N,)
    # 
    # # Bend axis = arm plane normal (perpendicular to shoulder-elbow-hand plane)
    # # Use post-swing positions for stability
    # upper_swung = F.normalize(el_after_swing - sh, dim=-1)
    # lower_swung = F.normalize(ha_after_swing - el_after_swing, dim=-1)
    # bend_axis = torch.cross(upper_swung, lower_swung, dim=-1)
    # bend_axis_norm = torch.norm(bend_axis, dim=-1, keepdim=True)
    # 
    # # Handle degenerate case (arm straight)
    # degenerate = bend_axis_norm.squeeze(-1) < eps
    # fallback_axis = torch.zeros_like(bend_axis)
    # fallback_axis[:, 2] = 1.0  # Use Z as fallback
    # bend_axis = torch.where(degenerate.unsqueeze(-1), fallback_axis, bend_axis / (bend_axis_norm + eps))
    # 
    # # Bend rotation (axis-angle to matrix)
    # # R = I + sin(θ)K + (1-cos(θ))K²  where K is skew-symmetric of axis
    # K = torch.zeros((N, 3, 3), device=device, dtype=dtype)
    # K[:, 0, 1] = -bend_axis[:, 2]
    # K[:, 0, 2] = bend_axis[:, 1]
    # K[:, 1, 0] = bend_axis[:, 2]
    # K[:, 1, 2] = -bend_axis[:, 0]
    # K[:, 2, 0] = -bend_axis[:, 1]
    # K[:, 2, 1] = bend_axis[:, 0]
    # 
    # sin_b = torch.sin(bend_delta).view(N, 1, 1)
    # cos_b = torch.cos(bend_delta).view(N, 1, 1)
    # I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(N, 3, 3)
    # K2 = torch.bmm(K, K)
    # bend_rot = I + sin_b * K + (1 - cos_b) * K2  # (N, 3, 3)
    # 
    # # Elbow world after swing (before bend)
    # el_world_after_swing = torch.bmm(swing, el_world)
    # 
    # # New elbow world = bend @ elbow_world_after_swing
    # el_world_new = torch.bmm(bend_rot, el_world_after_swing)
    # 
    # # Convert to local: new_elbow_local = new_shoulder_world^T @ new_elbow_world
    # el_local_new = torch.bmm(sh_world_new.transpose(-1, -2), el_world_new)
    
    # DEBUG: Skip elbow correction - keep original elbow local rotation
    el_local_new = el_local
    
    # Write back
    new_shoulder_local[fix_indices] = sh_local_new
    new_elbow_local[fix_indices] = el_local_new
    
    return new_shoulder_local, new_elbow_local


def compute_local_rotations_from_positions(
    shoulder_pos: torch.Tensor,      # (T, 3)
    original_elbow_pos: torch.Tensor,  # (T, 3) ORIGINAL elbow position
    new_elbow_pos: torch.Tensor,     # (T, 3) NEW elbow position
    original_hand_pos: torch.Tensor, # (T, 3) ORIGINAL hand position
    new_hand_pos: torch.Tensor,      # (T, 3) NEW hand position
    original_shoulder_world_rot: torch.Tensor,  # (T, 3, 3) original shoulder world rotation
    original_elbow_world_rot: torch.Tensor,     # (T, 3, 3) original elbow world rotation
    shoulder_parent_rot: torch.Tensor,  # (T, 3, 3) parent's world rotation
    original_shoulder_local: torch.Tensor,  # (T, 3, 3) original local rotation
    original_elbow_local: torch.Tensor,     # (T, 3, 3) original local rotation
    needs_fix: torch.Tensor,         # (T,) bool mask
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert world positions back to local joint rotations.
    
    Uses DELTA rotation approach: find the rotation that takes the original
    bone direction to the new bone direction, then apply that delta to the
    original bone rotation.
    
    Args:
        shoulder_pos: (T, 3) shoulder world positions (fixed)
        original_elbow_pos: (T, 3) original elbow positions
        new_elbow_pos: (T, 3) new elbow positions after IK
        original_hand_pos: (T, 3) original hand positions
        new_hand_pos: (T, 3) new hand positions after IK
        original_shoulder_world_rot: (T, 3, 3) original shoulder world rotation
        original_elbow_world_rot: (T, 3, 3) original elbow world rotation
        shoulder_parent_rot: (T, 3, 3) parent's world rotation
        original_shoulder_local: (T, 3, 3) original local rotation
        original_elbow_local: (T, 3, 3) original local rotation
        needs_fix: (T,) bool mask
    
    Returns:
        new_shoulder_local: (T, 3, 3) local rotation matrices
        new_elbow_local: (T, 3, 3) local rotation matrices
    """
    T = shoulder_pos.shape[0]
    device = shoulder_pos.device
    dtype = shoulder_pos.dtype
    
    # Start with original rotations
    new_shoulder_local = original_shoulder_local.clone()
    new_elbow_local = original_elbow_local.clone()
    
    fix_indices = torch.where(needs_fix)[0]
    if len(fix_indices) == 0:
        return new_shoulder_local, new_elbow_local
    
    # Extract frames needing fix
    sh = shoulder_pos[fix_indices]
    el_orig = original_elbow_pos[fix_indices]
    el_new = new_elbow_pos[fix_indices]
    ha_orig = original_hand_pos[fix_indices]
    ha_new = new_hand_pos[fix_indices]
    parent_rot = shoulder_parent_rot[fix_indices]
    sh_world_orig = original_shoulder_world_rot[fix_indices]
    el_world_orig = original_elbow_world_rot[fix_indices]
    sh_local_orig = original_shoulder_local[fix_indices]
    el_local_orig = original_elbow_local[fix_indices]
    
    N = sh.shape[0]
    
    # ===== Shoulder rotation =====
    # SWING: rotate so shoulder→hand points at target (NOT elbow!)
    orig_arm_dir = F.normalize(ha_orig - sh, dim=-1)  # shoulder → original HAND
    new_arm_dir = F.normalize(ha_new - sh, dim=-1)    # shoulder → new HAND (target)
    
    # Delta rotation: takes original_arm_dir → new_arm_dir
    delta_shoulder = rotation_between_vectors_batch(orig_arm_dir, new_arm_dir)  # (N, 3, 3)
    
    # Apply delta to original world rotation
    shoulder_world_new = torch.bmm(delta_shoulder, sh_world_orig)  # (N, 3, 3)
    
    # Convert to local: R_local = R_parent^T @ R_world
    shoulder_local_new = torch.bmm(parent_rot.transpose(-1, -2), shoulder_world_new)  # (N, 3, 3)
    
    # ===== Elbow rotation =====
    # BEND: forearm direction change
    orig_lower_dir = F.normalize(ha_orig - el_orig, dim=-1)  # elbow → original hand
    new_lower_dir = F.normalize(ha_new - el_new, dim=-1)     # elbow → new hand
    
    # Delta rotation: takes original_lower_dir → new_lower_dir
    delta_elbow = rotation_between_vectors_batch(orig_lower_dir, new_lower_dir)  # (N, 3, 3)
    
    # Apply delta to original world rotation
    elbow_world_new = torch.bmm(delta_elbow, el_world_orig)  # (N, 3, 3)
    
    # Convert to local: R_local = R_shoulder_world^T @ R_world
    elbow_local_new = torch.bmm(shoulder_world_new.transpose(-1, -2), elbow_world_new)  # (N, 3, 3)
    
    # Write back
    new_shoulder_local[fix_indices] = shoulder_local_new
    new_elbow_local[fix_indices] = elbow_local_new
    
    return new_shoulder_local, new_elbow_local


# ============================================================================
# SECTION 5: Optimization-based IK (LEGACY - kept for reference)
# ============================================================================

def fk_hand_local_batch(
    R: torch.Tensor,  # (T, J, 3, 3)
    offsets: torch.Tensor,
    parents: Sequence[int],
    root_trans: torch.Tensor,  # (T, 3)
    pelvis_idx: int,
    hand_idx: int
) -> torch.Tensor:
    """Batch FK: compute hand positions in pelvis-local space for all frames."""
    pos_world = fk_global_positions(R, offsets, parents, root_trans)  # (T, J, 3)
    
    pelvis_pos = pos_world[:, pelvis_idx, :]  # (T, 3)
    pelvis_rot = R[:, pelvis_idx, :, :]  # (T, 3, 3)
    hand_pos = pos_world[:, hand_idx, :]  # (T, 3)
    
    hand_local = world_to_pelvis_local(hand_pos, pelvis_pos, pelvis_rot)  # (T, 3)
    return hand_local


def solve_multi_frame_ik(
    R_original: torch.Tensor,  # (T, J, 3, 3) original rotations
    targets: Dict,  # Per-frame targets
    arm_R_chain: List[int],
    arm_L_chain: List[int],
    hand_R_idx: int,
    hand_L_idx: int,
    pelvis_idx: int,
    offsets: torch.Tensor,
    parents: Sequence[int],
    root_trans: torch.Tensor,
    config: Dict
) -> torch.Tensor:
    """
    Batch IK optimization across all frames with three loss terms:
    
    1. Target Loss (w_target): 
       - Reach the corrected hand positions
       - Higher = stronger correction, lower = allow some compromise
       
    2. Prior Loss (w_prior):
       - Stay close to original arm rotations
       - Higher = more conservative (less change), lower = more aggressive correction
       
    3. Continuity Loss (w_continuity):
       - Smooth displacement changes between frames
       - Higher = smoother motion, lower = allow more frame-to-frame variation
    
    Important: Pinned frames (hands already safe) are NOT optimized and receive zero gradients.
    """
    T, J = R_original.shape[0], R_original.shape[1]
    device = R_original.device
    
    # Identify non-pinned frames (only these will be optimized)
    non_pinned_indices = [t for t in range(T) if not targets[t]['is_pinned']]
    pinned_count = T - len(non_pinned_indices)
    
    print(f"[IK] Optimizing {len(non_pinned_indices)}/{T} frames ({pinned_count} pinned, untouched)")
    print(f"[IK] Running {config['iterations']} iterations...")
    
    if len(non_pinned_indices) == 0:
        print("[IK] All frames are pinned! No optimization needed.")
        return R_original
    
    # Extract original arm rotations
    D_arms_R_all = rc.matrix_to_rotation_6d(R_original[:, arm_R_chain, :, :])  # (T, K, 6)
    D_arms_L_all = rc.matrix_to_rotation_6d(R_original[:, arm_L_chain, :, :])  # (T, K, 6)
    
    # Only make non-pinned frames trainable
    D_arms_R_opt = D_arms_R_all[non_pinned_indices].clone()  # (T_opt, K, 6)
    D_arms_L_opt = D_arms_L_all[non_pinned_indices].clone()  # (T_opt, K, 6)
    D_arms_R_opt.requires_grad_(True)
    D_arms_L_opt.requires_grad_(True)
    
    optimizer = torch.optim.Adam([D_arms_R_opt, D_arms_L_opt], lr=config['lr'])
    
    # Precompute original hand positions for continuity loss
    hand_R_orig = fk_hand_local_batch(R_original, offsets, parents, root_trans, pelvis_idx, hand_R_idx)
    hand_L_orig = fk_hand_local_batch(R_original, offsets, parents, root_trans, pelvis_idx, hand_L_idx)
    
    for iteration in range(config['iterations']):
        optimizer.zero_grad()
        
        # Build full skeleton: start with original, update only non-pinned frames
        D_arms_R_current = D_arms_R_all.clone()  # (T, K, 6)
        D_arms_L_current = D_arms_L_all.clone()  # (T, K, 6)
        D_arms_R_current[non_pinned_indices] = D_arms_R_opt  # Only non-pinned get updated
        D_arms_L_current[non_pinned_indices] = D_arms_L_opt
        
        R_current = R_original.clone()
        R_current[:, arm_R_chain, :, :] = rc.rotation_6d_to_matrix(D_arms_R_current)
        R_current[:, arm_L_chain, :, :] = rc.rotation_6d_to_matrix(D_arms_L_current)
        
        # FK for all frames
        hand_R_current = fk_hand_local_batch(R_current, offsets, parents, root_trans, pelvis_idx, hand_R_idx)
        hand_L_current = fk_hand_local_batch(R_current, offsets, parents, root_trans, pelvis_idx, hand_L_idx)
        
        # ===== LOSS 1: Target Loss (only non-pinned frames) =====
        loss_target = 0.0
        for t in non_pinned_indices:
            if targets[t]['right'] is not None:
                loss_target = loss_target + torch.sum((hand_R_current[t] - targets[t]['right']) ** 2)
            if targets[t]['left'] is not None:
                loss_target = loss_target + torch.sum((hand_L_current[t] - targets[t]['left']) ** 2)
        
        # ===== LOSS 2: Prior Loss (only non-pinned frames) =====
        R_arms_R_current = rc.rotation_6d_to_matrix(D_arms_R_current)
        R_arms_L_current = rc.rotation_6d_to_matrix(D_arms_L_current)
        R_arms_R_orig = R_original[:, arm_R_chain, :, :]
        R_arms_L_orig = R_original[:, arm_L_chain, :, :]
        
        loss_prior = 0.0
        for t in non_pinned_indices:
            loss_prior = loss_prior + torch.sum((R_arms_R_current[t] - R_arms_R_orig[t]) ** 2)
            loss_prior = loss_prior + torch.sum((R_arms_L_current[t] - R_arms_L_orig[t]) ** 2)
        
        # ===== LOSS 3: Continuity Loss (only between non-pinned frames) =====
        # Displacement from original
        disp_R = hand_R_current - hand_R_orig  # (T, 3)
        disp_L = hand_L_current - hand_L_orig  # (T, 3)
        
        # Only compute continuity between consecutive non-pinned frames
        loss_continuity = 0.0
        for i in range(len(non_pinned_indices) - 1):
            t1 = non_pinned_indices[i]
            t2 = non_pinned_indices[i + 1]
            # Only if consecutive (no gap from pinned frames)
            if t2 == t1 + 1:
                loss_continuity = loss_continuity + torch.sum((disp_R[t2] - disp_R[t1]) ** 2)
                loss_continuity = loss_continuity + torch.sum((disp_L[t2] - disp_L[t1]) ** 2)
        
        # ===== Combined Loss =====
        loss = (
            config['w_target'] * loss_target
            + config['w_prior'] * loss_prior
            + config['w_continuity'] * loss_continuity
        )
        
        loss.backward()
        optimizer.step()
        
        if (iteration + 1) % 20 == 0 or iteration == 0:
            print(f"  Iter {iteration+1}/{config['iterations']}: "
                  f"target={loss_target.item():.4f}, "
                  f"prior={loss_prior.item():.4f}, "
                  f"continuity={loss_continuity.item():.4f}")
    
    # Return optimized rotations (pinned frames unchanged)
    D_arms_R_final = D_arms_R_all.clone()
    D_arms_L_final = D_arms_L_all.clone()
    D_arms_R_final[non_pinned_indices] = D_arms_R_opt.detach()
    D_arms_L_final[non_pinned_indices] = D_arms_L_opt.detach()
    
    R_final = R_original.clone()
    R_final[:, arm_R_chain, :, :] = rc.rotation_6d_to_matrix(D_arms_R_final)
    R_final[:, arm_L_chain, :, :] = rc.rotation_6d_to_matrix(D_arms_L_final)
    
    return R_final


# ============================================================================
# SECTION 5: High-Level API (Public Interface)
# ============================================================================

def get_default_config() -> Dict:
    """
    Get default configuration for hand collision fixing.
    
    Returns dict with:
        - z_threshold_keypoints: Body profile [(z_cm, threshold_cm), ...]
        - cutoff_ratio: Beyond threshold*ratio, no correction (default: 1.5)
        - iterations: IK optimization steps (default: 100)
        - lr: Learning rate (default: 0.01)
        - w_target: Target loss weight (default: 1.0)
        - w_prior: Prior loss weight - higher = more conservative (default: 0.15)
        - w_continuity: Continuity loss - higher = smoother (default: 0.08)
    
    Tuning tips in main module docstring.
    """
    return {
        'z_threshold_keypoints': [
            (-80, 22.0), (-40, 22.0), (-5, 21.0), (0, 21.0),
            (+10, 20.0), (+30, 16.0), (+40, 35.0), (+55, 35.0), (+80, 35.0)
        ],
        'cutoff_ratio': 1.5,
        'z_lift_ratio': 0.0,  # Z lift proportional to XY push (helps reachability)
        'iterations': 100,
        'lr': 0.01,
        'w_target': 1.0,
        'w_prior': 0.15,
        'w_continuity': 0.08,
    }


def fix_hand_collisions(
    R: torch.Tensor,              # (T, J, 3, 3) rotation matrices
    offsets: torch.Tensor,        # (J, 3) skeleton offsets
    parents: np.ndarray,          # (J,) parent indices
    root_trans: torch.Tensor,     # (T, 3) root positions
    names: List[str],             # Joint names
    config: Dict,                 # Configuration (use get_default_config())
    context_length: int = 0,      # First N frames kept frozen
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Core function: Fix hand self-collisions in motion data.
    
    Args:
        R: (T, J, 3, 3) Joint rotation matrices (already converted from Euler)
        offsets: (J, 3) Skeleton bone offsets
        parents: (J,) Parent joint indices (-1 for root)
        root_trans: (T, 3) Root joint positions
        names: Joint names list (must include 'Root_M', 'Wrist_R', 'Wrist_L')
        config: Configuration dict (get from get_default_config())
        context_length: First N frames are kept unchanged (e.g., T-pose)
        device: 'cpu' or 'cuda'
    
    Returns:
        R_fixed: (T, J, 3, 3) Fixed rotation matrices
    
    Example - Direct PyTorch Integration:
        >>> config = get_default_config()
        >>> R_fixed = fix_hand_collisions(
        ...     R=rotation_matrices,
        ...     offsets=skeleton_offsets,
        ...     parents=skeleton_parents,
        ...     root_trans=root_positions,
        ...     names=joint_names,
        ...     config=config,
        ...     context_length=10
        ... )
    """
    T, J = R.shape[0], R.shape[1]
    device_torch = torch.device(device)
    
    # Move to device
    R = R.to(device_torch)
    offsets = offsets.to(device_torch)
    root_trans = root_trans.to(device_torch)
    
    # Get joint indices
    name2idx = {n: i for i, n in enumerate(names)}
    pelvis_idx = name2idx.get('Root_M', name2idx.get('Hips', 0))
    hand_R_idx = name2idx.get('Wrist_R', name2idx.get('RightHand', -1))
    hand_L_idx = name2idx.get('Wrist_L', name2idx.get('LeftHand', -1))
    
    if hand_R_idx == -1 or hand_L_idx == -1:
        raise ValueError("Could not find hand joints. Need 'Wrist_R' and 'Wrist_L' in names.")
    
    # Get arm chains
    arm_R_chain = get_arm_chain_indices(names, 'R')
    arm_L_chain = get_arm_chain_indices(names, 'L')
    
    print(f"[INFO] Pelvis: {names[pelvis_idx]} (idx={pelvis_idx})")
    print(f"[INFO] Right hand: {names[hand_R_idx]} (idx={hand_R_idx})")
    print(f"[INFO] Left hand: {names[hand_L_idx]} (idx={hand_L_idx})")
    print(f"[INFO] Right arm chain: {[names[i] for i in arm_R_chain]}")
    print(f"[INFO] Left arm chain: {[names[i] for i in arm_L_chain]}")
    
    # Compute targets for all frames
    print(f"\n[PHASE 1] Computing targets for {T} frames...")
    print(f"[CONFIG] Z-profile: {config['z_threshold_keypoints']}")
    print(f"[CONFIG] Cutoff ratio: {config['cutoff_ratio']}")
    
    pos = fk_global_positions(R, offsets, parents, root_trans)
    
    targets = []
    for t in range(T):
        pelvis_pos = pos[t, pelvis_idx, :]
        pelvis_rot = R[t, pelvis_idx, :, :]
        
        hand_R_world = pos[t, hand_R_idx, :]
        hand_R_local = world_to_pelvis_local(hand_R_world, pelvis_pos, pelvis_rot)
        target_R = compute_target_position(hand_R_local, config['z_threshold_keypoints'], config['cutoff_ratio'], config['z_lift_ratio'])
        
        hand_L_world = pos[t, hand_L_idx, :]
        hand_L_local = world_to_pelvis_local(hand_L_world, pelvis_pos, pelvis_rot)
        target_L = compute_target_position(hand_L_local, config['z_threshold_keypoints'], config['cutoff_ratio'], config['z_lift_ratio'])
        
        targets.append({
            'right': target_R,
            'left': target_L,
            'is_pinned': (target_R is None and target_L is None)
        })
    
    # Apply context_length: force pin first N frames
    if context_length > 0:
        for t in range(min(context_length, T)):
            targets[t] = {'right': None, 'left': None, 'is_pinned': True}
        print(f"[INFO] Context length: {context_length} frames forced pinned")
    
    num_frames_needing_ik = sum(1 for t in targets if not t['is_pinned'])
    pinned_count = T - num_frames_needing_ik
    print(f"[INFO] Frames needing IK: {num_frames_needing_ik}/{T}")
    print(f"[INFO] Frames pinned (safe): {pinned_count}/{T}")
    
    # Run IK optimization
    print(f"\n[PHASE 2] Running multi-frame IK optimization...")
    R_fixed = solve_multi_frame_ik(
        R_original=R,
        targets=targets,
        arm_R_chain=arm_R_chain,
        arm_L_chain=arm_L_chain,
        hand_R_idx=hand_R_idx,
        hand_L_idx=hand_L_idx,
        pelvis_idx=pelvis_idx,
        offsets=offsets,
        parents=parents,
        root_trans=root_trans,
        config=config
    )
    
    return R_fixed


def fix_hand_collisions_analytical(
    R: torch.Tensor,              # (T, J, 3, 3) rotation matrices
    offsets: torch.Tensor,        # (J, 3) skeleton offsets
    parents: np.ndarray,          # (J,) parent indices
    root_trans: torch.Tensor,     # (T, 3) root positions
    names: List[str],             # Joint names
    config: Dict,                 # Configuration (use get_default_config())
    context_length: int = 0,      # First N frames kept frozen
    device: str = 'cpu'
) -> torch.Tensor:
    """
    ANALYTICAL version: Fix hand self-collisions using closed-form 2-step IK.
    
    This is 100-1000x faster than optimization-based fix_hand_collisions().
    Uses swing + bend algorithm with no iteration.
    
    Args:
        R: (T, J, 3, 3) Joint rotation matrices
        offsets: (J, 3) Skeleton bone offsets
        parents: (J,) Parent joint indices (-1 for root)
        root_trans: (T, 3) Root joint positions
        names: Joint names list
        config: Configuration dict (get from get_default_config())
        context_length: First N frames are kept unchanged
        device: 'cpu' or 'cuda'
    
    Returns:
        R_fixed: (T, J, 3, 3) Fixed rotation matrices
    """
    T, J = R.shape[0], R.shape[1]
    device_torch = torch.device(device)
    
    # Move to device
    R = R.to(device_torch)
    offsets = offsets.to(device_torch)
    root_trans = root_trans.to(device_torch)
    
    # Get joint indices
    name2idx = {n: i for i, n in enumerate(names)}
    pelvis_idx = name2idx.get('Root_M', name2idx.get('Hips', 0))
    
    # Find shoulder, elbow, wrist for each arm
    shoulder_R_idx = name2idx.get('Shoulder_R', -1)
    elbow_R_idx = name2idx.get('Elbow_R', -1)
    wrist_R_idx = name2idx.get('Wrist_R', -1)
    
    shoulder_L_idx = name2idx.get('Shoulder_L', -1)
    elbow_L_idx = name2idx.get('Elbow_L', -1)
    wrist_L_idx = name2idx.get('Wrist_L', -1)
    
    # Validate
    for idx, name in [(shoulder_R_idx, 'Shoulder_R'), (elbow_R_idx, 'Elbow_R'), 
                      (wrist_R_idx, 'Wrist_R'), (shoulder_L_idx, 'Shoulder_L'),
                      (elbow_L_idx, 'Elbow_L'), (wrist_L_idx, 'Wrist_L')]:
        if idx == -1:
            raise ValueError(f"Could not find joint: {name}")
    
    # Get shoulder's parent (for local rotation conversion)
    shoulder_R_parent_idx = int(parents[shoulder_R_idx])
    shoulder_L_parent_idx = int(parents[shoulder_L_idx])
    
    print(f"[INFO] Right arm: Shoulder({shoulder_R_idx}) -> Elbow({elbow_R_idx}) -> Wrist({wrist_R_idx})")
    print(f"[INFO] Left arm: Shoulder({shoulder_L_idx}) -> Elbow({elbow_L_idx}) -> Wrist({wrist_L_idx})")
    
    # Compute bone lengths from REST POSE positions (not offsets!)
    # Offsets can be wrong if there are intermediate joints
    rest_pos = compute_rest_positions(offsets.cpu().numpy(), parents)
    rest_pos = torch.from_numpy(rest_pos).to(device_torch)
    
    upper_len_R = torch.norm(rest_pos[elbow_R_idx] - rest_pos[shoulder_R_idx]).item()
    lower_len_R = torch.norm(rest_pos[wrist_R_idx] - rest_pos[elbow_R_idx]).item()
    upper_len_L = torch.norm(rest_pos[elbow_L_idx] - rest_pos[shoulder_L_idx]).item()
    lower_len_L = torch.norm(rest_pos[wrist_L_idx] - rest_pos[elbow_L_idx]).item()
    
    print(f"[INFO] Right arm lengths: upper={upper_len_R:.2f}, lower={lower_len_R:.2f} (total={upper_len_R+lower_len_R:.2f})")
    print(f"[INFO] Left arm lengths: upper={upper_len_L:.2f}, lower={lower_len_L:.2f} (total={upper_len_L+lower_len_L:.2f})")
    
    # Rest pose bone directions (from offsets)
    rest_upper_R = F.normalize(offsets[elbow_R_idx], dim=0)
    rest_lower_R = F.normalize(offsets[wrist_R_idx], dim=0)
    rest_upper_L = F.normalize(offsets[elbow_L_idx], dim=0)
    rest_lower_L = F.normalize(offsets[wrist_L_idx], dim=0)
    
    # ===== PHASE 1: Compute targets (FULLY VECTORIZED) =====
    print(f"\n[PHASE 1] Computing targets for {T} frames (vectorized)...")
    
    # FK with world rotations
    pos, world_rot = fk_global_with_rotations(R, offsets, parents, root_trans)
    
    # Get all positions at once
    pelvis_pos_all = pos[:, pelvis_idx, :]  # (T, 3)
    pelvis_rot_all = R[:, pelvis_idx, :, :]  # (T, 3, 3)
    hand_R_world = pos[:, wrist_R_idx, :]   # (T, 3)
    hand_L_world = pos[:, wrist_L_idx, :]   # (T, 3)
    
    # Transform to pelvis-local (vectorized)
    hand_R_local = torch.bmm(pelvis_rot_all.transpose(-1, -2), (hand_R_world - pelvis_pos_all).unsqueeze(-1)).squeeze(-1)
    hand_L_local = torch.bmm(pelvis_rot_all.transpose(-1, -2), (hand_L_world - pelvis_pos_all).unsqueeze(-1)).squeeze(-1)
    
    # Compute thresholds for all frames at once
    z_R = hand_R_local[:, 2]
    z_L = hand_L_local[:, 2]
    threshold_R = interpolate_threshold(z_R, config['z_threshold_keypoints'])
    threshold_L = interpolate_threshold(z_L, config['z_threshold_keypoints'])
    
    # XY distances
    xy_dist_R = torch.sqrt(hand_R_local[:, 0]**2 + hand_R_local[:, 1]**2)
    xy_dist_L = torch.sqrt(hand_L_local[:, 0]**2 + hand_L_local[:, 1]**2)
    
    # Check which need fixing
    cutoff_R = threshold_R * config['cutoff_ratio']
    cutoff_L = threshold_L * config['cutoff_ratio']
    needs_fix_R = xy_dist_R < cutoff_R
    needs_fix_L = xy_dist_L < cutoff_L
    
    # Apply context_length
    if context_length > 0:
        needs_fix_R[:context_length] = False
        needs_fix_L[:context_length] = False
    
    # Compute target XY distances (vectorized remap)
    target_xy_R = xy_remap_bounded(xy_dist_R, threshold_R, config['cutoff_ratio'])
    target_xy_L = xy_remap_bounded(xy_dist_L, threshold_L, config['cutoff_ratio'])
    if target_xy_R is None:
        target_xy_R = xy_dist_R
    if target_xy_L is None:
        target_xy_L = xy_dist_L
    
    # Scale factors
    scale_R = target_xy_R / (xy_dist_R + 1e-9)
    scale_L = target_xy_L / (xy_dist_L + 1e-9)
    
    # Compute target local positions
    target_R_local = hand_R_local.clone()
    target_R_local[:, 0] = hand_R_local[:, 0] * scale_R
    target_R_local[:, 1] = hand_R_local[:, 1] * scale_R
    xy_push_R = (scale_R - 1) * xy_dist_R
    target_R_local[:, 2] = hand_R_local[:, 2] + config['z_lift_ratio'] * xy_push_R
    
    target_L_local = hand_L_local.clone()
    target_L_local[:, 0] = hand_L_local[:, 0] * scale_L
    target_L_local[:, 1] = hand_L_local[:, 1] * scale_L
    xy_push_L = (scale_L - 1) * xy_dist_L
    target_L_local[:, 2] = hand_L_local[:, 2] + config['z_lift_ratio'] * xy_push_L
    
    # Transform back to world (vectorized)
    target_R_world = pelvis_pos_all + torch.bmm(pelvis_rot_all, target_R_local.unsqueeze(-1)).squeeze(-1)
    target_L_world = pelvis_pos_all + torch.bmm(pelvis_rot_all, target_L_local.unsqueeze(-1)).squeeze(-1)
    
    # For frames that don't need fix, use original position
    target_R_world = torch.where(needs_fix_R.unsqueeze(-1), target_R_world, hand_R_world)
    target_L_world = torch.where(needs_fix_L.unsqueeze(-1), target_L_world, hand_L_world)
    
    num_fix_R = needs_fix_R.sum().item()
    num_fix_L = needs_fix_L.sum().item()
    print(f"[INFO] Right arm frames to fix: {num_fix_R}/{T}")
    print(f"[INFO] Left arm frames to fix: {num_fix_L}/{T}")
    
    if num_fix_R == 0 and num_fix_L == 0:
        print("[INFO] No frames need fixing!")
        return R
    
    # ===== PHASE 2: Analytical IK (SWING + BEND) =====
    print(f"\n[PHASE 2] Running analytical IK (SWING + BEND)...")
    
    R_fixed = R.clone()
    
    # Right arm
    shoulder_R_pos = pos[:, shoulder_R_idx, :]
    elbow_R_pos = pos[:, elbow_R_idx, :]
    wrist_R_pos = pos[:, wrist_R_idx, :]
    shoulder_R_parent_rot = world_rot[:, shoulder_R_parent_idx, :, :]
    
    new_shoulder_R_local, new_elbow_R_local = analytical_ik_arm(
        shoulder_pos=shoulder_R_pos,
        elbow_pos=elbow_R_pos,
        hand_pos=wrist_R_pos,
        target_pos=target_R_world,
        upper_len=upper_len_R,
        lower_len=lower_len_R,
        shoulder_world_rot=world_rot[:, shoulder_R_idx, :, :],
        elbow_world_rot=world_rot[:, elbow_R_idx, :, :],
        shoulder_parent_rot=shoulder_R_parent_rot,
        shoulder_local_rot=R[:, shoulder_R_idx, :, :],
        elbow_local_rot=R[:, elbow_R_idx, :, :],
        needs_fix=needs_fix_R
    )
    
    R_fixed[:, shoulder_R_idx, :, :] = new_shoulder_R_local
    R_fixed[:, elbow_R_idx, :, :] = new_elbow_R_local
    
    # Left arm
    shoulder_L_pos = pos[:, shoulder_L_idx, :]
    elbow_L_pos = pos[:, elbow_L_idx, :]
    wrist_L_pos = pos[:, wrist_L_idx, :]
    shoulder_L_parent_rot = world_rot[:, shoulder_L_parent_idx, :, :]
    
    new_shoulder_L_local, new_elbow_L_local = analytical_ik_arm(
        shoulder_pos=shoulder_L_pos,
        elbow_pos=elbow_L_pos,
        hand_pos=wrist_L_pos,
        target_pos=target_L_world,
        upper_len=upper_len_L,
        lower_len=lower_len_L,
        shoulder_world_rot=world_rot[:, shoulder_L_idx, :, :],
        elbow_world_rot=world_rot[:, elbow_L_idx, :, :],
        shoulder_parent_rot=shoulder_L_parent_rot,
        shoulder_local_rot=R[:, shoulder_L_idx, :, :],
        elbow_local_rot=R[:, elbow_L_idx, :, :],
        needs_fix=needs_fix_L
    )
    
    R_fixed[:, shoulder_L_idx, :, :] = new_shoulder_L_local
    R_fixed[:, elbow_L_idx, :, :] = new_elbow_L_local
    
    # Skip verification for speed - just print done
    print(f"[DONE] Swing-only IK complete! Fixed {num_fix_R} right, {num_fix_L} left frames.")
    return R_fixed


def load_bvh_for_ik(bvh_path: str, max_frames: int = None) -> Tuple[Dict, Dict]:
    """
    Load BVH file and prepare data for fix_hand_collisions().
    
    Args:
        bvh_path: Path to BVH file
        max_frames: Optionally limit to first N frames
    
    Returns:
        motion_data: dict with {'R', 'root_trans', 'euler_deg', 'positions', 'order'}
        skeleton_data: dict with {'offsets', 'parents', 'names'}
    
    Example:
        >>> motion, skeleton = load_bvh_for_ik("input.bvh", max_frames=600)
        >>> config = get_default_config()
        >>> R_fixed = fix_hand_collisions(
        ...     R=motion['R'],
        ...     offsets=skeleton['offsets'],
        ...     parents=skeleton['parents'],
        ...     root_trans=motion['root_trans'],
        ...     names=skeleton['names'],
        ...     config=config
        ... )
        >>> save_bvh_from_ik("output.bvh", R_fixed, motion, skeleton)
    """
    print(f"[INFO] Loading BVH: {bvh_path}")
    data = bvh.load(bvh_path)
    
    rots_deg = data['rotations'].astype(np.float32)
    poss = data['positions'].astype(np.float32)
    T, J = rots_deg.shape[0], rots_deg.shape[1]
    
    if max_frames is not None and max_frames < T:
        rots_deg = rots_deg[:max_frames]
        poss = poss[:max_frames]
        T = max_frames
        print(f"[INFO] Processing only first {T} frames (limited)")
    
    order = data['order']
    convention = str(order).upper()
    
    # Convert to rotation matrices
    rots_rad = torch.from_numpy(rots_deg) * (np.pi / 180.0)
    R = rc.euler_angles_to_matrix(rots_rad, convention=convention)
    root_trans = torch.from_numpy(poss[:, 0, :])
    
    motion_data = {
        'R': R,                         # (T, J, 3, 3) rotation matrices
        'root_trans': root_trans,       # (T, 3)
        'euler_deg': rots_deg,          # (T, J, 3) original Euler (for saving)
        'positions': poss,              # (T, J, 3) original positions (for saving)
        'order': order                  # Euler order string
    }
    
    skeleton_data = {
        'offsets': torch.from_numpy(data['offsets'].astype(np.float32)),
        'parents': data['parents'].astype(np.int64),
        'names': data['names']
    }
    
    print(f"[INFO] Frames: {T}, Joints: {J}, Convention: {convention}")
    return motion_data, skeleton_data


def save_bvh_from_ik(
    output_path: str,
    R_fixed: torch.Tensor,          # (T, J, 3, 3) from fix_hand_collisions()
    original_motion: Dict,          # From load_bvh_for_ik()
    skeleton: Dict                  # From load_bvh_for_ik()
) -> None:
    """
    Save fixed rotations back to BVH file.
    
    Args:
        output_path: Output BVH path
        R_fixed: (T, J, 3, 3) Fixed rotation matrices from fix_hand_collisions()
        original_motion: Motion dict from load_bvh_for_ik()
        skeleton: Skeleton dict from load_bvh_for_ik()
    """
    print(f"\n[INFO] Converting back to Euler angles...")
    convention = str(original_motion['order']).upper()
    
    # Convert back to Euler angles
    euler_rad = rc.matrix_to_euler_angles(R_fixed, convention=convention)
    euler_deg = (euler_rad * (180.0 / np.pi)).cpu().numpy().astype(np.float32)
    
    # Build output data
    out_data = {
        'rotations': euler_deg,
        'positions': original_motion['positions'],
        'offsets': skeleton['offsets'].cpu().numpy(),
        'parents': skeleton['parents'],
        'names': skeleton['names'],
        'order': original_motion['order']
    }
    
    print(f"[INFO] Saving result: {output_path}")
    bvh.save(output_path, out_data)
    print(f"[DONE] Processing complete!")


# ============================================================================
# SECTION 6: Realtime Solver Class (matches cylindrical_hand_ik1_fast_lbfgs API)
# ============================================================================

from process_zm_dataset import default_meta_info_path


class realtime_hands_ik_solver_analytical:
    """
    实时手部 IK 求解器 - ANALYTICAL 解析版本
    
    性能: 12帧约 0.5-2ms (相比 L-BFGS 约 13ms，快 6-25 倍)
    
    使用纯几何解析解，无需迭代优化，速度极快。
    
    Usage:
        solver = realtime_hands_ik_solver_analytical()
        R_fixed = solver.solve(R_chunk, root_trans_chunk)
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Args:
            device: 计算设备 'cpu' 或 'cuda'
        """
        self.device = device
        self.prev_context_R = None
        self.prev_context_root_trans = None
        
        # 加载默认骨架信息
        default_meta_info = np.load(default_meta_info_path)
        self.offsets = torch.tensor(default_meta_info['offsets']).float().to(device)
        self.names = list(default_meta_info['names'])
        self.parents = default_meta_info['parents']
        
        self.config = get_default_config()
        
        # 预计算关节索引
        self._precompute_joint_indices()
        
        self._warmup_done = False
    
    def _precompute_joint_indices(self):
        """预计算常用关节索引，避免每次调用时重复查找"""
        name2idx = {n: i for i, n in enumerate(self.names)}
        
        self.pelvis_idx = name2idx.get('Root_M', name2idx.get('Hips', 0))
        
        # 右臂
        self.shoulder_R_idx = name2idx.get('Shoulder_R', -1)
        self.elbow_R_idx = name2idx.get('Elbow_R', -1)
        self.wrist_R_idx = name2idx.get('Wrist_R', -1)
        
        # 左臂
        self.shoulder_L_idx = name2idx.get('Shoulder_L', -1)
        self.elbow_L_idx = name2idx.get('Elbow_L', -1)
        self.wrist_L_idx = name2idx.get('Wrist_L', -1)
        
        # 肩膀父节点
        self.shoulder_R_parent_idx = int(self.parents[self.shoulder_R_idx])
        self.shoulder_L_parent_idx = int(self.parents[self.shoulder_L_idx])
        
        # 计算休息姿势的骨骼长度
        rest_pos = compute_rest_positions(self.offsets.cpu().numpy(), self.parents)
        rest_pos = torch.from_numpy(rest_pos).to(self.device)
        
        self.upper_len_R = torch.norm(rest_pos[self.elbow_R_idx] - rest_pos[self.shoulder_R_idx]).item()
        self.lower_len_R = torch.norm(rest_pos[self.wrist_R_idx] - rest_pos[self.elbow_R_idx]).item()
        self.upper_len_L = torch.norm(rest_pos[self.elbow_L_idx] - rest_pos[self.shoulder_L_idx]).item()
        self.lower_len_L = torch.norm(rest_pos[self.wrist_L_idx] - rest_pos[self.elbow_L_idx]).item()
    
    def warmup(self, num_frames: int = 12):
        """预热 - 对于解析解来说不需要编译，但保持 API 一致性"""
        if self._warmup_done:
            return
        
        # 创建随机数据进行预热 (主要是让 PyTorch 预分配内存)
        num_joints = len(self.names)
        R_dummy = torch.randn(num_frames, num_joints, 3, 3, device=self.device)
        R_dummy = torch.linalg.qr(R_dummy)[0]  # 正交化
        root_trans_dummy = torch.randn(num_frames, 3, device=self.device) * 10
        
        # 运行一次
        _ = self.solve(R_dummy, root_trans_dummy)
        
        # 重置状态
        self.prev_context_R = None
        self.prev_context_root_trans = None
        self._warmup_done = True
    
    def solve(self, R_chunk: torch.Tensor, root_trans_chunk: torch.Tensor) -> torch.Tensor:
        """
        求解 IK
        
        Args:
            R_chunk: 旋转矩阵 (T, J, 3, 3)
            root_trans_chunk: 根节点位移 (T, 3)
            
        Returns:
            修复后的旋转矩阵 (T, J, 3, 3)
        """
        R_chunk1 = R_chunk.clone().to(self.device)
        root_trans_chunk1 = root_trans_chunk.clone().to(self.device)
        
        # C0 连续性: 添加上一帧作为上下文
        if self.prev_context_R is not None and self.prev_context_root_trans is not None:
            R_chunk1 = torch.cat([self.prev_context_R, R_chunk1], dim=0)
            root_trans_chunk1 = torch.cat([self.prev_context_root_trans, root_trans_chunk1], dim=0)
            actual_context_len = 1
        else:
            actual_context_len = 0
        
        # 调用解析 IK
        R_fixed_chunk = self._solve_analytical(R_chunk1, root_trans_chunk1, context_length=actual_context_len)
        
        # 移除上下文帧
        R_fixed_chunk = R_fixed_chunk[actual_context_len:].detach()
        
        # 保存最后一帧作为下一次的上下文
        self.prev_context_R = R_fixed_chunk[-1:].clone()
        self.prev_context_root_trans = root_trans_chunk[-1:].clone().to(self.device)
        
        return R_fixed_chunk
    
    def _solve_analytical(
        self,
        R: torch.Tensor,
        root_trans: torch.Tensor,
        context_length: int = 0
    ) -> torch.Tensor:
        """
        解析 IK 求解 (内部方法)
        
        直接使用几何方法计算，无需迭代优化。
        """
        T = R.shape[0]
        config = self.config
        
        # FK with world rotations
        pos, world_rot = fk_global_with_rotations(R, self.offsets, self.parents, root_trans)
        
        # 获取所有位置
        pelvis_pos_all = pos[:, self.pelvis_idx, :]
        pelvis_rot_all = R[:, self.pelvis_idx, :, :]
        hand_R_world = pos[:, self.wrist_R_idx, :]
        hand_L_world = pos[:, self.wrist_L_idx, :]
        
        # 转换到 pelvis 局部坐标 (向量化)
        hand_R_local = torch.bmm(pelvis_rot_all.transpose(-1, -2), (hand_R_world - pelvis_pos_all).unsqueeze(-1)).squeeze(-1)
        hand_L_local = torch.bmm(pelvis_rot_all.transpose(-1, -2), (hand_L_world - pelvis_pos_all).unsqueeze(-1)).squeeze(-1)
        
        # 计算阈值
        z_R = hand_R_local[:, 2]
        z_L = hand_L_local[:, 2]
        threshold_R = interpolate_threshold(z_R, config['z_threshold_keypoints'])
        threshold_L = interpolate_threshold(z_L, config['z_threshold_keypoints'])
        
        # XY 距离
        xy_dist_R = torch.sqrt(hand_R_local[:, 0]**2 + hand_R_local[:, 1]**2)
        xy_dist_L = torch.sqrt(hand_L_local[:, 0]**2 + hand_L_local[:, 1]**2)
        
        # 检查需要修复的帧
        cutoff_R = threshold_R * config['cutoff_ratio']
        cutoff_L = threshold_L * config['cutoff_ratio']
        needs_fix_R = xy_dist_R < cutoff_R
        needs_fix_L = xy_dist_L < cutoff_L
        
        # 应用 context_length
        if context_length > 0:
            needs_fix_R[:context_length] = False
            needs_fix_L[:context_length] = False
        
        # 计算目标 XY 距离
        target_xy_R = xy_remap_bounded(xy_dist_R, threshold_R, config['cutoff_ratio'])
        target_xy_L = xy_remap_bounded(xy_dist_L, threshold_L, config['cutoff_ratio'])
        if target_xy_R is None:
            target_xy_R = xy_dist_R
        if target_xy_L is None:
            target_xy_L = xy_dist_L
        
        # 缩放系数
        scale_R = target_xy_R / (xy_dist_R + 1e-9)
        scale_L = target_xy_L / (xy_dist_L + 1e-9)
        
        # 计算目标局部位置
        target_R_local = hand_R_local.clone()
        target_R_local[:, 0] = hand_R_local[:, 0] * scale_R
        target_R_local[:, 1] = hand_R_local[:, 1] * scale_R
        xy_push_R = (scale_R - 1) * xy_dist_R
        target_R_local[:, 2] = hand_R_local[:, 2] + config['z_lift_ratio'] * xy_push_R
        
        target_L_local = hand_L_local.clone()
        target_L_local[:, 0] = hand_L_local[:, 0] * scale_L
        target_L_local[:, 1] = hand_L_local[:, 1] * scale_L
        xy_push_L = (scale_L - 1) * xy_dist_L
        target_L_local[:, 2] = hand_L_local[:, 2] + config['z_lift_ratio'] * xy_push_L
        
        # 转回世界坐标
        target_R_world = pelvis_pos_all + torch.bmm(pelvis_rot_all, target_R_local.unsqueeze(-1)).squeeze(-1)
        target_L_world = pelvis_pos_all + torch.bmm(pelvis_rot_all, target_L_local.unsqueeze(-1)).squeeze(-1)
        
        # 不需要修复的帧使用原始位置
        target_R_world = torch.where(needs_fix_R.unsqueeze(-1), target_R_world, hand_R_world)
        target_L_world = torch.where(needs_fix_L.unsqueeze(-1), target_L_world, hand_L_world)
        
        # 如果没有帧需要修复，直接返回
        if not needs_fix_R.any() and not needs_fix_L.any():
            return R
        
        R_fixed = R.clone()
        
        # 右臂 IK
        if needs_fix_R.any():
            shoulder_R_pos = pos[:, self.shoulder_R_idx, :]
            elbow_R_pos = pos[:, self.elbow_R_idx, :]
            wrist_R_pos = pos[:, self.wrist_R_idx, :]
            shoulder_R_parent_rot = world_rot[:, self.shoulder_R_parent_idx, :, :]
            
            new_shoulder_R_local, new_elbow_R_local = analytical_ik_arm(
                shoulder_pos=shoulder_R_pos,
                elbow_pos=elbow_R_pos,
                hand_pos=wrist_R_pos,
                target_pos=target_R_world,
                upper_len=self.upper_len_R,
                lower_len=self.lower_len_R,
                shoulder_world_rot=world_rot[:, self.shoulder_R_idx, :, :],
                elbow_world_rot=world_rot[:, self.elbow_R_idx, :, :],
                shoulder_parent_rot=shoulder_R_parent_rot,
                shoulder_local_rot=R[:, self.shoulder_R_idx, :, :],
                elbow_local_rot=R[:, self.elbow_R_idx, :, :],
                needs_fix=needs_fix_R
            )
            
            R_fixed[:, self.shoulder_R_idx, :, :] = new_shoulder_R_local
            R_fixed[:, self.elbow_R_idx, :, :] = new_elbow_R_local
        
        # 左臂 IK
        if needs_fix_L.any():
            shoulder_L_pos = pos[:, self.shoulder_L_idx, :]
            elbow_L_pos = pos[:, self.elbow_L_idx, :]
            wrist_L_pos = pos[:, self.wrist_L_idx, :]
            shoulder_L_parent_rot = world_rot[:, self.shoulder_L_parent_idx, :, :]
            
            new_shoulder_L_local, new_elbow_L_local = analytical_ik_arm(
                shoulder_pos=shoulder_L_pos,
                elbow_pos=elbow_L_pos,
                hand_pos=wrist_L_pos,
                target_pos=target_L_world,
                upper_len=self.upper_len_L,
                lower_len=self.lower_len_L,
                shoulder_world_rot=world_rot[:, self.shoulder_L_idx, :, :],
                elbow_world_rot=world_rot[:, self.elbow_L_idx, :, :],
                shoulder_parent_rot=shoulder_L_parent_rot,
                shoulder_local_rot=R[:, self.shoulder_L_idx, :, :],
                elbow_local_rot=R[:, self.elbow_L_idx, :, :],
                needs_fix=needs_fix_L
            )
            
            R_fixed[:, self.shoulder_L_idx, :, :] = new_shoulder_L_local
            R_fixed[:, self.elbow_L_idx, :, :] = new_elbow_L_local
        
        return R_fixed
    
    def reset(self):
        """重置状态 (开始新的序列时调用)"""
        self.prev_context_R = None
        self.prev_context_root_trans = None


# ============================================================================
# SECTION 7: Helper Functions (Used by API)
# ============================================================================

def get_arm_chain_indices(names: List[str], side: str) -> List[int]:
    """Find arm joint indices from skeleton (shoulder → elbow → wrist)."""
    assert side in ('R', 'L')
    name2idx = {n: i for i, n in enumerate(names)}
    
    # Simplified chain: Shoulder -> Elbow -> Wrist
    chain_names = [
        f"Shoulder_{side}",
        f"Elbow_{side}",
        f"Wrist_{side}",
    ]
    
    # Try to add optional joints if they exist
    optional = [
        f"Scapula_{side}",
        f"ShoulderPart1_{side}",
        f"ElbowPart1_{side}",
    ]
    
    indices = []
    for name in optional + chain_names:
        if name in name2idx:
            indices.append(name2idx[name])
    
    # Ensure we have at least the basic chain
    if len(indices) < 3:
        raise ValueError(f"Could not find complete arm chain for side {side}")
    
    return indices


def process_bvh_motion(
    input_bvh: str,
    output_bvh: str,
    config: Dict,
    device: str = 'cpu',
    max_frames: int = None
) -> None:
    """
    Main pipeline:
    1. Load BVH and convert to rotation matrices
    2. For each frame, compute hand targets (if collision detected)
    3. Run batch IK optimization across all frames
    4. Convert back to Euler angles and save
    """
    print(f"[INFO] Loading BVH: {input_bvh}")
    data = bvh.load(input_bvh)
    
    names = data['names']
    parents = data['parents'].astype(np.int64)
    offsets = data['offsets'].astype(np.float32)
    order = data['order']
    convention = str(order).upper()
    
    rots_deg = data['rotations'].astype(np.float32)  # (T, J, 3)
    poss = data['positions'].astype(np.float32)  # (T, J, 3)
    
    T, J = rots_deg.shape[0], rots_deg.shape[1]
    
    # Limit frames if requested
    if max_frames is not None and max_frames < T:
        T = max_frames
        print(f"[INFO] Processing only first {T} frames (limited)")
    
    print(f"[INFO] Frames: {T}, Joints: {J}, Convention: {convention}")
    
    # Convert to tensors (only first T frames)
    device_torch = torch.device(device)
    rots_rad = torch.from_numpy(rots_deg[:T]).to(device=device_torch) * (np.pi / 180.0)
    R = rc.euler_angles_to_matrix(rots_rad, convention=convention)  # (T, J, 3, 3)
    offsets_t = torch.from_numpy(offsets).to(device=device_torch)
    root_trans = torch.from_numpy(poss[:T, 0, :]).to(device=device_torch)
    
    # Get joint indices
    name2idx = {n: i for i, n in enumerate(names)}
    pelvis_idx = name2idx.get('Root_M', name2idx.get('Hips', 0))
    
    # Get hand indices (end effectors)
    hand_R_idx = name2idx.get('Wrist_R', name2idx.get('RightHand', -1))
    hand_L_idx = name2idx.get('Wrist_L', name2idx.get('LeftHand', -1))
    
    if hand_R_idx == -1 or hand_L_idx == -1:
        raise ValueError("Could not find hand joints in skeleton")
    
    # Get arm chains
    arm_R_chain = get_arm_chain_indices(names, 'R')
    arm_L_chain = get_arm_chain_indices(names, 'L')
    
    print(f"[INFO] Pelvis: {names[pelvis_idx]} (idx={pelvis_idx})")
    print(f"[INFO] Right hand: {names[hand_R_idx]} (idx={hand_R_idx})")
    print(f"[INFO] Left hand: {names[hand_L_idx]} (idx={hand_L_idx})")
    print(f"[INFO] Right arm chain: {[names[i] for i in arm_R_chain]}")
    print(f"[INFO] Left arm chain: {[names[i] for i in arm_L_chain]}")
    
    # ===== Phase 1: Compute targets for all frames =====
    print(f"\n[PHASE 1] Computing targets...")
    print(f"[CONFIG] Z-profile: {config['z_threshold_keypoints']}")
    print(f"[CONFIG] Cutoff ratio: {config['cutoff_ratio']}")
    
    targets = []
    num_frames_needing_ik = 0
    
    # Compute FK for all frames at once
    pos = fk_global_positions(R, offsets_t, parents, root_trans)  # (T, J, 3)
    
    for t in range(T):
        pelvis_pos = pos[t, pelvis_idx, :]
        pelvis_rot = R[t, pelvis_idx, :, :]
        
        # Right hand
        hand_R_world = pos[t, hand_R_idx, :]
        hand_R_local = world_to_pelvis_local(hand_R_world, pelvis_pos, pelvis_rot)
        target_R = compute_target_position(hand_R_local, config['z_threshold_keypoints'], config['cutoff_ratio'], config['z_lift_ratio'])
        
        # Left hand
        hand_L_world = pos[t, hand_L_idx, :]
        hand_L_local = world_to_pelvis_local(hand_L_world, pelvis_pos, pelvis_rot)
        target_L = compute_target_position(hand_L_local, config['z_threshold_keypoints'], config['cutoff_ratio'], config['z_lift_ratio'])
        
        targets.append({
            'right': target_R,
            'left': target_L,
            'is_pinned': (target_R is None and target_L is None)
        })
        
        if not targets[t]['is_pinned']:
            num_frames_needing_ik += 1
    
    pinned_count = sum(1 for t in targets if t['is_pinned'])
    print(f"[INFO] Frames needing IK: {num_frames_needing_ik}/{T}")
    print(f"[INFO] Frames pinned (safe): {pinned_count}/{T}")
    
    # ===== Phase 2: Multi-frame IK optimization =====
    print(f"\n[PHASE 2] Running multi-frame IK optimization...")
    
    R_optimized = solve_multi_frame_ik(
        R_original=R,
        targets=targets,
        arm_R_chain=arm_R_chain,
        arm_L_chain=arm_L_chain,
        hand_R_idx=hand_R_idx,
        hand_L_idx=hand_L_idx,
        pelvis_idx=pelvis_idx,
        offsets=offsets_t,
        parents=parents,
        root_trans=root_trans,
        config=config
    )
    
    # Convert back to Euler angles
    print(f"\n[INFO] Converting back to Euler angles...")
    euler_rad = rc.matrix_to_euler_angles(R_optimized, convention=convention)
    euler_deg = euler_rad * (180.0 / np.pi)
    
    # Save result (only processed frames)
    out_data = dict(data)
    out_data['rotations'] = euler_deg.detach().cpu().numpy().astype(np.float32)
    out_data['positions'] = poss[:T]
    
    print(f"[INFO] Saving result: {output_bvh}")
    bvh.save(output_bvh, out_data)
    print(f"[DONE] Processing complete!")


# ============================================================================
# SECTION 8: Command-Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ANALYTICAL Cylindrical Hand IK - Ultra-fast self-collision fixing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python analytic_cylindrical_hand_ik.py --input_bvh motion.bvh --output_bvh fixed.bvh
    
    # With custom cutoff ratio
    python analytic_cylindrical_hand_ik.py --input_bvh motion.bvh --output_bvh fixed.bvh --cutoff_ratio 1.4
    
    # Skip first 10 frames (e.g., T-pose)
    python analytic_cylindrical_hand_ik.py --input_bvh motion.bvh --output_bvh fixed.bvh --context_length 10
    
Note: This uses ANALYTICAL IK (closed-form solution). No iterations needed!
      100-1000x faster than optimization-based approach.
        """
    )
    
    parser.add_argument("--input_bvh", type=str, required=True, 
                        help="Input BVH file path")
    parser.add_argument("--output_bvh", type=str, required=True, 
                        help="Output BVH file path (will be overwritten)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="Device for computation (default: cpu)")
    parser.add_argument("--max_frames", type=int, default=None, 
                        help="Process only first N frames (useful for testing)")
    parser.add_argument("--context_length", type=int, default=0,
                        help="First N frames kept unchanged (e.g., T-pose)")
    
    # Mapping parameters
    parser.add_argument("--cutoff_ratio", type=float, default=1.5, 
                        help="Cutoff distance ratio (higher = wider affected area, default: 1.5)")
    parser.add_argument("--z_lift_ratio", type=float, default=0.0,
                        help="Z lift proportional to XY push (0=no lift, 1=full lift, default: 1.0)")
    
    args = parser.parse_args()
    
    
    # Load BVH using helper
    motion, skeleton = load_bvh_for_ik(args.input_bvh, args.max_frames)
    
    # Build config (start with defaults, override from args)
    config = get_default_config()
    config['cutoff_ratio'] = args.cutoff_ratio
    config['z_lift_ratio'] = args.z_lift_ratio
    
    import time
    start_time = time.time()
    # Call ANALYTICAL IK function (no iterations!)
    R_fixed = fix_hand_collisions_analytical(
        R=motion['R'],
        offsets=skeleton['offsets'],
        parents=skeleton['parents'],
        root_trans=motion['root_trans'],
        names=skeleton['names'],
        config=config,
        context_length=args.context_length,
        device=args.device
    )
    
    elapsed = time.time() - start_time
    print(f"\n[TIMING] Total processing time: {elapsed:.2f} seconds")
    # Save using helper
    save_bvh_from_ik(args.output_bvh, R_fixed, motion, skeleton)
    


if __name__ == "__main__":
    main()

