import torch
import math # For math.pi

# Helper to get device and dtype from a tensor, or default
def _get_tensor_props(*args):
    device = None
    dtype = torch.float32
    for arg in args:
        if isinstance(arg, torch.Tensor):
            device = arg.device
            dtype = arg.dtype
            break
    if device is None: # Default if no tensors provided
        device = torch.device('cpu')
    return device, dtype

def eye(shape=[], device=None, dtype=torch.float32):
    if device is None:
        device = torch.device('cpu') # Default device
    
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    if not shape: # Empty list check
        return identity_quat
    else:
        # Original: np.array([1,0,0,0]) * np.ones(np.concatenate([shape, [4]]))
        # Create ones with the target shape and then broadcast-multiply
        ones_shape = list(shape) + [4]
        return identity_quat * torch.ones(ones_shape, device=device, dtype=dtype)

def eye_like(x):
    # Creates an identity quaternion tensor with the same batch shape as x,
    # and last dimension 4.
    # Original: np.array([1,0,0,0]) * np.ones_like(x[..., np.newaxis].repeat(4, axis=-1))
    # This was a bit convoluted way to get the shape. A more direct PyTorch way:
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=x.device, dtype=x.dtype)
    if x.shape[-1] == 4: # If x is already a quaternion or batch of quaternions
        target_shape = x.shape
    else: # If x is, for example, just a batch shape descriptor
        target_shape = list(x.shape) + [4] # this might be an issue if x already has data

    # A safer way to mimic the original's shape intent:
    # The original seemed to imply x's last dim was not 4, and it was making it 4
    # For example, if x had shape (N, M), result would be (N, M, 4)
    # If x had shape (N, M, K), result would be (N, M, K, 4)
    # This means the intent of `x[..., np.newaxis].repeat(4, axis=-1)` was to create a shape like (*x.shape, 4)
    # But `np.ones_like` would then take this shape and make an array of ones.
    # Let's assume x is a tensor whose leading dimensions we want to copy, and the last dim should be 4
    # e.g. x might be x[..., 0] from a quaternion
    
    # If x's last dimension is already feature-like (e.g. not 4), the original intent
    # `x[..., np.newaxis].repeat(4, axis=-1)` would make shape e.g. (B, L, D, 4) from (B,L,D)
    # and then `ones_like` creates ones of that shape.
    # The current way:
    shape_prefix = list(x.shape[:-1]) # all dims except the last one of x
    # This means x is assumed to be like (*some_dims, K) and we want (*some_dims, 4)
    
    # A simpler interpretation for eye_like(quaternion_tensor):
    # result = torch.zeros_like(x)
    # result[..., 0] = 1.0
    # return result
    
    # Let's try to match the original `np.ones_like(x[..., np.newaxis].repeat(4, axis=-1))` more closely
    # This implies x itself is not a quaternion, but something whose shape is used as a base.
    # if x has shape (..., D), then x.unsqueeze(-1).repeat_interleave(4, dim=-1) has shape (..., D, 4)
    # then ones_like this.
    if x.dim() > 0 and x.shape[-1] == 4: # If x is already a quaternion array
        _ones = torch.ones_like(x)
    else: # If x is something else, e.g. x[..., 0] (scalar part)
        _ones = torch.ones( (*x.shape, 4) , device=x.device, dtype=x.dtype)

    return identity_quat * _ones


def mul(x, y):
    # Ensure x and y are on the same device and dtype
    common_device, common_dtype = _get_tensor_props(x, y)
    x = x.to(common_device, common_dtype)
    y = y.to(common_device, common_dtype)

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return torch.cat([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], dim=-1)


def _fast_cross(a, b):
    # Assuming a and b are 3-element vectors or batches of them
    # PyTorch has torch.linalg.cross or torch.cross
    # For compatibility with broadcasting of original, let's ensure shapes allow it.
    # The original np.broadcast might be slightly more general than torch.cross for non-standard shapes.
    # However, torch.cross (new name for torch.linalg.cross in recent PyTorch) handles broadcasting.
    # If a and b already have the last dim as 3
    if a.shape[-1] == 3 and b.shape[-1] == 3:
        return torch.cross(a, b, dim=-1)
    
    # Manual implementation if torch.cross is not suitable or for exact replication
    # This assumes a and b are broadcastable to a common shape ending in 3
    target_shape = torch.broadcast_shapes(a.shape, b.shape)
    o = torch.empty(target_shape, device=a.device, dtype=a.dtype)
    o[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    o[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    o[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return o


def mul_vec(q, v): # Renamed x to q, y to v for clarity (quaternion, vector)
    # Ensure q and v are on the same device and dtype
    common_device, common_dtype = _get_tensor_props(q, v)
    q = q.to(common_device, common_dtype)
    v = v.to(common_device, common_dtype)
    
    q_vec = q[..., 1:]
    q_scalar = q[..., 0].unsqueeze(-1) # Keep dim

    t = 2.0 * _fast_cross(q_vec, v)
    return v + q_scalar * t + _fast_cross(q_vec, t)


def mul_scalar(x, y_scalar): # y is a scalar factor here
    # slerp(eye_like(x component), x, scalar_factor)
    # x[..., 0] is used to get the shape/device/dtype for eye_like
    return slerp(eye_like(x[..., 0]), x, y_scalar)


def inv(x):
    # Ensure multiplier is on the same device and dtype as x
    multiplier = torch.tensor([1.0, -1.0, -1.0, -1.0], device=x.device, dtype=x.dtype)
    return multiplier * x


def abs_quat(x): # Renamed to avoid conflict with torch.abs
    # This is "canonical" quaternion, not element-wise abs
    scalar_part_sign = torch.sum(x * torch.tensor([1.0, 0.0, 0.0, 0.0], device=x.device, dtype=x.dtype), dim=-1)
    # condition must be broadcastable with x
    condition = (scalar_part_sign > 0.0).unsqueeze(-1)
    return torch.where(condition, x, -x)


def log(x, eps=1e-5):
    vec_part = x[..., 1:]
    scalar_part = x[..., 0:1] # Keep dim

    length_sq = torch.sum(torch.square(vec_part), dim=-1, keepdim=True)
    length = torch.sqrt(length_sq)
    
    # atan2(length, scalar_part) gives half angle.
    # If length is small, atan2(length, scalar_part) / length can be unstable.
    # For small length, quaternion is close to [1,0,0,0] or [-1,0,0,0].
    # If q = [cos(a/2), sin(a/2)*v], log(q) = a/2 * v.
    # angle_div_length = angle / length.
    # If length is small, angle is small. sin(a/2) ~ a/2. So length ~ a/2.
    # angle_div_length ~ (a/2) / (a/2) = 1.
    # The original `np.ones_like(length)` for `length < eps` implies this approximation.
    
    # halfangle_norm = atan2(length, scalar_part)
    # We need halfangle_norm / length
    # Use where to select: if length is small, use 1.0, else use atan2(length, scalar_part) / length
    # This seems to be the intent of the original `np.where(length < eps, np.ones_like(length), ...)`
    # However, it should be safe to compute atan2 always. The division by length is the issue.
    
    # Let's follow original logic path:
    # halfangle = np.where(length < eps, np.ones_like(length), np.arctan2(length, x[..., 0:1]) / length)
    # This `np.ones_like(length)` as the value when `length < eps` is a bit suspicious for `halfangle` directly.
    # It seems to be `factor = angle / length`.
    # log(q) = (angle / (2 * ||v||)) * v  (where v is vector part, ||v|| is length)
    # angle = 2 * atan2(||v||, w)
    # factor = atan2(||v||, w) / ||v||
    
    # Let's re-evaluate log_q = theta * v_unit, where theta is half angle `acos(w)`
    # or theta = atan2(norm_v, w).
    # v = q_v / norm_v
    # log_q = (atan2(norm_v, w) / norm_v) * q_v
    
    factor = torch.where(
        length < eps,
        torch.ones_like(length), # Approximation for small length (angle/length -> 1 if w is also ~1)
        torch.atan2(length, scalar_part) / (length + eps) # added eps to denominator for safety
    )
    return factor * vec_part


def exp(v_log, eps=1e-5): # input is the log of a quaternion (a 3-vector)
    halfangle_sq = torch.sum(torch.square(v_log), dim=-1, keepdim=True)
    halfangle = torch.sqrt(halfangle_sq)

    c = torch.cos(halfangle)
    # sinc(x) = sin(pi*x)/(pi*x). Here, np.sinc(halfangle / np.pi) = sin(halfangle)/halfangle
    # Handle halfangle == 0 case for s: limit is 1
    s_factor = torch.where(
        halfangle < eps,
        torch.ones_like(halfangle),
        torch.sin(halfangle) / (halfangle + eps) # Added eps to avoid division by zero if halfangle somehow becomes 0 despite check
    )
    
    return torch.cat([c, s_factor * v_log], dim=-1)


def to_helical(x, eps=1e-5):
    # Helical coordinates are often 2 * log(q)
    return 2.0 * log(x, eps)


def from_helical(h, eps=1e-5):
    # h = 2 * log(q) => log(q) = h / 2
    # q = exp(h / 2)
    return exp(h / 2.0, eps)


def to_angle_axis(x, eps=1e-10):
    vec_part = x[..., 1:]
    scalar_part = x[..., 0] # No keepdim needed for atan2's second arg if first is already shaped

    length_sq = torch.sum(torch.square(vec_part), dim=-1)
    length = torch.sqrt(length_sq)
    
    angle = 2.0 * torch.atan2(length, scalar_part)
    
    # Normalize axis part, add eps for stability
    axis = vec_part / (length.unsqueeze(-1) + eps)
    # Handle zero-rotation case (length is zero) - axis can be arbitrary (e.g. [1,0,0])
    # Current form gives NaN if length is 0.
    # If length is near zero, angle is near zero. Axis doesn't matter much.
    # We can set axis to a default like [1,0,0] if length is zero.
    is_zero_length = (length < eps).unsqueeze(-1)
    default_axis = torch.zeros_like(vec_part)
    default_axis[..., 0] = 1.0 # e.g. [1,0,0]
    
    safe_axis = torch.where(is_zero_length, default_axis, axis)
    return angle, safe_axis


def from_angle_axis(angle, axis):
    # angle shape: (...)
    # axis shape: (..., 3)
    # Ensure device and dtype are consistent
    common_device, common_dtype = _get_tensor_props(angle, axis)
    angle = angle.to(common_device, common_dtype)
    axis = axis.to(common_device, common_dtype)

    angle_half = (angle / 2.0).unsqueeze(-1) # Make it broadcastable with axis
    c = torch.cos(angle_half)
    s = torch.sin(angle_half)
    return torch.cat([c, s * axis], dim=-1)


def diff(x, y, world=True):
    # Ensure device and dtype
    common_device, common_dtype = _get_tensor_props(x, y)
    x = x.to(common_device, common_dtype)
    y = y.to(common_device, common_dtype)

    # Ensure quaternions are in the same hemisphere for stable diff
    dot_product = torch.sum(x * y, dim=-1, keepdim=True)
    # Use abs_quat's logic: if dot_product < 0, flip one of them.
    # Flipped x if necessary to be "closer" to y
    # x_prime = torch.where(dot_product < 0.0, -x, x)
    # The original code flips x if sum(x*y) > 0. This is slightly different.
    # diff = np.sum(x * y, axis=-1)[..., np.newaxis]
    # flip = np.where(diff > 0.0, x, -x)
    # The intent seems to be to ensure the scalar part of (x * inv(y)) or (inv(y) * x) is positive.
    # Let's use a simpler "closest" logic
    # If x and y are far apart (dot_product < 0), -x is closer to y than x is.
    # So we want to operate on (x, y) or (-x, y).
    # Let's use x_prime = abs_quat(x) and y_prime = abs_quat(y) and then check relative orientation.
    # The original logic:
    #   `diff = np.sum(x * y, axis=-1)[..., np.newaxis]`
    #   `flip = np.where(diff > 0.0, x, -x)`
    # This means if x and y are in roughly the same direction (dot > 0), use x.
    # If they are in opposite directions (dot < 0), use -x.
    # This ensures that `flip` and `y` are in the same hemisphere.
    
    current_dot = torch.sum(x * y, dim=-1, keepdim=True)
    x_adjusted = torch.where(current_dot >= 0.0, x, -x) # Ensure x_adjusted and y are in the same hemisphere

    y_inv = inv(y)
    if world:
        # diff_quat = x_adjusted * y_inv
        return mul(x_adjusted, y_inv)
    else:
        # diff_quat = y_inv * x_adjusted
        return mul(y_inv, x_adjusted)


def normalize(x, eps=1e-8): # Changed default eps, 0.0 can be problematic
    # norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True))
    norm = torch.linalg.norm(x, dim=-1, keepdim=True)
    return x / (norm + eps)


def between(v1, v2): # For vectors v1, v2
    # Ensure device and dtype
    common_device, common_dtype = _get_tensor_props(v1, v2)
    v1 = v1.to(common_device, common_dtype)
    v2 = v2.to(common_device, common_dtype)

    # This computes a quaternion q such that q * v1 = v2 (scaled).
    # Typically used for normalized v1, v2.
    # Formula: q_w = sqrt(||v1||^2 * ||v2||^2) + dot(v1,v2)
    #          q_xyz = cross(v1,v2)
    # Then normalize q.
    
    v1_norm_sq = torch.sum(v1 * v1, dim=-1, keepdim=True)
    v2_norm_sq = torch.sum(v2 * v2, dim=-1, keepdim=True)
    dot_prod = torch.sum(v1 * v2, dim=-1, keepdim=True)
    
    scalar_part = torch.sqrt(v1_norm_sq * v2_norm_sq) + dot_prod
    vector_part = _fast_cross(v1, v2)
    
    # Concatenate and then normalize
    q_unnormalized = torch.cat([scalar_part, vector_part], dim=-1)
    return normalize(q_unnormalized)


def slerp(q0, q1, t, eps=1e-10): # t is interpolation factor (scalar or tensor)
    # Ensure device and dtype
    common_device, common_dtype = _get_tensor_props(q0, q1, t if isinstance(t, torch.Tensor) else None)
    q0 = q0.to(common_device, common_dtype)
    q1 = q1.to(common_device, common_dtype)
    if isinstance(t, (float, int)):
        t = torch.tensor(t, device=common_device, dtype=common_dtype)
    else:
        t = t.to(common_device, common_dtype)

    # Ensure q1 is in the same hemisphere as q0
    dot = torch.sum(q0 * q1, dim=-1)
    q1_adjusted = torch.where(dot.unsqueeze(-1) < 0.0, -q1, q1)
    dot = torch.abs(dot) # Recalculate dot with adjusted q1, or just use abs of original

    # Angle between q0 and q1_adjusted
    # Clip dot to prevent acos domain errors due to numerical precision
    omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
    
    # sin_omega can be zero if omega is 0 or pi.
    # If omega is very small, use linear interpolation.
    sin_omega = torch.sin(omega)

    # Original factors:
    # a0 = np.sin((1.0 - a) * o) / (np.sin(o) + eps)
    # a1 = np.sin((a) * o) / (np.sin(o) + eps)
    # This is sensitive if sin(o) is near zero.

    # Reshape t and omega for broadcasting if necessary
    if t.dim() < omega.dim():
        t = t.unsqueeze(-1) # Make t broadcastable with omega, q0, q1

    # Handle linear interpolation case (omega is small)
    # Using a threshold for dot product is often more robust than checking sin_omega directly.
    # DOT_THRESHOLD = 0.9995 (or similar)
    # linear_interp_mask = dot > DOT_THRESHOLD
    
    # Original approach with eps:
    denom = sin_omega.unsqueeze(-1) + eps
    
    t0 = (1.0 - t) * omega.unsqueeze(-1)
    t1 = t * omega.unsqueeze(-1)
    
    s0 = torch.sin(t0) / denom
    s1 = torch.sin(t1) / denom
    
    # Fallback to LERP for colinearity
    # if omega is very small (dot very close to 1), sin_omega is very small.
    # then sin((1-a)o)/sin(o) -> 1-a and sin(ao)/sin(o) -> a
    # This will be handled if denom is small and numerators are also small by L'Hopital's.
    # But direct check is safer.
    
    # Simplified: if sin_omega is small, result is (1-t)*q0 + t*q1_adjusted (normalized)
    # The current formulation should approach this limit.
    # For robustness, explicitly handle near-zero omega:
    near_zero_omega = (omega < eps).unsqueeze(-1) # omega is per-pair, unsqueeze for broadcasting with quat components

    # LERP coefficients
    s0_lerp = 1.0 - t
    s1_lerp = t
    
    # Choose SLERP or LERP coefficients
    final_s0 = torch.where(near_zero_omega, s0_lerp, s0)
    final_s1 = torch.where(near_zero_omega, s1_lerp, s1)
    
    q_interp = final_s0 * q0 + final_s1 * q1_adjusted
    return normalize(q_interp) # Renormalize, esp. if LERP path taken or due to precision


def to_euler(x, order='zyx'):
    # Ensure device and dtype
    x = x.to(_get_tensor_props(x)[0], _get_tensor_props(x)[1]) # Use inferred props

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    if order == 'zyx':
        # Roll (X), Pitch (Y), Yaw (Z)
        # Equivalent to q_z * q_y * q_x
        # phi   (X-axis rotation) = atan2(2(w*x1 + x2*x3), 1 - 2(x1^2 + x2^2))
        # theta (Y-axis rotation) = asin (2(w*x2 - x3*x1))
        # psi   (Z-axis rotation) = atan2(2(w*x3 + x1*x2), 1 - 2(x2^2 + x3^2))
        # Original code seems to output [yaw, pitch, roll] based on names
        # For 'zyx' order, this means [angle_z, angle_y, angle_x]
        yaw = torch.atan2(2.0 * (x0 * x3 + x1 * x2), 1.0 - 2.0 * (x2 * x2 + x3 * x3))
        pitch = torch.asin(torch.clamp(2.0 * (x0 * x2 - x3 * x1), -1.0, 1.0))
        roll = torch.atan2(2.0 * (x0 * x1 + x2 * x3), 1.0 - 2.0 * (x1 * x1 + x2 * x2))
        return torch.cat([yaw, pitch, roll], dim=-1)
    elif order == 'xzy':
        # This is less standard. Using formulas from a reliable source or deriving is key.
        # Assuming order of application is Q = Qx * Qz * Qy
        # The original formulas were:
        # np.arctan2(2.0 * (x1*x0 - x2*x3), -x1*x1 + x2*x2 - x3*x3 + x0*x0),
        # np.arctan2(2.0 * (x2*x0 - x1*x3),  x1*x1 - x2*x2 - x3*x3 + x0*x0),
        # np.arcsin(np.clip(2.0 * (x1*x2 + x3*x0), -1.0, 1.0))
        # Let's verify the terms for a common xzy convention (e.g. intrinsic rotations: X, then new Z, then new Y)
        # For XZY intrinsic:
        # Angle X: atan2(2(x0*x1 - x2*x3), x0^2 - x1^2 + x2^2 - x3^2) WRONG. This is for specific rotation matrix entry.
        # Simpler: convert to matrix, then extract Euler angles from matrix.
        # Or use a known quaternion to Euler formula set.
        # E.g. from Wikipedia (conversion between Quaternions and Euler angles) for 'xzy' sequence:
        # Roll_x = atan2(-2*(x1*x2 - x0*x3), x0^2 - x1^2 + x2^2 - x3^2)  <- Note: sign difference from yours.
        # Pitch_z = asin(2*(x1*x3 + x0*x2))
        # Yaw_y = atan2(-2*(x2*x3 - x0*x1), x0^2 - x1^2 - x2^2 + x3^2)
        # Your original code does:
        eul_x = torch.atan2(2.0 * (x1 * x0 - x2 * x3), -x1 * x1 + x2 * x2 - x3 * x3 + x0 * x0) # Angle for 'x'
        eul_z = torch.atan2(2.0 * (x2 * x0 - x1 * x3),  x1 * x1 - x2 * x2 - x3 * x3 + x0 * x0) # Angle for 'z'
        eul_y = torch.asin(torch.clamp(2.0 * (x1 * x2 + x3 * x0), -1.0, 1.0))           # Angle for 'y'
        # The order of concatenation implies [angle_x, angle_z, angle_y]
        return torch.cat([eul_x, eul_z, eul_y], dim=-1)
    else:
        raise NotImplementedError('Cannot convert to ordering %s' % order)


def unroll(x): # x is typically a time series of quaternions, (Time, 4) or (Batch, Time, 4)
    y = x.clone()
    # Assuming first dim is time, or second if batched
    time_dim = 0 if x.dim() == 2 else 1
    
    for i in range(1, y.shape[time_dim]):
        # Select current and previous slices across all batches if present
        if x.dim() == 2: # (Time, 4)
            prev_q = y[i-1]
            curr_q = y[i]
        else: # (Batch, Time, 4) or more
            prev_q_slice = [slice(None)] * y.dim()
            prev_q_slice[time_dim] = i-1
            curr_q_slice = [slice(None)] * y.dim()
            curr_q_slice[time_dim] = i

            prev_q = y[tuple(prev_q_slice)]
            curr_q = y[tuple(curr_q_slice)]

        # d0 = sum(y[i] * y[i-1])  (dot product)
        # d1 = sum(-y[i] * y[i-1]) (dot product with -y[i])
        d0 = torch.sum(curr_q * prev_q, dim=-1)
        d1 = torch.sum(-curr_q * prev_q, dim=-1) # This is just -d0
        
        # Condition for flipping: d0 < d1  => d0 < -d0 => 2*d0 < 0 => d0 < 0
        # If dot product is negative, flip current quaternion
        flip_mask = d0 < 0.0
        
        # Apply flip. Ensure flip_mask is broadcastable if necessary.
        # For (Batch, Time, 4), d0 is (Batch,). flip_mask is (Batch,).
        # curr_q is (Batch, 4). Need flip_mask.unsqueeze(-1)
        if flip_mask.dim() < curr_q.dim() :
             flip_mask_expanded = flip_mask.unsqueeze(-1)
        else:
            flip_mask_expanded = flip_mask

        if x.dim() == 2:
            y[i] = torch.where(flip_mask_expanded, -curr_q, curr_q)
        else:
            y[tuple(curr_q_slice)] = torch.where(flip_mask_expanded, -curr_q, curr_q)
            
    return y


def to_xform(x): # Quaternion to 3x3 Rotation Matrix
    # Ensure device and dtype
    x = x.to(_get_tensor_props(x)[0], _get_tensor_props(x)[1])

    qw, qx, qy, qz = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]

    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2

    # Create rows then stack or concatenate
    row1 = torch.cat([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)
    row2 = torch.cat([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)
    row3 = torch.cat([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)

    # Stack rows to form matrices. Original used newaxis and then concatenate.
    # This means if input x is (..., 4), output should be (..., 3, 3)
    # row1, row2, row3 are (..., 3). We need to stack them along a new dimension.
    return torch.stack([row1, row2, row3], dim=-2)


def from_euler(e, order='zyx', device=None, dtype=torch.float32):
    # e is Euler angles (..., 3)
    # Infer device/dtype from e if it's a tensor
    if isinstance(e, torch.Tensor):
        device = e.device if device is None else device
        dtype = e.dtype if dtype is torch.float32 else dtype # careful with overriding
    elif device is None:
        device = torch.device('cpu')

    _axis_vals = {
        'x': torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
        'y': torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        'z': torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    }

    # Angles for each axis based on order
    # e[..., 0] corresponds to order[0], e[..., 1] to order[1], etc.
    q0 = from_angle_axis(e[..., 0], _axis_vals[order[0]])
    q1 = from_angle_axis(e[..., 1], _axis_vals[order[1]])
    q2 = from_angle_axis(e[..., 2], _axis_vals[order[2]])

    # Standard Euler composition: Q = Q(angle0, axis0) * Q(angle1, axis1) * Q(angle2, axis2)
    # This corresponds to intrinsic rotations in order 0, then 1, then 2.
    # Or extrinsic rotations in order 2, then 1, then 0.
    # The common 'zyx' convention is R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # If e = [yaw, pitch, roll] and order='zyx', then
    # q_yaw = from_angle_axis(e[...,0], axis_z)
    # q_pitch = from_angle_axis(e[...,1], axis_y)
    # q_roll = from_angle_axis(e[...,2], axis_x)
    # Result: q_yaw * q_pitch * q_roll
    # The original code implements mul(q0, mul(q1, q2)), so it's Q_first * Q_second * Q_third
    return mul(q0, mul(q1, q2))


def from_xform(ts, eps=1e-10): # 3x3 Rotation Matrix to Quaternion
    # Ensure device and dtype
    ts = ts.to(_get_tensor_props(ts)[0], _get_tensor_props(ts)[1])
    _eps = torch.tensor(eps, device=ts.device, dtype=ts.dtype)

    # Pre-allocate qs. Original used empty_like on a manipulated version of ts.
    # Shape of ts is (..., 3, 3). Shape of qs should be (..., 4).
    # ts[..., :1, 0] has shape (..., 1). .repeat(4, axis=-1) becomes .repeat_interleave(4, dim=-1) -> (..., 4)
    base_shape = ts.shape[:-2]
    qs = torch.empty(base_shape + (4,), device=ts.device, dtype=ts.dtype)

    # Trace of the matrix
    t = ts[..., 0, 0] + ts[..., 1, 1] + ts[..., 2, 2]

    # Case 1: trace > 0
    # s = 0.5 / np.sqrt(np.maximum(t + 1, eps))
    # qs_t_pos_w = (0.25 / s)
    # qs_t_pos_x = (s * (ts[..., 2, 1] - ts[..., 1, 2]))
    # qs_t_pos_y = (s * (ts[..., 0, 2] - ts[..., 2, 0]))
    # qs_t_pos_z = (s * (ts[..., 1, 0] - ts[..., 0, 1]))
    
    s_sqrt_val_case1 = torch.maximum(t + 1.0, _eps)
    s_case1 = 0.5 / torch.sqrt(s_sqrt_val_case1)
    
    val_w_case1 = (0.25 / s_case1)
    val_x_case1 = (s_case1 * (ts[..., 2, 1] - ts[..., 1, 2]))
    val_y_case1 = (s_case1 * (ts[..., 0, 2] - ts[..., 2, 0]))
    val_z_case1 = (s_case1 * (ts[..., 1, 0] - ts[..., 0, 1]))
    
    q_case1 = torch.stack([val_w_case1, val_x_case1, val_y_case1, val_z_case1], dim=-1)
    
    cond_case1 = (t > 0.0)
    qs = torch.where(cond_case1.unsqueeze(-1), q_case1, qs) # Fill qs where t > 0

    # Case 2: trace <= 0 AND ts[0,0] is largest diagonal element
    # c0 = (ts[..., 0, 0] > ts[..., 1, 1]) & (ts[..., 0, 0] > ts[..., 2, 2])
    # s0 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2], eps))
    # qs_c0_w = ((ts[..., 2, 1] - ts[..., 1, 2]) / s0)
    # qs_c0_x = (s0 * 0.25)
    # qs_c0_y = ((ts[..., 0, 1] + ts[..., 1, 0]) / s0)
    # qs_c0_z = ((ts[..., 0, 2] + ts[..., 2, 0]) / s0)

    s_sqrt_val_case2 = torch.maximum(1.0 + ts[..., 0, 0] - ts[..., 1, 1] - ts[..., 2, 2], _eps)
    s_case2 = 2.0 * torch.sqrt(s_sqrt_val_case2)

    val_w_case2 = ((ts[..., 2, 1] - ts[..., 1, 2]) / s_case2)
    val_x_case2 = (s_case2 * 0.25)
    val_y_case2 = ((ts[..., 0, 1] + ts[..., 1, 0]) / s_case2)
    val_z_case2 = ((ts[..., 0, 2] + ts[..., 2, 0]) / s_case2)

    q_case2 = torch.stack([val_w_case2, val_x_case2, val_y_case2, val_z_case2], dim=-1)

    cond_case2 = (~cond_case1) & (ts[..., 0, 0] > ts[..., 1, 1]) & (ts[..., 0, 0] > ts[..., 2, 2])
    qs = torch.where(cond_case2.unsqueeze(-1), q_case2, qs)

    # Case 3: trace <= 0 AND ts[1,1] is largest (and not covered by case 2)
    # c1 = (~c0) & (ts[..., 1, 1] > ts[..., 2, 2])
    # s1 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 1, 1] - ts[..., 0, 0] - ts[..., 2, 2], eps))
    # qs_c1_w = ((ts[..., 0, 2] - ts[..., 2, 0]) / s1)
    # qs_c1_x = ((ts[..., 0, 1] + ts[..., 1, 0]) / s1)
    # qs_c1_y = (s1 * 0.25)
    # qs_c1_z = ((ts[..., 1, 2] + ts[..., 2, 1]) / s1)

    s_sqrt_val_case3 = torch.maximum(1.0 + ts[..., 1, 1] - ts[..., 0, 0] - ts[..., 2, 2], _eps)
    s_case3 = 2.0 * torch.sqrt(s_sqrt_val_case3)

    val_w_case3 = ((ts[..., 0, 2] - ts[..., 2, 0]) / s_case3)
    val_x_case3 = ((ts[..., 0, 1] + ts[..., 1, 0]) / s_case3)
    val_y_case3 = (s_case3 * 0.25)
    val_z_case3 = ((ts[..., 1, 2] + ts[..., 2, 1]) / s_case3)

    q_case3 = torch.stack([val_w_case3, val_x_case3, val_y_case3, val_z_case3], dim=-1)
    
    cond_case3 = (~cond_case1) & (~((ts[..., 0, 0] > ts[..., 1, 1]) & (ts[..., 0, 0] > ts[..., 2, 2]))) & \
                 (ts[..., 1, 1] > ts[..., 2, 2])
    qs = torch.where(cond_case3.unsqueeze(-1), q_case3, qs)

    # Case 4: trace <= 0 AND ts[2,2] is largest (the rest)
    # c2 = (~c0) & (~c1)
    # s2 = 2.0 * np.sqrt(np.maximum(1.0 + ts[..., 2, 2] - ts[..., 0, 0] - ts[..., 1, 1], eps))
    # qs_c2_w = ((ts[..., 1, 0] - ts[..., 0, 1]) / s2)
    # qs_c2_x = ((ts[..., 0, 2] + ts[..., 2, 0]) / s2)
    # qs_c2_y = ((ts[..., 1, 2] + ts[..., 2, 1]) / s2)
    # qs_c2_z = (s2 * 0.25)

    s_sqrt_val_case4 = torch.maximum(1.0 + ts[..., 2, 2] - ts[..., 0, 0] - ts[..., 1, 1], _eps)
    s_case4 = 2.0 * torch.sqrt(s_sqrt_val_case4)
    
    val_w_case4 = ((ts[..., 1, 0] - ts[..., 0, 1]) / s_case4)
    val_x_case4 = ((ts[..., 0, 2] + ts[..., 2, 0]) / s_case4)
    val_y_case4 = ((ts[..., 1, 2] + ts[..., 2, 1]) / s_case4)
    val_z_case4 = (s_case4 * 0.25)

    q_case4 = torch.stack([val_w_case4, val_x_case4, val_y_case4, val_z_case4], dim=-1)

    cond_case4 = (~cond_case1) & (~cond_case2) & (~cond_case3) # This covers the remaining
    qs = torch.where(cond_case4.unsqueeze(-1), q_case4, qs)
    
    return qs


def fk(lrot, lpos, parents):
    # lrot: (Batch, Joints, 4) local rotations
    # lpos: (Batch, Joints, 3) local positions (offsets)
    # parents: list of parent indices
    
    # Ensure device and dtype consistency
    common_device, common_dtype = _get_tensor_props(lrot, lpos)
    lrot = lrot.to(common_device, common_dtype)
    lpos = lpos.to(common_device, common_dtype)

    num_joints = lrot.shape[-2]
    
    # Initialize lists for global positions and rotations
    # Store them as tensors directly if batch dim is consistent
    # Or use lists of tensors if shapes vary (not typical for FK)
    
    # Assuming lrot, lpos have shape (..., N, Dims) where N is num_joints
    # Slicing like lpos[..., :1, :] gives (..., 1, Dims)
    
    # Use lists to accumulate, then cat
    gp_list = [lpos[..., :1, :]] # Global position of root
    gr_list = [lrot[..., :1, :]] # Global rotation of root

    for i in range(1, num_joints):
        parent_idx = parents[i]
        
        # Global position of current joint:
        # gp_parent + rotate_vector(gr_parent, lpos_current)
        # gp_i = gp_list[parent_idx] + mul_vec(gr_list[parent_idx], lpos[..., i:i+1, :])
        # Need to use the ALREADY COMPUTED global rotation of the parent (gr_list[parent_idx])
        # NOT gr_list[parents[i]] which would be wrong if parents[i] is not parent_idx
        # It's correct, gr_list is indexed by joint index.
        
        # mul_vec expects quaternion and vector.
        # gr_list[parent_idx] is (..., 1, 4)
        # lpos[..., i:i+1, :] is (..., 1, 3)
        # This should work.
        
        current_global_pos = mul_vec(gr_list[parent_idx], lpos[..., i:i + 1, :]) + gp_list[parent_idx]
        gp_list.append(current_global_pos)
        
        # Global rotation of current joint:
        # gr_parent * lrot_current
        # gr_i = mul(gr_list[parent_idx], lrot[..., i:i+1, :])
        current_global_rot = mul(gr_list[parent_idx], lrot[..., i:i + 1, :])
        gr_list.append(current_global_rot)

    # Concatenate along the joint dimension (typically dim=-2 or dim=1 if Batch, Joints, Dim)
    # Original used axis=-2. If input is (Batch, Joints, Dim), then dim=1.
    # Let's assume shape is (..., NumJoints, Features)
    # Then concatenation should be on dim=-2
    
    # Check shapes: gr_list contains tensors of shape (..., 1, 4)
    # gp_list contains tensors of shape (..., 1, 3)
    # cat along dim=-2 will make (..., NumJoints, Features)
    
    global_rotations = torch.cat(gr_list, dim=-2)
    global_positions = torch.cat(gp_list, dim=-2)
    
    return global_rotations, global_positions


def fk_vel(lrot, lpos, lvrt, lvel, parents):
    # lrot: local rotations (quaternions)
    # lpos: local positions (offsets)
    # lvrt: local angular velocities (helical coordinates / log-quaternion derivatives)
    # lvel: local linear velocities
    # parents: list of parent indices

    common_device, common_dtype = _get_tensor_props(lrot, lpos, lvrt, lvel)
    lrot = lrot.to(common_device, common_dtype)
    lpos = lpos.to(common_device, common_dtype)
    lvrt = lvrt.to(common_device, common_dtype) # Angular velocities
    lvel = lvel.to(common_device, common_dtype) # Linear velocities

    num_joints = lrot.shape[-2]

    # Global lists
    gp_list = [lpos[..., :1, :]]  # Global positions
    gr_list = [lrot[..., :1, :]]  # Global rotations
    # Global angular velocities (often denoted omega or w)
    # These are 3D vectors in world space. lvrt are local.
    gt_list = [lvrt[..., :1, :]] # Global angular velocities (world frame)
    # Global linear velocities (world frame)
    gv_list = [lvel[..., :1, :]]  # Global linear velocities

    for i in range(1, num_joints):
        parent_idx = parents[i]
        
        # Parent's global state
        gr_parent = gr_list[parent_idx]
        gp_parent = gp_list[parent_idx]
        gt_parent = gt_list[parent_idx] # Parent's global angular velocity
        gv_parent = gv_list[parent_idx] # Parent's global linear velocity
        
        # Current joint's local state
        lpos_i = lpos[..., i:i + 1, :]
        lrot_i = lrot[..., i:i + 1, :]
        lvrt_i = lvrt[..., i:i + 1, :] # Local angular velocity of joint i
        lvel_i = lvel[..., i:i + 1, :] # Local linear velocity of joint i (relative to parent)

        # FK for position and rotation
        # R_i = R_p * r_i
        gr_i = mul(gr_parent, lrot_i)
        # p_i = p_p + R_p * t_i (where t_i is local offset lpos_i)
        rotated_lpos_i = mul_vec(gr_parent, lpos_i)
        gp_i = gp_parent + rotated_lpos_i
        
        gr_list.append(gr_i)
        gp_list.append(gp_i)

        # FK for velocities
        # Global angular velocity: w_i = w_p + R_p * w_local_i
        # where w_local_i is lvrt_i
        # gt_i = gt_parent + mul_vec(gr_parent, lvrt_i)
        # Note: lvrt might be axis-angle representation of angular velocity.
        # If lvrt is d(log(r_i))/dt, then this is slightly more complex.
        # Assuming lvrt_i is the angular velocity of joint i *in parent's frame*.
        # Then world angular velocity of joint i is w_parent + R_parent * lvrt_i.
        # The original code does: `gt.append(gt[parents[i]] + mul_vec(gr[parents[i]], lvrt[..., i:i + 1, :]))`
        # This seems correct if lvrt_i is angular velocity of child wrt parent, expressed in parent frame.
        rotated_lvrt_i = mul_vec(gr_parent, lvrt_i)
        gt_i = gt_parent + rotated_lvrt_i
        gt_list.append(gt_i)

        # Global linear velocity: v_i = v_p + w_p x (R_p * t_i) + R_p * v_local_i
        # v_i = v_parent + global_angular_vel_parent x (global_pos_child - global_pos_parent) + R_parent * local_linear_vel_child
        # The term (R_p * t_i) is `rotated_lpos_i`
        # The term (R_p * v_local_i) is `mul_vec(gr_parent, lvel_i)`
        # Original: gv.append(gv[p] + mul_vec(gr[p], lvel[i]) + _fast_cross(gt[p], mul_vec(gr[p], lpos[i])))
        # This is: v_parent + R_parent * v_local_child + w_parent x (R_parent * offset_child)
        
        term1_lin_vel = gv_parent
        term2_lin_vel = mul_vec(gr_parent, lvel_i) # R_p * v_local_i
        term3_lin_vel = _fast_cross(gt_parent, rotated_lpos_i) # w_p x (R_p * t_i)
        
        gv_i = term1_lin_vel + term2_lin_vel + term3_lin_vel
        gv_list.append(gv_i)

    global_rotations = torch.cat(gr_list, dim=-2)
    global_positions = torch.cat(gp_list, dim=-2)
    global_angular_velocities = torch.cat(gt_list, dim=-2)
    global_linear_velocities = torch.cat(gv_list, dim=-2)
    
    return global_rotations, global_positions, global_angular_velocities, global_linear_velocities