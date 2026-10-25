"""Pure public AUX query transport; the encoder sees RGB, q, L and valid only."""
import numpy as np
import torch

LAYOUT = {"full_CLS": [0,384], "fixed_Q": [384,768],
          "centre_CLS": [768,1152], "fractions": [1152,1154],
          "zero_condition_AUX_only": [1154,1186]}

def query_crop(rgb, q, L, crop):
    """V90 FP64 bilinear/half-up uint8; unchanged q at RAW96 47.5."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("uint8 HWC RGB required")
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (2,) or not np.isfinite(q).all() or not np.isfinite(L) or L <= 0:
        return np.zeros((3,96,96),np.uint8), np.zeros((1,96,96),bool), False
    raw, valid = crop(rgb, q, float(L), 96)
    if raw.shape != (3,96,96) or raw.dtype != np.uint8 or valid.shape != (1,96,96) or valid.dtype != bool:
        raise ValueError("Pinned crop output contract")
    supported = bool(valid[:,47:49,47:49].all() and valid.any())
    return raw, valid, supported

def append_aux_condition(pixel):
    if pixel.ndim != 2 or pixel.shape[1] != 1154 or pixel.dtype != np.float32 or not np.isfinite(pixel).all():
        raise ValueError("Finite FP32 canonical pixel1154 required")
    return np.concatenate((pixel,np.zeros((len(pixel),32),np.float32)),axis=1)

def encode_queries(model, raw, valid, node_pixel_batch, preprocess, crop_module, device):
    """One canonical full+centre forward; no teacher argument or classifier."""
    pixel, fractions, central4 = node_pixel_batch(
        model,torch.from_numpy(raw),torch.from_numpy(valid),preprocess,crop_module,device)
    return append_aux_condition(pixel), fractions, central4

def selected_grid_indices(frame_id):
    import hashlib
    ranked = sorted(range(4800),key=lambda i:(hashlib.sha256(
        f"V260_PUBLIC_TRANSPORT_PILOT_V1|{frame_id}|{i}".encode("utf-8")).digest(),i))
    return sorted(ranked[:8])
