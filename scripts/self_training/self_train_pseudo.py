"""self_train.py — pseudo-label 생성 모듈.

extract_peaks                   : belief map → 2D keypoint + confidence
generate_belief_maps_from_keypoints : keypoints → pseudo belief target
_apply_filter                    : filter_type 별 dispatcher (none/conf/ransac/bc/ransac_loo)
generate_pseudo_labels           : real_loader 순회 → forward → filter → accepted list
"""
from __future__ import annotations
import numpy as np
import torch
import torchvision.transforms as transforms

from utils import CreateBeliefMap
from canonical_filters import filter_B as canonical_filter_B
from canonical_filters import filter_C as canonical_filter_C

# camera-facing 0123 L-R mirror swap (180° yaw). flip TTA 일관성 필터용.
FLIP_PAIRS = [(0, 1), (3, 2), (4, 5), (7, 6)]


def _flip_score(model, img_tensor, keypoints_2d, image_size, device, normalize,
                peak_threshold=0.3):
    """L-R flip TTA 일관성: flip 추론 → un-flip x(size-x) + FLIP_PAIRS swap →
    원 예측과 코너별 거리. 양쪽 검출된 코너의 median px (겹침<4면 999)."""
    flip_img = torch.flip(img_tensor, dims=[2])            # width축 L-R flip
    with torch.no_grad():
        out_bel, _ = model(normalize(flip_img.clone()).unsqueeze(0).to(device))
    bmap = out_bel[-1][0].cpu().numpy()
    kpf_raw = extract_peaks(bmap, threshold=peak_threshold, image_size=image_size)
    kpf = [None] * 9
    for i in range(8):
        if kpf_raw[i] is not None:
            kpf[i] = (image_size - kpf_raw[i][0], kpf_raw[i][1])   # un-flip
    swap = list(range(9))
    for a, b in FLIP_PAIRS:
        swap[a], swap[b] = b, a
    kpf_sw = [kpf[swap[i]] for i in range(9)]
    dists = []
    for i in range(8):
        o, f = keypoints_2d[i], kpf_sw[i]
        if o is not None and f is not None:
            dists.append(float(np.hypot(o[0] - f[0], o[1] - f[1])))
    return float(np.median(dists)) if len(dists) >= 4 else 999.0


def extract_peaks(belief_maps, threshold=0.3, image_size=448):
    """belief maps (9, H, W) → 9 keypoints (u, v, conf) or None each.

    sub-pixel refinement: 5×5 patch weighted average around argmax.
    """
    if isinstance(belief_maps, torch.Tensor):
        belief_maps = belief_maps.cpu().numpy()

    num_kp, bh, bw = belief_maps.shape
    scale_x = image_size / bw
    scale_y = image_size / bh

    keypoints = []
    for i in range(num_kp):
        bmap = belief_maps[i]
        max_val = float(bmap.max())
        if max_val < threshold:
            keypoints.append(None)
            continue

        idx = bmap.argmax()
        py, px = divmod(int(idx), bw)
        win = 5; half = win // 2
        y_start = max(0, py - half); y_end = min(bh, py + half + 1)
        x_start = max(0, px - half); x_end = min(bw, px + half + 1)
        patch = bmap[y_start:y_end, x_start:x_end]
        if patch.sum() < 1e-8:
            keypoints.append(None)
            continue

        ys, xs = np.meshgrid(
            np.arange(y_start, y_end),
            np.arange(x_start, x_end), indexing='ij')
        cx = float(np.average(xs, weights=patch))
        cy = float(np.average(ys, weights=patch))
        keypoints.append((cx * scale_x, cy * scale_y, max_val))
    return keypoints


def generate_belief_maps_from_keypoints(keypoints_2d, image_size=448,
                                        belief_map_size=56, sigma=2.0):
    """keypoints → (9, H', W') belief tensor (pseudo-label target)."""
    scale = belief_map_size / image_size
    kps = []
    for pt in keypoints_2d:
        if pt is None:
            kps.append([-100, -100])
        else:
            kps.append([pt[0] * scale, pt[1] * scale])
    beliefs = CreateBeliefMap(
        size=belief_map_size, pointsBelief=[kps],
        sigma=sigma, nbpoints=9, save=False,
    )
    return torch.from_numpy(np.array(beliefs)).float()


def _apply_filter(filter_type, keypoints_2d, pnp_solver, geo_filter,
                  gf_cfg, image_size):
    """filter_type 에 따라 dispatch. Returns (is_valid, R, t, reason)."""
    if filter_type == "none":
        return True, None, None, None

    if filter_type == "conf":
        conf_min = float(gf_cfg.get("conf_min", 0.5))
        confs = [float(p[2]) for p in keypoints_2d[:8]
                 if p is not None and len(p) >= 3]
        if not confs or min(confs) < conf_min:
            return False, None, None, "filter_fail"
        return True, None, None, None

    if filter_type == "ransac":
        is_valid, R, t, _ = geo_filter.solve_and_validate(keypoints_2d)
        if R is None:
            return False, None, None, "pnp_fail"
        if not is_valid:
            return False, R, t, "filter_fail"
        return True, R, t, None

    if filter_type == "bc":
        success, R, t, _ = pnp_solver.solve(keypoints_2d)
        if not success or R is None:
            return False, None, None, "pnp_fail"
        b_pass, _ = canonical_filter_B(
            keypoints_2d, pnp_solver, R, t,
            tau_span=float(gf_cfg.get("bc_tau_span", 0.35)),
            tau_end=float(gf_cfg.get("bc_tau_end", 0.10)),
            tau_nc=float(gf_cfg.get("bc_tau_nc", 0.02)),
            min_kps=int(gf_cfg.get("bc_min_kps_B", 4)),
            img_size=(image_size, image_size),
        )
        if not b_pass:
            return False, R, t, "filter_fail"
        c_pass, _ = canonical_filter_C(
            keypoints_2d, pnp_solver, R, t,
            tau_C=float(gf_cfg.get("bc_tau_C", 0.05)),
            min_kps=int(gf_cfg.get("bc_min_kps_C", 5)),
        )
        if not c_pass:
            return False, R, t, "filter_fail"
        size_pass, _ = geo_filter._check_size(R, t)
        if not size_pass:
            return False, R, t, "filter_fail"
        return True, R, t, None

    if filter_type in ("ransac_loo", "ransac_loo_flip"):
        # flip 게이트는 generate_pseudo_labels 에서 (model/img 필요) 별도 적용.
        is_valid, R, t, _ = geo_filter.solve_and_validate(keypoints_2d)
        if R is None:
            return False, None, None, "pnp_fail"
        if not is_valid:
            return False, R, t, "filter_fail"
        loo_tau = float(gf_cfg.get("loo_tau", 0.05))
        loo_solver = gf_cfg.get("_loo_solver")
        if loo_solver is None:
            return True, R, t, None
        try:
            ok_loo, R_loo, t_loo, _, _ = loo_solver.solve_adaptive(keypoints_2d)
            if not ok_loo:
                return False, R, t, "filter_fail"
            _, loo_score = canonical_filter_C(
                keypoints_2d, loo_solver, R_loo, t_loo, tau_C=loo_tau)
            if loo_score >= loo_tau:
                return False, R, t, "filter_fail"
        except Exception:
            return False, R, t, "filter_fail"
        return True, R, t, None

    raise ValueError(f"Unknown filter_type: {filter_type}")


def generate_pseudo_labels(model, real_loader, pnp_solver, geo_filter,
                           weak_aug, config, device):
    """real_loader 순회 → DOPE forward → filter → accepted 반환.

    config.geometric_filter.filter_type 으로 dispatch:
      ransac (default) | bc | conf | ransac_loo | none
    """
    image_size = config["self_training"]["image_size"]
    sigma = config["self_training"]["sigma"]
    peak_threshold = 0.3

    gf_cfg = config["geometric_filter"]
    filter_type = str(gf_cfg.get("filter_type", "ransac")).lower()
    min_kps = int(gf_cfg.get("min_keypoints", 5))

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    accepted = []
    total = 0
    num_pnp_fail = 0
    num_filter_fail = 0
    reproj_errors = []

    model.eval()
    with torch.no_grad():
        for batch in real_loader:
            imgs = batch["img"]
            for i in range(imgs.size(0)):
                total += 1
                img = imgs[i]
                img_weak = weak_aug(img.clone())
                img_input = normalize(img_weak).unsqueeze(0).to(device)
                out_bel, _ = model(img_input)
                belief_maps = out_bel[-1][0].cpu().numpy()

                keypoints_2d = extract_peaks(
                    belief_maps, threshold=peak_threshold, image_size=image_size)

                valid_count = sum(1 for kp in keypoints_2d if kp is not None)
                if valid_count < min_kps:
                    num_pnp_fail += 1
                    continue

                is_valid, R, t, reason = _apply_filter(
                    filter_type, keypoints_2d, pnp_solver, geo_filter,
                    gf_cfg, image_size,
                )

                if R is not None:
                    reproj_all = pnp_solver.reproject(R, t)[:8]
                    det_idx = [k for k, p in enumerate(keypoints_2d[:8]) if p is not None]
                    det_2d = np.array(
                        [[keypoints_2d[k][0], keypoints_2d[k][1]] for k in det_idx],
                        dtype=np.float64)
                    if len(det_idx) > 0:
                        errs = np.linalg.norm(reproj_all[det_idx] - det_2d, axis=1)
                        reproj_errors.append(float(np.mean(errs)))

                if not is_valid:
                    if reason == "pnp_fail":
                        num_pnp_fail += 1
                    else:
                        num_filter_fail += 1
                    continue

                # flip 게이트 (ransac_loo_flip): L-R flip TTA 일관성 <= flip_tau
                if "flip" in filter_type:
                    flip_tau = float(gf_cfg.get("flip_tau", 10.0))
                    fscore = _flip_score(model, img, keypoints_2d, image_size,
                                         device, normalize, peak_threshold)
                    if fscore > flip_tau:
                        num_filter_fail += 1
                        continue

                pseudo_beliefs = generate_belief_maps_from_keypoints(
                    keypoints_2d, image_size=image_size,
                    belief_map_size=config["self_training"]["belief_map_size"],
                    sigma=sigma)
                accepted.append({"img": img.cpu(), "beliefs": pseudo_beliefs})

    num_accepted = len(accepted)
    stats = {
        "filter_type": filter_type,
        "total": total,
        "accepted": num_accepted,
        "acceptance_rate": num_accepted / total if total > 0 else 0,
        "pnp_fail": num_pnp_fail,
        "filter_fail": num_filter_fail,
        "reproj_error_mean": float(np.mean(reproj_errors)) if reproj_errors else 0,
    }
    return accepted, stats
