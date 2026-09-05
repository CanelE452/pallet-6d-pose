"""GATE E — 합성/실제 도메인 편향 진단. 학습 0(로지스틱 회귀만).

    1. R0 frozen backbone/neck 의 3 level feature 를 global average pooling
    2. SOURCE_DEV(합성 val) vs TARGET_UNLABELED 를 같은 수로 뽑아 로지스틱 회귀 5-fold
    3. domain AUROC
    4. 그 classifier 의 점수를 DEV_EVAL real 프레임에 붙여
       R0 의 keypoint 오차 / gross 프레임 지표와의 연관을 본다

★ PAPER_EVAL 로 classifier 를 학습하지 않는다. fit 은 SOURCE_DEV + TARGET_UNLABELED 만.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtcd_common as M
import mtcd_teachers as T

SYN = M.REPO_ROOT / "challenge/yolo_pose_one_model/datasets/g38_legacy_v1v2_p0_tex20k"
POOL = M.REPO_ROOT / "data/evaluation/pallet_eval_v1/adaptation/MAIN_UNLABELED_BALANCED.csv"
REF = "T0_R0_YOLO26N_G38LEGACY"
N_PER_DOMAIN = 500


class NeckTap:
    """Pose26 head 로 들어가는 3 level feature 를 가로챈다."""

    def __init__(self, yolo):
        self.features = None
        head = yolo.model.model[-1]
        self.handle = head.register_forward_pre_hook(self._hook, with_kwargs=False)

    def _hook(self, module, args):
        x = args[0]
        if isinstance(x, (list, tuple)):
            self.features = [t.detach() for t in x]
        return None

    def close(self):
        self.handle.remove()


@torch.no_grad()
def features_for(yolo, tap, image, already_padded=False):
    padded = image if already_padded else T.pad_image(image)
    yolo.predict(padded, conf=0.001, imgsz=T.YOLO_IMGSZ, augment=False, half=False,
                 device="0", verbose=False)
    if tap.features is None:
        return None
    return np.concatenate([f.float().mean(dim=(2, 3)).cpu().numpy().ravel()
                           for f in tap.features]), \
        [f.float().mean(dim=(2, 3)).cpu().numpy().ravel() for f in tap.features]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_PER_DOMAIN)
    args = parser.parse_args()

    registry = json.loads((M.TRACK / "TEACHER_REGISTRY.json").read_text())["teachers"]
    weights = M.REPO_ROOT / registry[REF]["checkpoint"]
    yolo = T.load_yolo(weights)
    tap = NeckTap(yolo)

    syn = sorted(p.name for p in (SYN / "images/val").glob("*.png"))
    random.Random(20260908).shuffle(syn)
    pool = list(csv.DictReader(POOL.open()))
    random.Random(20260908).shuffle(pool)

    X_levels, y, meta = [[], [], []], [], []
    for name in syn[:args.n]:
        img = cv2.imread(str(SYN / "images/val" / name))
        if img is None:
            continue
        got = features_for(yolo, tap, img, already_padded=True)
        if got is None:
            continue
        for i, v in enumerate(got[1]):
            X_levels[i].append(v)
        y.append(0)
        meta.append(("SOURCE_DEV", name))
    for row in pool[:args.n]:
        img = cv2.imread(str(M.REPO_ROOT / row["image_path"]))
        if img is None:
            continue
        got = features_for(yolo, tap, img)
        if got is None:
            continue
        for i, v in enumerate(got[1]):
            X_levels[i].append(v)
        y.append(1)
        meta.append(("TARGET_UNLABELED", row["capture_session"]))
    y = np.asarray(y)
    print(f"source {int((y == 0).sum())}  target {int((y == 1).sum())}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    level_auc, fitted = {}, {}
    for i, feats in enumerate(X_levels):
        X = np.asarray(feats)
        aucs, oof = [], np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], y[tr])
            s = clf.predict_proba(X[te])[:, 1]
            oof[te] = s
            aucs.append(roc_auc_score(y[te], s))
        level_auc[f"level{i}"] = {"auroc_mean": float(np.mean(aucs)),
                                  "auroc_folds": [float(a) for a in aucs],
                                  "feature_dim": int(X.shape[1])}
        full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        full.fit(X, y)
        fitted[i] = full
        print(f"  level{i}  dim {X.shape[1]:4d}  domain AUROC {np.mean(aucs):.4f}")

    best_level = max(level_auc, key=lambda k: level_auc[k]["auroc_mean"])
    best_i = int(best_level[-1])
    best_auroc = level_auc[best_level]["auroc_mean"]

    # ---- DEV_EVAL 에 점수를 붙인다 (classifier 는 여기서 학습되지 않았다)
    gts = [M.load_gt(f) for f in M.dev_eval_frames()]
    r0 = M.load_prediction_file(M.PREDICTIONS / f"{REF}.json")
    scores, kp_median, gross_frame, sess = [], [], [], []
    for gt in gts:
        img = cv2.imread(str(M.REPO_ROOT / gt["image"]))
        if img is None:
            continue
        got = features_for(yolo, tap, img)
        pts = M.prediction_keypoints(r0.get(gt["frame_id"]))
        if got is None or pts is None:
            continue
        s = fitted[best_i].predict_proba(got[1][best_i][None, :])[0, 1]
        d = np.linalg.norm(pts - gt["xy"], axis=1)
        m = gt["supervised"] & np.isfinite(d)
        if not m.any():
            continue
        scores.append(float(s))
        kp_median.append(float(np.median(d[m])))
        gross_frame.append(bool((d[m] > 20).any()))
        sess.append(gt["session_id"])
    tap.close()

    from scipy.stats import spearmanr
    scores = np.asarray(scores); kp_median = np.asarray(kp_median)
    gross = np.asarray(gross_frame)
    rho, pval = spearmanr(scores, kp_median)
    auc_gross = float(roc_auc_score(gross, scores)) if 0 < gross.sum() < len(gross) else None

    report = {
        "schema_version": "mtcd_gate_e_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": registry[REF]["checkpoint"],
        "backbone_sha256": registry[REF]["sha256"],
        "n_source": int((y == 0).sum()), "n_target": int((y == 1).sum()),
        "classifier_fit_population": "SOURCE_DEV + TARGET_UNLABELED only — PAPER_EVAL 미사용",
        "domain_auroc_by_level": level_auc,
        "best_level": best_level, "best_domain_auroc": best_auroc,
        "association_on_dev_eval": {
            "n_frames": len(scores),
            "spearman_score_vs_kp_median": float(rho),
            "spearman_p": float(pval),
            "auc_score_separates_gross_frames": auc_gross,
            "gross_frame_rate": float(gross.mean()),
        },
        "thresholds": {"domain_auroc": 0.85, "gross_separation_auc": 0.65},
    }
    strong_domain = best_auroc >= 0.85
    strong_link = (auc_gross or 0) >= 0.65
    report["TARGET_BIAS_SIGNAL"] = (
        "STRONG" if strong_domain and strong_link else
        "DOMAIN_SEPARABLE_BUT_NOT_ERROR_LINKED" if strong_domain or strong_link else
        "NO_ACTIONABLE_DOMAIN_BIAS")
    report["adapter_admission"] = (
        "RUN" if report["TARGET_BIAS_SIGNAL"] == "STRONG" else "NOT_RUN")

    out = M.GATE_E / "GATE_E_RESULT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float) + "\n")
    print(f"\nbest level {best_level}  domain AUROC {best_auroc:.4f}")
    print(f"DEV_EVAL n={len(scores)}  spearman(score, kp median) {rho:+.4f} (p={pval:.3g})  "
          f"gross-frame 분리 AUC {auc_gross}")
    print(f"TARGET_BIAS_SIGNAL = {report['TARGET_BIAS_SIGNAL']}   adapter {report['adapter_admission']}")
    print(f"-> {out.relative_to(M.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
