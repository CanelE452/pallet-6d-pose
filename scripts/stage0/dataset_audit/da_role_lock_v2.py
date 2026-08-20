"""DATA_ROLE_LOCK V2 — EDGE_HARD_TRUNC 학습 제외를 해제한다.

기록을 **덮어쓰지 않고 supersede** 한다. V1 은 그 시점에 알던 것으로 옳았고,
지워버리면 왜 바뀌었는지가 사라진다.

해제 근거는 두 갈래인데 세기가 다르다 — 섞지 않고 적는다.

 (강) 배제 기준이 범주 오류였다.  G1 게이트 `V_vis >= 4` 는 **평가에서 PnP 가
      성립하는가** 를 보는 것인데 그걸 **학습 배제** 기준으로 썼다.  corner loss 가
      실제로 학습하는 것은 가시 코너가 아니라 화면 안 채널(n_supervised)이고,
      EDGE 12,000 장 중 99.2% 가 n_supervised >= 4 다 (중앙값 5).
      line_valid_roles >= 6 도 99.1%.  즉 이 프레임들은 감독 신호가 있다.

 (약) line 이 corner 를 대체할 수 있다.  근거는 O12 oracle 감사(eval56 98.7%,
      4.68px)인데 그건 **GT 선**이다.  예측 선으로는 아직 그 수준이 아니며, 오늘
      REAL_DEV 에서 F3 의 paired 개선은 네 지표 모두 CI 가 0 을 포함했다.
      따라서 이 근거만으로는 해제하지 않았을 것이다 — [미검증] 으로 남긴다.

바뀌지 않는 것:
 * dev / untouched split 은 그대로 평가 전용.  코너 제한 해제는 train/dev 분리를
   푸는 것이 아니다.
 * F3 는 여전히 Point PnP 초기화를 요구한다.  EDGE 로 line 을 키워도 추론 시
   코너가 없으면 pose 가 안 나온다.  그건 별개의 열린 문제다.
 * CORNER_LA 는 여전히 제외.  그쪽 배제 사유는 코너 수가 아니라 효과 미확립이다.
"""
from __future__ import annotations

import hashlib, json, pathlib, sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA  # noqa: E402

TRAIN_SETS = ["BROAD_40K", "EDGE_HARD_TRUNC_TRAIN"]
EVAL_ONLY = ["EDGE_HARD_TRUNC_DEV", "EDGE_HARD_TRUNC_UNTOUCHED",
             "EDGE_HARD_CLEAN_UNTOUCHED"]


def main():
    frame = pd.read_parquet(DA.AUDIT / "positive_frame_features.parquet")
    edge = frame[frame["dataset_id"].str.startswith("EDGE_HARD_TRUNC")]
    evidence = {
        "n_edge_trunc": int(len(edge)),
        "V_vis_actual": {str(k): int(v) for k, v in
                         sorted(edge["V_vis_actual"].value_counts().items())},
        "n_supervised": {str(k): int(v) for k, v in
                         sorted(edge["n_supervised"].value_counts().items())},
        "n_supervised_ge4_rate": round(float((edge["n_supervised"] >= 4).mean()), 4),
        "line_valid_roles_ge6_rate":
            round(float((edge["line_valid_roles"] >= 6).mean()), 4),
    }

    broad = frame[frame["dataset_id"] == "BROAD_40K"]
    edge_train = frame[frame["dataset_id"] == "EDGE_HARD_TRUNC_TRAIN"]

    decision = {
        "name": "DATA_ROLE_LOCK_V2",
        "date": "2026-08-20",
        "supersedes": "FINAL_DATA_ROLE_LOCK.md (V1) — EDGE 항목만. 나머지는 유효",
        "authorised_by": "user",
        "CHANGE": "EDGE_HARD_TRUNC_TRAIN 10,000 을 학습에서 제외하던 규칙을 해제한다",
        "primary_reason": {
            "claim": "배제 기준이 범주 오류였다",
            "detail": "G1 게이트 V_vis>=4 는 평가에서 PnP 성립 여부를 보는 것인데 "
                      "학습 배제에 썼다. corner loss 가 학습하는 것은 화면 안 채널"
                      "(n_supervised)이고 EDGE 의 99.2% 가 >=4 다.",
            "evidence": evidence},
        "secondary_reason_UNVERIFIED": {
            "claim": "line 이 corner 없이도 pose 를 낸다",
            "status": "[미검증]",
            "detail": "근거인 O12 감사(eval56 98.7% / 4.68px)는 GT 선 oracle 이다. "
                      "예측 선으로는 미확인이고, REAL_DEV 에서 F3 의 paired 개선은 "
                      "R/t/ADD-S/IoU 네 지표 모두 CI 가 0 을 포함했다.",
            "note": "이 근거만으로는 해제하지 않는다. 위 primary 가 해제 사유다."},
        "roles": {
            "BROAD_40K": {"n": int(len(broad)), "role": "MAIN_TRAIN",
                          "branches": ["corner", "line"]},
            "EDGE_HARD_TRUNC_TRAIN": {
                "n": int(len(edge_train)), "role": "MAIN_TRAIN (V2 에서 승격)",
                "branches": ["corner", "line"],
                "was": "LINE_SUPPLEMENT_CANDIDATE, sampling weight 0"},
            "EDGE_HARD_TRUNC_DEV": {"role": "EVAL_ONLY", "unchanged": True},
            "EDGE_HARD_TRUNC_UNTOUCHED": {"role": "EVAL_ONLY", "unchanged": True},
            "EDGE_HARD_CLEAN_UNTOUCHED": {"role": "EVAL_ONLY", "unchanged": True},
            "CORNER_LA_*": {"role": "ABLATION_ONLY", "unchanged": True,
                            "why": "배제 사유가 코너 수가 아니라 효과 미확립이라 "
                                   "이번 변경과 무관"},
            "NEGATIVE_SYNTH_V1": {"role": "CALIBRATION_ONLY", "unchanged": True},
        },
        "still_true": [
            "dev / untouched 는 평가 전용 — 코너 제한 해제는 split 을 푸는 것이 아니다",
            "F3 는 여전히 Point PnP 초기화를 요구한다. 추론 시 코너가 없으면 "
            "pose 가 안 나온다 — 별개의 열린 문제",
            "이 변경으로 성능이 좋아진다는 증거는 아직 없다 (NOT_TESTED)",
        ],
    }

    # V2 학습 manifest
    rel_broad = (DA.RELEASE / "v2_prod40k_clean_merged").relative_to(DA.ROOT)
    src_edge = next(s for s in DA.POSITIVE_SOURCES
                    if s.dataset_id == "EDGE_HARD_TRUNC_TRAIN")
    rel_edge = src_edge.path.relative_to(DA.ROOT)
    total = len(broad) + len(edge_train)
    items = [{"dataset_id": "BROAD_40K", "frame_id": s,
              "frame_path": f"{rel_broad}/labels/{s}_label.json",
              "branch": "corner+line", "sampling_weight": 1.0 / total,
              "stratum": "BROAD"}
             for s in sorted(broad["frame_id"])]
    items += [{"dataset_id": "EDGE_HARD_TRUNC_TRAIN", "frame_id": s,
               "frame_path": f"{rel_edge}::labels/{s}_label.json",
               "branch": "corner+line", "sampling_weight": 1.0 / total,
               "stratum": "EDGE_TRUNC"}
              for s in sorted(edge_train["frame_id"])]
    manifest = {
        "manifest": "FINAL_SYNTH_TRAIN_V2",
        "supersedes": "FINAL_SYNTH_TRAIN_V1 (40,000)",
        "why": "EDGE_HARD_TRUNC_TRAIN 제외 해제 — DATA_ROLE_LOCK_V2 참조",
        "n_unique": len(items),
        "composition": {"BROAD_40K": int(len(broad)),
                        "EDGE_HARD_TRUNC_TRAIN": int(len(edge_train))},
        "edge_share": round(len(edge_train) / total, 4),
        "sampling": "uniform — EDGE 가 자연 비율(20.0%)로 들어간다. 별도 가중 없음. "
                    "비율을 조정하려면 그 근거를 따로 남길 것.",
        "MH_DEV_WARNING": "historical MH_DEV 6,242 가 BROAD 안에 포함돼 있다. "
                          "이 pool 로 학습한 checkpoint 로 MH_DEV 를 unseen 이라 "
                          "부르지 않는다.",
        "items": items,
    }
    text = json.dumps(manifest, indent=1, default=str)
    (DA.RELEASE_OUT / "FINAL_SYNTH_TRAIN_V2.json").write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (DA.RELEASE_OUT / "FINAL_SYNTH_TRAIN_V2.sha256").write_text(
        f"{digest}  FINAL_SYNTH_TRAIN_V2.json\n")
    decision["manifest"] = {"file": "FINAL_SYNTH_TRAIN_V2.json",
                            "n": len(items), "sha256": digest}
    (DA.RELEASE_OUT / "DATA_ROLE_LOCK_V2.json").write_text(
        json.dumps(decision, indent=1, default=str))

    print(f"  EDGE n_supervised>=4  {evidence['n_supervised_ge4_rate']:.1%}"
          f"   line_roles>=6 {evidence['line_valid_roles_ge6_rate']:.1%}")
    print(f"  FINAL_SYNTH_TRAIN_V2  {len(items):,} "
          f"(BROAD {len(broad):,} + EDGE {len(edge_train):,}, "
          f"EDGE {manifest['edge_share']:.1%})")
    print(f"  sha256 {digest}")


if __name__ == "__main__":
    main()
