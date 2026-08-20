"""Master table and the written reports.

The role column is derived from three things that are kept visible: the label
contract (gates), the experimental record (SUPPORTED / NOT_ESTABLISHED /
REJECTED / NOT_TESTED), and the split status.  Nothing is inferred from a
directory name.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA  # noqa: E402

# experimental record, carried over from the audits that produced it
EVIDENCE = {
    "BROAD_40K": ("SUPPORTED",
                  "E3/E4 and every arm trained on it; only pool with"
                  " established positive training evidence"),
    "CORNER_LA_Y15_30": ("NOT_ESTABLISHED",
                         "canonical re-classification puts its target cell at"
                         " near-baseline (9.62px, x1.14) -- control, not fix"),
    "CORNER_LA_Y30_PLUS": ("NOT_ESTABLISHED",
                           "C1 / C1_RESCUE: two seeds disagree in direction,"
                           " CI includes 0"),
    "CORNER_LA_FRONTAL": ("NOT_TESTED",
                          "0 frames rendered; FRONTAL_DATA_DECISION.md records"
                          " TARGETED_ENRICHMENT_NOT_ESTABLISHED, RENDER 0"),
    "EDGE_HARD_TRUNC_TRAIN": ("NOT_TESTED",
                              "V2 (2026-08-20): 학습 편입. 배제 사유였던 V_vis<4 는 "
                              "범주 오류 — corner loss 는 n_supervised 를 감독하고 "
                              "99.2% 가 >=4. 성능 이득은 여전히 NOT_TESTED"),
    "EDGE_HARD_TRUNC_DEV": ("NOT_TESTED", "frozen dev split"),
    "EDGE_HARD_TRUNC_UNTOUCHED": ("NOT_TESTED",
                                  "declared holdout, unread by any experiment"),
    "EDGE_HARD_CLEAN_UNTOUCHED": ("NOT_TESTED",
                                  "declared holdout; gates pass, so it is the"
                                  " one add-on that can serve corner eval"),
    "NEGATIVE_SYNTH_V1_TRAIN": ("REJECTED",
                                "dense negative suppression REJECTED (seed2"
                                " pose safety); presence gate uses score_4kp"),
    "NEGATIVE_SYNTH_V1_DEV": ("SUPPORTED",
                              "presence calibration / evaluation population"),
}

ROLE = {
    "BROAD_40K": "MAIN_TRAIN",
    "CORNER_LA_Y15_30": "ABLATION_ONLY",
    "CORNER_LA_Y30_PLUS": "ABLATION_ONLY",
    "CORNER_LA_FRONTAL": "PRESERVE_UNUSED",
    "EDGE_HARD_TRUNC_TRAIN": "MAIN_TRAIN",
    "EDGE_HARD_TRUNC_DEV": "EVAL_ONLY",
    "EDGE_HARD_TRUNC_UNTOUCHED": "EVAL_ONLY",
    "EDGE_HARD_CLEAN_UNTOUCHED": "EVAL_ONLY",
    "NEGATIVE_SYNTH_V1_TRAIN": "CALIBRATION_ONLY",
    "NEGATIVE_SYNTH_V1_DEV": "EVAL_ONLY",
}


LEGACY = [
    ("LEGACY_mixed_v8_train", "data/pallet/training_data/mixed_v8_train", 9000,
     "2026-07-13 이전 세대. camera-facing 0123 여부 미확인"),
    ("LEGACY_v4_split_base", "data/pallet/training_data/v4_split_base", 4000,
     "v4 계열이나 현 release 와 라벨 스키마 다름"),
    ("LEGACY_paper_4pallet_mask_v1",
     "data/pallet/training_data/paper_4pallet_mask_v1", 10000,
     "mask 보유. 현 트랙 미사용"),
    ("LEGACY_aug_squash_v2", "data/pallet/training_data/aug_squash_v2", 2212, "aug 계열"),
    ("LEGACY_aug_trunc_v2", "data/pallet/training_data/aug_trunc_v2", 2971, "aug 계열"),
    ("LEGACY_aug_scale_v2", "data/pallet/training_data/aug_scale_v2", 1125, "aug 계열"),
    ("LEGACY_val", "data/pallet/training_data/val", 1500, "구 val"),
    ("LEGACY_achieve_all", "data/pallet/training_data/achieve (24 폴더)", 81233,
     "2026-07-13 이관. achieve/README 가 현 모델의 입력이 아니라고 선언"),
    ("LEGACY_pl_achieve", "data/pallet/pl/achieve (5 폴더)", 24484,
     "paper_base 시대 pseudo-label. projected_cuboid 가 len=9 로 규약이 다름"),
    ("LEGACY_paper_s2_pl_family",
     "data/pallet/training_data/paper_s2_{fullpool,pl,plrf}*", 1682,
     "gt_source=pseudo"),
    ("QUARANTINE_win_search2k",
     "data/pallet/transfer/win_search2k/pallet6d_v2_10k", 2429,
     "★그 폴더 README 가 사용 금지를 선언 — 잘못된 전제로 만든 추출물"),
    ("BROKEN_addon_v1_train_val",
     "challenge/data/02_synthetic/training/addon_v1_{train,val}", 0,
     "★symlink 12,000개 전부 dangling (2026-08-14 재편으로 원본 삭제)"),
]


def main():
    pos = pd.read_parquet(DA.AUDIT / "positive_frame_features_binned.parquet")
    neg = pd.read_parquet(DA.AUDIT / "negative_frame_features.parquet")
    policy = json.loads((DA.RELEASE_OUT / "SAMPLING_POLICY.json").read_text())
    leak = json.loads((DA.AUDIT / "DUPLICATE_LEAKAGE_AUDIT.json").read_text()) \
        if (DA.AUDIT / "DUPLICATE_LEAKAGE_AUDIT.json").exists() else {}

    rows = []
    for src in DA.POSITIVE_SOURCES + DA.NEGATIVE_SOURCES:
        did = src.dataset_id
        is_neg = did.startswith("NEGATIVE")
        block = neg[neg["dataset_id"] == did] if is_neg \
            else pos[pos["dataset_id"] == did]
        evidence, why = EVIDENCE.get(did, ("NOT_TESTED", ""))
        row = {
            "dataset_id": did,
            "path": str(src.path.relative_to(DA.ROOT)),
            "kind": src.kind,
            "exists": src.exists,
            "N_unique": int(len(block)),
            "positive_negative": "negative" if is_neg else "positive",
            "object_present": (False if is_neg else True) if src.exists else None,
            "pose_valid": (False if is_neg else True) if src.exists else None,
            "corner_supervision_valid": None if is_neg or block.empty else bool(
                block["gate_G1_vvis_ge4"].fillna(False).all()),
            "line_supervision_valid": None if is_neg or block.empty else bool(
                (block["line_valid_roles"].fillna(0) > 0).mean() > .99),
            "gate_all_pass_rate": None if is_neg or block.empty else round(
                float(block["gate_all_pass"].fillna(False).mean()), 4),
            "keypoint_convention": None if is_neg or block.empty else
                "|".join(sorted(block["keypoint_convention"].dropna()
                                .astype(str).unique())) or None,
            "yaw_convention": None if is_neg else
                "abs_frontal_yaw = 45 - facing_margin (dataset definition)",
            "camera_convention": "OpenCV pinhole, camera_data.intrinsics",
            "split_status": (
                "MH_TRAIN 33758 / MH_DEV 6242" if did == "BROAD_40K" else
                "train" if did.endswith("TRAIN") else
                "dev" if did.endswith("DEV") else
                "untouched holdout" if "UNTOUCHED" in did else "train-only ship"),
            "evidence": evidence,
            "evidence_note": why,
            "recommended_role": ROLE.get(did, "ARCHIVE"),
        }
        rows.append(row)

    # PHASE 0 asks for legacy/archive to be recorded with metadata and count
    # only.  They are listed here rather than crawled again: none of them is a
    # candidate for any manifest, and their convention is unverified.
    for did, path, n, note in LEGACY:
        rows.append({"dataset_id": did, "path": path, "kind": "dir",
                     "exists": True, "N_unique": n,
                     "positive_negative": "positive",
                     "evidence": "NOT_TESTED", "evidence_note": note,
                     "recommended_role": "ARCHIVE", "split_status": "n/a"})

    table = pd.DataFrame(rows)
    DA.AUDIT.mkdir(parents=True, exist_ok=True)
    table.to_csv(DA.AUDIT / "DATASET_MASTER_TABLE.csv", index=False)

    ratio = policy["policy"]["CHOSEN_RATIO"] or 0.0
    lines = ["# DATASET CONTRACT REPORT", "",
             "생성 0 / 학습 0. 라벨의 실제 계약과 실험 기록만으로 역할을 정했다.",
             "디렉터리 이름에서 역할을 추론하지 않았다.", "",
             "## 계약 표", "", "```",
             f"{'dataset':30}{'N':>7}  {'gate':>6}  {'corner':>6} {'line':>5}"
             f"  {'evidence':17} role", "-" * 108]
    for _, r in table.iterrows():
        gate = "-" if r["gate_all_pass_rate"] is None else \
            f"{100 * r['gate_all_pass_rate']:.0f}%"
        lines.append(
            f"{r['dataset_id']:30}{r['N_unique']:>7}  {gate:>6}  "
            f"{str(r['corner_supervision_valid']):>6} "
            f"{str(r['line_supervision_valid']):>5}  {r['evidence']:17} "
            f"{r['recommended_role']}")
    lines += ["```", "",
              "## 이름이 아니라 계약으로 갈린 것", "",
              "### EDGE_HARD 는 두 계약이 섞여 있다", "", "```",
              "trunc_train / trunc_dev / trunc_untouched   gate 0%   V_vis<=3 100%"
              "   corner 감독 불가",
              "clean_untouched                            gate 100%  V_vis>=4"
              "      corner 평가 가능", "```", "",
              "같은 `edge_complement_v1` 접두어를 쓰지만 CLEAN 쪽만 point-valid 다."
              " 이름으로 묶으면 틀린다.", "",
              "### BROAD 에는 V_vis<=3 프레임이 한 장도 없다", "",
              "G1 게이트가 `V_vis >= 4` 를 요구하므로 설계상 0 이다. EDGE 가 채우는"
              " 영역은 BROAD 의 희소 영역이 아니라 **BROAD 가 정의상 배제한 영역**이다.",
              "", "### 세 카운트는 서로 다르다 (실측)", "", "```",
              "n_inframe      투영이 화면 안 (계산)          최대 8",
              "V_actual       화면 안 + 자기폐색 아님 (라벨)  최대 7  <- 볼록 육면체는"
              " 항상 >=1개가 뒤에 가려진다",
              "V_vis_actual   추가로 외부 폐색 아님 (라벨)",
              "n_supervised   corner loss 가 실제로 학습하는 채널 = 9채널의 화면 안 판정",
              "```", "",
              "`V` 를 visible count 로 읽으면 안 된다는 지시가 데이터로 확인됐다.", ""]
    (DA.AUDIT / "DATASET_CONTRACT_REPORT.md").write_text("\n".join(lines))

    cov = pd.read_csv(DA.AUDIT / "CELL_COVERAGE.csv")
    status = cov.groupby(["dataset_id", "status"]).size().unstack(fill_value=0)
    cov_lines = ["# CELL COVERAGE REPORT", "",
                 "bin 경계와 UNDER/OVER 판정 규칙은 결과를 보기 전에 고정했다."
                 " 판정은 절대 개수가 아니라 **BROAD 가 같은 cell 에 준 비중** 대비다"
                 " (모든 cell 을 같은 N 으로 만들지 않는다).", "", "```",
                 status.to_string(), "```", "",
                 "## line-hard 영역", "", "```"]
    for cell in policy["policy"]["line_hard_cells"]:
        cov_lines.append(f"  {cell}")
    cov_lines += ["```", "",
                  "이 cell 들의 BROAD 프레임 수는 전부 **0** 이다. 그래서"
                  " '`BROAD 대비 2배`' 조항은 분모가 0 이라 어떤 비율에서도 통과한다"
                  " — 비율을 가르지 못한다. 이 사실을 통과로 적지 않고"
                  " `clause_1_is_vacuous=True` 로 기록했다.", ""]
    (DA.AUDIT / "CELL_COVERAGE_REPORT.md").write_text("\n".join(cov_lines))

    broad_train = int(((pos["dataset_id"] == "BROAD_40K")
                       & (pos["mh_split"] == "MH_TRAIN")).sum())
    edge_train = int((pos["dataset_id"] == "EDGE_HARD_TRUNC_TRAIN").sum())
    ev = policy["policy"]["evaluation"]
    rationale = f"""# DEPLOYMENT DATA RATIONALE

## [BASE]

`BROAD_40K MH_TRAIN` {broad_train:,} 장이 core 다. 이유는 크기가 아니라 증거다 —
E3/E4 를 포함해 지금까지 **효과가 확인된 학습은 전부 이 pool 위에서 났다**.
다른 어떤 pool 도 positive 학습 증거가 없다.

## [EXCLUDED FROM MAIN]

```
Y15_30      2,500   ABLATION_ONLY   canonical 재분류 시 겨냥 cell 이 near-baseline
                                    (9.62px, x1.14) — 개입이 아니라 control
Y30_PLUS    2,500   ABLATION_ONLY   C1/C1_RESCUE 에서 두 seed 방향 충돌, CI 0 포함
FRONTAL         0   PRESERVE_UNUSED 렌더 0장. FRONTAL_DATA_DECISION.md 가
                                    TARGETED_ENRICHMENT_NOT_ESTABLISHED 로 종결
NEGATIVE   10,000   CALIBRATION     dense negative training 은 REJECTED
                                    (seed2 pose safety 대실패). 최종 rejection 은
                                    score_4kp 이고 pose network 를 건드리지 않는다
```

CORNER_LA 두 세트는 **5,000 장이 있다는 이유로는 들어가지 않는다.** 효과가
미확립이고, 넣으면 그 효과를 분리할 수 없게 된다.

## [LINE HARD SUPPLEMENT]

사용 여부 = **candidate 로만**. main 승격 아님.

```
ratio         = {ratio}  ({policy['policy']['CHOSEN']})
근거          = {policy['policy']['CHOSEN_BASIS']}
```

후보 3개의 실제 값:

```
                ratio   broad mode 보존(min)   EDGE 반복(broad 1회당)
CONSERVATIVE    0.05    {ev['CONSERVATIVE']['broad_mode_retention_min']:.3f}                  {ev['CONSERVATIVE']['edge_repeat_vs_broad_pass']:.2f}x
BALANCED        0.12    {ev['BALANCED']['broad_mode_retention_min']:.3f}                  {ev['BALANCED']['edge_repeat_vs_broad_pass']:.2f}x
AGGRESSIVE      0.20    {ev['AGGRESSIVE']['broad_mode_retention_min']:.3f}                  {ev['AGGRESSIVE']['edge_repeat_vs_broad_pass']:.2f}x   <- 0.85 미달로 탈락
```

**어떤 cell 이 얼마에서 얼마로 늘어나는가** — 정직하게 쓰면 이렇다.

```
cell (V_vis<=3)                      BROAD    EDGE      노출 5% 적용 후
──────────────────────────────────────────────────────────────────────
('<=3','truncated','0.40-0.60')          0    2,841     0  ->  0.0142
('<=3','truncated','0.60-0.85')          0    2,759     0  ->  0.0138
('<=3','truncated','>=0.85')             0    2,202     0  ->  0.0110
('<=3','truncated','0.25-0.40')          0    1,614     0  ->  0.0081
('<=3','truncated','<0.25')              0      583     0  ->  0.0029
('<=3','full','0.25-0.40')               0        1     0  ->  0.0000
합                                       0   10,000     0  ->  0.0500
```

즉 "N 에서 N 으로 늘었다" 가 아니라 **0 에서 생겼다**. BROAD 는 G1 게이트가
`V_vis >= 4` 를 요구해 이 영역을 정의상 배제한다. 그래서 coverage 논거만으로는
5% 와 20% 중 무엇이 필요한지 **가릴 수 없다**. 5% 는 보호 조항(broad mode 보존
0.950, EDGE 반복 0.18x)과 보수적 기본값으로 고른 값이지, coverage 가 요구한
값이 아니다. 이 구분을 뭉개지 않는다.

broad 는 그대로 95% 를 유지하고, EDGE 10,000 장은 broad 1회 통과당 0.18배만
재사용되므로 **같은 프레임을 반복 노출해 exposure 만 늘리는 상태가 아니다**.

## [DIVERSITY]

```
axis               BROAD unique   혼합 후 unique   BROAD 최대비중   혼합 후
"""
    val = policy["validation"]["diversity_preserved"]
    for axis, d in val.items():
        rationale += (f"{axis:18} {d['broad_unique']:>12} {d['mixture_unique']:>16}"
                      f" {d['broad_top_share']:>15.3f} {d['mixture_top_share']:>9.3f}\n")
    rationale += f"""```

## [FINAL]

```
Corner training pool = BROAD_40K MH_TRAIN {broad_train:,}   (effective exposure 1.00)
Line   training pool = BROAD_40K MH_TRAIN {broad_train:,}   (effective exposure {1 - ratio:.2f})
                     + EDGE_HARD_TRUNC_TRAIN {edge_train:,}  (effective exposure {ratio:.2f})
```

corner stream 에 EDGE 는 **절대 들어가지 않는다** — G1 이 False 라 4개 미만
코너로 pose 를 가르치게 된다. 이건 성능 판단이 아니라 계약 위반이다.
"""
    (DA.RELEASE_OUT / "DEPLOYMENT_DATA_RATIONALE.md").write_text(rationale)

    summary = f"""# DATASET RELEASE SUMMARY

물리 concat 없음. manifest 만 만들었고 원본 pool 은 그대로다.

```
PAPER_CORE_V1_corner_manifest.json          {policy['PAPER_CORE_V1']['corner_n']:,}
PAPER_CORE_V1_line_manifest.json            {policy['PAPER_CORE_V1']['line_n']:,}
DEPLOYMENT_CANDIDATE_V1_corner_manifest.json {policy['DEPLOYMENT_CANDIDATE_V1']['corner_n']:,}
DEPLOYMENT_CANDIDATE_V1_line_manifest.json  {policy['DEPLOYMENT_CANDIDATE_V1']['line_n']:,}
SAMPLING_POLICY.json
checksums.sha256
```

## 학습에서 제외된 것

```
"""
    for key, value in policy["excluded_from_training"].items():
        summary += f"{key:34}{value:>8}\n"
    summary += f"""```

## 누수

```
HARD_BLOCK        {len(leak.get('HARD_BLOCK', []))}
LEAKAGE_CLEAN     {leak.get('LEAKAGE_CLEAN')}
```

MH_DEV 6,242 는 어떤 training manifest 에도 들어가지 않는다 (manifest 생성 시
`mh_split == "MH_TRAIN"` 으로 필터). EDGE dev/untouched, NEGATIVE dev 도 동일.
real test 는 이 감사에서 아예 건드리지 않았다.
"""
    (DA.RELEASE_OUT / "DATASET_RELEASE_SUMMARY.md").write_text(summary)
    print("-> DATASET_MASTER_TABLE.csv / DATASET_CONTRACT_REPORT.md"
          " / CELL_COVERAGE_REPORT.md / DEPLOYMENT_DATA_RATIONALE.md"
          " / DATASET_RELEASE_SUMMARY.md")


if __name__ == "__main__":
    main()
