"""상담용 표를 기존 artifact 에서만 조립한다.  새 추론 0 · 새 통계 0.

    python3 scripts/advising/build_consult_tables.py

숫자는 authoritative JSON 에서 읽는다.  산문에서 복사하지 않는다.
각 표에 모집단과 출처 파일을 같이 적는다 — 층이 다르면 모집단도 다르기 때문이다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_docs/advising/2026-09-professor-consult/tables"
CLOSURE = ROOT / "data/pallet/results/paper_pose_metric_closure_v1"
ARMS = ROOT / "data/pallet/results/paper_eval_v1/arms"
FRAMING = ROOT / "data/pallet/results/paper_framing_closure_v1"
SELF1 = ROOT / "data/pallet/results/paper_selftrain_v1"

# 상담용 라벨 — teacher/student 대신 무엇을 한 모델인지 그대로 쓴다
ORDER = [("R0", "R0  합성만 학습"),
         ("R0_CONT", "R0-CONT  추가학습만 (ST 없음)"),
         ("R1_NAIVE", "R1  ST 필터없음"),
         ("R2_CONF", "R2  ST 신뢰도"),
         ("R3_CONF_REPROJ", "R3  ST +재투영"),
         ("R4_CONF_REMOVE", "R4  ST +코너제거"),
         ("R5_PROPOSED", "R5  ST 전체일관성 (제안)")]
LABEL_WIDTH = 32


def wide(text: str) -> int:
    """한글·전각 문자는 터미널에서 2 칸을 차지한다."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(text))


def pad(text: str, width: int) -> str:
    """표시폭 기준 왼쪽 정렬 — 한글이 섞여도 열이 안 밀린다."""
    return str(text) + " " * max(0, width - wide(text))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    sources = json.loads(
        (ROOT / "_docs/paper/final/PAPER_CANONICAL_NUMBER_SOURCES.json").read_text())
    stat = json.loads((FRAMING / "PAPER_STATIC_STAT_AUDIT.json").read_text())
    ranking = stat["G1_ranking_uncertainty"]["arms"]
    paired = stat["G1_ranking_uncertainty"]["paired_R5_minus_R0"]

    def two_d(arm):
        return json.loads((ARMS / f"{arm}.json").read_text())["metrics"]["box_and_keypoint_2d"]

    def six_d(arm):
        return json.loads((CLOSURE / f"POSE_EVALUATION_{arm}.json").read_text())

    # ---------------------------------------------------------- TABLE A : 6D
    lines = ["# Table A — downstream 6D pose (main comparison)", "",
             "population   PAPER_EVAL positive 319 (plastic 194 + wood 125), role DEV",
             "reference    geometry-reconstructed 6D reference pose "
             "(manual 2D keypoints + intrinsics + registered dimensions)",
             "source       data/pallet/results/paper_pose_metric_closure_v1/"
             "POSE_EVALUATION_<ARM>.json, key paths.MAIN.ALL", "",
             "```text",
             f"{'arm':32}{'PoseCov ↑':>11}{'AxisAcc ↑':>11}{'R med ↓':>10}"
             f"{'Yaw med ↓':>11}{'t cm ↓':>9}{'IoU3D ↑':>10}{'ADDsym ↑':>11}",
             "-" * 105]
    for arm, label in ORDER:
        block = six_d(arm)["paths"]["MAIN"]["ALL"]
        lines.append(f"{pad(label, 32)}{block['n'] / 319:11.3f}{block['axis_accuracy']:11.4f}"
                     f"{block['rotation_median_deg']:10.3f}{block['yaw_median_deg']:11.3f}"
                     f"{block['translation_median_cm']:9.3f}{block['iou3d_median']:10.5f}"
                     f"{block['add_sym_auc']:11.5f}")
    boot = sources["pose_bootstrap"]
    lines += ["```", "",
              "How to read it in the meeting",
              "",
              f"- {boot['metric_blocks']} metric blocks over {boot['comparisons']} paired "
              f"comparisons against R0.",
              f"- session-cluster intervals excluding zero, improvement direction: "
              f"**{boot['session_cluster_excluding_zero_in_improvement_direction']}**",
              f"- session-cluster intervals excluding zero, any direction: "
              f"{boot['session_cluster_excluding_zero']}",
              "- With 13 recording groups an interval containing zero means the data cannot",
              "  separate the arms — not that the arms are equal.",
              "- R0_CONT solves 318 of 319 frames (PoseCov 0.997); every other arm solves 319.",
              "",
              "Paper role: **main table**. 6D is reportable; a 6D improvement is not claimed."]
    (OUT / "TABLE_A_6D_MAIN.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------- TABLE B : detection / ranking / 2D
    lines = ["# Table B — detection, ranking and fine 2D localisation", "",
             "population   PAPER_EVAL 319 positive + 2,689 negative, role DEV",
             "source       data/pallet/results/paper_eval_v1/arms/<ARM>.json",
             "             ranking CIs: paper_framing_closure_v1/PAPER_STATIC_STAT_AUDIT.json",
             "", "```text",
             f"{'arm':32}{'AP50-95 ↑':>11}{'AP50 ↑':>9}{'AUROC ↑':>10}"
             f"{'FPR95 ↓':>10}{'kp med px ↓':>13}{'kp p90 px ↓':>13}",
             "-" * 99]
    for arm, label in ORDER:
        m, r = two_d(arm), ranking[arm]
        lines.append(f"{pad(label, 32)}{m['box_ap50_95']:11.4f}{m['box_ap50']:9.4f}"
                     f"{r['auroc']:10.4f}{r['fpr95']:10.4f}"
                     f"{m['keypoint_location_median_px']:13.4f}"
                     f"{m['keypoint_location_p90_px']:13.2f}")
    lines += ["```", "",
              "How to read it in the meeting", "",
              "- Ranking is the one axis where the proposed filter is best of all arms.",
              f"- paired R5 - R0 AUROC {paired['auroc_delta']:+.5f} "
              f"frame CI [{paired['auroc_frame_CI95'][0]:+.6f}, "
              f"{paired['auroc_frame_CI95'][1]:+.6f}] — excludes zero.",
              f"- paired R5 - R0 FPR95 {paired['fpr95_delta']:+.5f} "
              f"frame CI [{paired['fpr95_frame_CI95'][0]:+.5f}, "
              f"{paired['fpr95_frame_CI95'][1]:+.5f}] — contains zero.",
              "- The session-clustered ranking interval is NOT computable: negative rows",
              "  carry no session identifier. The frame-level interval was computed after",
              "  the point estimate was seen, so it is Tier B.",
              "- Fine 2D localisation: no arm is below R0's 6.6157 px. That is the",
              "  paper's central negative result.",
              "",
              "Paper role: **main table** (detection/ranking/2D)."]
    (OUT / "TABLE_B_DETECTION_RANKING_2D.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------- TABLE C : material / lighting
    lines = ["# Table C — 6D by material and by lighting", "",
             "population   the same 319 frames, split by the manifest's own fields",
             "source       POSE_EVALUATION_<ARM>.json, keys paths.MAIN.<subgroup>", "",
             "★ subgroups are descriptive. They never select the method.", "",
             "```text",
             f"{'arm':32}{'subgroup':12}{'n':>5}{'R med ↓':>10}{'t cm ↓':>9}"
             f"{'IoU3D ↑':>10}{'ADDsym ↑':>11}",
             "-" * 89]
    for arm, label in (("R0", "R0  합성만 학습"),
                       ("R5_PROPOSED", "R5  ST 전체일관성 (제안)")):
        paths = six_d(arm)["paths"]["MAIN"]
        for key in ("plastic", "wood", "daytime", "nighttime"):
            block = paths.get(key)
            if not block:
                continue
            lines.append(f"{pad(label, 32)}{key:12}{block['n']:5d}"
                         f"{block['rotation_median_deg']:10.3f}"
                         f"{block['translation_median_cm']:9.3f}"
                         f"{block['iou3d_median']:10.5f}{block['add_sym_auc']:11.5f}")
    lines += ["```", "",
              "How to read it in the meeting", "",
              "- The manifest labels lighting for only 120 of the 319 frames",
              "  (daytime 70, nighttime 50). The other 199 are unlabelled and are not",
              "  guessed from session names.",
              "- Nighttime here is 50 frames and plastic-only. It is a different subgroup",
              "  from the 106-frame lighting_night group used elsewhere — never mix them.",
              "",
              "Paper role: **appendix**."]
    (OUT / "TABLE_C_MATERIAL_LIGHTING.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------- TABLE D : uncertainty
    bootstrap = json.loads((CLOSURE / "POSE_PAIRED_BOOTSTRAP.json").read_text())
    lines = ["# Table D — is the ranking of the arms stable?", "",
             "population   319 frames in 13 recording groups",
             "source       POSE_PAIRED_BOOTSTRAP.json, 10,000 resamples, seed 20260903", "",
             "```text",
             f"{'contrast':30}{'metric':16}{'delta':>10}{'session CI95':>26}{'excl 0':>8}",
             "-" * 90]
    ARM_SHORT = {"R0_CONT": "R0-CONT 추가학습만", "R1_NAIVE": "R1 필터없음",
                 "R2_CONF": "R2 신뢰도", "R3_CONF_REPROJ": "R3 +재투영",
                 "R4_CONF_REMOVE": "R4 +코너제거", "R5_PROPOSED": "R5 제안"}
    for comparison in bootstrap["comparisons"]:
        for key in ("iou3d", "add_sym_m"):
            m = comparison["metrics"][key]
            s = m["session_cluster"]
            interval = "[{:+.4f}, {:+.4f}]".format(s["low"], s["high"])
            short = ARM_SHORT.get(comparison["arm"], comparison["arm"])
            lines.append(
                f"{pad(short + ' - R0', 30)}{m['title'][:15]:16}"
                f"{m['observed_difference']:10.4f}{interval:>26}"
                f"{str(s['excludes_zero']):>8}")
    lines += ["```", "",
              "How to read it in the meeting", "",
              "- Not one session-cluster interval excludes zero, in either direction.",
              "- Every point estimate is negative, but the spread across sessions is larger",
              "  than the spread across arms, so the ordering of the arms is not stable.",
              "- The honest sentence is 'this data cannot separate them', not 'they differ'.",
              "",
              "Paper role: **main table companion** (uncertainty for Table A)."]
    (OUT / "TABLE_D_UNCERTAINTY.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------- TABLE E : filter quality
    quality = json.loads((SELF1 / "M4_FILTER_QUALITY.json").read_text())
    lines = ["# Table E — what the filters actually keep (label quality, not student)", "",
             f"population   {quality['population']}, {quality['n_frames']} frames "
             f"(these HAVE manual GT)",
             f"criterion    {quality['criterion']['CORRECT_2D']}",
             "source       data/pallet/results/paper_selftrain_v1/M4_FILTER_QUALITY.json",
             "", "★ 이 표는 **라벨 품질**이다. 학생 모델 성능이 아니다. 둘을 같은 표에 섞지 않는다.",
             "", "```text",
             f"{'filter':24}{'kept':>6}{'retain':>8}{'pass med px ↓':>15}"
             f"{'reject med px ↑':>17}{'precision ↑':>13}{'recall ↑':>10}",
             "-" * 93]
    SHORT = {"F0_NAIVE": "F0 no filter", "F1_CONF": "F1 confidence",
             "F2_CONF_REPROJ": "F2 + reprojection", "F3_CONF_REMOVE": "F3 + kp removal",
             "F5_CONF_FLIP": "F5 + flip", "F4_PROPOSED": "F4 proposed (all)"}
    for key, block in quality["filters"].items():
        p, r = block["pass"], block["reject"]
        reject = "—" if r["median_px"] is None else f"{r['median_px']:.3f}"
        lines.append(f"{SHORT.get(key, key):24}"
                     f"{block['accepted']:6d}{block['retention']:8.3f}"
                     f"{p['median_px']:15.3f}{reject:>17}"
                     f"{block['precision']:13.4f}{block['recall']:10.4f}")
    lines += ["```", "",
              "How to read it in the meeting", "",
              "- The filters do work at the label level: the proposed filter raises the",
              "  kept-label median error and separates pass from reject.",
              "- That improvement does not appear in the student (Table B) or in the",
              "  downstream pose (Table A). That gap is the paper.",
              "",
              "Paper role: **main table** (the label-quality half of the story)."]
    (OUT / "TABLE_E_FILTER_QUALITY.md").write_text("\n".join(lines) + "\n")

    # ------------------------------------------- TABLE F : 다른 모델과의 비교
    summary = json.loads((ARMS / "ARM_RESULTS.json").read_text())["models"]
    CROSS = [("R0", "R0  합성만 학습 (YOLO26n)"),
             ("R5_PROPOSED", "R5  ST 전체일관성 (제안)"),
             ("DOPE", "DOPE  합성만 학습 (VGG19)"),
             ("REALFT_LV1V2", "실GT FT  legacy v1v2"),
             ("REALFT_A", "실GT FT  A (real157+neg259+synth12k)"),
             ("REALFT_B", "실GT FT  B (patience0 40ep)")]
    lines = ["# Table F — 다른 모델과의 비교 (같은 평가 계약)", "",
             "population   PAPER_EVAL 319 positive + 2,689 negative — 위 표들과 **같은 셋**",
             "evaluator    동일.  그래서 Table A/B 와 같은 줄에 놓고 읽어도 된다",
             "source       data/pallet/results/paper_eval_v1/arms/<ARM>.json + ARM_RESULTS.json",
             "", "```text",
             f"{pad('model', 38)}{'det@.5 ↑':>11}{'AP50-95 ↑':>11}{'AUROC ↑':>10}"
             f"{'FPR95 ↓':>10}{'kp med px ↓':>13}{'gross20 ↓':>11}",
             "-" * 104]
    for arm, label in CROSS:
        m = summary.get(arm)
        if not m:
            continue
        g = m.get("subgroups", {}).get("ALL", {})
        f = lambda v, spec=".4f": "—" if v is None else format(v, spec)
        lines.append(f"{pad(label, 38)}{f(g.get('detection_rate_iou50')):>11}"
                     f"{f(m.get('box_ap50_95')):>11}{f(g.get('auroc')):>10}"
                     f"{f(g.get('fpr95')):>10}{f(g.get('corner_median_px'), '.3f'):>13}"
                     f"{f(g.get('gross_rate')):>11}")
    lines += ["```", "",
              "★ 읽을 때 반드시 붙일 단서", "",
              "- DOPE 는 box head 가 없다. AP 는 검출된 코너의 bounding box 에서 유도한 값이라",
              "  YOLO 의 학습된 box AP 와 **같은 양이 아니다**. score 도 belief peak vs box",
              "  confidence 라 AUROC/FPR95 척도가 다르다. 직접 비교가 성립하는 열은",
              "  **kp med px 와 det** 뿐이다.",
              "- REALFT 3 개는 **실제 manual GT 로 학습**했다. 통제된 비교가 아니라 상한선이며,",
              "  논문에서 unlabeled adaptation arm 과 같은 열 블록에 넣지 않는다.",
              "- REALFT_LV1V2 는 실제 GT 로 학습했는데도 kp 9.324 px 로 R0 보다 나쁘다 —",
              "  '실제 라벨이면 무조건 낫다' 가 아니라는 반례다.",
              "- 6D 수치는 이 표에 없다. POSE_EVALUATION_* 는 R0~R5 일곱 arm 만 존재한다.",
              "  ARM_RESULTS 의 pose_status BLOCKED 는 pose closure 이전의 옛 표기다.",
              "",
              "Paper role: **appendix / baseline reference**."]
    (OUT / "TABLE_F_OTHER_MODELS.md").write_text("\n".join(lines) + "\n")

    # --------------------------- TABLE G : architecture + 속도 (다른 모집단)
    arch = ROOT / ("challenge/yolo_pose_one_model/runs_arch_baseline/"
                   "ARCHITECTURE_BASELINE_TABLE.md")
    lines = ["# Table G — backbone 비교와 속도 (★ 다른 모집단)", "",
             "★★ **이 표는 Table A/B/F 와 모집단이 다르다.** real n=128, 30 epoch 이고",
             "   PAPER_EVAL 319 가 아니다.  같은 줄에 놓고 비교하면 안 된다.",
             "",
             "source   challenge/yolo_pose_one_model/runs_arch_baseline/"
             "ARCHITECTURE_BASELINE_TABLE.md",
             "         (동일 recipe·동일 evaluator 로 세 backbone 을 G38 합성만으로 30ep 학습)",
             ""]
    body = arch.read_text().splitlines()
    def block(header):
        try:
            i = next(k for k, l in enumerate(body) if l.strip().startswith(header))
        except StopIteration:
            return []
        out, started = [], False
        for l in body[i:]:
            if l.startswith("```"):
                if started:
                    out.append(l); break
                started = True
            if started:
                out.append(l)
        return out
    lines += ["## real n=128 (실제 성능)", ""] + block("## real n=128")[0:1] + \
             [l for l in block("## real n=128")[1:] if "ALL" in l or l.startswith(("model", "-", "```"))]
    lines += ["", "## 속도 (RTX 3080, batch1, imgsz640, warmup 30 / run 200)", ""] + \
             block("## efficiency")
    lines += ["", "★ 읽을 때 붙일 단서", "",
              "- 합성 val 에서는 세 backbone 이 거의 같다(poseMAP 0.9006~0.9059).",
              "  차이는 real 에서만 나고, 축마다 승자가 다르다 —",
              "  cbox 는 11n(0.836), kp median 은 v8n(12.13), kp p90 은 26n(69.71) 이 최고다.",
              "  **단일 승자가 없다.**",
              "- 속도는 **RTX 3080 workstation** 값이다.  Jetson 값이 아니다.",
              "- n=128 은 FT 누수 12 장을 뺀 셋이며 PAPER_EVAL 319 와 다르다.",
              "",
              "Paper role: **appendix**.  상담에서는 '속도 이야기의 출발점' 으로만 쓴다."]
    (OUT / "TABLE_G_ARCHITECTURE_SPEED.md").write_text("\n".join(lines) + "\n")

    manifest = {
        "schema_version": "advising_tables_v1", "generated_utc": stamp,
        "new_inference": 0, "new_training": 0, "new_statistic": 0,
        "tables": {
            "TABLE_A_6D_MAIN.md": {"population": "PAPER_EVAL 319",
                                   "source": "POSE_EVALUATION_*.json paths.MAIN.ALL"},
            "TABLE_B_DETECTION_RANKING_2D.md": {"population": "319 pos + 2,689 neg",
                                                "source": "arms/<ARM>.json + PAPER_STATIC_STAT_AUDIT.json"},
            "TABLE_C_MATERIAL_LIGHTING.md": {"population": "319, subgroups",
                                             "source": "POSE_EVALUATION_*.json paths.MAIN.<subgroup>"},
            "TABLE_D_UNCERTAINTY.md": {"population": "319 in 13 clusters",
                                       "source": "POSE_PAIRED_BOOTSTRAP.json"},
            "TABLE_E_FILTER_QUALITY.md": {"population": "PAPER_EVAL_PLASTIC_POS 194",
                                          "source": "M4_FILTER_QUALITY.json"},
            "TABLE_F_OTHER_MODELS.md": {"population": "PAPER_EVAL 319 + 2,689 neg",
                                        "source": "arms/ARM_RESULTS.json"},
            "TABLE_G_ARCHITECTURE_SPEED.md": {"population": "real n=128, 30ep — DIFFERENT",
                                              "source": "runs_arch_baseline/ARCHITECTURE_BASELINE_TABLE.md"},
        },
        "population_warning": "Table E is 194 plastic frames; Tables A-D are 319. "
                              "Do not compare a number across the two.",
    }
    (ROOT / "_docs/advising/2026-09-professor-consult/manifests/TABLES.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for name in sorted(manifest["tables"]):
        print(f"  wrote tables/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
