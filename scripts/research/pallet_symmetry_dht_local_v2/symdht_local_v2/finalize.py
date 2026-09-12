"""Generate the small paper-closing records from immutable raw v2 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import immutable_json, read_json, sha256


def _cohorts(value):
    rows = value["records"]; difficulty = np.asarray([row["base_frame_mean_px"] for row in rows])
    gains = np.asarray([row["gain_frame_px"] for row in rows]); result = []
    for q, indices in enumerate(np.array_split(np.argsort(difficulty, kind="mergesort"), 4), 1):
        result.append({"quartile": q, "count": len(indices), "mean_baseline_frame_error_px": float(difficulty[indices].mean()),
                       "mean_method_gain_px": float(gains[indices].mean())})
    return result


def run(repo: Path, docs: Path, raw: Path):
    protocol_path = repo / "scripts/research/pallet_symmetry_dht_local_v2/PROTOCOL_LOCK.json"
    immutable_json(docs / "PROTOCOL_LOCK.json", read_json(protocol_path))
    starts = {}; checks = []
    for seed in (1, 2, 3):
        d = read_json(raw / f"heads/direct_seed{seed}/START.json"); h = read_json(raw / f"heads/hough_seed{seed}/START.json")
        dc = read_json(raw / f"heads/direct_seed{seed}/COMPLETION.json"); hc = read_json(raw / f"heads/hough_seed{seed}/COMPLETION.json")
        row = {"seed": seed, "steps_exact_2000": dc["steps"] == hc["steps"] == 2000,
               "both_complete": dc["complete"] and hc["complete"], "both_actual_cuda0": d["device_actual"] == h["device_actual"] == "cuda:0",
               "manifest_same": d["manifest_sha256"] == h["manifest_sha256"], "minibatch_order_same": d["minibatch_order_sha256"] == h["minibatch_order_sha256"],
               "common_init_same": d["common_parameter_sha256"] == h["common_parameter_sha256"], "utility_init_same": d["initial_utility_head_sha256"] == h["initial_utility_head_sha256"],
               "minibatch_order_sha256": d["minibatch_order_sha256"], "common_parameter_sha256": d["common_parameter_sha256"], "initial_utility_head_sha256": d["initial_utility_head_sha256"],
               "direct_checkpoint_sha256": dc["checkpoint_sha256"], "hough_checkpoint_sha256": hc["checkpoint_sha256"]}
        row["PASS"] = all(value for key, value in row.items() if key.endswith(("_same", "_complete", "_2000", "_cuda0")))
        checks.append(row)
    immutable_json(docs / "TRAINING_AUDIT.json", {"schema": "symdht_local_training_audit_v2", "PASS": all(row["PASS"] for row in checks), "rows": checks})
    names = ["point", "direct_seed1", "direct_seed2", "direct_seed3", "hough_seed1", "hough_seed2", "hough_seed3"]
    evaluations = {name: read_json(raw / f"evaluations/{name}_synth_val.json") for name in names}
    immutable_json(docs / "BASELINE_DIFFICULTY_COHORTS.json", {"schema": "symdht_local_baseline_difficulty_cohorts_v2",
                   "definition": "quartiles are equal-count ranks of baseline frame mean 8-corner error; gain is baseline minus method",
                   "arms": {name: _cohorts(value) for name, value in evaluations.items()}})
    summary = read_json(docs / "RESULT_SUMMARY.json"); oracle = read_json(docs / "ORACLE_DECOMPOSITION.json")
    point = summary["point"]; hough = summary["hough"]; direct = summary["direct"]
    def pct(value): return 100 * (point["primary_frame_mean_over_raw_diagonal"] - value["primary_frame_mean_over_raw_diagonal"]) / point["primary_frame_mean_over_raw_diagonal"]
    report = f"""# Symmetry-aware DHT local fusion v2 — 최종 bounded experiment

## 결론

사전 고정된 판정은 **`{summary['scientific_verdict']}`** 이다. Hough 세 seed 모두 Point 대비 primary 1% 개선 gate를 통과하지 못했고 median/P90도 모두 악화했다. 따라서 real DEV와 FINAL은 열지 않았으며 추가 DHT architecture/loss/seed/epoch/grid 탐색을 종료한다. C4는 데이터가 없어 `C4_NOT_EVALUATED`이다.

## 관찰

- Point: primary `{point['primary_frame_mean_over_raw_diagonal']:.9f}`, median `{point['pooled_symmetric_median_px']:.4f}px`, P90 `{point['pooled_symmetric_p90_px']:.4f}px`, coverage `{point['coverage_count']}`.
- Direct seed 1/2/3 primary 변화(Point 대비): `{pct(direct[0]):+.3f}%`, `{pct(direct[1]):+.3f}%`, `{pct(direct[2]):+.3f}%`.
- Hough seed 1/2/3 primary 변화(Point 대비): `{pct(hough[0]):+.3f}%`, `{pct(hough[1]):+.3f}%`, `{pct(hough[2]):+.3f}%` (양수가 개선). Hough median은 `{hough[0]['pooled_symmetric_median_px']:.4f}/{hough[1]['pooled_symmetric_median_px']:.4f}/{hough[2]['pooled_symmetric_median_px']:.4f}px`, P90은 `{hough[0]['pooled_symmetric_p90_px']:.4f}/{hough[1]['pooled_symmetric_p90_px']:.4f}/{hough[2]['pooled_symmetric_p90_px']:.4f}px이다.
- Hough good-point damage는 `{100*hough[0]['good_point_damage_rate']:.3f}%/{100*hough[1]['good_point_damage_rate']:.3f}%/{100*hough[2]['good_point_damage_rate']:.3f}%`; 새 catastrophic frame은 모두 0, coverage는 모두 Point와 동일하다.
- Hough continuous line endpoint error 평균은 `{hough[0]['continuous_line_endpoint_distance_px']['mean']:.3f}/{hough[1]['continuous_line_endpoint_distance_px']['mean']:.3f}/{hough[2]['continuous_line_endpoint_distance_px']['mean']:.3f}px`로 Direct보다 작았지만 point 개선으로 이어지지 않았다.
- Hough utility AUROC는 `{hough[0]['utility_calibration']['AUROC']:.3f}/{hough[1]['utility_calibration']['AUROC']:.3f}/{hough[2]['utility_calibration']['AUROC']:.3f}`, AUPRC는 `{hough[0]['utility_calibration']['AUPRC']:.3f}/{hough[1]['utility_calibration']['AUPRC']:.3f}/{hough[2]['utility_calibration']['AUPRC']:.3f}`이다. signed candidate gain과의 Spearman은 `{hough[0]['utility_calibration']['spearman_utility_vs_gain']:.3f}/{hough[1]['utility_calibration']['spearman_utility_vs_gain']:.3f}/{hough[2]['utility_calibration']['spearman_utility_vs_gain']:.3f}`로 음수였다.

## Oracle 원인 분해

GT를 사용하는 설명용 진단이며 배포 성능 주장이 아니다. primary는 baseline `{oracle['summary']['baseline']['frame_mean_over_raw_diagonal']:.9f}`에서 exact continuous GT line `{oracle['summary']['continuous_GT']['frame_mean_over_raw_diagonal']:.9f}`, 36×65 soft-quantized/decode GT line `{oracle['summary']['quantized_soft_GT']['frame_mean_over_raw_diagonal']:.9f}`, 기존 v1 seed1 predicted line `{oracle['summary']['v1_predicted']['frame_mean_over_raw_diagonal']:.9f}`가 됐다. GT-oracle utility가 있으면 세 단계 모두 baseline을 개선하므로 local geometry에 개선 가능성은 있었다. soft representation은 exact line보다 일부 손실이 있었고, predicted line에서 추가 손실이 있었다.

## 네 수정

1. anchor를 제거하고 corrected-point loss와 one-sided no-harm만 유지했다. 2/5/10px fixture에서 exact < partial < baseline을 확인했다.
2. structural support와 correction utility를 분리했다. utility는 detached selected line의 counterfactual gain이 0.25px를 초과하는지 감독하며 inference는 GT-free다.
3. 36×65 lattice는 그대로 두고 1-bin theta/rho bandwidth의 continuous soft target을 사용했다. hard-bin accuracy와 continuous raw-pixel line error를 별도 보고했다.
4. semantic edge당 MAP-local 한 mode만 WLS에 전달했다. 대안 mode들의 normal equation을 합산하지 않는다.

## 해석 범위

관찰된 결과는 네 수정 후에도 bounded synthetic gate가 실패했다는 뜻이다. 네 불일치가 과거 실패에 각각 얼마나 기여했는지, 또는 모든 가능한 DHT 설계가 실패한다는 인과 명제는 증명하지 않는다. 논문 결론은 현재 고정 데이터·예산·구조 범위에서 DHT line estimation 향상이 안정적인 point 개선으로 이어지지 않았다는 수준으로 제한한다.
"""
    code_diff = """# v1 → v2 code difference summary

- A: removed the independent linear anchor penalty; retained normalized corrected-point SmoothL1 and one-sided no-harm.
- B: replaced null/support proxy use gating with a separate common utility head trained from detached candidate counterfactual gain >0.25px.
- C: replaced nearest-bin hard CE with fixed one-bin-bandwidth continuous theta/rho soft targets; lattice remains 36×65.
- D: replaced simultaneous multi-mode normal-equation accumulation with one MAP-local alternative per semantic edge.
- Fairness: common stem and utility head use an arm-independent RNG stream; same-seed D/H state, manifest and minibatch-order SHAs are audited.
- Unchanged: stock R0 points/backbone/P3/P4 export, 1792/256/512 populations, C1/C2, AdamW, seeds 1/2/3, 2000 steps, batch 8, 1% shift cap, center8, GT-free scoring, and real-DEV gate.
- Not added: PCGrad, pose/translation solver or loss, candidate bank, beam search, fine-tuning, rendering, self-training, real labels, sweeps.
"""
    for path, text in ((docs / "REPORT_KO.md", report), (docs / "CODE_DIFF_SUMMARY.md", code_diff)):
        if path.exists(): raise FileExistsError(path)
        path.write_text(text)
    print(summary["scientific_verdict"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--docs", type=Path, required=True); parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args(); run(args.repo.resolve(), args.docs.resolve(), args.raw.resolve())


if __name__ == "__main__": main()
