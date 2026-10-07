#!/usr/bin/env python3
"""Publish existing verified results as a portable, read-only review packet.

No model imports, inference, training, human-label mutation or network calls.
Writes only beside this script. Original evidence files remain byte-identical.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
COMBINED = ROOT / "_docs/experiments/pallet_combined_closeout_20261003_v1"
CLOSE = COMBINED / "closeout_20261006_v1"
ASSIST = COMBINED / "pnp_assisted_lifter_20261006_v1"
STATIC = ROOT / "_docs/experiments/pallet_n3_static_closeout_v1"
COPIES: list[dict] = []
CELLS: list[dict] = []
SOURCE_TO_COPY: dict[str, str] = {}
START = time.perf_counter()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(name: str, value) -> None:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy(src: Path, dest: str) -> None:
    assert src.is_file(), src
    target = OUT / dest
    assert OUT in target.resolve().parents
    target.parent.mkdir(parents=True, exist_ok=True)
    before = sha(src)
    shutil.copyfile(src, target)
    assert sha(target) == before == sha(src)
    try:
        original = str(src.relative_to(ROOT))
    except ValueError:
        original = str(src)
    COPIES.append({"published_path": dest, "original_path": original,
                   "sha256": before, "bytes": target.stat().st_size,
                   "copy_equal": True, "original_preserved": True})
    SOURCE_TO_COPY[str(src)] = dest


def load(src: Path):
    return json.loads(src.read_text(encoding="utf-8"))


def read_csv(src: Path) -> list[dict]:
    with src.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(value, digits=3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def table(headers, rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def cell(src: Path, pointer: str, unit: str, denominator: dict,
         seeds=None, scale=1.0, digits=3, document="README_KO.md"):
    raw = load(src)
    for key in pointer.strip("/").split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        raw = raw[int(key)] if isinstance(raw, list) else raw[key]
    display_value = raw * scale if isinstance(raw, (int, float)) else raw
    CELLS.append({"document": document, "source_kind":"JSON", "source_copy": SOURCE_TO_COPY[str(src)],
                  "original_source": str(src.relative_to(ROOT)), "source_sha256": sha(src),
                  "json_pointer": pointer, "raw_value": raw, "scale": scale,
                  "unit": unit, "denominator": denominator, "seeds": seeds,
                  "display": fmt(display_value, digits)})
    return display_value


def csv_cell(src: Path, row_index: int, column: str, unit: str,
             denominator: dict, seeds=None, digits=3):
    raw = read_csv(src)[row_index][column]
    numeric = None if raw in ["", "NA"] else float(raw)
    CELLS.append({"document":"README_KO.md", "source_kind":"CSV", "source_copy":SOURCE_TO_COPY[str(src)],
                  "original_source":str(src.relative_to(ROOT)),"source_sha256":sha(src),
                  "data_row_zero_based":row_index,"column":column,"raw_value":raw,
                  "unit":unit,"denominator":denominator,"seeds":seeds,"display":fmt(numeric,digits)})
    return numeric


def rmetric(src, backbone, method, group, key, unit, scale=1):
    item = load(src)["backbones"][backbone][method]
    m = item["result"][group]
    denominator = {"frames": m["frames"],
                   "reference_corners": m["full_supervised_corners"],
                   "observed_corners": m["observed_corners"],
                   "pose_frames": m["pose_available_frames"]}
    return cell(src, f"/backbones/{backbone}/{method}/result/{group}/{key}",
                unit, denominator, item["raw_methods"], scale=scale)


def pair(src, b, group, key, unit, scale=1):
    return " → ".join(fmt(rmetric(src, b, m, group, key, unit, scale))
                       for m in ["Base", "N3"])


def write_md(name: str, text: str):
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def main():
    # Immutable audit and measurement evidence. Historical subgroup labels are
    # preserved in JSON but never presented as the current user-entered labels.
    mapping = {
        CLOSE / "static/STATIC_REAGGREGATION.json": "evidence/static/STATIC_REAGGREGATION.json",
        CLOSE / "static/DEV319_HEADLINE_AND_SEED.csv": "evidence/static/DEV319_HEADLINE_AND_SEED.csv",
        CLOSE / "static/COMPARATOR_REAGGREGATION.json": "evidence/static/COMPARATOR_REAGGREGATION.json",
        CLOSE / "static/COMPARATOR319_HEADLINE_AND_SEED.csv": "evidence/static/COMPARATOR319_HEADLINE_AND_SEED.csv",
        CLOSE / "static/STUDENT128_REAGGREGATION.json": "evidence/static/STUDENT128_REAGGREGATION.json",
        CLOSE / "static/STUDENT128_HEADLINE_AND_SEED.csv": "evidence/static/STUDENT128_HEADLINE_AND_SEED.csv",
        CLOSE / "static/SQUARE_REAGGREGATION.json": "evidence/static/SQUARE_REAGGREGATION.json",
        CLOSE / "static/SQUARE119_HEADLINE_AND_SEED.csv": "evidence/static/SQUARE119_HEADLINE_AND_SEED.csv",
        CLOSE / "static/SOURCE_BINDINGS.json": "evidence/static/SOURCE_BINDINGS.json",
        CLOSE / "static/LABEL_PROVENANCE_AUDIT.json": "evidence/static/LABEL_PROVENANCE_AUDIT.json",
        CLOSE / "static/STATIC_INVARIANCE_CHECK.json": "evidence/static/STATIC_INVARIANCE_CHECK.json",
        CLOSE / "static/EXECUTION_COST_LEDGER.json": "evidence/static/EXECUTION_COST_LEDGER.json",
        CLOSE / "visibility_square/VISIBILITY_RESULTS.json": "evidence/visibility/VISIBILITY_RESULTS.json",
        CLOSE / "visibility_square/VISIBILITY_RESULTS.csv": "evidence/visibility/VISIBILITY_RESULTS.csv",
        CLOSE / "visibility_square/VISIBILITY_PER_SEED.csv": "evidence/visibility/VISIBILITY_PER_SEED.csv",
        CLOSE / "visibility_square/SQUARE119_RESULTS.json": "evidence/square/SQUARE119_RESULTS.json",
        CLOSE / "visibility_square/SQUARE119_RESULTS.csv": "evidence/square/SQUARE119_RESULTS.csv",
        CLOSE / "visibility_square/SQUARE119_PER_SEED.csv": "evidence/square/SQUARE119_PER_SEED.csv",
        CLOSE / "visibility_square/SQUARE119_COMPARISONS.csv": "evidence/square/SQUARE119_COMPARISONS.csv",
        CLOSE / "visibility_square/HISTORICAL_SQUARE150.json": "evidence/square/HISTORICAL_SQUARE150.json",
        CLOSE / "visibility_square/HISTORICAL_SQUARE150.csv": "evidence/square/HISTORICAL_SQUARE150.csv",
        CLOSE / "visibility_square/EXECUTION_VALIDATION.json": "evidence/audits/VISIBILITY_SQUARE_VALIDATION.json",
        CLOSE / "visibility_square/SEMANTIC_TEST_VALIDATION.json": "evidence/audits/VISIBILITY_SEMANTIC_VALIDATION.json",
        CLOSE / "visibility_square/ACTUAL_CPU_COST_LEDGER.json": "evidence/audits/VISIBILITY_CPU_COST_LEDGER.json",
        CLOSE / "REUSED_RESULT_AUDIT.json": "evidence/audits/REUSED_RESULT_AUDIT.json",
        CLOSE / "final_review/INDEPENDENT_PAPER_AUDIT.json": "evidence/audits/INDEPENDENT_PAPER_AUDIT.json",
        CLOSE / "final_review/SUBGROUP_REVIEW.json": "evidence/static/SUBGROUP_REVIEW.json",
        CLOSE / "paper_patch/PAPER_CELL_MAP.json": "evidence/paper/STATIC_PAPER_CELL_MAP.json",
        CLOSE / "paper_patch/CLOSEOUT_VALIDATION.json": "evidence/paper/STATIC_CLOSEOUT_VALIDATION.json",
        CLOSE / "paper_patch/INTEGRATED.patch": "evidence/paper/STATIC_INTEGRATED.patch",
        CLOSE / "paper_patch/PAPER_GAP_MATRIX.md": "evidence/paper/STATIC_PAPER_GAP_MATRIX.md",
        STATIC / "N0_N1_POSE_RESULTS.json": "evidence/pose/N0_N1_POSE_RESULTS.json",
        STATIC / "N0_N1_POSE_RESULTS.csv": "evidence/pose/N0_N1_POSE_RESULTS.csv",
        STATIC / "PAIRED_POSE_ANALYSIS.json": "evidence/pose/PAIRED_POSE_ANALYSIS.json",
        STATIC / "PAIRED_POSE_ANALYSIS.csv": "evidence/pose/PAIRED_POSE_ANALYSIS.csv",
        STATIC / "ABLATION_POSE_UNCERTAINTY.json": "evidence/pose/ABLATION_POSE_UNCERTAINTY.json",
        STATIC / "PAIRED_2D_UNCERTAINTY.json": "evidence/pose/PAIRED_2D_UNCERTAINTY.json",
        STATIC / "POSE_TRACE_AND_TAIL.json": "evidence/pose/POSE_TRACE_AND_TAIL.json",
        STATIC / "RESNET_EFFECTIVE_PROTOCOL.json": "evidence/contracts/RESNET_EFFECTIVE_PROTOCOL.json",
        STATIC / "PROTOCOL_CORRECTIONS.md": "evidence/contracts/PROTOCOL_CORRECTIONS.md",
        STATIC / "TRAINING_REUSE_VERIFICATION.json": "evidence/contracts/TRAINING_REUSE_VERIFICATION.json",
        STATIC / "SOURCE_BINDINGS.json": "evidence/contracts/STATIC_ORIGINAL_SOURCE_BINDINGS.json",
        STATIC / "RUNTIME_PANEL.json": "evidence/runtime/RUNTIME_PANEL.json",
        STATIC / "FIGURE_PROVENANCE.json": "evidence/figures/FIGURE_PROVENANCE.json",
        STATIC / "EXAMPLE_SELECTION.json": "evidence/figures/EXAMPLE_SELECTION.json",
        STATIC / "figures/backbone_results.png": "images/backbone_results.png",
        STATIC / "figures/pose_uncertainty.png": "images/pose_uncertainty.png",
        STATIC / "figures/static_examples.png": "images/static_examples.png",
        CLOSE / "paper_updated/figures/current_review_summary.png": "images/current_review_summary.png",
        CLOSE / "paper_updated/figures/lifter_prediction_timeseries.png": "images/lifter_prediction_timeseries.png",
        COMBINED / "lifter/LIFTER_CONTINUITY_SUMMARY.json": "evidence/lifter/LIFTER_CONTINUITY_SUMMARY.json",
        COMBINED / "lifter/LIFTER_CONTINUITY_SUMMARY.csv": "evidence/lifter/LIFTER_CONTINUITY_SUMMARY.csv",
        COMBINED / "lifter/VALIDATION_RECEIPT.json": "evidence/lifter/CONTINUITY_VALIDATION_RECEIPT.json",
        ASSIST / "RESULT.json": "evidence/lifter/ASSISTED_RESULT.json",
        ASSIST / "POINT_ERRORS.csv": "evidence/lifter/ASSISTED_POINT_ERRORS.csv",
        ASSIST / "FRAME_RESULTS.json": "evidence/lifter/ASSISTED_FRAME_RESULTS.json",
        ASSIST / "EVALUATION_STATUS.json": "evidence/lifter/ASSISTED_EVALUATION_STATUS.json",
        ASSIST / "TEST_VALIDATION.json": "evidence/lifter/ASSISTED_TEST_VALIDATION.json",
        ASSIST / "PAPER_INTEGRATION_RECEIPT.json": "evidence/paper/ASSISTED_PAPER_INTEGRATION_RECEIPT.json",
        ASSIST / "paper_patch/ASSISTED_PAPER.patch": "evidence/paper/ASSISTED_PAPER.patch",
        ASSIST / "paper_patch/VALIDATION.json": "evidence/paper/ASSISTED_PAPER_VALIDATION.json",
        ASSIST / "paper_updated/evidence/ASSISTED_PAPER_CELL_MAP.json": "evidence/paper/ASSISTED_PAPER_CELL_MAP.json",
        ASSIST / "paper_updated/evidence/ASSISTED_SOURCE_MANIFEST.json": "evidence/paper/ASSISTED_SOURCE_MANIFEST.json",
    }
    # Only stable result tables are copied from the older retained CSV snapshot.
    for n in ["tab_backbones", "tab_pose", "tab_cost", "tab_comparators", "tab_update_alternative",
              "tab_square_manual_declared", "tab_square_manual_in_frame", "tab_pose_uncertainty"]:
        mapping[CLOSE / f"paper_updated/evidence/github_tables/{n}.csv"] = f"evidence/tables/{n}.csv"
    for src, dest in mapping.items():
        copy(src, dest)
    for src in sorted((ASSIST / "images").glob("*.png")):
        copy(src, "images/lifter_12/" + src.name)
    assert len(list((OUT / "images/lifter_12").glob("*.png"))) == 12

    # Bring the actual completed training receipts, with their pre-existing hashes.
    for head in load(STATIC / "TRAINING_REUSE_VERIFICATION.json")["heads"]:
        receipt = head["receipt"]
        src = Path(receipt["path"])
        assert sha(src) == receipt["sha256"]
        copy(src, "evidence/contracts/" + src.name)
    copy(ROOT / "_docs/experiments/pallet_dim_conditioned_p_v1/PAPER_TRAINING_COMPLETE.json",
         "evidence/contracts/YOLO_PAPER_TRAINING_COMPLETE.json")

    # Exact Markdown manuscript copies with the original relative image layout.
    for name in ["manuscript_ko.md", "supplement_ko.md"]:
        src = ASSIST / "paper_updated" / name
        copy(src, "paper_text/" + name)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", src.read_text()):
            if target.startswith("figures/"):
                figure = src.parent / target
                dest="paper_text/"+target
                if not any(c["published_path"]==dest for c in COPIES):
                    copy(figure,dest)

    src = CLOSE / "static/STATIC_REAGGREGATION.json"
    stats = load(src)
    assert stats["label_counts"] == {"clean": 153, "moderate": 92, "severe": 74}
    names = {"yolo": "YOLO", "dope": "DOPE", "resnet18": "ResNet-18"}
    headlines, grades, poses, seeds = [], [], [], []
    for b in names:
        base = stats["backbones"][b]["Base"]["result"]["all"]
        after = stats["backbones"][b]["N3"]["result"]["all"]
        assert base["full_supervised_corners"] == after["full_supervised_corners"] == 2499
        assert base["observed_corners"] == after["observed_corners"]
        assert base["pose_available_frames"] == after["pose_available_frames"]
        headlines.append([names[b], pair(src,b,"all","corner_median_px","px"),
                          pair(src,b,"all","corner_P90_px","px"),
                          pair(src,b,"all","PCK10_fraction","%",100),
                          pair(src,b,"all","translation_median_cm","cm"),
                          pair(src,b,"all","rotation_median_deg","deg"),
                          f"{base['observed_corners']} / 2499",
                          f"{base['pose_available_frames']} / 319"])
        for group,label in [("clean","없음"),("moderate","중간"),("severe","어려움")]:
            v = stats["backbones"][b]["Base"]["result"][group]
            grades.append([names[b],label,v["frames"],pair(src,b,group,"corner_median_px","px"),
                           pair(src,b,group,"corner_P90_px","px"),
                           pair(src,b,group,"PCK10_fraction","%",100),
                           pair(src,b,group,"translation_median_cm","cm"),
                           pair(src,b,group,"rotation_median_deg","deg"),
                           f"{v['pose_available_frames']} / {v['frames']}"])
        for arm, payload in stats["backbones"][b]["N3"]["per_seed"].items():
            v = payload["all"]
            getseed=lambda k,u,scale=1:fmt(cell(src,f"/backbones/{b}/N3/per_seed/{arm}/all/{k}",u,
                {"frames":319,"reference_corners":2499,"observed_corners":v["observed_corners"],
                 "pose_frames":v["pose_available_frames"]},[arm],scale=scale))
            seeds.append([names[b],arm,getseed("corner_median_px","px"),getseed("corner_P90_px","px"),
                          getseed("PCK10_fraction","%",100),getseed("translation_median_cm","cm"),
                          getseed("rotation_median_deg","deg"),f"{v['pose_available_frames']}/319"])
    for arm in ["Base","P","N0","N1","N2","N3"]:
        m = stats["backbones"]["yolo"][arm]["result"]["all"]
        get = lambda k,u: fmt(rmetric(src,"yolo",arm,"all",k,u))
        poses.append([arm,get("translation_median_cm","cm")+" / "+get("translation_P90_cm","cm"),
                      get("rotation_median_deg","deg")+" / "+get("rotation_P90_deg","deg"),
                      get("yaw_median_deg","deg")+" / "+get("yaw_P90_deg","deg"),
                      get("IoU3D_median","fraction"),get("ADDsym_AUC_full","fraction"),
                      f"{m['pose_available_frames']}/319"])

    uncertainty = load(STATIC / "PAIRED_POSE_ANALYSIS.json")
    uncertainty_rows, success_rows = [], []
    for b, results in uncertainty["backbones"].items():
        for k, v in results["mean_of_seed_statistics"].items():
            unit="cm" if k.startswith("translation") else "deg"
            for field in ["mean_seed_delta","seed_sd_ddof1","seed_min","seed_max"]:
                cell(STATIC/"PAIRED_POSE_ANALYSIS.json",f"/backbones/{b}/mean_of_seed_statistics/{k}/{field}",
                     unit,{"frames":319,"sessions":13,"bootstrap_replicates":10000},[1,2,3])
            for i in [0,1]:
                cell(STATIC/"PAIRED_POSE_ANALYSIS.json",f"/backbones/{b}/mean_of_seed_statistics/{k}/CI95/{i}",
                     unit,{"frames":319,"sessions":13,"bootstrap_replicates":10000},[1,2,3])
            uncertainty_rows.append([names[b],k,fmt(v["mean_seed_delta"]),
                                     "["+", ".join(fmt(n) for n in v["CI95"])+"]",
                                     fmt(v["seed_sd_ddof1"]),
                                     fmt(v["seed_min"])+" / "+fmt(v["seed_max"])])
        for seed, v in results["per_seed"].items():
            q = v["success_sets"]
            assert q["same_success_set"] is True
            for field in ["before_success","after_success","common_success"]:
                cell(STATIC/"PAIRED_POSE_ANALYSIS.json",f"/backbones/{b}/per_seed/{seed}/success_sets/{field}",
                     "frame",{"frames":319},[seed],digits=0)
            success_rows.append([names[b],seed,q["before_success"],q["after_success"],
                                 q["common_success"],len(q["new_failure_ids"]),len(q["recovered_ids"])])

    square = load(CLOSE / "visibility_square/SQUARE119_RESULTS.json")
    square_rows=[]
    for row_index,row in enumerate(square["family_rows"]):
        for field,unit in [("median_px","px"),("P90_px","px"),("PCK10_percent","%"),
                           ("observed_corners","corner"),("reference_corners","corner")]:
            cell(CLOSE/"visibility_square/SQUARE119_RESULTS.json",f"/family_rows/{row_index}/{field}",unit,
                 {"frames":119,"reference_corners":row["reference_corners"],"mode":row["mode"]},
                 [1,2,3] if row["method"]!="Base" else None)
        square_rows.append([names[row["backbone"]],row["mode"],row["method"],
                            fmt(row["median_px"]),fmt(row["P90_px"]),fmt(row["PCK10_percent"]),
                            f"{row['observed_corners']} / {row['reference_corners']}","x / x"])

    vis = read_csv(CLOSE / "visibility_square/VISIBILITY_RESULTS.csv")
    visible_rows = []
    for row_index,row in enumerate(vis):
        if row["population"]=="DEV319" and row["backbone"]=="yolo" and row["method"] in ["Base","N3"] and row["category"] not in ["ALL","UNKNOWN"]:
            for field,unit in [("median_px","px"),("P90_px","px"),("PCK10_percent","%")]:
                csv_cell(CLOSE/"visibility_square/VISIBILITY_RESULTS.csv",row_index,field,unit,
                         {"population":"DEV319","category":row["category"],"reference_corners":int(row["reference_corners"]),
                          "observed_corners":int(row["observed_corners"])},
                         [1,2,3] if row["method"]=="N3" else None)
            visible_rows.append([row["category"],row["method"],
                                 row["reference_corners"]+" / "+row["observed_corners"],
                                 fmt(float(row["median_px"])),fmt(float(row["P90_px"])),
                                 fmt(float(row["PCK10_percent"]))])

    comp = read_csv(CLOSE / "paper_updated/evidence/github_tables/tab_comparators.csv")
    for i,row in enumerate(comp):
        for column,unit in [("Median px","px"),("P90 px","px"),("PCK10 %","%"),("T med cm","cm"),("R med deg","deg")]:
            csv_cell(CLOSE/"paper_updated/evidence/github_tables/tab_comparators.csv",i,column,unit,
                     {"frames":319,"reference_corners":2499},[1,2,3] if row["Method"]!="R0" else None)
    comparator_rows = [[r["Method"]]+[fmt(float(r[k])) for k in
                         ["Median px","P90 px","PCK10 %","T med cm","R med deg"]] for r in comp]
    students = read_csv(CLOSE / "paper_updated/evidence/github_tables/tab_update_alternative.csv")
    student_sources=load(CLOSE/"static/STUDENT128_REAGGREGATION.json")["backbones"]["yolo"]
    for i,row in enumerate(students):
        source_arm={"R0":"Base","R0_plus_N3":"N3"}.get(row["Method"],row["Method"])
        for column,unit in [("Median px","px"),("P90 px","px"),("PCK10 %","%"),("T med cm","cm"),("R med deg","deg")]:
            csv_cell(CLOSE/"paper_updated/evidence/github_tables/tab_update_alternative.csv",i,column,unit,
                     {"frames":128,"reference_corners":985,"observed_corners":931},
                     [1,2,3] if row["Method"]=="R0_plus_N3" else None)
            CELLS[-1]["raw_method_identifiers"]=student_sources[source_arm]["raw_methods"]
    student_rows = [[r["Method"],r["Frames"],r["Full corners"]]+[fmt(float(r[k])) for k in
                      ["Median px","P90 px","PCK10 %","T med cm","R med deg"]] for r in students]
    costs = read_csv(CLOSE / "paper_updated/evidence/github_tables/tab_cost.csv")
    runtime_rows=[]
    for env in ["pallet-yolo26","pallet-pose"]:
        for i,r in enumerate(costs):
            if r["Environment"]==env:
                for column,unit in [("Added params","parameter"),("CUDA med ms","ms"),("CUDA P90 ms","ms"),
                                   ("Wall med ms","ms"),("N3 only med ms","ms"),("Peak allocated MiB","MiB")]:
                    csv_cell(CLOSE/"paper_updated/evidence/github_tables/tab_cost.csv",i,column,unit,
                             {"frames":26,"sessions":13,"warmup_per_path":20,"repeats_per_frame":5,
                              "measured_samples_per_path":130,"environment":env},[1])
                runtime_rows.append([env,names[r["Backbone"]],r["Path"],r["Added params"],
                                     fmt(float(r["CUDA med ms"])),fmt(float(r["CUDA P90 ms"])),
                                     fmt(float(r["Wall med ms"])),
                                     "NA" if r["N3 only med ms"]=="NA" else fmt(float(r["N3 only med ms"])),
                                     fmt(float(r["Peak allocated MiB"]),2)])

    lifter = load(COMBINED / "lifter/LIFTER_CONTINUITY_SUMMARY.json")
    overall = lifter["overall"]
    assert overall["stored_frame_count"] == 8910
    continuity_rows=[]
    for method, v in overall["methods"].items():
        assert v["fresh_count"]==8772 and v["no_pose_count"]==138 and v["held_count"]==0
        for field,unit in [("fresh_count","frame"),("no_pose_count","frame"),("held_count","frame"),
                           ("fresh_output_rate_pct","%"),("adjacent_output_change/pair_count","pair")]:
            cell(COMBINED/"lifter/LIFTER_CONTINUITY_SUMMARY.json",f"/overall/methods/{method}/{field}",unit,
                 {"frames":8910,"sessions":4},[1] if method=="N3" else None)
        continuity_rows.append([method,8910,v["fresh_count"],v["no_pose_count"],v["held_count"],
                                fmt(v["fresh_output_rate_pct"]),v["adjacent_output_change"]["pair_count"]])
    assist = load(ASSIST / "RESULT.json")
    m = assist["metrics"]
    assert m["decision_counts"]=={"same":12}
    assert m["reference_point_count"]==96 and m["reference_source_counts"]=={"manual_click":66,"pnp_projected":30}
    assist_rows=[]
    for method,v in m["methods"].items():
        for field,unit in [("conditional_error/median_px","px"),("conditional_error/p90_px","px"),
                           ("pck10_hit_count","corner"),("pck10_full_reference_percent","%"),
                           ("valid_matching_prediction_points","corner"),("failed_reference_points","corner")]:
            cell(ASSIST/"RESULT.json",f"/metrics/methods/{method}/{field}",unit,
                 {"frames":12,"reference_corners":96,"sessions":4},[1] if method=="N3" else [42])
        assist_rows.append([method,fmt(v["conditional_error"]["median_px"]),
                            fmt(v["conditional_error"]["p90_px"]),
                            f"{v['pck10_hit_count']}/96 ({fmt(v['pck10_full_reference_percent'])}%)",
                            v["valid_matching_prediction_points"],v["failed_reference_points"]])
    integration=load(ASSIST / "PAPER_INTEGRATION_RECEIPT.json")
    assert integration["status"]=="COMPLETE" and integration["numeric_cells"]==72
    assert integration["original_hash_verification"]["status"]=="PASS"
    assert integration["official_120_24_visible_x_preserved"]

    text = """# N3 실험·원고 검토 자료 — 2026-10-06

**YOLO·DOPE·ResNet-18의 완료한 N3 실험, 재집계·검산, 원고 반영 결과를 GitHub에서 확인하는 자료입니다.** 추가 근거가 필요한 세 항목(리프터 공식 120장/반복 24장, 정지 잡음, 정사각형·리프터 독립 실측 T/R)은 이번 완료 범위와 분리했습니다.

전체 정적 319장에서는 세 기반 모두 중앙 코너 오차와 중앙 이동·회전 오차가 감소했습니다. 일부 P90·어려움 집단은 악화됐습니다. 리프터 12장 보조 평가는 중앙값 악화, P90·PCK10 개선의 혼합 결과입니다. 모든 조건에서 안정적으로 개선되었다고 주장하지 않습니다.

[전체 원고 Markdown](paper_text/manuscript_ko.md) · [보충 원고 Markdown](paper_text/supplement_ko.md) · [12장 전체 비교 이미지](LIFTER_12_ALL_IMAGES_KO.md) · [모든 숫자의 출처](REVIEW_CELL_MAP.json) · [복사 원본 경로·크기·SHA-256](SOURCE_MANIFEST.json) · [문서·해시 검산](PUBLICATION_VALIDATION.json)

## 1. 무엇을 실제로 학습하고 평가했는가

| 구분 | 실제 입력·감독 | 공개하는 결과 |
| --- | --- | --- |
| Base | 동결된 RGB 초기 추정기 | YOLO, DOPE, ResNet-18 각각의 기존 출력 |
| N3 | 이미지 내부 특징 + 초기 코너 + 박스 + 등록 물리 치수 W/D/H; 대칭 감독 포함 | 각 기반에서 따로 학습한 보정기 seed 1·2·3 |
| ResNet Base | 실제 10-epoch CONSTANT-fold RGB; DSNT 공간 softmax 기댓값 decoder | epoch 10, step 34,990 체크포인트와 접기 근거 |
| N0 / N1 | YOLO 통제 재현 국소 보정 / 대칭만 포함 | 기존 각 3seed 원시 코너의 빠졌던 동일 PnP 평가 |

세 Base 자체를 모두 ‘이미지+치수 입력 모델’로 다시 학습한 결과가 아닙니다. **이미지 특징과 치수를 함께 받아 학습하는 부분은 N3 보정기입니다.** 세 기반이 동일한 보정 구조·규약을 사용하지만 가중치는 각각 따로 학습했습니다. 하나의 가중치를 다른 기반에 그대로 전이한 실험이 아닙니다.

기존 N3는 기반마다 3seed, 각 batch 16·6,000 optimizer updates·96,000 표본 노출의 완료 결과를 재사용합니다. DOPE와 ResNet의 유효 고유 합성 학습 행은 각각 44,063 / 55,806개이며, 초기 모델까지의 총학습량이 같다는 뜻은 아닙니다. 이번 마감과 공개 자료 생성의 **신규 학습·optimizer update·새 모델 추론은 0회**입니다.

ResNet의 옛 60-epoch/argmax 설명은 [정정 기록](evidence/contracts/PROTOCOL_CORRECTIONS.md)으로 분리했습니다. 원본 protocol·receipt·가중치를 소급 변경하지 않았습니다. [실제 모델 명세와 해시](evidence/contracts/RESNET_EFFECTIVE_PROTOCOL.json), [DOPE/ResNet 완료 checkpoint 연결](evidence/contracts/TRAINING_REUSE_VERIFICATION.json), [YOLO 완료 기록](evidence/contracts/YOLO_PAPER_TRAINING_COMPLETE.json)으로 확인할 수 있습니다.

## 2. 동일 319장의 세 기반 Base → N3

코너 0–7만 평가하고 중심점 8은 제외합니다. 같은 프레임·카메라·치수·물체 전체 대칭·PnP·실패 규약을 유지합니다. N3 headline은 **각 seed에서 먼저 계산한 통계의 평균**이며 seed 예측을 합치거나 좋은 seed를 선택하지 않았습니다.

"""
    text += table(["기반","코너 중앙값 px","코너 P90 px","PCK≤10px %","T 중앙값 cm","R 중앙값 °","유효 / 참조 코너","자세 산출"],headlines)
    text += """

중앙값/P90은 유효 예측 또는 산출된 자세의 조건부 통계입니다. PCK는 전체 2,499개 참조 코너, 자세 산출률은 전체 319장 분모를 유지합니다. DOPE의 109장 자세 실패와 702개 미관측 참조 코너를 성공 집단에 숨기지 않습니다. 직사각형 T/R은 영상 코너·기하로 재구성한 참조에 대한 값이며 독립 물리 실측 정확도가 아닙니다.

DOPE 코너 P90 51.282→53.570px, 회전 P90 80.583→82.173°와 ResNet 이동 P90 97.545→100.304cm는 악화됩니다. 상대적인 기반 순위에는 서로 다른 초기 모델·훈련 이력·유효 예측 수의 영향이 있습니다.

![세 기반의 동일319장 전후 결과](images/backbone_results.png)

위 그림은 기존 검산된 전체 결과를 그대로 복사했습니다. 마지막 패널의 ‘full-penalty P90’은 결측에 영상 대각선 벌점을 준 프로젝트 지표이며 조건부 코너 P90과 다른 값입니다. [그림 출처](evidence/figures/FIGURE_PROVENANCE.json), [전체·등급·재질·seed별 CSV](evidence/static/DEV319_HEADLINE_AND_SEED.csv), [최신 원시 지표 JSON](evidence/static/STATIC_REAGGREGATION.json)이 연결됩니다.

### 모든 N3 seed

"""
    text += table(["기반","실행","중앙값 px","P90 px","PCK10 %","T cm","R °","자세"],seeds)
    text += """

### 실제 예측점을 그린 개선·무차이·악화 예시

![세 기반 실제 전후 예시](images/static_examples.png)

노란 +는 기하 참조, 청록 원은 Base, 붉은 ×는 N3 seed 1입니다. PnP 재투영점을 모델 예측처럼 표시한 그림이 아닙니다. 각 기반의 개선·무차이·악화 범주에서 첫 사전식 frame ID를 고르는 기존 사후 규칙을 재사용했습니다. 질적 예시이며 전체 집단을 대표하는 무작위 표본은 아닙니다. [9개 예시 frame ID·원사진 해시·선택 규칙](evidence/figures/EXAMPLE_SELECTION.json).

## 3. N0/N1 누락 6D 계산과 짝지은 불확실성

기존 reuse.py의 CORE_ARMS 대상 밖이던 N0_BASE_REPLAY/N1_SYM_ONLY의 원시 코너를 읽어 동일 PnP 평가를 보완한 완료본을 재사용했습니다. 새 학습을 하지 않았고 R0/P/N2/N3는 같은 규약으로 회귀검산했습니다.

"""
    text += table(["YOLO 구성","T 중앙값 / P90 cm","R 중앙값 / P90 °","방향각 중앙값 / P90 °","IoU3D 중앙값","ADDsym AUC 전체","자세"],poses)
    text += """

[N0/N1 seed별 계산 CSV](evidence/pose/N0_N1_POSE_RESULTS.csv) · [원시 지표·분모·규약 JSON](evidence/pose/N0_N1_POSE_RESULTS.json).

같은 frame ID의 Base→N3 자세 성공 집합을 대조했습니다. 세 기반·세 seed 모두 전후 성공 집합이 같고, 신규 실패·복구는 실제 0건입니다. DOPE의 성공 집합은 210장으로 제한되지만 실패 109장은 전체 분모에 유지합니다. [전체 성공 집합과 공통 성공 보조 분석](evidence/pose/PAIRED_POSE_ANALYSIS.json).

"""
    text += table(["기반","seed","Base 성공","N3 성공","공통 성공","새 실패","복구"],success_rows)
    text += """

기존 **13세션 단위 paired bootstrap 10,000회, 난수 seed 20260917**을 재사용했습니다. 같은 재표집을 모든 기반·seed에 적용하고 각 재표집 안에서 seed별 통계 변화의 평균을 냈습니다. 아래 값은 `통계량(after)−통계량(before)`이며 `median(after−before)`가 아닙니다. 95% percentile 구간·seed 표준편차·최솟값·최댓값을 그대로 공개합니다.

"""
    text += table(["기반","통계·단위","평균 변화","95% 구간","seed SD","seed 최소 / 최대"],uncertainty_rows)
    text += """

이 구간은 재사용 DEV에 대한 사후 분석이며 독립 TEST의 확증 구간이 아닙니다. 다중비교 보정은 없습니다. 0을 포함하는 구간이나 악화 값을 임의로 개선 판정하지 않습니다. [짝지은 불확실성 CSV](evidence/pose/PAIRED_POSE_ANALYSIS.csv), [절제 구성 불확실성](evidence/pose/ABLATION_POSE_UNCERTAINTY.json), [2D 불확실성](evidence/pose/PAIRED_2D_UNCERTAINTY.json), [코너·프레임 개선/악화·큰 오류·손상·복구](evidence/pose/POSE_TRACE_AND_TAIL.json).

![세션 짝지은 자세 변화 구간](images/pose_uncertainty.png)

## 4. 최신 가림 등급과 가시성

직사각형 319장의 최신 사람 입력은 **없음 153 / 중간 92 / 어려움 74**입니다. 정사각형 119장은 **3 / 85 / 31**입니다. 세부 등급 판정 기준은 `NOT_CONFIRMED`로 남겨 사용자 입력 등급에 따른 보조 분석으로 제시합니다. 외부 차폐 비율이나 독립 블라인드 가림 강건성의 확증으로 바꾸지 않습니다. 이전 128장 가림 패널(29/20/79, 미분류191)과 최신 전체319장 등급을 섞지 않습니다.

2026-10-07 재확인에서 기존 GREEN150의 clean **103장**과 GREEN0918의 **3장**, 두 고정 패널의 저장 clean 분류 **106장**을 확인했습니다. 0918 원촬영 PNG는 **8,361장**이며 선별119장 밖에서 추가 시각적 후보9장을 찾았습니다. 두 패널의 분류 합계와 미주석 인접 후보를 현재119장 성능 분모에 합치지 않습니다. [공개된 사진115장·목록·분류 근거·검산 안내](../pallet_square_clean_recheck_20261007_v1/README_KO.md)에서 개별 이미지를 확인할 수 있습니다.

"""
    text += table(["기반","사람 입력 등급","영상","중앙값 px","P90 px","PCK10 %","T 중앙값 cm","R 중앙값 °","자세"],grades)
    text += """

YOLO 어려움의 T 중앙값 16.047→17.481cm, DOPE 중간 15.499→16.867cm, ResNet 어려움의 T 33.668→34.043cm·R 51.270→54.608° 악화도 그대로 유지했습니다. 재질별·등급별·각 seed의 모든 행은 [전체 CSV](evidence/static/DEV319_HEADLINE_AND_SEED.csv)와 [행별 JSON 포인터](evidence/static/SUBGROUP_REVIEW.json)에 있습니다.

가시성은 원래 좌표를 바꾸지 않고 기존 71개와 이번 3,030개 상태를 합쳐 **3,101개 참조 코너**를 연결했습니다. 전체 합계는 직접 가시 2,313 / 외부 가림 281 / 자체 가림 462 / 화면 밖 45 / UNKNOWN 0입니다. 참조 좌표가 없는 직사각형 53개·정사각형 350개 슬롯에 상태나 좌표를 새로 만들어 넣지 않았습니다.

아래는 직사각형 2,499점의 YOLO 예시이고, 모든 기반·정사각형 두 모드·각 seed는 별도 CSV에 있습니다. 물체 전체 대칭을 영상별로 먼저 정한 뒤 원래 참조 ID의 상태로 분리하며 상태별로 대칭을 재최적화하지 않습니다.

"""
    text += table(["상태","방법","참조 / 유효 코너","중앙값 px","P90 px","PCK10 %"],visible_rows)
    text += """

[가시성 모든 기반·모드 CSV](evidence/visibility/VISIBILITY_RESULTS.csv) · [각 seed CSV](evidence/visibility/VISIBILITY_PER_SEED.csv) · [분모·회귀검산](evidence/audits/VISIBILITY_SQUARE_VALIDATION.json).

![최신 입력 등급·가시성 구성](images/current_review_summary.png)

## 5. 정사각형 119장: 602점과 600점 분리

`manual_declared`는 저장된 수동 참조 **602점**, `manual_in_frame`은 그 중 화면 밖 2점을 제외한 **600점**입니다. 모든 방법에 같은 모드·분모를 적용했습니다. 한 촬영 세션이고 등록 치수는 [1.1,1.1,0.15]m입니다. 독립 물리 6D 참조가 없어 T/R은 x, 세션 간 일반화 구간은 NA입니다.

"""
    text += table(["기반","모드","방법","중앙값 px","P90 px","PCK10 %","유효 / 참조","T / R"],square_rows)
    text += """

YOLO 602점 모드에서 N3는 Base보다 중앙값 0.522px·P90 1.507px 개선, PCK10 +4.042%p입니다. **N2 대비로는 중앙값 +0.055px·P90 +0.050px·PCK10 −0.332%p로 악화**됩니다. 치수와 대칭의 추가 효과를 Base 대비 효과와 혼동하지 않습니다. 단일 치수·단일 세션만으로 치수 입력의 인과 효과를 주장하지 않습니다.

[두 모드 전체 결과·각 seed](evidence/square/SQUARE119_RESULTS.json) · [두 모드 headline CSV](evidence/square/SQUARE119_RESULTS.csv) · [각 seed CSV](evidence/square/SQUARE119_PER_SEED.csv) · [N3 대 Base / N2 변화](evidence/square/SQUARE119_COMPARISONS.csv).

과거 정사각형 150장(7세션)은 별도 결과로 보존했습니다. 직접 클릭 681점과 PnP 포함 1,200점 proxy를 별도 모드로 유지하고 119장에 합산하지 않았습니다. 해당 150장 동일 계약 N3/DOPE/ResNet 결과는 x입니다. [별도 150장 출처·모드·제한](evidence/square/HISTORICAL_SQUARE150.json).

## 6. 보정 비교군 319장과 학생 128장

### 동일 319장·8코너 보정 비교

"""
    text += table(["방법","중앙값 px","P90 px","PCK10 %","T 중앙값 cm","R 중앙값 °"],comparator_rows)
    text += """

D/L/PoseFix의 기존 3seed 원시 결과를 검산해 재사용했습니다. PoseFix는 팔레트 9점용 변형이며 원 논문 전체 수렴 학습 재현과 동일하지 않습니다. PoseFix가 N3보다 낮은 중앙값을 보이고 N3가 더 낮은 P90을 보이는 결과도 공개합니다. 비교군 전체 사전학습·튜닝·감독·비용이 완전히 같다고 소급 가정하지 않습니다. [같은 319장 각 seed CSV](evidence/static/COMPARATOR319_HEADLINE_AND_SEED.csv), [최신 재질·등급 재집계](evidence/static/COMPARATOR_REAGGREGATION.json), [기존 원시 파일 해시 검산](evidence/audits/REUSED_RESULT_AUDIT.json).

### 공통 비노출 128장 학생 대안

"""
    text += table(["방법","영상","참조 코너","중앙값 px","P90 px","PCK10 %","T 중앙값 cm","R 중앙값 °"],student_rows)
    text += """

이 표의 R0와 N3도 같은 128장을 사용합니다. 전체 319장 기준선을 옆에 붙여 비교하지 않습니다. 기존 고정 학생을 재사용했으며 학생 결과를 N3의 3seed 평균으로 소급 기록하지 않습니다. 이번에 자기학습을 하지 않았습니다. 노출 조건이 맞는 공통 128장 패널이지 새로운 독립 TEST가 아닙니다. [128장 frame ID·노출 조건·분모](evidence/static/STUDENT128_REAGGREGATION.json), [같은 128장 실행별 CSV](evidence/static/STUDENT128_HEADLINE_AND_SEED.csv).

## 7. 비용: 같은 GPU, 실행 환경 별도 표시

기존 측정은 RTX 3080, 고정 26프레임·13세션·batch 1·N3 seed 1·경로별 warmup 20·프레임당 5반복입니다. 전체 130개 측정에서 중앙값/P90을 계산했고 가장 빠른 반복을 골라 쓰지 않았습니다. 이미지 파일 읽기는 제외하고 전처리→동결 Base→decode→N3→고정 prediction-only PnP까지의 전체 경로와 보정 단독 경계를 분리했습니다. YOLO와 DOPE/ResNet의 Python/PyTorch 환경이 달라 별도 패널로 공개합니다.

"""
    text += table(["환경","기반","경로","추가 params","CUDA 중앙 ms","CUDA P90 ms","동기화 wall 중앙 ms","N3 단독 중앙 ms","Peak MiB"],runtime_rows)
    text += """

전체 중앙값의 차이와 N3 단독 중앙값은 다른 계측량입니다. 이 표를 Jetson 성능 또는 모든 비교군의 동일 환경 효율 순위로 해석하지 않습니다. 과거 시간을 이번 재측정값처럼 복사하지 않았습니다. [장비·라이브러리·계측 경계·raw 해시](evidence/runtime/RUNTIME_PANEL.json), [비용 CSV](evidence/tables/tab_cost.csv).

## 8. 리프터에서 이미 완료한 부분

### 고정 8910프레임 출력 연속성

"""
    text += table(["방법","저장 프레임","fresh","no-pose","held","fresh 비율 %","유효 인접쌍"],continuity_rows)
    text += """

네 촬영 세션의 frame ID·선택 객체·코너 결측 마스크를 전수 대조했고 두 방법이 일치합니다. 위 수치는 출력 가용성·연속성이며 참조가 필요한 정확도나 사람이 확인한 정지 잡음이 아닙니다. 실제 장비를 제어하거나 운용 모델을 바꾸지 않았습니다. [연속성 JSON](evidence/lifter/LIFTER_CONTINUITY_SUMMARY.json), [세션별 CSV](evidence/lifter/LIFTER_CONTINUITY_SUMMARY.csv), [검증 receipt](evidence/lifter/CONTINUITY_VALIDATION_RECEIPT.json).

![기존 고정 8910프레임 출력](images/lifter_prediction_timeseries.png)

### 사람이 확인한 PnP 보조 참조 12장·96점

기존 G 저장본의 직접 입력 66점+PnP 보완 30점, 8코너×12장(4세션)을 그대로 사용합니다. 이후 추가 수동 72점 버전으로 참조를 바꾸거나 좋은 결과를 만드는 주석을 선택하지 않았습니다. 실제 대상 확인은 **12장 모두 same**으로 저장됐고 같은 대상·같은 결측 규약으로 비교했습니다. PnP 보조 주석은 기하 참조로 활용할 수 있지만 독립 물리 계측 정답이나 모든 점이 직접 보인다는 뜻은 아닙니다.

"""
    text += table(["방법","중앙값 px","P90 px","전체96점 PCK≤10px","유효점","실패점"],assist_rows)
    paired=m["paired"]
    for field in ["median_after_minus_median_before_px","median_of_paired_after_minus_before_px", "p90_after_minus_p90_before_px"]:
        cell(ASSIST/"RESULT.json",f"/metrics/paired/{field}","px",{"frames":12,"reference_corners":96,"sessions":4},[1])
    text+=f"""

중앙값 변화는 **+{fmt(paired['median_after_minus_median_before_px'])}px(악화)**, P90 변화는 **{fmt(paired['p90_after_minus_p90_before_px'])}px(개선)**, PCK10은 **+3.125%p**입니다. 점별 46개선/50악화, 프레임 중앙값 5개선/7악화입니다. `median(N3)−median(Base)`는 {fmt(paired['median_after_minus_median_before_px'],6)}px이고 `median(N3−Base)`는 {fmt(paired['median_of_paired_after_minus_before_px'],6)}px로 서로 다릅니다.

이 패널은 **YOLO Base training seed 42, N3 seed 1의 탐색적 기하 참조 결과**입니다. 정적 319장·3seed와 분모를 섞지 않고 원래 120장/반복 24장 가시 코너 평가를 완료 처리하지 않습니다. 저장 시 이전 예측을 보지 않았다는 실제 답변은 있으나 최초 G 작성 시점의 독립 기록이 없어 블라인드 참조라는 주장은 하지 않습니다. 현재 선택 박스 노출은 대상 대응 검수 이력으로 분리했습니다.

[96점별 오차 CSV](evidence/lifter/ASSISTED_POINT_ERRORS.csv) · [12장별 JSON](evidence/lifter/ASSISTED_FRAME_RESULTS.json) · [전체 수치·원시 참조/예측/프로토콜 SHA](evidence/lifter/ASSISTED_RESULT.json) · [12장 전체 비교 이미지](LIFTER_12_ALL_IMAGES_KO.md).

## 9. 실제 원고에 반영한 내용과 아직 남은 x

기존 정적 원고 복사본에 본문 162개·보충 237개 숫자 셀을 연결하고, 634개 개별 출처 포인터를 검증했습니다. 최신 PnP 패널은 별도 복사본에 72개 수치 셀·표·설명·모든 12장 그림을 추가했습니다. 이는 전체 본문의 숫자 개수가 471개라는 뜻이 아니라 **두 작업에서 변경·연결한 셀 수**입니다.

| 완료 항목 | 결과 확보 | 실제 원고 반영 | 확인 근거 |
| --- | --- | --- | --- |
| 세 기반 319장·3seed | 완료 | 본문·보충표 | [원고 셀 634포인터](evidence/paper/STATIC_PAPER_CELL_MAP.json) |
| N0/N1 6D·짝지은 구간 | 완료 | 보충표·설명 | [독립 1465개 검산](evidence/audits/INDEPENDENT_PAPER_AUDIT.json) |
| 최신 등급 153/92/74·가시성 3101 | 완료 | 본문·보충표; 상세 행은 CSV | [정적 표 반영 검산](evidence/paper/STATIC_CLOSEOUT_VALIDATION.json) |
| 정사각형 119·602/600 | 완료 | 본문·보충표 | [602/600 동일 분모](evidence/square/SQUARE119_RESULTS.json) |
| D/L/PoseFix 319·학생 128·환경별 시간 | 완료 | 본문·보충표 | [원시 파일 대조](evidence/audits/REUSED_RESULT_AUDIT.json) |
| 리프터 8910 출력 연속성 | 완료 | 사례·보충표·시계열 | [연속성 결과](evidence/lifter/LIFTER_CONTINUITY_SUMMARY.json) |
| PnP 보조 12장·96점 | 완료 | 최신 LaTeX/MD 복사본·표·그림 | [72셀 실제 삽입 receipt](evidence/paper/ASSISTED_PAPER_INTEGRATION_RECEIPT.json) |

| 이번 추가 계산 대상에서 제외한 항목 | 유지 값 | 필요한 추가 근거 |
| --- | --- | --- |
| 리프터 공식 120장·반복 24장 가시 코너 평가 | x | 해당 고정 표본의 기존 참조·대상 대응·실제 반복 검수 |
| 리프터 정지 잡음 | x | 실제 카메라·파렛트가 정지한 원영상 구간과 확인 근거 |
| 정사각형·리프터 독립 물리 T/R | x | 촬영 당시 별도 실측 거리·각도·시각/좌표계 대응 |

그 외 정의되지 않는 값은 NA, 실제 측정된 무차이·실패 복구 0은 0으로 구분했습니다. 과거 150장에 같은 계약의 N3/DOPE/ResNet이 없는 것은 별도 역사 패널의 x이며, 현재 완료한 119장 결과를 삭제하거나 대체하지 않습니다. 비교군의 완전한 예산 동등성·초기 모델 선택 독립성 같은 미확인 방법론도 확인된 것처럼 쓰지 않았습니다.

실제 소스는 [최신 본문 LaTeX](../pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/paper_updated/main.tex)·[최신 보충 LaTeX](../pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1/paper_updated/supplement.tex)에 있습니다. [정적 통합 patch](evidence/paper/STATIC_INTEGRATED.patch)와 [최신 PnP 추가 patch](evidence/paper/ASSISTED_PAPER.patch)를 구분했습니다. 원본과 이전 복사본을 보존했고 PDF를 생성하거나 컴파일하지 않았습니다. 투고용 영문화·최종 편집·저자/소속 확정·교수님 검토는 별도 투고 준비입니다.

## 10. GitHub에서 검산하는 방법

1. 이 README의 표와 이미지를 읽고 원고 Markdown을 확인합니다.
2. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)의 published_path·original_path·SHA-256으로 각 파일이 원본의 정확한 복사본인지 확인합니다.
3. [REVIEW_CELL_MAP.json](REVIEW_CELL_MAP.json)의 JSON 포인터 또는 CSV 행·열, 단위·분모·seed를 따라 표시 수치의 출처를 확인합니다.
4. [공개 파일 SHA 목록](CHECKSUMS.sha256)을 이 폴더에서 `sha256sum -c CHECKSUMS.sha256`으로 확인합니다.

원시 대형 예측·영상·모델 파일은 이번 문서에 모두 복사하지 않았습니다. [원시 입력 경로·크기·SHA 인덱스](RAW_SOURCE_INDEX.csv)에 로컬 보존 위치와 연결 근거를 기록했습니다. 표·그림·각 seed 지표·96점별 오차는 이 폴더에서 직접 열 수 있습니다.

GitHub를 새로 내려받은 환경에서도 저장된 결과를 검산할 수 있도록 [압축 원시 근거와 독립 검산 안내](portable_evidence/README_KO.md)를 함께 제공합니다. 아래 verifier는 Python 표준 라이브러리만 사용해 8,910행의 두 방법 frame ID·결측 마스크와 12장·96점의 코너 오차·분모를 대조합니다. 모델을 불러오거나 새 추론을 실행하지 않습니다.

```bash
git clone https://github.com/CanelE452/pallet-6d-pose.git
cd pallet-6d-pose
python3 scripts/research/pallet_github_publication_20261006_v1/verify_published_evidence.py
```

코드·protocol·완료 학습 receipt·모델 SHA-256은 공개 자료에서 연결합니다. **새 모델 실행에는 인덱스에 적힌 원래 checkpoint와 데이터가 별도로 필요합니다.** 문서·압축 근거의 검산이 모델 재학습 완료를 뜻하지 않습니다. 출처 모델을 받았다면 파일 크기와 SHA-256이 잠금 값과 일치하는지 먼저 확인합니다.

공개 과정의 추가 검증은 가시성 분모·대칭 집계와 PnP 보조 지표의 안전 조건을 확인하는 **13개 fixture test**입니다. 세 기반 전체 학습을 새로 테스트했다는 의미가 아닙니다. 프로젝트의 NumPy/OpenCV 의존성을 갖춘 Python 환경에서 다음 명령으로 해당 평가 규약을 확인할 수 있습니다.

```bash
python3 -m unittest discover -s scripts/research/pallet_combined_closeout_20261003_v1 -p 'test*.py' -v
```

기존 완료본의 주요 검산은 정적 전체 불변 100/100, 기존 모델/대조군/학생 원시 36/36, 가시성·정사각형 72개, 원고 독립 1465/1465 PASS입니다. 최신 PnP 원고 삽입은 이전 원고 161파일의 해시 보존과 patch 적용 검산을 통과했습니다. 원고 페이지 모양은 PDF를 컴파일하지 않아 NA입니다.

이미 실행된 CPU 재집계 명령은 아래와 같습니다. 당시 실제 시간은 [정적 ledger](evidence/static/EXECUTION_COST_LEDGER.json)·[가시성 ledger](evidence/audits/VISIBILITY_CPU_COST_LEDGER.json)·[PnP 계산 기록](evidence/lifter/ASSISTED_EVALUATION_STATUS.json)에 있습니다. 서로 병행한 wall 시간을 합쳐 총작업 시간이나 CPU 사용초로 부르지 않습니다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_static_registry_review_20261003_v1.reaggregate_native_closeout_20261006
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_combined_closeout_20261003_v1/closeout_latest.py lifter-audit
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_combined_closeout_20261003_v1/closeout_latest.py finalize
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python scripts/research/pallet_combined_closeout_20261003_v1/evaluate_pnp_assisted_lifter_20261006.py
```

이 자료 묶음은 기존 결과를 읽어 복사·표현·링크 검산만 수행했습니다. 모델·주석·protocol 원본을 수정하지 않습니다. 공개 승인에 따른 commit/push 상태는 최종 Git commit에서 확인하며, 이 문서 생성 검산을 새 학습이나 새 측정으로 기록하지 않습니다.
"""
    # Keep technical content readable with Korean spacing throughout the prose.
    replacements = {
        "실제원고": "실제 원고", "독립물리": "독립 물리", "기존정적원고복사본": "기존 정적 원고 복사본",
        "최신 PnP패널": "최신 PnP 패널", "72개 수치셀": "72개 수치 셀",
        "634개 개별출처포인터": "634개 개별 출처 포인터",
        "본문162개·보충237개 숫자셀": "본문 162개·보충 237개 숫자 셀",
    }
    for before, after in replacements.items():
        text=text.replace(before,after)
    write_md("README_KO.md",text)

    gallery = "# 리프터 고정 12장 전체 비교\n\n기존 G 저장 참조 8점×12장을 모두 표시합니다. 왼쪽 참조(청록=직접 입력, 보라=PnP 보완), 가운데 Base, 오른쪽 N3입니다. 원은 참조, 십자는 예측입니다. 성능으로 사진을 골라내지 않았습니다.\n\n[전체 결과와 제한](README_KO.md) · [96점별 오차](evidence/lifter/ASSISTED_POINT_ERRORS.csv) · [12장별 JSON](evidence/lifter/ASSISTED_FRAME_RESULTS.json)\n"
    for i,frame in enumerate(load(ASSIST / "FRAME_RESULTS.json")):
        before=frame["methods"]["Base"]["conditional_error"]["median_px"]
        after=frame["methods"]["N3"]["conditional_error"]["median_px"]
        for method in ["Base","N3"]:
            cell(ASSIST/"FRAME_RESULTS.json",f"/{i}/methods/{method}/conditional_error/median_px","px",
                 {"frames":1,"reference_corners":8,"frame_id":frame["frame_id"]},
                 [1] if method=="N3" else [42],document="LIFTER_12_ALL_IMAGES_KO.md")
        filename=frame["frame_id"].replace(":","_")+".png"
        gallery+=f"\n## {frame['frame_id']}\n\n이 프레임의 코너 중앙값: **{fmt(before)} → {fmt(after)}px**, 변화 {fmt(after-before)}px.\n\n![참조·Base·N3 {frame['frame_id']}](images/lifter_12/{filename})\n"
    write_md("LIFTER_12_ALL_IMAGES_KO.md",gallery)
    dump("REVIEW_CELL_MAP.json",{"schema":"github_review_cell_map_v1","cells":CELLS,
                                 "scope":"derived_from_existing_completed_metrics_no_new_experiment"})

    dump("SOURCE_MANIFEST.json",{"schema":"github_review_exact_copy_manifest_v1", "files":COPIES,
                                 "source_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                                 "source_originals_modified":False, "new_training":0,
                                 "optimizer_updates":0,"new_model_forward":0})
    raw=[]
    for section, entries in load(CLOSE / "static/SOURCE_BINDINGS.json").items():
        if isinstance(entries,list):
            for item in entries:
                raw.append({"binding_document":"evidence/static/SOURCE_BINDINGS.json","section":section,
                            "path":item.get("path"),"sha256":item.get("sha256"),"bytes":item.get("bytes"),
                            "copied_in_this_packet":item.get("path") in [r["original_path"] for r in COPIES]})
    for key in ["direct"]:
        for item in load(STATIC / "SOURCE_BINDINGS.json").get(key,[]):
            raw.append({"binding_document":"evidence/contracts/STATIC_ORIGINAL_SOURCE_BINDINGS.json","section":key,
                        "path":item.get("path"),"sha256":item.get("sha256"),"bytes":item.get("bytes"),
                        "copied_in_this_packet":item.get("path") in [r["original_path"] for r in COPIES]})
    with (OUT / "RAW_SOURCE_INDEX.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    dump("PUBLICATION_VALIDATION.json", {"status":"BUILDING_NOT_VALIDATED"})
    (OUT / "CHECKSUMS.sha256").touch()
    validate()


def validate():
    checks=[]
    def check(name, ok, details=None):
        checks.append({"check":name,"passed":bool(ok),"details":details})
        if not ok:
            raise AssertionError((name,details))
    for item in COPIES:
        check("exact_copy_and_original_preserved:"+item["published_path"],
              sha(OUT/item["published_path"])==item["sha256"] and
              sha(ROOT/item["original_path"] if not item["original_path"].startswith("/") else Path(item["original_path"]))==item["sha256"])
    for c in CELLS:
        source=OUT/c["source_copy"]
        if c["source_kind"]=="JSON":
            value=load(source)
            for p in c["json_pointer"].strip("/").split("/"):
                value=value[int(p)] if isinstance(value,list) else value[p]
            name=c["json_pointer"]
        else:
            value=read_csv(source)[c["data_row_zero_based"]][c["column"]]
            name=f"row:{c['data_row_zero_based']}:{c['column']}"
        check("numeric_cell:"+name,value==c["raw_value"] and sha(source)==c["source_sha256"])
    link_count=0
    for md in OUT.rglob("*.md"):
        # Immutable copied provenance notes can refer to their original layout;
        # user-facing README, gallery and portable manuscript must be standalone.
        if md not in [OUT/"README_KO.md",OUT/"LIFTER_12_ALL_IMAGES_KO.md"] and "paper_text" not in md.parts:
            continue
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",md.read_text()):
            if re.match(r"https?://",target) or target.startswith("#"):
                continue
            check("relative_markdown_link:"+str(md.relative_to(OUT))+":"+target,
                  not target.startswith("/") and (md.parent/target.split("#")[0]).exists())
            link_count+=1
    check("all12_fixed_overlays",len(list((OUT/"images/lifter_12").glob("*.png")))==12)
    check("no_models_videos_pdf_archives_copied",not any(p.suffix.lower() in
          [".pt",".pth",".mp4",".avi",".pdf",".zip",".pptx"] for p in OUT.rglob("*")))
    dump("PUBLICATION_VALIDATION.json",{"schema":"github_review_publication_validation_v1","status":"PASS",
         "checks_passed":len(checks),"checks_total":len(checks),"relative_links_checked":link_count,
         "copied_files_checked":len(COPIES),"metric_cells_checked":len(CELLS),"checks":checks,
         "actual_wall_seconds":time.perf_counter()-START,"new_training":0,"optimizer_updates":0,
         "new_model_forward":0,"new_human_labels":0,"source_files_modified":False})
    # Exclude only the checksum file itself. All code, receipts and published
    # documentation are included; sha256sum needs no model/data installation.
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name!="CHECKSUMS.sha256")
    (OUT/"CHECKSUMS.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in files),encoding="utf-8")
    print(json.dumps({"status":"PASS","files":len(files)+1,"bytes":sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
                      "copy_hash_checks":len(COPIES),"metric_cells":len(CELLS),"links":link_count,
                      "wall_seconds":time.perf_counter()-START},ensure_ascii=False))


if __name__=="__main__":
    main()
