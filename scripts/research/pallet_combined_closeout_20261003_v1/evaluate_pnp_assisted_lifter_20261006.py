"""Offline exploratory geometry-reference evaluation, gated on real object review.

The fixed twelve images and original eight canonical PnP-assisted points are
reused.  These references are not promoted to direct-visible or independently
measured physical truth, and the original 120/24 evaluation remains untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/research/pallet_lifter_case_review_20261003_v1"
RAW = ROOT / "data/pallet/results/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1"
DOC = ROOT / "_docs/experiments/pallet_combined_closeout_20261003_v1/pnp_assisted_lifter_20261006_v1"
METHODS = ("Base", "N3")
THRESHOLD_PX = 10.0


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def describe(values):
    return dict(count=len(values), median_px=percentile(values, .5),
                p90_px=percentile(values, .9), mean_px=sum(values) / len(values) if values else None)


def prediction_points(method):
    """Respect frozen explicit masks; finite off-image coordinates remain valid."""
    points = method.get("keypoints")
    if points is None:
        if method.get("keypoints_mask") not in (None, []):
            raise ValueError("Absent keypoints have a nonempty frozen mask")
        return [None] * 8, [False] * 8
    if not isinstance(points, list) or len(points) != 8:
        raise ValueError("Exactly eight canonical prediction slots are required")
    explicit = method.get("keypoints_mask")
    if explicit is not None and (not isinstance(explicit, list) or len(explicit) != 8
                                 or any(type(value) is not bool for value in explicit)):
        raise ValueError("Frozen point-valid mask must be eight booleans")
    valid = []
    for index, point in enumerate(points):
        coordinates = isinstance(point, (list, tuple)) and len(point) == 2 and all(finite(v) for v in point)
        valid.append(bool(coordinates and (explicit is None or explicit[index])))
    return points, valid


def paired_summary(point_rows):
    common = [row for row in point_rows if row["Base_error_px"] is not None
              and row["N3_error_px"] is not None]
    before = [row["Base_error_px"] for row in common]
    after = [row["N3_error_px"] for row in common]
    delta = [b - a for a, b in zip(before, after)]
    return dict(common_valid_point_count=len(common),
                median_after_minus_median_before_px=(percentile(after, .5) - percentile(before, .5)) if common else None,
                median_of_paired_after_minus_before_px=percentile(delta, .5),
                p90_after_minus_p90_before_px=(percentile(after, .9) - percentile(before, .9)) if common else None,
                improved_points=sum(value < 0 for value in delta),
                worsened_points=sum(value > 0 for value in delta),
                tied_points=sum(value == 0 for value in delta),
                Base_only_valid_points=sum(row["Base_error_px"] is not None and row["N3_error_px"] is None for row in point_rows),
                N3_only_valid_points=sum(row["N3_error_px"] is not None and row["Base_error_px"] is None for row in point_rows))


def summary(point_rows):
    denominator = len(point_rows)
    methods = {}
    for method in METHODS:
        valid = [row[f"{method}_error_px"] for row in point_rows if row[f"{method}_error_px"] is not None]
        hit = sum(row[f"{method}_pck10_hit"] for row in point_rows)
        methods[method] = dict(conditional_error=describe(valid),
                              full_reference_point_count=denominator,
                              valid_matching_prediction_points=len(valid),
                              failed_reference_points=denominator - len(valid),
                              failure_reasons=dict(Counter(row[f"{method}_failure_reason"] for row in point_rows
                                                           if row[f"{method}_failure_reason"])),
                              pck10_hit_count=hit,
                              pck10_full_reference_percent=100 * hit / denominator if denominator else None,
                              pck10_conditional_percent=100 * hit / len(valid) if valid else None)
    return dict(reference_point_count=denominator, methods=methods, paired=paired_summary(point_rows))


def compute(reference_records, predictions, decisions):
    """Pure metric function. Undetermined correspondence cannot be discarded."""
    expected = [record["frame_id"] for record in reference_records]
    if len(expected) != len(set(expected)) or set(decisions) != set(expected):
        raise ValueError("Decisions must cover every fixed reference frame exactly once")
    if any(decisions[fid] not in ("same", "different") for fid in expected):
        raise ValueError("Undetermined correspondence is pending, not an evaluation exclusion")
    point_rows, frame_rows = [], []
    for record in reference_records:
        fid = record["frame_id"]
        points = record["points"]
        if len(points) != 8 or [p["id"] for p in points] != list(range(8)):
            raise ValueError("Reference IDs must be the eight original canonical corner IDs")
        if any(not finite(point.get("x")) or not finite(point.get("y")) for point in points):
            raise ValueError("A missing/nonfinite reference is not a new ground truth")
        predicted = predictions.get(fid)
        if predicted is None:
            raise ValueError("Frozen all-frame output is missing a fixed reference frame")
        base, n3 = (predicted["methods"][method] for method in METHODS)
        if base.get("selected_index") != n3.get("selected_index") or base.get("selected_object") != n3.get("selected_object"):
            raise ValueError("Base and N3 must retain the same frozen selected object")
        slots = {method: prediction_points(predicted["methods"][method]) for method in METHODS}
        if slots["Base"][1] != slots["N3"][1]:
            raise ValueError("N3 changed canonical point missingness")
        this_frame = []
        for point in points:
            index = point["id"]
            item = dict(frame_id=fid, session_id=record["session_id"], point_id=index,
                        reference_x=point["x"], reference_y=point["y"],
                        reference_source=point.get("source", "UNKNOWN"), object_match=decisions[fid])
            for method in METHODS:
                predicted_point, mask = slots[method]
                reason = None
                if decisions[fid] == "different":
                    reason = "wrong_selected_object"
                elif predicted["methods"][method].get("selected_index") is None:
                    reason = "no_selected_object"
                elif not mask[index]:
                    reason = "missing_or_invalid_prediction"
                error = None if reason else math.hypot(predicted_point[index][0] - point["x"],
                                                       predicted_point[index][1] - point["y"])
                item.update({f"{method}_error_px": error, f"{method}_failure_reason": reason,
                             f"{method}_pck10_hit": error is not None and error <= THRESHOLD_PX})
            item["paired_N3_minus_Base_px"] = (item["N3_error_px"] - item["Base_error_px"]
                                                if item["N3_error_px"] is not None and item["Base_error_px"] is not None else None)
            point_rows.append(item)
            this_frame.append(item)
        frame_summary = summary(this_frame)
        frame_rows.append(dict(frame_id=fid, session_id=record["session_id"], object_match=decisions[fid], **frame_summary))
    overall = summary(point_rows)
    overall["frame_count"] = len(reference_records)
    overall["decision_counts"] = dict(Counter(decisions.values()))
    overall["reference_source_counts"] = dict(Counter(row["reference_source"] for row in point_rows))
    overall["sessions"] = {}
    for session in dict.fromkeys(record["session_id"] for record in reference_records):
        selected = [row for row in point_rows if row["session_id"] == session]
        overall["sessions"][session] = dict(frame_count=sum(record["session_id"] == session for record in reference_records), **summary(selected))
    medians = [row for row in frame_rows if row["methods"]["Base"]["conditional_error"]["median_px"] is not None
               and row["methods"]["N3"]["conditional_error"]["median_px"] is not None]
    frame_delta = [row["methods"]["N3"]["conditional_error"]["median_px"] - row["methods"]["Base"]["conditional_error"]["median_px"] for row in medians]
    overall["paired_frame_medians"] = dict(common_valid_frames=len(medians), median_N3_minus_Base_frame_median_px=percentile(frame_delta, .5),
                                            improved_frames=sum(v < 0 for v in frame_delta), worsened_frames=sum(v > 0 for v in frame_delta), tied_frames=sum(v == 0 for v in frame_delta))
    return overall, point_rows, frame_rows


def write_csv(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def render_overlays(ctx, records, predictions, decisions, output):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    paths = []
    for record in records:
        fid = record["frame_id"]
        source = ctx.images[fid]
        # Independent images are preserved; every fixed image gets the same view.
        with Image.open(source) as image:
            rgb = image.convert("RGB")
        width, height = rgb.size
        panels = []
        for method in ("reference", "Base", "N3"):
            panel = Image.new("RGB", (width, height + 36), "#202020")
            panel.paste(rgb, (0, 36))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 8), f"{fid} | {method} | object: {decisions[fid]}", font=font, fill="white")
            for point in record["points"]:
                color = "#00e5e5" if point.get("source") == "manual_click" else "#dd66ff"
                x, y = point["x"], point["y"] + 36
                draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=2)
                if method == "reference":
                    draw.text((x+5, y-9), str(point["id"]), font=font, fill=color)
            if method != "reference":
                coords, valid = prediction_points(predictions[fid]["methods"][method])
                color = "#ffaa33" if method == "Base" else "#66aaff"
                for index, point in enumerate(coords):
                    if not valid[index]:
                        continue
                    x, y = point[0], point[1] + 36
                    draw.line((x-4, y, x+4, y), fill=color, width=2)
                    draw.line((x, y-4, x, y+4), fill=color, width=2)
                    draw.text((x+5, y+2), str(index), font=font, fill=color)
            panels.append(panel)
        combined = Image.new("RGB", (width * 3, height + 36))
        for index, panel in enumerate(panels):
            combined.paste(panel, (index * width, 0))
        name = fid.replace(":", "_") + ".png"
        target = output / name
        combined.save(target)
        paths.append(dict(frame_id=fid, path=str(target), sha256=sha(target), source_image_sha256=sha(source)))
    return paths


def markdown(result):
    overall = result["metrics"]
    lines = ["# 리프터 PnP 보조 8코너 부분 평가", "", "고정 12장·96점의 기존 PnP 보조 주석과 원시 Base/N3 예측을 비교했습니다.",
             "공식 120장·24장 반복 검수와는 별도인 탐색적 기하 참조 패널입니다. 모든 점이 직접 보인다는 의미나 독립 실측 정답이라는 의미가 아닙니다.", "",
             "| 방법 | 유효 일치점 중앙값(px) | P90(px) | PCK≤10px 전체 96점(%) | 유효점 / 실패점 |", "|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "NA" if value is None else f"{value:.3f}"
    for method in METHODS:
        row = overall["methods"][method]
        error = row["conditional_error"]
        lines.append(f"| {method} | {fmt(error['median_px'])} | {fmt(error['p90_px'])} | {fmt(row['pck10_full_reference_percent'])} | {row['valid_matching_prediction_points']} / {row['failed_reference_points']} |")
    paired = overall["paired"]
    lines += ["", "잘못 선택한 대상과 결측 점은 전체 분모에 남겨 PCK 실패로 처리합니다. 중앙값/P90은 실제 계산 가능한 일치점에서만 계산하며 실패율을 함께 공개합니다.", "",
              f"- 중앙값의 차이 median(N3)−median(Base): {fmt(paired['median_after_minus_median_before_px'])} px.",
              f"- 짝지은 차이의 중앙값 median(N3−Base): {fmt(paired['median_of_paired_after_minus_before_px'])} px.",
              f"- 점별 개선/악화/동일: {paired['improved_points']} / {paired['worsened_points']} / {paired['tied_points']}.",
              f"- 대상 판정: {json.dumps(overall['decision_counts'], ensure_ascii=False)}.",
              f"- 주석 출처: {json.dumps(overall['reference_source_counts'], ensure_ascii=False)}.", "",
              "이전 모델 예측 노출 여부는 UNKNOWN으로 남깁니다. 현재 대상 대응 화면에서 선택 박스를 본 이력은 실제 검수 기록에 별도로 저장됩니다.",
              "리프터 정지 잡음과 독립 물리 T/R은 x이며, 공식 가시 코너 정확도도 기존 계약의 완료로 바꾸지 않습니다.",
              "원고 삽입은 아직 하지 않았습니다. 새 학습·새 추론·장비 제어는 모두 0회입니다.", "", "## 모든 고정 표본 이미지", "",
              "왼쪽: 참조(청록=직접 클릭, 보라=PnP 생성), 가운데: Base, 오른쪽: N3. 12장 전부 표시하며 성능으로 골라내지 않았습니다.", ""]
    for artifact in result["overlays"]:
        lines += [f"### {artifact['frame_id']}", "", f"![{artifact['frame_id']}]({artifact['path']})", ""]
    return "\n".join(lines) + "\n"


def evaluate():
    from pnp_assisted_reference_20261006 import prepare_context
    ctx = prepare_context()
    reference = read(ctx.reviewed_path)
    records = reference["records"]
    if len(records) != 12 or sum(len(record["points"]) for record in records) != 96:
        raise ValueError("The fixed exploratory contract is exactly twelve frames and 96 reference points")
    if reference.get("source_kind") != "existing_human_assisted_pnp_annotations":
        raise ValueError("Assisted references must remain distinct from human_reviewed official references")
    protocol_path = ctx.reviewed_path.parent / "EVALUATION_PROTOCOL.json"
    if not protocol_path.is_file():
        raise ValueError("Missing locked exploratory evaluation protocol")
    protocol = read(protocol_path)
    bindings = protocol.get("bindings", {})
    required_bindings = {"predictions_sha256": sha(ctx.predictions_path),
                         "prediction_run_identity_sha256": sha(ctx.predictions_path.parent / "RUN_IDENTITY.json"),
                         "reviewed_reference_sha256": sha(ctx.reviewed_path),
                         "manifest_sha256": sha(ctx.manifest_path),
                         "queue_sha256": sha(ctx.queue_path)}
    if any(bindings.get(key) != value for key, value in required_bindings.items()):
        raise ValueError("Exploratory protocol hash binding changed")
    if (protocol.get("ordered_frame_ids") != [record["frame_id"] for record in records]
            or protocol.get("ordered_corner_ids") != list(range(8))
            or protocol.get("fixed_reference_point_denominator") != 96
            or protocol.get("pck_threshold_px") != THRESHOLD_PX
            or protocol.get("include_all_reference_corners") is not True):
        raise ValueError("Exploratory canonical geometry8 denominator/threshold changed")
    for name, binding in (("plan_sha256", reference["bindings"]),
                          ("corner_contract_sha256", reference["bindings"]),
                          ("batch_sha256", {"batch_sha256": reference["original_batch_binding"]["sha256"]})):
        if bindings.get(name) != binding.get(name):
            raise ValueError("Protocol plan/corner/batch binding changed")
    sys.path.insert(0, str(SCRIPTS / "combined_integration"))
    from bridge import validate_sidecar
    sidecar_path = ctx.store_path.parent / "LIFTER_OBJECT_MATCH_REVIEWED.json"
    if not sidecar_path.is_file():
        return dict(status="WAITING_REAL_HUMAN_OBJECT_MATCH", required_match_frames=12,
                    assisted_geometry_corner_accuracy="x", official_visible_corner_accuracy="x",
                    queue_path=str(ctx.queue_path), paper_inserted=False)
    sidecar = read(sidecar_path)
    if sidecar.get("queue_sha256") != sha(ctx.queue_path):
        raise ValueError("Human object decisions belong to a different frozen queue")
    validate_sidecar(ctx.queue, sidecar)
    decisions = {record["frame_id"]: record["decision"] for record in sidecar["records"]}
    expected = {record["frame_id"] for record in records}
    if set(decisions) != expected:
        return dict(status="WAITING_REAL_HUMAN_OBJECT_MATCH", required_match_frames=12,
                    reviewed_match_frames=len(decisions), assisted_geometry_corner_accuracy="x",
                    official_visible_corner_accuracy="x", paper_inserted=False)
    if any(decision == "undetermined" for decision in decisions.values()):
        return dict(status="WAITING_UNDETERMINED_OBJECT_MATCH", decision_counts=dict(Counter(decisions.values())),
                    assisted_geometry_corner_accuracy="x", official_visible_corner_accuracy="x", paper_inserted=False)
    sources = {"script": dict(path=str(Path(__file__)), sha256=sha(__file__)),
               "assisted_reference": dict(path=str(ctx.reviewed_path), sha256=sha(ctx.reviewed_path)),
               "predictions": dict(path=str(ctx.predictions_path), sha256=sha(ctx.predictions_path)),
               "run_identity": dict(path=str(ctx.predictions_path.parent / "RUN_IDENTITY.json"), sha256=sha(ctx.predictions_path.parent / "RUN_IDENTITY.json")),
               "human_match_sidecar": dict(path=str(sidecar_path), sha256=sha(sidecar_path)),
               "object_match_queue": dict(path=str(ctx.queue_path), sha256=sha(ctx.queue_path)),
               "evaluation_protocol": dict(path=str(protocol_path), sha256=sha(protocol_path)),
               "manifest": dict(path=str(ctx.manifest_path), sha256=sha(ctx.manifest_path))}
    latest = DOC / "RESULT.json"
    if latest.is_file():
        prior = read(latest)
        if prior.get("sources") == sources and Path(prior.get("output_dir", "__missing__")).is_dir():
            return dict(prior, reused_identical_result=True)
    predictions = {}
    total_rows = 0
    with ctx.predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total_rows += 1
            if row["frame_id"] in expected:
                if row["frame_id"] in predictions:
                    raise ValueError("Duplicate frozen prediction frame ID")
                predictions[row["frame_id"]] = row
    if total_rows != 8910:
        raise ValueError("Frozen prediction run is not the original 8910-frame run")
    for record in records:
        fid = record["frame_id"]
        row = predictions[fid]
        if row["session_id"] != record["session_id"] or row["decoded_bgr_sha256"] != record["decoded_bgr_sha256"]:
            raise ValueError("Decoded image/frame binding changed")
        for raw_key, reference_key in (("stored_index", "saved_frame_index"),
                                        ("sensor_timestamp_ms", "camera_sensor_timestamp_ms"),
                                        ("camera_frame_number", "camera_frame_number")):
            if row[raw_key] != record[reference_key]:
                raise ValueError("Frozen sensor/frame identity changed")
        if sha(ctx.images[fid]) != record["image_sha256"]:
            raise ValueError("Fixed reference source image changed")
        frozen = next(item for item in ctx.queue["records"] if item["frame_id"] == fid)
        for method in METHODS:
            actual = row["methods"][method]
            if actual["selected_index"] != frozen["selected_index"] or actual["selected_object"]["selected_box_xyxy"] != frozen["selected_box_xyxy"]:
                raise ValueError("Human match no longer refers to the frozen selected box")
        bound = record["source_pnp_binding"]
        source = Path(bound["path"])
        if not source.is_absolute():
            source = ROOT / source
        if sha(source) != bound["sha256"] or source.stat().st_size != bound["bytes"]:
            raise ValueError("Original annotation PnP source changed")
        original_points = read(source)["editor_kps_2d"][:8]
        if [[point["x"], point["y"]] for point in record["points"]] != original_points:
            raise ValueError("Assisted coordinates are not the original eight editor points")
    metrics, point_rows, frame_rows = compute(records, predictions, decisions)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = RAW / "evaluations" / stamp
    output.mkdir(parents=True, exist_ok=False)
    overlays = output / "overlays"
    overlays.mkdir()
    result = dict(status="EVALUATED_EXPLORATORY_ASSISTED_GEOMETRY_12_FRAMES", source_kind="exploratory_geometry_reference_panel",
                  source_reference_kind=reference["source_kind"], canonical_corner_ids=list(range(8)),
                  denominator_mode="all_96_original_geometric_reference_points", unit="px", pck_boundary="Euclidean error <= 10 px",
                  metrics=metrics, frame_results=frame_rows, sources=sources, output_dir=str(output),
                  previous_prediction_exposure="UNKNOWN", directly_visible_claim=False,
                  independent_physical_TR="x", stationary_noise="x", official_visible_corner_accuracy="x",
                  full_120_24_completed=False, machine_human_review_created=False, paper_inserted=False,
                  new_model_forward_frames=0, training_runs=0, optimizer_updates=0, hardware_control_calls=0,
                  overlays=render_overlays(ctx, records, predictions, decisions, overlays))
    write_csv(output / "POINT_ERRORS.csv", point_rows)
    write(output / "FRAME_RESULTS.json", frame_rows)
    write(output / "RESULT.json", result)
    (output / "RESULTS_KO.md").write_text(markdown(result), encoding="utf-8")
    DOC.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output / "RESULT.json", DOC / "RESULT.json")
    shutil.copyfile(output / "RESULTS_KO.md", DOC / "RESULTS_KO.md")
    return result


def main():
    started = time.perf_counter()
    result = evaluate()
    result.update(generated_at=datetime.now(timezone.utc).isoformat(), cpu_wall_seconds=time.perf_counter() - started,
                  training_runs=0, optimizer_updates=0, new_model_forward_frames=0, hardware_control_calls=0)
    DOC.mkdir(parents=True, exist_ok=True)
    gate_path = DOC / "EVALUATION_STATUS.json"
    if gate_path.is_file():
        with gate_path.with_suffix(".history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(read(gate_path), ensure_ascii=False, allow_nan=False) + "\n")
    write(gate_path, result)
    with (DOC / "CPU_COST_LEDGER.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(generated_at=result["generated_at"], status=result["status"],
                                    cpu_wall_seconds=result["cpu_wall_seconds"], script_sha256=sha(__file__),
                                    reused_identical_result=result.get("reused_identical_result", False),
                                    new_model_forward_frames=0, training_runs=0, optimizer_updates=0), ensure_ascii=False) + "\n")
    print(json.dumps(dict(status=result["status"], cpu_wall_seconds=result["cpu_wall_seconds"],
                          paper_inserted=result.get("paper_inserted", False), output_dir=result.get("output_dir")), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
