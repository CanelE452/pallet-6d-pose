"""Source-only supervision coverage; no models, private data, or relabeling."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import math
import statistics

V = Path(__file__).resolve().parent
P = V.parent
S = P / "query_visibility_independent_auxiliary_source_control_20261024_v155"
PROTOCOL = S / "source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json"
REF = P / "query_visibility_mr6d_full_public_source_20261024_v153/native_reference_preparation/provided_reference_v1/ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl"
OUT = V / "root_source_label_support"


def sha(path):
    assert "_v112" not in str(path).lower() and "private_evaluation" not in str(path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def checked(binding):
    assert sha(binding["path"]) == binding["sha256"]
    return Path(binding["path"])


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def write(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert not OUT.exists()
    protocol = read(PROTOCOL)
    rows = lines(checked(protocol["source_labels"]))
    seal = read(checked(protocol["source_supervision_seal"]))
    ref_binding = next(b for b in seal["inputs"] if b["path"] == str(REF))
    references = lines(checked(ref_binding))
    lookup = {(r["frame_index"], r["gt_instance_index"], r["native_slot"]): r for r in references}
    native_spec = read(checked(protocol["original_native35"]))
    pallox_rows = lines(checked(native_spec["public_labels"]))
    assert len(rows) == 1840 and len(references) == 2424 and len(pallox_rows) == 512
    counts = defaultdict(Counter)
    distances = []
    groups_distance = defaultdict(list)
    for i, row in enumerate(rows):
        assert row["global_index"] == i and row["frame_index"] == i // 8 and row["native_slot"] == i % 8
        assert row["point_quality_label"] == -1 and not row["human_gold"] and not row["exact_real_point_gold"]
        assert not row["source_visibility_scores_used"] and not row["coordinates_or_native_order_changed"]
        state = row["role_state"]
        avail = row["original_input_valid"]
        known = row["role_target"] in (0, 1)
        if known:
            assert row["entity_target"] == 1 and row["role_correspondence_supported"]
            assert state == ("D", "E")[row["role_target"]]
        if row["entity_target"] != 1:
            assert row["role_target"] == -1
        scene = row["source_id"].split("/")[1].split(":")[0]
        group_names = ("ALL", row["split"], f'{row["split"]}::native{i % 8}', f'{row["split"]}::scene{scene}')
        for name in group_names:
            c = counts[name]
            c["rows"] += 1
            c[f"state_{state}"] += 1
            c[f'entity_{row["entity_target"]}'] += 1
            c["available"] += int(avail)
            c["visibility_loss_rows"] += int(avail and known)
            if avail:
                c[f"available_{state}"] += 1
            if state == "E":
                c["E_occluder::" + row["occluder_or_uncertainty"]] += 1
            c["quality_unknown"] += 1
        gt = row["selected_gt_instance_index"]
        reference = lookup.get((i // 8, gt, i % 8)) if gt is not None else None
        distance = None
        if reference is not None:
            q = row["query_xy"]
            rxy = reference["reference_xy"]
            if all(math.isfinite(v) for v in q + rxy) and q != [-1.0, -1.0]:
                distance = math.dist(q, rxy)
        bucket = "UNKNOWN" if distance is None else "LE3" if distance <= 3 else "GT3_LE20" if distance <= 20 else "GT20"
        if avail and known:
            for name in (f'{row["split"]}::{state}', f'{row["split"]}::ALL_DE'):
                if distance is not None:
                    groups_distance[name].append(distance)
                counts[name]["nominal_reference_deviation::" + bucket] += 1
        distances.append({"global_index": i, "frame_index": i // 8, "split": row["split"], "native_slot": i % 8,
                          "weak_state": state, "input_available": avail,
                          "reference_deviation_px": distance, "nominal_reference_deviation_bin": bucket,
                          "reference_not_physical_or_pixel_gold": True})
    public_E = Counter()
    public_D = Counter()
    for key, expected, target in (("TRAIN_D35", "D", public_D), ("TRAIN_E8", "E", public_E)):
        for index in native_spec[key]:
            target[index % 8] += 1
    for row in rows:
        if row["split"] == "SOURCE_DEVELOPMENT" and row["original_input_valid"]:
            if row["role_state"] == "E":
                public_E[row["native_slot"]] += 1
            if row["role_state"] == "D":
                public_D[row["native_slot"]] += 1
    OUT.mkdir()
    with (OUT / "ALL1840_NOMINAL_REFERENCE_DEVIATION_DIAGNOSTIC.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(distances[0])); writer.writeheader(); writer.writerows(distances)
    result = {
        "schema": "V157_SOURCE_LABEL_LOSS_SUPPORT_DIAGNOSIS", "completed": True, "PASS": True,
        "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_only": True, "code": bind(__file__), "rules": bind(V / "ROOT_LABEL_SUPPORT_AUDIT_RULES_BEFORE_LABELS.json"),
        "inputs": [bind(PROTOCOL), protocol["source_labels"], protocol["source_supervision_seal"], ref_binding,
                   protocol["original_native35"], native_spec["public_labels"], protocol["code"]],
        "all1840_retained": True, "original_source_labels_changed": False,
        "MR_counts": {name: dict(c) for name, c in counts.items()},
        "MR_nominal_reference_deviation_summary": {name: {"N": len(x), "median_px": statistics.median(x),
            "min_px": min(x), "max_px": max(x)} for name, x in groups_distance.items()},
        "MR_available_TRAIN_E_by_native": {str(k): counts[f"SOURCE_DEVELOPMENT::native{k}"]["available_E"] for k in range(8)},
        "MR_available_HELD_E_by_native": {str(k): counts[f"SOURCE_VALIDATION::native{k}"]["available_E"] for k in range(8)},
        "all_existing_public_TRAIN_visibility_D_by_native": {str(k): public_D[k] for k in range(8)},
        "all_existing_public_TRAIN_visibility_E_by_native": {str(k): public_E[k] for k in range(8)},
        "Pallox_original_TRAIN_D_E": [len(native_spec["TRAIN_D35"]), len(native_spec["TRAIN_E8"])],
        "missing_public_TRAIN_E_native_slots": [k for k in range(8) if public_E[k] == 0],
        "semantic_contract_KO": [
            "MR D/E는 실제 초기query의 같은-native 역할 대응이 구조적으로 지지될 때 해당 GT-role의 약한 가시성 상태를 연결한 라벨이다. query 픽셀 자체의 정확한 표면/가림 정답이 아니다.",
            "wrong entity와 역할 미확정은 E로 바꾸지 않고 visibility loss에서 제외한다. entity 보조 손실과 별도이며 entity/quality는 primary 가림 gate가 아니다.",
            "모든 MR 실사 위치 품질 라벨은 -1이다. 제공 pose/CAD의 낮은 cm 정확도 때문에 위 거리 bin은 명목 참조와의 편차 진단일 뿐3px/far 픽셀 정답이 아니다.",
            "공개실사 supervision과 합성 native35 검증을 구분한다. public E 역할 범위가 비어 있어도 합성에 그 역할이 없는 것으로 해석하지 않는다.",
        ],
        "training_loss_code_evidence": {
            "visibility_MR": "available & SOURCE_DEVELOPMENT & role_target>=0; full batch class-balanced BCE D341/E49 per step",
            "visibility_Pallox": "fixed TRAIN_D35/TRAIN_E8, HALL excluded",
            "synthetic_visibility": "OLD/NEW native_bce, original native masks; no entity or point-quality filter",
            "entity": "balanced SOURCE_DEVELOPMENT available known entity only; separate auxiliary optimizer/clip in C37/C38",
            "point_quality": "synthetic-only known near/far train rows, real1840 unknown; never primary gate",
        },
        "source_only_next_investigation_KO": "공개실사의 E supervision이 어느 native 역할과 가림물에 집중됐는지 실제 source crop/GT-role 보드를 대조한다. 빠진 역할을 기존 held에서 옮기지 않고, TRAIN-only 신규 공개자료/합성에서 보완할 근거를 구한다. 명목pose 편차가 큰 라벨을 픽셀 정답으로 간주하는 학습 변경은 하지 않는다.",
        "new_model_or_training_or_private_or_V112_or_PnP_qL_calls": 0,
        "selected_model": None, "goal_status": "active_unsolved",
    }
    write(OUT / "LABEL_SUPPORT_DIAGNOSIS.json", result)
    print(json.dumps({"completed": True, "report": bind(OUT / "LABEL_SUPPORT_DIAGNOSIS.json"),
                      "TRAIN_MR_E_native": result["MR_available_TRAIN_E_by_native"],
                      "PUBLIC_TRAIN_E_native": result["all_existing_public_TRAIN_visibility_E_by_native"],
                      "reference_deviation": result["MR_nominal_reference_deviation_summary"]}))


if __name__ == "__main__":
    main()
