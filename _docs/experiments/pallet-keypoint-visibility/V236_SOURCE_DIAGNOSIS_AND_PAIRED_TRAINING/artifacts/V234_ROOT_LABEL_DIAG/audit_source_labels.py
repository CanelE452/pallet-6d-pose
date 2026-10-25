"""Saved SOURCE evidence only; no model, raw GT or private evaluation access."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math

H = Path(__file__).parent
R = H.parent


def checked(p, sha):
    p = Path(p)
    assert "v112" not in str(p).lower() and "__pycache__" not in p.parts
    assert not p.is_symlink()
    b = p.read_bytes()
    assert hashlib.sha256(b).hexdigest() == sha, str(p)
    return b


def write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    proto = json.loads((H / "PROTOCOL_BEFORE_SOURCE_DIAG.json").read_text())
    b = proto["allowed_saved_source_observation"]
    observations = [json.loads(line) for line in checked(b["path"], b["sha256"]).splitlines()]
    assert len(observations) == 8000
    fitcomplete = json.loads(checked(R / "V233_ROOT_CROP_ADAPTATION/common_local_crop_pair_v2/COMPLETE.json", "dd662d0a466e38972a386104f9a26f235f955b060c20da4a3b678283f70ed114"))
    models = {}
    for a, binding in fitcomplete["outputs"].items():
        import io
        models[a] = {r["row_id"]: r for r in csv.DictReader(io.StringIO(checked(binding["path"], binding["sha256"]).decode()))}
        assert len(models[a]) == 7617
    control = models["CONTROL_LOCAL_CROP"]
    treat = models["TREAT_LAST_QKV_LORA4"]
    full = [json.loads(line) for line in checked(R / "V233_ROOT_CROP_ADAPTATION/crop_samples_v1/ALL15040_ROWS.jsonl", "434e8a90c9b949f1755ff5fbc765e249f10e53e4df458396d8365f6572e1ff27").splitlines()]
    ycb = {int(r["source_index"]): r for r in full if r["family_code"] == 1}
    assert len(ycb) == 8000 and set(ycb) == set(range(8000))
    counts = collections.Counter()
    matrices = collections.defaultdict(collections.Counter)
    distances = collections.defaultdict(list)
    known = []
    checks = collections.Counter()
    for i, obs in enumerate(observations):
        row = ycb[i]
        old = obs["original_teacher_immutable"]
        state = old["raw_surface_teacher_state"]
        role = "FIT" if row["fit"] else "DEV"
        assert obs["slot_index"] == i and obs["input_id"] + ":aux" + str(obs["logical_aux_slot"]) == row["row_id"]
        assert state == row["original_state"] and obs["q"] == row["query_xy"] and obs["L_input"] == row["L"]
        assert row["target"] == ({"D": 0, "E": 1}.get(state, -1))
        assert obs["native_mapping"] == "UNKNOWN" and obs["training_admission"] is False
        q, ref = obs["q"], obs["r_GT_teacher_only"]
        distance = math.dist(q, ref)
        assert math.isclose(distance, obs["constructed_jitter_distance_px"], abs_tol=1e-12) and distance <= 4 + 1e-12
        qstate = obs["q_centre_observation"]["ownership"]
        counts[(role, state)] += 1
        matrices[(role, "ALL")][(state, qstate)] += 1
        matrices[(role, "LE3" if distance <= 3 else "GT3")][(state, qstate)] += 1
        distances[(role, state)].append(distance)
        if state not in ("D", "E"):
            continue
        assert row["row_id"] in control and row["row_id"] in treat
        c, t = control[row["row_id"]], treat[row["row_id"]]
        assert c["role"] == t["role"] == role
        scale = old["depth_scale_to_mm"]
        raw = old["neighbour_depth_raw_uint16"]
        inside = old["neighbour_inbounds"]
        owners = old["neighbour_visible_owner_count"]
        ownvis = old["neighbour_own_visible"]
        ownfull = old["neighbour_own_full"]
        ownerbits = old["neighbour_owner_bits_uint64"]
        positive = [bool(a and d > 0) for a, d in zip(inside, raw)]
        unique_own = [bool(a and n == 1 and v) for a, n, v in zip(inside, owners, ownvis)]
        assert old["geometry_state"] == "SELF_VISIBLE" and old["geometry_stable"] is True
        assert inside[4] and positive[4]
        if state == "D":
            assert unique_own[4] and sum(unique_own) >= 5 and sum(positive) >= 5
            z = old["centre_integer_ray_first_cameraZ_mm"][0]
            assert z is not None and abs(raw[4] * scale - z) <= 15
            checks["D_saved_mask_depth_and_centre_ray_replay"] += 1
        else:
            assert all(inside) and all(n == 1 for n in owners) and not any(ownvis)
            assert len(set(ownerbits)) == 1 and sum(positive) >= 7 and any(ownfull)
            checks["E_saved_foreign9_depth7_amodal_support_replay"] += 1
        nearest_ref = [math.floor(v + .5) for v in ref]
        offset96 = [(v - z) / row["L"] * 96 for v, z in zip(ref, q)]
        inside_crop = all(-.5 <= 47.5 + o < 95.5 for o in offset96)
        assert inside_crop
        known.append({"slot_index": i, "row_id": row["row_id"], "sample_index": row["sample_index"], "role": role, "scene": obs["scene_id"], "frame_id": obs["source_frame_id"], "object_id": obs["object_id"], "aux_slot": obs["logical_aux_slot"], "associated_r_state": state, "q_ownership_diagnostic_only": qstate, "q_x": q[0], "q_y": q[1], "r_x_GT_diagnostic_only": ref[0], "r_y_GT_diagnostic_only": ref[1], "q_reference_distance_px": distance, "L": row["L"], "reference_inside_original_crop": inside_crop, "raw96_ref_x": 47.5 + offset96[0], "raw96_ref_y": 47.5 + offset96[1], "reason": old["reason"], "positive_depth_count9": sum(positive), "own_full_count9": sum(ownfull), "reference_center_depth_mm": raw[4] * scale, "control": c["prediction"], "treatment": t["prediction"], "control_logit": c["selected_family_logit"], "treatment_logit": t["selected_family_logit"]})
    assert len(known) == 1521
    with (H / "YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(known[0])); w.writeheader(); w.writerows(known)
    def count_table(counter):
        return {f"{a}|{b}": n for (a, b), n in sorted(counter.items())}
    def quantiles(v):
        v = sorted(v)
        return {"N": len(v), "min": v[0], "median": (v[(len(v)-1)//2]+v[len(v)//2])/2, "max": v[-1], "LE3": sum(z <= 3 for z in v)}
    summary = {"PASS": True, "all8000_original_query_L_state_matched": True, "known1521_reference_inside_same_original_crop": True, "counts": count_table(counts), "saved_evidence_replay": dict(checks), "q_owner_is_not_semantic_confusion": {f"{role}_{part}": count_table(table) for (role, part), table in matrices.items()}, "q_r_distances": {f"{r}_{s}": quantiles(v) for (r, s), v in distances.items()}, "YCB_DEV_E_q_owners": dict(collections.Counter(r["q_ownership_diagnostic_only"] for r in known if r["role"] == "DEV" and r["associated_r_state"] == "E")), "YCB_DEV_E_NE3_q_owners": dict(collections.Counter(r["q_ownership_diagnostic_only"] for r in known if r["role"] == "DEV" and r["associated_r_state"] == "E" and r["q_reference_distance_px"] <= 3)), "limits": ["Saved exported masks/depth/rays only. No raw GT decode or new geometry replay.", "E <referenceZ-15mm inequality was replayed in sealed V228; referenceZ is not present in this exported NPZ-derived row, so it is not independently recomputed here.", "q-owner is a teacher-side diagnostic, never a learned-model prediction or substitute label.", "q/reference distance is constructed public emulation jitter, not actual detector error.", "No current/held data, target mutation, GT recrop, model inference, fit or PnP."]}
    write(H / "SUMMARY.json", summary)
    print(json.dumps({"PASS": True, "known_rows": len(known), "checks": dict(checks), "DEV_E_q_owners": summary["YCB_DEV_E_q_owners"], "DEV_E_LE3_q_owners": summary["YCB_DEV_E_NE3_q_owners"]}))


if __name__ == "__main__":
    main()
