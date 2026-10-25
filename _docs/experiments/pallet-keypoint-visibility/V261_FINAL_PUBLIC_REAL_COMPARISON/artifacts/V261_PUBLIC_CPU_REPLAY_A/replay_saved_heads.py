"""Portable V261 saved-feature CPU replay. No encoder, RGB, training or artifact imports."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import numpy as np
import torch

sys.dont_write_bytecode = True
ARMS = ("CONTROL_ORIGINAL_SOURCE_FIT", "TREAT_ORIGINAL_PLUS_PUBLIC_SURFACE_AUX")
GROUPS = ("ALL", "WOOD_PRIMARY105", "SQUARE118")
D, E, S = "DIRECT_VISIBLE", "EXTERNAL_OCCLUDED", "SELF_OCCLUDED"
PARAMS = {"head.0.weight": (64, 1186), "head.0.bias": (64,), "head.2.weight": (2, 64), "head.2.bias": (2,)}
META = {"supported", "original_available", "native_slot", "frame_index", "frame_id", "row_id", "query_xy", "L"}


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def lexical(value):
    s = str(value)
    need(not any(x in s.casefold() for x in ("v112", "cache", ".pyc")), "Forbidden path before FS")
    p = Path(s)
    need(p.is_absolute() and ".." not in p.parts, "Absolute closed path required")
    return p


def safe(value, fresh=False):
    p = lexical(value)
    for a in reversed([p, *p.parents]):
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            continue
        need(not stat.S_ISLNK(mode), "Symlink artifact or ancestor rejected")
    if fresh:
        need(not p.exists(), "Fresh output only; no automatic retry/resume")
    return p


def relative(value):
    need(isinstance(value, str) and value != "" and "\\" not in value, "POSIX relative path required")
    p = PurePosixPath(value)
    need(not p.is_absolute() and not any(x in ("", ".", "..") for x in value.split("/")), "No relative traversal")
    need(not any(x in value.casefold() for x in ("v112", "cache", ".pyc")), "Forbidden relative path before FS")
    return p


def checked(package, b):
    need(set(b) == {"relative_path", "sha256", "size_bytes"}, "Only package-relative exact bindings")
    rel = relative(b["relative_path"])
    need(isinstance(b["sha256"], str) and len(b["sha256"]) == 64
         and all(c in "0123456789abcdef" for c in b["sha256"])
         and type(b["size_bytes"]) is int and b["size_bytes"] >= 0, "Exact SHA/size")
    p = safe(package / rel)
    need(stat.S_ISREG(os.lstat(p).st_mode), "Regular package artifact required")
    h, size = hashlib.sha256(), 0
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    need(h.hexdigest() == b["sha256"] and size == b["size_bytes"], "Package artifact SHA/size mismatch")
    return p


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with safe(path, fresh=True).open("xb") as f:
        f.write(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def weights(cp, arm):
    need(cp["arm"] == arm and cp["updates"] == 1000 and cp["head_parameters"] == 76098
         and cp["use_lora"] is False, "Only exact final1000 CropFamilyHead76098")
    state = cp["head_state_dict"]
    need(set(state) == set(PARAMS), "Exactly four head tensors")
    out = {}
    for name, shape in PARAMS.items():
        t = state[name]
        need(isinstance(t, torch.Tensor) and tuple(t.shape) == shape and t.dtype == torch.float32
             and t.device.type == "cpu" and torch.isfinite(t).all(), "Finite CPU FP32 head tensor")
        out[name] = t.detach().numpy().copy()
    need(sum(v.size for v in out.values()) == 76098, "Fixed76098 parameters")
    return out


def numpy_forward(x, w):
    """Literal two Linear layers and ReLU, FP32; MAIN is column0, not argmax."""
    need(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 1186, "FP32 local1186")
    h = x @ w["head.0.weight"].T + w["head.0.bias"]
    h = np.maximum(h, np.float32(0))
    y = h @ w["head.2.weight"].T + w["head.2.bias"]
    need(y.dtype == np.float32 and np.isfinite(y).all(), "Finite FP32 logits")
    return y


def ratio(a, b):
    return a / b if b else None


def table(rows, model):
    counts = Counter(("D" if r["human_state"] == D else "E", r[model + "_D_E_U"]) for r in rows)
    dd, de, du = (counts[("D", p)] for p in ("D", "E", "U"))
    ed, ee, eu = (counts[("E", p)] for p in ("D", "E", "U"))
    nd, ne = dd + de + du, ed + ee + eu
    return {"N": len(rows), "N_D": nd, "N_E": ne, "D_D": dd, "D_E": de, "D_U": du,
            "E_D": ed, "E_E": ee, "E_U": eu, "D_precision": ratio(dd, dd + ed),
            "D_recall_including_U": ratio(dd, nd), "E_precision": ratio(ee, ee + de),
            "E_recall_including_U": ratio(ee, ne), "U": du + eu, "U_fraction": ratio(du + eu, len(rows)),
            "resolved_fraction": ratio(dd + de + ed + ee, len(rows)),
            "balanced_recall": None if not nd or not ne else (dd / nd + ee / ne) / 2}


def reduce_ledger(ledger):
    models = ("v61", "legacy", *ARMS)
    need(all(r[m + "_D_E_U"] in ("D", "E", "U") for r in ledger for m in models), "Stored binary/U namespace")
    primary = [r for r in ledger if r["paired_same_query_RGB_D_vs_E"] == "True" and r["human_state"] in (D, E)]
    need((len(primary), sum(r["human_state"] == D for r in primary), sum(r["human_state"] == E for r in primary))
         == (1237, 1147, 90), "Fixed primary1237 D1147 E90")
    need(sum(r["human_state"] == S for r in ledger) == 136, "Fixed humanSELF136")
    metrics, inventory, self_rows = [], [], []
    for group in GROUPS:
        pp = [r for r in primary if group == "ALL" or r["group"] == group]
        strata = {"PRIMARY": pp,
                  "ERROR_LE3": [r for r in pp if number(r["initial_reference_error_px"]) is not None and number(r["initial_reference_error_px"]) <= 3],
                  "ERROR_GT3": [r for r in pp if number(r["initial_reference_error_px"]) is not None and number(r["initial_reference_error_px"]) > 3],
                  "ERROR_MISSING": [r for r in pp if number(r["initial_reference_error_px"]) is None]}
        need(sum(len(strata[k]) for k in ("ERROR_LE3", "ERROR_GT3", "ERROR_MISSING")) == len(pp), "Error bins partition primary")
        allrows = [r for r in ledger if group == "ALL" or r["group"] == group]
        for model in models:
            metrics.extend(dict(group=group, stratum=name, model=model, **table(rows, model)) for name, rows in strata.items())
            c = Counter(r[model + "_D_E_U"] for r in allrows)
            inventory.append(dict(group=group, model=model, N=len(allrows), D=c["D"], E=c["E"], U=c["U"]))
            for cohort in ("ALL_REGISTERED", "PAIRED_SAME_QUERY"):
                if cohort == "PAIRED_SAME_QUERY" and not all("paired_same_query_RGB_D_vs_ES" in r for r in ledger):
                    self_rows.append(dict(group=group, model=model, cohort=cohort, N=None, status="UNKNOWN: inherited pairedSELF flag absent"))
                    continue
                rr = [r for r in allrows if r["human_state"] == S and (cohort == "ALL_REGISTERED" or r["paired_same_query_RGB_D_vs_ES"] == "True")]
                c = Counter(r[model + "_D_E_U"] for r in rr)
                self_rows.append(dict(group=group, model=model, cohort=cohort, N=len(rr), D=c["D"], E=c["E"], U=c["U"],
                                      interpretation="Binary response distribution on inherited humanSELF; no learnedSELF accuracy"))
    return {"primary": metrics, "all1784_inventory": inventory, "human_SELF_separate": self_rows}


def run(package, entry_path, output):
    entry_path = safe(entry_path)
    entry_bytes = entry_path.read_bytes()
    entry = json.loads(entry_bytes)
    need(entry.get("approved_saved_CPU_replay") is True, "Root true saved replay entry required before inputs/output")
    own = safe(__file__).read_bytes()
    need(hashlib.sha256(own).hexdigest() == entry["replay_code_sha256"], "Exact own replay CODE")
    need(set(entry["inputs"]) == {"features", "input_metadata", "ledger", *ARMS}, "Only exact five saved inputs")
    package = safe(package)
    paths = {k: checked(package, b) for k, b in entry["inputs"].items()}
    out = safe(output, fresh=True)
    need(out != package and package not in out.parents and out not in package.parents, "Output outside immutable package")
    out.mkdir()
    try:
        x = np.load(paths["features"], mmap_mode="r", allow_pickle=False)
        need(x.shape == (1784, 1186) and x.dtype == np.float32 and np.isfinite(x).all(), "All1784 finite saved local1186")
        with np.load(paths["input_metadata"], allow_pickle=False) as z:
            need(set(z.files) == META, "Exact input-only metadata8 fields")
            meta = {k: z[k] for k in META}
        need(meta["supported"].dtype == bool and meta["original_available"].dtype == bool
             and meta["supported"].shape == meta["original_available"].shape == (1784,), "Original support/availability bool")
        supported = meta["supported"]
        need(not supported[~meta["original_available"]].any() and int((~meta["original_available"]).sum()) == 45,
             "Original45 inputU retained; support does not override originalU")
        need(not x[~supported].any(), "Unsupported features allzero; never classify head bias on them")
        need(np.array_equal(meta["native_slot"], np.arange(1784) % 8)
             and np.array_equal(meta["frame_index"], np.arange(1784) // 8)
             and meta["query_xy"].shape == (1784, 2) and meta["L"].shape == (1784,), "Original native/frame/qL shape")
        ledger = list(csv.DictReader(io.StringIO(paths["ledger"].read_text(encoding="utf-8-sig"))))
        need(len(ledger) == 1784, "All1784 saved ledger rows")
        for i, r in enumerate(ledger):
            need(int(r["global_index"]) == i and int(r["native_corner"]) == i % 8 and int(r["frame_index"]) == i // 8
                 and r["row_id"] == str(meta["row_id"][i]) and r["id"] == str(meta["frame_id"][i]), "Exact row/native/frame IDs")
            need((r["original_query_valid"] == "True") == bool(meta["original_available"][i]), "Original availability identity")
            need(all(number(a) == number(b) for a, b in zip(meta["query_xy"][i], (r["query_x"], r["query_y"])))
                 and number(meta["L"][i]) == number(r["roi_side"]), "Original saved q/L identity")
        cps = {a: torch.load(paths[a], map_location="cpu", weights_only=True) for a in ARMS}
        need(cps[ARMS[0]]["initial_head_sha256"] == cps[ARMS[1]]["initial_head_sha256"], "Same original head initialization")
        rows, comparisons = [], {}
        take = np.flatnonzero(supported)
        for arm in ARMS:
            w = weights(cps[arm], arm)
            logits = np.empty((len(take), 2), np.float32)
            for start in range(0, len(take), 64):
                logits[start:start + 64] = numpy_forward(np.asarray(x[take[start:start + 64]]), w)
            saved = np.asarray([number(ledger[i][arm + "_MAIN0_logit"]) for i in take], np.float64)
            need(np.isfinite(saved).all() and np.allclose(logits[:, 0], saved, atol=1e-5, rtol=1e-5), "Saved MAIN0 logit reproduction atol=rtol1e-5")
            pos = {int(i): j for j, i in enumerate(take)}
            for i, r in enumerate(ledger):
                j = pos.get(i)
                value = None if j is None else float(logits[j, 0])
                pred = "U" if j is None else ("E" if value >= 0 else "D")
                need(pred == r[arm + "_D_E_U"], "Every saved MAIN0 prediction must match; threshold0 unchanged")
                if j is None:
                    need(number(r[arm + "_MAIN0_logit"]) is None, "Unsupported saved logit must be null/empty")
                rows.append({"global_index": i, "row_id": r["row_id"], "model": arm, "input_supported": bool(supported[i]),
                             "CPU_MAIN0_logit": value, "saved_MAIN0_logit": number(r[arm + "_MAIN0_logit"]),
                             "CPU_prediction": pred, "saved_prediction": r[arm + "_D_E_U"],
                             "absolute_error": None if j is None else abs(value - saved[j])})
            comparisons[arm] = {"rows": 1784, "supported": len(take), "U": int((~supported).sum()),
                                "all_predictions_equal": True, "MAIN0_allclose_atol_rtol": 1e-5,
                                "maximum_absolute_logit_error": float(np.max(np.abs(logits[:, 0] - saved))) if len(take) else None}
        metrics = reduce_ledger(ledger)
        rb = out / "REPLAY_ROWS.csv"
        with rb.open("x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        raw = rb.read_bytes()
        rows_b = {"path": str(rb), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
        summary = write_json(out / "SUMMARY.json", {"completed": True, "PASS": True, "saved_prediction_replay": comparisons,
            "tables": metrics, "primary_N": 1237, "primary_D": 1147, "primary_E": 90,
            "logit_semantics": "MAIN0>=0 E; negative D; unsupported U is input geometry, no model confidence threshold",
            "v61": "Stored original visible-target presence0.5 decisions; no v61 model forward",
            "limits": ["Repeated frozen current ledger; not fresh blind validation or new gold",
                       "HumanSELF response separate from D/E primary; no SELF correctness or physical native gold",
                       "Error<=3 is inherited q-reference separation, not coordinate correction",
                       "No encoder, RGB, artifact import, optimizer, fit, bank or threshold selection"],
            "head_forward_implementation": "NumPy FP32 four saved tensors; torchCPU weights_only load only", "optimizer_updates": 0})
        return write_json(out / "COMPLETE.json", {"completed": True, "PASS": True, "source_bindings": entry["inputs"],
             "replay_code_sha256": entry["replay_code_sha256"], "entry_sha256": hashlib.sha256(entry_bytes).hexdigest(),
             "summary": summary, "rows": rows_b, "input_relative_paths_only": True, "original_absolute_fallback": False,
             "encoder_or_RGB_or_training_or_Git_operations": 0})
    except Exception as exc:
        write_json(out / "FAILURE.json", {"completed": False, "PASS": False, "error_type": type(exc).__name__,
                   "error": str(exc), "partial_outputs_preserved": True, "automatic_retry": False})
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True)
    ap.add_argument("--entry", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.package, args.entry, args.output_dir), sort_keys=True), flush=True)
