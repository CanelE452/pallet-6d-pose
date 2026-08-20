"""PHASE 8 artefacts -- score definition, frozen thresholds, hashes.

Both operating points are frozen, because the brief specifies two that do not
coincide: the selection rule ("minimise FP subject to Recall >= 95%") and the
gate ("Recall drop <= 2pp", i.e. Recall >= 98%).  Freezing both keeps the
record honest; picking one is a decision, not a computation.
"""
from __future__ import annotations

import hashlib, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mh_data as MD      # noqa: E402
import mh_presence as PR  # noqa: E402

OUT = MD.OUT


def main():
    gate = json.loads((OUT / "presence_gate_result.json").read_text())
    curve = json.loads((OUT / "presence_recall_fp_curve.json").read_text())

    definition = {
        "score_name": "z",
        "definition": "per image, take the spatial max of each of the 8 corner"
                      " belief channels (channel 8, the centroid, is excluded),"
                      " sort descending -> z in R^8.",
        "detachment": "z is read under torch.no_grad from a .detach()ed stem;"
                      " no gradient path exists from any presence loss to the"
                      " pose network. Verified by measurement, not by argument:"
                      " see presence_pose_invariance.json.",
        "P1_SCORE4KP": "score = z[3], the 4th highest corner peak. No fit.",
        "P2_DETACHED_LINEAR": "score = sigmoid(w . z + b), 9 trainable"
                              " parameters, fitted on cached z only.",
        "pose_parameters_trained": 0,
    }
    (OUT / "presence_score_definition.json").write_text(
        json.dumps(definition, indent=1))

    frozen = {"frozen_on": "NEG dev + MH_DEV",
              "operating_points": {}, "seeds": {}}
    for seed in (1, 2):
        c = np.load(OUT / f"presence_z_cache_seed{seed}.npz", allow_pickle=True)
        layer, _ = PR.fit_linear(c["pos_train"], c["neg_train"], seed)
        w = layer.weight.detach().cpu().numpy().reshape(-1).tolist()
        b = float(layer.bias.detach().cpu().numpy().reshape(-1)[0])
        block = {"P2_weights": [round(v, 6) for v in w],
                 "P2_bias": round(b, 6), "thresholds": {}}
        for arm in ("P1_SCORE4KP", "P2_DETACHED_LINEAR"):
            entries = {f"recall_{e['recall_target']}": e
                       for e in curve["seeds"][f"seed{seed}"][arm]}
            block["thresholds"][arm] = {
                "selection_rule_recall_0.95":
                    gate["seeds"][f"seed{seed}"][arm]["operating_point"],
                "gate_recall_0.98": entries["recall_0.98"]}
        frozen["seeds"][f"seed{seed}"] = block
    frozen["operating_points"] = {
        "selection_rule": "minimise negative FP/image subject to positive"
                          " Recall >= 0.95 (brief, PHASE 6)",
        "gate": "positive Recall drop <= 2pp, i.e. Recall >= 0.98"
                " (brief, PHASE 7)",
        "note": "these two do not coincide; minimising FP walks the threshold"
                " down to the 0.95 floor. Both are frozen."}
    path = OUT / "presence_threshold.json"
    payload = json.dumps(frozen, indent=1)
    path.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (OUT / "threshold_sha256.txt").write_text(
        f"{digest}  presence_threshold.json\n")
    print(f"  presence_score_definition.json")
    print(f"  presence_threshold.json  sha256 {digest}")


if __name__ == "__main__":
    main()
