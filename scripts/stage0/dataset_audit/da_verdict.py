"""Final verdict.  Gates are hardcoded so the numbers cannot move them."""
from __future__ import annotations

import json, pathlib, sys
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import da_common as DA  # noqa: E402


def main():
    policy = json.loads((DA.RELEASE_OUT / "SAMPLING_POLICY.json").read_text())
    leak = json.loads((DA.AUDIT / "DUPLICATE_LEAKAGE_AUDIT.json").read_text())
    table = pd.read_csv(DA.AUDIT / "DATASET_MASTER_TABLE.csv")
    pol = policy["policy"]

    # a training manifest may never contain a dev/holdout frame
    train_ids = set()
    for name in ("PAPER_CORE_V1_corner_manifest.json",
                 "PAPER_CORE_V1_line_manifest.json",
                 "DEPLOYMENT_CANDIDATE_V1_corner_manifest.json",
                 "DEPLOYMENT_CANDIDATE_V1_line_manifest.json"):
        payload = json.loads((DA.RELEASE_OUT / name).read_text())
        for item in payload["items"]:
            train_ids.add((item["dataset_id"], item["frame_id"]))
    features = pd.read_parquet(DA.AUDIT / "positive_frame_features_binned.parquet")
    held = features[(features["mh_split"] == "MH_DEV")
                    | features["dataset_id"].isin(
                        ["EDGE_HARD_TRUNC_DEV", "EDGE_HARD_TRUNC_UNTOUCHED",
                         "EDGE_HARD_CLEAN_UNTOUCHED"])]
    intrusion = sum(1 for d, f in zip(held["dataset_id"], held["frame_id"])
                    if (d, f) in train_ids)

    lines = []
    lines.append("[PAPER CORE]")
    lines.append(f"  Corner = BROAD_40K MH_TRAIN {policy['PAPER_CORE_V1']['corner_n']:,}")
    lines.append(f"  Line   = BROAD_40K MH_TRAIN {policy['PAPER_CORE_V1']['line_n']:,}")
    # CASE A is checked, not assumed: the E3 run recorded its own pool size and
    # the sha256 of the split it drew from.  Both must match the contract in
    # force now, otherwise the existing checkpoint is not this composition.
    contract = json.loads((DA.OUT / "mh_data_contract.json").read_text())
    e3 = json.loads((DA.OUT / "mh_screen_meta_e3confirm25k_seed1.json").read_text())
    same_pool = int(e3.get("pool", -1)) == int(contract["train"])
    same_split = e3.get("split_sha256") == contract.get("split_sha256")
    case_a = bool(same_pool and same_split)
    lines.append(f"  PAPER_NEURAL_RETRAIN_REQUIRED = {not case_a}")
    lines.append(f"    E3 pool {e3.get('pool')} vs contract {contract['train']}"
                 f"  -> {same_pool}")
    lines.append(f"    split sha256 match -> {same_split}"
                 f"  ({str(contract.get('split_sha256'))[:16]}...)")
    lines.append("")
    lines.append("[DEPLOYMENT CANDIDATE]")
    lines.append(f"  Corner = BROAD_40K MH_TRAIN {policy['DEPLOYMENT_CANDIDATE_V1']['corner_n']:,}"
                 "  (CORNER_LA 미포함 — NOT_ESTABLISHED)")
    lines.append(f"  Line   = BROAD {1 - (pol['CHOSEN_RATIO'] or 0):.2f}"
                 f" + EDGE_HARD_TRUNC_TRAIN {pol['CHOSEN_RATIO']}")
    lines.append(f"  DEPLOYMENT_RETRAIN_REQUIRED = True (line stream 만)")
    lines.append("")
    lines.append("[LEAKAGE]")
    lines.append(f"  HARD_BLOCK {len(leak['HARD_BLOCK'])}"
                 f"   LEAKAGE_CLEAN {leak['LEAKAGE_CLEAN']}")
    lines.append(f"  dev/holdout frame in a training manifest: {intrusion}")
    lines.append("")
    lines.append("[EDGE RATIO]")
    lines.append(f"  {pol['CHOSEN']} = {pol['CHOSEN_RATIO']}")
    lines.append(f"  근거: {pol['CHOSEN_BASIS']}")
    lines.append("")
    lines.append("[RETRAIN]")
    ok = (not leak["HARD_BLOCK"]) and intrusion == 0
    lines.append("  DO_NOT_RUN — composition 만 lock. 장시간 학습은 실행하지 않는다."
                 if ok else "  BLOCKED — 누수가 있어 lock 무효")
    verdict = "\n".join(lines)
    (DA.RELEASE_OUT / "FINAL_DATA_VERDICT.txt").write_text(verdict + "\n")
    print(verdict)


if __name__ == "__main__":
    main()
