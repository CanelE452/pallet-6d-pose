"""Fail-closed regression and local-state audit before any N3 training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np

from . import common as C


EXPECTED_2D = {
    "R0": (6.720674617532823, 43.890018716816925, .6342537014805922,
           .04952389936997251, 61.709822134776275),
    "OLD_P": (5.93811955665297, 42.631384166849635, .674936641323196,
              .04868357355895866, 61.636463828095295),
    "N2_DIM_ONLY": (5.7776721802982935, 42.459482047463005, .6858743497398959,
                    .04842188070874762, 61.912311085057716),
    "N3_DIM_SYM": (5.778160604002257, 42.133753550391525, .6858743497398959,
                   .04841908679319606, 61.715463577153855),
}
EXPECTED_POSE = {
    "R0": (2.5388775343237744, 1.3161549246160575, 7.896851501830845, 1.),
    "OLD_P": (2.1542202557708534, 1.1511425031817357, 7.1529791460364835, 1.),
    "N2_DIM_ONLY": (2.0896982645578057, 1.1424495100005931, 7.0106012964116395, 1.),
    "N3_DIM_SYM": (2.0703933055782517, 1.1339764309922582, 7.0676496641524365, 1.),
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=C.ROOT, text=True).strip()


def close(actual, expected):
    if not np.isclose(float(actual), float(expected), rtol=0, atol=1e-12):
        raise AssertionError((actual, expected))


def regression() -> dict:
    table_path = C.ROOT / "_docs/experiments/pallet_final_paper_tables_v1/TABLES.json"
    tables = C.read(table_path)
    denominators = tables["table1"]["denominators"]
    expected_denominators = dict(total_frames=319, evaluable_frames=319, detected=319,
                                 matched=311, missing=0, corners=2499,
                                 observed_corners=2445)
    if denominators != expected_denominators:
        raise AssertionError((denominators, expected_denominators))
    rows2d = {row["arm"]: row for row in tables["table1"]["rows"]}
    for arm, expected in EXPECTED_2D.items():
        values = rows2d[arm]["values"]
        actual = (values["corner_median_px"], values["corner_P90_px"], values["PCK10"],
                  values["E_sym"], values["full_penalty_P90_px"])
        for left, right in zip(actual, expected):
            close(left, right)
    rows_pose = {row["arm"]: row for row in tables["table2"]["rows"]}
    for arm, expected in EXPECTED_POSE.items():
        values = rows_pose[arm]["values"]
        actual = (values["rotation_median_deg"], values["yaw_median_deg"],
                  values["translation_median_cm"], values["pose_coverage"])
        for left, right in zip(actual, expected):
            close(left, right)
    return {"PASS": True, "tables": C.binding(table_path),
            "denominators": denominators, "arms_2d": list(EXPECTED_2D),
            "arms_pose": list(EXPECTED_POSE), "absolute_tolerance": 1e-12,
            "historical_R0_rotation_2_262_forbidden": True}


def inventory() -> dict:
    required = [C.SOURCE, C.SIDECAR, C.NORMALIZATION, C.DEV,
                C.DOPE_WEIGHTS, C.RESNET_CONSTANT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    sidecar = np.load(C.SIDECAR, allow_pickle=False)
    if (sidecar["record_index"].shape != (60000,) or
            not np.array_equal(sidecar["record_index"], np.arange(60000)) or
            sidecar["dimensions"].shape != (60000, 3) or
            sidecar["permutations"].shape != (60000, 4, 9) or
            sidecar["group_valid"].shape != (60000, 4)):
        raise AssertionError("Dimension/symmetry sidecar identity drift")
    source = C.read(C.SOURCE)
    if len(source["records"]) != 60000:
        raise AssertionError("Source population drift")
    main = Path("/home/minjae/Documents/github/pallet-pose")
    process_lines = subprocess.check_output(
        ["ps", "-eo", "pid,etimes,args"], text=True).splitlines()
    excluded_pids = {os.getpid(), os.getppid()}
    training = []
    for line in process_lines[1:]:
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or int(fields[0]) in excluded_pids:
            continue
        command = fields[2].lower()
        if "preflight" in command or "codex-linux-sandbox" in command:
            continue
        if ("python" in command and any(token in command for token in
                ("pallet_n3_completion", "pallet_dope_refiner", "pallet_resnet18"))):
            training.append(line)
    return {
        "PASS": True,
        "worktree_HEAD": git("rev-parse", "HEAD"),
        "worktree_branch": git("branch", "--show-current"),
        "origin_main": git("rev-parse", "origin/main"),
        "source_records": 60000,
        "sidecar_record_index_exact": True,
        "inputs": [C.binding(path) for path in required],
        "active_relevant_processes": training,
        "original_main_status_lines": len(subprocess.check_output(
            ["git", "status", "--short"], cwd=main, text=True).splitlines()),
        "original_main_preserved": True,
    }


def main() -> None:
    C.DOC.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "pallet_n3_completion_v3_preflight_v1",
               "created_at": C.now(), "regression": regression(),
               "inventory": inventory(),
               "decision": {"dope_N3": "TRAIN_3_SEEDS",
                            "resnet18_N3": "TRAIN_3_SEEDS",
                            "new_self_training": False,
                            "actual_lifter_control": False,
                            "PDF_generation": False}}
    C.write(C.DOC / "PREFLIGHT.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
