"""Prepare and verify fresh C source locks without repeating extraction/fits.

Historical phased locks describe actual old executions. A clean replay of the
published final source needs its own current-source extraction and fitting
locks. This helper creates those in an empty private code snapshot, preserving
all published outputs and historical locks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.research.pallet_wd_info_fast_20261011 import common as C

PREFIX = Path("scripts/research/pallet_wd_info_fast_20261011")
DOC_REL = Path("_docs/experiments/pallet_wd_info_fast_20261011")


def run(fresh_root):
    started = time.monotonic()
    C.lock_ready()
    fresh_root = fresh_root.resolve()
    assert fresh_root.is_relative_to(C.PRIVATE), "Fresh replay snapshot must remain private"
    assert not fresh_root.exists(), "Existing fresh-root is preserved; choose an empty path"
    prior_bindings = [C.binding(C.DOC / name) for name in
        ("C_CODE_LOCK.json", "C_CODE_FINAL_LOCK.json", "C_IO_PREFIT_LOCK.json", "C_FITS.json", "C_RESULTS.json", "C5_SUBGROUPS_SUPPLEMENT.json")]
    prior_code = C.binding(ROOT / PREFIX / "classifier.py")
    (fresh_root / PREFIX).mkdir(parents=True)
    for source in sorted((ROOT / PREFIX).rglob("*")):
        if source.is_file() and (source.suffix == ".py" or "assets" in source.parts):
            target = fresh_root / PREFIX / source.relative_to(ROOT / PREFIX)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    replay_doc = fresh_root / DOC_REL
    replay_doc.mkdir(parents=True)
    for name in ("PLAN_LOCK.json", "INPUT_CONTRACT.json"):
        shutil.copyfile(C.DOC / name, replay_doc / name)
    replay_private = fresh_root / "private"
    replay_private.mkdir()
    environment = dict(os.environ, PALLET_FAST_PRIVATE=str(replay_private), PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    completed = subprocess.run([sys.executable, "-B", str(fresh_root / PREFIX / "classifier.py"), "lock"],
        cwd=fresh_root, env=environment, text=True, capture_output=True, check=True)
    source_lock = C.read(replay_doc / "C_CODE_LOCK.json")
    assert source_lock["code"]["sha256"] == prior_code["sha256"]
    assert not (replay_doc / "C_CODE_FINAL_LOCK.json").exists()
    prefit_lock = dict(source_lock, status="CLEAN_REPLAY_LOCKED_BEFORE_EIGHT_FITS",
        replay_preparation="Final published source; historical extraction/reporting/IO phase locks not copied",
        fitting_calls=0, optimizer_updates=0, PnP_calls=0, PnP_calls_already_executed=0)
    C.write(replay_doc / "C_IO_PREFIT_LOCK.json", prefit_lock)
    # Exercise the same concrete source/PLAN guards as extract() and fit().
    check = """import json
from pathlib import Path
from scripts.research.pallet_wd_info_fast_20261011 import common as C, classifier as s
C.lock_ready()
source = Path(s.__file__)
extract_name = 'C_CODE_FINAL_LOCK.json' if (C.DOC / 'C_CODE_FINAL_LOCK.json').exists() else 'C_CODE_LOCK.json'
assert C.sha(source) == C.read(C.DOC / extract_name)['code']['sha256']
assert C.sha(source) == C.read(C.DOC / 'C_IO_PREFIT_LOCK.json')['code']['sha256']
assert not (C.PRIVATE / 'C_SYNTH_PREDICTIONS_SEALED.npz').exists()
assert not (C.DOC / 'C_FITS.json').exists()
print(json.dumps({'PLAN_guard': 'PASS', 'extract_source_guard': 'PASS', 'fit_source_guard': 'PASS', 'source_sha256': C.sha(source), 'fits': 0, 'backbone_forwards': 0, 'OpenCV_calls': 0}))
"""
    verified = subprocess.run([sys.executable, "-B", "-c", check], cwd=fresh_root,
        env=environment, text=True, capture_output=True, check=True)
    guard_receipt = json.loads(verified.stdout)
    assert guard_receipt["source_sha256"] == prior_code["sha256"]
    for binding in prior_bindings:
        assert C.sha(ROOT / binding["path"]) == binding["sha256"]
    assert C.sha(ROOT / prior_code["path"]) == prior_code["sha256"]
    C.write(C.DOC / "C_CLEAN_REPLAY_PREPARATION.json", dict(status="FRESH_SOURCE_LOCK_GUARDS_ACTUALLY_EXECUTED_PASS",
        code=C.binding(Path(__file__)), classifier_source=prior_code, prior_published_outputs_unchanged=prior_bindings,
        fresh_extraction_lock=C.binding(replay_doc / "C_CODE_LOCK.json"), fresh_fit_lock=C.binding(replay_doc / "C_IO_PREFIT_LOCK.json"),
        actual_guard_receipt=guard_receipt, fresh_root_relative_to_private=str(fresh_root.relative_to(C.PRIVATE)),
        executed_commands=["$PALLET_PYTHON -B $PALLET_C_REPLAY_ROOT/scripts/research/pallet_wd_info_fast_20261011/classifier.py lock",
            "$PALLET_PYTHON -B -c '<same PLAN/extract/fit source guards; no extract/fit invocation>'"],
        scope="Concrete clean lock preparation only. Does not claim a second full experiment, extraction, fit, or solver replay.",
        new_fits=0, new_CNN_forwards=0, new_OpenCV_calls=0, wall_seconds=time.monotonic()-started,
        historical_phase_locks_preserved=True, repeated_experiment_authorized=False))
    # Verify the missing high-elevation TEST stratum from already sealed arrays.
    with np.load(C.PRIVATE / "C_SYNTH_PREDICTIONS_SEALED.npz", allow_pickle=False) as archive:
        split = archive["split"]
        ids = archive["ids"]
    with np.load(C.PRIVATE / "C_SYNTH_LABELS.npz", allow_pickle=False) as archive:
        label_ids = archive["ids"]
        elevation = archive["elevation"]
    assert np.array_equal(ids, label_ids) and len(ids) == 6144
    counts = {name: dict(frames=int((split == name).sum()),
        elevation_finite=int(((split == name) & np.isfinite(elevation)).sum()),
        elevation_ge30=int(((split == name) & (elevation >= 30)).sum())) for name in ("TRAIN", "VAL", "TEST")}
    assert counts["TRAIN"]["elevation_ge30"] == 1854 and counts["VAL"]["elevation_ge30"] == 2
    assert counts["TEST"]["elevation_ge30"] == 0
    C.write(C.DOC / "C5_CONTRACT_DETAILS_SUPPLEMENT.json", dict(status="HIGH_ELEVATION_TEST_EMPTY_AND_SIZE_BIN_CONTRACT_ABSENT",
        code=C.binding(Path(__file__)), counts=counts,
        sealed_predictions=C.binding(C.PRIVATE / "C_SYNTH_PREDICTIONS_SEALED.npz"),
        sealed_labels=C.binding(C.PRIVATE / "C_SYNTH_LABELS.npz"),
        high_elevation_test_metrics=None,
        high_elevation_test_reason="Original group-disjoint fixed TEST contains zero >=30 degree frames. No repartitioning or new threshold selection.",
        size_subgroups=C.binding(C.DOC / "C5_SUBGROUPS_SUPPLEMENT.json"),
        size_reason="Continuous runtime bbox-area feature exists, but no prelocked size boundaries; all methods retain explicit NA.",
        clean_replay_preparation=C.binding(C.DOC / "C_CLEAN_REPLAY_PREPARATION.json"),
        new_GT_reads=0, new_raw_RGB_reads=0, protected_per_ID_reads=0, new_fits=0, new_CNN_forwards=0, new_OpenCV_calls=0))
    print("C_CLEAN_REPLAY_LOCKS_DONE", guard_receipt, counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-root", type=Path, required=True)
    run(parser.parse_args().fresh_root)
