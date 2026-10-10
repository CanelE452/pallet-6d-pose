"""Copy completed actual stage artifacts for public review, without rerunning.

No model weights, source RGB, full mesh, features or private reference targets
are published. Existing files must already have exactly the expected bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
FILES = {
    'downstream': ('SUBSET_INFER_STARTED.json', 'SUBSET_INFER_ADAPTER_RECEIPT.json',
                   'OBSERVATION_SEAL.json', 'LEARNED_OBSERVATIONS.jsonl.gz',
                   'SUBSET_EVALUATE_STARTED.json', 'SUBSET_EVALUATE_ADAPTER_RECEIPT.json',
                   'SUBSET_EVALUATE_FAILURE.json', 'SCORING_RESUME_STARTED.json',
                   'POINT_LINE_CHECKS.json', 'LEARNED_GEOMETRY_CODE_LOCK.json',
                   'LEARNED_GEOMETRY_SEALED.jsonl.gz', 'LEARNED_PREDICTIONS.jsonl.gz',
                   'LEARNED_POSE_EXECUTION.json'),
    'runtime': ('RUNTIME_ADAPTER_STARTED.json', 'RUNTIME_ADAPTER_RECEIPT.json',
                'RUNTIME.json', 'RUNTIME_ROWS.jsonl.gz'),
    'stress': ('REAL_STRESS_STARTED.json', 'REAL_STRESS_EXECUTION.json',
               'REAL_STRESS_ROWS.jsonl.gz'),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def run(stage, source):
    source = Path(source).resolve()
    assert not source.is_relative_to(REPO) and not REPO.is_relative_to(source)
    exported = []
    for name in FILES[stage]:
        original, destination = source / name, DOC / name
        assert original.is_file(), 'Missing actual completed artifact ' + name
        expected = sha(original)
        if destination.exists():
            assert sha(destination) == expected, 'Preserve different existing artifact ' + name
        else:
            shutil.copyfile(original, destination)
        assert sha(destination) == expected
        exported.append(dict(path=name, sha256=expected, bytes=destination.stat().st_size,
                             transfer='byte_exact_actual_artifact'))
    receipt = dict(schema='completed_scoped_artifact_public_export_v1', stage=stage,
                   files=exported, model_forwards=0, PnP_calls=0, updates=0,
                   replay_used_as_fresh_measurement=False)
    path = DOC / (stage.upper() + '_EXPORT.json')
    if path.exists():
        assert json.loads(path.read_text()) == receipt
    else:
        path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(stage.upper() + '_PUBLIC_EXPORT', len(exported), 'byte_exact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=FILES)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    run(args.stage, args.source)
