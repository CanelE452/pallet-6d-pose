import json
import os
from pathlib import Path
from scripts.research.pallet_pose_objective_followup_v2 import common as V

ROOT = V.ROOT
DOC = ROOT / '_docs/experiments/pallet_gradient_transfer_diagnostic_v1'
RAW = ROOT / 'data/pallet/results/pallet_gradient_transfer_diagnostic_v1'
read, verify, sha, now, clean = V.read, V.verify, V.sha, V.now, V.clean


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                sha256=sha(path), bytes=path.stat().st_size)


def save(path, value, freeze=False):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    text = value if isinstance(value, str) else json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if freeze and path.exists():
        assert path.read_text() == text, path
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)
