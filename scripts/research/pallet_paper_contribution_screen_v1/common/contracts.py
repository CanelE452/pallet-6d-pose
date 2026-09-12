"""Immutable bindings and receipts for the paper contribution screen."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / 'data/pallet/results/pallet_paper_contribution_screen_v1'
DOC = ROOT / '_docs/experiments/pallet_paper_contribution_screen_v1'
R0 = ROOT / 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
R0_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(data, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text()==payload:return
        raise FileExistsError(f'Immutable receipt differs: {path}')
    path.write_text(payload)

def tensor_sha(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
