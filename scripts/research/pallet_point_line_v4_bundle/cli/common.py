"""Shared local bindings for the point/line v4 adapter.

Reads verified repository contracts only; never guesses field names.
"""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
KIT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'data/pallet/results/pallet_point_line_v4'
STRUCTURED_V2 = ROOT / 'data/pallet/results/pallet_dht_structured_v2'
DECODER_PROBE = ROOT / 'data/pallet/results/pallet_dht_decoder_probe_v1'
# [확인] read from decoder_probe TRAIN_PROTOCOL.json at runtime; never hardcoded here.
PADDED_FEATURE_HW = {8: (80, 80), 16: (40, 40)}
STRIDES = (8, 16)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(path)


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def backbone_binding():
    """Actual frozen predictor contract, read from the completed extraction."""
    protocol = read(DECODER_PROBE / 'TRAIN_PROTOCOL.json')
    extraction = read(DECODER_PROBE / 'EXTRACTION_PROTOCOL.json')
    checkpoint = Path(protocol['backbone']['checkpoint']).resolve()
    if protocol['backbone']['sha256'] != extraction['checkpoint_sha256']:
        raise ValueError('Backbone SHA disagrees between protocol and extraction')
    return dict(checkpoint=str(checkpoint), sha256=protocol['backbone']['sha256'],
                declared_in=str(DECODER_PROBE / 'TRAIN_PROTOCOL.json'))
