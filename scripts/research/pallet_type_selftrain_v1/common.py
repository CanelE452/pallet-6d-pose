"""Per-type R0 student experiment; original releases and evaluation labels immutable."""
import hashlib
import json
from pathlib import Path
from scripts.research.pallet_posefix_replay_v1 import core as N

ROOT=N.ROOT
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments/pallet_type_selftrain_v1'
RAW=ROOT/'data/pallet/results/pallet_type_selftrain_v1'
OUT=ROOT/'outputs/pallet_type_selftrain_v1'
TYPES={'PLASTIC':'plastic_standard_110x130x11','GREEN':'plastic_standard_110x110x15','WOOD':'wood_small_80x59x14'}
read=N.E.read
bound=N.E.bound
verify=N.F.verify


def freeze(path,obj):
    path=Path(path).resolve()
    assert any(path.is_relative_to(r) for r in (DOC,RAW,OUT))
    if path.exists():assert read(path)==obj,('Immutable artifact differs',path)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def write_text(path,text):
    path=Path(path).resolve();assert path.is_relative_to(RAW) or path.is_relative_to(OUT)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert path.read_text()==text
    else:
        with path.open('x') as f:f.write(text)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
