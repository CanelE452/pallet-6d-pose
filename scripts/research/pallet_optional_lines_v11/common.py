"""Isolated V11 paths and frozen V10 components."""
from pathlib import Path
import sys
import json
import hashlib
import torch

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(KIT.parent/'pallet_point_tokens_v10'))
from data import observations, read, sha, GuidedDHT, point_maps, PARENT, V6, EXPORT
from refiner import Refiner, loss

RUN = ROOT/'data/pallet/results/pallet_optional_lines_v11'
V10 = ROOT/'data/pallet/results/pallet_point_tokens_v10'
CONDITIONS = ('clean', 'occlude20', 'occlude40', 'crop20', 'crop40')
ARMS = ('local_extra', 'lines_frozen', 'lines_joint')
SEEDS = (1, 2, 3)

def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.pending.json')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)

def digest(s): return hashlib.sha256(s.encode()).hexdigest()

def setup():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def dht(seed, device='cuda:0'):
    model = GuidedDHT(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt', seed)
    model.load_state_dict(torch.load(PARENT/'models'/f'points_seed{seed}'/'final.pt', map_location='cpu', weights_only=False)['state_dict'])
    return model.to(device).eval()

def local(seed, device='cuda:0', arm='local'):
    model = Refiner(seed)
    model.load_state_dict(torch.load(V10/'models'/f'{arm}_seed{seed}'/'final.pt', map_location='cpu', weights_only=False)['state_dict'])
    return model.to(device).eval()

def load_observations(split, condition, device='cpu'):
    directory = RUN/'cache'/split/condition
    manifest = read(directory/'manifest.json')
    chunks = [torch.load(directory/p, map_location='cpu', weights_only=True) for p in manifest['chunks']]
    O = {k: torch.cat([c[k] for c in chunks]).to(device) for k in chunks[0]} if chunks else {}
    assert not O or len(O['points']) == sum(r['detected'] for r in manifest['rows'])
    return manifest, O

def frozen_lines(split, condition, seed, device='cpu'):
    p = RUN/'cache'/split/condition/f'lines_seed{seed}.pt'
    return {k: v.to(device) for k, v in torch.load(p, map_location='cpu', weights_only=True).items()}

def subset(O, ii): return {k:v[ii] for k,v in O.items()}
