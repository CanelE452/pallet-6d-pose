from pathlib import Path
import json
import torch
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as C
from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
from scripts.research.pallet_clean19_pose_sensitive_v1 import common as V

ROOT=C.ROOT
NAME='pallet_clean19_pose_sensitive_diag_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
read=C.read
save=D.save
bind=C.bind
verify=C.verify
MATS=('PLASTIC','WOOD')
ARMS=('M0','M1')
ATOL=1e-5
RTOL=1e-4

def plans():return [json.loads(x) for x in (C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()]

def immutable():
    for b in read(DOC/'INPUT_BINDINGS.json')['files']:verify(b)

def deterministic():
    from ultralytics.utils.torch_utils import init_seeds
    from ultralytics.utils import RANK
    C.H.P.N.setup()
    init_seeds(42+1+RANK,deterministic=True)

def settings():
    import os
    return dict(seed=42,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,AMP=False,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
