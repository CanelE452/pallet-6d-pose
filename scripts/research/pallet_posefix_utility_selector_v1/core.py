"""One bounded GPU utility-selector experiment; prior releases stay read-only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scripts.research.pallet_posefix_replay_v1 import core as N

ROOT=N.ROOT
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments/pallet_posefix_utility_selector_v1'
RAW=ROOT/'data/pallet/results/pallet_posefix_utility_selector_v1'
OUT=ROOT/'outputs/pallet_posefix_utility_selector_v1'
GATE_DOC=ROOT/'_docs/experiments/pallet_posefix_corner_gate_v1'
GATE_RAW=ROOT/'data/pallet/results/pallet_posefix_corner_gate_v1'
read=N.E.read
bound=N.E.bound
verify_binding=N.F.verify


def freeze(path,obj):
    path=Path(path).resolve()
    assert any(path.is_relative_to(p) for p in (DOC,RAW,OUT)),path
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert read(path)==obj,('Never overwrite completed artifact',path)
    else:
        with path.open('x') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_npz(path,**arrays):
    path=Path(path).resolve();assert path.is_relative_to(RAW)
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),('Never overwrite completed arrays',path)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)


def setup():
    N.setup()
    torch.set_num_interop_threads(1)


def gpu():
    from scripts.evaluation import final_dimension_release as F
    return F.setup().gpu()


def prepare():
    from .generate import prepare_split
    from .model import UtilitySelector
    N.verify();N.F.checked_lock()
    split=prepare_split();freeze(DOC/'SPLIT.json',split)
    files=['__init__.py','core.py','generate.py','model.py','test_model.py','train.py','evaluate.py']
    for name in files:assert (HERE/name).is_file(),name
    release=N.F.read(N.F.DOC/'MODEL_LOCK.json')
    checkpoints=[release['backbone'],next(r['checkpoint'] for r in release['heads'] if r['arm']=='N2_DIM_ONLY' and r['seed']==1),read(N.DOC/'FIT.json')['checkpoint']]
    sources=[N.F.DOC/'MODEL_LOCK.json',N.DOC/'PROTOCOL.json',N.DOC/'FIT.json',N.RAW/'PREDICTIONS.json',
        GATE_DOC/'RESULTS.json',GATE_DOC/'OUTPUTS_LOCK.json',GATE_RAW/'INFERENCE.json',
        GATE_RAW/'PREDICTIONS.json',GATE_RAW/'PER_FRAME_METRICS.json',
        ROOT/'scripts/research/pallet_posefix_corner_gate_v1/infer.py',
        ROOT/'scripts/research/pallet_posefix_corner_gate_v1/evaluate.py',
        ROOT/'scripts/research/pallet_posefix_large_error_v1/evaluate.py',
        ROOT/'scripts/research/pallet_posefix_replay_v1/source.py']
    model=UtilitySelector(26,64)
    protocol=dict(experiment='pallet_posefix_utility_selector_v1',user_authorized_training=True,
        purpose='Learn which fixed N2/Replay corner candidate reduces error; no generator fine-tuning or paper-model promotion',
        model='Shared local RGB CNN + global RGB/structure + paired endpoint context; outputs eight signed corner utilities',
        parameters=sum(p.numel() for p in model.parameters()),numeric_dim=26,global_dim=64,
        device='cuda',seed=1,steps=1500,batch=32,sampler_seed=8103,split_seed=8101,stress_seed=8102,
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,gradient_clip=5),
        loss='masked SmoothL1(beta=1) on 100*(N2 error - Replay error)/predicted-box diagonal, equal supported-corner weights; no class rebalance',
        target='One approved whole-object symmetry branch minimizing N2 supported mean is chosen ONCE for both candidates; GT supervision only, never input. Out-of-crop GT not dropped.',
        policy='Both endpoints of native vertical pair need predicted utility >0 and finite valid candidates; otherwise exact N2. No tuned threshold, blend, geometry/flip gate, GT reassignment or new coordinate.',
        pairs=[[0,3],[1,2],[4,7],[5,6]],
        variants=['clean','stress'],stress='Two random vertical pairs of R0 inputs each receive one uniform-angle .05-.15 predicted-box-diagonal displacement shared by their valid endpoints. RNG SeedSequence([8102,row]). RGB/boxes/features/centroid unchanged. NO GT used to construct stress inputs. Both frozen N2 and Replay run on each same variant.',
        split='Scenario-group disjoint ~80/20 among 1616 historical synthetic heldout rows after excluding whole Replay-probe scenario groups. Variants never cross split.',
        generator_exposure='No N2/PRIOR/Replay weight-gradient exposure to pool. R0 validation/checkpoint-selection and historical development exposure EXIST; not fully unseen candidate data.',
        C4_training_images=0,real_training_images=0,
        numerical_parity='Recompute clean source N2 with same frozen weights/decode; atol .001 prepared-image px against archived predictions accommodates batch16 cached vs batch1 live floating point. No GT in parity.',
        image_context='Predicted-box crop expanded1.25 at96x72 plus candidate-centered24x24 patches; not entire raw frame. Wrong or too small detector crop remains a limitation.',
        source_validation='Report fixed last1500 only. No early stopping, model/threshold selection, best-checkpoint search or rescue run.',
        real_evaluation='Fixed GREEN150 manual and DEV72 unknown-origin legacy reference, all denominators and detector metadata preserved. Known reused development evaluation, not independent confirmation.',
        real_arms=['A_N2','REPLAY','CORNER_FULL_PAIR','LEARNED_PAIR'],
        real_outputs='Freeze ALL222 GT-free choices before scoring. Real RGB/features first used only after fit checkpoint is fixed.',
        original_models_frozen=True,final_model_modified=False,pseudo_labels_generated=False,auto_promote=False,
        gpu_temperature_limit_C=80,foreign_compute_preserved=True,system_changes=False,
        checkpoint='last1500 only; separate root; fixed one seed',
        split_binding=bound(DOC/'SPLIT.json'),checkpoints=checkpoints,
        sources=[bound(p) for p in sources],code=[bound(HERE/n) for n in files])
    freeze(DOC/'PROTOCOL.json',protocol)
    print(json.dumps(dict(stage='PROTOCOL_LOCKED',parameters=protocol['parameters'],split=split.get('counts')),ensure_ascii=False),flush=True)


def verify():
    p=read(DOC/'PROTOCOL.json')
    for b in [p['split_binding'],*p['checkpoints'],*p['sources'],*p['code']]:verify_binding(b)
    split=read(DOC/'SPLIT.json')
    for b in split.get('bindings',[]):verify_binding(b)
    return p


if __name__=='__main__':prepare()
