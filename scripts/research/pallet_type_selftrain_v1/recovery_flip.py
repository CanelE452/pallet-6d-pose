"""Bounded horizontal-equivariance training control; no new pseudo-label filtering."""
import argparse
from pathlib import Path
from unittest.mock import patch
import ultralytics.data.augment as augmentation
from . import recovery_common as R
from . import recovery_pose as P
from .recovery_repeat import TracedPoseTrainer

C=R.C
PHASE='pose_flip'
ARMS={f'{target}_FLIP':dict(target=target,lr=1e-5) for target in ['SYN','RAW','REF']}


def prepare():
    parent=C.read(R.DOC/'pose_only/PROTOCOL.json')
    sources=parent['sources']+[C.bound(R.DOC/'pose_only/PROTOCOL.json'),C.bound(__file__),
        C.bound(Path(__file__).with_name('test_recovery_flip.py')),C.bound(Path(__file__).with_name('recovery_repeat.py'))]
    for b in sources+parent['inputs']:C.verify(b)
    args=dict(parent['args'],fliplr=.5)
    assert [k for k in args if args[k]!=parent['args'][k]]==['fliplr']
    protocol=dict(**{k:v for k,v in parent.items() if k not in ['arms','args','sources','controls']},arms=ARMS,args=args,sources=sources,
        intervention='Only fliplr0 -> .5; frozen detector/backbone/BN, pose-only updates at1e-5. Same original datasets/labels/217 unique real images/320steps. No new real GT.',
        horizontal_mapping=[1,0,3,2,5,4,7,6,8],
        coordinate_convention='Installed stock RandomFlip uses continuous x -> width-x (normalized1-x), not the raw-filter pixel-center width-1-x convention. Stock training transform is unchanged.',
        installed_augmentation=dict(path=augmentation.__file__,sha256=C.sha(augmentation.__file__)),
        rationale='Existing source and pseudo training disabled horizontal flips. Test whether training on corresponding flipped poses improves generalization; does not certify or fix90-degree identity mistakes.',
        controls='SYN/RAW/REF with flip share same configuration. Previous SYN_LR5/RAW_LR5/REF_LR5 are no-flip controls; never replace old results.',
        predeclared_decision='Run all3 regardless of intermediate output. Candidate must improve PCK20 over REF_LR5 and R0, preserve MAIN IoU3D and matched8cornerP90 versus REF_LR5 before repetition. Full paper metrics and repeats required before promotion; no threshold/LR sweep here.',
        literature_motivation='https://arxiv.org/abs/2001.07685 motivates learning under transformations of pseudo-labeled inputs; this is not an implementation or claimed reproduction of FixMatch.',
        no_geometry_filter=True,no_new_real_labels=True,no_inference_augmentation=True)
    with R.scope(PHASE):C.freeze(C.DOC/'PROTOCOL.json',protocol)
    R.evaluation_protocol(PHASE,list(ARMS),sources+[C.bound(R.DOC/PHASE/'PROTOCOL.json')])
    print('FLIP_PROTOCOL_LOCKED',list(ARMS),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare',*ARMS]);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:
        with patch.object(P,'PHASE',PHASE),patch.object(P,'PoseOnlyTrainer',TracedPoseTrainer):P.train(a.action)
