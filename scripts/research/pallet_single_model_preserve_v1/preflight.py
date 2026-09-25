import subprocess
from . import common as C
def main():
    assert not C.DOC.exists();C.DOC.mkdir(parents=True);C.RAW.mkdir(parents=True);C.OUT.mkdir(parents=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip();assert branch=='main'
    C.freeze(C.RAW/'WORKTREE_START.json',dict(head=head,branch=branch,status=subprocess.check_output(['git','status','--short','--branch'],text=True),created_at=C.now()))
    prior=C.read(C.P.DOC/'INPUT_BINDINGS.json');files=prior['files'][:]
    roots=[C.P.DOC,C.ROOT/'_docs/experiments/pallet_011067_corner_contract_v1',C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1']
    for root in roots:
        files.extend(C.bind(p) for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.md'))
    for b in files:C.verify(b)
    checkpoint=C.read(C.STRUCT/'FIT_PLASTIC_S1.json')['checkpoint'];scorer=C.read(C.P.sdoc(2)/'SCORER_SELECTION_LOCK.json')['checkpoint'];C.verify(checkpoint);C.verify(scorer)
    files.extend([checkpoint,scorer,C.bind(C.STRUCT/'SOURCE_PROBE_PLAN.json'),C.bind(C.P.PREV_RAW/'ANCHOR_POINT_METRICS.json')])
    C.freeze(C.DOC/'INPUT_BINDINGS.json',dict(head=head,branch=branch,files=list({b['path']:b for b in files}.values()),created_at=C.now()))
    C.freeze(C.RAW/'INFERENCE_BINDINGS.json',dict(checkpoint=checkpoint,scorer=scorer))
    C.freeze(C.DOC/'PROTOCOL_LOCK.json',dict(created_at=C.now(),model='S1',selector='frozen GEO_LINEAR',adapter='one2one_cv4_kpts final projection forward-hook per scale: h->Conv1x1(C,32)->SiLU->Conv1x1(32,27), zero last; xy mask only',
        training=dict(seed=42,lr=.001,weight_decay=.0001,batch=16,epochs=5,updates=320,selection='LAST_ONLY',lambda_occ=1.,
            clean_loss='SmoothL1 beta1 in 640-input pixel units on frozen highest-confidence detection decoded xy; sum / valid xy scalar count, v==2. No box matching or confidence threshold in task loss.',
            occ_loss='SmoothL1 beta1 in identical 640-input pixel units, all9 xy at frozen top1 detection, finite only, detached S1 target.',
            batch_composition='4 real-clean +4 synth-clean +8 synth-occluded-preserve; 64 batches/epoch',base_augmentation='reuse immutable original S1 cached augmented RGB/targets; no regenerated base transforms',
            occlusion='original S1 policy functions, supervised synthetic v==2, 640 cached canvas, deterministic hash seed42; no prediction-based placement'),
        preservation_rule='Clean AUC strictly higher AND Moderate/Severe nondecreasing; hard-anchor PCK10 warning separate',
        gap_rule=dict(both_fail_min=6,frames_min=3,corners_min=3,missing_support_points_min=6,severity_view_strata_min=2,trusted_cell_min=2),
        no_new_label_training=True,no_result_based_tuning=True,independent_test=False))
    C.save(C.DOC/'PREFLIGHT_AUDIT.md',f'# S1 preservation preflight\n\nHEAD `{head}` / {branch}. Previous outputs hash-bound and immutable. GPU checked separately. Original S1/GEO_LINEAR frozen. One adapter architecture, one fixed320-step run, no DEV validation during fitting.\n\nUnspecified clean coordinate regression is fixed before training as SmoothL1(beta1) in640-input pixel units at frozen top1 detection. The same units apply to distillation; each term is normalized by its valid xy scalar count. Cached original augmentations and teacher mask are reused.\n')
    print('PREFLIGHT',head,len(files),flush=True)
if __name__=='__main__':main()
