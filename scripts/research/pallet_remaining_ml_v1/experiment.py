"""Shared contracts for the two remaining bounded ML experiments."""
import importlib.util
import os
import subprocess
from pathlib import Path

_path=Path(__file__).resolve().parents[1]/'pallet_active_learning_v1/simulation_evaluate.py'
_spec=importlib.util.spec_from_file_location('remaining_previous_eval',_path)
E=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(E)
S=E.S
ROOT,P,read,write,sha=S.ROOT,S.P,S.read,S.write,S.sha
RAW=ROOT/'data/pallet/results/pallet_remaining_ml_v1'
DOC=ROOT/'_docs/experiments/pallet_remaining_ml_v1'
METHODS=('uniform','hard_loss','meta_weight')
SEEDS=(1,2,3)

def check_gpu():
    value=S.snapshot()
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
    assert not [v for v in lines if v.split(',')[0].strip()!=str(os.getpid())],'Foreign GPU process: do not terminate or wait'
    return value

def lock():
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    assert sha(S.R0)==S.R0_SHA
    original=read(S.DOC/'SPLIT.json')
    train,meta=S.split_records(original['pool'])
    parts=dict(train=train,meta_calibration=meta,evaluation=original['evaluation'])
    for a,x in parts.items():
        for b,y in parts.items():
            if a<b:
                assert not {r['image_sha256'] for r in x}&{r['image_sha256'] for r in y}
                assert not {r['capture_session'] for r in x}&{r['capture_session'] for r in y}
    write(DOC/'SPLIT.json',parts)
    sources=[S.R0,S.DOC/'SPLIT.json',S.DOC/'FINAL_AUDIT.json',S.SYNTH,
        Path(S.__file__),Path(E.__file__),Path(P.E.__file__),
        ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
        ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json',
        P.POSE/'GEOMETRY_RESOLVED_POSE_GT.json',P.POSE/'AXIS_REVIEW_MANIFEST.json',P.POSE/'POSE_EVAL_OBJECT_CONTRACT.json']
    write(DOC/'PROTOCOL_LOCK.json',dict(status='LOCKED_BEFORE_NEW_TRAINING',
        start_main=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_bindings={str(p.relative_to(ROOT)):sha(p) for p in sources},
        split_counts={k:len(v) for k,v in parts.items()},split_rule='Reuse existing session-connected pool/eval; apply same deterministic object-stratified group split within pool for train/meta.',
        evidence='RETROSPECTIVE_DEVELOPMENT; historical evaluation already inspected; no independent confirmation or novelty claim',
        sample_importance=dict(methods=list(METHODS),seeds=list(SEEDS),updates_per_fit=300,fits=9,total_student_updates=2700,
            synthetic_batch=24,real_batch=8,meta_batch=4,epochs=10,updates_per_epoch=30,
            initialization='Identical frozen-source R0; all parameters train except DFL; all BatchNorm running statistics frozen in every arm',
            data='All train-subset existing GT-v2 labels; disjoint meta labels only for weight learning; same source sets and schedules in controls',
            weighting='Per-image stock E2ELoss(PoseLoss26), includes vis1. Uniform=1; hard_loss=nonnegative current loss normalized to sum8; meta_weight=positive dot product of train and meta gradients on final keypoint coordinate projections, normalized to sum8. All-negative meta alignment gives zero real weight, synthetic replay continues.',
            derivation='Negative derivative of clean meta loss wrt zero-initialized example weights for virtual SGD in keypoint-projection subspace; first-order alignment, not full-parameter Ren reproduction or learned MLP.',
            proxy_parameters='Both cv4_kpts and one2one_cv4_kpts weights/biases, all three scales. Gradient alignment only; real weighted loss updates full trainable model.',
            fairness='Every arm computes same meta/proxy diagnostics, no meta optimizer update; actual augmented input hashes checked across arms. Real/meta mosaic and mixing disabled for unambiguous sample identity; synthetic recipe unchanged.',
            checkpoint='Last update only; no EMA, validation-best, retry or sweep',
            gate='Meta improves seed-mean three-arm common-frame kp median vs BOTH uniform/hard_loss; >=2/3 seeds each; paired P90/gross20 nonworse, AP50-95 and Det nonworse, translation/yaw <=1.1x and pose coverage nonworse'),
        pose_acceptance=dict(source='Frozen R0 only, no selection of best student',
            methods=['confidence','negative_reprojection','logistic','mlp_seed1','mlp_seed2','mlp_seed3'],
            learning='Logistic and one-hidden-layer32 ReLU MLP; full-batch Adam lr0.001 wd0.001 for1000updates; BCE unsafe label, standardize on train only; no class rebalancing',
            target='Unsafe if missing/invalid pose or IoU<0.5 or MAIN translation>10cm or yaw>5degrees. Geometry-reconstructed reference; exploratory operational tolerance, NOT safety certification.',
            inputs='Prediction-only score, image-normalized box/keypoint geometry, known intrinsics/object specification and prediction-only PnP residual, hypothesis gap, predicted translation/rotation. No GT errors, GT physical axis, session ID or filename features.',
            population='Positive pallet frames with known camera/object specification only; background/negative deployment acceptance NOT_TESTED',
            threshold='Calibration only: maximum accepted count at empirical unsafe fraction<=0.10 with at least10 accepted; whole score ties accepted together; no feasible threshold -> reject all',
            evaluation='145 fixed frames; report risk-coverage curves and tie-invariant AURC; threshold fixed before test labels opened; no oracle deployed',
            gate='MLP mean AURC beats confidence AND logistic, >=2/3 seeds each; all seeds test accepted coverage>=0.30 and unsafe risk<=0.10',
            budget='One logistic and3 MLPs x1000=4000 gate optimizer updates; R0 updates0'),
        additional_training_after_results=False,old_results_preserved=True))
    write(DOC/'GPU_PREFLIGHT.json',check_gpu())
    print('LOCKED', {k:len(v) for k,v in parts.items()},flush=True)

if __name__=='__main__':lock()
