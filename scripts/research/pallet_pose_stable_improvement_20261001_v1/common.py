"""Isolated output namespace; immutable input and experiment contracts."""
from pathlib import Path
import hashlib
import json
import time
from datetime import datetime,timezone
import numpy as np
from scripts.research.pallet_pose_diagnosis_20260930_v1 import run as D
from scripts.research.pallet_posefix_limited_adaptation_pilot_v1 import common as L

ROOT=D.ROOT
NAME='pallet_pose_stable_improvement_20261001_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
HERE=Path(__file__).resolve().parent
ARMS=('SINGLE251','DIVERSE251')
SEEDS=(1,2,3)
read=D.read
bind=D.bind
verify=D.verify

def now():return datetime.now(timezone.utc).isoformat()

def save(path,value):
    path=Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=value if isinstance(value,str) else json.dumps(D.M.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text()==encoded,('Refusing overwrite',path)
        return
    with path.open('x') as handle:handle.write(encoded)

def tensor_save(path,value):
    assert Path(path).resolve().is_relative_to(RAW)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('xb') as handle:D.torch.save(value,handle)

def protocol():
    verify(read(DOC/'PROTOCOL_SHA.json'))
    verify(read(DOC/'EFFECTIVE_PROTOCOL_SHA.json'))
    p=read(DOC/'PROTOCOL_EFFECTIVE.json')
    assert p['arms']==list(ARMS) and p['training_seeds']==list(SEEDS)
    return p

def amend_before_training():
    assert not list((RAW/'fits').glob('*/final.pt'))
    original=read(DOC/'GOAL_PROTOCOL.json')
    failed=read(DOC/'DATA_PREPARATION_GPU.json')
    amendment=dict(created_at=now(),stage='Input preparation only; no new fit or evaluation scoring',
        original=bind(DOC/'GOAL_PROTOCOL.json'),preparation=bind(DOC/'DATA_PREPARATION_GPU.json'),
        reason='One preselected training pair failed same-object IoU>=0.5. Preserve failed252 selection and guard; no replacement or evaluation-dependent selection.',
        failed_diverse_id='PLASTIC__10957de42de4bd188099b48c43743677039e60825001704bf19e1ca5e7ff3ff5',
        failed_pair_IoU=0.461980212,
        single_removal_rule='Lexicographically maximum (actual RGB sha256,id) in originalSINGLE252',
        single_removed_id='DAY264__005627',single_removed_sha256='ff9753762d8b57485072fc4e0cd3ee32012f11f0f928d784837aaaad36141eb3',
        arms=list(ARMS),sample_size=251,no_resampling=True,no_guard_relaxation=True,
        unchanged='Initialization, teacher, losses, optimizer, source/order seeds,6fits×300steps, evaluation contracts and all stability gates remain identical.')
    save(DOC/'PROTOCOL_AMENDMENT_01.json',amendment)
    effective=json.loads(json.dumps(original).replace('SINGLE252','SINGLE251').replace('DIVERSE252','DIVERSE251'))
    effective['arms']=list(ARMS);effective['sample_size']=251
    effective['changed_factor']=effective['changed_factor'].replace('recording252','recording251').replace('balanced252','balanced251')
    effective['pair_integrity']=effective['pair_integrity'].replace('selected252','selected251')
    effective['amendment']=bind(DOC/'PROTOCOL_AMENDMENT_01.json')
    effective['original_protocol']=bind(DOC/'GOAL_PROTOCOL.json')
    save(DOC/'PROTOCOL_EFFECTIVE.json',effective)
    save(DOC/'EFFECTIVE_PROTOCOL_SHA.json',bind(DOC/'PROTOCOL_EFFECTIVE.json'))
    print('PROTOCOL_AMENDED_BEFORE_TRAINING',list(ARMS),flush=True)

def make_protocol():
    import subprocess
    baseline=read(D.DOC/'MODEL_LINEAGE.json')
    oldfit=read(L.DOC/'FIT_FULL.json')
    save(DOC/'GOAL_PROTOCOL.json',dict(created_at=now(),objective='T와 R이 함께 안정적으로 개선되는 방법을 분석·구현·검증한다. 이 pilot의 완료를 전체 목표 달성으로 대체하지 않는다.',
        previous_goal_turn='The preceding diagnosis/publication provided evidence, but did not demonstrate stable FULL125 improvement. This is a new improvement phase.',
        authorization='User active goal requests analysis and solution for stable joint improvement; previous zero-new-fit limit described the completed diagnosis phase. This new bounded matched training attempt adds no new manual annotation.',
        supervision='Keep current teacher and existing synthetic/manual teacher lineage; no new real reference labels for optimization. Replay teacher already used38manualcorners/9images, so do not call the inherited method wholly real-GT-free.',
        local_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),previous_GitHub_commit='babef3568a02118f4e9fc5ab4fd07ad24aff2757',
        parent_diagnosis=bind(D.DOC/'PUBLICATION_MANIFEST.json'),arms=list(ARMS),training_seeds=list(SEEDS),
        changed_factor='Training recording composition: one DAY recording252 vs max-min balanced252 across eligible existing unlabeled recordings. Same fixed pseudo-target recipe, optimizer/update/source budget, augmentation family, initialization and BN policy.',
        motivation='Full125 restores large coordinate errors on clean29 but natural/occluded-RGB transfer uncertain. Previous SINGLE32/MULTI32 was small two-recording2D pilot with mixed clean/hard outcomes. Full-sized multirecording6D comparison with three training seeds not established.',
        prior_negative_results=['LoRA/SAME_LAYER and source preservation not adopted after failed/mixed controls','Final Huber/confidence-weighted PnP already tested and rejected historically; not claim a novel unused fix','Coordinate direction correlation alone not supported by current29 stress'],
        sample_size=252,exclude_original_DAY_id='DAY264__005516',excluded_reason='Exact SHA overlap with PAPER194; both new arms exclude all PAPER319 RGB hashes. LegacyFULL253 retained as historical baseline, not exact replication.',
        input_selection='Before GPU pairing/output evaluation: exact cached R0/Replay stage1 survivors, same hidden-only selfocclusionPnP and all8LOO target checks, standard-pallet type only, max-min recording allocation then deterministic SHA/ID ordering. No outcome-based resampling.',
        pair_integrity='Same object clean/OCC selected-box IoU>=0.5 and existing common crop target support; failure explicit, no clean-coordinate substitution. Training pauses if frozen selected252 cannot be prepared.',
        initialization=baseline['models']['PRIOR1']['checkpoint'],train_mode='FULL convolutional parameters, frozen BN running statistics and affine',
        optimizer=dict(name='TFAdam',lr=1e-4,betas=[.9,.999],eps=1e-8,original_conv_L2_half_coefficient=.5e-5),
        updates=300,real_batch=8,source_batch=8,microbatch=2,real_exposures_per_fit=2400,source_exposures_per_fit=2400,
        real_order='numpy default_rng(6400+training_seed),300x8 uniform integer sampling; identical indices for both arms',
        source_order='Existing locked source_rows300x8 unchanged across both arms and seeds; heldout256 excluded',
        source_corruption='Existing normal/stress recipe, numpy default_rng(7102+training_seed); identical to paired arm',
        checkpoint='Last300 only; no development checkpoint/seed/hyperparameter selection',
        primary_population='Current full128 split as natural99 and clean29, identical IDs/reference/GEO contracts to diagnosis. Repeated DEV; fitting/teacher-recording disjoint.',
        secondary_population='Wood45 existing DEV:38clean+7moderate,2recordings disjoint from new fitting/teacher. Shape-transfer stress only, weak2cluster uncertainty and reference provenance limits. No selecting/training on its result.',
        rejected_confirmation_population='Plastic66 allREC001/REC002 exposed to fitting/teacher and one directDAY image overlap; not independent confirmation.',
        primary_metrics=['natural99 median centerT_cm','natural99 median physical-frame C2 rotation_deg'],
        stability_criteria=dict(
            all_three_seeds_joint_gain='DIVERSE252 each seed has smaller natural99 T and R medians than pairedSINGLE252 and fixedR0/PRIOR1/FULL125. No best-seed reporting.',
            joint_uncertainty='For DIVERSE-SINGLE and DIVERSE-R0, paired recording-and-training-seed bootstrap2000/seed20261001 of mean per-seed population medians has upper95% endpoint<0 for BOTH T and R.',
            recording_sensitivity='Leaving out each of6naturalrecordings retains negative mean-seed median differences for BOTH metrics versus R0 and pairedSINGLE.',
            natural_tails='Mean acrossseed P90 T/R no more than5% above R0 and pairedSINGLE. Full pose failure count must not increase.',
            clean_preservation='Mean acrossseed clean29 median and P90 T/R no more than5% aboveR0; pose failure count notincrease.',
            tolerance_status='The5% guards are provisional engineering non-regression limits, not an empirical noise floor/MDE or evidence of statistical equivalence.',
            strong_generalization='Not claimed from reusedDEV,3seeds and6recordings. Independent new-recording confirmation remains a separate evidentiary limitation.'),
        budget=dict(max_new_fits=6,max_optimizer_updates=1800,masked_R0_preparation_forwards=253,
            new_model_inference_max=6*(128+45),frozen_PRIOR1_FULL125_wood_baselines_max=90,new_manual_labels=0,new_capture=0),
        resource_evidence=dict(host_GPU='RTX3080 10GiB verified accessible outside sandbox; no driver/config modifications',historical_full300_seconds=oldfit['seconds'],time_guarantee=False),
        stopping='One predeclared two-arm/three-seed attempt. Failed gates remain failures; no seed extension/threshold tuning. Diagnose failed mechanism before choosing further work under active goal.',
        completion='Goal remains active unless a deployable non-oracle method meets the original stable joint-improvement objective with adequate evidence. Protocol/artifact completion alone is not goal completion.'))
    save(DOC/'PROTOCOL_SHA.json',bind(DOC/'GOAL_PROTOCOL.json'))
    print('GOAL_PROTOCOL_LOCKED',flush=True)

if __name__=='__main__':make_protocol()
