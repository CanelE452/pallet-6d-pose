"""New-root IO and preflight; no training or target evaluation on import."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch import nn
from scripts.research.pallet_posefix_lora_preservation_v1 import run as L
from scripts.research.pallet_posefix_lora_preservation_v1 import adaptation as A
from scripts.research.pallet_occlusion_refiner_transfer_v2 import run as E
from scripts.research.pallet_occlusion_refiner_transfer_v2 import pilot as P

ROOT=E.ROOT;NAME='pallet_posefix_limited_adaptation_pilot_v1';HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments'/NAME;RAW=ROOT/'data/pallet/results'/NAME;OUT=ROOT/'outputs'/NAME
ARMS=('FULL','SAME_LAYER','LORA');read=L.read;bind=L.bind;state_hash=L.state_hash


def verify(binding):
    actual=bind(ROOT/binding['path'])
    assert actual['sha256']==binding['sha256'],binding['path']
    if 'bytes' in binding:assert actual['bytes']==binding['bytes'],binding['path']


def save(path,obj):
    path=Path(path).resolve();assert any(path.is_relative_to(p) for p in (DOC,RAW,OUT))
    path.parent.mkdir(parents=True,exist_ok=True)
    text=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with path.open('x') as f:f.write(text)


def tensor_save(path,obj):
    assert path.resolve().is_relative_to(RAW);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(obj,f)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def protocol():
    verify(read(DOC/'PROTOCOL_HASH_LOCK.json'));p=read(DOC/'EXPERIMENT_PROTOCOL.json')
    assert p['RUN_APPROVED'] and not p['preserve_loss'] and p['arms']==list(ARMS)
    for b in read(DOC/'CODE_LOCK.json')['files']:verify(b)
    return p


def model(arm,device='cpu'):
    L.setup(device);base=L.load_base();m=A.AdaptedPoseFix(base,arm).to(device)
    expected=read(DOC/'TRAINABLE_CONTRACTS.json')[arm]
    assert A.trainable_contract(m)==expected
    return m


def bn_state(m):
    return {n+'.'+k:v for n,q in m.named_modules() if isinstance(q,nn.BatchNorm2d) for k,v in q.state_dict().items()}


def frozen_state(m):
    return {**{n:p for n,p in m.named_parameters() if not p.requires_grad},**dict(m.named_buffers())}


def trainable_state(m):return {n:p for n,p in m.named_parameters() if p.requires_grad}


def load_inputs():
    lock=read(DOC/'INPUT_LOCK.json')
    for b in lock['paired_files']+[lock['real_order'],lock['source_orders']]:verify(b)
    real=[]
    for b in lock['paired_files']:
        entry=torch.load(ROOT/b['path'],map_location='cpu',weights_only=False)
        real.append(entry['pair']['OCC'])
    order=torch.load(ROOT/lock['real_order']['path'],map_location='cpu',weights_only=True).numpy()
    source=np.load(ROOT/lock['source_orders']['path'])
    return real,order,source


def preflight():
    L.setup();assert not (DOC/'PREFLIGHT.json').exists()
    for p in (RAW,OUT):
        assert not p.exists() or not any(p.iterdir()),('Nonempty namespace collision',p)
        p.mkdir(parents=True,exist_ok=True)
    old=read(E.DOC/'E2_PROTOCOL.json');lock=read(E.DOC/'E2_INPUT_LOCK.json')
    for key in ('input_lock','initialization','source_protocol','code'):verify(old[key])
    prior=read(L.DOC/'AUDIT.json');assert prior['PASS']
    # Previously frozen implementation/source/checkpoints must still be identical.
    for b in prior['artifacts']:verify(b)
    for b in read(L.DOC/'PREFLIGHT.json')['protected']:verify(b)
    assert read(L.DOC/'LORA_PARITY_AUDIT.json')['PASS'] and read(L.DOC/'SYNTHETIC_SANITY.json')['PASS']
    assert list(A.TARGETS)==[r['name'] for r in read(L.DOC/'LORA_TARGET_MODULES.json')['modules']]
    records=lock['records'];pairs=lock['paired_files'];assert len(records)==len(pairs)==253
    for b in pairs+[lock['real_order'],lock['source_orders']]:verify(b)
    # Input graphs reuse prepared OCC tensors; no annotation/evaluation loader.
    for r,b in zip(records,pairs):
        entry=torch.load(ROOT/b['path'],map_location='cpu',weights_only=False);meta=entry['metadata'];pair=entry['pair']
        assert meta==r and meta['R0_actually_rerun']
        assert P.array_sha(np.array(meta['target_original']))==meta['target_sha']
        assert P.array_sha(np.array(meta['mask'],dtype=bool))==meta['mask_sha']
        assert np.array_equal(pair['CLEAN']['target_valid'],pair['OCC']['target_valid'])
        assert not pair['OCC']['target_valid'][8]
        assert np.array_equal(pair['OCC']['target_valid'],np.array(meta['mask'],bool))
        for x in pair.values():
            mapped=E.N.C.transform_points(x['target'],np.linalg.inv(x['matrix']))
            np.testing.assert_allclose(mapped[x['target_valid']],np.array(meta['target_original'])[x['target_valid']],atol=1e-4,rtol=0)
    source=L.SourceData();orders=np.load(ROOT/lock['source_orders']['path'])
    assert orders['source_rows'].shape==(300,8) and len(orders['held_rows'])==256
    assert np.isin(orders['source_rows'],source.train_rows).all() and not np.intersect1d(orders['source_rows'],orders['held_rows']).size
    realorder=torch.load(ROOT/lock['real_order']['path'],map_location='cpu',weights_only=True).numpy()
    assert realorder.shape==(300,8) and realorder.min()>=0 and realorder.max()<253
    baseline=E.V.RAW/'FROZEN_PREDICTIONS.json';oldpredlock=read(E.DOC/'E2_PREDICTIONS_LOCK.json');verify(oldpredlock['baseline'])
    baseline_payload=read(baseline)['predictions'];ids={r['id'] for r in old['eval_records']}
    for arm in ('R0','N2','POSEFIX_SYNTH'):assert ids<=set(baseline_payload[arm])
    baseline_ck=read(E.DOC/'DATA_ROLE_MANIFEST.json')['checkpoint_bindings']
    for b in baseline_ck.values():verify(b)
    assert baseline_ck['POSEFIX_SYNTH']==old['initialization']
    source_lock=read(E.N.DOC/'INPUT_LOCK.json')
    for b in source_lock['cache_bindings']+[source_lock['orders']]:verify(b)
    input_lock=dict(e2_input_lock=old['input_lock'],paired_files=pairs,real_order=lock['real_order'],source_orders=lock['source_orders'],
        source_cache_bindings=source_lock['cache_bindings'],source_pool_rows=len(orders['source_pool']),source_heldout_rows=orders['held_rows'].tolist(),
        pseudo_target_sha=digest([r['target_original'] for r in records]),mask_sha=digest([r['mask'] for r in records]),
        occlusion_plan_sha=digest([r['plans'] for r in records]),OCC_R0_sha=digest([r['occluded_R0_prediction'] for r in records]),
        train_ids=[r['id'] for r in records],frozen_baseline=bind(baseline),baseline_checkpoints=baseline_ck,
        evaluation_protocol=bind(E.DOC/'E2_PROTOCOL.json'),evaluation_symmetry=bind(E.N.C.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'))
    save(DOC/'INPUT_LOCK.json',input_lock)
    m=L.load_base();contracts={};initial=state_hash(m.state_dict())
    for arm in ARMS:
        wrapper=A.AdaptedPoseFix(m,arm);contracts[arm]=A.trainable_contract(wrapper)
        assert state_hash(A.original_state(wrapper))==initial
    save(DOC/'TRAINABLE_CONTRACTS.json',contracts)
    p=dict(name=NAME,RUN_APPROVED=True,authorization='Explicit user decision-pilot instruction 9b891900; FULL/SAME_LAYER/LoRA only',
        arms=list(ARMS),base=old['initialization'],seed=1,optimizer='TFAdam',lr=1e-4,betas=[.9,.999],eps=1e-8,
        updates=300,real_batch=8,source_batch=8,microbatch=2,real_exposures=2400,source_exposures=2400,
        rank=4,alpha=4,target_modules=list(A.TARGETS),BN='running statistics + affine frozen ALL arms',
        real_mode='OCC only',pseudo_target=old['common_targets'],augmentation=old['augmentation'],source_corruption_rng=7103,
        source_probe_rng=7104,source_probe_heldout=256,preserve_loss=False,head_only=False,selector=False,student_training=False,
        regularization=dict(original_conv_weight_L2_half_coefficient=.5e-5,adapter_penalty=0,
            reason='Preserve previously audited explicit original_weight_l2; actual legacy recursion WOULD include Conv2d A/B, deliberately excluded. Different gradient scopes; NOT same strength.',
            contribution='once per update over real microbatches; none over source microbatches'),
        checkpoint='last300 only; no validation selection; exclusive create',no_rescue_run=True,no_eval_GT_in_training=True,
        inputs=bind(DOC/'INPUT_LOCK.json'),evaluation_populations=old['populations'],eval_records=old['eval_records'],
        symmetry=input_lock['evaluation_symmetry'],matching='reuse immutable R0 matching/detection; predictions cannot change boxes/score',
        denominator='primary93/713 incl 8 match-fail frames54 corners penalty800; matched median/P90 exclude unmatched',
        pilot_gates=dict(pck10_vs_FULL_pp_min=-.5,B3_recovery_vs_FULL_min=.8,primary_reference='BASE',
            reference_lost='strictly fewer BASE-correct losses than FULL; N2 secondary also reported',source_clean_delta_vs_BASE_pp_min=-1,
            P90_vs_FULL_ratio_max=1.05,GREEN='BASE-correct loss <= FULL',zero_FULL_recovery='undefined, no automatic pass',
            strong='PCK10>=SAME_LAYER and either fewer BASE-correct loss or Pareto recovery/loss advantage',
            equivalent=dict(PCK10_abs_pp_max=.5,B3_recovery_count_abs_max=1,BASE_correct_loss_abs_max=1),
            statistical_significance=False),
        route_priority=['LIMITED_ADAPTATION_ONLY if both pass plus equivalent','CONTINUE_LORA if LoRA passes',
            'ADD_PRESERVE_OBJECTIVE if ALL arms add B3 recovery over BASE and ALL have >0 B1/B2 BASE-correct losses',
            'MOVE_TO_REPRESENTATION_GEOMETRY if both limited arms B3 recovery<=BASE and <=1',
            'otherwise INCONCLUSIVE; DATA_DIVERSITY requires strong additional evidence, not automatic'],
        blind_review='REVIEW_PENDING; no external occlusion claim',thermal_stop_C=80,automatic_commit=False,automatic_push=False)
    save(DOC/'EXPERIMENT_PROTOCOL.json',p);save(DOC/'PROTOCOL_HASH_LOCK.json',bind(DOC/'EXPERIMENT_PROTOCOL.json'))
    protected={ROOT/b['path'] for b in read(L.DOC/'PREFLIGHT.json')['protected']}
    protected|={ROOT/b['path'] for b in prior['artifacts']}
    protected|={ROOT/b['path'] for b in source_lock['cache_bindings']}
    protected|={baseline,E.RAW/'E2_FRAME_METRICS.json',E.V.RAW/'E1_FRAME_METRICS.json',E.DOC/'SOURCE_BEFORE.json'}
    protected|={ROOT/r[k]['path'] for r in old['eval_records'] for k in ('image','annotation')}
    save(DOC/'PREFLIGHT.json',dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),
        protected=[bind(f) for f in sorted(protected)],all_provenance_resolved=True,initial_model_state_sha=initial,
        train_unique=253,real_augmentation_count=sum(bool(r['plans']) for r in records),eval_frames=len(ids),
        population_counts={k:len(v) for k,v in old['populations'].items()},missing=[]))
    save(DOC/'PREFLIGHT_AUDIT.md',f'''# Preflight PASS

main / HEAD `{read(DOC/'PREFLIGHT.json')['HEAD']}`. 기존 미추적 작업 유지. PRIOR1/checkpoint, LoRA targets/parity/sanity, E2 input/order/target/mask/augmentation, source TRAIN/heldout/order, N2/BASE frozen predictions 모두 실물과 hash 검증. MISSING 없음.

253개 paired 파일의 metadata 및 공통mask, inverse-crop target 일치 검증. 인공가림185장, 나머지는 기존 recipe의 no-patch branch. 새 pseudo-label·augmentation·R0 추론을 생성하지 않는다. 기존253장 OCC tensor를 그대로 재사용한다.

모든 군 BASE tensor SHA 동일, 계약상 FULL {contracts['FULL']['trainable']:,}, SAME_LAYER {contracts['SAME_LAYER']['trainable']:,}, LORA {contracts['LORA']['trainable']:,}. BASE는N2가 아닌PRIOR1. 이전A11은참조전용.

source order는TRAIN만, heldout256과분리. 실제eval278장=non-green128+GREEN150. PRIMARY93, CLEAN17, CAD18. 평가/학습recording 제외는 기존E2그대로. 평가whole-object 계약 불변. 가림subtype 미확인.

이전preserve lambda/mapping PENDING은 preserve를 사용하지 않는 이번실험의차단조건이아니다. 입력/타깃의 기존supervised native계약은변경하지않는다. 학습전 PRETRAIN_TESTS PASS까지 실행금지.
''')
    print('PREFLIGHT_PASS',len(protected),flush=True)


if __name__=='__main__':preflight()
