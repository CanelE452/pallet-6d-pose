"""Bounded implementation audit. NO full-pilot training entry point."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch import nn
from .adaptation import (TARGETS, AdaptedPoseFix, PointwiseLoRA, trainable_contract,
    original_state, supervised_loss, original_weight_l2, preserve_mask, preservation_kl)
from scripts.research.pallet_posefix_large_error_v1 import core as C
from scripts.research.pallet_posefix_large_error_v1.train import corrupted
from scripts.research.pallet_posefix_replay_v1.source import SourceData
from scripts.research.pallet_posefix_replay_v1 import core as REPLAY
from scripts.research.pallet_sensors_submission_v1.prior_model import expectation, TFAdam

ROOT=C.ROOT; NAME='pallet_posefix_lora_preservation_v1'
DOC=ROOT/'_docs/experiments'/NAME; RAW=ROOT/'data/pallet/results'/NAME; OUT=ROOT/'outputs'/NAME
HERE=Path(__file__).resolve().parent
OLD=ROOT/'_docs/experiments/pallet_occlusion_refiner_transfer_v2'
PREV=ROOT/'_docs/experiments/pallet_oracle_visibility_preservation_v1'
read=lambda p:json.loads(Path(p).read_text())


def bind(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),sha256=h.hexdigest(),bytes=path.stat().st_size)


def freeze(path,obj):
    path=Path(path).resolve()
    if not any(path.is_relative_to(p) for p in (DOC,RAW,OUT)):raise ValueError('Outside new namespace')
    path.parent.mkdir(exist_ok=True,parents=True)
    text=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with path.open('x') as f:f.write(text)


def tensor_save(path,obj):
    if not path.resolve().is_relative_to(RAW):raise ValueError('Outside new namespace')
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('xb') as f:torch.save(obj,f)


def verify(b):
    assert bind(ROOT/b['path'])==b,b['path']


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def setup(device='cpu'):
    torch.set_num_threads(4)
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if device=='cuda':
        assert torch.cuda.is_available(),'CUDA unavailable: stop, do not silently CPU train'
        torch.cuda.manual_seed_all(1);gpu_guard()


def gpu_guard():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    temp,used,total=map(int,s.splitlines()[0].split(','))
    assert temp<80,('Thermal stop',temp)
    return dict(temperature_C=temp,memory_used_MiB=used,memory_total_MiB=total)


def load_base():
    # CPU strict load, exact tensor equality to checkpoint, not just missing-key check.
    ck=torch.load(C.PRIOR_CK,map_location='cpu',weights_only=False)
    assert ck['complete'] and ck['step']==6000
    assert bind(C.PRIOR_CK)['sha256']==read(C.PRIOR_DOC/'PRIOR_SELECTION.json')['checkpoints']['1']
    m=C.PoseFixPallet9();m.load_state_dict(ck['model_state_dict'],strict=True)
    assert all(torch.equal(v,ck['model_state_dict'][k]) for k,v in m.state_dict().items())
    m.eval().requires_grad_(False)
    return m


def preflight():
    setup();assert not (DOC/'PREFLIGHT.json').exists()
    for p in (RAW,OUT):
        assert not p.exists(),('Namespace collision',p)
        p.mkdir(parents=True)
    protected={ROOT/b['path'] for b in read(PREV/'PREFLIGHT.json')['sources']}
    protected|={p for base in (PREV,OLD) for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    old=read(OLD/'E2_PROTOCOL.json');lock=read(OLD/'E2_INPUT_LOCK.json')
    for key in ('input_lock','initialization','source_protocol','code'):verify(old[key]);protected.add(ROOT/old[key]['path'])
    for b in lock['paired_files']+[lock['real_order'],lock['source_orders']]:verify(b);protected.add(ROOT/b['path'])
    frozen_arms={}
    for arm in ('A10','A11'):
        fit=read(OLD/f'FIT_{arm}.json');verify(fit['checkpoint'])
        pred=ROOT/f'data/pallet/results/pallet_occlusion_refiner_transfer_v2/PREDICTIONS_{arm}.json'
        protected|={ROOT/fit['checkpoint']['path'],pred}
        frozen_arms[arm]=dict(checkpoint=fit['checkpoint'],predictions=bind(pred))
    for base in ('pallet_posefix_replay_v1','pallet_posefix_large_error_v1','pallet_occlusion_refiner_transfer_v2'):
        protected|={p for p in (ROOT/'scripts/research'/base).glob('*.py')}
    source=SourceData();orders=np.load(REPLAY.RAW/'ORDERS.npz')
    assert np.isin(orders['source_rows'],source.train_rows).all()
    assert not np.intersect1d(orders['source_rows'],orders['held_rows']).size
    protected|={source.data.run_dir/'SOURCE_MANIFEST.json',source.data.directory/'CACHE_MANIFEST.json',source.data.directory/'CACHE_COMPLETE.json'}
    m=load_base();modules=[]
    for n,mod in m.named_modules():
        if isinstance(mod,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            row=dict(name=n,type=type(mod).__name__,parent=n.rsplit('.',1)[0],weight_shape=list(mod.weight.shape),bias=mod.bias is not None)
            for a in ('stride','padding','groups','dilation','kernel_size'):
                if hasattr(mod,a):row[a]=getattr(mod,a)
            modules.append(row)
    inv=dict(checkpoint=bind(C.PRIOR_CK),device='cpu',strict_tensor_equality=True,state_sha256=state_hash(m.state_dict()),
        total_params=sum(p.numel() for p in m.parameters()),trainable_params=0,BN_count=sum(isinstance(q,nn.BatchNorm2d) for q in m.modules()),
        out_head=dict(weight_shape=list(m.out.weight.shape),bias_shape=list(m.out.bias.shape)),modules=modules)
    targets=[]
    for n in TARGETS:
        mod=m.get_submodule(n);assert type(mod) is nn.Conv2d and mod.kernel_size==mod.stride==(1,1) and mod.groups==1
        targets.append(dict(name=n,shape=list(mod.weight.shape),bias=mod.bias is not None))
    freeze(DOC/'POSEFIX_MODULE_INVENTORY.json',inv)
    freeze(DOC/'LORA_TARGET_MODULES.json',dict(scope='first bounded candidate set; not optimality claim',rank=4,alpha=4,modules=targets))
    for mode in ('FULL','SAME_LAYER','HEAD_ONLY','LORA'):
        # Same model reused only before wrapping, with all flags reset each time.
        wrapped=AdaptedPoseFix(m,mode)
        freeze(DOC/f'{mode}_TRAINABLE_CONTRACT.json',trainable_contract(wrapped))
    assert state_hash(original_state(wrapped))==inv['state_sha256']
    protocol=make_protocol(old,lock)
    freeze(DOC/'EXPERIMENT_PROTOCOL_PENDING.json',protocol)
    freeze(DOC/'PROTOCOL_HASH_LOCK.json',bind(DOC/'EXPERIMENT_PROTOCOL_PENDING.json'))
    audit=dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),protected=[bind(p) for p in sorted(protected)],
        old_BN='running frozen, affine trainable',new_BN='running + affine frozen for ALL arms',frozen_arms=frozen_arms,
        paired_count=len(lock['paired_files']),populations={k:len(v) for k,v in old['populations'].items()},
        source_train_order_verified=True,source_heldout_disjoint=True,source_mapping='Fixed native source cache GT; no online symmetry reselection',
        missing=['Independent approval of native whole-object mapping for preservation','Evidence-based preservation lambda'],
        system_changes=False,full_training=False)
    freeze(DOC/'PREFLIGHT.json',audit)
    freeze(DOC/'PREFLIGHT_AUDIT.md',f'''# Preflight

HEAD `{audit['HEAD']}`, branch `{audit['branch']}`. 기존 무관한 미추적 파일 유지. 원본 보호 binding {len(protected)}개는 PREFLIGHT.json에 기록했다.

PRIOR1 CPU strict-load 및 checkpoint tensor equality 통과. 총 {inv['total_params']:,} parameters, BN {inv['BN_count']}개, out (9,256,1,1)+bias9. 6개 후기 pointwise 후보 실모델 확인.

E2 paired input {len(lock['paired_files'])}개와 real/source order SHA, A10/A11 checkpoint/prediction을 잠갔다. augmentation 및 타깃은 기존 E2 계약을 재사용할 계획이며 이번 실사 학습은 없다. source order는 TRAIN만, heldout256과 분리 확인.

과거 BN affine는 trainable이었다. 이번 모든 대조군에서 affine까지 frozen이므로 A11을 F1로 재사용하지 않는다. 기존 losses()는 모든 하위 Conv2d/ConvTranspose2d를 순회하므로 이번 A/B Conv2d도 자동 포함된다. 이를 방지해 원래 weight L2와 adapter penalty0을 명시적으로 분리한다.

MISSING/PENDING: preserve native whole-object GT mapping의 승인 근거 및 preservation lambda. 성능 파일럿 NO-GO. 코드 검사와 합성 sanity는 허용 범위 내 진행.
''')
    print('PREFLIGHT_PASS',inv['total_params'],len(protected),flush=True)


def make_protocol(old,lock):
    return dict(status='PROTOCOL_PENDING',RUN_APPROVED=False,base=old['initialization'],seed=1,
        optimizer=dict(name='TFAdam',betas=[.9,.999],eps=1e-8,weight_decay=0),
        lr={a:1e-4 for a in ('F1','F2','F3','F4','F5','F6','H0')},
        lr_reason='Prior E2 TFAdam scale, single predeclared conservative pilot setting; NOT each-arm optimum; no rescue tuning',
        updates=300,real_batch=8,source_batch=8,microbatch=2,real_exposure=2400,source_exposure=2400,
        BN_policy='all running buffers and affine frozen across all arms',
        arms=dict(F0='BASE',F1='FULL',F2='SAME_LAYER',F3='LORA',F4='FULL+PRESERVE',F5='SAME_LAYER+PRESERVE',F6='LORA+PRESERVE',H0='HEAD_ONLY preserve OFF'),
        real_input='OCC branch from immutable E2 paired files; actual augmented R0, not clean R0 substitution',
        augmentation=old['augmentation'],pseudo_target_source=old['common_targets'],
        input_lock=old['input_lock'],real_order=lock['real_order'],source_orders=lock['source_orders'],
        source_replay_source=old['source_protocol'],source_rng=7103,source_loss_weight=1,
        source_normal_preserve='If approved: same source rows, extra normal teacher/student forward for preservation; report extra compute, no hidden source GT exposure',
        target_modules=list(TARGETS),rank=4,alpha=4,rank_status='pilot proposal, not optimum',
        preserve=dict(type='KL(teacher_softmax || student_softmax)',temperature=1,lambda_value=None,
            mask='synthetic TRAIN normal input, frozen base original-image error <=5px, support-valid corner0..7 only',
            mapping='Fixed native source GT whole-object identity, no student/pointwise reselection; approval pending',
            teacher_stop_gradient=True,normalization='sum over spatial cells then mean over selected corners; zero if no selected corners'),
        regularization=dict(original_conv_weight_half_L2=0.5e-5,adapter_penalty=0,
            accumulation='real microbatches weighted2/8 accumulate once per update; no second source L2',
            note='Legacy losses would include Conv2d A/B; use explicit original-weight-only helper'),
        checkpoint_policy='last300 only, exclusive create in new root',evaluation_populations=old['populations'],
        primary_denominator='same primary93/713 incl8 failed frames/54 corners with800px penalty; matched median/P90 on659',
        no_validation_selection=True,no_eval_GT_training_dependency=True,
        metrics=['PCK10','PCK20','matched median','matched P90','R0 (20,40] to <=10','R0 >20 to <=10',
            'BASE <=10 to >10','N2 <=10 to >10','R0 <5 to >10','R0 (5,10] and (10,20] bands lost',
            'source clean/stress','real clean','GREEN manual','non-green legacy','occlusion set'],
        gates=dict(base_correct_losses='fewer than F1',hard_recovery_fraction_of_F1=.8,pck10_delta_pp_min=-.5,
            source_clean_delta_pp_min=-1,p90_relative_increase_max=.05,
            pck20='no worsening',clean_GREEN='no worsening and preservation improvement, report each separately',
            zero_F1_recovery='ratio undefined, cannot pass recovery gate',
            caveat='pilot rules not statistical significance; F0-level recovery is not success'),
        interpretation=dict(F2_equals_F3='location restriction candidate',F3_above_F2='low-rank additional value candidate',
            F4_above_F1='preservation effect',F6_above_F3_and_F4='complementarity candidate',
            low_damage_low_recovery='capacity reduction, not success',no_hard_recovery='representation/geometry/targets',source_only_preservation='distribution mismatch'),
        blind_review='REVIEW_PENDING until validated human response; external subset identityCONFIRMED visibilityEXTERNAL_OCCLUDED confidenceHIGH, exposure groups separate; no automatic GT promotion',
        pending=['preserve lambda justification/lock','native whole-object source mapping approval','RUN_APPROVED'],
        sanity=dict(seed=1,updates=5,lr=1e-4,rank=4,alpha=4,condition='LORA only, no preserve',
            samples='first2 unique source order TRAIN rows; both normal and fixed stress',microbatch=1,batch=2,
            performance_claim=False,real_training=False))


def sample_rows():
    verify(read(DOC/'PROTOCOL_HASH_LOCK.json'))
    source=SourceData();orders=np.load(REPLAY.RAW/'ORDERS.npz')
    indices=list(dict.fromkeys(map(int,orders['source_rows'].ravel())))[:2]
    assert set(indices)<=set(source.train_rows) and not set(indices)&set(orders['held_rows'])
    rows=[source.item(i) for i in indices]
    for x in rows:assert x['partition']==x['source_partition']=='train'
    return rows


def batch(rows,device):
    return {k:torch.as_tensor(np.stack([x[k] for x in rows]),device=device) for k in ('rgb','points','valid','target','target_valid')}


def output(m,b):return m(b['rgb'],b['points'],b['valid'])


def assert_equal(a,b):
    assert torch.equal(a,b),(float((a-b).abs().max()),a.shape)


def parity():
    setup('cuda');assert not (DOC/'LORA_PARITY_AUDIT.json').exists()
    rows=sample_rows();b=batch(rows[:1],'cuda');base=load_base();checkpoint_hash=state_hash(base.state_dict())
    assert checkpoint_hash==read(DOC/'POSEFIX_MODULE_INVENTORY.json')['state_sha256']
    base.cuda()
    with torch.no_grad():q0=output(base,b);e0=expectation(q0)
    refs={n:base.get_submodule(n).weight for n in TARGETS}
    model=AdaptedPoseFix(base,'LORA')
    assert all(model.model.get_submodule(n).base.weight is refs[n] for n in TARGETS)
    assert state_hash(original_state(model))==checkpoint_hash
    with torch.no_grad():q1=output(model,b);e1=expectation(q1)
    assert_equal(q0,q1);assert_equal(e0,e1)
    mapped0=C.transform_points(e0[0].cpu().numpy(),np.linalg.inv(rows[0]['matrix']))
    mapped1=C.transform_points(e1[0].cpu().numpy(),np.linalg.inv(rows[0]['matrix']))
    assert np.array_equal(mapped0,mapped1)
    # Real production prediction adapter, actual synthetic RGB + predicted box/points.
    import cv2
    image=cv2.imread(str(ROOT/rows[0]['image_binding']['path']))
    raw=dict(selected_index=0,candidates=[dict(box_xyxy=rows[0]['box'].tolist(),keypoints_xy=rows[0]['original_points'].tolist(),score=.73,fixture_identity='synthetic-cache-row')])
    untouched=copy.deepcopy(raw)
    model.enable(False);pred0=C.predict(model,image,raw)
    model.enable(True);pred1=C.predict(model,image,raw)
    assert pred0==pred1 and raw==untouched
    assert pred1['selected_index']==raw['selected_index']
    for key in ('box_xyxy','score','fixture_identity'):assert pred1['candidates'][0][key]==raw['candidates'][0][key]
    assert pred1['candidates'][0]['keypoints_xy'][8]==raw['candidates'][0]['keypoints_xy'][8]
    model.train();assert all(not m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d))
    model.eval();assert state_hash(original_state(model))==checkpoint_hash
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=TFAdam(params,lr=1e-4)
    expected={id(p) for m in model.modules() if isinstance(m,PointwiseLoRA) for p in (m.A.weight,m.B.weight)}
    assert {id(p) for g in optimizer.param_groups for p in g['params']}==expected
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=supervised_loss(output(model,b),b['target'],b['target_valid']);loss.backward()
    grads0={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad}
    assert all(v>0 for k,v in grads0.items() if '.B.' in k)
    assert all(v==0 for k,v in grads0.items() if '.A.' in k)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    update=float(optimizer.step());assert update>0
    optimizer.zero_grad(set_to_none=True);supervised_loss(output(model,b),b['target'],b['target_valid']).backward()
    grads1={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad}
    assert all(v>0 and np.isfinite(v) for v in grads1.values())
    model.eval();saved=model.adapter_state()
    adapter_path=RAW/'parity_adapter_attempt3.pt'
    tensor_save(adapter_path,dict(adapter=saved,base=bind(C.PRIOR_CK),rank=4,alpha=4,purpose='one diagnostic step only'))
    with torch.no_grad():trained=output(model,b)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m,PointwiseLoRA):m.A.weight.normal_(0,.01);m.B.weight.normal_(0,.01)
    model.enable(False)
    with torch.no_grad():off=output(model,b)
    assert_equal(off,q0)
    model.load_adapter(torch.load(adapter_path,map_location='cpu',weights_only=False)['adapter']);model.enable(True)
    with torch.no_grad():restored=output(model,b)
    assert_equal(restored,trained)
    # Explicit legacy L2 membership audit, without using it for adapter training.
    legacy=sum(m.weight.square().sum() for m in model.modules() if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)))*.5e-5
    ab=sum(p.square().sum() for p in params)*.5e-5
    torch.testing.assert_close(legacy,original_weight_l2(model)+ab,rtol=1e-5,atol=1e-5)
    if state_hash(original_state(model))!=checkpoint_hash:
        reference=load_base().state_dict()
        differences={k:dict(max_abs=float((v.detach().cpu()-reference[k]).abs().max()),shape=list(v.shape))
            for k,v in original_state(model).items() if not torch.equal(v.detach().cpu(),reference[k])}
        freeze(DOC/'PARITY_FAILURE_DIAGNOSTIC.json',dict(differences=differences))
        raise AssertionError(('Base changed',differences))
    checks=['test_base_load_hash','test_lora_target_module_names','test_lora_zero_delta','test_lora_on_init_output_parity',
        'test_expectation_inverse_crop_candidate_center_parity','test_lora_off_output_parity','test_base_weight_immutable',
        'test_bn_all_frozen','test_train_eval_no_mutation','test_optimizer_only_lora_params','test_backward_B_grad',
        'test_backward_A_grad_after_update','test_adapter_serialization','test_legacy_L2_includes_AB_and_explicit_exclusion']
    audit=dict(PASS=True,device='cuda',dtype='float32',tests={k:'PASS' for k in checks},logit_max_abs_diff=float((q0-q1).abs().max()),
        expectation_max_abs_diff=float((e0-e1).abs().max()),inverse_crop_max_abs_diff=float(np.max(np.abs(mapped0-mapped1))),
        adapter_off_max_abs_diff=float((q0-off).abs().max()),serialization_max_abs_diff=float((trained-restored).abs().max()),
        candidate_metadata_fixture='Actual synthetic RGB/cached predicted box+points; constant score fixture to verify pass-through, not detector inference',
        diagnostic_optimizer_steps=1,grads_initial=grads0,grads_after_step=grads1,update_norm=update,
        base_BN_head_unchanged=True,optimizer_AB_only=True,trainable=trainable_contract(model),gpu=gpu_guard())
    freeze(DOC/'LORA_PARITY_AUDIT.json',audit)
    freeze(DOC/'LORA_PARITY_AUDIT.md',f'# LoRA parity PASS\n\nGPU float32: initial logits / expectation / inverse crop / adapter off / serialized restore max difference 모두 0. 실제 production predict()를 합성 RGB+cached 예측 fixture에 적용하여 center/bbox/score/candidate pass-through 확인. 새 detector 추론 검사는 아니다.\n\nA/B만 optimizer, 최초 B gradient 모두 양수, A는 0. 진단용 1-step 후 A/B gradient 모두 양수. Base/BN/head tensor hash 동일, train/eval 왕복 및 backward 이후 BN 통계·affine 고정.\n\n학습 가능 수 {audit["trainable"]["trainable"]:,}. 이 결과는 성능 평가가 아니다.\n')
    print('PARITY_PASS',audit['trainable']['trainable'],flush=True)


def sanity():
    setup('cuda');assert read(DOC/'LORA_PARITY_AUDIT.json')['PASS']
    assert not (DOC/'SYNTHETIC_SANITY.json').exists()
    protocol=read(DOC/'EXPERIMENT_PROTOCOL_PENDING.json');verify(read(DOC/'PROTOCOL_HASH_LOCK.json'))
    rows=sample_rows();rng=np.random.default_rng(7103);stress=[corrupted(x,rng,True) for x in rows]
    freeze(DOC/'SANITY_INPUT_LOCK.json',dict(rows=[dict(id=x['id'],row=int(x['row']),partition=x['partition'],image=x['image_binding']) for x in rows],
        source_order=protocol['source_orders'],normal_stress_fixed=True,GT_source='synthetic TRAIN only',updates=5,lr=1e-4,
        sample_selection='first two distinct immutable source order rows, no outcome filtering'))
    model=AdaptedPoseFix(load_base(),'LORA').cuda();original=state_hash(original_state(model));ad0=state_hash(model.adapter_state())
    probes=[batch(rows[:1],'cuda'),batch(stress[:1],'cuda')]
    with torch.no_grad():before=[output(model,b).cpu() for b in probes]
    optimizer=TFAdam([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    history=[];start=time.monotonic();model.train()
    for step in range(5):
        gpu_guard();optimizer.zero_grad(set_to_none=True);losses=[]
        # Two rows/update: paired normal and stress, alternate fixed TRAIN image.
        for item in (rows[step%2],stress[step%2]):
            b=batch([item],'cuda');loss=supervised_loss(output(model,b),b['target'],b['target_valid'])/2
            assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach()))
        update=float(optimizer.step());assert np.isfinite(update) and update>0
        history.append(dict(step=step+1,loss=sum(losses),update_norm=update))
        print('SANITY',step+1,history[-1],flush=True)
    model.eval()
    with torch.no_grad():after=[output(model,b).cpu() for b in probes]
    delta=[float((a-b).abs().max()) for a,b in zip(after,before)]
    assert all(v>0 and np.isfinite(v) for v in delta)
    assert original==state_hash(original_state(model)) and ad0!=state_hash(model.adapter_state())
    model.enable(False)
    with torch.no_grad():off=[output(model,b).cpu() for b in probes]
    for a,b in zip(before,off):assert_equal(a,b)
    path=RAW/'sanity_adapter_step5.pt';tensor_save(path,dict(adapter=model.adapter_state(),purpose='synthetic trainability only',step=5,base=bind(C.PRIOR_CK)))
    model.enable(True);saved=model.adapter_state()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m,PointwiseLoRA):m.B.weight.zero_()
    model.load_adapter(torch.load(path,map_location='cpu',weights_only=False)['adapter'])
    with torch.no_grad():reloaded=[output(model,b).cpu() for b in probes]
    for a,b in zip(after,reloaded):assert_equal(a,b)
    result=dict(PASS=True,scope='trainability only; no generalization or performance conclusion',full_pilot_training=False,
        updates=5,real_exposure=0,synthetic_exposure=10,normal_exposure=5,stress_exposure=5,unique_TRAIN_images=2,
        fresh_PRIOR1_init_not_parity_checkpoint=True,history=history,normal_stress_max_logit_change=delta,
        base_BN_head_unchanged=True,adapter_changed=True,adapter_off_exact=True,serialization_exact=True,
        seconds=time.monotonic()-start,gpu=gpu_guard(),checkpoint=bind(path),preserve_loss_used=False)
    freeze(DOC/'SYNTHETIC_SANITY.json',result)
    freeze(DOC/'SYNTHETIC_SANITY.md',f'# Synthetic mini-sanity PASS\n\n합성 TRAIN 2장, normal/stress 각각 5회 노출, optimizer5 updates. loss finite, A/B 변경, base/BN/head 불변, adapter OFF exact 복원, 저장/복원 exact. 실사 노출 0, preserve 미사용.\n\nnormal/stress max logit 변화: {delta}. Loss 감소나 성능 향상은 PASS 조건이 아니며 **LoRA performance not evaluated yet**.\n')
    print('SANITY_PASS',delta,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','parity','sanity']);args=p.parse_args()
    globals()[args.stage]()
