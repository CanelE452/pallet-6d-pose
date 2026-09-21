"""Two-arm pose-only clean-vs-occlusion experiment; frozen CAD8 pseudo targets."""
import argparse
import copy
import csv
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad8_selftrain_v1 import run as B
from scripts.research.pallet_cad8_occlusion_v1 import augmentation as A
from scripts.research.pallet_type_selftrain_v1.recovery_pose_trainer import pose_parameter

RAW=ROOT/'data/pallet/results/pallet_cad8_occlusion_v1'
DOC=ROOT/'_docs/experiments/pallet_cad8_occlusion_v1'
OUT=ROOT/'outputs/pallet_cad8_occlusion_v1'
ARMS={'CLEAN':'REPLAY_PNP','OCCLUDED':'REPLAY_PNP'}
read,write=B.read,B.write


def prepare():
    parent=B.check();tests=A.tests()
    p=copy.deepcopy(parent)
    p.update(arms=ARMS,datasets={a:parent['datasets']['REPLAY_PNP'] for a in ARMS},
        frozen='Only pose branch parameters/flow trainable; backbone/neck/detection weights AND all BN buffers exact R0. Pose BN affine remains trainable.',
        budget='Two arms from R0,5epochs/320updates,seed42,lr1e-4 unchanged. Identical dataset/order/labels/native augmentation. Same fixed last.pt rule.',
        teacher='Existing Replay+PnP targets on the8 selected CAD frames; fixed before this experiment, no teacher reselection on these outcomes.',
        occlusion='Post-geometric-augmentation RGB only. 75% of real exposures:1-2 textured-color patches, each18-35% bbox width/height, centered at trusted top corner/edge; leave>=4 trusted corners uncovered. Synthetic replay unchanged.',
        supervision='Keep pseudo coordinates AND original true-ignore masks. Artificially hidden trusted corners remain supervised2 (including keypoint existence target); no new invisible0 or ignore1 labels.',
        RNG='Independent numpy902106 in both arms, no consumption of model/global augmentation RNG. Clean generates same plans but does not paint.',
        controls='CLEAN vs OCCLUDED isolates artificial occlusion under same pose-only update. R0 and archived full-model Replay CAD8 run provide context only.',
        causal_limit='Only this synthetic-occlusion recipe/320steps/8images/one seed tested; not proof about all occlusion adaptation methods.',
        tests=tests,
        sources=parent['sources']+[B.G.binding(B.DOC/'PROTOCOL.json'),B.G.binding(Path(__file__)),B.G.binding(Path(A.__file__)),
            B.G.binding(ROOT/'scripts/research/pallet_type_selftrain_v1/recovery_pose_trainer.py')])
    write(DOC/'PROTOCOL.json',p)
    print('OCCLUSION_PROTOCOL_LOCKED',tests,flush=True)


def scope():
    from contextlib import ExitStack
    stack=ExitStack()
    for k,v in [('DOC',DOC),('RAW',RAW),('ARMS',ARMS)]:stack.enter_context(patch.object(B,k,v))
    return stack


def train(arm):
    with scope():p=B.check()
    fit=DOC/f'FIT_{arm}.json'
    if fit.exists():B.G.verify(read(fit)['checkpoint']);return
    B.C.N.setup();assert torch.cuda.is_available();print('GPU',B.gpu(),flush=True)
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))//1024
    assert available>=6000
    folder=RAW/'runs'/arm;assert not folder.exists(),'Preserve incomplete run'
    trainer=A.OcclusionPoseTrainer(overrides=dict(p['args'],model=str(ROOT/p['initialization']['path']),
        data=str(ROOT/p['datasets'][arm]['data']['path']),project=str(RAW/'runs'),name=arm,exist_ok=False))
    base=torch.load(ROOT/p['initialization']['path'],map_location='cpu',weights_only=False)['model'].float().state_dict()
    steps=[];history=[];start=time.monotonic()
    def begin(t):
        actual=t.model.state_dict();assert base.keys()==actual.keys()
        assert all(torch.equal(v,actual[k].detach().cpu()) for k,v in base.items())
        assert all(pose_parameter(k) for k in t.recovery_trainable)
        t.optimizer.register_step_post_hook(lambda opt,args,kwargs:steps.append(len(steps)+1))
        print('POSE_ONLY_R0_EXACT',arm,len(t.recovery_trainable),flush=True)
    def epoch(t):
        fixed=t.check_frozen();status=B.gpu();history.append(dict(epoch=t.epoch+1,steps=len(steps),protected=fixed,gpu=status))
        print('POSE_OCC_PROGRESS',arm,t.epoch+1,len(steps),status,flush=True)
    def batch(t):
        if len(steps)%32==0:B.gpu()
    trainer.add_callback('on_train_start',begin);trainer.add_callback('on_train_epoch_end',epoch);trainer.add_callback('on_train_batch_end',batch)
    trainer.train();checkpoint=folder/'weights/last.pt'
    final=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'].float().state_dict()
    protected=[k for k in base if not pose_parameter(k) or k.endswith(('.running_mean','.running_var','.num_batches_tracked'))]
    assert all(torch.equal(base[k],final[k]) for k in protected),'Saved frozen state changed'
    changed=[k for k in base if not torch.equal(base[k],final[k])];assert changed and all(pose_parameter(k) for k in changed)
    assert len(steps)==len(trainer.occlusion_trace)==320
    assert len(list(csv.DictReader((folder/'results.csv').open())))==5
    write(RAW/f'TRACE_{arm}.json',trainer.occlusion_trace)
    if arm=='OCCLUDED':
        OUT.mkdir(exist_ok=True,parents=True);previews=[]
        for i,r in enumerate(trainer.occlusion_previews,1):
            to_bgr=lambda x:cv2.cvtColor(np.rint(x.transpose(1,2,0)*255).clip(0,255).astype('uint8'),cv2.COLOR_RGB2BGR)
            pixels=np.concatenate([to_bgr(r['before']),to_bgr(r['after'])],axis=1)
            B.G.image_write(OUT/f'augmentation_{i:02}.png',pixels)
            previews.append({k:v for k,v in r.items() if k not in ['before','after']})
        write(OUT/'AUGMENTATION_PREVIEWS.json',previews)
    write(fit,dict(complete=True,checkpoint=B.G.binding(checkpoint),protocol=B.G.binding(DOC/'PROTOCOL.json'),
        steps=len(steps),epochs=history,seconds=time.monotonic()-start,protected_state_exact=True,protected_tensors=len(protected),
        changed_pose_tensors=changed,trace=B.G.binding(RAW/f'TRACE_{arm}.json')))
    del trainer;gc.collect();torch.cuda.empty_cache();print('POSE_OCC_TRAIN_COMPLETE',arm,flush=True)


def audit():
    with scope():p=B.check()
    traces={a:read(RAW/f'TRACE_{a}.json') for a in ARMS}
    assert traces['CLEAN']==traces['OCCLUDED'],'Input order/target/native-augmentation trace differs'
    trace=traces['OCCLUDED'];real_count=sum(sum(Path(x).name.startswith('eval_cad__') for x in t['files']) for t in trace)
    masked=sum(len(t['plan']) for t in trace);covered=sum(len(x['covered_corners']) for t in trace for x in t['plan'])
    old={r['id']:r['prediction'] for r in read(B.C.RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    parity={}
    for arm in ARMS:
        maximum=0.
        for r in read(RAW/f'PREDICTIONS_{arm}.json')['records']:
            before=old[r['id']];after=r['prediction']
            assert before['selected_index']==after['selected_index'] and len(before['candidates'])==len(after['candidates'])
            for a,b in zip(before['candidates'],after['candidates']):
                delta=np.max(np.abs(np.array([a['score'],*a['box_xyxy']])-np.array([b['score'],*b['box_xyxy']])))
                maximum=max(maximum,float(delta))
        assert maximum<=1e-4,('Detection inference parity',arm,maximum)
        parity[arm]=maximum
    results=read(DOC/'RESULTS.json');metrics=read(RAW/'METRICS.json')
    from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as M
    ids=set(p['primary_occlusion_ids']);clean=[r for r in metrics['CLEAN'] if r['id'] in ids];occ=[r for r in metrics['OCCLUDED'] if r['id'] in ids]
    effect=M.damage(clean,occ)
    write(DOC/'AUDIT.json',dict(native_input_and_label_plan_traces_exact=True,real_exposures=real_count,occluded_exposures=masked,
        covered_corner_exposures=covered,detector_max_abs_difference=parity,protected_checkpoint_state_exact=True,
        occluded_vs_clean_primary=effect,comparison='Paired fixed320steps, identical inputs/targets before painting. Primary OCC96, noCAD.'))
    lines=['# 포즈 전용 학습: 깨끗한 수도레이블 vs 인공 가림','',
        '같은 CAD8 Replay+PnP 수도레이블, 합성512 replay,320update,seed42,lr1e-4. Backbone/검출/BN 통계 고정. 포즈만 학습.',
        'CLEAN과 OCCLUDED의 입력 순서·가림 전 RGB 샘플 해시·전체 타깃·가림 계획이 일치. OCCLUDED만 RGB 패치를 적용. 가려도 타깃 좌표와 학습 마스크 유지.',
        f'실사 노출{real_count}회 중{masked}회 인공 가림. 가린 supervised corner 노출 합계{covered}.',
        '평가: 기존 occlusion96장, CAD 전체 제외, Replay teacher 학습 세션 제외. 학생 단독 추론. 재사용 DEV·단일 seed, 독립 확인 아님.','',
        '| 모델 | 중앙값 px ↓ | P90 px ↓ | PCK10 ↑ | PCK20 ↑ | 매칭 |','|---|---:|---:|---:|---:|---:|']
    for a,s in results['summary']['PRIMARY_OCC96'].items():
        lines.append(f'| {a} | {s["matched_pooled_corner8_median_px"]:.2f} | {s["matched_pooled_corner8_P90_px"]:.2f} | {100*s["PCK"]["10"]:.2f}% | {100*s["PCK"]["20"]:.2f}% | {s["matched"]}/96 |')
    lines+=['',f'OCCLUDED vs CLEAN: 개선{effect["improved_frames"]}장 / 악화{effect["harmed_frames"]}장 / 유지{effect["unchanged_frames"]}장.',
        '검출 가중치·BN 버퍼 보존 및 검출 출력 parity 확인. 학습 후 보정기/PnP는 붙이지 않음. 원래 모델·평가 split 변경/자동 승격 없음.']
    write(DOC/'RESULTS_KO.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(primary=results['summary']['PRIMARY_OCC96'],audit=effect,masked=masked,real=real_count,detector_parity=parity),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','all','audit',*ARMS]);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='audit':audit()
    else:
        for arm in ARMS if a.stage=='all' else [a.stage]:
            train(arm)
            with scope():B.infer(arm)
        if a.stage=='all':
            with scope():B.score()
            audit()
