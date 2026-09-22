"""One 300-step manual-Clean19 + synthetic replay screen; same-session image split."""
import argparse
import copy
import csv
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_posefix_replay_v1 import train as T
from scripts.research.pallet_posefix_replay_v1.source import SourceData
from scripts.research.pallet_posefix_large_error_v1 import train as O
from scripts.research.pallet_posefix_large_error_v1.evaluate import assert_same
from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved
from scripts.research.pallet_sensors_submission_v1.prior_model import TFAdam
from scripts.research.pallet_dim_conditioned_p_v1 import dev_evaluate as D
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as M

C=N.C; ROOT=N.ROOT
NAME='pallet_replay_clean19_v1'
DOC=ROOT/'_docs/experiments'/NAME;RAW=ROOT/'data/pallet/results'/NAME;OUT=ROOT/'outputs'/NAME
SEVERITIES=('CLEAN','MODERATE_OCCLUSION','SEVERE_OCCLUSION')
ARMS=('R0','N2_DIM_ONLY','N3_DIM_SYM','PRIOR1','CLEAN19_REPLAY','CLEAN19_REPLAY_CAP8')
read=N.E.read;bind=N.E.bound


def save(p,obj):
    assert any(p.resolve().is_relative_to(x) for x in (DOC,RAW,OUT))
    p.parent.mkdir(parents=True,exist_ok=True)
    text=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('x') as f:f.write(text)


def torchsave(p,obj):
    assert p.is_relative_to(RAW) and not p.exists()
    torch.save(obj,p)


def verify():
    for b in read(DOC/'INPUT_LOCK.json')['files']:N.F.verify(b)


def prepare():
    assert not any(p.exists() for p in (DOC,RAW,OUT))
    ws=ROOT/'data/evaluation/pallet_eval_v1'
    source=ws/'manifests/ALL_AVAILABLE_POSITIVE.csv'
    rows=list(csv.DictReader(source.open()));assert len(rows)==319
    response=Path('/home/minjae/Downloads/COMBINED319_SEVERITY_RESPONSES.json')
    rr=read(response);mapping=read(ROOT/'data/pallet/results/pallet_combined319_severity_review_v1/PRIVATE_FRAME_MAPPING.json')['rows']
    order=ROOT/'data/pallet/results/pallet_combined319_severity_review_v1/REVIEW_ORDER.json'
    assert rr['order_sha256']==N.E.sha(order)
    assert set(rr['responses'])==set(read(order)['review_ids'])
    assert all(v in SEVERITIES for v in rr['responses'].values())
    severity={r['frame_id']:rr['responses'][r['review_id']] for r in mapping}
    groupspath=ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json'
    gd=read(groupspath)
    group={s['session_key']:g['recording_id'] for g in gd['groups'] if not g['is_collection'] for s in g['sessions']}
    cache=read(C.E.DOC/'DEV_CACHE_COMPLETE.json')
    cache_by_key={r['key']:r for r in cache['records']}
    baselines={a:read(C.E.RAW/f'predictions/REAL_DEV/{a}_seed1.json') for a in ('N2_DIM_ONLY','N3_DIM_SYM')}
    preds={a:{r['id']:r for r in p['records']} for a,p in baselines.items()}
    r0file=C.E.LINE/'baseline/FULL_CANDIDATES.json';r0=read(r0file)
    records=[];baseline={'R0':{},**preds};protected=[bind(source),bind(order),bind(groupspath),bind(r0file),bind(C.PRIOR_CK),bind(C.E.DOC/'DEV_CACHE_COMPLETE.json'),bind(C.E.RAW/'REAL_DEV_METRICS.json')]
    protected += [bind(C.E.RAW/f'predictions/REAL_DEV/{a}_seed1.json') for a in baselines]
    for r in rows:
        image=ws/r['image_path'];ann=ws/r['annotation_path'];key=str(image.relative_to(ROOT))
        assert N.E.sha(image)==r['image_sha256']
        fid=cache_by_key[key]['id'];sessionkey=str(image.parent.parent.relative_to(ROOT))
        g=group.get(sessionkey)
        assert g is not None,sessionkey
        gt,mask=C.R.manual_target(read(ann))
        cands=copy.deepcopy(r0['frames'][key])
        for i,c in enumerate(cands):
            c['candidate_index']=preds['N2_DIM_ONLY'][fid]['candidates'][i]['candidate_index']
            c['keypoints_conf']=preds['N2_DIM_ONLY'][fid]['candidates'][i]['keypoints_conf']
        p=dict(candidates=cands,selected_index=preds['N2_DIM_ONLY'][fid]['selected_index'])
        assert_preserved(p,preds['N2_DIM_ONLY'][fid]);assert_preserved(p,preds['N3_DIM_SYM'][fid])
        baseline['R0'][fid]=p
        protected += [bind(image),bind(ann)]
        records.append(dict(id=fid,frame_id=r['frame_id'],image=bind(image),annotation=bind(ann),
            session=r['session_id'],recording=g,object_type=r['object_type'],severity=severity[r['frame_id']],
            manual_count=int(mask.sum())))
    # User explicitly permits same-session adaptation. Choose diversity without model errors.
    thumbs=[];features=[]
    for r in records:
        im=cv2.imread(str(ROOT/r['image']['path']));assert im is not None
        thumbs.append(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(64,48)).astype(np.float32))
        features.append(cv2.resize(im,(16,12)).astype(np.float32).ravel()/255.)
    thumbs=np.stack(thumbs);features=np.stack(features)
    distances=np.stack([np.abs(thumbs-x).mean((1,2)) for x in thumbs]);np.fill_diagonal(distances,np.inf)
    near=distances<=2.0
    eligible=[r for i,r in enumerate(records) if r['severity']=='CLEAN' and r['manual_count']>=3 and not near[i].any()]
    selected=[];quota={'plastic_day_01':5,'plastic_night_01':5,'wood_day_01':4,'wood_night_01':5}
    position={r['id']:i for i,r in enumerate(records)}
    for session,count in quota.items():
        pool=sorted([r for r in eligible if r['session']==session],key=lambda r:r['id'])
        assert len(pool)>=count,(session,len(pool),'insufficient near-duplicate-safe Clean manual candidates')
        random.Random(20260922).shuffle(pool);chosen=[]
        while len(chosen)<count:
            reference=[position[r['id']] for r in chosen]
            r=pool[0] if not chosen else max(pool,key=lambda r:min(float(np.square(features[position[r['id']]]-features[j]).mean()) for j in reference))
            chosen.append(r);pool.remove(r)
        selected+=chosen
    assert len(selected)==19
    selected_ids={r['id'] for r in selected}
    evaluation=[r for r in records if r['id'] not in selected_ids];withheld=[]
    assert len(evaluation)==300
    cross=distances[np.ix_([position[r['id']] for r in selected],[position[r['id']] for r in evaluation])]
    assert not (cross<=2.).any()
    assert not {r['image']['sha256'] for r in selected}&{r['image']['sha256'] for r in evaluation}
    for p in (DOC,RAW,OUT):p.mkdir(parents=True)
    save(RAW/'HUMAN_SEVERITY_RESPONSES.json',rr)
    split=dict(train=selected,evaluation=evaluation,withheld=withheld,
        same_session_allowed_by_user=True,recording_disjoint=False,
        near_duplicate_audit=dict(method='64x48 grayscale mean absolute pixel difference <=2/255 flagged before selection',minimum_cross_split_MAD=float(cross.min()),flagged_cross_split_pairs=0,not_a_guarantee_of_scene_independence=True),
        rule='Clean manual>=3; session quota5/5/4/5; fixed seeded first point and RGB-thumbnail farthest sampling; no errors used')
    save(DOC/'SPLIT.json',split)
    save(RAW/'BASELINE_PREDICTIONS.json',baseline)
    code=[Path(__file__),C.HERE/'core.py',C.HERE/'train.py',N.HERE/'train.py',N.HERE/'source.py',N.HERE/'core.py',ROOT/'scripts/research/pallet_dim_conditioned_p_v1/eval_math.py']
    protected += [bind(p) for p in code]+[bind(N.DOC/'INPUT_LOCK.json'),bind(N.RAW/'ORDERS.npz')]
    # Synthetic-only prior provenance and historical source TRAIN/HOLDOUT separation retained.
    protocol=dict(seed=1,steps=300,batch_real=8,batch_synthetic=8,microbatch=2,
        init='PRIOR1 synthetic-only 6000 steps; optimizer reset; not former real-trained checkpoint',
        optimizer='TFAdam lr=1e-4',BN='frozen running statistics',
        real_supervision='manual_click finite in-frame crop-supported corners only; center excluded',
        corruption='unchanged legacy O.corrupted p=.5; .05-.15 bbox diagonal jitter',
        loss='unchanged CE+coordinate real and source; L2 once',
        source_order='historical replay ORDERS source_rows; synthetic RNG7103',
        train=19,withheld=0,evaluation=300,main_arms=list(ARMS[:-1]),secondary_arm=ARMS[-1],
        primary_metric='existing reference corner8 PCK10 on ALL300',
        secondary=['PCK5','PCK20','observed median/P90','full-penalty median/P90','gross20','severity','material','source heldout'],
        selected_checkpoint='last300 only; no threshold/checkpoint/seed search',
        old_real_replay_excluded='old nine-label adaptation overlaps evaluation nighttime; not a fair held-out comparison',
        pilot_only=True,independent_unseen_test=False,auto_promote=False,
        original319_results_preserved=True,green_used=False,commit_push=False)
    save(DOC/'PROTOCOL.json',protocol)
    protected += [bind(DOC/'SPLIT.json'),bind(DOC/'PROTOCOL.json'),bind(RAW/'BASELINE_PREDICTIONS.json'),bind(RAW/'HUMAN_SEVERITY_RESPONSES.json')]
    save(DOC/'INPUT_LOCK.json',dict(files=protected,source_response_sha256=N.E.sha(response)))
    save(DOC/'PURPOSE_AND_PLAN.md','# Clean19 학습 / 동일 세션 허용 평가300\n\n사용자 명시적 요청으로 같은 촬영 세션을 허용한다. 플라스틱 주간5/야간5, 목재 주간4/야간5의 Clean·직접클릭≥3코너19장. 원본 SHA중복0, 사전규칙 grayscale MAD<=2 근접중복 교차0 확인. 썸네일 분산은 시점 다양성의 보조이며 환경독립 증명이 아니다. 나머지300장 동일 분모 평가. 합성전용PRIOR1에서 seed1/300step Replay, 마지막checkpoint 고정. 기존319 결과·원본GT·모델 변경 없음. 독립 새 환경 일반화 주장이 아닌 동일 환경 적응 파일럿이다.\n')
    print('PREPARED',19,len(evaluation),len(withheld),quota,flush=True)


def real_items():
    baseline=read(RAW/'BASELINE_PREDICTIONS.json')['R0'];result=[]
    for r in read(DOC/'SPLIT.json')['train']:
        im=cv2.imread(str(ROOT/r['image']['path']));assert im is not None
        x=C.prepare_input(im,baseline[r['id']]);assert x is not None,r['id']
        gt,valid=C.R.manual_target(read(ROOT/r['annotation']['path']))
        target=C.transform_points(np.where(valid[:,None],gt,0),x['matrix']).astype(np.float32)
        supported=valid&(target>=0).all(-1)&(target[:,0]<288)&(target[:,1]<384);supported[8]=False
        assert supported.any(),r['id']
        x.update(id=r['id'],target=target,target_valid=supported,original_gt=gt,manual_mask=valid)
        result.append(x)
    return result


def train():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    assert torch.cuda.is_available()
    real=real_items();source=SourceData();historical=dict(np.load(N.RAW/'ORDERS.npz'))
    order=N.make_real_order(real);order['source_rows']=historical['source_rows'];order['held_rows']=historical['held_rows']
    assert set(order['source_rows'].ravel())<=set(source.train_rows)
    assert not set(order['source_rows'].ravel())&set(order['held_rows'])
    prior_lock=read(N.DOC/'INPUT_LOCK.json')
    chosen=np.unique(np.r_[order['source_rows'].ravel(),order['held_rows']])
    assert N.source_cache_signature(source.data,chosen)==prior_lock['source_cache_values']
    for r in prior_lock['source_images']:N.F.verify(r['image'])
    np.savez(RAW/'ORDERS.npz',**order)
    save(DOC/'TRAIN_SUPPORT.json',dict(records=[dict(id=x['id'],manual=x['manual_mask'].tolist(),crop_supported=x['target_valid'].tolist()) for x in real],
        manual=sum(int(x['manual_mask'].sum()) for x in real),supported=sum(int(x['target_valid'].sum()) for x in real)))
    held=[source.item(int(i)) for i in order['held_rows']]
    torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    model=C.load_model().requires_grad_(True)
    before=dict(real=O.probe(model,real),source=T.source_probe(model,held))
    save(DOC/'BEFORE.json',before)
    optimizer=TFAdam(model.parameters(),lr=1e-4);model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.BatchNorm2d):m.eval()
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    initial=C.E.state_sha(model.state_dict());gs=np.random.default_rng(7103);history=[]
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    save(DOC/'TRAIN_START.json',dict(protocol=bind(DOC/'PROTOCOL.json'),input_lock=bind(DOC/'INPUT_LOCK.json'),initial_state_sha=initial,orders=bind(RAW/'ORDERS.npz'),gpu=N.E.gpu()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(source.item,int(i)) for i in order['source_rows'][0]]
        for idx in range(300):
            synthetic=[f.result() for f in pending]
            pending=[pool.submit(source.item,int(i)) for i in order['source_rows'][idx+1]] if idx<299 else []
            branches=dict(real=T.real_rows(real,order,idx),source=[O.corrupted(x,gs) for x in synthetic])
            optimizer.zero_grad(set_to_none=True);losses={}
            for branch,items in branches.items():
                parts=[]
                for j in range(0,8,2):
                    b=O.batch(items[j:j+2]);q=model(b['rgb'],b['points'],b['valid'])
                    _,detail=model.losses(q,b['target'],b['target_valid'])
                    loss=T.combined_micro_loss(detail,branch);assert torch.isfinite(loss)
                    loss.backward();parts.append({k:float(v.detach()) for k,v in detail.items()})
                    del b,q,detail,loss
                losses[branch]={k:float(np.mean([p[k] for p in parts])) for k in parts[0]}
            norm=float(torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt())
            update=float(optimizer.step());assert np.isfinite(norm) and np.isfinite(update) and update>0
            history.append(dict(step=idx+1,loss=losses,gradient_norm=norm,update_norm=update))
            if idx==0 or (idx+1)%25==0:print('TRAIN',idx+1,round(time.monotonic()-start,1),losses,'GPU',N.E.gpu(),flush=True)
            if (idx+1)%100==0:
                torchsave(RAW/f'step{idx+1}.pt',dict(step=idx+1,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),history=history))
    assert all(torch.equal(buffers[k],v) for k,v in model.named_buffers())
    assert C.E.state_sha(model.state_dict())!=initial
    model.eval()
    after=dict(real=O.probe(model,real),source=T.source_probe(model,held))
    save(DOC/'FIT.json',dict(complete=True,step=300,checkpoint=bind(RAW/'step300.pt'),before=before,after=after,
        BN_buffers_unchanged=True,real_exposures=2400,synthetic_exposures=2400,
        seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/2**20,auto_promoted=False))
    save(RAW/'TRAIN_STEPS.json',history);verify();print('TRAIN_COMPLETE',flush=True)


@torch.no_grad()
def infer():
    verify();N.setup();print(N.E.gpu(),flush=True)
    fit=read(DOC/'FIT.json');assert fit['complete'];N.F.verify(fit['checkpoint'])
    baseline=read(RAW/'BASELINE_PREDICTIONS.json');records=read(DOC/'SPLIT.json')['evaluation']
    result={a:{r['id']:baseline[a][r['id']] for r in records} for a in ('R0','N2_DIM_ONLY','N3_DIM_SYM')}
    for arm in ('PRIOR1','CLEAN19_REPLAY'):
        model=C.load_model()
        if arm=='CLEAN19_REPLAY':model.load_state_dict(torch.load(ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict'])
        result[arm]={}
        if arm=='CLEAN19_REPLAY':result['CLEAN19_REPLAY_CAP8']={}
        for idx,r in enumerate(records):
            fid=r['id'];im=cv2.imread(str(ROOT/r['image']['path']));p=baseline['R0'][fid]
            q=C.predict(model,im,p,None);assert_preserved(p,q);result[arm][fid]=q
            if arm=='CLEAN19_REPLAY':result['CLEAN19_REPLAY_CAP8'][fid]=C.cap_prediction(p,q,.01,im.shape[:2])
            if (idx+1)%50==0:print('INFER',arm,idx+1,len(records),N.E.gpu(),flush=True)
        del model;torch.cuda.empty_cache()
    save(RAW/'PREDICTIONS.json',dict(predictions=result,GT_input=False,complete=True))
    save(DOC/'PREDICTION_LOCK.json',dict(predictions=bind(RAW/'PREDICTIONS.json'),checkpoint=fit['checkpoint'],frozen_before_scoring=True))
    print('PREDICTIONS_FROZEN',flush=True)


def evaluate():
    verify();lock=read(DOC/'PREDICTION_LOCK.json');assert lock['frozen_before_scoring'];N.F.verify(lock['predictions'])
    split=read(DOC/'SPLIT.json');records=split['evaluation'];ids={r['id'] for r in records}
    preds=read(RAW/'PREDICTIONS.json')['predictions'];pe,pop=D.population_metadata()
    sym={r['object_type']:r['permutations'] for r in read(C.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    metrics={a:{} for a in ARMS};truth={}
    old=read(C.E.RAW/'REAL_DEV_METRICS.json')
    legacy={a:{r['id']:r for r in old[a+'_seed1']} for a in ('N2_DIM_ONLY','N3_DIM_SYM')}
    for item,meta in pop:
        fid=item.frame_id
        if fid not in ids:continue
        target=pe.E._legacy_forbidden_target(item)
        g=np.array(target.keypoints_xy);v=np.array(target.keypoint_supervision_mask);box=np.array(target.box_xyxy)
        im=cv2.imread(str(ROOT/item.image));hw=im.shape[:2]
        truth[fid]=dict(gt=g.tolist(),valid=v.tolist(),box=box.tolist(),hw=list(hw),permutations=sym[meta['object_type']])
        for arm in ARMS:
            p=preds[arm][fid];c=C.selected(p);detected=c is not None;matched=detected and D.iou(c['box_xyxy'],box)>=.5
            q=np.full((9,2),np.nan) if c is None else c['keypoints_xy']
            row=M.measure(q,g,v,sym[meta['object_type']],hw,matched,detected)
            if arm in legacy:
                for k,value in row.items():assert_same(value,legacy[arm][fid][k],f'{arm}/{fid}/{k}')
            metrics[arm][fid]=dict(id=fid,**row)
    assert all(set(m)==ids for m in metrics.values())
    groups={'ALL300':[r['id'] for r in records]}
    for c in SEVERITIES:groups[c]=[r['id'] for r in records if r['severity']==c]
    for material in ('plastic','wood'):
        groups[material]=[r['id'] for r in records if r['object_type']==material]
        for c in SEVERITIES:groups[material+'_'+c]=[r['id'] for r in records if r['object_type']==material and r['severity']==c]
    summary={}
    for name,group in groups.items():
        summary[name]={}
        for arm in ARMS:
            rows=[metrics[arm][i] for i in group];s=M.summary(rows)
            s.update(correct10=sum(e<=10 for r in rows for e in r.get('errors',[])),gross20_count=sum(e>20 for r in rows for e in r.get('errors',[])))
            summary[name][arm]=s
        assert len({(s['corners'],s['total_frames'],s['detected'],s['matched']) for s in summary[name].values()})==1
    save(RAW/'FRAME_METRICS.json',metrics);save(RAW/'TRUTH_FOR_DISPLAY_ONLY.json',truth)
    save(DOC/'RESULTS.json',summary)
    report(summary,metrics,records)
    verify()
    save(DOC/'AUDIT.json',dict(complete=True,train=19,evaluation=300,withheld=0,
        recording_disjoint=False,same_session_user_authorized=True,image_sha_disjoint=True,BN_unchanged=True,legacy_N2_N3_parity=True,
        same_denominator=True,failed_matches_retained=True,existing319_unchanged=True,
        new_training_steps=300,seed=1,checkpoint='step300',independent_confirmation=False,auto_promoted=False))
    print('RESULTS',json.dumps({a:summary['ALL300'][a]['PCK'] for a in ARMS}),flush=True)


def report(summary,metrics,records):
    fmt=lambda x:'—' if x is None else f'{x:.2f}'
    lines=['# Clean19 + synthetic replay — 동일 세션 허용 평가300 결과','',
        '학습19장(플라스틱10/목재9), 나머지300장 평가. 동일 이미지는 분리했으나 촬영 세션은 공유한다. 기존319장 표와 직접 비교하지 않고 같은300장 내에서만 비교한다.', '',
        '합성전용PRIOR1에서 seed1/300step, 실사8+합성8, manual_click 코너만 감독. 마지막300step 고정. cap8은 사전등록 secondary이며 좋은 출력만 골라 사용하지 않는다. 예측에 GT/PnP/치수를 넣지 않는다(N2/N3 기존 dimension baseline은 예외).', '',
        '이것은 self-training이 아니라 소량 수동지도 보정기 적응이다. 기존Replay는 평가에 포함된 과거학습9장 때문에 공정비교에서 제외했다. 이전 연구에서 확인한 데이터이므로 독립 미사용 test 주장은 하지 않는다.', '']
    for group in ('ALL300','plastic','wood',*SEVERITIES):
        ref=summary[group]['R0'];lines+=['## '+group,'',f'프레임{ref["total_frames"]}, 유효코너{ref["corners"]}, 검출{ref["detected"]}, 매칭{ref["matched"]}.', '',
            '| 모델 | PCK5% | PCK10% | PCK20% | observed med px | observed P90 px | >20 코너 |', '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            r=summary[group][arm];vals=[fmt(None if r['PCK'][k] is None else 100*r['PCK'][k]) for k in ('5','10','20')]
            vals += [fmt(r['matched_pooled_corner8_median_px']),fmt(r['matched_pooled_corner8_P90_px']),str(r['gross20_count'])]
            lines.append('| '+arm+' | '+' | '.join(vals)+' |')
        lines.append('')
    delta={a:100*(summary['ALL300']['CLEAN19_REPLAY']['PCK']['10']-summary['ALL300'][a]['PCK']['10']) for a in ('R0','N2_DIM_ONLY','N3_DIM_SYM','PRIOR1')}
    decision=dict(status='PILOT_ONLY',delta_PCK10_pp=delta,positive_vs_synthetic_prior=delta['PRIOR1']>0,
        exceeds_N2=delta['N2_DIM_ONLY']>0,exceeds_N3=delta['N3_DIM_SYM']>0,next='Inspect fixed-seed paired examples; no automatic further training or final model replacement')
    save(DOC/'DECISION.json',decision)
    lines+=['## 한계와 다음 단계','',f'RAW Replay의 PCK10 차이(pp): {delta}.', '',
        '유효 정답 전체 분모에 실패 penalty를 유지한다. observed med/P90만 매칭된 유효 예측 통계이며 penalty 통계는 JSON에 별도 있다. 기존 reference 출처가 불확실한 hidden point를 독립 물리 GT라 부르지 않는다. 보이는 점/가려진 점을 임의로 나누지 않는다.', '',
        '주간·야간 동일 촬영 세션을 공유하므로 환경 간 일반화 검증이 아니다.19장 조합·seed를 탐색하지 않았다. 단일 bounded pilot이며 최종 모델 자동 교체 없음.', '',
        '비교 예시는 outputs/pallet_replay_clean19_v1/GALLERY.html. 다음은 고정 예시 확인 한 가지이며 추가학습은 실행하지 않는다.']
    save(DOC/'RESULTS_KO.md','\n'.join(lines)+'\n')
    # Original RGB plus native SVG overlays; fixed random, gain, and damage examples.
    preds=read(RAW/'PREDICTIONS.json')['predictions'];truth=read(RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    byid={r['id']:r for r in records};rng=random.Random(20260922);ids=list(byid);rng.shuffle(ids)
    diff=lambda i:sum(e<=10 for e in metrics['CLEAN19_REPLAY'][i].get('errors',[]))-sum(e<=10 for e in metrics['PRIOR1'][i].get('errors',[]))
    cases={'random':ids[:8],'gain_vs_PRIOR1':[i for i in ids if diff(i)>0][:4],'damage_vs_PRIOR1':[i for i in ids if diff(i)<0][:4]}
    save(DOC/'GALLERY_SELECTION.json',cases);(OUT/'images').mkdir()
    import shutil,html
    for index,fid in enumerate(dict.fromkeys(i for v in cases.values() for i in v)):
        r=byid[fid];name=f'{index:03d}.png';shutil.copyfile(ROOT/r['image']['path'],OUT/'images'/name);r['gallery_image']='images/'+name
    parts=['<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;background:#142631;color:white}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}svg{width:100%}</style><h1>Clean19 replay · 동일 평가300장</h1><p>초록=기존 reference / 노랑=예측. 원본 픽셀 기준, whole-object symmetry 대응. 학습19장 아님.</p>']
    for kind,subset in cases.items():
        parts.append(f'<h2>{kind}</h2>')
        for fid in subset:
            r=byid[fid];t=truth[fid];parts.append(f'<h3>{html.escape(fid)} · {r["severity"]}</h3><div class="grid">')
            for arm in ARMS:
                m=metrics[arm][fid];p=preds[arm][fid];q=C.selected(p);overlay=[]
                if q is not None and m['evaluable']:
                    xy=np.array(q['keypoints_xy']);aligned=np.empty_like(xy);aligned[t['permutations'][m['branch']]]=xy
                    for j,v in enumerate(m['canonical_valid']):
                        if not v:continue
                        x,y=t['gt'][j];u,w=aligned[j]
                        if not np.isfinite([x,y,u,w]).all():continue
                        overlay.append(f'<circle cx="{x}" cy="{y}" r="4" stroke="#58ff60" fill="none" stroke-width="2"/><circle cx="{u}" cy="{w}" r="3" fill="#ffdc55"/><line x1="{x}" y1="{y}" x2="{u}" y2="{w}" stroke="#ffdc55"/>')
                err=m.get('errors',[]);h,w=t['hw'];parts.append(f'<div>{arm} · ≤10px {sum(e<=10 for e in err)}/{len(err)}<svg viewBox="0 0 {w} {h}"><image href="{r["gallery_image"]}" width="{w}" height="{h}"/>{"".join(overlay)}</svg></div>')
            parts.append('</div>')
    save(OUT/'GALLERY.html',''.join(parts))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','infer','evaluate']);args=ap.parse_args();globals()[args.stage]()
