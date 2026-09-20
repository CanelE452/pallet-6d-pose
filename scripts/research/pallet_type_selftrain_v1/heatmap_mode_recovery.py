"""Bounded frozen-model decoder screen on actual coordinate recovery."""
import argparse
import copy
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import torch
from . import recovery_pseudo_denoise as P
from . import identity_rank as I
from . import heatmap_mode_decoder as D

C=P.C;R=P.R;N=P.N
PHASE='heatmap_mode_recovery'
DOC=R.DOC/PHASE;RAW=R.RAW/PHASE
MODELS=['BASE','DENOISE']
ARMS=[f'{m}_{d}' for m in MODELS for d in ['MEAN','MODE']]


def prepare():
    p=P.verify();source=P.SourceData();order=np.load(N.RAW/'ORDERS.npz')
    held=order['held_rows'].astype(int)
    assert len(held)==256 and set(held)<=set(source.held_rows)
    assert not set(held)&set(np.asarray(p['source_samples']).ravel())
    rows=[dict(row=int(row),id=source.data.source['records'][int(source.data.indices[row])]['id'],
               image=C.bound(source.data.source['records'][int(source.data.indices[row])]['image'])) for row in held]
    checkpoints=dict(BASE=p['initialization'],DENOISE=C.read(P.DOC/'FIT_DENOISE.json')['checkpoint'])
    for b in checkpoints.values():C.verify(b)
    protocol=dict(objective='Test whether global heatmap expectation hides a correct distant dominant mode in existing RGB coordinate refiners. Do not confuse identity reassignment with this genuine coordinate change.',
        architecture='Exactly frozen prior Replay BASE and fixed pseudo DENOISE last300;no training or new labels',
        decode='MEAN is existing full96x72 softmax expectation. MODE is softmax conditional expectation in fixed5x5 cells centered on global logit argmax,clipped by image grid boundary,stride4. No temperature/radius search. Ties use flattened first index. No reference coordinate or GT given to decoder.',
        inputs='Original R0 predicted box/RGB/points and unchanged384x288 crop;no GT,dims,PnP,depth,CAD,tag input. Same logits feed both decoders.',
        controls='Paired fixed decoder intervention within each of2existing models;not a new model architecture or retraining claim. Existing BASE/DENOISE mean predictions must reproduce within .003 original-image pixels.',
        source_rows=rows,source_cache_signature=N.source_cache_signature(source.data,held),
        source_screen='Natural uncorrupted256heldout only. MODE versus same-model MEAN: PCK10 drop<=1pp,P90 ratio<=1.10. Against R0: good<5to>10damage<=1percent,hard>20to<=10 recoveries>=same-modelMEAN. No synthetic perturbation recoveries in this gate.',
        decision='No checkpoint/parameter selection on realGT. Freeze source screen and both decoder configurations before all194real predictions. Evaluate all four candidates even if source fails,but explicitly not deployable. No new frame/corner filter and no automatic promotion.',
        real_screen='Full194>=5hard>20to<=10recoveries across>=3frames,good<5to>10damage<=1percent,PCK20>=R0. Also fixed159 subset and >40,>80,>100tails. This screen is not full objective completion.',
        diagnostic='Report GT-free mode-mean displacement,entropy,local probability mass;not a correctness score or selection rule. Evaluation GT used only after all candidate predictions fixed.',
        checkpoints=checkpoints,arms=ARMS,new_training=0,new_annotations=0,new_tags=0,
        caveats='Repeated DEV,prior Replay teacher9manual images includes3evaloverlaps. DENOISE pseudo labels correlated and potentially wrong. Source heldout reused development. No independent confirmation or final-model promotion.',
        sources=p['sources']+[C.bound(x) for x in [__file__,D.__file__,Path(__file__).with_name('test_heatmap_mode_decoder.py'),
            P.__file__,I.__file__,P.DOC/'PROTOCOL.json',P.DOC/'FIT_DENOISE.json',P.DOC/'RESULTS.json',N.RAW/'ORDERS.npz']]+list(checkpoints.values()))
    C.freeze(DOC/'PROTOCOL.json',protocol)
    R.evaluation_protocol(PHASE,ARMS,protocol['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('HEATMAP_MODE_PROTOCOL_FROZEN',len(held),ARMS,flush=True)


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']:C.verify(b)
    return p


def load_model(name,p):
    b=p['checkpoints'][name];C.verify(b)
    ck=torch.load(C.ROOT/b['path'],map_location='cpu',weights_only=False)
    assert ck['step']==300
    m=N.C.PoseFixPallet9();m.load_state_dict(ck['model_state_dict']);del ck
    return m.cuda().eval().requires_grad_(False)


def errors_summary(initial,final):
    a=np.asarray(initial);b=np.asarray(final)
    assert len(a)==len(b) and len(a)>0 and np.isfinite(a).all() and np.isfinite(b).all()
    h=a>20;g=a<5
    return dict(corners=len(b),PCK10=float((b<=10).mean()),PCK20=float((b<=20).mean()),
        median_px=float(np.median(b)),P90_px=float(np.quantile(b,.9)),hard=int(h.sum()),
        recovered=int((h&(b<=10)).sum()),good=int(g.sum()),damaged=int((g&(b>10)).sum()),
        damage_rate=float((g&(b>10)).sum()/max(g.sum(),1)))


@torch.inference_mode()
def source():
    p=verify();N.setup();N.E.gpu();assert torch.cuda.is_available()
    if (DOC/'SOURCE_SCREEN.json').exists():
        for b in C.read(DOC/'SOURCE_SCREEN.json')['artifacts']:C.verify(b)
        print('SOURCE_SCREEN_ALREADY_COMPLETE');return
    data=P.SourceData();rows=np.array([r['row'] for r in p['source_rows']])
    assert N.source_cache_signature(data.data,rows)==p['source_cache_signature']
    for r in p['source_rows']:C.verify(r['image'])
    with ThreadPoolExecutor(max_workers=4) as workers:items=list(workers.map(data.item,rows))
    result={};artifacts=[]
    for name in MODELS:
        model=load_model(name,p);initial=[];errors=dict(MEAN=[],MODE=[]);qs=[];logits_all=[];targets=[];masks=[];matrices=[]
        for start in range(0,len(items),2):
            xs=items[start:start+2];b=P.O.batch(xs);logits=model(b['rgb'],b['points'],b['valid'])
            a=D.ambiguity(logits);logits_all.append(logits.cpu().numpy())
            for j,x in enumerate(xs):
                mask=x['target_valid'];scale=x['matrix'][0,0]
                initial.extend((np.linalg.norm(x['points'][mask]-x['target'][mask],axis=-1)/scale).tolist())
                qs.append(x['points']);targets.append(x['target']);masks.append(mask);matrices.append(x['matrix'])
                for decoder in errors:
                    q=a[decoder.lower()][j].cpu().numpy()
                    errors[decoder].extend((np.linalg.norm(q[mask]-x['target'][mask],axis=-1)/scale).tolist())
            if (start+2)%64==0:print('HEATMAP_SOURCE',name,start+2,'/256',N.E.gpu(),flush=True)
        path=RAW/f'SOURCE_LOGITS_{name}.npz'
        I.save_npz(path,logits=np.concatenate(logits_all),points=np.asarray(qs),target=np.asarray(targets),valid=np.asarray(masks),matrix=np.asarray(matrices),rows=rows)
        artifacts.append(C.bound(path));result[name]={d:errors_summary(initial,e) for d,e in errors.items()}
        m=result[name]['MEAN'];n=result[name]['MODE']
        checks=dict(PCK10=n['PCK10']>=m['PCK10']-.01,P90=n['P90_px']<=m['P90_px']*1.1,
            damage1percent=n['damage_rate']<=.01,natural_recovery=n['recovered']>=m['recovered'])
        result[name].update(checks=checks,passed=all(checks.values()))
        print('HEATMAP_SOURCE_SCREEN',name,result[name],flush=True)
        del model,b,logits,a;gc.collect();torch.cuda.empty_cache()
    C.freeze(DOC/'SOURCE_SCREEN.json',dict(results=result,artifacts=artifacts,protocol=C.bound(DOC/'PROTOCOL.json'),
        natural_source_only=True,artificially_corrupted_inputs=0,real_GT_used=False))


def replace(prediction,item,crop_points):
    result=copy.deepcopy(prediction)
    raw=N.C.transform_points(np.asarray(crop_points),np.linalg.inv(item['matrix']))
    raw[~item['valid']]=item['original_points'][~item['valid']]
    raw[8]=item['original_points'][8]
    N.C.selected(result)['keypoints_xy']=raw.tolist()
    P.assert_preserved(prediction,result)
    return result


@torch.inference_mode()
def infer():
    p=verify();N.setup();N.E.gpu();assert torch.cuda.is_available()
    screen=C.read(DOC/'SOURCE_SCREEN.json')
    for b in screen['artifacts']:C.verify(b)
    C.freeze(DOC/'DECISION_LOCK.json',dict(protocol=C.bound(DOC/'PROTOCOL.json'),source_screen=C.bound(DOC/'SOURCE_SCREEN.json'),
        checkpoints=p['checkpoints'],decoder_radius=2,before_real_forward=True))
    records=C.read(DOC/'EVAL_PROTOCOL.json')['records'];assert len(records)==194
    original={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    prepared={}
    for r in records:
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        item=N.C.prepare_input(im,original[r['id']]['prediction']);assert item is not None
        prepared[r['id']]=item
    parity={};diagnostics=[]
    for name in MODELS:
        path=RAW/f'EVAL_PREDICTIONS_{name}_MODE.json'
        if path.exists():raise RuntimeError('Partial/completed inference exists; audit saved artifacts instead of overwriting')
        model=load_model(name,p);preds=dict(MEAN=[],MODE=[]);logits_all=[];max_delta=0.
        old={r['id']:r for r in C.read(P.RAW/f'EVAL_PREDICTIONS_{name}.json')['records']}
        for idx,r in enumerate(records):
            key=r['id'];item=prepared[key];base=original[key]
            args=[torch.as_tensor(item[k],device='cuda')[None] for k in ['rgb','points','valid']]
            logits=model(*args);a=D.ambiguity(logits);logits_all.append(logits[0].cpu().numpy())
            for decoder in preds:
                out=replace(base['prediction'],item,a[decoder.lower()][0].cpu().numpy())
                preds[decoder].append(dict(id=key,kind='PLASTIC',raw_hw=base['raw_hw'],prediction=out))
                if decoder=='MEAN':
                    previous=np.asarray(P.top(old[key]['prediction'])['keypoints_xy']);now=np.asarray(P.top(out)['keypoints_xy'])
                    max_delta=max(max_delta,float(np.max(np.abs(previous-now))))
                    np.testing.assert_allclose(now,previous,atol=.003,rtol=0)
            diagnostics.append(dict(id=key,model=name,mode_mean_px=(torch.linalg.vector_norm(a['mode']-a['mean'],dim=-1)[0]/item['matrix'][0,0]).cpu().tolist(),
                local_mass=a['local_mass'][0].cpu().tolist(),entropy=a['entropy'][0].cpu().tolist()))
            if (idx+1)%50==0:print('HEATMAP_REAL',name,idx+1,'/194',N.E.gpu(),flush=True)
        for decoder,rows in preds.items():
            C.freeze(RAW/f'EVAL_PREDICTIONS_{name}_{decoder}.json',dict(complete=True,arm=f'{name}_{decoder}',records=rows,
                checkpoint=p['checkpoints'][name],GT_free=True,diagnostic_candidate_not_deployed=True))
        I.save_npz(RAW/f'REAL_LOGITS_{name}.npz',logits=np.asarray(logits_all),ids=np.array([r['id'] for r in records]),
            matrix=np.array([prepared[r['id']]['matrix'] for r in records]))
        parity[name]=max_delta;del model,logits,a;gc.collect();torch.cuda.empty_cache()
    C.freeze(RAW/'AMBIGUITY.json',dict(rows=diagnostics,used_for_selection=False))
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in ARMS]+
        [C.bound(RAW/f'REAL_LOGITS_{m}.npz') for m in MODELS]+[C.bound(RAW/'AMBIGUITY.json')],
        before_real_GT_scoring=True,mean_parity_max_abs_px=parity))


def report():
    verify()
    for b in C.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    scores={a:R.score(PHASE,a,False)['metrics'] for a in ARMS}
    scores['R0']=[r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC']
    order=[r['id'] for r in scores['R0']]
    scores={a:[{r['id']:r for r in rr}[i] for i in order] for a,rr in scores.items()}
    subset=C.read(R.BASE_DOC/'large_corner_recovery_v1/extreme_low_subset_v1/SUBSET_MANIFEST.json')
    keep={r['id'] for r in subset['records'] if r['kind']=='PLASTIC'};results={}
    for label,keys in [('full194',set(order)),('retained159',keep)]:
        base=[r for r in scores['R0'] if r['id'] in keys];table={}
        for arm,rr in scores.items():
            n=[r for r in rr if r['id'] in keys];d=P.recovery_damage(base,n,True)
            frames=sum(b['matched'] and any(v is not None and v>20 and w<=10 for v,w in zip(b['canonical_errors'],q['canonical_errors'])) for b,q in zip(base,n))
            tails={str(t):dict(hard=sum(v>t for b in base if b['matched'] for v in b['canonical_errors'] if v is not None),
                recovered=sum(v is not None and v>t and w<=10 for b,q in zip(base,n) if b['matched'] for v,w in zip(b['canonical_errors'],q['canonical_errors']))) for t in [40,80,100]}
            table[arm]=dict(summary=P.summary(n),recovery=d,recovered_frames=frames,tails=tails)
        results[label]=table
    source=C.read(DOC/'SOURCE_SCREEN.json')['results'];checks={}
    for m in MODELS:
        arm=m+'_MODE';x=results['full194'][arm]
        checks[arm]=dict(source=source[m]['passed'],recovery5=x['recovery']['recovered']>=5,frames3=x['recovered_frames']>=3,
            damage1percent=x['recovery']['damage_rate']<=.01,PCK20=x['summary']['PCK']['20']>=results['full194']['R0']['summary']['PCK']['20'])
    ex='eval_pallet07:1778652166837872128'
    result=dict(results=results,checks=checks,passed={a:all(v.values()) for a,v in checks.items()},
        example={a:next(r for r in rr if r['id']==ex) for a,rr in scores.items()},
        goal_complete=False,new_training=0,new_annotations=0,new_tags=0,auto_promoted=False,independent_confirmation=False)
    C.freeze(DOC/'RESULTS.json',result)
    lines=['# 기존 PoseFix 확률 분포의 평균 vs 최상위 봉우리 주변 좌표','',
        '가중치와 입력·crop은 그대로다. 새 레이블/태그/학습0. MODE는 고정5x5국소 평균으로, GT 선택이나 새 필터가 아니다. 네 후보 모두 미배포 진단 결과다.','',
        '| 평가 | 모델/디코더 | PCK20 % | 중앙 px | P90 px | >20→≤10 | <5→>10손상 | >40→≤10 | >100→≤10 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,table in results.items():
        for arm in ['R0',*ARMS]:
            x=table[arm];s=x['summary'];d=x['recovery'];t=x['tails']
            lines.append(f'|{label}|{arm}|{100*s["PCK"]["20"]:.2f}|{s["matched_pooled_corner8_median_px"]:.3f}|{s["matched_pooled_corner8_P90_px"]:.3f}|{d["recovered"]}/{d["hard"]}|{d["damaged"]}/{d["good"]}|{t["40"]["recovered"]}/{t["40"]["hard"]}|{t["100"]["recovered"]}/{t["100"]["hard"]}|')
    lines+=['',f'사전 screen: {result["passed"]}.','의자·콘 사진 평균 px: '+str({a:r['frame_mean_px'] for a,r in result['example'].items()}),
        '반복 DEV,역사적 teacher3이미지 overlap. 원래 모델·수도레이블 보존. 확률 봉우리가 있다고 그 위치가 실제 정답인 것은 아니다.']
    C.write_text(RAW/'REPORT_KO.md','\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','source','infer','report']);args=p.parse_args()
    globals()[args.action]()
