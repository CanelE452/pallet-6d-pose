"""Same pseudo targets/exposures, one extra input-hint masking intervention."""
import argparse
import copy
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import torch
from . import recovery_pseudo_denoise as P
from . import hint_dropout as M
from . import heatmap_mode_recovery as H

C=P.C;R=P.R;N=P.N
PHASE='hint_dropout'
DOC=R.DOC/PHASE;RAW=R.RAW/PHASE
ARMS=['DENOISE_NATIVE','DENOISE_BLIND','MASKED_NATIVE','MASKED_BLIND']


def prepare():
    parent=P.verify();p=copy.deepcopy(parent)
    control=C.read(P.DOC/'FIT_DENOISE.json')['checkpoint'];C.verify(control)
    p.update(arms=ARMS,control=control,
        objective='Learn to recover a corner from RGB and other corner hints rather than requiring its own possibly wrong hint. No new labels/tags or realGT supervision.',
        controls='Existing DENOISE last300 versus new MASKED last300, both from same Replay initialization,identical data ordering/targets/perturbation RNG/loss/optimizer/exposures/BN policy. Only add independent50percent chance of hiding one supervised valid corner hint in each real/source image. No model-size/lr/step sweep.',
        corruption='EXACT DENOISE real corruption and source corruption with original SeedSequence([20260922,step,branch]);additional mask RNG SeedSequence([20260926,step,branch]). Mask only gaussian conditioning validity; preserve coordinates,target,target_valid,RGB,other inputs. Center never hidden.',
        evaluation='Four fixed arms:each model NATIVE unchanged full-hint expectation,BLIND eight forward passes each hiding only the output corner own hint,then collecting that corner. Original R0 crop/box preserved,uncapped mean decoder. No GT selection,corner/frame rejection,mode decoder or new geometry filter. BLIND is NOT PnP LOO filtering.',
        source_gate='Same64source-heldout as prior DENOISE. NATIVE and BLIND outputs compared with frozen DENOISE_NATIVE:clean PCK10 drop<=1pp,P90 ratio<=1.1. Natural source inputs only;separate stress training diagnostics not counted as real recovery.',
        real_gate='Full194:>=5hard>20to<=10 across>=3frames,good<5to>10damage<=1percent,PCK20>=R0,plus source gate. Report fixed159 and >40/80/100tails. No realGT checkpoint/threshold selection. Screen not full objective success.',
        output_lock='Freeze both-model4arm predictions before any realGT scoring. Model/decoder choice not selected on realGT. No automatic promotion or new pseudo-label generation.',
        new_annotations=0,new_tags=0,auto_promote=False,
        sources=parent['sources']+[C.bound(x) for x in [__file__,M.__file__,Path(__file__).with_name('test_hint_dropout.py'),
            P.__file__,H.__file__,P.DOC/'PROTOCOL.json',P.DOC/'FIT_DENOISE.json',P.DOC/'RESULTS.json',
            R.DOC/'heatmap_mode_recovery/RESULTS.json',R.DOC/'heatmap_mode_recovery/SPATIAL_DECOMPOSITION.json']]+[control])
    C.freeze(DOC/'PROTOCOL.json',p)
    R.evaluation_protocol(PHASE,ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('HINT_DROPOUT_PROTOCOL_FROZEN','same217pseudo source exposures300x8','300steps',flush=True)


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']+[p['initialization'],p['control']]:C.verify(b)
    return p


def save(path,obj):
    assert path.resolve().is_relative_to(RAW) and not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(obj,path)


def train():
    p=verify();N.setup();N.E.gpu();assert torch.cuda.is_available()
    if (DOC/'FIT_MASKED.json').exists():C.verify(C.read(DOC/'FIT_MASKED.json')['checkpoint']);print('MASKED_ALREADY_FIT');return
    real,source,held=P.load_inputs(p);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    model=N.load_model().requires_grad_(True);optimizer=P.TFAdam(model.parameters(),lr=1e-4)
    before=P.NT.source_probe(model,held);model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.BatchNorm2d):m.eval()
    buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    history=[];start_step=0;prior_seconds=0.;hidden=dict(real=0,source=0)
    resumes=sorted((RAW/'MASKED').glob('resume_*.pt')) if (RAW/'MASKED').exists() else []
    if resumes:
        ck=torch.load(resumes[-1],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
        model.load_state_dict(ck['model_state_dict']);optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_step=ck['step'];history=ck['history'];prior_seconds=ck['seconds'];hidden=ck['hidden'];del ck
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as workers:
        pending=[workers.submit(source.item,i) for i in p['source_samples'][start_step]] if start_step<300 else []
        for idx in range(start_step,300):
            syn=[f.result() for f in pending]
            pending=[workers.submit(source.item,i) for i in p['source_samples'][idx+1]] if idx+1<300 else []
            rng=np.random.default_rng(np.random.SeedSequence([20260922,idx,0]))
            rr=[P.perturb(real[i],rng) for i in p['real_samples'][idx]]
            srng=np.random.default_rng(np.random.SeedSequence([20260922,idx,1]))
            branches=dict(real=rr,source=[P.O.corrupted(x,srng) for x in syn])
            for b,(branch,items) in enumerate(branches.items()):
                mrng=np.random.default_rng(np.random.SeedSequence([20260926,idx,b]))
                masked=[M.mask_one(x,mrng) for x in items]
                for old,new in zip(items,masked):
                    np.testing.assert_array_equal(old['target'],new['target'])
                    np.testing.assert_array_equal(old['target_valid'],new['target_valid'])
                    np.testing.assert_array_equal(old['points'],new['points'])
                    hidden[branch]+=int((old['valid']&~new['valid']).sum())
                branches[branch]=masked
            optimizer.zero_grad(set_to_none=True);details={}
            for branch,items in branches.items():
                parts=[]
                for j in range(0,8,2):
                    b=P.O.batch(items[j:j+2]);q=model(b['rgb'],b['points'],b['valid'])
                    _,detail=model.losses(q,b['target'],b['target_valid'])
                    loss=P.NT.combined_micro_loss(detail,branch);assert torch.isfinite(loss)
                    loss.backward();parts.append({k:float(v.detach()) for k,v in detail.items()})
                    del b,q,detail,loss
                details[branch]={k:float(np.mean([v[k] for v in parts])) for k in parts[0]}
            update=float(optimizer.step());assert np.isfinite(update) and update>0
            history.append(dict(step=idx+1,loss=details,update_norm=update))
            if (idx+1)%100==0:
                save(RAW/'MASKED'/f'resume_{idx+1:03d}.pt',dict(step=idx+1,history=history,hidden=hidden,
                    protocol_sha256=C.sha(DOC/'PROTOCOL.json'),model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),seconds=prior_seconds+time.monotonic()-started))
            if idx==start_step or (idx+1)%50==0:
                print('HINT_MASK_TRAIN',idx+1,'/300',round(prior_seconds+time.monotonic()-started,1),'hidden',hidden,N.E.gpu(),flush=True)
    assert all(torch.equal(v,buffers[k]) for k,v in model.named_buffers())
    after=P.NT.source_probe(model,held)
    pseudo=P.NT.source_probe(model,[real[r['id']] for r in p['real_records'] if not r['train']])
    path=RAW/'MASKED/last300.pt'
    save(path,dict(step=300,model_state_dict=model.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')))
    C.freeze(DOC/'FIT_MASKED.json',dict(complete=True,step=300,checkpoint=C.bound(path),protocol=C.bound(DOC/'PROTOCOL.json'),
        source_before=before,source_after=after,held32_pseudo_consistency_NOT_accuracy=pseudo,
        masked_hint_exposures=hidden,BN_buffers_bit_exact=True,history=history,
        seconds=prior_seconds+time.monotonic()-started,new_annotations=0,new_tags=0,auto_promoted=False))
    print('HINT_MASK_TRAIN_COMPLETE',flush=True)


def load_model(name,p):
    ckpt=p['control'] if name=='DENOISE' else C.read(DOC/'FIT_MASKED.json')['checkpoint']
    C.verify(ckpt);ck=torch.load(C.ROOT/ckpt['path'],map_location='cpu',weights_only=False)
    assert ck['step']==300
    model=N.C.PoseFixPallet9();model.load_state_dict(ck['model_state_dict']);del ck
    return model.cuda().eval().requires_grad_(False),ckpt


@torch.no_grad()
def probe(model,items,blind):
    initial=[];final=[];max_move=[]
    for idx in range(0,len(items),2):
        xs=items[idx:idx+2];b=P.O.batch(xs)
        q=M.predict_crop(model,b['rgb'],b['points'],b['valid'],blind).cpu().numpy()
        for x,y in zip(xs,q):
            mask=x['target_valid'];scale=x['matrix'][0,0]
            initial.extend((np.linalg.norm(x['points'][mask]-x['target'][mask],axis=-1)/scale).tolist())
            final.extend((np.linalg.norm(y[mask]-x['target'][mask],axis=-1)/scale).tolist())
            max_move.extend((np.linalg.norm(y[mask]-x['points'][mask],axis=-1)/scale).tolist())
    r=H.errors_summary(initial,final);r['displacement_quantiles']=np.quantile(max_move,[.5,.9,1]).tolist()
    return r


def source_screen():
    p=verify();N.setup();N.E.gpu();assert torch.cuda.is_available()
    data=P.SourceData();held=[data.item(i) for i in p['source_held_rows']]
    result={}
    for name in ['DENOISE','MASKED']:
        model,_=load_model(name,p)
        for method in ['NATIVE','BLIND']:
            result[f'{name}_{method}']=probe(model,held,method=='BLIND')
            print('HINT_SOURCE',name,method,result[f'{name}_{method}'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    b=result['DENOISE_NATIVE'];checks={}
    for arm,n in result.items():
        checks[arm]=dict(PCK10=n['PCK10']>=b['PCK10']-.01,P90=n['P90_px']<=b['P90_px']*1.1)
    C.freeze(DOC/'SOURCE_SCREEN.json',dict(results=result,checks=checks,passed={a:all(v.values()) for a,v in checks.items()},
        real_GT_used=False,natural_inputs_only=True,rows=p['source_held_rows'],fit=C.bound(DOC/'FIT_MASKED.json')))


@torch.no_grad()
def infer():
    p=verify();N.setup();N.E.gpu();assert torch.cuda.is_available()
    C.freeze(DOC/'DECISION_LOCK.json',dict(fit=C.bound(DOC/'FIT_MASKED.json'),source_screen=C.bound(DOC/'SOURCE_SCREEN.json'),
        protocol=C.bound(DOC/'PROTOCOL.json'),before_real_forward=True,auto_selection=False))
    records=C.read(DOC/'EVAL_PROTOCOL.json')['records']
    original={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    old={r['id']:r for r in C.read(P.RAW/'EVAL_PREDICTIONS_DENOISE.json')['records']}
    prepared={}
    for r in records:
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        prepared[r['id']]=N.C.prepare_input(im,original[r['id']]['prediction']);assert prepared[r['id']] is not None
    parity=0.;timings={}
    for name in ['DENOISE','MASKED']:
        model,ckpt=load_model(name,p);results=dict(NATIVE=[],BLIND=[])
        times=dict(NATIVE=0.,BLIND=0.)
        for idx,r in enumerate(records):
            key=r['id'];item=prepared[key];base=original[key]
            args=[torch.as_tensor(item[k],device='cuda')[None] for k in ['rgb','points','valid']]
            for method in results:
                torch.cuda.synchronize();start=time.monotonic()
                q=M.predict_crop(model,*args,blind=method=='BLIND')[0].cpu().numpy()
                times[method]+=time.monotonic()-start
                out=H.replace(base['prediction'],item,q)
                if name=='DENOISE' and method=='NATIVE':
                    a=np.asarray(P.top(old[key]['prediction'])['keypoints_xy']);b=np.asarray(P.top(out)['keypoints_xy'])
                    parity=max(parity,float(np.abs(a-b).max()));np.testing.assert_allclose(a,b,atol=.003,rtol=0)
                results[method].append(dict(id=key,kind='PLASTIC',raw_hw=base['raw_hw'],prediction=out))
            if (idx+1)%50==0:print('HINT_REAL',name,idx+1,'/194',N.E.gpu(),flush=True)
        for method,rows in results.items():
            arm=f'{name}_{method}'
            C.freeze(RAW/f'EVAL_PREDICTIONS_{arm}.json',dict(complete=True,arm=arm,checkpoint=ckpt,records=rows,GT_free=True))
            timings[arm]=times[method]/len(records)*1000
        del model;gc.collect();torch.cuda.empty_cache()
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in ARMS],
        before_real_GT_scoring=True,control_parity_max_abs_px=parity,observed_network_ms_per_image=timings,
        latency_note='Diagnostic timing only:includesGPUtransfer/decode,excludesdisk/crop/detector;BLIND8forwards versusNATIVE1'))


def report():
    verify()
    for b in C.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    scores={a:R.score(PHASE,a,False)['metrics'] for a in ARMS}
    scores['R0']=[r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC']
    order=[r['id'] for r in scores['R0']];scores={a:[{r['id']:r for r in rr}[i] for i in order] for a,rr in scores.items()}
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
    source=C.read(DOC/'SOURCE_SCREEN.json');checks={}
    for arm in ARMS:
        x=results['full194'][arm]
        checks[arm]=dict(source=source['passed'][arm],recovery5=x['recovery']['recovered']>=5,frames3=x['recovered_frames']>=3,
            damage1percent=x['recovery']['damage_rate']<=.01,PCK20=x['summary']['PCK']['20']>=results['full194']['R0']['summary']['PCK']['20'])
    ex='eval_pallet07:1778652166837872128'
    result=dict(results=results,checks=checks,passed={a:all(v.values()) for a,v in checks.items()},
        example={a:next(r for r in rr if r['id']==ex) for a,rr in scores.items()},
        goal_complete=False,new_annotations=0,new_tags=0,auto_promoted=False,independent_confirmation=False)
    C.freeze(DOC/'RESULTS.json',result)
    lines=['# 입력 코너 힌트 dropout: 같은 수도레이블, 같은300step','',
        '새 정답/태그0. MASKED는 DENOISE와 동일 프로토콜에50%단일 hint dropout만 추가했다. NATIVE는 평소 입력,BLIND는 각 출력 코너의 자기 hint를 가린8회 추론이다. PnP LOO 필터가 아니며 프레임/코너를 거절하지 않는다.','',
        '| 평가 | 모델/입력 | PCK20 % | 중앙 px | P90 px | >20→≤10 | <5→>10손상 | >40→≤10 | >100→≤10 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,table in results.items():
        for arm in ['R0',*ARMS]:
            x=table[arm];s=x['summary'];d=x['recovery'];t=x['tails']
            lines.append(f'|{label}|{arm}|{100*s["PCK"]["20"]:.2f}|{s["matched_pooled_corner8_median_px"]:.3f}|{s["matched_pooled_corner8_P90_px"]:.3f}|{d["recovered"]}/{d["hard"]}|{d["damaged"]}/{d["good"]}|{t["40"]["recovered"]}/{t["40"]["hard"]}|{t["100"]["recovered"]}/{t["100"]["hard"]}|')
    lines+=['',f'사전 screen: {result["passed"]}.','의자·콘 사진 평균 px: '+str({a:r['frame_mean_px'] for a,r in result['example'].items()}),
        '반복 DEV이며 기존 teacher 수동 학습3장이 평가와 겹친다. 독립 확인 아님. 기존 모델·데이터는 보존했고 자동 승격하지 않았다.']
    C.write_text(RAW/'REPORT_KO.md','\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','source_screen','infer','report']);a=p.parse_args();globals()[a.action]()
