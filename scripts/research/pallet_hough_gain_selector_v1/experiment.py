"""Phase-ordered, no-sweep frozen proposal selector reproduction."""
import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset, collate, observation_batch
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.util import immutable_json, read_json, sha256, canonical_sha
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.runner import load_checkpoint, seed_all, sequence
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.assessment import _average_ranks
from .core import action_features, assigned_target, choose, GainSelector, metrics, gate

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'scripts/research/pallet_hough_gain_selector_v1'
DOC=ROOT/'_docs/experiments/pallet_hough_gain_selector_v1'
RAW=ROOT/'data/pallet/results/pallet_hough_gain_selector_v1'
EXPORT=ROOT/'data/pallet/results/pallet_symmetry_dht_local_v1/export'
CORRECTED=ROOT/'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction'
CHECKPOINT=CORRECTED/'heads/hough_seed1/checkpoint_final.pt'
TEST_Q=CORRECTED/'predictions/hough_seed1_synth_val.json'
SPLITS={'train':'train','calibration':'calibration','test':'synth_val'}
TAUS=[0.,.05,.1,.25,.5,1.]


def save(path, value):
    immutable_json(path, value)


def preserved():
    names=('pallet_symmetry_dht_local_v1','pallet_symmetry_dht_local_v2',
           'pallet_symmetry_dht_local_v2_wls_correction','pallet_point_line_v4','pallet_line_pose_v1')
    return {str(p.relative_to(ROOT)):sha256(p) for base in ('data/pallet/results','_docs/experiments')
            for name in names for p in sorted((ROOT/base/name).rglob('*')) if p.is_file()}


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def distribution(a):
    a=np.asarray(a,float)
    if not len(a): return {'count':0,'mean':None,'median':None,'std':None}
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),std=float(a.std()),
                positive_rate=float((a>0).mean()),negative_rate=float((a<0).mean()),
                quantiles=dict(zip(('min','p01','p10','p90','p99','max'),map(float,np.quantile(a,[0,.01,.1,.9,.99,1])))))


def correlation(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    mask=np.isfinite(a)&np.isfinite(b);a,b=a[mask],b[mask]
    if len(a)<2 or a.std()==0 or b.std()==0:return dict(Pearson=None,Spearman=None,count=len(a))
    return dict(Pearson=float(np.corrcoef(a,b)[0,1]),
                Spearman=float(np.corrcoef(_average_ranks(a),_average_ranks(b))[0,1]),count=len(a))


def cohorts(rows):
    result={}
    for field in ('source','session_id'):
        result[field]={key:distribution([max(0,r['gain_px']) for r in rows if r[field]==key]) for key in sorted({r[field] for r in rows})}
    order=np.argsort([r['p_mean'] for r in rows],kind='stable')
    result['baseline_difficulty_quartiles']=[dict(quartile=i+1,mean_P_px=float(np.mean([rows[k]['p_mean'] for k in ids])),
        oracle_gain=distribution([max(0,rows[k]['gain_px']) for k in ids])) for i,ids in enumerate(np.array_split(order,4))]
    return result


def target_rows(split, prediction):
    data=ObservationDataset(EXPORT/(SPLITS[split]+'.json'),targets=True)
    saved={r['frame_id']:r for r in prediction['records']};rows=[]
    assert set(saved)=={r['frame_id'] for r in data.records}
    for i,item in enumerate(data):
        pred=saved[item['frame_id']]; p=item['base_points'].numpy();q=np.asarray(pred['points'])
        assert np.array_equal(p[8],q[8]) and pred['point_valid']==item['point_valid'].tolist()
        d=float(item['image_hw'].norm())
        r=assigned_target(p,q,item['point_valid'],item['target_points'],item['target_valid'],item['symmetry_permutations'],d)
        for key in ('p_error','q_error','mask'):r[key]=r[key].tolist()
        perm=item['symmetry_permutations'][r['choice']]; y=item['target_points'][perm].numpy();valid=item['target_valid'][perm].numpy()
        lines=np.asarray(pred['raw_line']);before=[];line_errors=[]
        from .core import EDGES
        for e,(a,b) in enumerate(EDGES):
            before.extend(np.abs(p[[a,b]]@lines[e,:2]+lines[e,2]).tolist())
            if valid[[a,b]].all():line_errors.extend(np.abs(y[[a,b]]@lines[e,:2]+lines[e,2]).tolist())
        r.update(frame_id=item['frame_id'],diagonal=d,source=data.records[i]['source'],session_id=data.records[i]['session_id'],
                 correction_mean_px=float(np.linalg.norm(q[:8]-p[:8],axis=-1).mean()),utility_mean=float(np.mean(pred['utility'])),
                 point_line_distance_px=float(np.mean(before)),GT_line_error_px=float(np.mean(line_errors)) if line_errors else None)
        rows.append(r)
    return rows


def lock():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='main'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor','4a6c0abfd185fef9d06d9db92a22cc0c63249b9f','HEAD'],cwd=ROOT,check=True)
    DOC.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    manifests={s:read_json(EXPORT/(name+'.json')) for s,name in SPLITS.items()}
    overlap={}
    for field in ('frame_id','source_image','source_image_sha256','observation_sha256'):
        values={s:{r[field] for r in m['records']} for s,m in manifests.items()}
        overlap[field]={f'{a}/{b}':len(values[a]&values[b]) for a,b in (('train','calibration'),('train','test'),('calibration','test'))}
    assert all(n==0 for v in overlap.values() for n in v.values()),overlap
    split_audit={'counts':{s:len(m['records']) for s,m in manifests.items()},'overlap':overlap,
        'manifest_sha256':{s:sha256(EXPORT/(name+'.json')) for s,name in SPLITS.items()},
        'session_counts':{s:{key:sum(r['session_id']==key for r in m['records']) for key in sorted({r['session_id'] for r in m['records']})} for s,m in manifests.items()},
        'frame_disjoint':True,'session_disjoint':False,'limitation':'Hough trained on selector train; session/source categories overlap; no session-held-out claim',
        'selector_inputs_exclude_GT_ID_source_session_symmetry':True}
    save(DOC/'DATA_SPLIT_AUDIT.json',split_audit)
    save(DOC/'PRESERVED_ARTIFACT_SHA.json',preserved())
    save(DOC/'PROTOCOL_LOCK.json',dict(start_main=head,start_origin_main=remote,branch=branch,start_worktree_clean_before_new_files=True,
        ancestor_verified='4a6c0abfd185fef9d06d9db92a22cc0c63249b9f',frozen_hough_seed=1,upstream_seed_selection='first numeric seed; no upstream seed comparison',
        frozen_checkpoint=str(CHECKPOINT.relative_to(ROOT)),frozen_checkpoint_sha256=sha256(CHECKPOINT),test_Q_sha256=sha256(TEST_Q),
        assignment='baseline-selected whole-object C1/C2, identical assignment for P and Q in targets AND reported metrics',
        target='frame gain / diagonal',network_output_scale=1000.,loss='SmoothL1 beta=1 on 1000*normalized gain',
        hidden=[64,64],dropout=.1,seeds=[1,2,3],steps_per_seed=2000,batch=64,optimizer='AdamW',lr=.001,weight_decay=.0001,
        gradient_clip=10.,checkpoint_selection='last step only',tau_units='raw px equivalent = predicted normalized gain * frame raw diagonal',tau_grid=TAUS,
        calibration='min primary subject to median/P90 nonworse and good damage<=.005; tie tolerance1e-12 higher tau; infeasible -> always P without extra grid',
        oracle_gate='primary relative gain>=.01 AND median/P90 nonworse',success='all 3 seeds: primary>=1%, median/P90/coverage nonworse, damage<=.005, catastrophic0; all selected-Q mean gains>0 and seed mean realized gain>0',
        real_DEV='only after all 3 seeds pass; fixed weights/tau',FINAL_open_allowed=False,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (PACKAGE/'core.py',PACKAGE/'experiment.py',DOC/'PURPOSE_AND_SCOPE.md')}))
    print('PROTOCOL AND SPLIT LOCKED',flush=True)


def oracle():
    read_json(DOC/'PROTOCOL_LOCK.json')
    q=read_json(TEST_Q);rows=target_rows('test',q)
    p=metrics(rows,[False]*len(rows));always=metrics(rows,[True]*len(rows));selected=np.array([r['gain_px']>0 for r in rows])
    best=metrics(rows,selected);checks=gate(p,best)['checks'];passed=all(checks[k] for k in ('primary_gain_ge_1pct','median_nonworse','p90_nonworse'))
    gains=[max(0,r['gain_px']) for r in rows]
    result=dict(Point=p,Always_Q=always,Oracle_P_or_Q=best,oracle_selected_Q_rate=float(selected.mean()),
        oracle_mean_gain_px=float(np.mean(gains)),oracle_gain_distribution=distribution(gains),cohorts=cohorts(rows),
        headroom_gate=passed,relative_primary_gain=(p['primary']-best['primary'])/p['primary'],
        status='SELECTOR_HEADROOM_PASS' if passed else 'SELECTOR_NO_HEADROOM',GT_assisted_not_deployment=True,
        Q_source_sha256=sha256(TEST_Q),records=rows)
    save(DOC/'ORACLE_HEADROOM.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','cohorts')},indent=2),flush=True)


@torch.inference_mode()
def cache():
    assert read_json(DOC/'ORACLE_HEADROOM.json')['headroom_gate']
    model,_=load_checkpoint(CHECKPOINT,torch.device('cuda:0'));model.requires_grad_(False);before=state_hash(model)
    provenance=[]
    for split,name in SPLITS.items():
        data=ObservationDataset(EXPORT/(name+'.json'),targets=False);xs=[];ps=[];qs=[];ds=[];records=[]
        saved={r['frame_id']:r for r in read_json(TEST_Q)['records']} if split=='test' else None
        for start in range(0,len(data),32):
            items=[data[i] for i in range(start,min(start+32,len(data)))]
            obs=observation_batch(collate(items),torch.device('cuda:0'))
            keys=('points','raw_line','utility','mode_mass','ambiguity')
            if saved is None:
                output=model(obs);proposal={k:output[k] for k in keys}
                for j,item in enumerate(items):
                    records.append(dict(frame_id=item['frame_id'],point_valid=item['point_valid'].tolist(),
                        **{k:proposal[k][j].cpu().tolist() for k in keys}))
            else:
                proposal={k:torch.tensor([saved[it['frame_id']][k] for it in items],device='cuda:0') for k in keys}
                records.extend(saved[it['frame_id']] for it in items)
            x,schema=action_features(obs,proposal)
            xs.append(x.cpu());ps.append(obs['base_points'].cpu());qs.append(proposal['points'].cpu());ds.append(obs['image_hw'].norm(dim=-1).cpu())
        path=RAW/f'{split}_observations.pt'
        if path.exists():raise FileExistsError(path)
        torch.save(dict(x=torch.cat(xs),p=torch.cat(ps),q=torch.cat(qs),diagonal=torch.cat(ds),
                        frame_ids=[r['frame_id'] for r in records],GT_opened=False),path)
        pred=dict(records=records,GT_opened=False,checkpoint_sha256=sha256(CHECKPOINT))
        save(RAW/f'{split}_Q.json',pred)
        rows=target_rows(split,pred);save(RAW/f'{split}_targets.json',rows)
        provenance.append(dict(split=split,count=len(data),observation_cache_sha256=sha256(path),prediction_sha256=sha256(RAW/f'{split}_Q.json'),
            target_sha256=sha256(RAW/f'{split}_targets.json'),upstream_forwards=0 if saved is not None else len(data),GT_access_in_feature_path=0))
        print(f'CACHED {split}: {len(data)} frames, {xs[0].shape[1]} features',flush=True)
    after=state_hash(model);assert before==after and all(not p.requires_grad for p in model.parameters())
    save(DOC/'FEATURE_SCHEMA.json',dict(dimension=sum(v[1]-v[0] for v in schema.values()),groups=schema,
        forbidden_inputs=['GT','gain_target','GT_line_error','frame_id','source','session','symmetry_assignment']))
    save(DOC/'FROZEN_UPSTREAM_AUDIT.json',dict(before_state_sha256=before,after_state_sha256=after,parameter_diff=0,
        all_parameters_requires_grad_false=True,gradients_present=False,point_WLS_training_updates=0,
        source_checkpoint_sha256=sha256(CHECKPOINT),caches=provenance))


def sanity():
    result={}
    for split in SPLITS:
        rows=read_json(RAW/f'{split}_targets.json');g=np.array([r['gain_px'] for r in rows])
        positive=np.sort(np.maximum(g,0))[::-1];total=positive.sum()
        by={key:{value:distribution([r['gain_px'] for r in rows if r[key]==value]) for value in sorted({r[key] for r in rows})} for key in ('source','session_id')}
        order=np.argsort([r['p_mean'] for r in rows],kind='stable')
        by['P_difficulty_quartiles']=[distribution([rows[k]['gain_px'] for k in ids]) for ids in np.array_split(order,4)]
        result[split]=dict(gain_px=distribution(g),gain_normalized=distribution([r['gain_normalized'] for r in rows]),
            positive_tail=distribution(g[g>0]),negative_tail=distribution(g[g<0]),groups=by,
            positive_gain_top_1pct_share=float(positive[:max(1,int(np.ceil(len(g)*.01)))].sum()/total) if total else 0.,
            correlations={key:correlation([np.nan if r[key] is None else r[key] for r in rows],g) for key in
                ('correction_mean_px','GT_line_error_px','utility_mean','point_line_distance_px')})
    save(DOC/'GAIN_TARGET_AUDIT.json',dict(splits=result,GT_line_error_diagnostic_only=True,
        interpretation='Univariate diagnostics do not establish leakage-free generalization; no design/threshold changes permitted based on these results'))
    print('TARGET SANITY AUDIT COMPLETE',flush=True)


def train():
    protocol=read_json(DOC/'PROTOCOL_LOCK.json');assert read_json(DOC/'ORACLE_HEADROOM.json')['headroom_gate']
    read_json(DOC/'GAIN_TARGET_AUDIT.json');read_json(DOC/'REGRESSION_TESTS.json')
    for path,digest in protocol['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    data=torch.load(RAW/'train_observations.pt',map_location='cpu');rows=read_json(RAW/'train_targets.json')
    assert data['GT_opened'] is False and data['frame_ids']==[r['frame_id'] for r in rows]
    x=data['x'].detach().cuda();y=torch.tensor([1000*r['gain_normalized'] for r in rows],device='cuda:0')
    mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-5);runs=[]
    for seed in (1,2,3):
        path=RAW/f'seed{seed}';path.mkdir(exist_ok=False)
        seed_all(seed);model=GainSelector(x.shape[1],mean,std).cuda();initial=state_hash(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters()}
        order=sequence(len(x),2000*64,seed);trace=[];started=time.perf_counter()
        for step in range(2000):
            index=torch.tensor(order[step*64:(step+1)*64],device='cuda:0')
            optimizer.zero_grad(set_to_none=True);loss=F.smooth_l1_loss(model(x[index]),y[index],beta=1.)
            assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm)
            optimizer.step()
            if step==0 or (step+1)%500==0:
                row=dict(step=step+1,loss=float(loss),gradient_norm=float(norm));trace.append(row)
                print(f'SELECTOR seed{seed} {json.dumps(row)}',flush=True)
        final=state_hash(model);assert final!=initial and x.grad is None
        checkpoint=path/'checkpoint_final.pt'
        torch.save(dict(state_dict=model.state_dict(),dimension=x.shape[1],seed=seed,steps=2000,protocol_sha256=sha256(DOC/'PROTOCOL_LOCK.json')),checkpoint)
        runs.append(dict(seed=seed,updates=2000,batch=64,parameter_count=sum(p.numel() for p in model.parameters()),initial_sha256=initial,final_sha256=final,
            order_sha256=canonical_sha(order),train_cache_sha256=sha256(RAW/'train_observations.pt'),train_targets_sha256=sha256(RAW/'train_targets.json'),
            checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha256(checkpoint),trace=trace,elapsed_seconds=time.perf_counter()-started,
            optimizer_only_selector=True,upstream_updates=0,calibration_used_in_training=False,test_used_in_training=False))
    assert sha256(CHECKPOINT)==protocol['frozen_checkpoint_sha256']
    save(DOC/'TRAINING_AUDIT.json',dict(runs=runs,PASS=True,device='cuda:0',checkpoint_selection='last step; no validation selection'))


def selector_stats(rows,predicted,pick,tau):
    g=np.array([r['gain_px'] for r in rows]);d=np.array([r['diagonal'] for r in rows]);gh=predicted*d
    chosen=g[pick];positive=g>0;sign=gh>0
    return dict(tau_px=tau,selected_Q_rate=float(pick.mean()),selected_count=int(pick.sum()),
        true_positive_gain_rate_among_selected=float((chosen>0).mean()) if len(chosen) else None,
        false_harm_rate_among_selected=float((chosen<0).mean()) if len(chosen) else None,
        selected_Q_realized_mean_gain_px=float(chosen.mean()) if len(chosen) else None,
        selected_Q_realized_median_gain_px=float(np.median(chosen)) if len(chosen) else None,
        population_realized_mean_gain_px=float(np.where(pick,g,0).mean()),
        G_hat_MAE_px=float(np.abs(gh-g).mean()),G_hat_MAE_normalized=float(np.abs(predicted-g/d).mean()),
        correlation=correlation(gh,g),sign_accuracy=float((sign==positive).mean()),
        positive_gain_precision=float(positive[sign].mean()) if sign.any() else None,
        positive_gain_recall=float(sign[positive].mean()) if positive.any() else None)


@torch.inference_mode()
def evaluate():
    cal=torch.load(RAW/'calibration_observations.pt',map_location='cpu');test=torch.load(RAW/'test_observations.pt',map_location='cpu')
    cr=read_json(RAW/'calibration_targets.json');tr=read_json(RAW/'test_targets.json')
    baseline_cal=metrics(cr,[False]*len(cr));baseline=metrics(tr,[False]*len(tr));results=[];diagnostics=[];calibrations=[];csvrows=[]
    for seed in (1,2,3):
        ck=torch.load(RAW/f'seed{seed}/checkpoint_final.pt',map_location='cpu')
        model=GainSelector(ck['dimension'],ck['state_dict']['mean'],ck['state_dict']['std']);model.load_state_dict(ck['state_dict']);model.eval()
        cp=model(cal['x']).numpy()/1000.;tp=model(test['x']).numpy()/1000.
        candidates=[]
        for tau in TAUS:
            _,pick=choose(cal['p'],cal['q'],torch.from_numpy(cp),cal['diagonal'],tau)
            m=metrics(cr,pick.numpy());feasible=m['median_px']<=baseline_cal['median_px'] and m['p90_px']<=baseline_cal['p90_px'] and m['good_point_damage_rate']<=.005
            candidates.append(dict(tau_px=tau,metrics=m,feasible=feasible))
        feasible=[r for r in candidates if r['feasible']]
        if feasible:
            best=min(r['metrics']['primary'] for r in feasible)
            tau=max(r['tau_px'] for r in feasible if r['metrics']['primary']<=best+1e-12);force=False
        else:tau=None;force=True
        calibrations.append(dict(seed=seed,tau_px=tau,force_Point=force,candidates=candidates,calibration_only=True))
        output,pick=choose(test['p'],test['q'],torch.from_numpy(tp),test['diagonal'],tau if tau is not None else 0.,force)
        pick=pick.numpy();m=metrics(tr,pick);s=selector_stats(tr,tp,pick,tau);g=gate(baseline,m)
        g['selected_subset_positive']=s['selected_Q_realized_mean_gain_px'] is not None and s['selected_Q_realized_mean_gain_px']>0
        g['PASS'] &= g['selected_subset_positive']
        results.append(dict(seed=seed,metrics=m,gate=g));diagnostics.append(dict(seed=seed,**s))
        torch.save(dict(points=output,selected_Q=torch.from_numpy(pick),predicted_gain_normalized=torch.from_numpy(tp),GT_opened=False),RAW/f'seed{seed}/test_inference.pt')
        for r,gh,selected in zip(tr,tp,pick):
            csvrows.append(dict(seed=seed,frame_id=r['frame_id'],source=r['source'],session_id=r['session_id'],P_mean_px=r['p_mean'],Q_mean_px=r['q_mean'],
                gain_px=r['gain_px'],gain_normalized=r['gain_normalized'],G_hat_normalized=float(gh),G_hat_px=float(gh*r['diagonal']),
                tau_px=tau,selected_Q=bool(selected),output_mean_px=r['q_mean'] if selected else r['p_mean'],
                output_normalized=(r['q_mean'] if selected else r['p_mean'])/r['diagonal'],symmetry_choice=r['choice']))
    avg=float(np.mean([s['population_realized_mean_gain_px'] for s in diagnostics]));passed=all(r['gate']['PASS'] for r in results) and avg>0
    summary=dict(verdict='GAIN_SELECTOR_SYNTHETIC_GO' if passed else 'GAIN_SELECTOR_SYNTHETIC_FAIL',
        Point=baseline,Always_Q=metrics(tr,[True]*len(tr)),selectors=results,seed_average_realized_gain_px=avg,
        real_DEV_allowed=passed,real_DEV_executed=False,FINAL_opened=False,upstream_frozen=True)
    save(DOC/'CALIBRATION_SELECTION.json',calibrations);save(DOC/'SYNTH_PER_SEED.json',results)
    save(DOC/'SELECTOR_DIAGNOSTICS.json',diagnostics);save(DOC/'RESULT_SUMMARY.json',summary)
    path=DOC/'SYNTH_PER_FRAME.csv'
    if path.exists():raise FileExistsError(path)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows)
    print(json.dumps(dict(summary=summary,diagnostics=diagnostics),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['lock','oracle','cache','sanity','train','evaluate']);a=p.parse_args()
    globals()[a.phase]()
