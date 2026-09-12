"""Student-relative normal gain, scenario-separated trust prediction, no real GT.

The frozen local line distribution is decoded without targets. GT is used only
in the separate shared assignment and supervised trust targets/evaluation.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.contracts import ROOT,RAW,DOC,sha,write
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts/research/pallet_line_pose_v1'))
from readout import prepare_readout
from model import SIDE_EDGES
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.export import legacy_metadata,metadata,symmetry,SYMMETRY_CONTRACT
from scripts.research.pallet_dht_pseudoline_selftrain_v1.geometry import baseline_assignment
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.data import ObservationDataset
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES

OUT=DOC/'D_student_relative_geometry'
OLD=ROOT/'data/pallet/results/pallet_line_pose_v1'
TAUS=[0.,.1,.25,.5,1.]

def split_for(group):
    v=int(hashlib.sha256(('trust-v1:'+group).encode()).hexdigest(),16)%10
    return 'trust_train' if v<6 else 'trust_cal' if v<8 else 'trust_test'

def selected_stats(gain,selected,groups,seed=817):
    chosen=gain[selected]
    if not len(chosen):return dict(coverage=0.,n=0,mean_gain_px=None,harm_fraction=None,lower95=None,PASS=False)
    unique=np.unique(groups)
    sums=np.array([gain[(groups==g)&selected].sum() for g in unique])
    counts=np.array([((groups==g)&selected).sum() for g in unique])
    rng=np.random.default_rng(seed);values=[]
    for _ in range(2000):
        sample=rng.integers(0,len(unique),len(unique));n=counts[sample].sum()
        if n:values.append(sums[sample].sum()/n)
    low=float(np.quantile(values,.025));coverage=float(selected.mean());mean=float(chosen.mean());harm=float((chosen<0).mean())
    return dict(coverage=coverage,n=len(chosen),mean_gain_px=mean,harm_fraction=harm,lower95=low,
        PASS=bool(coverage>=.1 and harm<.25 and low>0 and mean>=.25))

def build():
    source=json.loads((OLD/'SOURCE_MANIFEST.json').read_text())
    binding=json.loads((OLD/'validation/image_line_only_seed1_logits.json').read_text())
    for key,filename in [('logits_sha256','validation/image_line_only_seed1_logits.npy'),
                         ('validation_indices_sha256','validation/validation_indices.npy'),
                         ('source_manifest_sha256','SOURCE_MANIFEST.json')]:assert sha(OLD/filename)==binding[key]
    assert sha(binding['checkpoint'])==binding['checkpoint_sha256']
    train=[r for r in source['records'] if r['partition']=='train']
    held=[r for r in source['records'] if r['partition']=='heldout']
    train_hash={r['image_sha256'] for r in train};train_group={r['scenario_id'] for r in train}
    assert not train_hash & {r['image_sha256'] for r in held}
    assert not train_group & {r['scenario_id'] for r in held}
    members={k:[r for r in held if split_for(r['scenario_id'])==k] for k in ['trust_train','trust_cal','trust_test']}
    for a in members:
        for b in members:
            if a>=b:continue
            assert not {r['image_sha256'] for r in members[a]} & {r['image_sha256'] for r in members[b]}
            assert not {r['scenario_id'] for r in members[a]} & {r['scenario_id'] for r in members[b]}
    write(OUT/'SPLIT_AUDIT.json',dict(status='PASS',teacher_train_frames=len(train),teacher_untrained_frames=len(held),
        partitions={k:[dict(id=r['id'],image_sha256=r['image_sha256'],scenario=r['scenario_id']) for r in rows] for k,rows in members.items()},
        image_hash_overlap=0,scenario_overlap=0,source_manifest_sha256=sha(OLD/'SOURCE_MANIFEST.json')))
    cache={key:np.load(OLD/'cache'/f'{key}.npy',mmap_mode='r') for key in ['record_index','points','boxes','point_valid','gt_points','gt_valid','gain','matched','input_shape','score']}
    loc={int(v):i for i,v in enumerate(cache['record_index'])}
    logloc={int(v):i for i,v in enumerate(np.load(OLD/'validation/validation_indices.npy'))}
    logits=np.load(OLD/'validation/image_line_only_seed1_logits.npy',mmap_mode='r')
    legacy=legacy_metadata();contract=json.loads(SYMMETRY_CONTRACT.read_text());rows=[];choices=[]
    for begin in range(0,len(held),64):
        records=held[begin:begin+64];ix=[loc[r['index']] for r in records]
        tensor=lambda k:torch.from_numpy(np.array(cache[k][ix]))
        points=tensor('points');valid=tensor('point_valid').bool()
        with torch.no_grad():state=prepare_readout(torch.from_numpy(np.array(logits[[logloc[r['index']] for r in records]])),points,tensor('boxes'),valid,tensor('input_shape'))
        for j,r in enumerate(records):
            if not cache['matched'][ix[j]]:continue
            _,asset,_=metadata(r,legacy);order,basis,perms=symmetry(asset,contract)
            diagonal=float(np.hypot(*r['raw_shape_hw']));gain=float(cache['gain'][ix[j]])
            gt,yv,choice=baseline_assignment(points[j],tensor('gt_points')[j],tensor('gt_valid')[j].bool(),torch.tensor(perms),diagonal*gain,valid[j])
            choices.append(dict(id=r['id'],order=order,choice=choice))
            for e,(a,b) in enumerate(SIDE_EDGES):
                if not (state['line_valid'][j,e] and valid[j,[a,b]].all() and yv[[a,b]].all()):continue
                line=state['h_mean'][j,e].numpy();p=points[j,[a,b]].numpy();y=gt[[a,b]].numpy()
                if not np.isfinite(y).all():continue
                normal=line[:2];student=float(np.abs((p-y)@normal).mean()/gain)
                teacher=float(np.abs(y@normal+line[2]).mean()/gain)
                residual=(p@normal+line[2])/gain
                variance=float(torch.einsum('i,ij,j->',torch.cat([points[j,[a,b]].mean(0),torch.ones(1)]),state['moment'][j,e],torch.cat([points[j,[a,b]].mean(0),torch.ones(1)])))
                features=[*list(residual/diagonal),*list(np.abs(residual)/diagonal),
                    float(state['entropy_normalized'][j,e]),float(state['null_probability'][j,e]),
                    float(state['sample_coverage'][j,e].mean()),float(state['length'][j,e])/gain/diagonal,
                    float(state['anchor_std'][j])/gain/diagonal,float(cache['score'][ix[j]]),
                    np.sqrt(max(variance,0.))/gain/diagonal,*np.eye(8)[e].tolist()]
                rows.append(dict(id=r['id'],scenario=r['scenario_id'],split=split_for(r['scenario_id']),edge=e,
                    diagonal=diagonal,features=list(map(float,features)),gain_px=student-teacher,student_normal_px=student,teacher_normal_px=teacher))
    write(RAW/'D_student_relative_geometry/EDGES.json',rows)
    write(OUT/'SHARED_ASSIGNMENT.json',dict(evaluation_only=True,records=choices,C4_claim=False))
    return rows

def control(local):
    """Frozen corrected-DHT control on the existing exact overlap, no selection."""
    path=ROOT/'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction/predictions/hough_seed1_synth_val.json'
    payload=json.loads(path.read_text());pred={r['frame_id']:r for r in payload['records']}
    ds=ObservationDataset(ROOT/'data/pallet/results/pallet_symmetry_dht_local_v1/export/synth_val.json',targets=True)
    lookup={(r['id'],r['edge']):r for r in local};rows=[]
    role_map={frozenset(e):i for i,e in enumerate(EDGES)}
    local_ids={r['id'] for r in local}
    for i,meta in enumerate(ds.records):
        if meta['frame_id'] not in local_ids:continue
        item=ds[i];p=item['base_points'].double();D=float(torch.linalg.vector_norm(item['image_hw'].double()))
        y,valid,choice=baseline_assignment(p,item['target_points'].double(),item['target_valid'],item['symmetry_permutations'],D,item['point_valid'])
        lines=np.asarray(pred[meta['frame_id']]['raw_line'],float)
        for e,(a,b) in enumerate(SIDE_EDGES):
            if (meta['frame_id'],e) not in lookup or not valid[[a,b]].all():continue
            line=lines[role_map[frozenset([a,b])]];line=line/np.linalg.norm(line[:2])
            pp=p[[a,b]].numpy();yy=y[[a,b]].numpy()
            gain=float(np.abs((pp-yy)@line[:2]).mean()-np.abs(yy@line[:2]+line[2]).mean())
            rows.append(dict(id=meta['frame_id'],edge=e,dht_gain_px=gain,local_gain_px=lookup[(meta['frame_id'],e)]['gain_px'],shared_choice=choice))
    write(RAW/'D_student_relative_geometry/DHT_CONTROL_EDGES.json',rows)
    write(OUT/'DHT_CONTROL.json',dict(status='PASS',scope='descriptive negative/control teacher on pre-existing overlap; no teacher selection',
        checkpoint_sha256=payload['checkpoint_sha256'],prediction_sha256=sha(path),n_edges=len(rows),
        local_mean_gain_px=float(np.mean([r['local_gain_px'] for r in rows])) if rows else None,
        dht_mean_gain_px=float(np.mean([r['dht_gain_px'] for r in rows])) if rows else None))

def main():
    torch.set_num_threads(4)
    lock=dict(status='LOCKED_BEFORE_GAIN_MEASUREMENT',teacher='A frozen image_line_only seed1',seeds=[1,2,3],
        split='existing local teacher heldout only; SHA256 scenario modulo10 60/20/20; no target outcomes used',
        architecture='Linear(input,64),ReLU,Dropout(0.1),Linear(64,1)',optimizer='AdamW lr0.001 wd0.0001',
        steps=1500,batch=256,loss='MSE of signed G/raw_diagonal; divide target by trust_train RMS only for numeric conditioning',
        normalization='features trust_train mean/std only',thresholds_px=TAUS,
        cal_selection='maximum mean selected gain among coverage>=0.1,harm<0.25,cluster-bootstrap lower>0; no cal-valid threshold means abstain',
        test_gate='all3 positive mean and >=2 full PASS, including >=0.25px mean',
        GT_free_features='signed/abs disagreement, entropy/null, coverage, length, anchor uncertainty, baseline score, line variance, semantic role onehot',
        forbidden_feature_audit=['no GT','no gain','no frame/session/source identifiers','no symmetry choice'],
        bootstrap='2000 scenario-cluster samples; fixed seed817',student_updates=0,
        dht_control_sha256=sha(ROOT/'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction/heads/hough_seed1/checkpoint_final.pt'))
    write(OUT/'PROTOCOL_LOCK.json',lock)
    rows=build();control(rows);results=[]
    x=np.asarray([r['features'] for r in rows],np.float32);g=np.array([r['gain_px'] for r in rows]);diag=np.array([r['diagonal'] for r in rows]);groups=np.array([r['scenario'] for r in rows]);parts=np.array([r['split'] for r in rows])
    tr=parts=='trust_train';cal=parts=='trust_cal';test=parts=='trust_test'
    assert min(tr.sum(),cal.sum(),test.sum())>=64
    mu=x[tr].mean(0);sd=x[tr].std(0).clip(1e-6);scale=max(float(np.sqrt(np.mean((g[tr]/diag[tr])**2))),1e-6)
    X=torch.from_numpy((x-mu)/sd);Y=torch.from_numpy((g/diag/scale).astype(np.float32));train_indices=np.flatnonzero(tr)
    for seed in [1,2,3]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        for step in range(1500):
            ix=rng.choice(train_indices,256,replace=True);opt.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(X[ix]).squeeze(-1),Y[ix]);assert torch.isfinite(loss)
            loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
        model.eval()
        with torch.no_grad():pred=model(X).squeeze(-1).numpy()*scale*diag
        options=[]
        for tau in TAUS:
            stat=selected_stats(g[cal],pred[cal]>tau,groups[cal])
            if stat['n'] and stat['coverage']>=.1 and stat['harm_fraction']<.25 and stat['lower95']>0:options.append((stat['mean_gain_px'],tau))
        tau=max(options,key=lambda v:(v[0],-v[1]))[1] if options else None
        selected=pred[test]>tau if tau is not None else np.zeros(test.sum(),bool)
        stat=selected_stats(g[test],selected,groups[test]);result=dict(seed=seed,tau=tau,updates=1500,**stat)
        results.append(result);print(result,flush=True)
        path=RAW/f'D_student_relative_geometry/trust_seed{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        assert not path.exists();torch.save(dict(state_dict=model.state_dict(),mu=mu,sd=sd,target_scale=scale,tau=tau),path)
        write(RAW/f'D_student_relative_geometry/PREDICTIONS_seed{seed}.json',dict(predicted_gain_px=pred.tolist(),selected_test_mask=selected.tolist()))
    passed=all(r['mean_gain_px'] is not None and r['mean_gain_px']>0 for r in results) and sum(r['PASS'] for r in results)>=2
    write(OUT/'MECHANISM_RESULT.json',dict(status='PASS' if passed else 'FAIL',per_seed=results,
        eligible_edges={p:int((parts==p).sum()) for p in np.unique(parts)},
        verdict='D_MECHANISM_PASS_REQUIRES_STUDENT' if passed else 'D_COMPLEMENTARITY_NOT_PREDICTABLE',
        evidence_level='MECHANISM_ONLY',student_optimizer_updates=0))

if __name__=='__main__':main()
