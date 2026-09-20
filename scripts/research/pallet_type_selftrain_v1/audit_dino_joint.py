"""Reproduce joint choices and distinguish role edits from new locations."""
import hashlib
import re
import subprocess
import sys
from collections import Counter
import cv2
import numpy as np
import torch
from . import dino_joint as X
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=X.C;J=X.J;D=X.D;W=X.W;V=X.V;N=X.N;P=X.P


@torch.no_grad()
def main():
    p=X.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_joint_model.py','test_dino_large_gate.py','test_dino_evidence_gate_model.py','test_dino_wide_visual_model.py','test_dino_wide_model.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0,run.stderr
    ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==21
    train=set(p['source_train_rows']);cal=set(p['calibration_rows']);test=set(p['test_rows'])
    assert [len(train),len(cal),len(test)]==[1412,32,32] and not train&cal and not train&test and not cal&test
    sourcebinding=C.read(X.DOC/'SOURCE_CACHE.json')['artifact'];C.verify(sourcebinding)
    with np.load(C.ROOT/sourcebinding['path']) as z:data={k:np.array(z[k]) for k in z.files}
    assert data['features'].shape==(1476*6,12,98) and set(data['view'])==set(range(6)) and set(data['row'])==train|cal|test
    model,mean,std,t=X.load();fit=C.read(X.DOC/'FIT_JOINT.json');assert fit['steps']==1000 and len(fit['history'])==1000
    assert np.isfinite(fit['history']).all();features=torch.as_tensor(data['features'],device='cuda')
    training=np.isin(data['row'],list(train));tx=features[training,1:].reshape(-1,J.DIM)
    torch.testing.assert_close(mean,tx.mean(0),atol=0,rtol=0)
    torch.testing.assert_close(std,tx.std(0,unbiased=False).clamp_min(1e-4),atol=0,rtol=0)
    probability=model(X.E.normalized(features,mean,std)).sigmoid().cpu().numpy()
    cm=np.isin(data['row'],list(cal));best,grid=J.calibrate(probability[cm],data['errors'][cm],data['view'][cm]==0)
    assert best==fit['calibration'] and grid==fit['calibration_grid'] and t==best['threshold']
    tm=np.isin(data['row'],list(test))&(data['view']==0)
    assert J.stats(data['errors'][tm],J.choose(probability[tm],t))==fit['source_checks']['test32']
    changed=sum(N.array_sha(v.detach().cpu().numpy())!=fit['initial_state'][k] for k,v in model.state_dict().items());assert changed>0
    heads=X.E.heads()
    with W.scope():bank={r['row']:r for r in D.load_bank() if r['domain']=='source'}
    selected=sorted(train,key=lambda i:hashlib.sha256(('jointaudit:'+str(i)).encode()).hexdigest())[:4]+sorted(cal)[:2]+sorted(test)[:2]
    side=np.load(X.SIDECAR);checks=0;source_feature_max=0.
    for row in selected:
        r=bank[row];b=D.tensor_batch([r]);logits={a:m(b['feature'],b['points'],b['valid']) for a,m in heads.items()}
        diag=torch.tensor([r['bbox_diagonal']*r['matrix'][0,0]],device='cuda',dtype=torch.float32)
        perms=side['permutations'][row,:side['order'][row]]
        for view in range(6):
            q=torch.as_tensor(X.view(r,view)['points'],device='cuda')[None]
            feat,candidates=J.describe(logits['SYN'],logits['MIX'],q,diag);cand=candidates[0].cpu().numpy()
            idx=np.flatnonzero((data['row']==row)&(data['view']==view));assert len(idx)==1;idx=int(idx[0])
            delta=float(np.max(np.abs(feat[0].cpu().numpy()-data['features'][idx])));source_feature_max=max(source_feature_max,delta)
            np.testing.assert_allclose(feat[0].cpu().numpy(),data['features'][idx],atol=3e-4,rtol=1e-5)
            for h in range(12):
                actual=np.array(EM.measure(cand[h],r['target'],r['target_valid'],perms,[768,576])['canonical_errors'],float)/r['matrix'][0,0]
                np.testing.assert_allclose(actual,data['errors'][idx,h],atol=3e-4,rtol=1e-5,equal_nan=True);checks+=1
    for b in C.read(X.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    C.verify(C.read(X.DOC/'DECISION_LOCK.json')['fit'])
    original={r['id']:r for r in C.read(X.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    baseline={r['id']:r for r in C.read(X.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    output={r['id']:r for r in C.read(X.RAW/'EVAL_PREDICTIONS_JOINT.json')['records']}
    scores={r['id']:r for r in C.read(X.RAW/'SCREEN_JOINT.json')['metrics']}
    receipts={r['id']:r for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')}
    previous_features={r['id']:r['feature_sha'] for r in C.read(V.RAW/'INFERENCE_RECEIPTS.json')}
    assert set(original)==set(output)==set(scores)==set(receipts) and len(output)==194
    backbone,_=D.A.load();hist=Counter();maxdiff=0.
    for i,r in enumerate(C.read(X.DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);key=r['id'];old=original[key];im=cv2.imread(str(C.ROOT/r['image']['path']));x=W.prepare_input(im,old['prediction'])
        feature=W.extract(backbone,[x])[0];assert N.array_sha(feature)==receipts[key]['feature_sha']==previous_features[key]
        f=torch.as_tensor(feature,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
        logits={a:m(f,q,v) for a,m in heads.items()};diag=torch.tensor([x['bbox_diagonal']*x['matrix'][0,0]],device='cuda',dtype=torch.float32)
        feat,candidates=J.describe(logits['SYN'],logits['MIX'],q,diag);prob=model(X.E.normalized(feat,mean,std)).sigmoid()[0].cpu().numpy()
        choice=int(J.choose(prob,t)) if x['valid'][:8].all() else 0
        assert choice==receipts[key]['choice'];hist[J.NAMES[choice]]+=1
        np.testing.assert_array_equal(prob,receipts[key]['probabilities']);np.testing.assert_array_equal(feat[0].cpu().numpy(),receipts[key]['features'])
        restored=X.restore(x['original_points'],candidates[0,choice].cpu().numpy(),choice,x['matrix'])
        pred=output[key]['prediction'];qout=np.asarray(P.top(pred)['keypoints_xy']);np.testing.assert_array_equal(restored,qout)
        P.assert_preserved(old['prediction'],pred);maxdiff=max(maxdiff,float(np.abs(restored-qout).max()))
        if choice<4:np.testing.assert_array_equal(qout[:8],x['original_points'][J.PERMS[choice]][:8])
        if (i+1)%50==0:print('JOINT_REPRODUCE',i+1,N.E.gpu(),flush=True)
    pe,pop=X.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in original}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    assert len(perms)==2
    rows=[];cases=[]
    for key,r in output.items():
        b=baseline[key];n=scores[key];gt=np.asarray(targets[key].keypoints_xy);q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
        actual=EM.measure(q,gt,targets[key].keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
        np.testing.assert_allclose(np.array(actual['canonical_errors'],float),np.array(n['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
        assert b['matched']==n['matched'] and b['detected']==n['detected']
        if not b['matched']:continue
        oldq=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
        for j,(before,after) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])):
            if before is None:continue
            row=dict(id=key,GT_corner=j,before=before,after=after,nearest_R0_point_distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min()),
                candidate=receipts[key]['candidate'],choice=receipts[key]['choice'],branches=[b['branch'],n['branch']])
            rows.append(row)
            if before>40 and after<=10:cases.append(row)
    spatial={}
    for d in [10,20,40]:
        ss=[r for r in rows if r['nearest_R0_point_distance']>d and r['before']>20]
        spatial[f'no_old_point_within{d}']=dict(hard=len(ss),recovered=sum(r['after']<=10 for r in ss))
    C.freeze(X.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_ONLY_NOT_SELECTION',rows=rows))
    result=C.read(X.DOC/'RESULTS.json')
    C.freeze(X.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        source_split_disjoint=True,source_calibration_reproduced=True,source_GT_scoring_independently_checked=checks,source_feature_max_abs_delta=source_feature_max,
        actual_steps=1000,changed_weight_tensors=changed,all194_choices_and_predictions_reproduced=True,max_reproduction_px=maxdiff,
        all_choices_one_whole_hypothesis=True,choice_histogram=dict(hist),spatial_decomposition=spatial,recovered_over40_cases=cases,
        new_annotations=0,new_tags=0,auto_promoted=False,bounded_screen=result['passed'],
        evidence=[C.bound(__file__),C.bound(X.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(X.DOC/n) for n in
            ['PROTOCOL.json','SOURCE_CACHE.json','FIT_JOINT.json','DECISION_LOCK.json','OUTPUTS_LOCK.json','RESULTS.json']]+[C.bound(t) for t in tests]))
    print('JOINT_AUDIT_COMPLETE',dict(histogram=dict(hist),spatial=spatial,cases=cases),flush=True)


if __name__=='__main__':main()
