"""Audit optimizer continuation, immutable labels and final spatial recovery."""
import re
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import dino_diverse_continue as L
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=L.C;X=L.X;D=L.D;N=L.N;P=L.P;V=L.V;W=L.W


def main():
    p=L.verify();parent=X.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_diverse_continue.py','test_dino_source_diversity.py','test_dino_visual_long.py','test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True);print(run.stdout,flush=True)
    assert run.returncode==0,run.stderr;ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==30
    for k in ['source_records','source_held_rows','source_cache_signature','targets','head','parent_source_rows']:assert p[k]==parent[k]
    assert p['arms']==['SYN'] and p['real_training_images']==0
    train=sorted(r['row'] for r in p['source_records'] if r['partition']=='train')
    assert p['source_samples']==X.sample_schedule(train,20000) and p['source_samples'][:5000]==parent['source_samples']
    assert not set(np.asarray(p['source_samples']).ravel())&set(p['source_held_rows'])
    assert C.read(L.DOC/'CACHE_COMPLETE.json')['records']==C.read(X.DOC/'CACHE_COMPLETE.json')['records']
    parent_audit=C.read(X.DOC/'COMPLETION_AUDIT.json')
    for b in parent_audit['evidence']:C.verify(b)
    bank=X.load_bank();sm={r['row']:r for r in bank if r['domain']=='source'}
    fit=C.read(L.DOC/'FIT_SYN.json');oldfit=C.read(X.DOC/'FIT_SYN.json');C.verify(fit['checkpoint']);C.verify(fit['continued_from'])
    assert fit['step']==20000 and fit['new_updates']==15000 and fit['history'][:5000]==oldfit['history']
    assert [h['step'] for h in fit['history']]==list(range(1,20001)) and all(np.isfinite(h['loss']) for h in fit['history'])
    assert fit['source_curves'][:5]==oldfit['source_curves']
    assert [r['step'] for r in fit['source_curves'][5:]]==[7500,10000,12500,15000,17500,20000]
    attempts=[]
    for repeat in range(2):
        ck=torch.load(C.ROOT/fit['continued_from']['path'],map_location='cpu',weights_only=False)
        m=V.V.Head().cuda();m.load_state_dict(ck['model']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(ck['optimizer'])
        for k,v in m.state_dict().items():torch.testing.assert_close(v.cpu(),ck['model'][k],atol=0,rtol=0)
        state=opt.state_dict()
        for k,s in ck['optimizer']['state'].items():
            for name,v in s.items():torch.testing.assert_close(state['state'][k][name].cpu(),v.cpu(),atol=0,rtol=0)
        assert all(float(s['step'])==5000 for s in state['state'].values())
        m.train();losses=[]
        for step in range(5000,5010):losses.append(L.L.update(m,opt,[[sm[i] for i in p['source_samples'][step]]],step))
        attempts.append(losses)
    expected=[r['loss'] for r in fit['history'][5000:5010]]
    np.testing.assert_allclose(attempts,[expected,expected],atol=1e-5,rtol=0)
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    resume=torch.load(L.RAW/'SYN/resume_20000.pt',map_location='cpu',weights_only=False)
    assert ck['step']==20000 and ck['protocol_sha256']==C.sha(L.DOC/'PROTOCOL.json')
    for k,v in ck['model'].items():torch.testing.assert_close(v,resume['model'][k],atol=0,rtol=0)
    assert all(float(s['step'])==20000 for s in resume['optimizer']['state'].values())
    m.load_state_dict(ck['model']);m.eval()
    with L.scope():probe=D.probe(m,[sm[i] for i in p['source_held_rows']])
    assert probe==fit['source_after']
    for b in C.read(L.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    C.verify(C.read(L.DOC/'DECISION_LOCK.json')['fits']['SYN'])
    outputs={r['id']:r for r in C.read(L.RAW/'EVAL_PREDICTIONS_SYN.json')['records']}
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    scores={r['id']:r for r in C.read(L.RAW/'SCREEN_SYN.json')['metrics']}
    receipts={r['id']:r for r in C.read(L.RAW/'INFERENCE_RECEIPTS.json')};oldfeatures={r['id']:r['feature_sha'] for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')}
    assert set(outputs)==set(original)==set(scores) and len(outputs)==194
    backbone,_=D.A.load();assert all(not t.requires_grad for t in backbone.parameters())
    with torch.no_grad():
        for i,r in enumerate(C.read(L.DOC/'EVAL_PROTOCOL.json')['records']):
            C.verify(r['image']);key=r['id'];old=original[key];im=cv2.imread(str(C.ROOT/r['image']['path']));x=W.prepare_input(im,old['prediction'])
            feat=W.extract(backbone,[x])[0];assert N.array_sha(feat)==receipts[key]['feature_sha']==oldfeatures[key]
            t=torch.as_tensor(feat,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
            z=m(t,q,v);torch.testing.assert_close(z,m(t,q+1000,~v),atol=0,rtol=0)
            predicted=V.V.decode(z)[0].cpu().numpy();new=N.C.transform_points(predicted,np.linalg.inv(x['matrix']))
            new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
            pred=outputs[key]['prediction'];np.testing.assert_array_equal(new,np.asarray(P.top(pred)['keypoints_xy']));P.assert_preserved(old['prediction'],pred)
            if (i+1)%50==0:print('DIVERSE_CONTINUE_REPRODUCE',i+1,N.E.gpu(),flush=True)
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in outputs}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    rows=[];pool=[]
    for key,r in outputs.items():
        b=base[key];s=scores[key];target=targets[key];gt=np.asarray(target.keypoints_xy);q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
        actual=EM.measure(q,gt,target.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
        np.testing.assert_allclose(np.asarray(actual['canonical_errors'],float),np.asarray(s['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
        assert b['matched']==s['matched'] and b['detected']==s['detected']
        if not b['matched']:continue
        oldq=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
        for j,(before,after) in enumerate(zip(b['canonical_errors'],s['canonical_errors'])):
            if before is None:continue
            distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min())
            rows.append(dict(id=key,arm='SYN',GT_corner=j,before=before,after=after,nearest_R0_point_distance=distance,branches=[b['branch'],s['branch']]))
            if distance>40:
                pv=np.isfinite(q[:8]).all(-1)&~(q[:8]==-1).all(-1)
                pool.append(dict(id=key,GT_corner=j,nearest_R0_px=distance,nearest_new_point_px=float(np.linalg.norm(q[:8][pv]-gt[j],axis=-1).min())))
    assert len(pool)==64
    spatial={}
    for dist in [10,20,40]:
        rr=[r for r in rows if r['nearest_R0_point_distance']>dist and r['before']>20]
        spatial[f'no_old_point_within{dist}']=dict(hard=len(rr),recovered=sum(r['after']<=10 for r in rr))
    C.freeze(L.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_ONLY_NOT_SELECTION',rows=rows))
    C.freeze(L.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json',dict(status='POSTHOC_POINT_POOL_NOT_METHOD',rows=pool,
        far_spatial_corners=64,any_new_point_within10=sum(r['nearest_new_point_px']<=10 for r in pool),predictions_changed=False))
    result=C.read(L.DOC/'RESULTS.json')
    C.freeze(L.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        unchanged_cached_items=8505,source_training_images=8192,real_training_images=0,actual_steps=20000,
        model_and_optimizer_initial_state_exact=True,history5000prefix_exact=True,
        next10_update_loss_atol=1e-5,next10_update_max_difference=float(np.abs(np.asarray(attempts)-expected).max()),
        training_trajectory_bit_exact_claim=False,predictions_reproduced=194,max_reproduction_px=0,
        learned_hint_invariance_verified=194,spatial_decomposition=spatial,
        far_spatial_candidate_support=sum(r['nearest_new_point_px']<=10 for r in pool),new_annotations=0,new_tags=0,auto_promoted=False,
        screen=result['passed'],evidence=[C.bound(__file__),C.bound(L.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(t) for t in tests]+[C.bound(L.DOC/f) for f in
            ['PROTOCOL.json','CACHE_COMPLETE.json','FIT_SYN.json','OUTPUTS_LOCK.json','RESULTS.json','SPATIAL_SUPPORT_DIAGNOSTIC.json']]))
    print('DIVERSE_CONTINUE_AUDIT_COMPLETE',dict(spatial=spatial,candidate_support=sum(r['nearest_new_point_px']<=10 for r in pool)),flush=True)


if __name__=='__main__':main()
