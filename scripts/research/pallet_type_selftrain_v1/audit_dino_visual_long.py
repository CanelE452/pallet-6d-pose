"""Optimizer-state continuation, fixed-final inference and spatial support audit.

CUDA training is not bit-exact: repeated identical starts differ after backward.
Check state restoration exactly and ten next losses numerically, not a fictitious
bit-exact thousand-update trajectory. Stored final inference is checked exactly.
"""
import re
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import dino_visual_long as L
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=L.C;D=L.D;V=L.V;W=L.W;N=L.N;P=L.P


def main():
    p=L.verify();parent=V.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_visual_long.py','test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0,run.stderr
    ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==24
    equal=['source_records','real_records','source_held_rows','source_cache_signature','targets','head']
    for k in equal:assert p[k]==parent[k],k
    for k in ['source_samples','real_samples']:
        assert p[k][:1000]==parent[k] and len(p[k])==5000
        assert p[k]==[parent[k][i%300] for i in range(5000)]
    assert C.read(L.DOC/'CACHE_COMPLETE.json')['records']==C.read(V.DOC/'CACHE_COMPLETE.json')['records']
    with L.scope():bank=D.load_bank()
    sm={r['row']:r for r in bank if r['domain']=='source'};rm={r['id']:r for r in bank if r['domain']=='real'}
    models={};continuation={};fits={}
    for a in D.ARMS:
        fit=C.read(L.DOC/f'FIT_{a}.json');fits[a]=fit;C.verify(fit['checkpoint']);C.verify(fit['continued_from'])
        previous=C.read(V.DOC/f'FIT_{a}.json')
        assert fit['step']==5000 and fit['new_updates']==4000 and fit['history'][:1000]==previous['history']
        assert [h['step'] for h in fit['history']]==list(range(1,5001)) and all(np.isfinite(h['loss']) for h in fit['history'])
        assert [r['step'] for r in fit['source_curves']]==[2000,3000,4000,5000]
        ck=torch.load(C.ROOT/fit['continued_from']['path'],map_location='cpu',weights_only=False)
        attempts=[]
        for repeat in range(2):
            # Reload tensors: AdamW may share/mutate the loaded state dictionary.
            ck=torch.load(C.ROOT/fit['continued_from']['path'],map_location='cpu',weights_only=False)
            m=V.V.Head().cuda();m.load_state_dict(ck['model']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(ck['optimizer'])
            for k,tensor in m.state_dict().items():torch.testing.assert_close(tensor.cpu(),ck['model'][k],atol=0,rtol=0)
            state=opt.state_dict()
            for k,s in ck['optimizer']['state'].items():
                for key,value in s.items():torch.testing.assert_close(state['state'][k][key].cpu(),value.cpu(),atol=0,rtol=0)
            assert all(float(s['step'])==1000 for s in state['state'].values())
            m.train();losses=[]
            for step in range(1000,1010):
                src=[sm[i] for i in p['source_samples'][step]];real=[rm[i] for i in p['real_samples'][step]]
                losses.append(L.update(m,opt,[src] if a=='SYN' else [src,real],step))
            attempts.append(losses)
        expected=[h['loss'] for h in fit['history'][1000:1010]]
        np.testing.assert_allclose(attempts,[expected,expected],atol=1e-5,rtol=0)
        final=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
        resume=torch.load(L.RAW/a/'resume_5000.pt',map_location='cpu',weights_only=False)
        assert final['step']==5000 and final['protocol_sha256']==C.sha(L.DOC/'PROTOCOL.json')
        for k,tensor in final['model'].items():torch.testing.assert_close(tensor,resume['model'][k],atol=0,rtol=0)
        assert all(float(s['step'])==5000 for s in resume['optimizer']['state'].values())
        m.load_state_dict(final['model']);models[a]=m.eval()
        with L.scope():probe=D.probe(m,[sm[i] for i in p['source_held_rows']])
        assert probe==fit['source_after']
        continuation[a]=dict(initial_model_and_optimizer_state_restored_exactly=True,
            first10_update_losses=attempts,stored_first10_update_losses=expected,
            loss_check_atol=1e-5,max_loss_difference=float(np.abs(np.asarray(attempts)-expected).max()),
            repeat_max_loss_difference=float(np.abs(np.asarray(attempts[0])-attempts[1]).max()),
            training_bit_exact_claim=False,optimizer_steps=5000,original_prefix_preserved=True)
        print('LONG_OPTIMIZER_REPRODUCED',a,N.E.gpu(),flush=True)
    for b in C.read(L.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    for b in C.read(L.DOC/'DECISION_LOCK.json')['fits'].values():C.verify(b)
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    outputs={a:{r['id']:r for r in C.read(L.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in D.ARMS}
    scores={a:{r['id']:r for r in C.read(L.RAW/f'SCREEN_{a}.json')['metrics']} for a in D.ARMS}
    receipts={r['id']:r for r in C.read(L.RAW/'INFERENCE_RECEIPTS.json')};oldfeatures={r['id']:r['feature_sha'] for r in C.read(V.RAW/'INFERENCE_RECEIPTS.json')}
    for a in D.ARMS:assert set(outputs[a])==set(scores[a])==set(original) and len(outputs[a])==194
    backbone,_=D.A.load();assert all(not t.requires_grad for t in backbone.parameters())
    maxdelta={a:0. for a in D.ARMS}
    with torch.no_grad():
        for i,r in enumerate(C.read(L.DOC/'EVAL_PROTOCOL.json')['records']):
            C.verify(r['image']);key=r['id'];old=original[key];im=cv2.imread(str(C.ROOT/r['image']['path']));x=W.prepare_input(im,old['prediction'])
            feature=W.extract(backbone,[x])[0];assert N.array_sha(feature)==receipts[key]['feature_sha']==oldfeatures[key]
            t=torch.as_tensor(feature,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
            for a,m in models.items():
                points=V.V.decode(m(t,q,v))[0].cpu().numpy();new=N.C.transform_points(points,np.linalg.inv(x['matrix']))
                new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
                pred=outputs[a][key]['prediction'];observed=np.asarray(P.top(pred)['keypoints_xy'])
                np.testing.assert_array_equal(new,observed);P.assert_preserved(old['prediction'],pred)
                maxdelta[a]=max(maxdelta[a],float(np.abs(new-observed).max()))
            if (i+1)%50==0:print('LONG_REPRODUCE',i+1,N.E.gpu(),flush=True)
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in original}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    support={(r['id'],r['GT_corner']):r for r in C.read(W.DOC/'CROP_SUPPORT_DIAGNOSTIC.json')['rows']}
    rows=[];spatial={};cases=[];pool=[]
    for a in D.ARMS:
        ar=[]
        for key,r in outputs[a].items():
            b=base[key];n=scores[a][key];target=targets[key];gt=np.asarray(target.keypoints_xy);q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
            actual=EM.measure(q,gt,target.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(actual['canonical_errors'],float),np.asarray(n['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            assert b['matched']==n['matched'] and b['detected']==n['detected']
            if not b['matched']:continue
            oldq=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
            for j,(before,after) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])):
                if before is None:continue
                distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min())
                row=dict(id=key,arm=a,GT_corner=j,before=before,after=after,nearest_R0_point_distance=distance,
                    parent_unreachable_within10=support[(key,j)]['minimum_possible_error_px']['parent']>10,branches=[b['branch'],n['branch']])
                ar.append(row);rows.append(row)
                if before>40 and after<=10:cases.append(row)
                if a=='SYN' and distance>40:
                    qs=[oldq[valid]]+[np.asarray(P.top(outputs[arm][key]['prediction'])['keypoints_xy'])[:8] for arm in D.ARMS]
                    distances=[float(np.linalg.norm(qq-gt[j],axis=-1).min()) for qq in qs]
                    pool.append(dict(id=key,GT_corner=j,nearest_R0_px=distance,nearest_SYN_px=distances[1],nearest_MIX_px=distances[2],available_within10=min(distances)<=10))
        spatial[a]={}
        for d in [10,20,40]:
            ss=[r for r in ar if r['nearest_R0_point_distance']>d and r['before']>20]
            spatial[a][f'no_old_point_within{d}']=dict(hard=len(ss),recovered=sum(r['after']<=10 for r in ss))
    assert len(pool)==64
    C.freeze(L.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_ONLY_NOT_SELECTION',rows=rows))
    C.freeze(L.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json',dict(status='POSTHOC_GT_POINT_POOL_NOT_METHOD',rows=pool,
        far_spatial_corners=64,any_point_available_within10=sum(r['available_within10'] for r in pool),predictions_changed=False))
    result=C.read(L.DOC/'RESULTS.json')
    C.freeze(L.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        identical_parent_protocol_fields=equal,identical_cached_items=1725,optimizer_continuation=continuation,
        actual_steps_per_arm=5000,predictions_reproduced=388,max_reproduction_px=maxdelta,spatial_decomposition=spatial,
        recovered_over40_cases=cases,far_spatial_pool_available=sum(r['available_within10'] for r in pool),
        new_annotations=0,new_tags=0,auto_promoted=False,screen=result['passed'],
        evidence=[C.bound(__file__),C.bound(L.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(L.DOC/n) for n in
            ['PROTOCOL.json','SOURCE_CACHE.json'] if (L.DOC/n).exists()]+[C.bound(L.DOC/n) for n in
            ['CACHE_COMPLETE.json','FIT_SYN.json','FIT_MIX.json','OUTPUTS_LOCK.json','RESULTS.json','SPATIAL_SUPPORT_DIAGNOSTIC.json']]+[C.bound(t) for t in tests]))
    print('LONG_AUDIT_COMPLETE',dict(spatial=spatial,pool_available=sum(r['available_within10'] for r in pool),cases=cases),flush=True)


if __name__=='__main__':main()
