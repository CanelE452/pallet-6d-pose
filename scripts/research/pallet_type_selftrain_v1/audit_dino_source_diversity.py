"""Verify larger existing-source training and reproduce every real prediction."""
import re
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import dino_source_diversity as X
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=X.C;D=X.D;V=X.V;W=X.W;N=X.N;P=X.P


def main():
    p=X.verify();parent=V.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_source_diversity.py','test_dino_visual_long.py','test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0,run.stderr
    ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==28
    for k in ['real_records','source_held_rows','targets','head']:assert p[k]==parent[k],k
    assert p['real_samples']==[parent['real_samples'][i%300] for i in range(5000)]
    source=P.SourceData();train=sorted(r['row'] for r in p['source_records'] if r['partition']=='train')
    assert train==X.choose_rows(source.train_rows,p['parent_source_rows']) and len(train)==8192
    assert p['source_samples']==X.sample_schedule(train)
    assert set(np.asarray(p['source_samples']).ravel())==set(train)
    assert not set(train)&set(source.held_rows)
    assert N.source_cache_signature(source.data,np.r_[train,p['source_held_rows']])==p['source_cache_signature']
    before={r['key']:r for r in C.read(V.DOC/'CACHE_COMPLETE.json')['records']}
    receipt=C.read(X.DOC/'CACHE_COMPLETE.json');cache={r['key']:r for r in receipt['records']}
    assert len(cache)==8505 and receipt['new_items']==6780
    for key,r in before.items():assert cache[key]==r
    train_hash={r['image']['sha256'] for r in p['source_records'] if r['partition']=='train'}
    held_hash={r['image']['sha256'] for r in p['source_records'] if r['partition']=='heldout'}
    eval_hash={r['image']['sha256'] for r in C.read(X.DOC/'EVAL_PROTOCOL.json')['records']}
    assert not train_hash&held_hash and not (train_hash|held_hash)&eval_hash
    bank=X.load_bank();sm={r['row']:r for r in bank if r['domain']=='source'}
    # Reconstruct every source target from historical canvas GT and R0 boxes,
    # without trusting the new cache target arrays or selecting by error.
    a=source.data.arrays
    for i,row in enumerate(sorted(sm)):
        r=source.data.source['records'][int(source.data.indices[row])]
        gain,offset=source._canvas_affine(np.asarray(r['prepared_shape_hw']),np.asarray(a['input_shape'][row]))
        points=(a['points'][row]-offset)/gain;box=((a['boxes'][row].reshape(2,2)-offset)/gain).reshape(4)
        oldmatrix=N.C.axis_aligned_crop_matrix(box);matrix=oldmatrix.copy();matrix[:2,2]+=[144,192]
        gt=(a['gt_points'][row]-offset)/gain;mask=a['gt_valid'][row].astype(bool)&np.isfinite(gt).all(-1);mask[8]=False
        target=N.C.transform_points(np.where(mask[:,None],gt,0),matrix).astype(np.float32)
        mask&=(target>=0).all(-1)&(target[:,0]<576)&(target[:,1]<768)
        valid=a['point_valid'][row].astype(bool)&np.isfinite(points).all(-1)
        q=N.C.transform_points(np.where(valid[:,None],points,0),oldmatrix).astype(np.float32)+np.array([144,192],np.float32)
        for name,value in [('matrix',matrix),('target',target),('target_valid',mask),('points',q),('valid',valid),('original_points',points)]:
            np.testing.assert_array_equal(sm[row][name],value,err_msg=f'{row} {name}')
        if (i+1)%2000==0:print('DIVERSITY_SOURCE_TARGETS',i+1,flush=True)
    models={};fits={}
    for arm in D.ARMS:
        f=C.read(X.DOC/f'FIT_{arm}.json');fits[arm]=f;C.verify(f['checkpoint'])
        assert f['initial_state']==C.read(V.DOC/f'FIT_{arm}.json')['initial_state']
        assert [h['step'] for h in f['history']]==list(range(1,5001)) and all(np.isfinite(h['loss']) for h in f['history'])
        assert [r['step'] for r in f['source_curves']]==[1000,2000,3000,4000,5000]
        ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        resume=torch.load(X.RAW/arm/'resume_5000.pt',map_location='cpu',weights_only=False)
        assert ck['step']==5000 and ck['protocol_sha256']==C.sha(X.DOC/'PROTOCOL.json')
        for k,v in ck['model'].items():torch.testing.assert_close(v,resume['model'][k],atol=0,rtol=0)
        assert all(float(s['step'])==5000 for s in resume['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for s in resume['optimizer']['state'].values() for v in s.values())
        m=V.V.Head().cuda();m.load_state_dict(ck['model']);models[arm]=m.eval()
        with X.scope():probe=D.probe(m,[sm[i] for i in p['source_held_rows']])
        assert probe==f['source_after'];print('DIVERSITY_FIT_VERIFIED',arm,flush=True)
    X.read_feature.cache_clear()
    backbone,_=D.A.load();assert all(not t.requires_grad for t in backbone.parameters())
    # Feature audit uses predeclared deterministic train rows, not good results.
    selected=np.random.default_rng(20261005).choice(np.setdiff1d(train,p['parent_source_rows']),16,replace=False)
    for row in selected:
        feature=W.extract(backbone,[W.source_item(source,int(row))])[0]
        np.testing.assert_array_equal(feature,np.asarray(sm[int(row)]['feature']))
    for b in C.read(X.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    outputs={a:{r['id']:r for r in C.read(X.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in D.ARMS}
    scores={a:{r['id']:r for r in C.read(X.RAW/f'SCREEN_{a}.json')['metrics']} for a in D.ARMS}
    receipts={r['id']:r for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')};oldfeatures={r['id']:r['feature_sha'] for r in C.read(V.RAW/'INFERENCE_RECEIPTS.json')}
    for a in D.ARMS:assert set(outputs[a])==set(scores[a])==set(original) and len(outputs[a])==194
    with torch.no_grad():
        for i,r in enumerate(C.read(X.DOC/'EVAL_PROTOCOL.json')['records']):
            C.verify(r['image']);key=r['id'];old=original[key];im=cv2.imread(str(C.ROOT/r['image']['path']));x=W.prepare_input(im,old['prediction'])
            feature=W.extract(backbone,[x])[0];assert N.array_sha(feature)==receipts[key]['feature_sha']==oldfeatures[key]
            t=torch.as_tensor(feature,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
            for arm,m in models.items():
                points=V.V.decode(m(t,q,v))[0].cpu().numpy();new=N.C.transform_points(points,np.linalg.inv(x['matrix']))
                new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
                pred=outputs[arm][key]['prediction'];np.testing.assert_array_equal(new,np.asarray(P.top(pred)['keypoints_xy']));P.assert_preserved(old['prediction'],pred)
            if (i+1)%50==0:print('DIVERSITY_REPRODUCE',i+1,N.E.gpu(),flush=True)
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in original}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    support={(r['id'],r['GT_corner']):r for r in C.read(W.DOC/'CROP_SUPPORT_DIAGNOSTIC.json')['rows']}
    rows=[];spatial={};pool=[]
    for arm in D.ARMS:
        ar=[]
        for key,r in outputs[arm].items():
            b=base[key];n=scores[arm][key];target=targets[key];gt=np.asarray(target.keypoints_xy);q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
            actual=EM.measure(q,gt,target.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(actual['canonical_errors'],float),np.asarray(n['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            assert b['matched']==n['matched'] and b['detected']==n['detected']
            if not b['matched']:continue
            oldq=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
            for j,(before,after) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])):
                if before is None:continue
                distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min())
                row=dict(id=key,arm=arm,GT_corner=j,before=before,after=after,nearest_R0_point_distance=distance,
                    parent_unreachable_within10=support[(key,j)]['minimum_possible_error_px']['parent']>10,branches=[b['branch'],n['branch']])
                ar.append(row);rows.append(row)
                if arm=='SYN' and distance>40:
                    qs=[oldq[valid]]+[np.asarray(P.top(outputs[a][key]['prediction'])['keypoints_xy'])[:8] for a in D.ARMS]
                    distances=[float(np.linalg.norm(qq-gt[j],axis=-1).min()) for qq in qs]
                    pool.append(dict(id=key,GT_corner=j,nearest_R0_px=distance,nearest_SYN_px=distances[1],nearest_MIX_px=distances[2],available_within10=min(distances)<=10))
        spatial[arm]={}
        for d in [10,20,40]:
            ss=[r for r in ar if r['nearest_R0_point_distance']>d and r['before']>20]
            spatial[arm][f'no_old_point_within{d}']=dict(hard=len(ss),recovered=sum(r['after']<=10 for r in ss))
    assert len(pool)==64
    C.freeze(X.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_ONLY_NOT_SELECTION',rows=rows))
    C.freeze(X.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json',dict(status='POSTHOC_GT_POINT_POOL_NOT_METHOD',rows=pool,
        far_spatial_corners=64,any_point_available_within10=sum(r['available_within10'] for r in pool),predictions_changed=False))
    result=C.read(X.DOC/'RESULTS.json')
    C.freeze(X.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        source_images=8192,source_targets_reconstructed=8256,common_parent_items_byte_exact=1725,new_feature_reproductions=16,
        all_8192_source_rows_exposed=True,scenario_and_image_hash_split_disjoint=True,actual_steps_per_arm=5000,
        predictions_reproduced=388,max_reproduction_px=0,spatial_decomposition=spatial,
        far_spatial_pool_available=sum(r['available_within10'] for r in pool),new_annotations=0,new_tags=0,
        auto_promoted=False,screen=result['passed'],evidence=[C.bound(__file__),C.bound(X.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(X.DOC/n) for n in
            ['PROTOCOL.json','CACHE_COMPLETE.json','FIT_SYN.json','FIT_MIX.json','OUTPUTS_LOCK.json','RESULTS.json','SPATIAL_SUPPORT_DIAGNOSTIC.json']]+[C.bound(t) for t in tests]))
    print('DIVERSITY_AUDIT_COMPLETE',dict(spatial=spatial,pool_available=sum(r['available_within10'] for r in pool)),flush=True)


if __name__=='__main__':main()
