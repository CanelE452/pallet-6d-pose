"""Audit controlled sampling, continued Adam state and genuine spatial recovery."""
import re
import subprocess
import sys
from collections import Counter
import cv2
import numpy as np
import torch
from . import dino_tail_sampling as T
from . import dino_joint_model as J
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=T.C;U=T.U;D=T.D;N=T.N;P=T.P;W=T.W;M=T.M


def main():
    p=T.verify();parent=U.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_tail_sampling.py','test_dino_mid_feature_model.py','test_dino_source_diversity.py',
        'test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py',
        'test_recovery_pseudo_denoise.py','test_dino_source_difficulty_full.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True);print(run.stdout,flush=True)
    assert run.returncode==0,run.stderr;ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==37
    for b in C.read(U.DOC/'COMPLETION_AUDIT.json')['evidence']:C.verify(b)
    train=sorted(r['row'] for r in p['source_records'] if r['partition']=='train');held=p['source_held_rows'];hard=set(p['hard_train_rows'])
    assert len(train)==8357 and len(held)==77 and len(hard)==196 and not set(train)&set(held)
    assert p['source_samples_by_arm']==T.schedule(train,hard)
    assert p['head']==parent['head'] and p['targets']==parent['targets'] and p['real_training_images']==0
    assert p['preservation_held_rows']==parent['source_held_rows'] and len(p['far_held_rows'])==13
    assert not set(p['preservation_held_rows'])&set(p['far_held_rows'])
    assert not {r['image']['sha256'] for r in p['source_records'] if r['partition']=='train'}&{r['image']['sha256'] for r in p['source_records'] if r['partition']=='heldout'}
    strata={}
    for arm in T.ARMS:
        ss=np.asarray(p['source_samples_by_arm'][arm]);assert ss.shape==(5000,8) and set(ss.ravel())==set(train)
        hc=int(np.isin(ss,list(hard)).sum());strata[arm]=dict(total=40000,hard=hc,other=40000-hc)
        if arm=='TAIL':assert np.isin(ss[:,:4],list(hard)).all() and not np.isin(ss[:,4:],list(hard)).any() and hc==20000
    cache=C.read(T.DOC/'CACHE_COMPLETE.json');old={r['row']:r for r in C.read(U.DOC/'CACHE_COMPLETE.json')['records']}
    assert len(cache['records'])==8434
    new=[]
    for r in cache['records']:
        C.verify(r['cache']);C.verify(r['mid_cache'])
        if r['row'] in old:assert r==old[r['row']]
        else:new.append(r)
    assert len(new)==178
    source=P.SourceData();assert N.source_cache_signature(source.data,np.r_[train,held])==p['source_cache_signature']
    bank=T.load_bank();sm={r['row']:r for r in bank};backbone,_=D.A.load()
    for r in new:
        x=W.source_item(source,r['row'])
        for k,v in D.item_arrays(x).items():np.testing.assert_array_equal(sm[r['row']][k],v)
    for r in new[:8]+new[-8:]:
        x=W.source_item(source,r['row'])
        with torch.no_grad():ref=backbone.get_intermediate_layers(U.image_tensor([x]),n=[5,11],norm=True)
        np.testing.assert_array_equal(U.array(ref[0])[0],np.asarray(sm[r['row']]['mid_feature']))
        np.testing.assert_array_equal(U.array(ref[1])[0],np.asarray(sm[r['row']]['feature']))
    models={};fits={};repeat_errors={}
    for arm in T.ARMS:
        fit=C.read(T.DOC/f'FIT_{arm}.json');C.verify(fit['checkpoint']);fits[arm]=fit
        assert fit['step']==10000 and fit['new_updates']==5000 and fit['initialization']==p['initialization']
        assert [r['step'] for r in fit['history']]==list(range(5001,10001))
        assert [r['step'] for r in fit['source_curves']]==[6000,7000,8000,9000,10000]
        ck=torch.load(C.ROOT/p['initialization']['path'],map_location='cpu',weights_only=False)
        m=M.Head().cuda();m.load_state_dict(ck['model']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(ck['optimizer'])
        assert {k:N.array_sha(v.detach().cpu().numpy()) for k,v in m.state_dict().items()}==fit['initial_state']
        for key,state in ck['optimizer']['state'].items():
            for name,v in state.items():torch.testing.assert_close(opt.state_dict()['state'][key][name].cpu(),v.cpu(),atol=0,rtol=0)
        assert all(float(s['step'])==5000 for s in opt.state_dict()['state'].values())
        if arm==T.ARMS[0]:
            m.eval()
            C.freeze(T.DOC/'PARENT_SOURCE_REFERENCE.json',dict(status='SOURCE_ONLY_REFERENCE_NOT_SELECTION',
                initialization=p['initialization'],far_held13=T.spatial_probe(m,[sm[i] for i in p['far_held_rows']]),
                hard_train196=T.spatial_probe(m,[sm[i] for i in p['hard_train_rows']]),new_predictions_applied=False))
        losses=[];m.train()
        for offset in range(5):
            rows=[sm[i] for i in p['source_samples_by_arm'][arm][offset]]
            with T.scope():losses.append(T.X.L.update(m,opt,[rows],5000+offset))
        expected=[r['loss'] for r in fit['history'][:5]];np.testing.assert_allclose(losses,expected,atol=1e-5,rtol=0)
        repeat_errors[arm]=float(np.max(np.abs(np.asarray(losses)-expected)))
        final=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
        resume=torch.load(T.RAW/arm/'resume_10000.pt',map_location='cpu',weights_only=False)
        assert final['step']==10000 and final['protocol_sha256']==C.sha(T.DOC/'PROTOCOL.json')
        for k,v in final['model'].items():torch.testing.assert_close(v,resume['model'][k],atol=0,rtol=0)
        assert all(float(s['step'])==10000 for s in resume['optimizer']['state'].values())
        m.load_state_dict(final['model']);m.eval();models[arm]=m
        with T.scope():assert D.probe(m,[sm[i] for i in p['preservation_held_rows']])==fit['source_after']
        far=T.spatial_probe(m,[sm[i] for i in p['far_held_rows']]);assert far==fit['far_held_after'] and far['far']==36
    assert fits['UNIFORM']['initial_state']==fits['TAIL']['initial_state']
    # Distinguish an unseen-position failure from an official symmetry mismatch.
    sidepath=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz'
    side=np.load(sidepath);np.testing.assert_array_equal(side['record_index'],source.data.indices)
    role={a:{k:[] for k in ['fixed','official','nearest_oracle','far']} for a in T.ARMS}
    with torch.no_grad():
        for row in p['far_held_rows']:
            r=sm[row];b=U.tensor_batch([r]);mask=r['target_valid'][:8];gt=r['target'][:8];gain=r['matrix'][0,0]
            old=r['points'][:8][r['valid'][:8]];far=(np.linalg.norm(gt[:,None]-old[None],axis=-1).min(-1)/gain>40)[mask]
            perms=side['permutations'][row,:int(side['order'][row])]
            for arm,m in models.items():
                q=M.decode(m(b['feature'],b['points'],b['valid']))[0].cpu().numpy()
                fixed=np.linalg.norm(q[:8]-gt,axis=-1)[mask]/gain
                official=J.canonical_errors(q[None],r['target'],r['target_valid'],perms,gain)[0][mask]
                nearest=np.linalg.norm(gt[:,None]-q[None,:8],axis=-1).min(-1)[mask]/gain
                for key,values in [('fixed',fixed),('official',official),('nearest_oracle',nearest),('far',far)]:role[arm][key].extend(values.tolist())
    role_summary={}
    for arm,rr in role.items():
        f=np.asarray(rr['far'],bool);assert f.sum()==36
        role_summary[arm]={k:dict(n=len(v),PCK10=float((np.asarray(v)<=10).mean()),far_recovered=int(((np.asarray(v)<=10)&f).sum())) for k,v in rr.items() if k!='far'}
    C.freeze(T.DOC/'SOURCE_FAR_ROLE_DIAGNOSTIC.json',dict(status='SOURCE_GT_DIAGNOSTIC_NOT_INFERENCE',metrics=role_summary,
        official_group_order_counts=dict(Counter(str(int(side['order'][i])) for i in p['far_held_rows'])),
        predictions_changed=False,thresholds_changed=False,warning='Nearest oracle can reuse a point and ignores roles;not method performance.',
        evidence=[C.bound(J.__file__),C.bound(sidepath),C.bound(T.DOC/'PROTOCOL.json')]))
    for b in C.read(T.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    for b in C.read(T.DOC/'DECISION_LOCK.json')['fits'].values():C.verify(b)
    outputs={a:{r['id']:r for r in C.read(T.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in T.ARMS}
    scores={a:{r['id']:r for r in C.read(T.RAW/f'SCREEN_{a}.json')['metrics']} for a in T.ARMS}
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    receipts={r['id']:r for r in C.read(T.RAW/'INFERENCE_RECEIPTS.json')};prior={r['id']:r for r in C.read(U.RAW/'INFERENCE_RECEIPTS.json')}
    for a in T.ARMS:assert set(outputs[a])==set(scores[a])==set(original) and len(outputs[a])==194
    assert all(not p.requires_grad for p in backbone.parameters())
    with torch.no_grad():
        for i,r in enumerate(C.read(T.DOC/'EVAL_PROTOCOL.json')['records']):
            C.verify(r['image']);key=r['id'];old=original[key];x=W.prepare_input(cv2.imread(str(C.ROOT/r['image']['path'])),old['prediction'])
            mid,late=U.extract(backbone,[x],True);mid=mid[0];late=late[0]
            assert N.array_sha(mid)==receipts[key]['mid_sha']==prior[key]['mid_sha'] and N.array_sha(late)==receipts[key]['feature_sha']==prior[key]['feature_sha']
            t=tuple(torch.as_tensor(f,device='cuda').float()[None] for f in [late,mid]);q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
            for a,m in models.items():
                z=m(t,q,v);torch.testing.assert_close(z,m(t,q+1000,~v),atol=0,rtol=0)
                points=M.decode(z)[0].cpu().numpy();newq=N.C.transform_points(points,np.linalg.inv(x['matrix']));newq[~x['valid']]=x['original_points'][~x['valid']];newq[8]=x['original_points'][8]
                pred=outputs[a][key]['prediction'];np.testing.assert_array_equal(newq,np.asarray(P.top(pred)['keypoints_xy']));P.assert_preserved(old['prediction'],pred)
            if (i+1)%50==0:print('TAIL_REPRODUCE',i+1,'/194 x2',N.E.gpu(),flush=True)
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in original}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    corners=[];spatial={};support={};pool=[]
    for a in T.ARMS:
        rows=[];pp=[]
        for key,r in outputs[a].items():
            b=base[key];s=scores[a][key];target=targets[key];gt=np.asarray(target.keypoints_xy);q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
            actual=EM.measure(q,gt,target.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(actual['canonical_errors'],float),np.asarray(s['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            assert b['matched']==s['matched'] and b['detected']==s['detected']
            if not b['matched']:continue
            oldq=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
            for j,(before,after) in enumerate(zip(b['canonical_errors'],s['canonical_errors'])):
                if before is None:continue
                dist=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min())
                rows.append(dict(id=key,arm=a,GT_corner=j,before=before,after=after,nearest_R0_point_distance=dist,branches=[b['branch'],s['branch']]))
                if dist>40:
                    pv=np.isfinite(q[:8]).all(-1)&~(q[:8]==-1).all(-1)
                    pp.append(dict(id=key,arm=a,GT_corner=j,nearest_R0_px=dist,nearest_new_point_px=float(np.linalg.norm(q[:8][pv]-gt[j],axis=-1).min())))
        assert len(pp)==64;spatial[a]={};support[a]=sum(r['nearest_new_point_px']<=10 for r in pp)
        for d in [10,20,40]:
            rr=[r for r in rows if r['before']>20 and r['nearest_R0_point_distance']>d]
            spatial[a][f'no_old_point_within{d}']=dict(hard=len(rr),recovered=sum(r['after']<=10 for r in rr))
        corners.extend(rows);pool.extend(pp)
    C.freeze(T.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_ONLY_NOT_SELECTION',rows=corners))
    C.freeze(T.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json',dict(status='POSTHOC_POINT_POOL_NOT_METHOD',rows=pool,counts=support,predictions_changed=False))
    C.freeze(T.DOC/'SAMPLING_EXPOSURE_AUDIT.json',dict(strata=strata,source_train=8357,source_held=77,hard_train=196,far_held=13,no_held_training=True,schedules_reproduced=True))
    C.freeze(T.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        source_training_images=8357,source_held_images=77,far_held_corners=36,real_training_images=0,reused_cache_items=8256,new_cache_items=178,
        new_targets_reconstructed=178,official_features_reproduced=16,initial_model_and_optimizer_exact=True,
        first5_losses_atol=1e-5,first5_max_error=repeat_errors,full_training_bit_exact_claim=False,
        actual_new_updates_each=5000,final_step_each=10000,predictions_reproduced=388,max_reproduction_px=0,
        spatial_decomposition=spatial,far_spatial_candidate_support=support,new_annotations=0,new_tags=0,auto_promoted=False,
        screen=C.read(T.DOC/'RESULTS.json')['passed'],evidence=[C.bound(__file__),C.bound(T.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(f) for f in tests]+[C.bound(T.DOC/f) for f in
            ['PROTOCOL.json','CACHE_COMPLETE.json','FIT_UNIFORM.json','FIT_TAIL.json','OUTPUTS_LOCK.json','RESULTS.json','SPATIAL_SUPPORT_DIAGNOSTIC.json','SAMPLING_EXPOSURE_AUDIT.json','PARENT_SOURCE_REFERENCE.json','SOURCE_FAR_ROLE_DIAGNOSTIC.json']]))
    print('TAIL_AUDIT_COMPLETE',spatial,flush=True)


if __name__=='__main__':main()
