"""Reproduce mid-feature predictions, official layer extraction and spatial recovery."""
import re
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import dino_mid_feature as T
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=T.C;X=T.X;D=T.D;N=T.N;P=T.P;W=T.W;M=T.M


def main():
    p=T.verify();parent=X.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_mid_feature_model.py','test_dino_source_diversity.py','test_dino_wide_visual_model.py',
        'test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py','test_dino_source_difficulty_full.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True);print(run.stdout,flush=True)
    assert run.returncode==0,run.stderr;ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==34
    for k in ['source_records','source_held_rows','source_cache_signature','targets','source_samples','parent_source_rows']:assert p[k]==parent[k]
    assert p['arms']==['SYN'] and p['real_training_images']==0
    train={r['row'] for r in p['source_records'] if r['partition']=='train'};held=set(p['source_held_rows'])
    assert len(train)==8192 and len(held)==64 and not train&held and set(np.asarray(p['source_samples']).ravel())==train
    C.verify(C.read(T.DOC/'EXTRACTOR_PREFLIGHT.json')['source'])
    difficulty=C.read(T.DOC/'SOURCE_DIFFICULTY_AUDIT.json')
    full_difficulty=C.read(T.DOC/'FULL_SOURCE_DIFFICULTY_AUDIT.json')
    for b in difficulty['evidence']+full_difficulty['evidence']:C.verify(b)
    assert difficulty['summary']['train']['far40']==77 and difficulty['summary']['held']['far40']==0
    cache=C.read(T.DOC/'CACHE_COMPLETE.json')
    chunk_rows=[]
    for b in cache['chunks']:
        C.verify(b);part=C.read(C.ROOT/b['path']);C.verify(part['protocol']);chunk_rows.extend(part['records'])
    assert chunk_rows==cache['records']
    old={r['key']:r for r in C.read(X.DOC/'CACHE_COMPLETE.json')['records']}
    assert len(cache['records'])==8256 and all(r['domain']=='source' for r in cache['records'])
    for r in cache['records']:
        C.verify(r['cache']);C.verify(r['mid_cache']);assert {k:v for k,v in r.items() if k!='mid_cache'}==old[r['key']]
    source=P.SourceData();bank=T.load_bank();sm={r['row']:r for r in bank};backbone,_=D.A.load()
    assert N.source_cache_signature(source.data,np.r_[sorted(train),p['source_held_rows']])==p['source_cache_signature']
    checkrows=sorted(train)[::1024]+p['source_held_rows'][::8]
    for row in checkrows:
        x=W.source_item(source,row)
        for k,v in D.item_arrays(x).items():np.testing.assert_array_equal(sm[row][k],v)
        with torch.no_grad():
            reference=backbone.get_intermediate_layers(T.image_tensor([x]),n=[5,11],norm=True)
            mid,late=T.extract(backbone,[x],True)
        np.testing.assert_array_equal(np.asarray(sm[row]['mid_feature']),mid[0])
        np.testing.assert_array_equal(mid,T.array(reference[0]));np.testing.assert_array_equal(late,T.array(reference[1]))
        np.testing.assert_array_equal(late[0],np.asarray(sm[row]['feature']))
    fit=C.read(T.DOC/'FIT_SYN.json');C.verify(fit['checkpoint']);assert fit['step']==5000 and fit['real_training_images']==0
    assert [h['step'] for h in fit['history']]==list(range(1,5001)) and all(np.isfinite(h['loss']) for h in fit['history'])
    assert [r['step'] for r in fit['source_curves']]==[1000,2000,3000,4000,5000]
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929);m=M.Head().cuda()
    initial={k:N.array_sha(v.detach().cpu().numpy()) for k,v in m.state_dict().items()};assert initial==fit['initial_state']
    for k,v in C.read(X.DOC/'FIT_SYN.json')['initial_state'].items():assert initial[k]==v
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    resume=torch.load(T.RAW/'SYN/resume_5000.pt',map_location='cpu',weights_only=False)
    assert ck['step']==5000 and ck['protocol_sha256']==C.sha(T.DOC/'PROTOCOL.json')
    for k,v in ck['model'].items():torch.testing.assert_close(v,resume['model'][k],atol=0,rtol=0)
    assert all(float(s['step'])==5000 for s in resume['optimizer']['state'].values())
    m.load_state_dict(ck['model']);m.eval()
    with T.scope():assert D.probe(m,[sm[i] for i in p['source_held_rows']])==fit['source_after']
    assert ck['model']['mid.3.weight'].abs().sum()>0
    # Source-only explanatory ablation, never a checkpoint/threshold selector.
    with torch.no_grad():m.mid[-1].weight.zero_();m.mid[-1].bias.zero_()
    with T.scope():mid_disabled=D.probe(m,[sm[i] for i in p['source_held_rows']])
    m.load_state_dict(ck['model'])
    for k,v in m.state_dict().items():torch.testing.assert_close(v.cpu(),ck['model'][k],atol=0,rtol=0)
    C.freeze(T.DOC/'SOURCE_MID_ABLATION.json',dict(status='POSTHOC_SOURCE_ONLY_NO_SELECTION',
        full=fit['source_after'],mid_branch_disabled=mid_disabled,trained_weights_restored_exactly=True,
        note='Disabling a co-trained branch is not a separately trained architecture ablation. No real GT used.'))
    for b in C.read(T.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    C.verify(C.read(T.DOC/'DECISION_LOCK.json')['fits']['SYN'])
    outputs={r['id']:r for r in C.read(T.RAW/'EVAL_PREDICTIONS_SYN.json')['records']}
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    scores={r['id']:r for r in C.read(T.RAW/'SCREEN_SYN.json')['metrics']}
    receipts={r['id']:r for r in C.read(T.RAW/'INFERENCE_RECEIPTS.json')};prior={r['id']:r for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')}
    assert set(outputs)==set(original)==set(scores) and len(outputs)==194
    assert all(not v.requires_grad for v in backbone.parameters())
    with torch.no_grad():
        for i,r in enumerate(C.read(T.DOC/'EVAL_PROTOCOL.json')['records']):
            C.verify(r['image']);key=r['id'];old=original[key];im=cv2.imread(str(C.ROOT/r['image']['path']));x=W.prepare_input(im,old['prediction'])
            mid,f=T.extract(backbone,[x],True);mid=mid[0];f=f[0]
            assert N.array_sha(f)==receipts[key]['feature_sha']==prior[key]['feature_sha'] and N.array_sha(mid)==receipts[key]['mid_sha']
            t=(torch.as_tensor(f,device='cuda').float()[None],torch.as_tensor(mid,device='cuda').float()[None])
            q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
            z=m(t,q,v);torch.testing.assert_close(z,m(t,q+1000,~v),atol=0,rtol=0)
            out=M.decode(z)[0].cpu().numpy();new=N.C.transform_points(out,np.linalg.inv(x['matrix']))
            new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
            pred=outputs[key]['prediction'];np.testing.assert_array_equal(new,np.asarray(P.top(pred)['keypoints_xy']));P.assert_preserved(old['prediction'],pred)
            if (i+1)%50==0:print('MID_FEATURE_REPRODUCE',i+1,N.E.gpu(),flush=True)
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
            dist=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min())
            rows.append(dict(id=key,arm='SYN',GT_corner=j,before=before,after=after,nearest_R0_point_distance=dist,branches=[b['branch'],s['branch']]))
            if dist>40:
                pv=np.isfinite(q[:8]).all(-1)&~(q[:8]==-1).all(-1)
                pool.append(dict(id=key,GT_corner=j,nearest_R0_px=dist,nearest_new_point_px=float(np.linalg.norm(q[:8][pv]-gt[j],axis=-1).min())))
    assert len(pool)==64
    spatial={}
    for dist in [10,20,40]:
        rr=[r for r in rows if r['nearest_R0_point_distance']>dist and r['before']>20]
        spatial[f'no_old_point_within{dist}']=dict(hard=len(rr),recovered=sum(r['after']<=10 for r in rr))
    C.freeze(T.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_ONLY_NOT_SELECTION',rows=rows))
    C.freeze(T.DOC/'SPATIAL_SUPPORT_DIAGNOSTIC.json',dict(status='POSTHOC_POINT_POOL_NOT_METHOD',rows=pool,
        far_spatial_corners=64,any_new_point_within10=sum(r['nearest_new_point_px']<=10 for r in pool),predictions_changed=False))
    C.freeze(T.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        source_training_images=8192,real_training_images=0,actual_steps=5000,unchanged_parent_feature_items=8256,
        mid_and_official_late_items_reproduced=len(checkrows),parent_parameter_initialization_exact=True,
        source_train_far40_corners=77,source_held64_far40_corners=0,
        predictions_reproduced=194,max_reproduction_px=0,learned_hint_invariance_verified=194,
        spatial_decomposition=spatial,far_spatial_candidate_support=sum(r['nearest_new_point_px']<=10 for r in pool),
        new_annotations=0,new_tags=0,auto_promoted=False,screen=C.read(T.DOC/'RESULTS.json')['passed'],
        evidence=[C.bound(__file__),C.bound(T.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(t) for t in tests]+[C.bound(T.DOC/f) for f in
        ['PROTOCOL.json','CACHE_COMPLETE.json','FIT_SYN.json','OUTPUTS_LOCK.json','RESULTS.json','SPATIAL_SUPPORT_DIAGNOSTIC.json','SOURCE_MID_ABLATION.json','EXTRACTOR_PREFLIGHT.json','SOURCE_DIFFICULTY_AUDIT.json','FULL_SOURCE_DIFFICULTY_AUDIT.json']]))
    print('MID_FEATURE_AUDIT_COMPLETE',spatial,flush=True)


if __name__=='__main__':main()
