"""Independent forward reproduction, hint invariance and scoring of visual control."""
import re
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import dino_wide_visual as V
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

W=V.W;D=V.D;C=V.C;P=V.P;N=V.N


@torch.no_grad()
def main():
    p=V.verify();parent=W.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    keys=['steps','source_records','real_records','source_samples','real_samples','source_held_rows','source_cache_signature','optimizer','targets']
    for k in keys:assert p[k]==parent[k],k
    tests=[C.HERE/n for n in ['test_dino_wide_visual_model.py','test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    run=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True)
    print(run.stdout,flush=True);assert run.returncode==0,run.stderr
    ntests=int(re.search(r'(\d+) passed',run.stdout).group(1));assert ntests==22
    cache=C.read(V.DOC/'CACHE_COMPLETE.json');C.verify(cache['reused_from'])
    assert cache['records']==C.read(W.DOC/'CACHE_COMPLETE.json')['records']
    with V.scope():bank=D.load_bank()
    assert len(bank)==1725
    train={r['image']['sha256'] for r in p['real_records'] if r['train']}
    evaluation={r['image']['sha256'] for r in C.read(D.R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    assert len(train)==217 and not train&evaluation
    models={};changed={};source_common={}
    for a in D.ARMS:
        f=C.read(V.DOC/f'FIT_{a}.json');C.verify(f['checkpoint'])
        assert f['initial_state']==C.read(W.DOC/f'FIT_{a}.json')['initial_state']
        assert [r['step'] for r in f['history']]==list(range(1,1001))
        assert all(np.isfinite(r['loss']) and r['loss']>0 for r in f['history'])
        ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['step']==1000 and ck['protocol_sha256']==C.sha(V.DOC/'PROTOCOL.json')
        changed[a]=sum(N.array_sha(t.numpy())!=f['initial_state'][k] for k,t in ck['model'].items());assert changed[a]>0
        m=V.V.Head().cuda();m.load_state_dict(ck['model']);models[a]=m.eval()
        held=[dict(r,target_valid=r['old_target_valid']) for r in bank if r['domain']=='source' and not r['train']]
        with V.scope():source_common[a]=D.probe(m,held)
        assert source_common[a]['n']==506
    for b in C.read(V.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    for b in C.read(V.DOC/'DECISION_LOCK.json')['fits'].values():C.verify(b)
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    base={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    outputs={a:{r['id']:r for r in C.read(V.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in D.ARMS}
    scores={a:{r['id']:r for r in C.read(V.RAW/f'SCREEN_{a}.json')['metrics']} for a in D.ARMS}
    receipts={r['id']:r for r in C.read(V.RAW/'INFERENCE_RECEIPTS.json')}
    parent_receipts={r['id']:r for r in C.read(W.RAW/'INFERENCE_RECEIPTS.json')}
    for a in D.ARMS:assert set(outputs[a])==set(scores[a])==set(original) and len(outputs[a])==194
    backbone,_=D.A.load();assert all(not t.requires_grad for t in backbone.parameters())
    max_delta={a:0. for a in D.ARMS};checked=0
    for i,r in enumerate(C.read(V.DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));key=r['id'];old=original[key]
        x=W.prepare_input(im,old['prediction']);feature=W.extract(backbone,[x])[0]
        assert N.array_sha(feature)==receipts[key]['feature_sha']==parent_receipts[key]['feature_sha']
        np.testing.assert_array_equal(x['matrix'],receipts[key]['matrix'])
        feat=torch.as_tensor(feature,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];valid=torch.as_tensor(x['valid'],device='cuda')[None]
        for a,m in models.items():
            logits=m(feat,q,valid)
            torch.testing.assert_close(logits,m(feat,q+1000,~valid),atol=0,rtol=0)
            points=V.V.decode(logits)[0].cpu().numpy();new=N.C.transform_points(points,np.linalg.inv(x['matrix']))
            new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
            pred=outputs[a][key]['prediction'];delta=float(np.max(np.abs(new-np.asarray(P.top(pred)['keypoints_xy']))))
            assert delta<=.001;max_delta[a]=max(max_delta[a],delta)
            P.assert_preserved(old['prediction'],pred);checked+=1
        if (i+1)%50==0:print('VISUAL_REPRODUCE',i+1,'/194',N.E.gpu(),flush=True)
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in original}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    support={(r['id'],r['GT_corner']):r for r in C.read(W.DOC/'CROP_SUPPORT_DIAGNOSTIC.json')['rows']}
    rows=[];spatial={};cases=[]
    for a in D.ARMS:
        ar=[]
        for key,r in outputs[a].items():
            b=base[key];n=scores[a][key];t=targets[key];gt=np.asarray(t.keypoints_xy);q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
            measured=EM.measure(q,gt,t.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(measured['canonical_errors'],float),np.asarray(n['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            assert n['matched']==b['matched'] and n['detected']==b['detected']
            if not b['matched']:continue
            oldq=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
            for j,(before,after) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])):
                if before is None:continue
                distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min())
                row=dict(id=key,arm=a,GT_corner=j,before=before,after=after,nearest_R0_point_distance=distance,
                    parent_unreachable_within10=support[(key,j)]['minimum_possible_error_px']['parent']>10,branches=[b['branch'],n['branch']])
                ar.append(row);rows.append(row)
                if before>40 and after<=10:cases.append(row)
        table={}
        for d in [10,20,40]:
            ss=[r for r in ar if r['nearest_R0_point_distance']>d and r['before']>20]
            table[f'no_old_point_within{d}']=dict(hard=len(ss),recovered=sum(r['after']<=10 for r in ss))
        spatial[a]=table
    C.freeze(V.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_DIAGNOSTIC_NOT_TRAINING',rows=rows))
    result=C.read(V.DOC/'RESULTS.json')
    C.freeze(V.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        identical_parent_protocol_fields=keys,byte_identical_parent_cached_items=len(bank),source_shared506_probe=source_common,
        backbone_frozen=True,same_initial_weights_as_parent=True,actual_steps_per_arm=1000,changed_weight_tensors=changed,
        new_train_eval_exact_hash_overlap=0,real_train_unique=217,output_preservation_and_metric_checks=checked,
        learned_model_hint_invariance_checks=checked,prediction_reproduction_max_px=max_delta,spatial_decomposition=spatial,
        recovered_over40_cases=cases,new_annotations=0,new_tags=0,auto_promoted=False,screen=result['passed'],
        evidence=[C.bound(__file__),C.bound(V.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(V.DOC/f) for f in
            ['PROTOCOL.json','RESULTS.json','CACHE_COMPLETE.json','FIT_SYN.json','FIT_MIX.json','OUTPUTS_LOCK.json']]+[C.bound(t) for t in tests]))
    print('VISUAL_AUDIT_COMPLETE',dict(tests=ntests,outputs=checked,spatial=spatial,max_delta=max_delta),flush=True)


if __name__=='__main__':main()
