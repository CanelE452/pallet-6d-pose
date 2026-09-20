"""Verify physical-scale preservation, wide outputs, and the previously blocked tail."""
import hashlib
import re
import subprocess
import sys
import cv2
import numpy as np
import torch
from . import dino_wide as W
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

D=W.D;C=W.C;P=W.P;N=W.N


@torch.no_grad()
def main():
    p=W.verify();parent=D.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    equal=['steps','source_records','real_records','source_samples','real_samples','source_held_rows','source_cache_signature','optimizer']
    for key in equal:assert p[key]==parent[key],key
    tests=[C.HERE/n for n in ['test_dino_wide_model.py','test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    test=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True)
    print(test.stdout,flush=True);assert test.returncode==0,test.stderr
    ntests=int(re.search(r'(\d+) passed',test.stdout).group(1));assert ntests==18
    with W.scope():bank=D.load_bank()
    pool={r['id']:r for r in C.read(D.R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC'}
    source=P.SourceData();old_cache={r['id']:r for r in C.read(D.DOC/'CACHE_COMPLETE.json')['records']}
    counts={d:dict(old=0,new=0) for d in ['source_train','source_heldout','real_train','real_heldout']}
    for r in bank:
        x=W.source_item(source,r['row']) if r['domain']=='source' else W.real_item(pool[r['id']])
        for key,v in D.item_arrays(x).items():np.testing.assert_array_equal(r[key],v)
        with np.load(C.ROOT/old_cache[r['id']]['cache']['path']) as old:
            np.testing.assert_array_equal(r['matrix'][:2,:2],old['matrix'][:2,:2])
            np.testing.assert_allclose(r['matrix'][:2,2]-old['matrix'][:2,2],[144,192],atol=1e-10)
            np.testing.assert_array_equal(r['original_points'],old['original_points'])
            np.testing.assert_array_equal(r['valid'],old['valid'])
            np.testing.assert_array_equal(r['old_target_valid'],old['target_valid'])
            mask=old['target_valid'];assert not (mask&~r['target_valid']).any()
            np.testing.assert_allclose(r['target'][mask]-[144,192],old['target'][mask],atol=1e-4,rtol=0)
        group=r['domain']+('_train' if r['train'] else '_heldout')
        counts[group]['old']+=int(r['old_target_valid'].sum());counts[group]['new']+=int(r['target_valid'].sum())
    assert counts==C.read(W.DOC/'CACHE_COMPLETE.json')['supervision_counts']
    train={r['image']['sha256'] for r in p['real_records'] if r['train']}
    evaluation={r['image']['sha256'] for r in C.read(D.R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    assert len(train)==217 and not train&evaluation
    backbone,_=D.A.load();assert all(not t.requires_grad for t in backbone.parameters())
    selected=[]
    for domain in ['source','real']:
        selected+=sorted([r for r in bank if r['domain']==domain],key=lambda r:hashlib.sha256(('wide-feature-audit:'+r['id']).encode()).hexdigest())[:4]
    for r in selected:
        x=W.source_item(source,r['row']) if r['domain']=='source' else W.real_item(pool[r['id']])
        np.testing.assert_array_equal(W.extract(backbone,[x])[0],r['feature'])
    models={};fits={};changed={};source_common={}
    for a in D.ARMS:
        f=C.read(W.DOC/f'FIT_{a}.json');fits[a]=f;C.verify(f['checkpoint'])
        assert f['initial_state']==C.read(D.DOC/f'FIT_{a}.json')['initial_state']
        assert [r['step'] for r in f['history']]==list(range(1,1001))
        assert all(np.isfinite(r['loss']) and r['loss']>0 for r in f['history'])
        ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==C.sha(W.DOC/'PROTOCOL.json')
        changed[a]=sum(N.array_sha(t.numpy())!=f['initial_state'][k] for k,t in ck['model'].items());assert changed[a]>0
        model=W.W.Head().cuda();model.load_state_dict(ck['model']);models[a]=model.eval()
        held=[dict(r,target_valid=r['old_target_valid']) for r in bank if r['domain']=='source' and not r['train']]
        with W.scope():source_common[a]=D.probe(model,held)
        assert source_common[a]['n']==506
    for b in C.read(W.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    for b in C.read(W.DOC/'DECISION_LOCK.json')['fits'].values():C.verify(b)
    baseline={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    original={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    outputs={a:{r['id']:r for r in C.read(W.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in D.ARMS}
    scores={a:{r['id']:r for r in C.read(W.RAW/f'SCREEN_{a}.json')['metrics']} for a in D.ARMS}
    for a in D.ARMS:assert set(outputs[a])==set(scores[a])==set(original) and len(outputs[a])==194
    receipts={r['id']:r for r in C.read(W.RAW/'INFERENCE_RECEIPTS.json')}
    max_delta={a:0. for a in D.ARMS};checked=0
    for i,r in enumerate(C.read(W.DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));key=r['id'];old=original[key]
        x=W.prepare_input(im,old['prediction']);feature=W.extract(backbone,[x])[0]
        assert N.array_sha(feature)==receipts[key]['feature_sha']
        np.testing.assert_array_equal(x['matrix'],receipts[key]['matrix'])
        feat=torch.as_tensor(feature,device='cuda').float()[None];q=torch.as_tensor(x['points'],device='cuda')[None];valid=torch.as_tensor(x['valid'],device='cuda')[None]
        for a,m in models.items():
            crop=W.W.decode(m(feat,q,valid))[0].cpu().numpy();new=N.C.transform_points(crop,np.linalg.inv(x['matrix']))
            new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
            pred=outputs[a][key]['prediction'];delta=float(np.max(np.abs(new-np.asarray(P.top(pred)['keypoints_xy']))))
            assert delta<=.001;(max_delta.__setitem__(a,max(max_delta[a],delta)))
            P.assert_preserved(old['prediction'],pred);checked+=1
        if (i+1)%50==0:print('WIDE_REPRODUCE',i+1,'/194',flush=True)
    pe,pop=D.R.E.O.population_metadata();targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in original}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    crop_audit=C.read(W.DOC/'CROP_SUPPORT_DIAGNOSTIC.json');crop_rows={(r['id'],r['GT_corner']):r for r in crop_audit['rows']}
    rows=[];spatial={};cases=[]
    for a in D.ARMS:
        ar=[]
        for key,r in outputs[a].items():
            b=baseline[key];n=scores[a][key];t=targets[key];gt=np.asarray(t.keypoints_xy);q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
            measured=EM.measure(q,gt,t.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(measured['canonical_errors'],float),np.asarray(n['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            assert n['matched']==b['matched'] and n['detected']==b['detected']
            if not b['matched']:continue
            oldq=np.asarray(P.top(original[key]['prediction'])['keypoints_xy'])[:8];valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
            for j,(before,after) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])):
                if before is None:continue
                distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min());c=crop_rows[(key,j)]
                row=dict(id=key,arm=a,GT_corner=j,before=before,after=after,nearest_R0_point_distance=distance,
                    parent_unreachable_within10=c['minimum_possible_error_px']['parent']>10,branches=[b['branch'],n['branch']])
                ar.append(row);rows.append(row)
                if before>40 and after<=10:cases.append(row)
        table={}
        for distance in [10,20,40]:
            ss=[r for r in ar if r['nearest_R0_point_distance']>distance and r['before']>20]
            table[f'no_old_point_within{distance}']=dict(hard=len(ss),recovered=sum(r['after']<=10 for r in ss))
        far=[r for r in ar if r['nearest_R0_point_distance']>40]
        table['far64_by_parent_domain']={str(blocked):dict(hard=sum(r['parent_unreachable_within10']==blocked for r in far),
            recovered=sum(r['parent_unreachable_within10']==blocked and r['after']<=10 for r in far)) for blocked in [False,True]}
        spatial[a]=table
    C.freeze(W.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_DIAGNOSTIC_NOT_TRAINING',rows=rows))
    result=C.read(W.DOC/'RESULTS.json')
    C.freeze(W.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,tests_passed=ntests,
        identical_parent_protocol_fields=equal,cached_input_and_scale_checks=len(bank),supervision_counts=counts,
        shared506_source_probe=source_common,backbone_frozen=True,cached_feature_reproductions=len(selected),
        same_initial_weights_as_parent=True,actual_steps_per_arm=1000,changed_weight_tensors=changed,
        new_train_eval_exact_hash_overlap=0,real_train_unique=217,output_preservation_and_metric_checks=checked,
        prediction_reproduction_max_px=max_delta,spatial_decomposition=spatial,recovered_over40_cases=cases,
        new_annotations=0,new_tags=0,auto_promoted=False,screen=result['passed'],
        evidence=[C.bound(__file__),C.bound(W.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(W.DOC/f) for f in
            ['PROTOCOL.json','RESULTS.json','CROP_SUPPORT_DIAGNOSTIC.json','FIT_SYN.json','FIT_MIX.json','OUTPUTS_LOCK.json']]+[C.bound(t) for t in tests]))
    print('WIDE_AUDIT_COMPLETE',dict(tests=ntests,outputs=checked,spatial=spatial,cases=cases),flush=True)


if __name__=='__main__':main()
