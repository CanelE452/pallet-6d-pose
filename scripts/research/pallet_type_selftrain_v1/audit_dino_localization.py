"""Independent cache/input/output reproduction and spatial-recovery accounting."""
import hashlib
import re
import subprocess
import sys
import cv2
import numpy as np
import torch

from . import dino_localization as D
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=D.C;P=D.P;N=D.N


@torch.no_grad()
def main():
    p=D.verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    tests=[C.HERE/n for n in ['test_dino_localization_model.py','test_heatmap_mode_decoder.py','test_recovery_pseudo_denoise.py']]
    test=subprocess.run([sys.executable,'-m','pytest','-q',*map(str,tests)],capture_output=True,text=True)
    print(test.stdout,flush=True);assert test.returncode==0,test.stderr
    count=int(re.search(r'(\d+) passed',test.stdout).group(1));assert count==14
    train_hash={r['image']['sha256'] for r in p['real_records'] if r['train']}
    eval_hash={r['image']['sha256'] for r in C.read(D.R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    assert len(train_hash)==217 and not train_hash&eval_hash
    train_scenario={r['scenario'] for r in p['source_records'] if r['partition']=='train'}
    held_scenario={r['scenario'] for r in p['source_records'] if r['partition']=='heldout'}
    assert not train_scenario&held_scenario
    bank=D.load_bank();source=P.SourceData();old=P.verify();real,_,_=P.load_inputs(old)
    for r in bank:
        x=source.item(r['row']) if r['domain']=='source' else real[r['id']]
        for k,v in D.item_arrays(x).items():np.testing.assert_array_equal(r[k],v)
    backbone,_=D.A.load();assert all(not x.requires_grad for x in backbone.parameters())
    # Fixed hash-selected cache sample, unrelated to accuracy or eventual recovery.
    selected=[]
    for domain in ['source','real']:
        selected+=sorted([r for r in bank if r['domain']==domain],key=lambda r:hashlib.sha256(('dino-cache-audit:'+r['id']).encode()).hexdigest())[:4]
    feature_differences=[]
    for r in selected:
        x=source.item(r['row']) if r['domain']=='source' else real[r['id']]
        z=D.extract(backbone,[x])[0]
        delta=float(np.max(np.abs(z.astype(float)-r['feature'].astype(float))))
        assert delta<=.032,(r['id'],delta)  # FP16 storage, B4 cache vs B1 reproduction
        feature_differences.append(dict(id=r['id'],max_abs_delta=delta))
    models={};fits={};changed={}
    for arm in D.ARMS:
        f=C.read(D.DOC/f'FIT_{arm}.json');fits[arm]=f;C.verify(f['checkpoint'])
        assert [r['step'] for r in f['history']]==list(range(1,1001))
        assert all(np.isfinite(r['loss']) and r['loss']>0 for r in f['history'])
        ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['step']==1000 and ck['protocol_sha256']==C.sha(D.DOC/'PROTOCOL.json')
        changed[arm]=sum(N.array_sha(v.numpy())!=f['initial_state'][k] for k,v in ck['model'].items())
        assert changed[arm]>0
        model=D.M.Head().cuda();model.load_state_dict(ck['model']);models[arm]=model.eval()
    assert fits['SYN']['initial_state']==fits['MIX']['initial_state']
    for b in C.read(D.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    for b in C.read(D.DOC/'DECISION_LOCK.json')['fits'].values():C.verify(b)
    baseline={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    basepred={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    predictions={a:{r['id']:r for r in C.read(D.RAW/f'EVAL_PREDICTIONS_{a}.json')['records']} for a in D.ARMS}
    scores={a:{r['id']:r for r in C.read(D.RAW/f'SCREEN_{a}.json')['metrics']} for a in D.ARMS}
    receipts={r['id']:r for r in C.read(D.RAW/'INFERENCE_RECEIPTS.json')}
    for arm in D.ARMS:assert set(predictions[arm])==set(scores[arm])==set(basepred) and len(predictions[arm])==194
    max_delta={a:0. for a in D.ARMS};checked=0
    for i,r in enumerate(C.read(D.DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));key=r['id'];oldp=basepred[key]
        x=N.C.prepare_input(im,oldp['prediction']);z=D.extract(backbone,[x])[0]
        assert N.array_sha(z)==receipts[key]['feature_sha'],('Eval feature reproducibility',key)
        np.testing.assert_array_equal(x['matrix'],receipts[key]['matrix'])
        feat=torch.as_tensor(z,device='cuda').float()[None]
        q=torch.as_tensor(x['points'],device='cuda')[None];valid=torch.as_tensor(x['valid'],device='cuda')[None]
        for arm,m in models.items():
            crop=D.M.decode(m(feat,q,valid))[0].cpu().numpy()
            new=N.C.transform_points(crop,np.linalg.inv(x['matrix']))
            new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
            saved=predictions[arm][key]['prediction']
            delta=float(np.max(np.abs(new-np.asarray(P.top(saved)['keypoints_xy']))))
            assert delta<=.001,(arm,key,delta);max_delta[arm]=max(max_delta[arm],delta)
            P.assert_preserved(oldp['prediction'],saved);checked+=1
        if (i+1)%50==0:print('DINO_REPRODUCE',i+1,'/194',flush=True)
    # Predictions/checkpoints have now been fully reproduced without GT inputs.
    pe,pop=D.R.E.O.population_metadata()
    targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in basepred}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    spatial={};recoveries=[];corner_rows=[]
    for arm in D.ARMS:
        corners=[]
        for key,r in predictions[arm].items():
            b=baseline[key];n=scores[arm][key];t=targets[key];gt=np.asarray(t.keypoints_xy)
            q=np.asarray(P.top(r['prediction'])['keypoints_xy'])
            measured=EM.measure(q,gt,t.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(measured['canonical_errors'],float),np.asarray(n['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            assert b['matched']==n['matched'] and b['detected']==n['detected']
            if not b['matched']:continue
            oldq=np.asarray(P.top(basepred[key]['prediction'])['keypoints_xy'])[:8]
            valid=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
            for j,(a,z) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])):
                if a is None:continue
                distance=float(np.linalg.norm(oldq[valid]-gt[j],axis=-1).min())
                row=dict(id=key,arm=arm,GT_corner=j,before=a,after=z,nearest_R0_point_distance=distance,
                    no_R0_point_within10=distance>10,branches=[b['branch'],n['branch']])
                corners.append(row);corner_rows.append(row)
                if a>40 and z<=10:recoveries.append(row)
        table={}
        for threshold in [20,40,100]:
            table[str(threshold)]={}
            for label,absent in [('existing_set_near_reference',False),('no_old_point_near_reference',True)]:
                rows=[r for r in corners if r['before']>threshold and r['no_R0_point_within10']==absent]
                table[str(threshold)][label]=dict(hard=len(rows),recovered=sum(r['after']<=10 for r in rows))
        spatial[arm]=table
    C.freeze(D.RAW/'CORNER_DIAGNOSTICS.json',dict(status='POSTHOC_GT_DIAGNOSTIC_NOT_TRAINING',rows=corner_rows))
    result=C.read(D.DOC/'RESULTS.json')
    receipt=dict(experiment_complete=True,goal_complete=False,tests_passed=count,
        source_real_cached_target_point_crop_checks=len(bank),source_scenario_disjoint=True,real_train_images=217,
        new_train_eval_exact_hash_overlap=0,initial_heads_identical=True,changed_weight_tensors=changed,actual_steps_per_arm=1000,
        frozen_backbone=True,backbone_checkpoint_unchanged=True,feature_reproduction=feature_differences,
        full194_prediction_reproduction_max_px=max_delta,output_preservation_and_metric_checks=checked,
        spatial_decomposition=spatial,recovered_over40_cases=recoveries,
        limitations='No-old-point-within10 bucket is a thresholded spatial diagnostic,not proof of a missing physical corner. Report nearest distance:10.03 is marginal. No oracle selection/new pseudo labels.',
        new_annotations=0,new_tags=0,auto_promoted=False,screen=result['passed'],
        evidence=[C.bound(D.DOC/f) for f in ['PROTOCOL.json','BACKBONE.json','FIT_SYN.json','FIT_MIX.json','OUTPUTS_LOCK.json','RESULTS.json']]
            +[C.bound(__file__),C.bound(D.RAW/'CORNER_DIAGNOSTICS.json')]+[C.bound(t) for t in tests])
    C.freeze(D.DOC/'COMPLETION_AUDIT.json',receipt)
    print('DINO_AUDIT_COMPLETE',dict(tests=count,reproduced=checked,spatial=spatial,recovered=recoveries),flush=True)


if __name__=='__main__':main()
