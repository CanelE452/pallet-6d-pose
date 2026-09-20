"""Checkpoint/input/evaluation audit, including new spatial-support recovery."""
import re
import subprocess
import sys
import numpy as np
import torch
from . import recovery_hint_dropout as H
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM

C=H.C


def main():
    p=H.verify();H.N.setup();old=H.P.verify()
    keys=['steps','initialization','real_records','real_samples','source_samples','source_held_rows','source_cache_signature']
    for k in keys:assert p[k]==old[k],k
    train={r['image']['sha256'] for r in p['real_records'] if r['train']}
    evaluation={r['image']['sha256'] for r in C.read(H.R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    assert len(train)==217 and not train&evaluation
    fit=C.read(H.DOC/'FIT_MASKED.json');C.verify(fit['checkpoint'])
    init=torch.load(C.ROOT/p['initialization']['path'],map_location='cpu',weights_only=False)['model_state_dict']
    last=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict']
    assert set(init)==set(last);buffers=[k for k in init if any(s in k for s in ['running_mean','running_var','num_batches_tracked'])]
    assert buffers and all(torch.equal(init[k],last[k]) for k in buffers)
    changed=[k for k in init if not torch.equal(init[k],last[k])];assert changed
    del init,last
    assert [r['step'] for r in fit['history']]==list(range(1,301))
    assert all(np.isfinite(r['update_norm']) and r['update_norm']>0 for r in fit['history'])
    for b in C.read(H.DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    basepred={r['id']:r['prediction'] for r in C.read(H.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC'}
    baseline={r['id']:r for r in C.read(H.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    pe,pop=H.R.E.O.population_metadata()
    targets={i.frame_id:pe.E._legacy_forbidden_target(i) for i,m in pop if i.frame_id in baseline}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    checked=0;spatial={};recoveries=[]
    for arm in H.ARMS:
        scores={r['id']:r for r in C.read(H.RAW/f'SCREEN_{arm}.json')['metrics']}
        rows=C.read(H.RAW/f'EVAL_PREDICTIONS_{arm}.json')['records'];assert len(rows)==194 and {r['id'] for r in rows}==set(baseline)
        corners=[]
        for r in rows:
            key=r['id'];b=baseline[key];n=scores[key];t=targets[key]
            H.P.assert_preserved(basepred[key],r['prediction']);checked+=1
            assert n['matched']==b['matched'] and n['detected']==b['detected']
            q=np.asarray(H.P.top(r['prediction'])['keypoints_xy']);gt=np.asarray(t.keypoints_xy)
            m=EM.measure(q,gt,t.keypoint_supervision_mask,perms,r['raw_hw'],b['matched'],b['detected'])
            np.testing.assert_allclose(np.asarray(m['canonical_errors'],float),np.asarray(n['canonical_errors'],float),atol=1e-9,rtol=0,equal_nan=True)
            if not b['matched']:continue
            oldq=np.asarray(H.P.top(basepred[key])['keypoints_xy'])[:8]
            usable=np.isfinite(oldq).all(-1)&~(oldq==-1).all(-1)
            for j,(a,z) in enumerate(zip(b['canonical_errors'],n['canonical_errors'])):
                if a is None:continue
                distance=float(np.linalg.norm(oldq[usable]-gt[j],axis=-1).min())
                row=dict(id=key,arm=arm,GT_corner=j,before=a,after=z,nearest_R0_point_distance=distance,
                    no_R0_point_within10=distance>10,branches=[b['branch'],n['branch']])
                corners.append(row)
                if a>40 and z<=10:recoveries.append(row)
        table={}
        for threshold in [20,40,100]:
            table[str(threshold)]={}
            for label,absent in [('existing_set_near_reference',False),('no_old_point_near_reference',True)]:
                rr=[r for r in corners if r['before']>threshold and r['no_R0_point_within10']==absent]
                table[str(threshold)][label]=dict(hard=len(rr),recovered=sum(r['after']<=10 for r in rr))
        spatial[arm]=table
    tests=[C.ROOT/'scripts/research/pallet_type_selftrain_v1'/n for n in ['test_hint_dropout.py','test_recovery_pseudo_denoise.py']]
    tests += [C.ROOT/'scripts/research/pallet_posefix_replay_v1'/n for n in ['test_replay.py','test_source.py']]
    test=subprocess.run([sys.executable,'-m','pytest','-q',*[str(t) for t in tests]],capture_output=True,text=True)
    print(test.stdout,flush=True);assert test.returncode==0,test.stderr
    count=int(re.search(r'(\d+) passed',test.stdout).group(1))
    result=C.read(H.DOC/'RESULTS.json')
    C.freeze(H.DOC/'COMPLETION_AUDIT.json',dict(experiment_complete=True,goal_complete=False,
        identical_control_fields=keys,real_train_unique=len(train),new_train_eval_exact_hash_overlap=0,
        historical_teacher_eval_overlap_still_present=3,BN_buffer_tensors_bit_exact=len(buffers),changed_weight_tensors=len(changed),
        actual_optimizer_steps=300,tests_passed=count,output_contract_and_recomputed_metric_checks=checked,
        spatial_decomposition=spatial,recovered_over40_cases=recoveries,
        interpretation='Existing-point proximity is a diagnostic,not proof of correspondence cause. No-old-point bucket is stricter new spatial-support recovery. No oracle selection or GT gate.',
        new_manual_labels=0,new_tags=0,auto_promoted=False,screen=result['passed'],
        evidence=[C.bound(__file__),C.bound(H.DOC/'PROTOCOL.json'),C.bound(H.DOC/'FIT_MASKED.json'),
            C.bound(H.DOC/'RESULTS.json'),C.bound(H.DOC/'OUTPUTS_LOCK.json')]+[C.bound(t) for t in tests]))
    print('HINT_AUDIT_COMPLETE',dict(tests=count,outputs=checked,BN_buffers=len(buffers),weights_changed=len(changed),
        spatial=spatial,recovered_over40=recoveries),flush=True)


if __name__=='__main__':main()
