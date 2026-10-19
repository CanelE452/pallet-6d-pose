"""Adversarial metadata fixtures for the frozen final selection protocol.

No actual dataset, checkpoint tensor, target, pose or scientific fit is used.
"""
import argparse
import json
from pathlib import Path
import tempfile

from . import f_finish as F, s1_union as U


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def own(root,arm,counts,loss,*,training=None,prediction=None):
    training=training or root/'training';prediction=prediction or root/'prediction'
    runs=[]
    for seed,count in zip(F.SEEDS,counts):
        checkpoint=training/str(seed)/'best.pt'
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        checkpoint.write_bytes(('constructed fixture, not a tensor: '+str(seed)).encode())
        runs.append(dict(seed=seed,complete=True,checkpoint=str(checkpoint),
            checkpoint_sha256=U.sha(checkpoint),best_epoch=1,best_validation_loss=loss,
            best_metrics=dict(validation=dict(frames=235,loss=loss,success_5cm5deg=count))))
    write(training/'TRAINING_MANIFEST.json',dict(complete=True,runs=runs,
        protocol=dict(seeds=list(F.SEEDS),representative_seed=F.SEEDS[0],
            split=[768,235],parameters=16321,max_epochs=50,patience=5)))
    pp=prediction/'PREDICTIONS.jsonl.gz';pp.parent.mkdir(parents=True,exist_ok=True)
    pp.write_bytes(b'constructed metadata fixture; no dataset predictions')
    write(prediction/'PREDICTION_SEAL.json',dict(complete=True,rows=238,before_reference_decode=True,
        training_manifest_sha256=U.sha(training/'TRAINING_MANIFEST.json'),predictions_sha256=U.sha(pp)))
    return training,prediction


def fixture(root):
    s2_core,s2_heads=root/'S2_CORE',root/'S2_HEADS'
    for name,value in {
        'CAMPAIGN_PLAN.json':dict(seeds=list(F.SEEDS)),
        'F_SELECTION_RULE_CLARIFICATION.json':dict(rule=['fixture']*6),
        'F_AUXILIARY_MAPPING_IMPLEMENTATION_NOTE.json':dict(root_authorized_mapping={
            'S1':['PF2'],**{'S2_'+a:[]for a in ('C0','CP','CB')},
            **{a:['PF2',a]for a in F.MATERIAL}}),
        'F_OBSERVER_POLICY_LOCK_V2.json':{},'F_CORE_PRECOMPUTE_LOCK.json':{},
        's3_s4_postprocess/FINISH_RECEIPT.json':dict(complete=True,H1_fits=15,sealed_artifact_paths={}),
        'S1_EXECUTION_SUMMARY.json':dict(phase_wall_seconds={
            'new_synthetic_SubPix_and_F':1.,'real_union_feature_build':1.,'synthetic_union_feature_build':1.}),
        's3_s4_production/FULL_RECEIPT.json':dict(invocation_wall_seconds=7.),
    }.items():write(root/name,value)
    write(s2_core/'SUMMARY.json',dict(status='COMPLETE_NEW',missing_rows=[],technical_failures=[],wall_seconds=9.))
    own(root,'S1',[90,1,1],.1,training=root/'s1_training',prediction=root/'s1_predictions')
    for a,count in (('C0',44),('CP',45),('CB',44)):
        own(s2_heads/a,'S2_'+a,[count]*3,.2)
    write(s2_heads/'ALL_REAL_PREDICTIONS_SEAL.json',dict(complete=True,rows_per_arm=238,
        prediction_seal_sha256={a:U.sha(s2_heads/a/'prediction/PREDICTION_SEAL.json')for a in ('C0','CP','CB')}))
    for a,count,loss in (('S3_COMMON',48,.3),('S3_BI',48,.2),('S4_PL',47,.1),
                         ('S4_CONTOUR',30,.1),('S4_IMAGE',29,.1)):
        own(root,a,[count]*3,loss,training=root/'s3_s4_training'/a,
            prediction=root/'s3_s4_predictions'/a)
    for ev,ind,arms in (
        ('S1_EVALUATION','S1_INDEPENDENT_FINAL',['S1_UNION']),
        ('S2_EVALUATION','S2_INDEPENDENT',['S2_C0','S2_CP','S2_CB']),
        ('S3_S4_EVALUATION','S3_S4_INDEPENDENT',list(F.MATERIAL))):
        summary=dict(PRIMARY192={a+'_s20261018':dict(denominator=192,successes=180 if a=='S4_PL' else 119)for a in arms})
        write(root/ev/'SUMMARY.json',summary)
        write(root/ev/'EVALUATION_SEAL.json',dict(artifacts={'SUMMARY.json':U.sha(root/ev/'SUMMARY.json')}))
        write(root/ind/'INDEPENDENT_VERIFICATION.json',dict(PASS=True,rows=238,
            canonical_evaluation_seal_sha256=U.sha(root/ev/'EVALUATION_SEAL.json')))
    return s2_core,s2_heads


def test(output):
    with tempfile.TemporaryDirectory()as temporary:
        root=Path(temporary);core,heads=fixture(root)
        choice=F.choose(root,core,heads)
        assert choice['selected_configuration']=='S3_BI'
        assert choice['ranked_configuration_names'][0:2]==['S3_BI','S3_COMMON']
        assert choice['own_validation']['S1']['median_joint_success']==1
        assert choice['primary_counts_used_only_for_fallback']['S4_PL']==180
        assert not choice['primary_configuration_ranking']
        assert F.choose(root,core,heads)==choice
        cp=root/'F_SYSTEM_CHOICE.json';cp.rename(root/'VALID_FIXTURE_CHOICE.json')
        summary=root/'S3_S4_EVALUATION/SUMMARY.json';original=summary.read_bytes()
        data=U.read(summary);data['PRIMARY192']['S4_PL_s20261018']['successes']=181
        write(summary,data)
        try:F.choose(root,core,heads)
        except ValueError as e:assert 'Canonical score artifact changed'in str(e)
        else:raise AssertionError('Tampered canonical summary was accepted')
        summary.write_bytes(original)
        aggregate=heads/'ALL_REAL_PREDICTIONS_SEAL.json';data=U.read(aggregate)
        data['prediction_seal_sha256']['C0']='0'*64;write(aggregate,data)
        try:F.choose(root,core,heads)
        except ValueError as e:assert 'S2 aggregate prediction ancestry differs'in str(e)
        else:raise AssertionError('Detached aggregate ancestry accepted')
    result=dict(PASS=True,median_not_best_seed=True,VAL_loss_tie_break=True,
        PRIMARY_winner_not_selected=True,completed_choice_exact_reused=True,
        tampered_canonical_summary_rejected=True,detached_S2_aggregate_rejected=True,
        constructed_metadata_only=True,campaign_dataset_calls=0,model_forwards=0,pose_calls=0,fits=0,reference_reads=0)
    U.write(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(test(parser.parse_args().output))
