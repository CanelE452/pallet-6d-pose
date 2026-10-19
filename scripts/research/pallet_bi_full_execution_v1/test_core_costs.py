"""Small constructed cost receipts only; no dataset/scientific execution."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from scripts.research.pallet_bi_full_execution_v1 import core_costs as C


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True)+'\n')


def parallel(root,calls=7,*,complete=True,quiesced=False,arms=('C0','CP'),legacy=False):
    source={C.BATCH_KEY:C.FROZEN_BATCH};arms=list(arms);fid='image';signature='one_exact_packet'
    put(root/'PRODUCER_LOCK.json',dict(arms=arms,static_ordered_row_ids=dict(real=[fid]),workers=2,driver_sha256='helper',numerical_kernel_source_binding=source))
    for index,arm in enumerate(arms):
        current=calls if index==0 else 0
        folder=root/arm/'real/0000';intrinsic=calls if index==0 else 999999
        seal=dict(complete=True,id=fid,domain='real',ordinal=0,arm='C0',source_binding=source,
            cache_payload_sha256=signature,resource=dict(actual_pose_optimizer_calls=intrinsic,wall_seconds=99999.))
        put(folder/'FRAME_SEAL.json',seal);proof=None
        if index:
            origin=root.parent/(root.name+'_old_origin')/'C0/real/0000';put(origin/'FRAME_SEAL.json',seal)
            proof=dict(source_folder=str(origin),source_frame_seal_sha256=C.sha(origin/'FRAME_SEAL.json'))
        row=dict(complete=True,id=fid,arm=arm,domain='real',ordinal=0,driver_sha256='helper',numerical_kernel_source_binding=source,
            original_frame_seal_sha256=C.sha(folder/'FRAME_SEAL.json'),cache_payload_sha256=signature,exact_reused=bool(index),
            pilot_reuse_proof=proof,reuse_origin='SEALED_PILOT'if index else'C0',resource=dict(actual_new_pose_optimizer_calls=current,
                current_execution_CPU_seconds=2. if not index else .1,current_execution_wall_seconds=3. if not index else .2,worker_peak_RSS_bytes=4000))
        put(folder/'ROW_EXECUTION.json',row)
    put(root/'row_jobs/real/0000.json',dict(id=fid,domain='real',ordinal=0,current_job_CPU_seconds=2.3,current_job_wall_seconds=3.5,actual_new_optimizer_calls_this_job=calls))
    spec=dict(name=root.name,component='S2',mode='rows',root=str(root))
    if complete:put(root/'SUMMARY.json',dict(status='COMPLETE_NEW',new_pose_optimizer_calls=calls,wall_seconds=10.,missing_rows=[],technical_failures=[]))
    if quiesced:
        spec['terminal']=str(root.parent/(root.name+'_QUIESCE.json'))
        terminal=dict(unfinished_inmemory_rows=0,unknown_interrupted_optimizer_calls=0,completed_jobs=1,
            completed_arm_rows=len(arms),current_attempt_actual_new_optimizer_calls=calls,epoch=20.)
        if legacy:terminal['all_started_rows_completed_all3arms']=True
        else:
            if not complete:put(root/'SUMMARY.json',dict(status='INCOMPLETE',new_pose_optimizer_calls=calls,missing_rows=[],technical_failures=[]))
            put(root/'DISPATCH_COMPLETION.json',dict(active_rows_drained_before_exit=True,checkpoint_stop_observed=True,
                submitted_jobs=1,unsubmitted_jobs=0,summary_sha256=C.sha(root/'SUMMARY.json')))
            terminal.update(all_started_rows_completed_all_arms=True,active_rows_drained_before_exit=True,checkpoint_stop_observed=True,
                submitted_jobs=1,unsubmitted_jobs=0,orphan_scientific_artifacts=[],technical_failure_receipts=[],
                summary_sha256=C.sha(root/'SUMMARY.json'),dispatch_completion_sha256=C.sha(root/'DISPATCH_COMPLETION.json'),
                producer_lock_sha256=C.sha(root/'PRODUCER_LOCK.json'),row_execution_bindings=[dict(path=str(root/arm/'real/0000/ROW_EXECUTION.json'),
                    sha256=C.sha(root/arm/'real/0000/ROW_EXECUTION.json'),frame_seal_sha256=C.sha(root/arm/'real/0000/FRAME_SEAL.json'),
                    current_optimizer_calls=calls if index==0 else 0)for index,arm in enumerate(arms)])
        put(spec['terminal'],terminal)
    return spec


def pilot(root,calls):
    source={C.BATCH_KEY:C.FROZEN_BATCH};sp=root/'C0/real/0000/FRAME_SEAL.json';resource=dict(actual_pose_optimizer_calls=calls,CPU_seconds=5.,wall_seconds=6.)
    put(root/'PRODUCER_LOCK.json',dict(source_binding=source));put(sp,dict(complete=True,id='pilot',exact_reused=False,resource=resource,cache_payload_sha256='pilot_signature'))
    put(root/'SUMMARY.json',dict(status='DIAGNOSTIC_ONLY_TECHNICAL_PILOT',rows=1,new_optimizer_calls=calls,wall_seconds=7.,
        manifest=[dict(id='pilot',arm='C0',domain='real',ordinal=0,seal=str(sp),resource=resource,exact_reused=False)]))
    return dict(name='pilot',component='S2',mode='pilot',root=str(root),prior_S2=True,expected_calls=calls)


def test_copied_intrinsic_cost_never_enters_current_calls_or_tails(tmp_path):
    spec=parallel(tmp_path/'run');report=C.audit_attempt(spec)
    assert report['cost_complete']and report['actual_optimizer_calls']==7
    assert report['exact_reused_rows']==report['staged_consumed_rows']==1
    assert report['requested_arm_rows']==2 and report['observed_complete_packet_keys']==1
    assert report['source_implied_completed_family_dispatches']==5
    assert report['actual_family_dispatches']==C.UNKNOWN
    assert report['longest_completed_fresh_arm_rows'][0]['wall']==3.
    assert report['timing']['arm_CPU_seconds']['total']==2.1
    assert report['timing']['aggregate_wall_seconds']==C.UNKNOWN
    assert report['peak_worker_RSS_bytes']==4000


def test_missing_current_envelope_is_unknown_not_zero_even_with_intrinsic_frame_cost(tmp_path):
    spec=parallel(tmp_path/'active',complete=False);(Path(spec['root'])/'CP/real/0000/ROW_EXECUTION.json').unlink()
    report=C.audit_attempt(spec)
    assert report['known_optimizer_calls']==7 and report['actual_optimizer_calls']==C.UNKNOWN
    assert report['missing_current_receipt_frames']and not report['cost_complete']
    assert 'INCOMPLETE'in report['status']


def test_missing_call_field_and_unsealed_failed_work_never_become_zero(tmp_path):
    spec=parallel(tmp_path/'partial',complete=False);path=Path(spec['root'])/'C0/real/0000/ROW_EXECUTION.json';row=C.read(path)
    row['resource'].pop('actual_new_pose_optimizer_calls');put(path,row)
    unsealed=Path(spec['root'])/'C0/real/0001/RESULT.json.gz';unsealed.parent.mkdir(parents=True);unsealed.write_bytes(b'not parsed')
    put(Path(spec['root'])/'row_jobs/real/0001_FAILURE.json',dict(error='after unknown amount of work'))
    report=C.audit_attempt(spec)
    assert report['actual_optimizer_calls']==C.UNKNOWN and report['known_optimizer_calls']==0
    assert report['technical_failure_count']==1 and report['unsealed_partial_artifacts']
    assert any('missing/nonfinite'in reason for reason in report['unknown_cost_reasons'])


def test_preserved62807_once_with_F_reuse_and_duplicate_root_rejection(tmp_path):
    specs=[pilot(tmp_path/'pilot',7490)]
    for name,calls in [('confounded',37820),('corrected2',17497)]:
        spec=parallel(tmp_path/name,calls,complete=False,quiesced=True);spec.update(prior_S2=True,expected_calls=calls);specs.append(spec)
    f=parallel(tmp_path/'F',17);f['component']='F_UNCONDITIONAL_CORE';specs.append(f)
    report=C.aggregate(specs,expected_preserved_prior_S2=62807)
    assert report['preserved_prior_S2']['actual_optimizer_calls']==62807
    assert report['actual_optimizer_calls']==62824
    assert report['cost_complete']and len(report['attempts'])==4
    assert 'D diagnostics and D confound repair'in report['excluded_cost_scopes']
    with pytest.raises(ValueError,match='exactly once'):C.aggregate(specs+[deepcopy(specs[0])])


def test_active_snapshot_is_not_final_and_resumed_job_timing_not_negative_overhead(tmp_path):
    spec=parallel(tmp_path/'active',complete=False)
    job=Path(spec['root'])/'row_jobs/real/0000.json';put(job,dict(id='image',domain='real',ordinal=0,
        current_job_CPU_seconds=.2,current_job_wall_seconds=.2,actual_new_optimizer_calls_this_job=0))
    report=C.aggregate([spec]);attempt=report['attempts'][0]
    assert not report['cost_complete']and report['actual_optimizer_calls']==C.UNKNOWN
    assert attempt['timing']['job_wall_outside_arm_unattributable_envelopes']==1
    assert attempt['timing']['job_wall_outside_arm_known_sum']==0
    seal=C.run(tmp_path,tmp_path/'snapshot',specs=[spec])
    assert seal['snapshot_incomplete']and not seal['accounting_complete']


def test_missing_copy_call_field_or_job_envelope_remains_unknown(tmp_path):
    spec=parallel(tmp_path/'missing',complete=False)
    path=Path(spec['root'])/'CP/real/0000/ROW_EXECUTION.json';row=C.read(path);row['resource'].pop('actual_new_pose_optimizer_calls');put(path,row)
    (Path(spec['root'])/'row_jobs/real/0000.json').unlink()
    report=C.audit_attempt(spec)
    assert report['known_optimizer_calls']==7 and report['actual_optimizer_calls']==C.UNKNOWN
    assert report['timing']['job_CPU_seconds']['total']==C.UNKNOWN
    assert report['timing']['completed_image_keys_missing_job_envelope']==[dict(domain='real',ordinal=0)]


def test_bad_copy_claim_or_changed_origin_bytes_rejected(tmp_path):
    spec=parallel(tmp_path/'copy');path=Path(spec['root'])/'CP/real/0000/ROW_EXECUTION.json';row=C.read(path)
    row['resource']['actual_new_pose_optimizer_calls']=10;put(path,row)
    with pytest.raises(ValueError,match='exact copied row'):C.audit_attempt(spec)
    row['resource']['actual_new_pose_optimizer_calls']=0;put(path,row)
    origin=Path(row['pilot_reuse_proof']['source_folder'])/'FRAME_SEAL.json';origin.write_text('changed')
    with pytest.raises((ValueError,json.JSONDecodeError)):C.audit_attempt(spec)


def test_completed_feature_seal_authenticates_current_cost_envelopes(tmp_path):
    spec=parallel(tmp_path/'sealed');root=Path(spec['root']);lock=C.read(root/'PRODUCER_LOCK.json')
    put(root/'C0/real/FEATURE_SEAL.json',dict(complete=True,producer_lock_sha256=C.sha(root/'PRODUCER_LOCK.json'),
        source_binding=lock['numerical_kernel_source_binding'],row_execution_sha256={'0000':C.sha(root/'C0/real/0000/ROW_EXECUTION.json')},
        frame_seal_sha256={'0000':C.sha(root/'C0/real/0000/FRAME_SEAL.json')}))
    assert C.audit_attempt(spec)['actual_optimizer_calls']==7
    path=root/'C0/real/0000/ROW_EXECUTION.json';row=C.read(path);row['resource']['current_execution_CPU_seconds']=19.;put(path,row)
    with pytest.raises(ValueError,match='cost envelope differs'):C.audit_attempt(spec)


@pytest.mark.parametrize('arms',[('C0','CP'),('C0','CP','CB')])
def test_generic_zero_unknown_quiesce_uses_actual_all_arm_count(tmp_path,arms):
    spec=parallel(tmp_path/'generic',complete=False,quiesced=True,arms=arms)
    terminal=C.read(spec['terminal']);assert 'all_started_rows_completed_all3arms'not in terminal
    report=C.audit_attempt(spec)
    assert report['cost_complete']and report['actual_optimizer_calls']==7
    assert report['represented_arm_rows']==len(arms)and not report['requested_population_complete']
    assert report['exact_reused_rows']==len(arms)-1


def test_legacy_three_arm_receipt_kept_but_cannot_relabel_two_arm_F(tmp_path):
    three=parallel(tmp_path/'legacy3',complete=False,quiesced=True,arms=('C0','CP','CB'),legacy=True)
    assert C.audit_attempt(three)['actual_optimizer_calls']==7
    two=parallel(tmp_path/'fake3',complete=False,quiesced=True,legacy=True)
    report=C.audit_attempt(two);assert not report['cost_complete']and report['actual_optimizer_calls']==C.UNKNOWN


@pytest.mark.parametrize('field,value',[('unknown_interrupted_optimizer_calls',1),('unfinished_inmemory_rows',1),
    ('active_rows_drained_before_exit',False),('orphan_scientific_artifacts',['unknown result'])])
def test_generic_quiesce_does_not_promote_incomplete_work_to_zero(tmp_path,field,value):
    spec=parallel(tmp_path/'incomplete',complete=False,quiesced=True);term=C.read(spec['terminal']);term[field]=value;put(spec['terminal'],term)
    report=C.audit_attempt(spec);assert not report['cost_complete']and report['actual_optimizer_calls']==C.UNKNOWN


def test_generic_quiesce_requires_all_job_envelopes_and_bound_ROW_bytes(tmp_path):
    spec=parallel(tmp_path/'missing_job',complete=False,quiesced=True);(Path(spec['root'])/'row_jobs/real/0000.json').unlink()
    assert C.audit_attempt(spec)['actual_optimizer_calls']==C.UNKNOWN
    changed=parallel(tmp_path/'changed_row',complete=False,quiesced=True);path=Path(changed['root'])/'C0/real/0000/ROW_EXECUTION.json'
    row=C.read(path);row['resource']['current_execution_CPU_seconds']=99.;put(path,row)
    with pytest.raises(ValueError,match='Quiesce row binding differs'):C.audit_attempt(changed)


def test_explicit_catalog_guards62807_once_and_keeps_D4_snapshot_incomplete(tmp_path):
    specs=[pilot(tmp_path/'pilot',7490)]
    for name,calls in [('confounded',37820),('corrected2',17497)]:
        spec=parallel(tmp_path/name,calls,complete=False,quiesced=True);spec.update(prior_S2=True,expected_calls=calls);specs.append(spec)
    specs.append(parallel(tmp_path/'D4_active',11,complete=False))
    catalog=tmp_path/'CATALOG.json';put(catalog,dict(specs=specs,expected_preserved_prior_S2_optimizer_calls=62807))
    seal=C.run(tmp_path,tmp_path/'snapshot',catalog_path=catalog);report=C.read(tmp_path/'snapshot/CORE_COST_REPORT.json')
    assert seal['snapshot_incomplete']and not seal['accounting_complete']
    assert seal['input_catalog']['sha256']==C.sha(catalog)
    assert report['preserved_prior_S2']['actual_optimizer_calls']==62807
    assert report['known_optimizer_call_lower_bound']==62818 and report['actual_optimizer_calls']==C.UNKNOWN
    bad=tmp_path/'BAD.json';put(bad,dict(specs=specs,expected_preserved_prior_S2_optimizer_calls=62808))
    with pytest.raises(ValueError,match='exactly once'):C.run(tmp_path,tmp_path/'bad_output',catalog_path=bad)
