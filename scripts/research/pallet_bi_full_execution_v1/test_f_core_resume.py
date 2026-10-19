"""Constructed files only; no scientific import/dispatcher/solve/fit/reference."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from . import f_core_resume as R


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open if path.suffix=='.gz'else open)(path,'wt')as stream:json.dump(value,stream,sort_keys=True)


def fixture(root):
    output=root/'F_CORE_THREE_CPU';source={'constructed.py':'frozen'};helper='constructed-row-helper'
    lock=dict(arms=list(R.ARMS),static_ordered_row_ids={'real':['real0'],'synthetic':['synthetic0']},
        numerical_kernel_source_binding=source,driver_sha256=helper,stop_file=str(root/'F_CORE_CHECKPOINT_STOP'))
    put(output/'PRODUCER_LOCK.json',lock);packets={}
    for domain,ids in lock['static_ordered_row_ids'].items():
        packets[(domain,0)]=dict(sha256=domain+'-packet',selection_sha256={arm:domain+'-'+arm+'-selection'for arm in R.ARMS})
        for arm in R.ARMS:
            folder=output/arm/domain/'0000';calls=5 if arm=='F_COMMON'else 0
            put(folder/'RESULT.json.gz',dict(id=ids[0],constructed_metadata_only=True))
            feature=dict(id=ids[0],split='train',constructed_metadata_only=True)
            put(folder/'FEATURE.json.gz',feature)
            seal=dict(id=ids[0],arm=arm,domain=domain,ordinal=0,complete=True,source_binding=source,
                numeric_reference_inputs=False,cache_payload_sha256=domain+arm+'-cache',
                artifacts={name:R.sha(folder/name)for name in ('RESULT.json.gz','FEATURE.json.gz')},resource=dict(actual_pose_optimizer_calls=calls))
            put(folder/'FRAME_SEAL.json',seal)
            receipt=dict(id=ids[0],arm=arm,domain=domain,ordinal=0,complete=True,driver_sha256=helper,
                numerical_kernel_source_binding=source,original_frame_seal_sha256=R.sha(folder/'FRAME_SEAL.json'),
                numeric_reference_inputs=False,cache_payload_sha256=seal['cache_payload_sha256'],
                requested_packet_file_sha256=packets[(domain,0)]['sha256'],
                requested_selection_sha256=packets[(domain,0)]['selection_sha256'][arm],resource=dict(actual_new_pose_optimizer_calls=calls))
            put(folder/'ROW_EXECUTION.json',receipt)
            features=folder.parent/'FEATURES.jsonl.gz'
            with gzip.open(features,'wt')as stream:stream.write(json.dumps(feature)+'\n')
            fs=dict(schema='BI_FULL_NORMAL_FEATURE_SEAL_V1',complete=True,domain=domain,arm=arm,rows=1,
                source_binding=source,driver_sha256=helper,producer_lock_sha256=R.sha(output/'PRODUCER_LOCK.json'),
                numeric_reference_inputs=False,features_sha256=R.sha(features),head_features_sha256=R.sha(features),
                frame_seal_sha256={'0000':R.sha(folder/'FRAME_SEAL.json')},row_execution_sha256={'0000':R.sha(folder/'ROW_EXECUTION.json')})
            put(features.parent/'FEATURE_SEAL.json',fs)
    put(output/'SUMMARY.json',dict(status='INCOMPLETE',historical_metadata=True));put(output/'row_jobs/real/0000_FAILURE.json',dict(error='constructed prior failure'))
    return output,lock,packets


def tests():
    with TemporaryDirectory(prefix='f-core-resume-metadata-')as temporary:
        root=Path(temporary);output,lock,packets=fixture(root)
        spawned=['python','-c','from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=7, pipe_handle=12)','--multiprocessing-fork']
        assert R.classify_owner(root,output,spawned,{'1':str(root/'F_CORE_STDOUT.log')})
        assert R.classify_owner(root,output,spawned,{'2':str(root/'F_CORE_STDOUT.log')})
        assert R.classify_owner(root,output,spawned,{'1':str(root/'S2_THREE_CPU_STDOUT.log')})is None
        assert R.classify_owner(root,output,['python','-c','from multiprocessing.resource_tracker import main;main(7)'],{'1':str(root/'F_CORE_STDOUT.log')})is None
        assert R.classify_owner(root,output,['python','-m','scripts.research.pallet_bi_full_execution_v1.heads','--campaign',str(root)],{'1':str(root/'F_CORE_STDOUT.log')})is None
        completed=R.authenticate_bank(output,lock,packets)
        assert completed['bank_complete']and completed['completed_rows']==4
        assert completed['stored_ROW_EXECUTION_optimizer_calls']==10
        # Exact prior pilot rows retain their original C0 seal header even
        # when requested as F_COMMON; verify the original and staged bytes.
        cross_output,cross_lock,cross_packets=fixture(root/'cross')
        current=cross_output/'F_COMMON/real/0000';original=root/'cross/original_pilot';original.mkdir()
        seal=R.read(current/'FRAME_SEAL.json');seal['arm']='C0';put(current/'FRAME_SEAL.json',seal)
        for path in current.iterdir():shutil.copy2(path,original/path.name)
        staged=root/'cross/reuse/F_COMMON/real/0000';staged.mkdir(parents=True)
        for name in ('FRAME_SEAL.json','RESULT.json.gz','FEATURE.json.gz'):shutil.copy2(original/name,staged/name)
        reuse_manifest=root/'cross/reuse/REUSE_MANIFEST.json';packet_sha='constructed-manifest'
        put(reuse_manifest,dict(packet_seal_sha256=packet_sha,rows=[dict(domain='real',ordinal=0,arm='F_COMMON',source=str(original),
            source_frame_seal_sha256=R.sha(original/'FRAME_SEAL.json'),desired_cache_payload_sha256=seal['cache_payload_sha256'])]))
        cross_lock.update(packet_seal_sha256=packet_sha,exact_reuse_binding=dict(root=str(reuse_manifest.parent),manifest_sha256=R.sha(reuse_manifest)))
        put(cross_output/'PRODUCER_LOCK.json',cross_lock)
        rr=R.read(current/'ROW_EXECUTION.json');rr.update(original_frame_seal_sha256=R.sha(current/'FRAME_SEAL.json'),exact_reused=True,reuse_origin='SEALED_PILOT',
            pilot_reuse_proof=dict(source_folder=str(staged),source_frame_seal_sha256=R.sha(current/'FRAME_SEAL.json'),
                copied_artifact_sha256=seal['artifacts'],dependency_payload_equality_verified=True,pilot_source_preserved=True))
        put(current/'ROW_EXECUTION.json',rr)
        for fs_path in cross_output.glob('*/*/FEATURE_SEAL.json'):
            fs0=R.read(fs_path);fs0['producer_lock_sha256']=R.sha(cross_output/'PRODUCER_LOCK.json')
            if fs_path.parent==current.parent:
                fs0['frame_seal_sha256']['0000']=R.sha(current/'FRAME_SEAL.json');fs0['row_execution_sha256']['0000']=R.sha(current/'ROW_EXECUTION.json')
            put(fs_path,fs0)
        cross=R.authenticate_bank(cross_output,cross_lock,cross_packets)
        assert cross['bank_complete']and cross['verified_cross_arm_original_seal_reuses'][0]['original_preserved_arm']=='C0'
        put(original/'RESULT.json.gz',dict(changed_original=True))
        try:R.authenticate_bank(cross_output,cross_lock,cross_packets)
        except ValueError:pass
        else:raise AssertionError('Changed original cross-arm artifact accepted')
        bad=deepcopy(packets);bad[('real',0)]['selection_sha256']['F_COMMON']='changed-selection'
        try:R.authenticate_bank(output,lock,bad)
        except ValueError:pass
        else:raise AssertionError('Wrong requested physical selection digest accepted')
        fp=output/'F_COMMON/real/FEATURES.jsonl.gz';fs=fp.parent/'FEATURE_SEAL.json';saved=fp.read_bytes();savedseal=fs.read_bytes()
        with gzip.open(fp,'wt')as stream:stream.write(json.dumps(dict(id='substituted_frame',split='train'))+'\n')
        resealed=R.read(fs);resealed['features_sha256']=resealed['head_features_sha256']=R.sha(fp);put(fs,resealed)
        try:R.authenticate_bank(output,lock,packets)
        except ValueError:pass
        else:raise AssertionError('Self-resealed wrong frame IDs accepted')
        fp.write_bytes(saved);fs.write_bytes(savedseal)
        # A completed bank must never dispatch or aggregate on re-entry.
        completed['bindings']={};fake_calls=[]
        def forbidden_dispatch(*args,**kwargs):raise AssertionError('Completed bank invoked dispatcher/aggregator')
        fake=SimpleNamespace(source_bindings=lambda:{},run=forbidden_dispatch)
        module=R.__package__+'.fullnormal_dispatch'
        with patch.dict(sys.modules,{module:fake}),patch.object(R,'authenticate',return_value=(lock,completed)),patch.object(R,'active_owners',return_value=[]):
            reused=R.run(root,root/'prior',GO=True)
        assert reused['dispatcher_calls']==reused['aggregator_calls']==reused['new_pose_optimizer_calls_this_invocation']==0
        assert reused['original_stored_optimizer_costs']==10
        # Preserve a partial aggregate and its old failure log before retry.
        folder=output/'F_BI/real/0000';(folder/'ROW_EXECUTION.json').unlink()
        for name in ('FEATURES.jsonl.gz','FEATURE_SEAL.json'):(folder.parent/name).unlink()
        partial=R.authenticate_bank(output,lock,packets);partial['bindings']={}
        assert not partial['bank_complete']and len(partial['sealed_frames_missing_execution_envelopes'])==1
        assert partial['historical_cost_unknown_possible']and not partial['unsealed_partial_artifacts']
        invocation=root/'snapshot';R.preserve_before(output,invocation,partial)
        old_summary=(output/'SUMMARY.json').read_bytes();put(output/'SUMMARY.json',dict(status='simulated_new_attempt'))
        assert (invocation/'before/SUMMARY.json').read_bytes()==old_summary
        assert (invocation/'before/row_jobs/real/0000_FAILURE.json').is_file()
        stop=Path(lock['stop_file']);stop.write_text('explicit human pause or checkpoint;preserve until root clear')
        with patch.dict(sys.modules,{module:fake}),patch.object(R,'authenticate',return_value=(lock,partial)),patch.object(R,'active_owners',return_value=[]):
            try:R.run(root,root/'prior',GO=True,resume_partial=True,resource_GO=True)
            except PermissionError:pass
            else:raise AssertionError('STOP automatically cleared or resumed')
        assert stop.exists()
        def simulated_dispatch(*args,**kwargs):
            fake_calls.append(kwargs);return dict(current_invocation_new_pose_optimizer_calls=7,technical_failures=[])
        fake.run=simulated_dispatch
        with patch.dict(sys.modules,{module:fake}),patch.object(R,'authenticate',return_value=(lock,partial)),patch.object(R,'active_owners',return_value=[]):
            resumed=R.run(root,root/'prior',GO=True,resume_partial=True,resource_GO=True,clear_stop=True)
        assert len(fake_calls)==1 and not stop.exists()
        assert resumed['new_pose_optimizer_calls_this_invocation']==7 and resumed['original_stored_optimizer_costs']==10
        assert (Path(resumed['invocation'])/'PRESERVED_STOP').read_text().startswith('explicit human pause')
    return dict(PASS=True,constructed_metadata_only=True,actual_dispatcher_calls=0,actual_optimizer_calls=0,
        actual_H1_fits=0,dataset_rows_read=0,reference_reads=0,completed_reuse_never_dispatches_or_aggregates=True,
        changed_packet_selection_and_wrong_feature_identity_rejected=True,
        partial_summaries_feature_seals_and_failure_logs_preserved=True,STOP_never_automatically_cleared=True,
        original_C0_cross_arm_seals_require_exact_registered_staging_origin_payload_and_bytes=True,
        missing_execution_envelope_preserves_unknown_prior_cost_flag=True,
        orphan_F_spawn_workers_recognized_by_exact_stdout_or_stderr_path=True,
        unrelated_S2_spawn_H1_and_idle_resource_tracker_not_rejected=True,
        simulated_current_calls_separate_from_original_stored_costs=True,source_sha256=R.sha(R.__file__))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();receipt=tests();R.write(a.output,receipt);print(json.dumps(receipt,sort_keys=True))
