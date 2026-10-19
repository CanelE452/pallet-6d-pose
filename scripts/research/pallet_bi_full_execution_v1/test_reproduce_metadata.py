"""Constructed metadata-source registration tests;no pipeline execution."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from . import reproduce as R

def tests():
    with TemporaryDirectory(prefix='reproduce-metadata-')as tmp:
        root=Path(tmp)
        def save(name,value):
            p=root/name;p.write_text(json.dumps(value)+'\n');return p
        save('PROCESS.json',dict(processes=dict(F_controller=dict(argv=['python','F_controller.py'],pid=123)),working_directory='CONSTRUCTED',launch_environment=dict(OMP_NUM_THREADS='1',FAKE_SECRET='MUST_NOT_EXPORT')))
        save('API.json',dict(delegated_call=dict(workers=4,output='CONSTRUCTED'),delegated_argv=['not','an','actual','CLI']))
        save('SUMMARY.json',dict(status='COMPLETE_NEW',rows=1241))
        (root/'LEDGER.jsonl').write_text('\n'.join(json.dumps(dict(event=e,phase='CONSTRUCTED',argv=['python','actual_child.py'],return_code=0))for e in ['START','COMPLETE'])+'\n')
        with (root/'LEDGER.jsonl').open('a')as stream:
            stream.write(json.dumps(dict(event='CONTROLLER_START',argv=['python','outer_controller.py']))+'\n')
            stream.write(json.dumps(dict(event='CONTROLLER_START',actual_argv=['python','derived_controller.py']))+'\n')
            stream.write(json.dumps(dict(event='METADATA_COMPLETE',phase='metadata',function_call='helper.metadata',subprocess=False))+'\n')
            stream.write(json.dumps(dict(event='CHECKPOINT_ACTION',phase='saved_stage',action='ADOPT',checkpoint_sha256='CONSTRUCTED'))+'\n')
        mp=save('MAP.json',dict(schema='REPRODUCTION_METADATA_SOURCES_V1',ledgers=['LEDGER.jsonl'],process_snapshots=['PROCESS.json'],API_records=[dict(path='API.json',fields=['delegated_call'],scope='API_only')],state_overrides=dict(S2_PRODUCER='SUMMARY.json',F_FINAL='PENDING.json')))
        calls=[]
        def record(*args,**kwargs):calls.append((args,kwargs))
        states,apis,binding,missing=R.extension_metadata(root,mp,record)
        assert len(calls)==5 and len(apis)==3 and all(x['CLI_invocation_claimed']is False for x in apis)
        assert apis[0]['explicit_API_metadata']['function_call']=='helper.metadata'
        assert calls[2][0][1:]==('CONTROLLER_START',['python','outer_controller.py'],'CONTROLLER_START')
        assert calls[3][0][1:]==('CONTROLLER_START',['python','derived_controller.py'],'CONTROLLER_START')
        assert all(args[2]!=['not','an','actual','CLI']for args,metadata in calls)
        assert 'FAKE_SECRET'not in calls[-1][1]['environment']and calls[-1][1]['environment']['OMP_NUM_THREADS']=='1'
        assert R.sealed_metadata(root,states['S2_PRODUCER'])['complete']is True
        assert states['F_FINAL']in missing and R.sealed_metadata(root,states['F_FINAL'])['status']=='PENDING'
        assert binding['sha256']==R.sha(mp)
        bad=save('BAD.json',dict(schema='REPRODUCTION_METADATA_SOURCES_V1',state_overrides=dict(NO='ISOLATED_REFERENCES.json')))
        try:R.extension_metadata(root,bad,record)
        except ValueError:pass
        else:raise AssertionError('Reference source metadata registration accepted')
    return dict(PASS=True,constructed_metadata_only=True,actual_dataset_rows_or_reference_decodes=0,scientific_calls=0,
        literal_outer_argv_and_ledger_only=True,delegated_API_not_faked_as_CLI=True,explicit_D4_state_overrides_and_missing_pending_supported=True,
        source_sha256=R.sha(R.__file__))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=tests();R.write_snapshot(a.output,r);print(json.dumps(r))
