"""Prepare a small closing receipt after every sealed archive batch is remote.

The receipt binds the payload commit rather than its own future commit, avoiding
self-reference. Existing archive and its verification manifest remain immutable.
No Git mutations are performed here. Root uses the same sealed push runner next.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--repository',required=True,type=Path)
    ap.add_argument('--admin',required=True,type=Path)
    a=ap.parse_args()
    state=json.loads((a.admin/'MAIN_PUSH_PROGRESS.json').read_text())
    plan=json.loads((a.admin/'ROOT_SEALED_MAIN_PUSH_PLAN.json').read_text())
    assert state['completed_all'] and state['pending'] is None
    assert len(state['completed'])==len(plan['batches'])
    tip=state['expected_remote']
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=a.repository).decode().split()[0]==tip
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repository).decode().strip()==tip
    target=a.repository/'_docs/experiments/pallet-keypoint-visibility'
    verification=json.loads((target/'publication/PUBLICATION_VERIFICATION_SUMMARY.json').read_text())
    assert verification['PASS']
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt=dict(completed=True,UTC=now,remote='https://github.com/CanelE452/pallet-6d-pose.git',branch='main',
                 payload_main_commit=tip,payload_batches=state['completed'],
                 payload_files=sum(b['files'] for b in state['completed']),
                 payload_stored_bytes=sum(b['stored_bytes'] for b in state['completed']),
                 main_readme='https://github.com/CanelE452/pallet-6d-pose/blob/main/_docs/experiments/pallet-keypoint-visibility/README.md',
                 covered_versions='All permitted V63..V186 directories plus v60/v61/v62, with failures and UNKNOWN; V112 excluded before access',
                 verification=verification,goal_status='active_unsolved',model_adopted=False,
                 ongoing_policy='Freeze and validate each completed experiment; publish original code/method/results/failures/crops/tables and update index; push main and verify remote.',
                 source_data_and_model_scope='Raw dataset banks, input/feature caches, environments, large foundation weights and V112 excluded with provenance. Review figures, saved outputs and small own heads are published.',
                 receipt_commit_note='This record binds the completed payload commit above. The subsequent closing receipt commit is verified externally, avoiding circular commit identifiers.')
    out=a.source/'MAIN_PUBLICATION_RECEIPT_20261024.json'
    with out.open('x') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
    for name in ['PUBLICATION_POLICY_KO.json','ACTIVE_HANDOFF_KO.json']:
        original=(a.source/name).read_bytes()
        (a.admin/('BEFORE_MAIN_PUBLICATION_COMPLETE_'+name)).write_bytes(original)
        obj=json.loads(original)
        if name=='PUBLICATION_POLICY_KO.json':
            obj['main_pushing_status']='Completed verified payload main publication; closing receipt publication follows.'
            obj['last_payload_main_commit']=tip
            obj['last_payload_UTC']=now
            obj['local_publication_receipt']=str(out)
        else:
            obj['latest_main_publication']=dict(payload_commit=tip,receipt=str(out),root_url=receipt['main_readme'],ongoing_completed_experiment_push_authorized=True)
            obj['active_work']='V186 source-only registration diagnosis complete, V184 remains latest actual model comparison; all permitted historical review artifacts published to main. Overall visibility/external-occlusion goal remains unsolved. Continue only under prior eval-only/no-PnP/full-WOOD-SQUARE-native8/UNKNOWN/V112 restrictions and push each completed attempt with verified results.'
        (a.source/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    sources=[(out,'publication/MAIN_PUBLICATION_RECEIPT_20261024.json'),
             (a.source/'PUBLICATION_POLICY_KO.json','publication/PUBLICATION_POLICY_AFTER_PAYLOAD_PUSH.json'),
             (a.admin/'ROOT_SEALED_MAIN_PUSH_PLAN.json','publication/ROOT_MAIN_PUSH_PLAN_20261024.json'),
             (a.admin/'MAIN_PUSH_PROGRESS.json','publication/ALL14_REMOTE_PUSH_VERIFICATIONS.json'),
             (a.admin/'make_push_plan.py','tools/make_push_plan.py'),
             (a.source/'publication_preparation_c/MAKE_PUSH_PLAN_STATIC_REVIEW_V1.json','publication/MAKE_PUSH_PLAN_STATIC_REVIEW_V1.json'),
             (Path(__file__),'tools/record_main_publication.py')]
    files=[]
    for origin,relative in sources:
        assert 'v112' not in str(origin).lower() and 'v112' not in relative.lower()
        data=origin.read_bytes();dest=target/relative
        assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(data)
        files.append(dict(path='_docs/experiments/pallet-keypoint-visibility/'+relative,sha256=sha(data),size_bytes=len(data)))
    closing=dict(authorized_main_push=True,repository=str(a.repository),remote=receipt['remote'],branch='main',
                 initial_remote_main=tip,manifest_sha256=sha((target/'PUBLICATION_MANIFEST.json').read_bytes()),
                 batch_max_bytes=480*1024**2,batches=[dict(files=files,message='Record verified visibility publication and continuing main push policy')])
    assert sum(f['size_bytes'] for f in files)<closing['batch_max_bytes']
    f=a.admin/'ROOT_CLOSING_RECEIPT_PUSH_PLAN.json'
    with f.open('x') as stream:json.dump(closing,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(completed=True,payload_tip=tip,closing_files=len(files),closing_plan=str(f))))


if __name__=='__main__':main()
