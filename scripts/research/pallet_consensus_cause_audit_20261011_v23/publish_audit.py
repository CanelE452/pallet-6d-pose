"""Publish only the explicit saved-audit files to the authorized branch."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = Path('/tmp/pallet-observation-publication-stage-20261011')
BRANCH = 'research/observation-refiner-robust-pnp-20261009'
PARENT = 'ff34ce4ba8ae52a6eed0d6f88f37a0b7bffd6178'
MAIN = 'f565418e6a63acd5fb920a5741308d84eb7c131f'
DOC = '_docs/experiments/pallet_consensus_cause_audit_20261011_v23'
CODE = 'scripts/research/pallet_consensus_cause_audit_20261011_v23'


def git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def state(repo):
    commands={'index':('ls-files','--stage','-z'),
        'diff':('diff','--no-ext-diff','--binary'),
        'staged':('diff','--cached','--no-ext-diff','--binary'),
        'status':('status','--porcelain=v1','-z','-uall')}
    return {'HEAD':git(repo,'rev-parse','HEAD').decode().strip()} | {
        name:sha(git(repo,*args)) for name,args in commands.items()}


def protected():
    baseline=json.loads(Path('/tmp/pallet-publication-copied-reader-next-20261011/ROOT_PROTECTED_BASELINE.json').read_text())['expected']
    actual={name:state(Path(repo)) for name,repo in (
        ('original','/home/minjae/Documents/github/pallet-pose'),
        ('source','/tmp/pallet-kp-publication-20261010-v6'))}
    assert actual==baseline, 'preserve original and SOURCE canonical states'
    return actual


def remote():
    result=git(STAGE,'ls-remote','origin','refs/heads/main','refs/heads/'+BRANCH).decode()
    return {ref:commit for commit,ref in (line.split() for line in result.splitlines())}


def save(name,value):
    path=ROOT/name
    with path.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    assert git(STAGE,'rev-parse','HEAD').decode().strip()==PARENT
    assert git(STAGE,'rev-parse','--abbrev-ref','HEAD').decode().strip()!='main'
    assert not git(STAGE,'diff','--cached','--name-only','-z')
    assert not git(STAGE,'diff','--name-only','-z')
    before=protected()
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:PARENT}
    scripts=['saved_cause_audit.py','saved_boundary_state_audit.py','plot_saved_cases.py','check_saved_public.py','publish_audit.py']
    docs=['RESULT_KO.md','PROTOCOL.json','SUMMARY.json','CHECKS.json','ROWS.jsonl.gz','ROWS_STORAGE_CHECKS.json',
        'SELECTED_BOUNDARY_ROWS.jsonl','BOUNDARY_STATE_PROTOCOL.json','BOUNDARY_STATE_SUMMARY.json','BOUNDARY_STATE_CHECKS.json',
        'PLOT_PROTOCOL.json','PLOT_CHECKS.json','ALL24_SAVED_ERRORS.png','ALL24_SAVED_ERRORS.svg',
        'ROOT_VISUAL_REVIEW.json','SOURCE_REVIEW.json']
    items=[(ROOT/name,CODE+'/'+name) for name in scripts]+[(ROOT/name,DOC+'/'+name) for name in docs]
    items += [(ROOT/'initial_label_review'/name,DOC+'/initial_label_review/'+name) for name in (
        'saved_cause_audit.py','PROTOCOL.json','SUMMARY.json','CHECKS.json','REVIEW.json')]
    selected=[]; manifest=[]
    for src,relative in items:
        dst=STAGE/relative;assert src.is_file() and not dst.exists()
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert src.read_bytes()==dst.read_bytes()
        selected.append(relative);manifest.append(dict(path=relative,bytes=dst.stat().st_size,sha256=sha(dst.read_bytes())))
    check= subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',str(STAGE/CODE/'check_saved_public.py'),str(STAGE/DOC)])
    derived=json.loads(check);assert derived['passed'] and derived['original_upstream245_replay_performed'] is False
    save('PUBLIC_DERIVED_CHECKS.json',derived)
    dst=STAGE/DOC/'PUBLIC_DERIVED_CHECKS.json';shutil.copy2(ROOT/'PUBLIC_DERIVED_CHECKS.json',dst)
    selected.append(str(dst.relative_to(STAGE)));manifest.append(dict(path=selected[-1],bytes=dst.stat().st_size,sha256=sha(dst.read_bytes())))
    latest='_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md'
    path=STAGE/latest;old=path.read_bytes()
    addition='\n추가 원인 감사: [59장 4점 가설→재적합→선택 추적과 전체24장 그림](../pallet_consensus_cause_audit_20261011_v23/RESULT_KO.md). 저장 결과만 대조했으며 새 학습·PnP·자세 성능 실험은0이다. 성능 목표는 아직 미달이다.\n'
    assert addition.encode() not in old
    path.write_bytes(old+addition.encode());selected.append(latest)
    manifest.append(dict(path=latest,bytes=path.stat().st_size,sha256=sha(path.read_bytes()),previous_sha256=sha(old)))
    protocol=dict(schema='explicit_saved_audit_publication_v23',parent=PARENT,remote_main_before=MAIN,
        branch=BRANCH,protected_states=before,files=manifest,
        own_protocol_not_in_its_self_hash_manifest=True,explicit_git_add_force_for_ignored_research_files=True,
        force_push=False,main_mutation=False,new_PnP=0,new_training_updates=0,new_pose_accuracy_evaluations=0,
        accuracy_goal_achieved=False)
    save('PUBLICATION_PROTOCOL.json',protocol)
    dst=STAGE/DOC/'PUBLICATION_PROTOCOL.json';shutil.copy2(ROOT/'PUBLICATION_PROTOCOL.json',dst);selected.append(str(dst.relative_to(STAGE)))
    specification=ROOT/'SELECTED_PATHS.nul'
    with specification.open('xb') as f:f.write(b''.join(p.encode()+b'\0' for p in selected))
    save('PUBLICATION_STARTED.json',dict(parent=PARENT,explicit_paths=len(selected),protocol_sha256=sha((ROOT/'PUBLICATION_PROTOCOL.json').read_bytes())))
    assert protected()==before and git(STAGE,'rev-parse','HEAD').decode().strip()==PARENT
    git(STAGE,'add','-f','--pathspec-from-file='+str(specification),'--pathspec-file-nul')
    staged=git(STAGE,'diff','--cached','--name-only','-z').decode().split('\0')[:-1]
    assert sorted(staged)==sorted(selected)
    git(STAGE,'diff','--cached','--check')
    git(STAGE,'commit','-m','research: trace saved good-corner hypotheses and consensus failures')
    head=git(STAGE,'rev-parse','HEAD').decode().strip()
    changed=git(STAGE,'diff-tree','--no-commit-id','--no-renames','--name-only','-r','-z',PARENT,head).decode().split('\0')[:-1]
    assert sorted(changed)==sorted(selected)
    for item in manifest:
        assert sha(git(STAGE,'show',head+':'+item['path']))==item['sha256']
    assert protected()==before
    git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    after=remote()
    assert after=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:head}
    assert protected()==before
    assert not git(STAGE,'diff','--cached','--name-only','-z')
    assert not git(STAGE,'diff','--name-only','-z')
    save('REMOTE_CONFIRMATION.json',dict(schema='actual_saved_cause_audit_remote_confirmation_v23',
        complete=True,passed=True,parent=PARENT,HEAD=head,remote=after,remote_SHA_equals_HEAD=True,
        remote_main_unchanged=True,original_SOURCE_protected=True,stage_index_empty=True,
        selected_files=len(selected),committed_manifest_blobs_verified=len(manifest),
        public_derived_reader_passed=True,accuracy_goal_achieved=False))
    print(json.dumps(dict(complete=True,HEAD=head,remote_verified=True,main_unchanged=True,files=len(selected))))


if __name__=='__main__':
    main()
