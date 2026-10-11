"""Resume the stopped pre-commit publication after preserving its EOF-only failure."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v25_original_publisher',ROOT/'publish.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
STAGE=original.STAGE;DOC=original.DOC;CODE=original.CODE


def sha(b):return hashlib.sha256(b).hexdigest()


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def git(*args):return original.git(STAGE,*args)


def main():
    before=original.protected();assert git('rev-parse','HEAD').decode().strip()==original.PARENT
    assert original.remote()=={'refs/heads/main':original.MAIN,'refs/heads/'+original.BRANCH:original.PARENT}
    selected=(ROOT/'SELECTED_PATHS.nul').read_bytes().decode().split('\0')[:-1]
    assert sorted(selected)==sorted(git('diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    assert not git('diff','--name-only','-z')
    old=json.loads((ROOT/'PUBLICATION_PROTOCOL.json').read_text())
    for m in old['files']:
        b=(STAGE/m['path']).read_bytes();assert len(b)==m['bytes'] and sha(b)==m['sha256'],m['path']
    assert (STAGE/DOC/'PUBLICATION_PROTOCOL.json').read_bytes()==(ROOT/'PUBLICATION_PROTOCOL.json').read_bytes()
    check=subprocess.run(['git','-C',str(STAGE),'diff','--cached','--check'],capture_output=True,text=True)
    expected=DOC+'/RESULT_KO.md:110: new blank line at EOF.\n'
    assert check.returncode==2 and check.stdout==expected and not check.stderr
    save(ROOT/'PUBLICATION_PREFLIGHT_FAILURE.json',dict(phase='before commit',HEAD=original.PARENT,
        returncode=check.returncode,stdout=check.stdout,stderr=check.stderr,
        commit_performed=False,push_performed=False,numerical_audit_failure=False))
    p=ROOT/'RESULT_KO.md';old_report=p.read_bytes();assert old_report==(STAGE/DOC/'RESULT_KO.md').read_bytes()
    with (ROOT/'RESULT_KO_BEFORE_EOF_FIX.md').open('xb') as f:f.write(old_report)
    corrected=old_report.rstrip(b'\n')+b'\n';assert corrected!=old_report
    assert b''.join(corrected.split())==b''.join(old_report.split())
    p.write_bytes(corrected);(STAGE/DOC/'RESULT_KO.md').write_bytes(corrected)
    assert all(sha((STAGE/m['path']).read_bytes())==m['sha256'] for m in old['files'] if m['path']!=DOC+'/RESULT_KO.md')
    save(ROOT/'WHITESPACE_REPAIR_CHECKS.json',dict(passed=True,before_report_sha256=sha(old_report),
        after_report_sha256=sha(corrected),removed_bytes=len(old_report)-len(corrected),
        only_EOF_newline_removed=True,non_whitespace_report_content_identical=True,
        all_other_previous_staged_files_unchanged=True,all_scientific_inputs_rows_images_codes_unchanged=True,
        new_PnP=0,new_training=0,new_pose_accuracy=0))
    additions={DOC+'/PUBLICATION_PREFLIGHT_PROTOCOL.json':ROOT/'PUBLICATION_PROTOCOL.json',
        DOC+'/PUBLICATION_PREFLIGHT_FAILURE.json':ROOT/'PUBLICATION_PREFLIGHT_FAILURE.json',
        DOC+'/WHITESPACE_REPAIR_CHECKS.json':ROOT/'WHITESPACE_REPAIR_CHECKS.json',
        CODE+'/resume_publication_after_eof_check.py':Path(__file__)}
    for rel,src in additions.items():original.copy(src,rel);selected.append(rel)
    protocol_path=DOC+'/PUBLICATION_PROTOCOL.json'
    manifest=[dict(path=rel,bytes=(STAGE/rel).stat().st_size,sha256=sha((STAGE/rel).read_bytes()))
        for rel in selected if rel!=protocol_path]
    save(ROOT/'PUBLICATION_RESUME_PROTOCOL.json',dict(schema='actual_V25_EOF_only_publication_resume',
        parent=original.PARENT,branch=original.BRANCH,remote_main_before=original.MAIN,
        protected_states=before,files=manifest,preserved_preflight_failure=True,
        original_preflight_protocol=DOC+'/PUBLICATION_PREFLIGHT_PROTOCOL.json',
        scientific_artifacts_unchanged=True,report_nonwhitespace_identical=True,
        public_derived_reader_passed=True,source_join_replay_passed=True,real_join_replay_passed=True,
        fresh_clone=False,force_push=False,main_mutation=False,new_PnP=0,new_training=0,
        accuracy_goal_achieved=False,own_protocol_not_in_self_hash_manifest=True))
    original.copy(ROOT/'PUBLICATION_RESUME_PROTOCOL.json',protocol_path)
    specfile=ROOT/'RESUME_SELECTED_PATHS.nul';specfile.write_bytes(b''.join(s.encode()+b'\0' for s in selected))
    git('add','-f','--pathspec-from-file='+str(specfile),'--pathspec-file-nul')
    assert sorted(selected)==sorted(git('diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    git('diff','--cached','--check');assert original.protected()==before
    git('commit','-m','research: audit point admission and all adopted real LOO observations')
    head=git('rev-parse','HEAD').decode().strip()
    assert sorted(selected)==sorted(git('diff-tree','--no-commit-id','--no-renames','--name-only','-r','-z',original.PARENT,head).decode().split('\0')[:-1])
    for rel in selected:assert git('show',head+':'+rel)==(STAGE/rel).read_bytes()
    for m in manifest:assert sha(git('show',head+':'+m['path']))==m['sha256']
    assert original.protected()==before
    git('push','origin','HEAD:refs/heads/'+original.BRANCH)
    remote=original.remote();assert remote=={'refs/heads/main':original.MAIN,'refs/heads/'+original.BRANCH:head}
    assert original.protected()==before
    receipt=dict(complete=True,passed=True,published_audit_commit=head,remote=remote,
        remote_SHA_equals_published_commit=True,main_unchanged=True,original_SOURCE_protected=True,
        selected_files=len(selected),all_committed_files_byte_verified=True,
        public_derived_reader_passed=True,source_join_replay_passed=True,real_join_replay_passed=True,
        preflight_EOF_failure_preserved=True,preflight_repair_scientific_bytes_unchanged=True,
        new_PnP=0,new_pose_accuracy=0,new_training=0,accuracy_goal_achieved=False)
    save(ROOT/'REMOTE_CONFIRMATION.json',receipt);original.copy(ROOT/'REMOTE_CONFIRMATION.json',DOC+'/REMOTE_CONFIRMATION.json')
    git('add','-f','--',DOC+'/REMOTE_CONFIRMATION.json')
    assert git('diff','--cached','--name-only').decode().strip()==DOC+'/REMOTE_CONFIRMATION.json'
    git('diff','--cached','--check')
    git('commit','-m','research: record verified point-admission audit publication')
    sealed=git('rev-parse','HEAD').decode().strip()
    git('push','origin','HEAD:refs/heads/'+original.BRANCH)
    assert original.remote()=={'refs/heads/main':original.MAIN,'refs/heads/'+original.BRANCH:sealed}
    assert original.protected()==before
    assert not git('diff','--cached','--name-only','-z') and not git('diff','--name-only','-z')
    save(ROOT/'FINAL_REMOTE_CHECKS.json',dict(complete=True,passed=True,HEAD=sealed,audit_commit=head,
        remote_SHA_equals_HEAD=True,main_unchanged=True,original_SOURCE_protected=True,
        published_receipt_verified=True,accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True,HEAD=sealed,audit_commit=head,remote_verified=True,main_unchanged=True,files=len(selected))))


if __name__=='__main__':main()
