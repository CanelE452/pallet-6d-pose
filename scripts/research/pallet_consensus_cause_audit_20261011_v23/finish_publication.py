"""Resume the preserved publication after formatting generated SVG whitespace."""
import gzip
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STAGE=Path('/tmp/pallet-observation-publication-stage-20261011')
DOC='_docs/experiments/pallet_consensus_cause_audit_20261011_v23'
CODE='scripts/research/pallet_consensus_cause_audit_20261011_v23'
PARENT='ff34ce4ba8ae52a6eed0d6f88f37a0b7bffd6178'
BRANCH='research/observation-refiner-robust-pnp-20261009'
MAIN='f565418e6a63acd5fb920a5741308d84eb7c131f'


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],stderr=subprocess.PIPE)


def sha(b):return hashlib.sha256(b).hexdigest()


def protected():
    expected=json.loads(Path('/tmp/pallet-publication-copied-reader-next-20261011/ROOT_PROTECTED_BASELINE.json').read_text())['expected']
    cmds={'index':('ls-files','--stage','-z'),'diff':('diff','--no-ext-diff','--binary'),
        'staged':('diff','--cached','--no-ext-diff','--binary'),'status':('status','--porcelain=v1','-z','-uall')}
    actual={}
    for name,repo in [('original','/home/minjae/Documents/github/pallet-pose'),('source','/tmp/pallet-kp-publication-20261010-v6')]:
        actual[name]={'HEAD':git(repo,'rev-parse','HEAD').decode().strip()}|{k:sha(git(repo,*v)) for k,v in cmds.items()}
    assert actual==expected
    return actual


def remote():
    lines=git(STAGE,'ls-remote','origin','refs/heads/main','refs/heads/'+BRANCH).decode().splitlines()
    return {ref:commit for commit,ref in (line.split() for line in lines)}


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def copy(relative,src):
    dst=STAGE/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    assert dst.read_bytes()==src.read_bytes()


def main():
    before=protected();assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:PARENT}
    assert git(STAGE,'rev-parse','HEAD').decode().strip()==PARENT
    selected=(ROOT/'SELECTED_PATHS.nul').read_bytes().decode().split('\0')[:-1]
    assert sorted(selected)==sorted(git(STAGE,'diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    initial=ROOT/'initial_publication';initial.mkdir()
    archived=('PUBLICATION_PROTOCOL.json','PUBLICATION_STARTED.json','PLOT_CHECKS.json','ROOT_VISUAL_REVIEW.json')
    for name in archived:
        shutil.copy2(ROOT/name,initial/name)
        relative=DOC+'/initial_publication/'+name;copy(relative,initial/name);selected.append(relative)
    old=(ROOT/'ALL24_SAVED_ERRORS.svg').read_bytes();ET.fromstring(old)
    new=('\n'.join(line.rstrip() for line in old.decode().splitlines())+'\n').encode();ET.fromstring(new)
    assert old.split()==new.split()
    assert new != old and not any(line.rstrip()!=line for line in new.decode().splitlines())
    with (initial/'ALL24_SAVED_ERRORS.svg.gz').open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(old)
    relative=DOC+'/initial_publication/ALL24_SAVED_ERRORS.svg.gz';copy(relative,initial/'ALL24_SAVED_ERRORS.svg.gz');selected.append(relative)
    (ROOT/'ALL24_SAVED_ERRORS.svg').write_bytes(new);copy(DOC+'/ALL24_SAVED_ERRORS.svg',ROOT/'ALL24_SAVED_ERRORS.svg')
    repair=dict(schema='actual_generated_SVG_whitespace_repair_v23',original_sha256=sha(old),current_sha256=sha(new),
        only_trailing_whitespace_removed=True,non_whitespace_tokens_identical=True,both_XML_parse=True,
        PNG_unchanged=True,new_render=False,new_PnP=0,new_pose_accuracy_evaluations=0,
        initial_git_diff_check_exit=2,initial_failure='Matplotlib SVG path/glyph attributes have trailing whitespace',
        initial_publication_commit_created=False,initial_publication_push_performed=False)
    save(ROOT/'SVG_WHITESPACE_REPAIR.json',repair);copy(DOC+'/SVG_WHITESPACE_REPAIR.json',ROOT/'SVG_WHITESPACE_REPAIR.json');selected.append(DOC+'/SVG_WHITESPACE_REPAIR.json')
    for name in ('PLOT_CHECKS.json','ROOT_VISUAL_REVIEW.json'):
        r=json.loads((ROOT/name).read_text());key='SVG_sha256'
        assert r[key]==sha(old);r[key]=sha(new);r['original_render_SVG_sha256']=sha(old)
        r['SVG_post_render_whitespace_repair']='SVG_WHITESPACE_REPAIR.json'
        save(ROOT/name,r);copy(DOC+'/'+name,ROOT/name)
    check=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',str(STAGE/CODE/'check_saved_public.py'),str(STAGE/DOC)]))
    assert check['passed'];save(ROOT/'PUBLIC_DERIVED_CHECKS.json',check);copy(DOC+'/PUBLIC_DERIVED_CHECKS.json',ROOT/'PUBLIC_DERIVED_CHECKS.json')
    relative=CODE+'/finish_publication.py';copy(relative,Path(__file__));selected.append(relative)
    manifest=[dict(path=p,bytes=(STAGE/p).stat().st_size,sha256=sha((STAGE/p).read_bytes())) for p in selected if p!=DOC+'/PUBLICATION_PROTOCOL.json']
    save(ROOT/'PUBLICATION_PROTOCOL.json',dict(schema='explicit_saved_audit_publication_ready_v23',parent=PARENT,
        remote_main_before=MAIN,branch=BRANCH,protected_states=before,files=manifest,
        initial_publication_check_failure_preserved='initial_publication',own_protocol_not_in_self_hash_manifest=True,
        force_push=False,main_mutation=False,new_PnP=0,new_training_updates=0,new_pose_accuracy_evaluations=0,
        accuracy_goal_achieved=False))
    copy(DOC+'/PUBLICATION_PROTOCOL.json',ROOT/'PUBLICATION_PROTOCOL.json')
    spec=ROOT/'READY_SELECTED_PATHS.nul';assert not spec.exists()
    spec.write_bytes(b''.join(p.encode()+b'\0' for p in selected))
    assert protected()==before
    git(STAGE,'add','-f','--pathspec-from-file='+str(spec),'--pathspec-file-nul')
    assert sorted(selected)==sorted(git(STAGE,'diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    git(STAGE,'diff','--cached','--check')
    git(STAGE,'commit','-m','research: trace saved good-corner hypotheses and consensus failures')
    head=git(STAGE,'rev-parse','HEAD').decode().strip()
    assert sorted(selected)==sorted(git(STAGE,'diff-tree','--no-commit-id','--no-renames','--name-only','-r','-z',PARENT,head).decode().split('\0')[:-1])
    for item in manifest:assert sha(git(STAGE,'show',head+':'+item['path']))==item['sha256']
    assert protected()==before
    git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    after=remote();assert after=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:head}
    assert protected()==before
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    save(ROOT/'REMOTE_CONFIRMATION.json',dict(complete=True,passed=True,parent=PARENT,HEAD=head,remote=after,
        remote_SHA_equals_HEAD=True,remote_main_unchanged=True,original_SOURCE_protected=True,stage_index_empty=True,
        selected_files=len(selected),committed_manifest_blobs_verified=len(manifest),public_derived_reader_passed=True,
        original_git_diff_check_passed_after_SVG_whitespace_repair=True,accuracy_goal_achieved=False))
    print(json.dumps(dict(complete=True,HEAD=head,remote_verified=True,main_unchanged=True,files=len(selected))))


if __name__=='__main__':main()
