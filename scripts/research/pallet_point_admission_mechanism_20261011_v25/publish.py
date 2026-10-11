"""Publish only completed V25 saved audits on the authorized research branch."""
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
STAGE=Path('/tmp/pallet-observation-publication-stage-20261011')
DOC='_docs/experiments/pallet_point_admission_mechanism_20261011_v25'
CODE='scripts/research/pallet_point_admission_mechanism_20261011_v25'
LATEST='_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md'
PARENT='c1d80d29265729901d571b9d5ffcf57ab3970327'
BRANCH='research/observation-refiner-robust-pnp-20261009'
MAIN='f565418e6a63acd5fb920a5741308d84eb7c131f'


def sha(b):return hashlib.sha256(b).hexdigest()


def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE)


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def protected():
    expected=json.loads(Path('/tmp/pallet-publication-copied-reader-next-20261011/ROOT_PROTECTED_BASELINE.json').read_text())['expected']
    cmds=dict(index=('ls-files','--stage','-z'),diff=('diff','--no-ext-diff','--binary'),
              staged=('diff','--cached','--no-ext-diff','--binary'),status=('status','--porcelain=v1','-z','-uall'))
    actual={}
    for name,path in [('original','/home/minjae/Documents/github/pallet-pose'),('source','/tmp/pallet-kp-publication-20261010-v6')]:
        actual[name]=dict(HEAD=git(path,'rev-parse','HEAD').decode().strip(),**{k:sha(git(path,*v)) for k,v in cmds.items()})
    assert actual==expected
    return actual


def remote():
    return {ref:commit for commit,ref in (s.split() for s in git(STAGE,'ls-remote','origin','refs/heads/main','refs/heads/'+BRANCH).decode().splitlines())}


def copy(src,relative):
    p=STAGE/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
    assert p.read_bytes()==src.read_bytes()


def main():
    before=protected();assert git(STAGE,'rev-parse','HEAD').decode().strip()==PARENT
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:PARENT}
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    assert not (STAGE/DOC).exists() and not (STAGE/CODE).exists()
    for name in ('ROOT_FINAL_DERIVED_CHECKS.json','PUBLIC_SOURCE_REPLAY_CHECKS.json','PUBLIC_REAL_REPLAY_CHECKS.json'):
        assert json.loads((ROOT/name).read_text())['passed']
    source_fixture=json.loads((ROOT/'PUBLIC_SOURCE_REPLAY_PROTOCOL.json').read_text())
    for b in source_fixture['files']:
        if b.get('source_commit'):
            blob=git(STAGE,'show',PARENT+':'+b['path']);assert len(blob)==b['bytes'] and sha(blob)==b['sha256']
    real_fixture=json.loads((ROOT/'PUBLIC_REAL_FIXTURE_PROTOCOL.json').read_text())
    for b in real_fixture['files']:
        if b['provenance']=='exact parent Git blob':
            blob=git(STAGE,'show',PARENT+':'+b['path']);assert len(blob)==b['bytes'] and sha(blob)==b['sha256']
    files={}
    expected={'source':('PROTOCOL.json','SUMMARY.json','CHECKS.json','LINE_ROWS.jsonl.gz','CORNER_ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz'),
        'real_loo':('PROTOCOL.json','SUMMARY.json','CHECKS.json','ROWS.jsonl.gz'),
        'bounds':('PROTOCOL.json','SUMMARY.json','ROWS.jsonl.gz'),
        'plot':('PROTOCOL.json','CHECKS.json','ALL133_SAVED_ADOPTED_POINTS.png'),
        'source_plot':('PROTOCOL.json','CHECKS.json','ALL4_SOURCE_BOUNDARY_AND_CORNER.png')}
    for folder,names in expected.items():
        original='output' if folder=='source' else folder
        for name in names:files[DOC+'/'+folder+'/'+name]=ROOT/original/name
    for name in ('RESULT_KO.md','GOAL_STATUS.json','ROOT_VISUAL_REVIEW.json','SOURCE_REVIEW.json',
        'PRIVATE_PUBLIC_DERIVED_CHECKS.json','PUBLIC_SOURCE_REPLAY_PROTOCOL.json','PUBLIC_SOURCE_REPLAY_CHECKS.json',
        'PUBLIC_REAL_FIXTURE_PROTOCOL.json','PUBLIC_REAL_REPLAY_CHECKS.json','ROOT_FINAL_DERIVED_CHECKS.json','REPORT_EDITORIAL_CHECKS.json'):
        files[DOC+'/'+name]=ROOT/name
    for name in ('audit_saved_mechanism.py','audit_saved_real_loo.py','derive_validation_bounds.py',
        'plot_saved_real.py','check_public.py','build_report.py','replay_public_source.py','replay_public_real.py',
        'prepare_public.py','prepare_real_replay.py','plot_saved_source.py','finalize_editorial.py','publish.py'):
        files[CODE+'/'+name]=ROOT/name
    original_latest=(STAGE/LATEST).read_bytes()
    addition='\n추가 실제 관측 승인 검산: [기존 radius gate의 3/4 제외와 실사 133점 LOO·최종 inlier 추적](../pallet_point_admission_mechanism_20261011_v25/RESULT_KO.md). 잘못된 실사 채택점 13개가 모두 최종 fit/inlier에 남은 기록을 검산했다. 공개 파생 reader와 원 saved join의 경로만 옮긴 reader를 실제 실행했다. 새 학습·LOO/PnP·자세 성능 재실험·설정 변경은 0이며 정확도 목표는 아직 미달이다.\n'
    latest=ROOT/'LATEST_KO.md';latest.write_bytes(original_latest+addition.encode());files[LATEST]=latest
    manifest=[dict(path=k,bytes=v.stat().st_size,sha256=sha(v.read_bytes())) for k,v in files.items()]
    save(ROOT/'PUBLICATION_PROTOCOL.json',dict(schema='explicit_saved_point_admission_and_real_LOO_publication_v25',
        parent=PARENT,branch=BRANCH,remote_main_before=MAIN,protected_states=before,files=manifest,
        source_fixture_parent_Git_blobs_verified=True,real_small_parent_Git_blobs_verified=True,
        geometry_matches_existing_archive_manifest=True,new_archive_restore=False,
        fresh_clone=False,latest_previous_content_preserved=True,force_push=False,main_mutation=False,
        new_head=0,new_line_fit=0,new_LOO=0,new_PnP=0,new_pose_scoring=0,new_training=0,
        accuracy_goal_achieved=False,own_protocol_not_in_self_hash_manifest=True))
    files[DOC+'/PUBLICATION_PROTOCOL.json']=ROOT/'PUBLICATION_PROTOCOL.json'
    for relative,src in files.items():copy(src,relative)
    checked=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(STAGE/CODE/'check_public.py'),str(STAGE/DOC)]))
    assert checked['passed'];save(ROOT/'PUBLIC_DERIVED_CHECKS.json',checked)
    copy(ROOT/'PUBLIC_DERIVED_CHECKS.json',DOC+'/PUBLIC_DERIVED_CHECKS.json');files[DOC+'/PUBLIC_DERIVED_CHECKS.json']=ROOT/'PUBLIC_DERIVED_CHECKS.json'
    links=[]
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',(ROOT/'RESULT_KO.md').read_text()):
        if '://' in target:continue
        relative=posixpath.normpath(DOC+'/'+target.split('#',1)[0]);assert not relative.startswith('../')
        if relative in files:assert (STAGE/relative).is_file()
        else:git(STAGE,'cat-file','-e',PARENT+':'+relative)
        links.append(dict(target=target,resolved=relative))
    save(ROOT/'PUBLIC_LINK_CHECKS.json',dict(passed=True,all_report_links_resolved=True,links=links))
    copy(ROOT/'PUBLIC_LINK_CHECKS.json',DOC+'/PUBLIC_LINK_CHECKS.json');files[DOC+'/PUBLIC_LINK_CHECKS.json']=ROOT/'PUBLIC_LINK_CHECKS.json'
    assert protected()==before
    selected=list(files);spec=ROOT/'SELECTED_PATHS.nul';spec.write_bytes(b''.join(s.encode()+b'\0' for s in selected))
    git(STAGE,'add','-f','--pathspec-from-file='+str(spec),'--pathspec-file-nul')
    assert sorted(selected)==sorted(git(STAGE,'diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    git(STAGE,'diff','--cached','--check')
    git(STAGE,'commit','-m','research: audit point admission and all adopted real LOO observations')
    head=git(STAGE,'rev-parse','HEAD').decode().strip()
    assert sorted(selected)==sorted(git(STAGE,'diff-tree','--no-commit-id','--no-renames','--name-only','-r','-z',PARENT,head).decode().split('\0')[:-1])
    for relative,src in files.items():assert sha(git(STAGE,'show',head+':'+relative))==sha(src.read_bytes())
    assert protected()==before
    git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    after=remote();assert after=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:head}
    assert protected()==before
    receipt=dict(complete=True,passed=True,published_audit_commit=head,remote=after,
        remote_SHA_equals_published_commit=True,main_unchanged=True,original_SOURCE_protected=True,
        selected_files=len(selected),all_committed_files_byte_verified=True,
        public_derived_reader_passed=True,source_join_replay_passed=True,real_join_replay_passed=True,
        new_PnP=0,new_pose_accuracy=0,new_training=0,accuracy_goal_achieved=False)
    save(ROOT/'REMOTE_CONFIRMATION.json',receipt);copy(ROOT/'REMOTE_CONFIRMATION.json',DOC+'/REMOTE_CONFIRMATION.json')
    git(STAGE,'add','-f','--',DOC+'/REMOTE_CONFIRMATION.json')
    assert git(STAGE,'diff','--cached','--name-only').decode().strip()==DOC+'/REMOTE_CONFIRMATION.json'
    git(STAGE,'diff','--cached','--check')
    git(STAGE,'commit','-m','research: record verified point-admission audit publication')
    sealed=git(STAGE,'rev-parse','HEAD').decode().strip()
    git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:sealed}
    assert protected()==before
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    save(ROOT/'FINAL_REMOTE_CHECKS.json',dict(complete=True,passed=True,HEAD=sealed,audit_commit=head,
        remote_SHA_equals_HEAD=True,main_unchanged=True,original_SOURCE_protected=True,
        published_receipt_verified=True,accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True,HEAD=sealed,audit_commit=head,remote_verified=True,main_unchanged=True,files=len(selected))))


if __name__=='__main__':main()
