"""Publish the completed source semantic audit; no main change or force push."""
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
STAGE=Path('/tmp/pallet-observation-publication-stage-20261011')
DOC='_docs/experiments/pallet_source_corner_semantics_20261011_v26'
CODE='scripts/research/pallet_source_corner_semantics_20261011_v26'
LATEST='_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md'
PARENT='942d162d0657eeb7c18d53946e9b1971220ce490'
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


def remote():return {ref:commit for commit,ref in (s.split() for s in git(STAGE,'ls-remote','origin','refs/heads/main','refs/heads/'+BRANCH).decode().splitlines())}


def copy(src,rel):
    p=STAGE/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p);assert p.read_bytes()==src.read_bytes()


def main():
    before=protected();assert git(STAGE,'rev-parse','HEAD').decode().strip()==PARENT
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:PARENT}
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    assert not (STAGE/DOC).exists() and not (STAGE/CODE).exists()
    assert json.loads((ROOT/'PRIVATE_PUBLIC_CHECKS.json').read_text())['passed']
    protocol=json.loads((ROOT/'output/PROTOCOL.json').read_text())
    for key,b in protocol['inputs'].items():
        if key=='code':continue
        if key=='saved_corners':rel='_docs/experiments/pallet_point_admission_mechanism_20261011_v25/source/CORNER_ROWS.jsonl.gz'
        else:rel=b['path'].split('/pallet-kp-publication-20261010-v6/',1)[1]
        blob=git(STAGE,'show',PARENT+':'+rel);assert sha(blob)==b['sha256'] and len(blob)==b['bytes'],key
    assert sha((ROOT/'audit_source_reference.py').read_bytes())==protocol['inputs']['code']['sha256']
    files={}
    for name in ('PROTOCOL.json','SUMMARY.json','CHECKS.json','FRAME_ROWS.jsonl.gz','CORNER_ROWS.jsonl.gz',
        'RESULT_KO.md','EXECUTION_UNIT_CLARIFICATION.json','SOURCE_REVIEW.json','GOAL_STATUS.json'):
        files[DOC+'/'+name]=ROOT/'output'/name
    for name in ('PUBLIC_READER_INITIAL_LAYOUT_FAILURE.json','PUBLIC_FIXTURE_PROTOCOL.json','PRIVATE_PUBLIC_CHECKS.json'):
        files[DOC+'/'+name]=ROOT/name
    for name in ('audit_source_reference.py','check_public.py','build_report.py','prepare_public.py','publish.py'):
        files[CODE+'/'+name]=ROOT/name
    previous_latest=(STAGE/LATEST).read_bytes()
    addition='\n추가 source 의미 검산: [573개 actual-wire virtual reference와 같은 canonical corner ID의 기존 합성 투영 비교](../pallet_source_corner_semantics_20261011_v26/RESULT_KO.md). 검사한 source에서 참조 정의 차이는 최대0.000033px로8~19px 예측 오차를 설명하지 못했다. 새 학습·PnP·실사 성능 재평가는0이며 source 기존자세 순방향 투영128프레임/1024 scalar 호출은 별도로 기록했다. 정확도 목표는 아직 미달이다.\n'
    p=ROOT/'LATEST_KO.md';p.write_bytes(previous_latest+addition.encode());files[LATEST]=p
    for rel,src in files.items():copy(src,rel)
    reader=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(STAGE/CODE/'check_public.py'),str(STAGE/DOC),'--checkout',str(ROOT/'git_blob_reader_fixture')]))
    assert reader['passed'];save(ROOT/'PUBLIC_CHECKS.json',reader);copy(ROOT/'PUBLIC_CHECKS.json',DOC+'/PUBLIC_CHECKS.json');files[DOC+'/PUBLIC_CHECKS.json']=ROOT/'PUBLIC_CHECKS.json'
    ledger=dict(passed=True,source_known_Rt_frame_operations=128,source_scalar_projection_calls=1024,
        initial_sparse_reader_failure_scalar_projection_calls=0,private_fixture_reader_scalar_projection_calls=1024,
        staged_public_reader_scalar_projection_calls=1024,total_actual_scalar_projection_calls=3072,
        new_pose_estimations=0,new_line_fits=0,new_models=0,new_training=0,new_RGB=0,new_real_pose_accuracy=0)
    save(ROOT/'EXECUTION_LEDGER.json',ledger);copy(ROOT/'EXECUTION_LEDGER.json',DOC+'/EXECUTION_LEDGER.json');files[DOC+'/EXECUTION_LEDGER.json']=ROOT/'EXECUTION_LEDGER.json'
    links=[]
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',(ROOT/'output/RESULT_KO.md').read_text()):
        if '://' in target:continue
        rel=posixpath.normpath(DOC+'/'+target.split('#',1)[0]);assert not rel.startswith('../')
        if rel in files:assert (STAGE/rel).is_file()
        else:git(STAGE,'cat-file','-e',PARENT+':'+rel)
        links.append(dict(target=target,resolved=rel))
    save(ROOT/'PUBLIC_LINK_CHECKS.json',dict(passed=True,links=links));copy(ROOT/'PUBLIC_LINK_CHECKS.json',DOC+'/PUBLIC_LINK_CHECKS.json');files[DOC+'/PUBLIC_LINK_CHECKS.json']=ROOT/'PUBLIC_LINK_CHECKS.json'
    manifest=[dict(path=rel,bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for rel,p in files.items()]
    save(ROOT/'PUBLICATION_PROTOCOL.json',dict(parent=PARENT,branch=BRANCH,remote_main_before=MAIN,
        protected_states=before,files=manifest,all_original_upstream_inputs_match_parent_Git_blobs=True,
        original_audit_code_byte_hash_preserved=True,fixture_fresh_clone=False,source_labels_originals_preserved=True,
        projection_count_units_explicit=True,force_push=False,main_mutation=False,latest_previous_content_preserved=True,
        own_protocol_not_in_self_hash_manifest=True,accuracy_goal_achieved=False))
    copy(ROOT/'PUBLICATION_PROTOCOL.json',DOC+'/PUBLICATION_PROTOCOL.json');files[DOC+'/PUBLICATION_PROTOCOL.json']=ROOT/'PUBLICATION_PROTOCOL.json'
    assert protected()==before
    selected=list(files);spec=ROOT/'SELECTED_PATHS.nul';spec.write_bytes(b''.join(s.encode()+b'\0' for s in selected))
    git(STAGE,'add','-f','--pathspec-from-file='+str(spec),'--pathspec-file-nul')
    assert sorted(selected)==sorted(git(STAGE,'diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    git(STAGE,'diff','--cached','--check')
    git(STAGE,'commit','-m','research: verify virtual boundary references against canonical source corners')
    head=git(STAGE,'rev-parse','HEAD').decode().strip()
    assert sorted(selected)==sorted(git(STAGE,'diff-tree','--no-commit-id','--no-renames','--name-only','-r','-z',PARENT,head).decode().split('\0')[:-1])
    for rel,p in files.items():assert git(STAGE,'show',head+':'+rel)==p.read_bytes()
    assert protected()==before
    git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    actual=remote();assert actual=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:head}
    assert protected()==before
    receipt=dict(passed=True,complete=True,published_audit_commit=head,remote=actual,
        remote_SHA_equals_published_commit=True,main_unchanged=True,original_SOURCE_protected=True,
        selected_files=len(selected),all_committed_files_byte_verified=True,public_reader_passed=True,
        source_scalar_projection_calls=1024,total_with_two_readers_scalar_projection_calls=3072,
        new_PnP=0,new_real_pose_accuracy=0,new_training=0,accuracy_goal_achieved=False)
    save(ROOT/'REMOTE_CONFIRMATION.json',receipt);copy(ROOT/'REMOTE_CONFIRMATION.json',DOC+'/REMOTE_CONFIRMATION.json')
    git(STAGE,'add','-f','--',DOC+'/REMOTE_CONFIRMATION.json')
    assert git(STAGE,'diff','--cached','--name-only').decode().strip()==DOC+'/REMOTE_CONFIRMATION.json'
    git(STAGE,'diff','--cached','--check');git(STAGE,'commit','-m','research: record verified source semantic audit publication')
    sealed=git(STAGE,'rev-parse','HEAD').decode().strip();git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:sealed} and protected()==before
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    save(ROOT/'FINAL_REMOTE_CHECKS.json',dict(passed=True,complete=True,HEAD=sealed,audit_commit=head,
        remote_SHA_equals_HEAD=True,main_unchanged=True,original_SOURCE_protected=True,accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True,HEAD=sealed,audit_commit=head,files=len(selected),remote_verified=True,main_unchanged=True)))


if __name__=='__main__':main()
