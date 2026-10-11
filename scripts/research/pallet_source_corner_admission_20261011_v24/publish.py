"""Publish the completed audit on the authorized dedicated branch only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
STAGE=Path('/tmp/pallet-observation-publication-stage-20261011')
DOC='_docs/experiments/pallet_source_corner_admission_20261011_v24'
CODE='scripts/research/pallet_source_corner_admission_20261011_v24'
LATEST='_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md'
PARENT='c3b4df8ab3cbc8eb84cfd9be2a8263c3e7f60709'
BRANCH='research/observation-refiner-robust-pnp-20261009'
MAIN='f565418e6a63acd5fb920a5741308d84eb7c131f'


def sha(b):return hashlib.sha256(b).hexdigest()


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE)


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
    assert not (STAGE/DOC).exists()
    for folder in ('output','decoder'):
        protocol=json.loads((ROOT/folder/'PROTOCOL.json').read_text())
        for relative,b in protocol['inputs'].items():
            blob=git(STAGE,'show',PARENT+':'+relative)
            assert len(blob)==b['bytes'] and sha(blob)==b['sha256'],relative
    files={}
    for name in ('PROTOCOL.json','SUMMARY.json','CHECKS.json','ROWS.jsonl.gz','ROWS_STORAGE_CHECKS.json',
                 'PLOT_PROTOCOL.json','PLOT_CHECKS.json','ALL573_SAVED_CANDIDATES.png','ROOT_VISUAL_REVIEW.json','SOURCE_REVIEW.json'):
        files[DOC+'/'+name]=ROOT/'output'/name
    files[DOC+'/NECESSARY_ONLY_REPORT_KO.md']=ROOT/'output/RESULT_KO.md'
    for name in ('RESULT_KO.md','GOAL_STATUS.json'):files[DOC+'/'+name]=ROOT/'final'/name
    for name in ('PROTOCOL.json','SUMMARY.json','CHECKS.json','DECISIONS.jsonl.gz','DECODED_FRAMES.jsonl.gz',
                 'SOURCE_RAW_HW.json','CALIBRATION_GEOMETRY_ONLY.json','CALIBRATION_IMAGE_NO_ROLE.json','CALIBRATION_IMAGE_ROLE.json'):
        files[DOC+'/decoder/'+name]=ROOT/'decoder'/name
    for name in ('saved_admission_audit.py','replay_saved_decoder.py','plot_saved_candidates.py',
                 'check_saved_public.py','check_decoder_public.py','prepare_report.py','finalize_report.py','publish.py'):
        files[CODE+'/'+name]=ROOT/name
    original_latest=(STAGE/LATEST).read_bytes()
    addition='\n추가 코너 승인 검산: [합성 573후보 → 고정 decoder 검산과 실사 선택 문제](../pallet_source_corner_admission_20261011_v24/RESULT_KO.md). 미확정이었던 source 4개가 실제로 선택됨을 확인했다. 새 학습·모델 추론·PnP·자세 성능 실험은 0이며, 기존 선·교점 계산을 검산했다. 최종 self-occ fit 제외·재투영 계약은 유지하고 정확도 목표는 아직 미달이다.\n'
    latest=ROOT/'LATEST_KO.md';latest.write_bytes(original_latest+addition.encode());files[LATEST]=latest
    manifest=[dict(path=k,bytes=v.stat().st_size,sha256=sha(v.read_bytes())) for k,v in files.items()]
    save(ROOT/'PUBLICATION_PROTOCOL.json',dict(schema='explicit_saved_CAL_decoder_audit_publication_v24',
        parent=PARENT,branch=BRANCH,remote_main_before=MAIN,protected_states=before,files=manifest,
        all_upstream_inputs_match_parent_Git_blobs=True,latest_previous_content_preserved=True,
        force_push=False,main_mutation=False,new_head=0,new_PnP=0,new_pose_scoring=0,new_training=0,
        accuracy_goal_achieved=False,own_protocol_not_in_self_hash_manifest=True))
    files[DOC+'/PUBLICATION_PROTOCOL.json']=ROOT/'PUBLICATION_PROTOCOL.json'
    for relative,src in files.items():copy(src,relative)
    for script,folder,name in [('check_saved_public.py',STAGE/DOC,'PUBLIC_DERIVED_CHECKS.json'),
                               ('check_decoder_public.py',STAGE/DOC/'decoder','PUBLIC_DECODER_CHECKS.json')]:
        result=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',str(STAGE/CODE/script),str(folder)]))
        assert result['passed'];save(ROOT/name,result);copy(ROOT/name,DOC+'/'+name);files[DOC+'/'+name]=ROOT/name
    assert protected()==before
    selected=list(files);spec=ROOT/'SELECTED_PATHS.nul';spec.write_bytes(b''.join(s.encode()+b'\0' for s in selected))
    git(STAGE,'add','-f','--pathspec-from-file='+str(spec),'--pathspec-file-nul')
    assert sorted(selected)==sorted(git(STAGE,'diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    git(STAGE,'diff','--cached','--check')
    git(STAGE,'commit','-m','research: verify saved boundary decoder admission and corner errors')
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
        public_derived_reader_passed=True,public_decoder_reader_passed=True,accuracy_goal_achieved=False)
    save(ROOT/'REMOTE_CONFIRMATION.json',receipt);copy(ROOT/'REMOTE_CONFIRMATION.json',DOC+'/REMOTE_CONFIRMATION.json')
    git(STAGE,'add','-f','--',DOC+'/REMOTE_CONFIRMATION.json')
    assert git(STAGE,'diff','--cached','--name-only').decode().strip()==DOC+'/REMOTE_CONFIRMATION.json'
    git(STAGE,'diff','--cached','--check')
    git(STAGE,'commit','-m','research: record verified dedicated-branch decoder audit publication')
    sealed=git(STAGE,'rev-parse','HEAD').decode().strip()
    git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:sealed}
    assert protected()==before
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    save(ROOT/'FINAL_REMOTE_CHECKS.json',dict(complete=True,passed=True,HEAD=sealed,audit_commit=head,
        remote_SHA_equals_HEAD=True,main_unchanged=True,original_SOURCE_protected=True,
        published_receipt_verified=True,accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True,HEAD=sealed,audit_commit=head,remote_verified=True,main_unchanged=True,files=len(selected))))


if __name__=='__main__':
    main()
