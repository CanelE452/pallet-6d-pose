"""Publish completed V27 artifacts only to the authorized research branch."""
import hashlib
import json
from pathlib import Path
import posixpath
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
STAGE=Path('/tmp/pallet-observation-publication-stage-20261011')
DOC='_docs/experiments/pallet_observation_uncertainty_20261011_v27'
CODE='scripts/research/pallet_observation_uncertainty_20261011_v27'
LATEST='_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md'
PARENT='c2d93dd34a7b1374c1863f4fa65eb85a406bf8b7'
BRANCH='research/observation-refiner-robust-pnp-20261009'
MAIN='1a2d9bbf0947d15c0cd9fe8c770f2522f022292e'


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
    private=json.loads((ROOT/'PRIVATE_RAW_READER_CHECKS.json').read_text())
    assert private['passed'] and private['raw_prediction_reference_extraction_verified']
    protocol=json.loads((ROOT/'output/PROTOCOL.json').read_text())
    parents=dict(real='_docs/experiments/pallet_point_admission_mechanism_20261011_v25/real_loo/ROWS.jsonl.gz',
        source='_docs/experiments/pallet_source_corner_semantics_20261011_v26/CORNER_ROWS.jsonl.gz',
        bounds='_docs/experiments/pallet_point_admission_mechanism_20261011_v25/bounds/ROWS.jsonl.gz',
        archive_manifest='_docs/experiments/pallet_whole_u_closure_20261011_v21/ARCHIVE_MANIFEST.json')
    for name,rel in parents.items():
        blob=git(STAGE,'show',PARENT+':'+rel);b=protocol['inputs'][name]
        assert sha(blob)==b['sha256'] and len(blob)==b['bytes'],name
    for name,b in protocol['inputs'].items():
        path=Path(b['path']);assert sha(path.read_bytes())==b['sha256'] and path.stat().st_size==b['bytes']
    assert private['code_sha256']==sha((ROOT/'check_public.py').read_bytes())
    assert json.loads((ROOT/'READER_PROTOCOL.json').read_text())['code_sha256']==private['code_sha256']
    figures=json.loads((ROOT/'figures/CHECKS.json').read_text());assert figures['passed']
    for n,b in figures['files'].items():assert sha((ROOT/'figures'/n).read_bytes())==b['sha256']
    assert json.loads((ROOT/'ROOT_VISUAL_CHECKS.json').read_text())['all_figures_viewed']
    files={}
    for n in ('PROTOCOL.json','SUMMARY.json','CHECKS.json','REFERENCE_FRAMES.jsonl.gz','REAL_ROWS.jsonl.gz',
        'SOURCE_ROWS.jsonl.gz','RESULT_KO.md','GOAL_STATUS.json'):files[DOC+'/'+n]=ROOT/'output'/n
    for n in ('PROTOCOL.json','CHECKS.json','ALL133_LOO_GAIN_AND_RADIUS.png','ALL13_BAD_SAVED_PIXEL_GEOMETRY.png'):
        files[DOC+'/figures/'+n]=ROOT/'figures'/n
    for n in ('READER_PROTOCOL.json','PRIVATE_RAW_READER_CHECKS.json','ROOT_VISUAL_CHECKS.json','SOURCE_REVIEW.json',
        'REMOTE_MAIN_OBSERVATION.json','REPORT_BUILD_CHECKS.json'):files[DOC+'/'+n]=ROOT/n
    for n in ('audit_saved_uncertainty.py','check_public.py','plot_saved_pixels.py','build_report.py','publish.py'):files[CODE+'/'+n]=ROOT/n
    previous_latest=(STAGE/LATEST).read_bytes()
    addition='\n추가 LOO/불확실성 감사: [전체133점 저장 LOO 참조오차·gain 분해 및573 source pair 반경 진단](../pallet_observation_uncertainty_20261011_v27/RESULT_KO.md). 악화53점도 LOO 양수 gain으로 통과했고 BAD13점 중10점은 LOO 참조오차도8px 초과였다. 자기 가림 제외 계약은 유지됐으며 후보 반경을 LOO 자세 불확실성으로 취급할 수 없다. 실제 raw 재검산과2개 전체 사례 그림을 포함했다. 새 PnP·투영·학습·실사 자세 재평가0, 정확도 목표는 미달이다.\n'
    p=ROOT/'LATEST_KO.md';p.write_bytes(previous_latest+addition.encode());files[LATEST]=p
    for rel,src in files.items():copy(src,rel)
    reader=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(STAGE/CODE/'check_public.py'),str(STAGE/DOC),'--checkout',str(STAGE)]))
    assert reader['passed'] and not reader['raw_prediction_reference_extraction_verified']
    save(ROOT/'PUBLIC_COMPACT_READER_CHECKS.json',reader);copy(ROOT/'PUBLIC_COMPACT_READER_CHECKS.json',DOC+'/PUBLIC_COMPACT_READER_CHECKS.json')
    files[DOC+'/PUBLIC_COMPACT_READER_CHECKS.json']=ROOT/'PUBLIC_COMPACT_READER_CHECKS.json'
    ledger=dict(passed=True,primary_saved_prediction_streams=1,primary_saved_prediction_frames=245,
        private_independent_raw_reader_streams=1,private_independent_raw_reader_frames=245,
        staged_compact_reader_streams_original_predictions=0,
        real_points_audited_per_pass=dict(primary=133,independent_raw_reader=133,staged_compact_reader=133),
        source_pairs_audited_per_pass=dict(primary=573,independent_raw_reader=573,staged_compact_reader=573),
        independent_distribution_cells_per_reader=378,original_predictions_not_restored_this_turn=True,
        fresh_clone=False,new_inference=0,new_fits=0,new_PnP=0,new_projection=0,new_pose_accuracy=0,
        new_training_updates=0,new_RGB=0,new_real_fullpath_runtime_measurements=0,thresholds_changed=False,
        actual_ROOT_figures_viewed=2,accuracy_goal_achieved=False)
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
        previous_v26_main_different_movement_recorded=True,protected_states=before,files=manifest,
        all_parent_inputs_match_committed_Git_blobs=True,audit_code_hash_preserved=True,
        original_raw_matches_existing_committed_archive_manifest=True,raw_reader_and_compact_reader_scopes_distinguished=True,
        force_push=False,main_mutation=False,latest_previous_content_preserved=True,
        own_protocol_not_in_self_hash_manifest=True,accuracy_goal_achieved=False))
    copy(ROOT/'PUBLICATION_PROTOCOL.json',DOC+'/PUBLICATION_PROTOCOL.json');files[DOC+'/PUBLICATION_PROTOCOL.json']=ROOT/'PUBLICATION_PROTOCOL.json'
    assert protected()==before
    selected=list(files);spec=ROOT/'SELECTED_PATHS.nul';spec.write_bytes(b''.join(s.encode()+b'\0' for s in selected))
    git(STAGE,'add','-f','--pathspec-from-file='+str(spec),'--pathspec-file-nul')
    assert sorted(selected)==sorted(git(STAGE,'diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    git(STAGE,'diff','--cached','--check')
    git(STAGE,'commit','-m','research: audit saved LOO validation error and observation uncertainty')
    head=git(STAGE,'rev-parse','HEAD').decode().strip()
    assert sorted(selected)==sorted(git(STAGE,'diff-tree','--no-commit-id','--no-renames','--name-only','-r','-z',PARENT,head).decode().split('\0')[:-1])
    for rel,p in files.items():assert git(STAGE,'show',head+':'+rel)==p.read_bytes()
    assert protected()==before
    git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    actual=remote();assert actual=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:head}
    assert protected()==before
    receipt=dict(passed=True,complete=True,published_audit_commit=head,remote=actual,
        remote_SHA_equals_published_commit=True,main_SHA_unchanged_during_this_publication=True,
        main_changed_since_previous_v26_record_before_this_turn=True,main_pushed_by_this_audit=False,
        original_SOURCE_protected=True,selected_files=len(selected),all_committed_files_byte_verified=True,
        private_raw_reader_passed=True,staged_compact_reader_passed=True,
        new_PnP=0,new_projection=0,new_pose_accuracy=0,new_training=0,accuracy_goal_achieved=False)
    save(ROOT/'REMOTE_CONFIRMATION.json',receipt);copy(ROOT/'REMOTE_CONFIRMATION.json',DOC+'/REMOTE_CONFIRMATION.json')
    git(STAGE,'add','-f','--',DOC+'/REMOTE_CONFIRMATION.json')
    assert git(STAGE,'diff','--cached','--name-only').decode().strip()==DOC+'/REMOTE_CONFIRMATION.json'
    git(STAGE,'diff','--cached','--check');git(STAGE,'commit','-m','research: record verified LOO uncertainty audit publication')
    sealed=git(STAGE,'rev-parse','HEAD').decode().strip();git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:sealed} and protected()==before
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    save(ROOT/'FINAL_REMOTE_CHECKS.json',dict(passed=True,complete=True,HEAD=sealed,audit_commit=head,
        remote_SHA_equals_HEAD=True,remote_main=MAIN,main_pushed_by_this_audit=False,
        main_unchanged_during_this_publication=True,original_SOURCE_protected=True,accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True,HEAD=sealed,audit_commit=head,files=len(selected),remote_verified=True,
        main_pushed_by_this_audit=False,main_unchanged_during_this_publication=True,accuracy_goal_achieved=False)))


if __name__=='__main__':main()
