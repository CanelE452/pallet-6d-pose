"""Publish completed frozen root audit, preserving failed audit/figure versions."""
import importlib.util
import json
from pathlib import Path
import posixpath
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
STAGE=Path('/tmp/pallet-observation-publication-stage-20261011')
DOC='_docs/experiments/pallet_loo_root_envelope_20261011_v28'
CODE='scripts/research/pallet_loo_root_envelope_20261011_v28'
LATEST='_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md'
PARENT='9451d533132961f54a0c543db7907f92ede8ffa7'
BRANCH='research/observation-refiner-robust-pnp-20261009'
MAIN='1a2d9bbf0947d15c0cd9fe8c770f2522f022292e'
HELPER=Path('/tmp/pallet-observation-uncertainty-private-20261011-v27/publish.py')
spec=importlib.util.spec_from_file_location('immutable_v27_publication_helpers',HELPER)
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
git,sha,protected,remote=H.git,H.sha,H.protected,H.remote


def save(p,r):
    with p.open('x') as f:json.dump(r,f,indent=2,ensure_ascii=False);f.write('\n')


def copy(src,rel):
    p=STAGE/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p);assert p.read_bytes()==src.read_bytes()


def main():
    assert HELPER.read_bytes()==git(STAGE,'show',PARENT+':scripts/research/pallet_observation_uncertainty_20261011_v27/publish.py')
    before=protected();assert git(STAGE,'rev-parse','HEAD').decode().strip()==PARENT
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:PARENT}
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    assert not (STAGE/DOC).exists() and not (STAGE/CODE).exists()
    raw_reader=json.loads((ROOT/'PRIVATE_RAW_READER_RESULT.json').read_text())
    assert raw_reader['exit_code']==0 and raw_reader['checks']['original_raw_root_extraction_verified']
    protocol=json.loads((ROOT/'output_corrected/PROTOCOL.json').read_text())
    rels=dict(v27_real='_docs/experiments/pallet_observation_uncertainty_20261011_v27/REAL_ROWS.jsonl.gz',
        archive_manifest='_docs/experiments/pallet_whole_u_closure_20261011_v21/ARCHIVE_MANIFEST.json',
        selection_source='scripts/research/pallet_cornerwise_independent_20261010_v4/selection.py',
        pose_source='scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py')
    for name,rel in rels.items():
        b=protocol['inputs'][name];blob=git(STAGE,'show',PARENT+':'+rel)
        assert sha(blob)==b['sha256'] and len(blob)==b['bytes']
    for name,b in protocol['inputs'].items():
        p=Path(b['path']);assert sha(p.read_bytes())==b['sha256'] and p.stat().st_size==b['bytes']
    initial=json.loads((ROOT/'output/PROTOCOL.json').read_text())
    assert sha((ROOT/'audit_saved_roots_initial.py').read_bytes())==initial['inputs']['code']['sha256']
    assert raw_reader['code_sha256']==sha((ROOT/'check_public.py').read_bytes())
    for directory in ('figures','figures_readable'):
        checks=json.loads((ROOT/directory/'CHECKS.json').read_text());assert checks['passed']
        assert sha((ROOT/directory/checks['file']['path']).read_bytes())==checks['file']['sha256']
    assert json.loads((ROOT/'ROOT_VISUAL_CHECKS.json').read_text())['readable_figure_viewed']
    files={}
    for n in ('PROTOCOL.json','SUMMARY.json','CHECKS.json','FRAME_ROWS.jsonl.gz','DECISION_ROWS.jsonl.gz','RESULT_KO.md','GOAL_STATUS.json'):
        files[DOC+'/'+n]=ROOT/'output_corrected'/n
    files[DOC+'/initial_failure/PROTOCOL.json']=ROOT/'output/PROTOCOL.json'
    files[DOC+'/initial_failure/INITIAL_FAILURE.json']=ROOT/'INITIAL_FAILURE.json'
    for directory,target in [('figures','initial_figure'),('figures_readable','figures')]:
        for n in ('PROTOCOL.json','CHECKS.json','ALL133_ROOT_PREFERENCE_ENVELOPES.png'):
            files[DOC+'/'+target+'/'+n]=ROOT/directory/n
    for n in ('READER_PROTOCOL.json','PRIVATE_RAW_READER_RESULT.json','VISUAL_REPAIR_PROTOCOL.json','ROOT_VISUAL_CHECKS.json','REPORT_BUILD_CHECKS.json'):
        files[DOC+'/'+n]=ROOT/n
    for n in ('audit_saved_roots.py','audit_saved_roots_initial.py','check_public.py','plot_saved_roots.py',
        'plot_saved_roots_readable.py','build_report.py','publish.py'):files[CODE+'/'+n]=ROOT/n
    previous=(STAGE/LATEST).read_bytes()
    text='\n추가 저장 LOO 다중해 감사: [전체423결정·12178 root의 GT-free 선호 범위](../pallet_loo_root_envelope_20261011_v28/RESULT_KO.md). 채택133점 중69점은 최대 inlier 수의 다른 해가 N3를 선호했으며, 그 해들이 모두 후보를 선호한64점 중22점도 실제 코너 좌표가 악화됐다. 여러 해의 동의만으로 해결했다고 표시하지 않았다. 전체 원행·독립 raw 검산·그림·최초 감사기 실패 기록까지 보존했다. 새 PnP·투영·학습·성능 재실험0, 정확도 목표는 미달이다.\n'
    p=ROOT/'LATEST_KO.md';p.write_bytes(previous+text.encode());files[LATEST]=p
    for rel,p in files.items():copy(p,rel)
    reader=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(STAGE/CODE/'check_public.py'),str(STAGE/DOC),'--checkout',str(STAGE)]))
    assert reader['passed'] and not reader['original_raw_root_extraction_verified']
    save(ROOT/'PUBLIC_COMPACT_READER_CHECKS.json',reader);copy(ROOT/'PUBLIC_COMPACT_READER_CHECKS.json',DOC+'/PUBLIC_COMPACT_READER_CHECKS.json')
    files[DOC+'/PUBLIC_COMPACT_READER_CHECKS.json']=ROOT/'PUBLIC_COMPACT_READER_CHECKS.json'
    ledger=dict(passed=True,initial_audit_attempts_failed=1,corrected_primary_completed=1,
        initial_failure_reason='duplicate serialized best diagnostics; unique-entry assumption corrected',
        all_saved_decisions_per_complete_audit=423,all_saved_root_entries_per_complete_audit=12178,
        private_independent_raw_reader_completed=1,staged_independent_compact_reader_completed=1,
        original_prediction_streams_completed=2,completed_original_prediction_frame_rows=490,
        schema_probe_and_initial_failed_partial_read_not_counted_as_complete_passes=True,
        ROOT_figures_viewed=2,initial_and_readable_renders=2,new_inference=0,new_fits=0,new_PnP=0,
        new_projection=0,new_training_updates=0,new_RGB=0,new_pose_accuracy=0,new_real_fullpath_runtime_measurements=0,
        deployment_policy_changed=False,root_pools_changed_after_results=False,accuracy_goal_achieved=False)
    save(ROOT/'EXECUTION_LEDGER.json',ledger);copy(ROOT/'EXECUTION_LEDGER.json',DOC+'/EXECUTION_LEDGER.json');files[DOC+'/EXECUTION_LEDGER.json']=ROOT/'EXECUTION_LEDGER.json'
    links=[]
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',(ROOT/'output_corrected/RESULT_KO.md').read_text()):
        if '://' in target:continue
        rel=posixpath.normpath(DOC+'/'+target.split('#',1)[0]);assert not rel.startswith('../')
        if rel in files:assert (STAGE/rel).is_file()
        else:git(STAGE,'cat-file','-e',PARENT+':'+rel)
        links.append(dict(target=target,resolved=rel))
    save(ROOT/'PUBLIC_LINK_CHECKS.json',dict(passed=True,links=links));copy(ROOT/'PUBLIC_LINK_CHECKS.json',DOC+'/PUBLIC_LINK_CHECKS.json');files[DOC+'/PUBLIC_LINK_CHECKS.json']=ROOT/'PUBLIC_LINK_CHECKS.json'
    manifest=[dict(path=rel,bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for rel,p in files.items()]
    save(ROOT/'PUBLICATION_PROTOCOL.json',dict(parent=PARENT,branch=BRANCH,remote_main_before=MAIN,
        protected_states=before,files=manifest,publication_helper_matches_parent_committed_code=True,
        publication_helper_sha256=sha(HELPER.read_bytes()),all_inputs_match_parent_Git_blobs=True,
        corrected_primary_code_hash_preserved=True,failed_initial_code_protocol_receipt_preserved=True,
        all_root_entries_and_matching_best_indices_preserved=True,initial_and_corrected_visual_versions_preserved=True,
        raw_and_compact_readers_scopes_distinguished=True,force_push=False,main_mutation=False,
        latest_previous_content_preserved=True,own_protocol_not_in_self_hash_manifest=True,accuracy_goal_achieved=False))
    copy(ROOT/'PUBLICATION_PROTOCOL.json',DOC+'/PUBLICATION_PROTOCOL.json');files[DOC+'/PUBLICATION_PROTOCOL.json']=ROOT/'PUBLICATION_PROTOCOL.json'
    assert protected()==before
    selected=list(files);spec=ROOT/'SELECTED_PATHS.nul';spec.write_bytes(b''.join(s.encode()+b'\0' for s in selected))
    git(STAGE,'add','-f','--pathspec-from-file='+str(spec),'--pathspec-file-nul')
    assert sorted(selected)==sorted(git(STAGE,'diff','--cached','--name-only','-z').decode().split('\0')[:-1])
    git(STAGE,'diff','--cached','--check');git(STAGE,'commit','-m','research: audit all saved LOO root preference envelopes')
    head=git(STAGE,'rev-parse','HEAD').decode().strip()
    assert sorted(selected)==sorted(git(STAGE,'diff-tree','--no-commit-id','--no-renames','--name-only','-r','-z',PARENT,head).decode().split('\0')[:-1])
    for rel,p in files.items():assert git(STAGE,'show',head+':'+rel)==p.read_bytes()
    assert protected()==before;git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    actual=remote();assert actual=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:head} and protected()==before
    receipt=dict(passed=True,complete=True,published_audit_commit=head,remote=actual,
        remote_SHA_equals_published_commit=True,main_pushed_by_this_audit=False,
        main_SHA_unchanged_during_this_publication=True,original_SOURCE_protected=True,
        selected_files=len(selected),all_committed_files_byte_verified=True,private_raw_reader_passed=True,
        staged_compact_reader_passed=True,all423_decisions_and12178_entries_retained=True,
        initial_failed_audit_and_visual_versions_preserved=True,new_PnP=0,new_projection=0,new_training=0,
        new_pose_accuracy=0,accuracy_goal_achieved=False)
    save(ROOT/'REMOTE_CONFIRMATION.json',receipt);copy(ROOT/'REMOTE_CONFIRMATION.json',DOC+'/REMOTE_CONFIRMATION.json')
    git(STAGE,'add','-f','--',DOC+'/REMOTE_CONFIRMATION.json')
    assert git(STAGE,'diff','--cached','--name-only').decode().strip()==DOC+'/REMOTE_CONFIRMATION.json'
    git(STAGE,'diff','--cached','--check');git(STAGE,'commit','-m','research: record verified saved LOO root audit publication')
    sealed=git(STAGE,'rev-parse','HEAD').decode().strip();git(STAGE,'push','origin','HEAD:refs/heads/'+BRANCH)
    assert remote()=={'refs/heads/main':MAIN,'refs/heads/'+BRANCH:sealed} and protected()==before
    assert not git(STAGE,'diff','--cached','--name-only','-z') and not git(STAGE,'diff','--name-only','-z')
    save(ROOT/'FINAL_REMOTE_CHECKS.json',dict(passed=True,complete=True,HEAD=sealed,audit_commit=head,
        remote_SHA_equals_HEAD=True,remote_main=MAIN,main_pushed_by_this_audit=False,
        main_unchanged_during_this_publication=True,original_SOURCE_protected=True,accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True,HEAD=sealed,audit_commit=head,files=len(selected),remote_verified=True,
        main_pushed_by_this_audit=False,accuracy_goal_achieved=False)))


if __name__=='__main__':main()
