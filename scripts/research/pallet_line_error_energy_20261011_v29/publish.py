"""One-use publication of completed V29 audit, with protected working copies."""
import importlib.util
import json
from pathlib import Path
import posixpath
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
STAGE = Path('/tmp/pallet-observation-publication-stage-20261011')
DOC = '_docs/experiments/pallet_line_error_energy_20261011_v29'
CODE = 'scripts/research/pallet_line_error_energy_20261011_v29'
LATEST = '_docs/experiments/pallet_observation_refiner_status_20261011/LATEST_KO.md'
PARENT = '0c210572582057a8e614e899dbf41205f97ec897'
MAIN = '1a2d9bbf0947d15c0cd9fe8c770f2522f022292e'
BRANCH = 'research/observation-refiner-robust-pnp-20261009'
HELPER = Path('/tmp/pallet-observation-uncertainty-private-20261011-v27/publish.py')
spec = importlib.util.spec_from_file_location('immutable_v27_publication_helpers', HELPER)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
git, sha, protected, remote = H.git, H.sha, H.protected, H.remote


def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def copy(src, relative):
    dest = STAGE/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    assert dest.read_bytes() == src.read_bytes()


def main():
    assert HELPER.read_bytes() == git(STAGE, 'show', PARENT+':scripts/research/pallet_observation_uncertainty_20261011_v27/publish.py')
    before = protected()
    assert git(STAGE, 'rev-parse', 'HEAD').decode().strip() == PARENT
    assert remote() == {'refs/heads/main': MAIN, 'refs/heads/'+BRANCH: PARENT}
    assert not git(STAGE, 'diff', '--cached', '--name-only', '-z')
    assert not git(STAGE, 'diff', '--name-only', '-z')
    assert not (STAGE/DOC).exists() and not (STAGE/CODE).exists()
    protocol = json.loads((ROOT/'output/PROTOCOL.json').read_text())
    for name, entry in protocol['inputs'].items():
        p = Path(entry['path'])
        assert p.stat().st_size == entry['bytes'] and sha(p.read_bytes()) == entry['sha256']
    reader_protocol = json.loads((ROOT/'READER_PROTOCOL.json').read_text())
    private = json.loads((ROOT/'PRIVATE_READER_RESULT.json').read_text())
    assert private['exit_code'] == 0 and private['checks']['passed']
    assert private['code_sha256'] == reader_protocol['code_sha256'] == sha((ROOT/'check_public.py').read_bytes())
    assert private['checks']['all_upstream_Git_blobs_SHA_verified']
    assert private['checks']['line_rows'] == 1302 and private['checks']['certified_POS_lines'] == 1194
    figure = json.loads((ROOT/'figures/CHECKS.json').read_text())
    assert figure['passed'] and figure['all1194_rows_plotted']
    for n, b in figure['files'].items():
        assert sha((ROOT/'figures'/n).read_bytes()) == b['sha256']
    visual = json.loads((ROOT/'ROOT_VISUAL_CHECKS.json').read_text())
    assert visual['actual_ROOT_view_image'] and visual['all1194_inside_plot_axes'] and visual['report_whole_88_lines_ROOT_read']
    report = json.loads((ROOT/'REPORT_BUILD_CHECKS.json').read_text())
    assert sha((ROOT/'output/RESULT_KO.md').read_bytes()) == report['sha256']
    files = {}
    for n in ('PROTOCOL.json', 'SUMMARY.json', 'CHECKS.json', 'LINE_ROWS.jsonl.gz', 'FRAME_ROWS.jsonl.gz', 'RESULT_KO.md', 'audit_saved_line_energy.py'):
        files[DOC+'/'+n] = ROOT/'output'/n
    for n in ('READER_PROTOCOL.json', 'PRIVATE_READER_RESULT.json', 'SOURCE_REVIEW.json', 'ROOT_VISUAL_CHECKS.json', 'REPORT_BUILD_CHECKS.json', 'GOAL_STATUS.json'):
        files[DOC+'/'+n] = ROOT/n
    for n in ('PROTOCOL.json', 'CHECKS.json', 'ALL1194_LINE_ERROR_ENERGY.png', 'ALL1194_LINE_ERROR_ENERGY.svg'):
        files[DOC+'/figures/'+n] = ROOT/'figures'/n
    for n in ('audit_saved_line_energy.py', 'check_public.py', 'plot_saved_energy.py', 'build_report.py', 'publish.py'):
        files[CODE+'/'+n] = ROOT/n
    assert (ROOT/'audit_saved_line_energy.py').read_bytes() == (ROOT/'output/audit_saved_line_energy.py').read_bytes()
    previous = (STAGE/LATEST).read_bytes()
    addition = '\n추가 경계 관측 감사: [전체1302 선·1194 POS 선의 실제 normal 오차 에너지](../pallet_line_error_energy_20261011_v29/RESULT_KO.md). 모든 POS 선에서 실제 normal 오차가 TLS 내부 오차보다 컸다. 선 전체가 함께 치우치는 평균 성분의 총에너지 비율은 기하85.15%/영상·역할 없음75.11%/영상·역할 포함72.18%였다. LOO 동의나 작은 선 잔차가 정확한 관측을 인증하지 않는 문제를 별도로 기록했다. 새 학습·PnP·투영·성능 재실험0이며 정확도 개선은 미검증이다. 모든 원행과 Git SHA를 확인하는 독립 stdlib reader, 그림·이미지·코드 위치를 포함했다.\n'
    latest = ROOT/'LATEST_KO.md'
    latest.write_bytes(previous+addition.encode())
    files[LATEST] = latest
    for rel, src in files.items(): copy(src, rel)
    argv = ['/home/minjae/anaconda3/bin/python3', '-I', '-S', '-B', str(STAGE/CODE/'check_public.py'), str(STAGE/DOC), '--checkout', str(STAGE)]
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    checked = json.loads(result.stdout)
    assert checked['passed'] and checked['all_upstream_Git_blobs_SHA_verified']
    save(ROOT/'PUBLIC_READER_RESULT.json', dict(command_argv=argv, exit_code=result.returncode, stdout=result.stdout,
        stderr=result.stderr, checks=checked, actual_ROOT_execution=True, code_sha256=reader_protocol['code_sha256']))
    copy(ROOT/'PUBLIC_READER_RESULT.json', DOC+'/PUBLIC_READER_RESULT.json')
    files[DOC+'/PUBLIC_READER_RESULT.json'] = ROOT/'PUBLIC_READER_RESULT.json'
    save(ROOT/'EXECUTION_LEDGER.json', dict(passed=True, primary_complete_audits=1, private_independent_readers=1,
        staged_independent_readers=1, per_complete_pass=dict(arm_frames=384, lines=1302, all_POS_lines=1194, undefined_POS_lines=108),
        complete_total_arm_frame_reads=1152, complete_total_line_reads=3906, complete_total_POS_energy_checks=3582,
        independent_distribution_cells_per_reader=147, independently_verified_large_cases_per_reader=4,
        one_scientific_figure_two_export_formats=True, actual_ROOT_figure_views=1,
        new_head_inference=0, new_line_fits=0, new_PnP=0, new_projection=0, new_training_updates=0,
        new_RGB=0, new_pose_accuracy=0, new_fullpath_runtime_measurements=0,
        deployment_policy_changed=False, statistical_query_correlation_estimated=False,
        source_CAL_same_selection_not_independent_coverage=True, accuracy_goal_achieved=False))
    copy(ROOT/'EXECUTION_LEDGER.json', DOC+'/EXECUTION_LEDGER.json')
    files[DOC+'/EXECUTION_LEDGER.json'] = ROOT/'EXECUTION_LEDGER.json'
    links = []
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', (ROOT/'output/RESULT_KO.md').read_text()):
        if '://' in target: continue
        rel = posixpath.normpath(DOC+'/'+target.split('#', 1)[0])
        assert not rel.startswith('../')
        if rel == DOC+'/REMOTE_CONFIRMATION.json': status = 'post_push_receipt_written_after_remote_verification'
        elif rel in files:
            assert (STAGE/rel).is_file()
            status = 'copied_byte_verified'
        else:
            git(STAGE, 'cat-file', '-e', PARENT+':'+rel)
            status = 'existing_parent_blob_verified'
        links.append(dict(target=target, resolved=rel, status=status))
    save(ROOT/'PUBLIC_LINK_CHECKS.json', dict(passed=True, links=links))
    copy(ROOT/'PUBLIC_LINK_CHECKS.json', DOC+'/PUBLIC_LINK_CHECKS.json')
    files[DOC+'/PUBLIC_LINK_CHECKS.json'] = ROOT/'PUBLIC_LINK_CHECKS.json'
    manifest = [dict(path=rel, bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for rel, p in files.items()]
    save(ROOT/'PUBLICATION_PROTOCOL.json', dict(parent=PARENT, branch=BRANCH, remote_main_before=MAIN,
        protected_states=before, files=manifest, all_upstream_Git_blobs_verified_by_both_independent_readers=True,
        primary_code_hash_preserved=True, primary_code_duplicate_in_doc_is_byte_identical=True,
        publication_helper_matches_parent_committed_code=True, publication_helper_sha256=sha(HELPER.read_bytes()),
        selected_files_only=True, original_source_unmodified=True, latest_previous_content_preserved=True,
        own_protocol_not_in_self_hash_manifest=True, normal_push=True, force_push=False, main_mutation=False,
        accuracy_goal_achieved=False))
    copy(ROOT/'PUBLICATION_PROTOCOL.json', DOC+'/PUBLICATION_PROTOCOL.json')
    files[DOC+'/PUBLICATION_PROTOCOL.json'] = ROOT/'PUBLICATION_PROTOCOL.json'
    assert protected() == before
    selected = list(files)
    specpath = ROOT/'SELECTED_PATHS.nul'
    specpath.write_bytes(b''.join(p.encode()+b'\0' for p in selected))
    git(STAGE, 'add', '-f', '--pathspec-from-file='+str(specpath), '--pathspec-file-nul')
    assert sorted(selected) == sorted(git(STAGE, 'diff', '--cached', '--name-only', '-z').decode().split('\0')[:-1])
    git(STAGE, 'diff', '--cached', '--check')
    git(STAGE, 'commit', '-m', 'research: audit saved boundary normal error energy and uncertainty limits')
    head = git(STAGE, 'rev-parse', 'HEAD').decode().strip()
    assert sorted(selected) == sorted(git(STAGE, 'diff-tree', '--no-commit-id', '--no-renames', '--name-only', '-r', '-z', PARENT, head).decode().split('\0')[:-1])
    for rel, p in files.items(): assert git(STAGE, 'show', head+':'+rel) == p.read_bytes()
    assert protected() == before
    git(STAGE, 'push', 'origin', 'HEAD:refs/heads/'+BRANCH)
    actual = remote()
    assert actual == {'refs/heads/main': MAIN, 'refs/heads/'+BRANCH: head} and protected() == before
    save(ROOT/'REMOTE_CONFIRMATION.json', dict(passed=True, complete=True, published_audit_commit=head, remote=actual,
        remote_SHA_equals_published_commit=True, all_committed_files_byte_verified=True, selected_files=len(selected),
        private_and_public_independent_readers_passed=True, original_SOURCE_protected=True,
        main_pushed_by_this_audit=False, main_SHA_unchanged_during_this_publication=True,
        new_PnP=0, new_projection=0, new_training=0, new_pose_accuracy=0, accuracy_goal_achieved=False))
    copy(ROOT/'REMOTE_CONFIRMATION.json', DOC+'/REMOTE_CONFIRMATION.json')
    git(STAGE, 'add', '-f', '--', DOC+'/REMOTE_CONFIRMATION.json')
    assert git(STAGE, 'diff', '--cached', '--name-only').decode().strip() == DOC+'/REMOTE_CONFIRMATION.json'
    git(STAGE, 'diff', '--cached', '--check')
    git(STAGE, 'commit', '-m', 'research: record verified boundary error energy audit publication')
    sealed = git(STAGE, 'rev-parse', 'HEAD').decode().strip()
    git(STAGE, 'push', 'origin', 'HEAD:refs/heads/'+BRANCH)
    assert remote() == {'refs/heads/main': MAIN, 'refs/heads/'+BRANCH: sealed} and protected() == before
    assert git(STAGE, 'show', sealed+':'+DOC+'/REMOTE_CONFIRMATION.json') == (ROOT/'REMOTE_CONFIRMATION.json').read_bytes()
    assert not git(STAGE, 'diff', '--cached', '--name-only', '-z') and not git(STAGE, 'diff', '--name-only', '-z')
    save(ROOT/'FINAL_REMOTE_CHECKS.json', dict(passed=True, HEAD=sealed, audit_commit=head, remote_SHA_equals_HEAD=True,
        remote_main=MAIN, main_pushed_by_this_audit=False, original_SOURCE_protected=True, accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True, HEAD=sealed, audit_commit=head, selected_files=len(selected),
        remote_verified=True, main_pushed_by_this_audit=False, accuracy_goal_achieved=False)))


if __name__ == '__main__': main()
