"""Resume publication after SVG-only whitespace check failure; no science rerun."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_initial_v29_publisher', ROOT/'publish.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
git, sha, protected, remote = P.git, P.sha, P.protected, P.remote
STAGE, DOC, CODE, PARENT, MAIN, BRANCH = P.STAGE, P.DOC, P.CODE, P.PARENT, P.MAIN, P.BRANCH


def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def binding(b): return dict(bytes=len(b), sha256=sha(b))


def xml_semantics(b):
    def node(e):
        attrs = {k: ' '.join(v.split()) for k, v in e.attrib.items()}
        return (e.tag, attrs, ' '.join((e.text or '').split()), [node(c) for c in e])
    return node(ET.fromstring(b))


def main():
    initial = json.loads((ROOT/'PUBLICATION_PROTOCOL.json').read_text())
    before = protected()
    assert before == initial['protected_states']
    assert git(STAGE, 'rev-parse', 'HEAD').decode().strip() == PARENT
    assert remote() == {'refs/heads/main': MAIN, 'refs/heads/'+BRANCH: PARENT}
    files = {r['path']: r for r in initial['files']}
    initial_protocol = (ROOT/'PUBLICATION_PROTOCOL.json').read_bytes()
    files[DOC+'/PUBLICATION_PROTOCOL.json'] = dict(path=DOC+'/PUBLICATION_PROTOCOL.json', **binding(initial_protocol))
    selected = git(STAGE, 'diff', '--cached', '--name-only', '-z').decode().split('\0')[:-1]
    assert sorted(selected) == sorted(files)
    for rel, entry in files.items():
        blob = git(STAGE, 'show', ':'+rel)
        assert binding(blob) == {k: entry[k] for k in ('bytes', 'sha256')}
        assert (STAGE/rel).read_bytes() == blob
    check = subprocess.run(['git', '-C', str(STAGE), 'diff', '--cached', '--check'], capture_output=True)
    assert check.returncode == 2
    svgrel = DOC+'/figures/ALL1194_LINE_ERROR_ENERGY.svg'
    headers = [s for s in check.stdout.decode().splitlines() if ': trailing whitespace.' in s]
    assert len(headers) == 3235 and all(s.startswith(svgrel+':') for s in headers)
    assert not check.stderr
    original = (ROOT/'figures/ALL1194_LINE_ERROR_ENERGY.svg').read_bytes()
    assert original == (STAGE/svgrel).read_bytes()
    cleaned = b'\n'.join(s.rstrip(b' \t') for s in original.splitlines())+b'\n'
    assert xml_semantics(original) == xml_semantics(cleaned)
    additions = {}
    def add_bytes(rel, data):
        dest = STAGE/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        dest.write_bytes(data)
        additions[rel] = dict(path=rel, **binding(data))
    add_bytes(DOC+'/preflight/original_svg.svg.gz', gzip.compress(original, mtime=0))
    add_bytes(DOC+'/preflight/GIT_DIFF_CHECK_STDOUT.txt.gz', gzip.compress(check.stdout, mtime=0))
    add_bytes(DOC+'/preflight/FIGURE_CHECKS_ORIGINAL.json', (ROOT/'figures/CHECKS.json').read_bytes())
    add_bytes(CODE+'/complete_staged_publication.py', Path(__file__).read_bytes())
    failure = dict(actual_ROOT_initial_publisher_exit_code=1, stage='git diff --cached --check before any commit or push',
        git_diff_check_exit_code=check.returncode, SVG_trailing_whitespace_lines=len(headers), other_errors=0,
        initial_publisher_code=binding((ROOT/'publish.py').read_bytes()), original_SVG=binding(original),
        cleaned_SVG=binding(cleaned), XML_semantics_equal=True, correction='Only line-end spaces/tabs removed; all SVG tags, attributes, text and coordinates normalized-semantic identical.',
        original_SVG_gzip_preserved=True, primary_reader_plot_report_not_rerun=True,
        no_science_inputs_policy_or_results_changed=True, new_PnP=0, new_projection=0, new_training=0)
    save(ROOT/'PUBLICATION_PREFLIGHT_FAILURE.json', failure)
    add_bytes(DOC+'/PUBLICATION_PREFLIGHT_FAILURE.json', (ROOT/'PUBLICATION_PREFLIGHT_FAILURE.json').read_bytes())
    (STAGE/svgrel).write_bytes(cleaned)
    files[svgrel] = dict(path=svgrel, **binding(cleaned))
    fc = json.loads((ROOT/'figures/CHECKS.json').read_text())
    fc['files']['ALL1194_LINE_ERROR_ENERGY.svg'] = binding(cleaned)
    fc['SVG_trailing_whitespace_normalized_after_preserved_initial_export'] = True
    fc['original_SVG_gzip'] = 'preflight/original_svg.svg.gz'
    fbytes = (json.dumps(fc, indent=2)+'\n').encode()
    rel = DOC+'/figures/CHECKS.json'
    (STAGE/rel).write_bytes(fbytes)
    files[rel] = dict(path=rel, **binding(fbytes))
    assert gzip.decompress((STAGE/DOC/'preflight/original_svg.svg.gz').read_bytes()) == original
    assert sha((STAGE/DOC/'figures/ALL1194_LINE_ERROR_ENERGY.png').read_bytes()) == fc['files']['ALL1194_LINE_ERROR_ENERGY.png']['sha256']
    files.update(additions)
    resume = dict(parent=PARENT, branch=BRANCH, remote_main_before=MAIN, protected_states=before,
        initial_publication_protocol_sha256=sha(initial_protocol), initial_explicit_staged_files=len(selected),
        files=list(files.values()), only_public_SVG_and_its_check_binding_changed=True,
        SVG_XML_semantics_equal=True, preserved_original_svg_and_failed_check_stdout=True,
        independent_scientific_checks_already_passed_and_unchanged=True,
        report_code_primary_protocol_reader_figure_PNG_unchanged=True,
        one_use_resume_no_initial_publisher_rerun=True, force_push=False, main_mutation=False,
        accuracy_goal_achieved=False)
    save(ROOT/'PUBLICATION_RESUME_PROTOCOL.json', resume)
    add_bytes(DOC+'/PUBLICATION_RESUME_PROTOCOL.json', (ROOT/'PUBLICATION_RESUME_PROTOCOL.json').read_bytes())
    files[DOC+'/PUBLICATION_RESUME_PROTOCOL.json'] = additions[DOC+'/PUBLICATION_RESUME_PROTOCOL.json']
    selected = list(files)
    specs = ROOT/'RESUME_SELECTED_PATHS.nul'
    specs.write_bytes(b''.join(p.encode()+b'\0' for p in selected))
    assert protected() == before
    git(STAGE, 'add', '-f', '--pathspec-from-file='+str(specs), '--pathspec-file-nul')
    assert sorted(selected) == sorted(git(STAGE, 'diff', '--cached', '--name-only', '-z').decode().split('\0')[:-1])
    git(STAGE, 'diff', '--cached', '--check')
    git(STAGE, 'commit', '-m', 'research: audit saved boundary normal error energy and uncertainty limits')
    head = git(STAGE, 'rev-parse', 'HEAD').decode().strip()
    assert sorted(selected) == sorted(git(STAGE, 'diff-tree', '--no-commit-id', '--no-renames', '--name-only', '-r', '-z', PARENT, head).decode().split('\0')[:-1])
    for rel, entry in files.items():
        assert binding(git(STAGE, 'show', head+':'+rel)) == {k: entry[k] for k in ('bytes', 'sha256')}
    assert protected() == before
    git(STAGE, 'push', 'origin', 'HEAD:refs/heads/'+BRANCH)
    actual = remote()
    assert actual == {'refs/heads/main': MAIN, 'refs/heads/'+BRANCH: head} and protected() == before
    save(ROOT/'REMOTE_CONFIRMATION.json', dict(passed=True, complete=True, published_audit_commit=head, remote=actual,
        remote_SHA_equals_published_commit=True, all_committed_files_byte_verified=True, selected_files=len(selected),
        private_and_public_independent_readers_passed=True, original_SOURCE_protected=True,
        initial_SVG_whitespace_failure_preserved_and_semantic_equal_repair=True,
        main_pushed_by_this_audit=False, main_SHA_unchanged_during_this_publication=True,
        new_PnP=0, new_projection=0, new_training=0, new_pose_accuracy=0, accuracy_goal_achieved=False))
    receipt_rel = DOC+'/REMOTE_CONFIRMATION.json'
    assert not (STAGE/receipt_rel).exists()
    shutil.copy2(ROOT/'REMOTE_CONFIRMATION.json', STAGE/receipt_rel)
    git(STAGE, 'add', '-f', '--', receipt_rel)
    assert git(STAGE, 'diff', '--cached', '--name-only').decode().strip() == receipt_rel
    git(STAGE, 'diff', '--cached', '--check')
    git(STAGE, 'commit', '-m', 'research: record verified boundary error energy audit publication')
    sealed = git(STAGE, 'rev-parse', 'HEAD').decode().strip()
    git(STAGE, 'push', 'origin', 'HEAD:refs/heads/'+BRANCH)
    assert remote() == {'refs/heads/main': MAIN, 'refs/heads/'+BRANCH: sealed} and protected() == before
    assert git(STAGE, 'show', sealed+':'+receipt_rel) == (ROOT/'REMOTE_CONFIRMATION.json').read_bytes()
    assert not git(STAGE, 'diff', '--cached', '--name-only', '-z') and not git(STAGE, 'diff', '--name-only', '-z')
    save(ROOT/'FINAL_REMOTE_CHECKS.json', dict(passed=True, HEAD=sealed, audit_commit=head,
        remote_SHA_equals_HEAD=True, remote_main=MAIN, main_pushed_by_this_audit=False,
        original_SOURCE_protected=True, accuracy_goal_achieved=False))
    print(json.dumps(dict(passed=True, HEAD=sealed, audit_commit=head, selected_files=len(selected),
        remote_verified=True, main_pushed_by_this_audit=False, accuracy_goal_achieved=False)))


if __name__ == '__main__': main()
