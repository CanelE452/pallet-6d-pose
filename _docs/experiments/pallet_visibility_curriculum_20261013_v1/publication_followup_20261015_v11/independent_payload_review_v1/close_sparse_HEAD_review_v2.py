"""Close only two sparse-worktree availability assumptions in preserved V1 QA."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

ROOT = Path('/home/minjae/Documents/github/pallet-pose')
WT = Path('/tmp/pallet-visibility-main-publication-20261015-v1')
OUT = Path(__file__).parent
PUB = OUT.parent
NAMES = (
    'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
    'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    result_path = OUT / 'REVIEW_V2.json'
    if result_path.exists():
        raise FileExistsError('Preserve the completed review')
    previous_path = OUT / 'REVIEW.json'
    previous = json.loads(previous_path.read_text())
    assert sha(previous_path.read_bytes()) == '9212a84bc93f91c657e99da473599bdd7739dc157661f28a4bab1dbf63670d1b'
    expected_failed = {'HEAD_reuse_tracked_and_WT_exact:' + name for name in NAMES}
    assert {x['name'] for x in previous['failures']} == expected_failed
    assert len(previous['checks_detail']) == 4075
    assert all(x['passed'] for x in previous['checks_detail'] if x['name'] not in expected_failed)
    manifest_path = PUB / 'CLOSED_PAYLOAD_MANIFEST_V1.json'
    validation_path = PUB / 'PAYLOAD_VALIDATION_V1.json'
    assert sha(manifest_path.read_bytes()) == previous['manifest_sha256']
    assert sha(validation_path.read_bytes()) == previous['validation_sha256']
    manifest = json.loads(manifest_path.read_text())
    sources = {r['path']: r for r in manifest['source_publication_status']['rows']}
    corrected = []
    for name in NAMES:
        row = sources[name]
        head = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=WT)
        index = subprocess.check_output(['git', 'show', ':' + name], cwd=WT)
        tracked = subprocess.check_output(['git', 'ls-files', '-t', '--', name], cwd=WT).decode().strip()
        assert tracked and sha(head) == sha(index) == sha((ROOT / name).read_bytes()) == row['sha256'] == row['published_HEAD_sha256']
        corrected.append(dict(path=name, sha256=row['sha256'], HEAD_exact=True, index_exact=True,
                              current_source_exact=True, worktree_materialized=(WT / name).is_file(),
                              git_index_flag=tracked.split()[0]))
    result = dict(
        schema='INDEPENDENT_V11_CLOSED_PUBLICATION_PAYLOAD_REVIEW_V2',
        status='PASS', UTC=datetime.now(timezone.utc).isoformat(), failure_count=0, failures=[],
        checks=4075, unchanged_prior_checks_reused=4073, corrected_sparse_HEAD_checks=2,
        no_additional_distinct_test_claim=True,
        correction='V1 wrongly required every previously published source to be materialized in a sparse worktree. Both missing working files are present in Git HEAD and index with exact source hashes. Publication artifacts required no correction.',
        prior_review_sha256=sha(previous_path.read_bytes()), corrected_records=corrected,
        manifest_sha256=previous['manifest_sha256'], validation_sha256=previous['validation_sha256'],
        paths=509, manifest_rows=507, manifest_bytes=91670780,
        image_files=132, HTML_files=12, source_AST_files=59, local_links=723,
        maximum_file_bytes=1764277, source_closure=previous['source_closure'],
        raw_exception_count=23, protected6_excluded_and_unchanged=True,
        manual_report_review=previous['manual_report_review'], limits=previous['limits'],
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__).read_bytes())},
        input_sha256={str(previous_path.relative_to(ROOT)):sha(previous_path.read_bytes()),
                      str(manifest_path.relative_to(ROOT)):sha(manifest_path.read_bytes()),
                      str(validation_path.relative_to(ROOT)):sha(validation_path.read_bytes())},
    )
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    seal = dict(receipt_sha256=sha(result_path.read_bytes()), source_sha256=result['source_sha256'],
                input_sha256=result['input_sha256'])
    (OUT / 'REVIEW_SEAL_V2.json').write_text(json.dumps(seal, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', checks=4075, failure_count=0,
                          receipt=str(result_path.relative_to(ROOT)), sha256=seal['receipt_sha256']), ensure_ascii=False))

if __name__ == '__main__':
    main()
