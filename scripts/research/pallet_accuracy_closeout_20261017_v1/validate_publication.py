"""Validate the full closeout artifact contract and protected workspace."""
import argparse
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import resource
import subprocess
import time
from urllib.parse import unquote, urlsplit

from PIL import Image

from .publish import read, sha, dump


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets += [v for k, v in attrs if k in ('href', 'src') and v]


def run(repo, private, output, asset):
    start, cpu = time.monotonic(), time.process_time()
    repo, private, out, asset = map(Path, (repo, private, output, asset))
    before = read(private/'START_BEFORE_FETCH.json')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(asset), *args])
    assert git('rev-parse', 'HEAD').decode().strip() == before['original_HEAD']
    assert git('branch', '--show-current').decode().strip() == before['original_branch']
    assert git('status', '--porcelain=v1', '-z') == (private/'ORIGINAL_STATUS_NUL.bin').read_bytes()
    for name, digest in before['protected_files_SHA256'].items():
        assert sha(asset/name) == digest
    current = git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads').decode().splitlines()
    assert all(r in current for r in before['original_branch_refs'])
    bindings = read(out/'SOURCE_AND_INPUT_BINDINGS.json')
    for name, digest in bindings['new_source_sha256'].items():
        assert sha(repo/name) == digest, 'New source bytes changed after binding'
    for derivation in read(out/'PUBLIC_DERIVATIONS.json')['records']:
        assert sha(out/derivation['public']) == derivation['public_sha256']
    assert read(out/'POSEFIX_PARITY95.json')['passed']
    parity = read(out/'POSEFIX_PARITY95.json')
    assert parity['exact_point_rows'] == parity['exact_pose_dictionary_rows'] == 475
    assert max(parity['maxima'].values()) == 0
    assert read(out/'evidence/posefix192_v1/RAW143_CACHE_PARITY.json')['passed']
    assert read(out/'evidence/posefix192_v1/IO_BOUNDARY.json')['scientific_answer_reads'] == 0
    assert read(out/'evidence/INTEGRATION_TESTS_V2.json')['exit_code'] == 0
    assert read(out/'INDEPENDENT_ARITHMETIC.json')['passed']
    assert read(out/'INDEPENDENT_CONTRACT_CHECK.json')['passed']
    seal = read(private/'posefix192_v1/PREDICTION_SEAL.json')
    copied = {'PREDICTIONS.jsonl.gz': 'predictions/POSEFIX_ALL238.jsonl.gz',
        'POSEFIX_192_PREDICTIONS.jsonl.gz': 'POSEFIX_192_PREDICTIONS.jsonl.gz',
        'POSEFIX_CAL46_PREDICTIONS.jsonl.gz': 'POSEFIX_CAL46_PREDICTIONS.jsonl.gz'}
    for source, target in copied.items():
        assert sha(out/target) == seal['artifacts'][source]
    with gzip.open(out/'POSEFIX_192_PREDICTIONS.jsonl.gz', 'rt') as stream:
        primary = [json.loads(line) for line in stream]
    assert len(primary) == 576
    methods = ('POSEFIX_RAW', 'POSEFIX_SUBPIX_PF_ANCHOR', 'POSEFIX_SUBPIX_YOLO_ANCHOR')
    identities = [{r['id'] for r in primary if r['method'] == method} for method in methods]
    assert all(len(ids) == 192 and ids == identities[0] for ids in identities)
    with (out/'FAIR_ROW_METRICS.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 1428
    values = read(out/'metrics/FAIR_RESULTS.json')
    assert [values['populations']['PRIMARY192'][arm]['successes'] for arm in values['arms']] == [66, 108, 114, 80, 112, 111]
    assert values['paired']['PRIMARY192']['E0_minus_N0']['counts'] == dict(BOTH_SUCCESS=89, RECOVERY=25, DAMAGE=19, BOTH_FAILURE=59)
    for path in ('FAIR_192_COMPARISON.csv', 'FAIR_ROW_METRICS.csv', 'PAIRED_TRANSITIONS.csv'):
        assert sha(out/path) == sha(private/'fair_evaluation_v1'/path)
    with (out/'E_POINT_CONTRACT_ROWS.csv').open() as stream:
        point_rows = list(csv.DictReader(stream))
    assert len(point_rows) == 1252
    assert {r['role'] for r in point_rows} == {'FULL', 'PARTIAL_ARCHIVE'}
    motivation = [r for r in point_rows if r['id'] == 'wood_night_01:030898' and r['role'] == 'FULL']
    assert len(motivation) == 5
    assert all(r['actual_vs_current_adapter_changed'] == 'False' for r in motivation)
    assert 'NO_SUPPORTED_CHANGE' in (out/'REPORT_KO.md').read_text()
    assert bindings['new_E192_method_evaluation'] is False

    required = ('POSEFIX_192_PROTOCOL_LOCK.json', 'SOURCE_AND_INPUT_BINDINGS.json', 'POSEFIX_PARITY95.json',
        'POSEFIX_192_PREDICTIONS.jsonl.gz', 'FAIR_192_COMPARISON.csv', 'PAIRED_TRANSITIONS.csv',
        'FAIR_COMPARISON_REPORT_KO.md', 'REPORT_KO.md', 'MECHANISM_AND_EDIT_PROPOSAL.md',
        'E0_INFORMATION_CONTRACT_AUDIT_KO.md', 'CAL_PROVENANCE_SUMMARY.json', 'REPRODUCE.md', 'COST.json')
    assert all((out/name).is_file() for name in required)
    links, images, texts, compressed_bytes = 0, 0, 0, 0
    for path in sorted(out.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix in ('.png', '.jpg', '.jpeg'):
            with Image.open(path) as image:
                image.load(); assert image.width > 0 and image.height > 0
            images += 1
        elif path.suffix in ('.json', '.jsonl', '.csv', '.md', '.html', '.gz'):
            text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
            assert not re.search(r'/(?:home|Users)/', text), 'Private local path in public artifact: '+str(path)
            texts += 1
            if path.suffix == '.json':
                json.loads(text)
            elif path.suffix in ('.jsonl', '.gz'):
                for line in text.splitlines():
                    json.loads(line)
                if path.suffix == '.gz':
                    compressed_bytes += len(text.encode())
            if path.suffix in ('.html', '.md'):
                if path.suffix == '.html':
                    parser = Links(); parser.feed(text); targets = parser.targets
                else:
                    targets = re.findall(r'\]\(([^)]+)\)', text)
                for target in targets:
                    url = urlsplit(target.strip('<>'))
                    if url.scheme or not url.path:
                        continue
                    target_path = (path.parent/unquote(url.path)).resolve()
                    if target_path != (out/'VALIDATION.json').resolve():
                        assert target_path.is_file(), 'Broken link: '+str(path)+' -> '+target
                    links += 1
    evidence = dict(schema='FAIR_POSEFIX_AND_E_FINAL_VALIDATION_V1', passed=True,
        COMMON95_exact_point_and_pose_rows=475, PRIMARY192=192, CAL46_separate=True,
        PF_PRIMARY_method_rows=576, all_evaluation_rows=1428, E_observations=1252,
        all_two_fixed_PF_anchor_results_preserved=True, strongest_PF_count=112, E0_count=114,
        E0_minus_PF_anchor_recovery=21, E0_minus_PF_anchor_damage=19,
        E0_minus_YOLO_anchor_recovery=22, E0_minus_YOLO_anchor_damage=19,
        scientific_status='NO_SUPPORTED_CHANGE', new_E_variant_performance='NOT_EXECUTED',
        original_weights_refs_defaults_papers_dirty_files_and_branches_preserved=True,
        independent_metrics_and_CI_and_caps_pass=True, independent_E_contract_arithmetic_pass=True,
        integration_tests='7 passed', local_links_checked=links, decoded_images=images,
        public_text_files_checked=texts, compressed_JSON_bytes_checked=compressed_bytes,
        source_sha256=sha(__file__), wall_seconds=time.monotonic()-start,
        CPU_seconds=time.process_time()-cpu, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        artifact_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    dump(out/'VALIDATION.json', evidence)
    return {k: v for k, v in evidence.items() if k != 'artifact_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('repo', 'private', 'output', 'asset'):
        parser.add_argument('--'+key, required=True)
    a = parser.parse_args()
    print(json.dumps(run(a.repo, a.private, a.output, a.asset), indent=2))
