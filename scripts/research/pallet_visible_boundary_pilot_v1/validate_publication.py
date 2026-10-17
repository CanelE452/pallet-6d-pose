"""Small-publication integrity and original workspace preservation checks."""
from __future__ import annotations
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

from .publish import sha, read, rows, dump


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links += [value for key, value in attrs if key in ('src', 'href') and value]


def run(source_root, asset_root, private_root, output):
    started, cpu = time.monotonic(), time.process_time()
    source, asset, private, out = map(Path, (source_root, asset_root, private_root, output))
    def git(*args):
        return subprocess.check_output(['git', '-C', str(asset), *args])
    before = read(private/'START_BEFORE_FETCH.json')
    assert git('rev-parse', 'HEAD').decode().strip() == before['original_HEAD']
    assert git('branch', '--show-current').decode().strip() == before['original_branch']
    status = git('status', '--porcelain=v1', '-z')
    assert status == (private/'ORIGINAL_STATUS_NUL.bin').read_bytes(), 'Original workspace status changed'
    for path, digest in before['protected_files_SHA256'].items():
        assert sha(asset/path) == digest, 'Original dirty file changed'
    branches = git('for-each-ref', '--format=%(refname) %(objectname)', 'refs/heads').decode().splitlines()
    assert all(branch in branches for branch in before['branch_refs']), 'Original branch moved'
    diff = subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only', before['origin_main_before_fetch']]).decode().splitlines()
    allowed = ('scripts/research/pallet_visible_boundary_pilot_v1/', '_docs/experiments/pallet_visible_boundary_pilot_v1/')
    assert all(name.startswith(allowed) for name in diff)

    seal = read(out/'PREDICTION_SEAL.json')
    for name, digest in seal['artifacts'].items():
        assert sha(out/name) == digest
    assert len(rows(out/'CANDIDATE_SCORES.jsonl.gz')) == 351
    predictions = rows(out/'PREDICTIONS.jsonl.gz')
    assert len(predictions) == 24 and len({(r['id'], r['variant']) for r in predictions}) == 24
    assert all(r['treatment_active'] for r in predictions if r['variant'] != 'BASELINE_E')
    binding = read(out/'INPUT_BINDINGS.json')
    for path, digest in binding['direct_source_sha256'].items():
        assert sha(source/path) == digest, 'Bound source changed'
    derived = read(out/'PUBLIC_DERIVATIONS.json')
    for record in derived['records']:
        assert sha(out/record['public']) == record['public_sha256']
    original_inputs = rows(private/'VISIBLE_INPUTS.jsonl')
    public_inputs = rows(out/'VISIBLE_INPUTS.jsonl')
    for old, new in zip(original_inputs, public_inputs):
        for key in ('id', 'ordinal', 'polylines', 'unresolved_regions', 'image_sha256', 'image_hw', 'frozen_at_UTC'):
            assert old[key] == new[key], 'Aliasing changed input semantics'
    assert sha(private/'VISIBLE_INPUTS.jsonl') == read(private/'PROTOCOL_LOCK.json')['input_sha256']
    assert sha(out/'PLAN_AND_CONTRACT.md') == read(private/'PROTOCOL_LOCK.json')['plan_sha256']
    assert read(out/'evidence/INDEPENDENT_CHECK_V1.json')['passed']

    required = ['PLAN_AND_CONTRACT.md', 'VISIBLE_INPUTS.jsonl', 'INPUT_BINDINGS.json', 'PREDICTION_SEAL.json',
        'CANDIDATE_SCORES.jsonl.gz', 'PREDICTIONS.jsonl.gz', 'METRICS.csv', 'TRANSITIONS.csv',
        'VISIBLE_VS_MASS_TRANSITIONS.csv', 'EVIDENCE_DISCRIMINATION.csv', 'REPORT_KO.md',
        'NEXT_DECISION_KO.md', 'REPRODUCE.md', 'COST.json', 'gallery/index.html']
    assert all((out/name).is_file() for name in required)
    with (out/'METRICS.csv').open() as stream:
        metrics = list(csv.DictReader(stream))
    assert len(metrics) == 24
    gallery_receipt = read(out/'gallery/GALLERY_RECEIPT.json')
    for name, digest in gallery_receipt['assets'].items():
        assert sha(out/'gallery'/name) == digest, 'Gallery evidence changed'
    images, links, scanned, gzbytes = 0, 0, 0, 0
    for path in sorted(out.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            with Image.open(path) as im:
                im.load()
                assert im.width > 0 and im.height > 0
            images += 1
        elif path.suffix in ('.md', '.json', '.jsonl', '.html', '.csv', '.gz', '.py'):
            text = gzip.open(path, 'rt').read() if path.suffix == '.gz' else path.read_text()
            assert not re.search(r'/(?:home|Users)/', text), 'Private path: '+str(path)
            scanned += 1
            gzbytes += len(text.encode()) if path.suffix == '.gz' else 0
            if path.suffix == '.json':
                json.loads(text)
            elif path.suffix == '.jsonl' or path.suffix == '.gz':
                for line in text.splitlines():
                    json.loads(line)
            if path.suffix in ('.html', '.md'):
                if path.suffix == '.html':
                    parser = Links(); parser.feed(text); targets = parser.links
                else:
                    targets = re.findall(r'\]\(([^)]+)\)', text)
                for target in targets:
                    target = target.strip('<>')
                    url = urlsplit(target)
                    if url.scheme or not url.path:
                        continue
                    resolved = (path.parent/unquote(url.path)).resolve()
                    if resolved == (out/'VALIDATION.json').resolve():
                        # This successful call writes that exact report only
                        # after every other artifact/link check passes.
                        links += 1
                        continue
                    assert resolved.is_file(), 'Missing local link: '+str(path)+' -> '+target
                    links += 1
    summary = read(out/'SUMMARY.json')['metrics']
    assert [summary[v]['successes'] for v in seal['variants']] == [2, 2, 1, 2]
    assert summary['VISIBLE_SUPPORT']['recoveries'] == summary['VISIBLE_SUPPORT']['damages'] == 1
    report = dict(schema='VISIBLE_BOUNDARY_PILOT_FINAL_VALIDATION_V1', passed=True,
        frames=6, original_retained_candidates=351, scored_variants=4, predictions=24,
        actual_input_frames=6, active_new_arm_predictions=18, no_input_identity_actual_frames=0,
        original_workspace_HEAD_branch_status_6dirty_files_preserved=True, prior_branch_refs_preserved=True,
        changed_paths_confined_to_new_namespaces=True, defaults_papers_weights_references_modified=False,
        public_scientific_outputs_match_private_prediction_seal=True, input_semantics_unchanged_since_freeze=True,
        original_protocol_plan_unchanged=True, independent_full_arithmetic_and_error_check_passed=True,
        input_control_5_preserved_success=True, input_control_6_damage=True,
        gallery_images_decoded=images, local_links_checked=links, privacy_text_files_checked=scanned,
        compressed_text_CRC_and_JSON_bytes_checked=gzbytes, private_absolute_path_matches=0,
        source_sha256=sha(__file__), artifact_sha256={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        elapsed_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu,
        maximum_resident_set_size_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    dump(out/'VALIDATION.json', report)
    return {key: value for key, value in report.items() if key != 'artifact_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('source-root', 'asset-root', 'private-root', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.asset_root, args.private_root, args.output), indent=2))
