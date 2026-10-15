"""Validate copied small closed evidence, without raw journals or execution."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL = Path('/home/minjae/Documents/github/pallet-pose')
D = Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
B = D / 'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
CAMPAIGN = D / 'nonsevere_v1/remainder_after_test49_v1/repair_campaign_20261015_v1'
PUB = D / 'publication_followup_20261015_v3'
ARMS = ['SAME_FROZEN_E', 'GEOMETRY_EDGE_ALL_BI_NATIVE', 'GEOMETRY_EDGE_ALL_BI_CONTAINMENT',
        'GEOMETRY_EDGE_SOURCE_NATIVE', 'GEOMETRY_EDGE_SOURCE_CONTAINMENT']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('href', 'src'))


def main():
    out = ROOT / PUB
    if (out / 'VALIDATION.json').exists(): raise FileExistsError('Preserve validation')
    manifest = read(out / 'COPIED_MANIFEST.json')
    self_rel = str(Path(__file__).relative_to(ROOT))
    manifest['entries'][self_rel] = dict(sha256=sha(__file__), bytes=Path(__file__).stat().st_size, kind='publication_validation_source')
    manifest['file_count'] = len(manifest['entries'])
    manifest['total_bytes'] = sum(row['bytes'] for row in manifest['entries'].values())
    (out / 'COPIED_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    checks = []; failures = []; links_checked = 0; parsed = []
    def check(condition, label):
        checks.append(label)
        if not condition: failures.append(label)
    check(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == manifest['base_commit'], 'base HEAD preserved/no commit')
    for rel, row in manifest['entries'].items():
        file = ROOT / rel
        check(file.is_file() and sha(file) == row['sha256'], 'manifest SHA ' + rel)
        check(file.stat().st_size == row['bytes'], 'manifest size ' + rel)
        if row['kind'] not in ['publication_documentation', 'publication_validation_source']:
            check(sha(ORIGINAL / rel) == row['sha256'], 'original current copy equality ' + rel)
        check(not any(p in Path(rel).parts for p in ['code_snapshot', 'code_snapshots', 'frames', 'spool']), 'excluded raw directory absent ' + rel)
        if file.suffix == '.py':
            ast.parse(file.read_text(), filename=rel); parsed.append(rel)
        if file.suffix == '.gz':
            check(file.name in ['CAL_ERRORS.jsonl.gz', 'SELECTED_POSE_COMPONENTS.jsonl.gz'] and file.stat().st_size < 1024 * 1024, 'small saved-error gzip only ' + rel)
    cutoff = read(out / 'CUTOFF.json')
    for root, previous in cutoff['protected_before_after'].items():
        for rel, value in previous.items():
            file = Path(root) / rel
            check(file.is_file() == value['exists'] and (sha(file) if file.is_file() else None) == value['sha256'], 'protected unchanged ' + root + '/' + rel)
    check(cutoff['V29']['fixed_scores_complete'] and not cutoff['V29']['adopted'], 'closed V29/no adoption')
    check(not cutoff['V30']['full_primary_evaluation_complete_at_this_cutoff'] and not cutoff['V30']['primary_accuracy_claim'], 'V30 cutoff no primary claim')
    progress = read(ROOT / CAMPAIGN / 'PROGRESS_V10_V29.json')
    check(progress['method_count'] == 33 and progress['population_table_rows'] == 132 and progress['paired_control_count'] == 8, 'V10 33methods/132rows/8paired')
    check(progress['default_method'] == 'SAME_FROZEN_E' and not progress['default_changed'] and not progress['goal_completion_declared'], 'V10 default/goal unchanged')
    methods = {row['method']: row for row in progress['methods']}
    canonical = ROOT / B / 'geometry_guided_edge_eval_v4_29/REPORT_KO.md'
    text = canonical.read_text(); tables = []
    for section in [text.split('## 고정 TEST49')[1].split('## 주 평가')[0], text.split('## 주 평가 192장')[1].split('## 확인된')[0]]:
        tables.append([line.split('|')[1:-1] for line in section.splitlines() if line.startswith('| ')][1:])
    for population, rows in zip(['TEST', 'PRIMARY192'], tables):
        check(len(rows) == 5, 'canonical five table rows ' + population)
        for arm, cells in zip(ARMS, rows):
            pop = methods[arm]['populations'][population]; count = pop['count']
            expected = [f"{pop['translation_le5']}/{count}", f"{pop['successes_5cm_5deg']}/{count}",
                f"{pop['translation_cm']['mean']:.4f}", f"{pop['translation_cm']['max']:.4f}", f"{pop['rotation_deg']['mean']:.4f}"]
            if population == 'PRIMARY192': expected.append(str(pop['fallback_count']))
            check([x.strip() for x in cells[1:]] == expected, 'canonical metrics V10 ' + population + '/' + arm)
    pubtext = (out / 'REPORT_KO.md').read_text()
    pubrows = [line.split('|')[1:-1] for line in pubtext.splitlines() if line.startswith('| ')][1:]
    check(len(pubrows) == 5, 'PUB3 five table rows')
    for arm, cells in zip(ARMS, pubrows):
        pop = methods[arm]['populations']; expected = [str(pop['TEST']['translation_le5']), str(pop['REMAINDER']['translation_le5']),
            str(pop['PRIMARY192']['translation_le5']), f"{pop['PRIMARY192']['translation_cm']['mean']:.4f} / {pop['PRIMARY192']['translation_cm']['max']:.4f}"]
        check([x.strip() for x in cells[1:]] == expected, 'PUB3 table metrics ' + arm)
    for partition, count in [('CAL', 46), ('TEST', 49), ('REMAINDER', 143)]:
        score = read(ROOT / B / f'geometry_guided_edge_eval_v4_29/{partition}_SCORE.json')
        seal = read(ROOT / B / 'geometry_guided_edge_eval_v4_29/SCORE_SEAL.json')
        check(sha(ROOT / B / f'geometry_guided_edge_eval_v4_29/{partition}_SCORE.json') == seal['score_sha256'][partition], 'closed score sealed ' + partition)
        check(len(score['rows']) == count, 'closed score all rows ' + partition)
        for arm in ARMS:
            values = [r['metrics'][arm]['translation_cm'] for r in score['rows']]
            p = methods[arm]['populations'][partition]
            check(sum(v <= 5 for v in values) == p['translation_le5'], 'fixed score counts V10 ' + partition + '/' + arm)
            check(abs(sum(values) / count - p['translation_cm']['mean']) < 1e-10 and max(values) == p['translation_cm']['max'], 'fixed score mean/max V10 ' + partition + '/' + arm)
    audit = read(ROOT / B / 'geometry_guided_edge_eval_v4_29/independent_root_audit_v1/AUDIT.json')
    check(audit['checks'] == 77571 and audit['failure_count'] == 0, 'V29 audit 77571/0')
    accounting = read(ROOT / B / 'independent_campaign_accounting_review_v4_29/v2/ACCOUNTING_REVIEW_V2.json')
    check(accounting.get('checks', accounting.get('check_count')) == 8448 and accounting['failure_count'] == 0, 'V10 accounting 8448/0')
    noise = ROOT / B / 'CAL_geometry_guided_edge_noise_v4_30'
    for policy in ['ALL_BI_REPLACE', 'RGB_NONFALLBACK_PRESERVE']:
        model = read(noise / policy / 'MODEL.json'); tangent = read(noise / policy / 'TANGENT_MODEL.json')
        check(model['source_policy'] == tangent['source_policy'] == policy and model['fit_partition'] == tangent['fit_partition'] == 'CAL', 'policy-specific CAL models ' + policy)
        check(model['selection_profile'] == tangent['selection_profile'] and model['noise_population'] == tangent['noise_population'], 'new V30 profile/population ' + policy)
        check(model['observation_generation'] == tangent['observation_generation'] == 'OBSERVED_BI_ASSOCIATED_MULTIGUIDE_ACTUAL_NATIVE_EDGES_V30', 'V30 family ' + policy)
    replay = read(ROOT / B / 'CAL_geometry_guided_edge_noise_v4_30_checks/SAVED_CAL_MODEL_REPLAY_V1.json')
    check(replay.get('checks', replay.get('check_count')) == 9342 and replay['failure_count'] == 0, 'saved-error replay 9342/0')
    for folder, expected in [('geometry_guided_actual_runtime_smoke_v4_30', 53), ('geometry_guided_fitted_native_independent_smoke_v4_30', 293)]:
        smoke = read(ROOT / B / folder / 'RESULT.json')
        check(smoke['checks'] == expected and smoke['failure_count'] == 0, 'closed CAL smoke ' + folder)
    # Check new human-facing documents/HTML only. Hash-map paths in machine
    # seals intentionally point to omitted original evidence and are not web links.
    for rel in manifest['entries']:
        file = ROOT / rel
        if file.suffix not in ['.md', '.html']: continue
        if file.suffix == '.html':
            parser = Links(); parser.feed(file.read_text()); links = parser.links
        else: links = re.findall(r'\]\(([^)]+)\)', file.read_text())
        for link in links:
            parts = urlsplit(link.strip('<>'))
            if parts.scheme or parts.netloc or not parts.path: continue
            target = (file.parent / unquote(parts.path)).resolve()
            if target == (out / 'VALIDATION.json').resolve():
                check(True, 'pending own validation link')
            else: check(target.is_file(), 'human local link ' + str(file.relative_to(ROOT)) + ' -> ' + link)
            links_checked += 1
    report = dict(schema='CURATED_PUBLICATION_VALIDATION_V3', UTC=datetime.now(timezone.utc).isoformat(),
        checks=len(checks), failure_count=len(failures), failures=failures,
        copied_manifest_sha256=sha(out / 'COPIED_MANIFEST.json'),
        copied_payload_files=manifest['file_count'], copied_payload_bytes=manifest['total_bytes'],
        AST_parsed_files=parsed, human_local_links_checked=links_checked,
        canonical_V29_report_sha256=sha(canonical), publication_report_sha256=sha(out / 'REPORT_KO.md'),
        original_protected_and_existing_publication_evidence_unchanged=True,
        tests_not_rerun='Previously sealed 77571/9342/53/293/273 and accounting8448 receipts are copied, not added as new independent checks.',
        current_validation_scope='Byte copies/AST/small saved score tables/local links only; no raw journals, new inference, labels, image measurements or pose fits.',
        commit_or_push_performed=False)
    (out / 'VALIDATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'PUBLICATION_PREPARATION_SEAL.json').write_text(json.dumps(dict(schema='CURATED_PUBLICATION_PREPARATION_SEAL_V3',
        manifest_sha256=sha(out / 'COPIED_MANIFEST.json'), cutoff_sha256=sha(out / 'CUTOFF.json'),
        validation_sha256=sha(out / 'VALIDATION.json'), validator_sha256=sha(__file__),
        report_sha256=sha(out / 'REPORT_KO.md'), commit_or_push_performed=False), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ['checks', 'failure_count', 'failures', 'copied_payload_files', 'copied_payload_bytes', 'human_local_links_checked']}, ensure_ascii=False))
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
