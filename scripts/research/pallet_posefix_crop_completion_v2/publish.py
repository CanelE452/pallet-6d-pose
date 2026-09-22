"""Package an explicit public allowlist; no Git operations or model execution."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_posefix_crop_completion_v2'
DOC = ROOT / '_docs/experiments' / NAME
CODE = Path(__file__).resolve().parent


def read(name):
    return json.loads((DOC / name).read_text())


def write(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x') as handle:
        handle.write(payload)


def clean(value):
    # Per-case raw tables and bindings are not input to this publication.
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if k not in {'non_evaluable_ids', 'sessions', 'real_sessions'}}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    audit = read('AUDIT.json')
    assert audit['PASS'] and audit['new_train_runs'] == 1 and audit['updates'] == 300
    assert read('FINAL_TESTS.json')['PASS']
    # Verify every audited experiment source remained unchanged.
    for item in audit['code']:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256']
    results = read('RESULTS.json')
    public = {key: clean(results[key]) for key in
              ('real', 'source', 'heatmap', 'decision', 'supervision', 'review')}
    public['train_fit'] = dict(results=results['train_fit']['results'], not_validation=True)
    public['audit'] = {key: audit[key] for key in (
        'PASS', 'new_train_runs', 'updates', 'all_predictions_before_scoring',
        'old_artifacts_unchanged', 'source_corruption_all300_trace_exact',
        'real_original_parity', 'same_orders', 'BN_frozen', 'denominators_unchanged',
        'failed_match_included', 'all_tests', 'max_gpu_temperature',
        'max_gpu_memory_MiB', 'no_sweep', 'no_rescue', 'final_model_changed', 'review', 'figures')}
    write(DOC / 'PUBLIC_RESULTS.json', public)
    doc_names = (
        'REPORT_KO.md', 'COMPLETION.md', 'GALLERY.html', 'PUBLIC_RESULTS.json',
        'NEXT_STAGE_PLAN.md', 'CROP_CONTRACT.md', 'PURPOSE_AND_PLAN.md',
        'PREFLIGHT_AUDIT.md', 'IDENTITY_MAPPING_AUDIT.md', 'PREPARATION_IO_NOTE.md',
        'SCORING_IO_NOTE.md', 'FINAL_TESTS.json', 'DECISION.json')
    code_names = (
        '__init__.py', 'common.py', 'data.py', 'inference.py', 'train.py',
        'test_contracts.py', 'test_analysis.py', 'pretrain.py', 'evaluate.py',
        'probes.py', 'review.py', 'report.py', 'audit.py', 'publish.py', 'README.md')
    files = [DOC / n for n in doc_names] + [CODE / n for n in code_names]
    figures = sorted((DOC / 'figures').glob('comparison_*.jpg'))
    assert len(figures) == 38
    files += figures
    allowed = {p.resolve() for p in files}
    for p in files:
        if p.suffix in {'.md', '.html'}:
            text = p.read_text()
            links = re.findall(r'\]\(([^)]+)\)', text) if p.suffix == '.md' else re.findall(r'(?:src|href)=[\"\x27]([^\"\x27]+)', text)
            for link in links:
                if link.startswith(('http:', 'https:', '#')):
                    continue
                assert (p.parent / link.split('#')[0]).resolve() in allowed, (p, link)
    manifest = {'files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
                'private_excluded': True, 'independent_reproduction_package': False}
    write(DOC / 'PUBLICATION_MANIFEST.json', manifest)
    print('PUBLICATION_READY', len(files), sum(r['bytes'] for r in manifest['files']))


if __name__ == '__main__':
    main()
