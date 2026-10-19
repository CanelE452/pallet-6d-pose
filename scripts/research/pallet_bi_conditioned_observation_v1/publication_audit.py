"""Independent offline audit of this experiment's small public derivatives.

Only the new source/public namespaces and the six-line root README prefix are
reviewed. All original RGB, model, feature-bank and reference bytes stay private.
The checker hashes those private inputs to detect accidental copying; it does
not publish them, change Git state, or claim a human pixel review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import unquote, urlsplit

from PIL import Image

from .registry import ARMS, LEARNED


NAMESPACE = 'pallet_bi_conditioned_observation_v1'
METADATA = {'PUBLIC_PACK_MANIFEST.json', 'PUBLIC_PACK_AUDIT.json', 'PUBLICATION_RECEIPT.json'}
CAP = 16 * 1024 * 1024
STREAM_CAP = 128 * 1024 * 1024
CELL_CAP = 64 * 1024
PRIVATE = re.compile(r'/(?:home/[^/\s\"\'<>`]+|Users/[^/\s\"\'<>`]+|tmp|root|mnt|media|private|var/tmp|run)/[^\s\"\'<>`]*')
WINDOWS = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"\'<>`]*')
SECRET = re.compile(r'(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"\']?[A-Za-z0-9_+/.=-]{24,}')
BULK = {'ground_truth', 'reference_poses', 'FIXED_REFERENCES', 'native_GT_xyz', 'native_GT_corners',
    'model_state_dict', 'optimizer_state_dict', 'feature_bank', 'corner_rows', 'x_values', 'x_validity',
    'initial_points', 'native_points', 'raw_points', 'original_raw_points', 'observed_points', 'inlier_mask',
    'candidate_poses', 'candidate_ids', 'candidate_scores', 'logits', 'action_logits', 'risk_scores',
    'frozen_BI_fit', 'frozen_visible_measurements', 'partial_original_visible', 'MASK', 'pallet_mask'}
POSE_FIELDS = {'actual_pose', 'status', 'prediction_status', 'fallback', 'accepted', 'technical_failure',
    'solver_rejected', 'historical_copy', 'historical_source_arm', 'external_complete_system',
    'selected_candidate_id', 'original_selected_record_sha256'}
POSE_FIELDS |= {'observation_signature', 'solver_origin_arm'}
CHOICE_FIELDS = {'native_corner', 'automatic_state', 'selected_candidate_id', 'selected_candidate_index',
    'selected_kind', 'selected_score', 'scorer_condition', 'q0', 'raw_intersection_xy', 'blend_alpha',
    'applied_xy', 'covariance_px2', 'source_line_ids', 'physical_source_ids', 'partial', 'partial_normals'}
JSON_NAMES = {'EXPERIMENT_PLAN.json', 'INPUT_AND_FEATURE_CONTRACT.json', 'FEATURE_CONTRACT.json',
    'TRAINING_MANIFEST.json', 'VERIFICATION.json', 'EXECUTION_AND_COST.json', 'VISUAL_REVIEW.json',
    'SUMMARY.json', 'PAIRED.json', 'OBSERVATION_SUMMARY.json', 'DIAGNOSTIC_RESULTS.json',
    'PREDICTION_MANIFEST.json', 'PREDICTION_SEAL_PUBLIC.json', 'EVALUATION_SEAL.json', 'DIAGNOSTIC_SEAL.json', 'GALLERY_MANIFEST.json',
    'INDEPENDENT_AUDIT.json', 'SOURCE_BINDINGS.json', 'RIGHTS_AND_PRIVACY.json'} | METADATA
JSON_NAMES |= {'DIAGNOSTIC_SUMMARY_PUBLIC.json', 'USER_WORK_PRESERVATION.json', 'FIGURES.json'}
JSON_NAMES |= {'CAUSE_DECOMPOSITION.json', 'CANDIDATE_DIAGNOSTIC_SUMMARY.json'}
CSV_NAMES = {'COMPARISON_ALL_ARMS.csv', 'ROW_METRICS.csv.gz', 'OBSERVATION_ROWS.csv.gz',
    'TRAINING_CURVES.csv', 'TRAINING_CURVES.csv.gz', 'TAIL_TRANSITIONS.csv.gz'}
CSV_NAMES |= {'DIAGNOSTIC_ROWS.csv.gz'}
CSV_NAMES |= {'CANDIDATE_REFERENCE_DIAGNOSTIC.csv.gz'}
REQUIRED = {'README.md', 'REPORT_KO.md', 'REPRODUCE.md', 'DIAGNOSTIC_RESULTS.md',
    'COMPARISON_ALL_ARMS.csv', 'ROW_METRICS.csv.gz', 'OBSERVATION_ROWS.csv.gz',
    'SELECTED_POSES.jsonl.gz', 'gallery/index.html', 'EXPERIMENT_PLAN.json',
    'INPUT_AND_FEATURE_CONTRACT.json', 'TRAINING_MANIFEST.json', 'PAIRED_RESULTS.md',
    'VERIFICATION.json', 'EXECUTION_AND_COST.json', 'VISUAL_REVIEW.json'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rb') as stream: value = stream.read(CAP + 1)
    if len(value) > CAP: raise ValueError('Decoded JSON exceeds bounded metadata size')
    return json.loads(value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def json_rows(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        while True:
            line = stream.readline(CAP + 1)
            if not line: break
            if len(line.encode()) > CAP: raise ValueError('Oversized JSONL row')
            if line.strip(): yield json.loads(line)


def relative(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts: raise ValueError('Explicit relative path required')
    return path


def decoded(value):
    for _ in range(5):
        new = html.unescape(unquote(value))
        new = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), new)
        new = new.replace('\\/', '/').replace('\\\\', '\\')
        if new == value: return value
        value = new
    return value


def scan_text(value, label):
    text = decoded(value)
    if PRIVATE.search(text) or WINDOWS.search(text): raise ValueError('Decoded private absolute path in ' + label)
    if SECRET.search(text) or re.search(r'\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{16,}', text):
        raise ValueError('Credential-like content in ' + label)


def scan_source(value, label):
    # A sanitizer may explicitly name a generic root prefix as a whole Python
    # string. It contains no username or filename. Strip only that exact
    # quoted literal; actual user/file paths still undergo the full scan.
    value = re.sub(r"(?P<quote>['\"])/(?:home|tmp|usr|opt|Users|root|mnt|media|private|run)/(?P=quote)",
        "'GENERIC_ROOT_PREFIX'", value)
    scan_text(value, label)


def no_bulk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in BULK and isinstance(child, (dict, list)):
                if not (isinstance(child, dict) and child.get('publication') == 'PRIVATE_BULK_NOT_COPIED'
                        and set(child) <= {'publication', 'items', 'canonical_payload_sha256'}):
                    raise ValueError('Private bank/feature/target payload field: ' + key)
            no_bulk(child)
    elif isinstance(value, list):
        if len(value) > 16 and all(isinstance(v, (float, int)) and not isinstance(v, bool) for v in value):
            raise ValueError('Unexpected large numeric vector in a public metadata artifact')
        for child in value: no_bulk(child)
    elif isinstance(value, str) and value.lstrip().startswith(('{', '[')):
        try: parsed = json.loads(value)
        except (ValueError, TypeError): return
        no_bulk(parsed)


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.values = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value and key in ('href', 'src', 'xlink:href'):
                self.values.append((value, key != 'href' or tag in ('img', 'image', 'script', 'link')))


def check_link(source, value, repo, embedded=False):
    parsed = urlsplit(decoded(value.strip()))
    if parsed.scheme:
        if parsed.scheme not in ('https', 'http', 'mailto') or embedded: raise ValueError('Unsafe/external embedded resource')
        return
    if parsed.netloc: raise ValueError('Protocol-relative link')
    if not parsed.path: return
    path = re.sub(r':\d+$', '', unquote(parsed.path))
    if path.startswith('/') or WINDOWS.match(path): raise ValueError('Absolute artifact link')
    target = (source.parent / path).resolve()
    if not target.is_relative_to(repo.resolve()) or not target.exists(): raise ValueError('Missing or escaping relative link: ' + path)


def links(path, text, repo):
    if path.suffix in ('.html', '.svg'):
        parser = Links(); parser.feed(text)
        for value, embedded in parser.values: check_link(path, value, repo, embedded)
    if path.suffix == '.md':
        for match in re.finditer(r'(!?)\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)', text):
            check_link(path, match[2].strip('<>'), repo, bool(match[1]))
        for match in re.finditer(r'(?m)^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)', text): check_link(path, match[1].strip('<>'), repo)
    for match in re.finditer(r'url\(\s*[\"\']?([^\s)\"\']+)', text, re.I): check_link(path, match[1], repo, True)


def inventory(root, source=False):
    result = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if '__pycache__' in rel.parts or path.suffix in ('.pyc', '.pyo'): continue
        if path.is_symlink(): raise ValueError('Symlink in reviewed namespace')
        if path.is_file():
            if path.stat().st_size > CAP: raise ValueError('Physical artifact exceeds16MiB cap')
            if source and path.suffix not in ('.py', '.md'): raise ValueError('Unexpected source namespace payload')
            result[str(rel)] = sha(path)
    return result


def bounded_text(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rb') as stream:
        value = stream.read(CAP + 1)
    if len(value) > CAP: raise ValueError('Decoded text exceeds16MiB cap')
    return value.decode('utf-8')


def csv_rows(path):
    """Every compressed row is inspected through bounded streaming decode."""
    total = 0
    def lines():
        nonlocal total
        with (gzip.open if path.suffix == '.gz' else open)(path, 'rb') as stream:
            while True:
                line = stream.readline(CELL_CAP * 128 + 1)
                if not line: break
                total += len(line)
                if len(line) > CELL_CAP * 128 or total > STREAM_CAP: raise ValueError('CSV bounded decode/line cap exceeded')
                yield line.decode('utf-8')
    old_limit = csv.field_size_limit(CELL_CAP)
    try:
        reader = csv.DictReader(lines()); fields = reader.fieldnames
        if not fields or len(fields) > 128 or len(fields) != len(set(fields)) or set(fields) & BULK:
            raise ValueError('CSV header outside scalar whitelist')
        for row in reader:
            if None in row or any(v is None for v in row.values()): raise ValueError('CSV width differs from header')
            if any(len(v.encode()) > CELL_CAP for v in row.values()): raise ValueError('CSV cell exceeds64KiB')
            scan_text('\t'.join(row.values()), path.name)
            for value in row.values(): no_bulk(value)
            yield row
    finally: csv.field_size_limit(old_limit)


def pixel_hash(image):
    rgb = image.convert('RGB')
    return hashlib.sha256(str(rgb.size).encode() + b'RGB\0' + rgb.tobytes()).hexdigest()


def raw_inventory(upstream):
    frames = list(json_rows(upstream)); selected = [f for f in frames if f['partition'] in ('TEST', 'REMAINDER')]
    if len(frames) != 238 or len(selected) != 192: raise ValueError('Exact raw real238/PRIMARY192 inventory required')
    byte_hashes = set(); pixels = set()
    for frame in selected:
        path = Path(frame['image'])
        if sha(path) != frame['image_sha256']: raise ValueError('Private original RGB bytes changed')
        with Image.open(path) as image:
            image.load()
            if [image.height, image.width] != frame['raw_hw']: raise ValueError('Original RGB canvas changed')
            pixels.add(pixel_hash(image))
        byte_hashes.add(frame['image_sha256'])
    return frames, byte_hashes, pixels


def forbidden_inventory(run):
    """Hash all current checkpoints, raw references and complete cache exports."""
    result = {}; paths = []
    p0 = read(run / 'P0_INPUT_CONTRACT.json')
    for section, key in (('real_reference', 'path'), ('synthetic_targets', 'source')):
        item = p0[section].get(key)
        if isinstance(item, str):
            path = Path(item)
            if sha(path) != p0[section]['sha256']: raise ValueError('Frozen private GT/reference bytes changed')
            paths.append(path)
    for name in LEARNED:
        path = run / 'training' / name / 'best.pt'
        receipt = read(path.with_name('RUN_RECEIPT.json'))
        if receipt['checkpoint_sha256'] != sha(path) or receipt['status'] != 'COMPLETE':
            raise ValueError('Completed model checkpoint no-copy binding changed')
        paths.append(path)
    for path in (run / 'common_bank').rglob('*'):
        if path.is_file() and path.name in ('BANK.json.gz', 'FEATURES.json.gz', 'LEGACY.json.gz'):
            paths.append(path)
    paths.extend((run / 'common_bank').glob('*_FEATURES.jsonl.gz'))
    for path in paths:
        if not path.is_file(): raise ValueError('Required private model/GT/cache inventory file missing')
        result[str(path)] = sha(path)
        if path.name in ('BANK.json.gz', 'FEATURES.json.gz', 'LEGACY.json.gz'):
            seal = read(path.parent / 'SEAL.json')
            if not seal['complete'] or seal['outputs'][path.name] != result[str(path)]:
                raise ValueError('Private cached bank/feature source bytes changed')
    if len([p for p in paths if p.name == 'best.pt']) != 6:
        raise ValueError('Every actual checkpoint must be included in the no-copy audit')
    if sum(path.name == 'BANK.json.gz' for path in paths) != 1241:
        raise ValueError('Every real238 and synthetic1003 cached bank must be inventoried')
    return result


def readme_pointer(repo, base):
    if not re.fullmatch('[0-9a-f]{40}', base): raise ValueError('Literal frozen base commit SHA required')
    process = subprocess.run(['git', 'show', base + ':readme.md'], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    original = process.stdout; path = repo / 'readme.md'; current = path.read_bytes()
    if not original or not current.endswith(original): raise ValueError('Original root README suffix changed')
    prefix = current[:-len(original)]
    if len(prefix.splitlines()) != 6 or not prefix.endswith(b'\n\n'): raise ValueError('Expected exactly six new README prefix lines')
    text = prefix.decode(); scan_text(text, 'new README prefix'); links(path, text, repo)
    return dict(whole_current_file_sha256=sha(path), base_commit=base,
        original_suffix_sha256=hashlib.sha256(original).hexdigest(), original_suffix_bytes=len(original),
        original_suffix_exact=True, introduced_prefix_lines=6, introduced_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        old_body_rescanned=False, scope='NEW_SIX_LINE_PREFIX_ONLY_OLD_GIT_BLOB_SUFFIX_BYTE_EXACT')


def selected_poses(run, public, originals, ids):
    path = public / 'SELECTED_POSES.jsonl.gz'; selected = list(json_rows(path))
    if [r['id'] for r in selected] != ids: raise ValueError('Public selected pose order/population changed')
    count = 0
    for row in selected:
        original = originals[row['id']]
        if set(row) - {'id', 'partition', 'session', 'cluster_id', 'predictions'}:
            raise ValueError('Selected row metadata outside whitelist')
        if set(row['predictions']) != set(ARMS): raise ValueError('Selected thirteen-arm coverage changed')
        for key in set(row) - {'predictions'}:
            if row[key] != original[key]: raise ValueError('Selected identity metadata changed')
        for arm, entry in row['predictions'].items():
            source = original['predictions'][arm]
            if set(entry) - POSE_FIELDS or entry['actual_pose'] != source['actual_pose']:
                raise ValueError('Selected actual pose differs or extra bank fields exported')
            for key, value in entry.items():
                if key == 'original_selected_record_sha256':
                    if value != hashlib.sha256(canonical(source)).hexdigest(): raise ValueError('Selected original record binding changed')
                elif value != source.get(key): raise ValueError('Selected status/pose field differs')
            count += 1
    if count != 238 * 13: raise ValueError('Selected pose denominator changed')
    return dict(rows=238, arms=13, exact_actual_poses=count)


def extra_scalar_derivatives(run, public):
    candidate = 'CANDIDATE_REFERENCE_DIAGNOSTIC.csv.gz'
    if (public / candidate).exists():
        original = run / 'candidate_diagnostics' / candidate
        if sha(public / candidate) != sha(original): raise ValueError('Candidate diagnostic scalar byte-exact derivative changed')
    path = public / 'DIAGNOSTIC_ROWS.csv.gz'
    if not path.exists(): return dict(candidate_scalar_byte_exact=True, conditional_diagnostic_rows=0)
    private = run / 'diagnostics_v1'; seal = read(private / 'DIAGNOSTIC_SEAL.json')
    rows_path = private / 'SYNTHETIC_ROWS.jsonl.gz'
    if seal['artifacts']['SYNTHETIC_ROWS.jsonl.gz'] != sha(rows_path): raise ValueError('Private conditional diagnostic scalar source changed')
    values = list(json_rows(rows_path)); expected = {}
    if len(values) != 235: raise ValueError('Original conditional validation235 required')
    for frame in values:
        if len(frame['groups']) != 3: raise ValueError('Exactly three fixed conditional diagnostic groups required')
        for name, group in frame['groups'].items():
            row = dict(id=frame['id'], split='validation', reference_kind=name, **group['metric'],
                accepted=group['accepted'], fallback=group['fallback'], status=group['status'],
                exact_packet_cache_reused=group['exact_packet_cache_reused'],
                same_observations_as_automatic=group['same_observations_as_automatic'],
                changed_scorer_condition_corners_in_frame=len(frame['changed_condition_ids']),
                point_replacements_in_frame=frame['point_replacement_count'])
            expected[(frame['id'], name)] = {k: '' if v is None else str(v) for k, v in row.items()}
    actual_rows = list(csv_rows(path)); actual = {(r['id'], r['reference_kind']): r for r in actual_rows}
    if len(actual_rows) != 705 or actual != expected:
        raise ValueError('Published conditional diagnostic scalar rows differ from sealed actual diagnostics')
    return dict(candidate_scalar_byte_exact=True, conditional_diagnostic_rows=705,
        conditional_scalar_fields_reconstructed_from_sealed_private_rows=True)


def gallery_row(path, originals, scored, automatic_states):
    value = read(path); original = originals[value['id']]; evaluation = scored[value['id']]
    if set(value) != {'id', 'partition', 'states', 'predictions'} or value['partition'] != original['partition']:
        raise ValueError('Gallery row schema/identity changed')
    if value['states'] != automatic_states[value['id']]: raise ValueError('Public gallery automatic states changed')
    if set(value['predictions']) != set(ARMS): raise ValueError('Gallery row thirteen-arm coverage changed')
    for arm, entry in value['predictions'].items():
        source = original['predictions'][arm]
        if set(entry) != {'actual_pose', 'metrics', 'status', 'choices'}:
            raise ValueError('Gallery prediction schema changed')
        if entry['actual_pose'] != source['actual_pose'] or entry['metrics'] != evaluation['metrics'][arm] or entry['status'] != evaluation['status'][arm]:
            raise ValueError('Gallery pose/metrics/status do not match sealed actual output')
        if arm not in LEARNED:
            if entry['choices'] is not None: raise ValueError('Historical system cannot invent new local choices')
            continue
        old = {c['native_corner']: c for c in source['choices']}
        if len(entry['choices']) != len(old): raise ValueError('Gallery chosen observation coverage changed')
        for choice in entry['choices']:
            if set(choice) - CHOICE_FIELDS: raise ValueError('Gallery choice outside selected-only whitelist')
            prior = old[choice['native_corner']]
            for key, child in choice.items():
                if key == 'partial_normals':
                    expected = [{k: line[k] for k in ('normal_unit_xy', 'offset_px', 'normal_variance_px2', 'physical_source_ids')}
                        for line in prior['typed_partial']['lines']]
                    if child != expected: raise ValueError('Gallery partial normal differs from actual typed observation')
                elif child != prior[key]: raise ValueError('Gallery actual selected observation changed')
    return value['id']


def context(a):
    repo, public, run, source = (Path(p).resolve() for p in (a.worktree, a.public_dir, a.run_dir, a.source_dir))
    if public != repo / '_docs/experiments' / NAMESPACE or source != repo / 'scripts/research' / NAMESPACE:
        raise ValueError('Only the exact new public/source namespaces are in scope')
    pm = read(run / 'PREDICTION_MANIFEST.json'); ev = read(run / 'evaluation/EVALUATION_SEAL.json')
    if pm['before_scoring'] is not True or pm['real_reference_inputs'] is not False or pm['arms'] != list(ARMS):
        raise ValueError('Complete blind fixed-arm prediction seal required')
    if pm['artifacts']['PREDICTIONS.jsonl.gz'] != sha(run / 'PREDICTIONS.jsonl.gz') or ev['prediction_manifest_sha256'] != sha(run / 'PREDICTION_MANIFEST.json'):
        raise ValueError('Prediction/evaluation seal changed')
    for name, expected in ev['artifacts'].items():
        if sha(run / 'evaluation' / relative(name)) != expected: raise ValueError('Sealed private evaluation artifact changed')
    for item in pm.get('model_and_input_bindings', []):
        if sha(item['path']) != item['sha256']: raise ValueError('Actual predictor model/input/source binding changed')
    originals = {r['id']: r for r in json_rows(run / 'PREDICTIONS.jsonl.gz')}
    ids = pm['ordered_ids']
    if len(ids) != 238 or len(set(ids)) != 238 or set(originals) != set(ids): raise ValueError('Exact real238 required')
    scored = {r['id']: r for r in json_rows(run / 'evaluation/SCORED_ROWS.jsonl.gz')}
    if set(scored) != set(ids): raise ValueError('Scored rows do not cover actual predictions')
    frames, raw_hashes, raw_pixels = raw_inventory(a.upstream)
    forbidden = forbidden_inventory(run)
    return repo, public, run, source, pm, ev, originals, scored, ids, frames, raw_hashes, raw_pixels, forbidden


def scan_asset(path, public, repo, raw_hashes, raw_pixels, forbidden_hashes):
    name = str(path.relative_to(public)); ext = '.csv.gz' if name.endswith('.csv.gz') else '.jsonl.gz' if name.endswith('.jsonl.gz') else path.suffix
    if sha(path) in raw_hashes | forbidden_hashes: raise ValueError('Original RGB/model/feature/cache/GT bytes copied into public payload')
    if ext in ('.png', '.jpg', '.jpeg'):
        with Image.open(path) as image:
            image.load()
            if image.getexif() or image.info.get('exif'): raise ValueError('Image EXIF metadata is forbidden')
            if pixel_hash(image) in raw_pixels: raise ValueError('Transcoded original RGB pixels copied without a derivative')
            for value in image.info.values():
                if isinstance(value, str): scan_text(value, name + ' metadata')
            return dict(kind='raster', width=image.width, height=image.height, mode=image.mode, EXIF_entries=0)
    if ext in ('.csv', '.csv.gz'):
        if path.name not in CSV_NAMES: raise ValueError('Unregistered public CSV')
        count = sum(1 for _ in csv_rows(path))
        return dict(kind='bounded_stream_csv', rows=count)
    if ext == '.jsonl.gz':
        if name != 'SELECTED_POSES.jsonl.gz': raise ValueError('Only selected-pose JSONL may be published')
    elif ext == '.json':
        if path.name not in JSON_NAMES and not name.startswith('gallery/rows/'):
            raise ValueError('Unregistered public JSON')
    elif ext not in ('.md', '.html', '.css', '.js', '.svg'):
        raise ValueError('Forbidden public extension; models/fonts/raw banks excluded')
    text = bounded_text(path); scan_text(text, name)
    if ext == '.json': no_bulk(json.loads(text))
    elif ext == '.jsonl.gz':
        for line in text.splitlines(): no_bulk(json.loads(line))
    if ext in ('.md', '.html', '.css', '.svg'): links(path, text, repo)
    return dict(kind='decoded_safe_text')


def sanitized(value):
    if isinstance(value, dict): return {k: sanitized(v) for k, v in value.items()}
    if isinstance(value, list): return [sanitized(v) for v in value]
    if isinstance(value, str): return PRIVATE.sub('${PRIVATE_PATH}', WINDOWS.sub('${PRIVATE_PATH}', value))
    return value


def write_report(path, value, run, immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'
    if path.exists():
        if path.read_bytes() == data: return
        if immutable: raise FileExistsError('Immutable payload manifest already exists with different bytes')
        backup = run / 'publication_audit_history' / sha(path) / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists(): shutil.copyfile(path, backup)
    path.write_bytes(data)


def run(a):
    started = time.monotonic(); problems = []; checks = Counter(); evidence = {}
    repo, public, run_dir, source, pm, ev, originals, scored, ids, frames, raw_hashes, raw_pixels, forbidden = context(a)
    def checked(name, function):
        checks[name] += 1
        try: return function()
        except Exception as error:
            problems.append(dict(check=name, error_type=type(error).__name__, reason=str(error))); return None
    payload = inventory(public); sources = inventory(source, source=True)
    checked('required_artifact_inventory', lambda: _equal(REQUIRED <= set(payload), True, 'Required final artifact absent'))
    for name in sources:
        checked('source:' + name, lambda name=name: scan_source((source / name).read_text(), name))
    raster = {}; csv_info = {}
    for name in payload:
        info = checked('public:' + name, lambda name=name: scan_asset(public / name, public, repo, raw_hashes, raw_pixels, set(forbidden.values())))
        if info and info['kind'] == 'raster': raster[name] = info
        if info and info['kind'] == 'bounded_stream_csv': csv_info[name] = info
    evidence['selected_poses'] = checked('selected_poses_238_x_13', lambda: selected_poses(run_dir, public, originals, ids))
    evidence['additional_diagnostic_scalars'] = checked('additional_diagnostic_scalar_derivatives',
        lambda: extra_scalar_derivatives(run_dir, public))
    for name in ('ROW_METRICS.csv.gz', 'OBSERVATION_ROWS.csv.gz', 'COMPARISON_ALL_ARMS.csv'):
        checked('exact_private_scalar_derivative:' + name, lambda name=name: _equal(
            list(csv_rows(public / name)), list(csv_rows(run_dir / 'evaluation' / name)), 'Public scalar CSV differs from sealed evaluation'))
    metrics = checked('full_metric_coverage', lambda: list(csv_rows(public / 'ROW_METRICS.csv.gz')))
    if metrics is not None:
        checked('238_x_13_unique_metric_rows', lambda: _equal(
            {(r['id'], r['arm']) for r in metrics}, {(fid, arm) for fid in ids for arm in ARMS}, 'Public metric ID/arm coverage differs'))
        checked('metric_no_duplicate_rows', lambda: _equal(len(metrics), 238 * 13, 'Duplicate/missing public metric rows'))
    observation = checked('public_observation_rows', lambda: list(csv_rows(public / 'OBSERVATION_ROWS.csv.gz')))
    if observation is not None:
        expected_obs = {(row['id'], arm, str(c['native_corner'])) for row in originals.values() for arm in LEARNED
            for c in row['predictions'][arm]['choices']}
        checked('all_visible_learned_observation_coverage', lambda: _equal(
            {(r['id'], r['arm'], r['native_corner']) for r in observation}, expected_obs, 'Public chosen observation coverage differs'))
        checked('observation_no_duplicate_rows', lambda: _equal(len(observation), len(expected_obs), 'Duplicate/missing observation rows'))
    automatic_states = {item['id']: item['states'] for item in read(run_dir / 'P0_INPUT_CONTRACT.json')['real']['inventory']}
    gallery_ids = [checked('gallery_selected_row:' + path.name, lambda path=path: gallery_row(path, originals, scored, automatic_states))
        for path in sorted((public / 'gallery/rows').glob('*.json'))]
    checked('whole_192_gallery_rows', lambda: _equal(set(gallery_ids), {f['id'] for f in frames if f['partition'] != 'CAL'}, 'Gallery identity coverage differs from PRIMARY192'))
    checked('gallery_row_no_duplicates', lambda: _equal(len(gallery_ids), 192, 'Gallery row count differs'))
    pointer = checked('root_README_prefix_and_original_suffix', lambda: readme_pointer(repo, a.base))
    source_files = {str(source.relative_to(repo) / name): value for name, value in sources.items()}
    manifest = dict(schema='BI_CONDITIONED_OBSERVATION_PUBLIC_PAYLOAD_MANIFEST_V1',
        artifacts={name: value for name, value in payload.items() if name not in METADATA},
        source_files=source_files, pointer_files={'readme.md': pointer},
        registered_arms=list(ARMS), rows=238, primary_rows=192, selected_poses=238*13,
        prediction_manifest_sha256=sha(run_dir / 'PREDICTION_MANIFEST.json'),
        evaluation_seal_sha256=sha(run_dir / 'evaluation/EVALUATION_SEAL.json'),
        upstream_sha256=sha(a.upstream), excluded_self_hash_metadata=sorted(METADATA),
        raw_RGB_bytes_checked=192, raw_RGB_pixels_checked=192,
        private_no_copy_inventory_count=len(forbidden),
        private_no_copy_inventory_digest=hashlib.sha256(canonical(forbidden)).hexdigest(),
        file_limits=dict(physical_bytes=CAP, decoded_text_bytes=CAP, streamed_CSV_bytes=STREAM_CAP, CSV_cell_bytes=CELL_CAP))
    manifest_path = public / 'PUBLIC_PACK_MANIFEST.json'
    if a.command == 'prepare':
        if not problems: write_report(manifest_path, manifest, run_dir, immutable=True)
    else:
        checked('immutable_reviewed_payload_manifest', lambda: _equal(read(manifest_path), manifest, 'Payload/source/pointer/input bindings changed after prepare'))
    report = dict(schema='BI_CONDITIONED_OBSERVATION_PUBLICATION_AUDIT_V1', status='FAIL' if problems else 'PASS',
        command=a.command, failure_count=len(problems), problems=problems, checks=dict(checks),
        manifest_sha256=sha(manifest_path) if manifest_path.exists() else None,
        payload_files=len(manifest['artifacts']), source_files=len(source_files),
        raster_checks=raster, bounded_CSV_checks=csv_info, evidence=evidence,
        checked_raw_RGB_images=192, checked_private_models_GT_and_cache_files=len(forbidden),
        physical_material_correspondence_certified=False, human_pixel_review_performed=False,
        remote_verification_performed=False, git_mutations=0, wall_seconds=time.monotonic()-started,
        auditor_source_sha256=sha(__file__))
    write_report(a.private_output, report, run_dir, immutable=True)
    public_report = sanitized(report); scan_text(json.dumps(public_report), 'sanitized public audit')
    write_report(a.public_output, public_report, run_dir)
    print(json.dumps({k: report[k] for k in ('status', 'failure_count', 'payload_files', 'source_files', 'wall_seconds')}), flush=True)
    if problems: raise ValueError('Offline publication audit failed: ' + str(len(problems)))
    return report


def _equal(actual, expected, reason):
    if actual != expected: raise ValueError(reason)


def parser():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=('prepare', 'verify'))
    for name in ('worktree', 'public-dir', 'run-dir', 'source-dir', 'upstream', 'private-output', 'public-output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--base', required=True)
    return p


def main(a):
    try: return run(a)
    except Exception as error:
        # Context/setup failures also get a permanent private failure record.
        # A later repaired attempt must choose another private output name.
        if not a.private_output.exists():
            report = dict(schema='BI_CONDITIONED_OBSERVATION_PUBLICATION_AUDIT_V1', status='FAIL',
                command=a.command, failure_count=1, problems=[dict(check='audit_setup_or_execution',
                    error_type=type(error).__name__, reason=str(error))], git_mutations=0,
                remote_verification_performed=False, auditor_source_sha256=sha(__file__))
            write_report(a.private_output, report, a.run_dir, immutable=True)
            write_report(a.public_output, sanitized(report), a.run_dir)
        raise


if __name__ == '__main__': main(parser().parse_args())
