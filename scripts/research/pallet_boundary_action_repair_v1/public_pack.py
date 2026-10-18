"""Explicit small-result publication derivatives and offline verification.

No Git operation, remote publication, pose fitting, or pixel review is performed.
Existing reports, READMEs, figures and galleries are read and preserved. Only
the declared derivative targets and this tool's manifests may be written.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit

NAMESPACE = 'pallet_boundary_action_repair_v1'
EXPECTED_ARMS = 37
PUBLIC_EXTENSIONS = {'.md', '.json', '.csv', '.png', '.jpg', '.jpeg', '.html', '.css', '.js', '.svg'}
PRIVATE_PATH = re.compile(r'/(?:home/[^/\s\"\'<>`]+|Users/[^/\s\"\'<>`]+|tmp/[^/\s\"\'<>`]+|root|mnt|media|private|var/tmp|run)/[^\s\"\'<>`\]\[),;]*')
SECRETS = [re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{24,}\b'), re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'), re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b'),
    re.compile(r'(?i)authorization\s*[:=]\s*[\"\']?Bearer\s+[A-Za-z0-9._-]{20,}')]
POSE_KEYS = {'R_physical', 'R_fixed', 'R_cf', 'centroid', 'native_t', 'model_state_dict', 'native_GT_xyz', 'projected_GT_native_xy'}
BULK_PAYLOAD_KEYS = {'candidates', 'banks', 'predictions', 'poses', 'actual_pose', 'endpoint_journal', 'generated_candidates',
    'feasible_pose_samples', 'optimizer_endpoints', 'optimizer_state_dict', 'state_dict', 'raw_RGB'}
RESULT_DETAIL_KEYS = {'all_deltas', 'largest_worsening', 'largest_improvement', 'ids', 'failures'}
REQUIRED_FIGURES = ['joint_success.png', 'translation_cdf.png', 'rotation_cdf.png', 'paired_recovery_damage.png',
    'actual_action_sources.png', 'same_bank_2x2_pair.png', 'training_curves_all_seeds.png']

# This is a file allowlist, not a recursive private-directory publication rule.
# Optional paths may be absent while a branch is honestly blocked; verification
# still requires every final public deliverable that the report claims to have.
ALLOWLIST = [
    ('evaluation_v1/COMPARISON_ALL_ARMS.csv', 'COMPARISON_ALL_ARMS.csv', 'csv', True),
    ('evaluation_v1/ROW_METRICS.csv.gz', 'ROW_METRICS.csv.gz', 'csv', True),
    ('evaluation_v1/PAIRED_TRANSITIONS.csv.gz', 'PAIRED_TRANSITIONS.csv.gz', 'csv', True),
    ('evaluation_v1/ACTION_SOURCE_COUNTS.csv', 'ACTION_SOURCE_COUNTS.csv', 'csv', True),
    ('evaluation_v1/CANDIDATE_COVERAGE.csv', 'CANDIDATE_COVERAGE.csv', 'csv', True),
    ('evaluation_v1/RESULTS.json', 'RESULTS.json', 'results', True),
    ('evaluation_v1/EVALUATION_SEAL.json', 'EVALUATION_SEAL.json', 'summary', True),
    ('action_training_v1/TRAINING_MANIFEST.json', 'TRAINING_MANIFEST.json', 'summary', True),
    ('PLAN_LOCK.json', 'PLAN_AND_CONFIG.json', 'summary', True),
    ('ENVIRONMENT.json', 'ENVIRONMENT.json', 'summary', True),
    ('RUN_STATE.json', 'RUN_STATE.json', 'summary', True),
    ('EVALUATION_PROTOCOL.json', 'EVALUATION_PROTOCOL.json', 'summary', True),
    ('PREDICTION_SEAL.json', 'PREDICTION_MANIFEST.json', 'summary', True),
    ('ALL_OUTPUTS_SCORING_GO.json', 'ALL_OUTPUTS_SCORING_GO.json', 'summary', True),
    ('SOURCE_BINDINGS.json', 'SOURCE_BINDINGS.json', 'summary', False),
    ('P0_BASELINE_INPUT_INVENTORY.json', 'P0_BASELINE_INPUT_INVENTORY.json', 'summary', False),
    ('B_GEOMETRY_TESTS.json', 'B_GEOMETRY_TESTS_ORIGINAL.json', 'summary', False),
    ('B_GEOMETRY_TESTS_v4.json', 'B_GEOMETRY_TESTS.json', 'summary', False),
    ('B_LOCAL_REGULARIZATION.json', 'B_LOCAL_REGULARIZATION.json', 'summary', False),
    ('B_SOURCE_LOCK_v2.json', 'B_SOURCE_LOCK.json', 'summary', False),
    ('L_ACTION_TESTS.json', 'L_ACTION_TESTS.json', 'summary', False),
    ('L_D0_TRAINING_GO_LOCK.json', 'L_D0_TRAINING_GO_LOCK.json', 'summary', False),
    ('D0_H1_TRAINING_VERIFIED.json', 'D0_H1_TRAINING_VERIFIED.json', 'summary', False),
    ('LEGACY_TRAINING_CURVE_LIMITATION.json', 'LEGACY_TRAINING_CURVE_LIMITATION.json', 'summary', False),
    ('DATA_PIPELINE_PARITY.json', 'DATA_PIPELINE_PARITY.json', 'summary', False),
    ('D1_PREFLIGHT_V2.json', 'D1_PREFLIGHT.json', 'summary', False),
    ('d1_real_candidates_v1/FEATURE_SEAL.json', 'D1_REAL_FEATURE_SEAL.json', 'summary', False),
    ('d1_synthetic_candidates_v1/FEATURE_SEAL.json', 'D1_SYNTHETIC_FEATURE_SEAL.json', 'summary', False),
    ('d1_real_candidates_v1/D1_EXECUTION_RECEIPT.json', 'D1_REAL_EXECUTION_RECEIPT.json', 'summary', False),
    ('d1_synthetic_candidates_v1/D1_EXECUTION_RECEIPT.json', 'D1_SYNTHETIC_EXECUTION_RECEIPT.json', 'summary', False),
    ('b_finite_v1/BATCH_SEAL.json', 'B_BATCH_SEAL.json', 'summary', False),
    ('action_predictions_v1/PREDICTION_SEAL.json', 'L_PREDICTION_SEAL.json', 'summary', False),
    ('independent_numeric_v1/INDEPENDENT_CHECK.json', 'INDEPENDENT_CHECK.json', 'summary', False),
    ('independent_action_d0_v1/INDEPENDENT_ACTION_AUDIT.json', 'INDEPENDENT_ACTION_D0_INITIAL_FAIL.json', 'summary', False),
    ('independent_action_d0_v2/INDEPENDENT_ACTION_AUDIT.json', 'INDEPENDENT_ACTION_D0_RECHECK.json', 'summary', False),
    ('INDEPENDENT_ACTION_REPRESENTATION_RECONCILIATION.json', 'INDEPENDENT_ACTION_REPRESENTATION_RECONCILIATION.json', 'summary', False),
    ('independent_action_full_v1/INDEPENDENT_ACTION_AUDIT.json', 'INDEPENDENT_ACTION_AUDIT.json', 'summary', False),
    ('b_finite_independent_v1/INDEPENDENT_FINITE_CHECK.json', 'INDEPENDENT_FINITE_INITIAL_FAIL.json', 'summary', True),
    ('b_finite_independent_v2/INDEPENDENT_FINITE_CHECK.json', 'INDEPENDENT_FINITE_CHECK.json', 'summary', True),
    ('COST_AND_EXECUTION.json', 'COST_AND_EXECUTION.json', 'summary', False),
    ('VALIDATION.json', 'VALIDATION.json', 'summary', False),
    ('FINAL_DECISION.json', 'FINAL_DECISION.json', 'summary', False),
    ('VISUAL_REVIEW.json', 'VISUAL_REVIEW.json', 'summary', False),
    ('RIGHTS_RECEIPT.json', 'RIGHTS_RECEIPT.json', 'summary', False),
    ('RIGHTS_SCOPE_BINDING.json', 'RIGHTS_SCOPE_BINDING.json', 'summary', False),
    ('DISPLAY_PRIVACY_PLAN.json', 'DISPLAY_PRIVACY_PLAN.json', 'summary', False),
    ('PRIVACY_REVIEW.json', 'PRIVACY_REVIEW.json', 'summary', False),
    ('PRIVACY_DISPLAY_MASKS.json', 'PRIVACY_DISPLAY_MASKS.json', 'summary', False),
    ('posthoc_v1/HISTORICAL_STRATA.csv', 'HISTORICAL_STRATA.csv', 'csv', False),
    ('posthoc_v1/FACTOR_AND_PAIR_COUNTS.csv', 'FACTOR_AND_PAIR_COUNTS.csv', 'csv', False),
    ('posthoc_v1/GAIN_DAMAGE_ACTUAL_SOURCE.csv', 'GAIN_DAMAGE_ACTUAL_SOURCE.csv', 'csv', False),
    ('posthoc_v1/SEED_VARIABILITY.json', 'SEED_VARIABILITY.json', 'summary', False),
    ('posthoc_v1/POSTHOC_MECHANISM.json', 'POSTHOC_MECHANISM.json', 'summary', False),
]


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def encoded(value): return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def inside(path, root):
    try: Path(path).resolve().relative_to(Path(root).resolve()); return True
    except ValueError: return False


def source_rel(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts: raise ValueError('Private source must be an explicit relative path')
    return path


def build_aliases(run_dir, repo_root, aliases_path=None):
    aliases = {str(Path(run_dir).resolve()): '${RUN_DIR}', str(Path(repo_root).resolve()): '${REPO_ROOT}'}
    if aliases_path:
        for old, new in read(aliases_path).items():
            if not Path(old).is_absolute() or not re.fullmatch(r'\$\{[A-Z][A-Z0-9_]*\}', new): raise ValueError('Aliases require absolute source paths and symbolic ${NAME} values')
            aliases[str(Path(old).resolve())] = new
    return sorted(aliases.items(), key=lambda x: -len(x[0]))


def redact(text, aliases, counts):
    for old, new in aliases:
        occurrences = text.count(old)
        if occurrences: counts['declared_path_replacements'] += occurrences; text = text.replace(old, new)
    def generic(match):
        counts['other_private_path_replacements'] += 1
        # Preserve path identity without publishing the private directory. Two
        # unrelated common.py/model.json files must not collapse into one key.
        identity = hashlib.sha256(match.group().encode()).hexdigest()[:16]
        return '${PRIVATE_PATH}/' + identity + '/' + Path(match.group()).name
    return PRIVATE_PATH.sub(generic, text)


def forbidden_pose(value):
    if isinstance(value, dict):
        if set(value) & POSE_KEYS: return True
        if 'R' in value and 't' in value:
            R, t = value['R'], value['t']
            if isinstance(R, list) and len(R) == 3 and isinstance(t, list) and len(t) == 3: return True
        return any(forbidden_pose(v) for v in value.values())
    if isinstance(value, list): return any(forbidden_pose(v) for v in value)
    return False


def summarize(value, aliases, counts, *, results=False, key=''):
    if isinstance(value, dict):
        out = {}
        def put(name, item):
            name = redact(str(name), aliases, counts)
            if name in out: raise ValueError('Public summary key collision: ' + name)
            out[name] = item
        for k, v in value.items():
            if k in BULK_PAYLOAD_KEYS and isinstance(v, (list, dict)):
                put(k + '_payload_summary', dict(items=len(v), canonical_payload_sha256=hashlib.sha256(encoded(v)).hexdigest(), publication='raw bank/pose/model payload not copied'))
                counts['raw_bank_or_pose_payloads_summarized'] += 1; continue
            if k in POSE_KEYS or (k in ('R', 't') and isinstance(v, list)):
                counts['pose_or_weight_fields_omitted'] += 1; continue
            if results and k in RESULT_DETAIL_KEYS and isinstance(v, (list, dict)):
                counts['repeated_result_details_omitted'] += 1; continue
            if k == 'training_frame_ids' and isinstance(v, list):
                put('training_frame_count', len(v)); put('training_frame_ids_sha256', hashlib.sha256(encoded(v)).hexdigest()); counts['repeated_train_ID_lists_summarized'] += 1; continue
            # Preserve every scalar in a compact result, including failures and
            # bad seeds. Very large file/frame manifests are reduced to a bound
            # count/hash rather than copied as private bulk journals.
            if isinstance(v, (list, dict)) and len(v) > 256:
                put(k + '_summary', dict(items=len(v), canonical_payload_sha256=hashlib.sha256(encoded(v)).hexdigest(), publication='bulk list not copied; original SHA retained'))
                counts['bulk_lists_summarized'] += 1; continue
            put(k, summarize(v, aliases, counts, results=results, key=k))
        return out
    if isinstance(value, list): return [summarize(v, aliases, counts, results=results, key=key) for v in value]
    if isinstance(value, str): return redact(value, aliases, counts)
    return value


def numeric_csv(path, aliases, counts):
    raw = gzip.decompress(path.read_bytes()) if path.name.endswith('.gz') else path.read_bytes()
    text = io.StringIO(raw.decode(), newline=''); reader = csv.DictReader(text)
    if not reader.fieldnames: raise ValueError('Empty CSV is not a scientific result')
    output = io.StringIO(newline=''); writer = csv.DictWriter(output, fieldnames=reader.fieldnames); writer.writeheader(); n = 0
    banned_headers = POSE_KEYS | {'R', 't', 'actual_pose', 'native_xyz', 'model_state_dict'}
    if set(reader.fieldnames) & banned_headers: raise ValueError('A pose/bank CSV is not allowed for public small-results packing')
    for row in reader:
        writer.writerow({k: redact(v, aliases, counts) if isinstance(v, str) else v for k, v in row.items()}); n += 1
    counts['CSV_rows'] = n; data = output.getvalue().encode()
    return gzip.compress(data, mtime=0) if path.name.endswith('.gz') else data


def scan_text(value, label):
    if PRIVATE_PATH.search(value): raise ValueError('Private absolute path remains in public text: ' + label)
    if any(pattern.search(value) for pattern in SECRETS): raise ValueError('Credential-like content detected; no automatic publication: ' + label)


def prepare(run_dir, repo_root, output, *, aliases_path=None, extra_json=None, GO=False):
    if not GO: raise PermissionError('Actual publication derivation requires explicit root GO')
    started = time.monotonic(); run_dir = Path(run_dir).resolve(); repo_root = Path(repo_root).resolve(); output = Path(output).resolve()
    if not inside(output, repo_root / '_docs/experiments') or output.name != NAMESPACE: raise ValueError('Only the new research documentation namespace is writable')
    output.mkdir(parents=True, exist_ok=True); aliases = build_aliases(run_dir, repo_root, aliases_path)
    evaluation_path = run_dir / 'evaluation_v1/EVALUATION_SEAL.json'
    evaluation = read(evaluation_path); evaluation_sha = sha(evaluation_path)
    if not evaluation.get('complete'): raise ValueError('A completed sealed evaluation is required')
    for name, expected in evaluation['artifacts'].items():
        path = evaluation_path.parent / source_rel(name)
        if not path.is_file() or sha(path) != expected: raise ValueError('Original evaluated artifact changed: ' + name)
    for relative in ('figures/FIGURE_MANIFEST.json', 'gallery/GALLERY_MANIFEST.json'):
        if read(output / relative).get('evaluation_seal_sha256') != evaluation_sha:
            raise ValueError('Figure/gallery does not use the sealed evaluation: ' + relative)
    figure_manifest = read(output / 'figures/FIGURE_MANIFEST.json')
    for name, expected in figure_manifest['artifacts'].items():
        if sha(output / 'figures' / source_rel(name)) != expected:
            raise ValueError('Rendered figure/data changed since generation: ' + name)
    items = list(ALLOWLIST)
    if extra_json:
        for item in read(extra_json):
            src, dst = item['source'], item['target']
            mode = item.get('mode', 'summary' if src.endswith('.json') else 'csv' if src.endswith(('.csv', '.csv.gz')) else 'md' if src.endswith('.md') else '')
            reserved = {'PUBLICATION_MANIFEST.json', 'PUBLICATION_CHECK.json', 'README.md', 'FINAL_REPORT_KO.md', 'REPRODUCE.md'}
            if mode not in ('summary', 'csv', 'md') or Path(dst).name != dst or dst in reserved or dst.startswith('METHOD'):
                raise ValueError('Extra files require an explicit small JSON/CSV/MD source and a nonreserved public basename; root reports and method documents are preserved')
            if not src.endswith(('.json', '.csv', '.csv.gz', '.md')) or not dst.endswith(('.json', '.csv', '.csv.gz', '.md')): raise ValueError('Raw journals, banks, RGB, weights and font files are not accepted as extra publication sources')
            if mode == 'summary' and not (src.endswith('.json') and dst.endswith('.json')): raise ValueError('JSON summary source/target extensions must match')
            if mode == 'md' and not (src.endswith('.md') and dst.endswith('.md')): raise ValueError('Markdown source/target extensions must match')
            if mode == 'csv' and not (src.endswith(('.csv', '.csv.gz')) and dst.endswith(('.csv', '.csv.gz')) and src.endswith('.gz') == dst.endswith('.gz')): raise ValueError('CSV source/target compression and extensions must match')
            if item.get('replace_default', False): items = [v for v in items if v[1] != dst]
            items.append((src, dst, mode, item.get('required', True)))
    targets = [v[1] for v in items]
    if len(targets) != len(set(targets)): raise ValueError('Derivative target names must be unique')
    previous_path = output / 'PUBLICATION_MANIFEST.json'; previous = read(previous_path) if previous_path.exists() else {}
    owned = {r['target']: r for r in previous.get('derivatives', [])}; derivatives = []; missing = []
    for rel, target, mode, required in items:
        src = run_dir / source_rel(rel); dst = output / target
        if not src.is_file():
            if required: raise FileNotFoundError('Required completed result missing: ' + rel)
            missing.append(rel); continue
        if src.is_symlink() or not inside(src, run_dir): raise ValueError('Source must remain a real allowlisted private-run file')
        if src.stat().st_size > 256 * 1024 ** 2: raise ValueError('Source too large for a small-summary publication; provide a separately bound compact receipt')
        original_sha = sha(src); counts = Counter()
        if mode == 'csv': data = numeric_csv(src, aliases, counts)
        elif mode == 'md':
            text = src.read_text()
            if re.search(r'\b(?:R_physical|R_fixed|R_cf|centroid|model_state_dict)\s*[\"\']?\s*[:=]\s*\[', text): raise ValueError('Raw pose or weight arrays cannot be published through Markdown notes')
            data = redact(text, aliases, counts).encode()
        else:
            value = summarize(read(src), aliases, counts, results=mode == 'results')
            if forbidden_pose(value): raise ValueError('Full pose or model payload survived summary extraction: ' + rel)
            data = encoded(value)
        if len(data) > 32 * 1024 ** 2: raise ValueError('Public derivative exceeds32MiB; keep full scalar rows in compressed CSV and provide compact JSON summaries')
        scan_text(gzip.decompress(data).decode() if dst.name.endswith('.gz') else data.decode(), target)
        public_sha = hashlib.sha256(data).hexdigest()
        if dst.exists() and sha(dst) != public_sha:
            equivalent = False
            if mode in ('summary', 'results'):
                try: equivalent = json.loads(dst.read_text()) == json.loads(data.decode())
                except (OSError, UnicodeError, json.JSONDecodeError): pass
            elif dst.name.endswith('.gz'):
                try: equivalent = gzip.decompress(dst.read_bytes()) == gzip.decompress(data)
                except OSError: pass
            if equivalent:
                data = dst.read_bytes(); public_sha = sha(dst); counts['preexisting_semantically_identical_bytes_preserved'] += 1
            elif target not in owned or sha(dst) != owned[target]['public_sha256']: raise FileExistsError('Preserve preexisting or user-edited public artifact: ' + target)
        if not dst.exists() or sha(dst) != public_sha: dst.write_bytes(data)
        if sha(src) != original_sha or sha(dst) != public_sha: raise ValueError('Source or derivative changed during publication preparation')
        derivatives.append(dict(source='${RUN_DIR}/' + rel, original_sha256=original_sha, original_bytes=src.stat().st_size,
            target=target, public_sha256=public_sha, public_bytes=len(data), transformation=mode, transformations=dict(counts), source_modified=False))
    artifacts = []
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.name in ('PUBLICATION_MANIFEST.json', 'PUBLICATION_CHECK.json'): continue
        if path.is_symlink(): raise ValueError('Public assets must not symlink to private/raw data')
        artifacts.append(dict(path=str(path.relative_to(output)), sha256=sha(path), bytes=path.stat().st_size,
            ownership='generated small derivative' if str(path.relative_to(output)) in targets else 'preserved report/document/figure/gallery'))
    manifest = dict(schema='EXPLICIT_SMALL_RESULT_PUBLICATION_MANIFEST_V1', complete_preparation=True,
        source_sha256=sha(__file__), derivatives=derivatives, optional_sources_absent=missing, artifacts=artifacts,
        original_evaluation_seal_sha256=evaluation_sha,
        policy=dict(private_tree_recursive_copy=False, raw_pose_journals=False, candidate_banks=False, raw_RGB_bulk=False,
            checkpoints=False, fonts=False, operating_defaults_changed=False, prior_reports_preserved=True),
        result_detail_policy='Repeated IDs/failure-ID lists/all_deltas/extreme lists are omitted from RESULTS derivative; full IDs, all failures and paired transitions remain in the public scalar CSV ledgers.',
        visual_review='NOT performed by this packing tool; inspect separate actual pixel-review receipt',
        remote_publication='NOT performed or verified by this tool', wall_seconds=time.monotonic() - started)
    previous_path.write_bytes(encoded(manifest)); return manifest


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []; self.images = []; self.ids = set(); self.arms = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'): self.ids.add(attrs['id'])
        if attrs.get('name'): self.ids.add(attrs['name'])
        if tag == 'tr' and attrs.get('data-arm'): self.arms.append(attrs['data-arm'])
        for key in ('href', 'src', 'poster'):
            if attrs.get(key): self.links.append(attrs[key])
        if tag == 'img' and attrs.get('src'): self.images.append(attrs['src'])


def links_in(path):
    text = path.read_text(); parser = Links(); parser.feed(text)
    links = list(parser.links); images = list(parser.images)
    if path.suffix == '.md':
        # Strip code blocks so example shell paths and commented templates are
        # not mistaken for links. Inline-code destinations still are forbidden.
        prose = re.sub(r'```.*?```', '', text, flags=re.S)
        for match in re.finditer(r'(!?)\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[\"\'][^\"\']*[\"\'])?\)', prose):
            target = match.group(2).strip('<>'); links.append(target)
            if match.group(1): images.append(target)
        references = {m.group(1).casefold(): m.group(2).strip('<>') for m in re.finditer(r'^\s*\[([^\]]+)\]:\s*(<[^>]+>|\S+)', prose, re.M)}
        for match in re.finditer(r'(!?)\[([^\]]+)\]\[([^\]]*)\]', prose):
            target = references.get((match.group(3) or match.group(2)).casefold())
            if target: links.append(target)
            if target and match.group(1): images.append(target)
    return links, images


def resolve_local(link, current, repo_root):
    url = urlsplit(link)
    if url.scheme in ('file', 'vscode'): raise ValueError('Machine-local URI is not a portable published repository link')
    if url.scheme or url.netloc: return None
    if not url.path: return current if url.fragment else None
    value = unquote(url.path)
    target = repo_root / value.lstrip('/') if value.startswith('/') else current.parent / value
    if not inside(target, repo_root): raise ValueError('Local link leaves repository: ' + str(current.name) + '/' + link)
    return target.resolve()


def check_anchor(link, target):
    fragment = unquote(urlsplit(link).fragment)
    if not fragment: return True
    if target.suffix == '.html':
        parser = Links(); parser.feed(target.read_text()); return fragment in parser.ids
    if target.suffix == '.md':
        text = re.sub(r'```.*?```', '', target.read_text(), flags=re.S); slugs = set(); repeated = Counter()
        parser = Links(); parser.feed(text); slugs.update(parser.ids)
        for heading in re.finditer(r'^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$', text, re.M):
            clean = re.sub(r'<[^>]*>', '', heading.group(1)).lower(); base = re.sub(r'[^\w\s-]', '', clean).replace(' ', '-')
            occurrence = repeated[base]; repeated[base] += 1; slugs.add(base if occurrence == 0 else base + '-' + str(occurrence))
        return fragment in slugs
    if target.suffix in ('.py', '.js', '.css') and re.fullmatch(r'L\d+(?:-L\d+)?', fragment):
        return max(map(int, re.findall(r'\d+', fragment))) <= len(target.read_text().splitlines())
    return True


def csv_rows(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', newline='') as f: return list(csv.DictReader(f))


def boolean(text):
    if text in ('True', 'true', '1'): return True
    if text in ('False', 'false', '0'): return False
    raise ValueError('Explicit CSV boolean required')


def verify_numbers(output):
    plan = read(output / 'PLAN_AND_CONFIG.json'); arms = plan['arms']
    if len(arms) != EXPECTED_ARMS or len(set(arms)) != EXPECTED_ARMS: raise ValueError('Expected37 frozen registered methods')
    from .registry import ALL_ARMS, contrasts
    protocol = read(output / 'EVALUATION_PROTOCOL.json'); prediction = read(output / 'PREDICTION_MANIFEST.json')
    if any(tuple(x) != tuple(ALL_ARMS) for x in (arms, protocol['arms'], prediction['arms'])):
        raise ValueError('Public plan/protocol/prediction method inventory differs from frozen registry')
    if tuple(map(tuple, protocol['contrasts'])) != contrasts(): raise ValueError('Frozen contrast inventory changed')
    rows = csv_rows(output / 'ROW_METRICS.csv.gz'); indexed = {}; ids = defaultdict(set)
    for row in rows:
        key = row['id'], row['arm']
        if key in indexed or row['arm'] not in arms: raise ValueError('Duplicated or unknown public scalar row')
        if row['partition'] not in ('TEST', 'REMAINDER', 'CAL'): raise ValueError('Unknown partition')
        indexed[key] = row; ids[row['partition']].add(row['id']); boolean(row['success'])
        available = boolean(row['pose_available'])
        if not available and boolean(row['success']): raise ValueError('Missing pose cannot be a binary success')
        for k in ('translation_cm', 'rotation_deg'):
            if available != (row[k] != '') or row[k] != '' and not math.isfinite(float(row[k])): raise ValueError('Public available/missing/numeric error mismatch')
    if {k: len(v) for k, v in ids.items()} != dict(TEST=49, REMAINDER=143, CAL=46): raise ValueError('Original49/143/separate46 public denominators changed')
    all_ids = set.union(*ids.values())
    if len(all_ids) != 238 or len(indexed) != 238 * EXPECTED_ARMS: raise ValueError('Every238 image must have all37 scalar method rows')
    for fid in all_ids:
        if {a for i, a in indexed if i == fid} != set(arms): raise ValueError('A method vanished on a failing image')
    populations = dict(PRIMARY192=ids['TEST'] | ids['REMAINDER'], TEST49=ids['TEST'], REMAINDER143=ids['REMAINDER'], CAL46=ids['CAL'])
    table = csv_rows(output / 'COMPARISON_ALL_ARMS.csv'); table_index = {(r['population'], r['arm']): r for r in table}
    if len(table_index) != 4 * EXPECTED_ARMS or len(table_index) != len(table): raise ValueError('Four full-population comparison tables required')
    for pop, frames in populations.items():
        for arm in arms:
            row = table_index[pop, arm]; success = sum(boolean(indexed[fid, arm]['success']) for fid in frames)
            if int(row['denominator']) != len(frames) or int(row['successes']) != success: raise ValueError('Comparison CSV disagrees with complete scalar ledger')
    pairs = defaultdict(list)
    for row in csv_rows(output / 'PAIRED_TRANSITIONS.csv.gz'):
        pop, before, after = row['population'], row['before'], row['after']; fid = row['id']
        if fid not in populations[pop] or before not in arms or after not in arms: raise ValueError('Paired CSV identity mismatch')
        a, b = boolean(indexed[fid, before]['success']), boolean(indexed[fid, after]['success'])
        expected = 'BOTH_SUCCESS' if a and b else 'RECOVERY' if b else 'DAMAGE' if a else 'BOTH_FAILURE'
        if row['transition'] != expected: raise ValueError('Public paired transition label changed')
        pairs[pop, before, after].append(fid)
    for (pop, before, after), frames in pairs.items():
        if len(frames) != len(set(frames)) or set(frames) != populations[pop]: raise ValueError('Paired contrast drops/duplicates failures')
    expected_pairs = {(pop, before, after) for pop in populations for before, after in protocol['contrasts']}
    if set(pairs) != expected_pairs: raise ValueError('A frozen whole contrast or population is absent from the public paired ledger')
    source_rows = csv_rows(output / 'ACTION_SOURCE_COUNTS.csv'); source_keys = set(); source_groups = Counter()
    for row in source_rows:
        pop, arm, role = row['population'], row['arm'], row['source']; key = pop, arm, role
        if key in source_keys: raise ValueError('Duplicated action-source table row')
        source_keys.add(key); source_groups[pop, arm] += int(row['n'])
        count = sum(indexed[fid, arm]['source'] == role for fid in populations[pop])
        if int(row['n']) != count or int(row['denominator']) != len(populations[pop]): raise ValueError('Action-source table disagrees with actual public source ledger')
    if set(source_groups) != {(pop, arm) for pop in populations for arm in arms} or any(n != len(populations[pop]) for (pop, arm), n in source_groups.items()): raise ValueError('Action-source table drops outputs')
    if source_keys != {(pop, arm, role) for pop in populations for arm in arms for role in ('N0','E0_REGENERATED','HISTORICAL_E0','OTHER','NO_POSE')}:
        raise ValueError('Action-source table drops a zero-count role or introduces a role')
    coverage = csv_rows(output / 'CANDIDATE_COVERAGE.csv'); coverage_index = {(r['population'], r['arm']): r for r in coverage}
    candidate_arms = [a for a in arms if a.startswith(('B_FINITE_', 'EDGE_FINITE_', 'D0_', 'D1_'))]
    if len(coverage) != len(coverage_index) or set(coverage_index) != {(pop, arm) for pop in populations for arm in candidate_arms}: raise ValueError('Candidate-coverage table drops a frozen candidate arm or population')
    if any(int(r['denominator']) != len(populations[pop]) for (pop, arm), r in coverage_index.items()): raise ValueError('Candidate-coverage denominator changed')
    return dict(registered_arms=EXPECTED_ARMS, scalar_rows=len(rows), image_rows=238, PRIMARY192=192, TEST49=49, REMAINDER143=143, CAL46=46,
        comparison_rows=len(table), complete_paired_population_contrasts=len(pairs), arithmetic_scope='public scalar/transition identity and count verification, not independent GT scoring')


def inspect_public_image(path, relative, problems):
    """Verify a fresh stream, then independently decode pixels and inspect EXIF."""
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        size = list(image.size)
        if image.getexif():
            problems.append(dict(kind='IMAGE_EXIF', path=relative))
    return size


def verify(repo_root, output, *, index_files=None, GO=False):
    if not GO: raise PermissionError('Actual publication verification requires explicit root GO')
    started = time.monotonic(); repo_root = Path(repo_root).resolve(); output = Path(output).resolve(); problems = []; checked_links = 0; images = []
    if not inside(output, repo_root / '_docs/experiments') or output.name != NAMESPACE: raise ValueError('Only the new public namespace is verified')
    manifest = read(output / 'PUBLICATION_MANIFEST.json')
    derivative_map = {r['target']: r for r in manifest['derivatives']}
    if len(derivative_map) != len(manifest['derivatives']): problems.append(dict(kind='DUPLICATE_DERIVATIVE_TARGET'))
    for entry in manifest['derivatives']:
        path = output / source_rel(entry['target'])
        if not path.is_file() or sha(path) != entry['public_sha256']:
            problems.append(dict(kind='DERIVATIVE_HASH', path=entry['target']))
    try:
        prediction = read(output / 'PREDICTION_MANIFEST.json')
        evaluation = read(output / 'EVALUATION_SEAL.json')
        gate = read(output / 'ALL_OUTPUTS_SCORING_GO.json')
        links = [(prediction['plan_lock_sha256'], 'PLAN_AND_CONFIG.json'),
                 (prediction['evaluation_protocol_sha256'], 'EVALUATION_PROTOCOL.json'),
                 (evaluation['prediction_seal_sha256'], 'PREDICTION_MANIFEST.json'),
                 (gate['prediction_seal_sha256'], 'PREDICTION_MANIFEST.json'),
                 (gate['plan_sha256'], 'PLAN_AND_CONFIG.json')]
        for expected, target in links:
            if expected != derivative_map[target]['original_sha256']:
                problems.append(dict(kind='FROZEN_INPUT_SEAL_CHAIN', target=target))
    except (KeyError, FileNotFoundError) as error:
        problems.append(dict(kind='FROZEN_INPUT_SEAL_CHAIN_MISSING', detail=str(error)))
    evaluations = [r for r in manifest['derivatives'] if r['target'] == 'EVALUATION_SEAL.json']
    evaluation_sha = evaluations[0]['original_sha256'] if len(evaluations) == 1 else None
    if not evaluation_sha or evaluation_sha != manifest.get('original_evaluation_seal_sha256'):
        problems.append(dict(kind='EVALUATION_SEAL_CHAIN'))
    for relative in ('figures/FIGURE_MANIFEST.json', 'gallery/GALLERY_MANIFEST.json'):
        path = output / relative
        if not path.is_file() or read(path).get('evaluation_seal_sha256') != evaluation_sha:
            problems.append(dict(kind='RENDER_EVALUATION_CHAIN', path=relative))
    figure_path = output / 'figures/FIGURE_MANIFEST.json'
    if figure_path.is_file():
        figure_artifacts = read(figure_path)['artifacts']
        for name in REQUIRED_FIGURES:
            if name not in figure_artifacts or Path(name).with_suffix('.csv').name not in figure_artifacts:
                problems.append(dict(kind='REQUIRED_FIGURE_AND_DATA', path=name))
        for name, expected in figure_artifacts.items():
            path = figure_path.parent / source_rel(name)
            if not path.is_file() or sha(path) != expected: problems.append(dict(kind='FIGURE_DATA_HASH', path=name))
    for entry in manifest['artifacts']:
        path = output / entry['path']
        if not path.is_file() or sha(path) != entry['sha256']: problems.append(dict(kind='ARTIFACT_HASH', path=entry['path']))
    tracked = {r['path'] for r in manifest['artifacts']}
    all_paths = [p for p in output.rglob('*') if p.is_file()]
    for path in all_paths:
        rel = str(path.relative_to(output))
        if path.name in ('PUBLICATION_MANIFEST.json', 'PUBLICATION_CHECK.json'): continue
        if path.is_symlink() or not inside(path, output): problems.append(dict(kind='SYMLINK_OR_ESCAPE', path=rel)); continue
        if rel not in tracked: problems.append(dict(kind='UNMANIFESTED_ARTIFACT', path=rel))
        suffix = '.csv.gz' if path.name.endswith('.csv.gz') else path.suffix.lower()
        if suffix not in PUBLIC_EXTENSIONS | {'.csv.gz'}: problems.append(dict(kind='FORBIDDEN_EXTENSION', path=rel)); continue
        if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            try:
                size = inspect_public_image(path, rel, problems)
                images.append(dict(path=rel, sha256=sha(path), pixels=size, actual_pixels_reviewed_by_this_tool=False))
            except Exception as error: problems.append(dict(kind='IMAGE_DECODE', path=rel, error=repr(error)))
        elif path.suffix.lower() != '.gz':
            text = path.read_text()
            try: scan_text(text, rel)
            except ValueError as error: problems.append(dict(kind='PRIVACY_OR_CREDENTIAL', path=rel, error=str(error)))
            if path.suffix.lower() == '.json':
                try:
                    if forbidden_pose(json.loads(text)): problems.append(dict(kind='POSE_OR_WEIGHT_PAYLOAD', path=rel))
                except json.JSONDecodeError: problems.append(dict(kind='INVALID_JSON', path=rel))
            if path.suffix.lower() in ('.md', '.html'):
                for link in links_in(path)[0]:
                    try:
                        local = resolve_local(link, path, repo_root)
                        if local is not None:
                            checked_links += 1
                            if not local.exists(): problems.append(dict(kind='BROKEN_RELATIVE_LINK', path=rel, target=link))
                            elif local.is_file() and not check_anchor(link, local): problems.append(dict(kind='BROKEN_LOCAL_ANCHOR', path=rel, target=link))
                    except ValueError as error: problems.append(dict(kind='LINK_ESCAPE', path=rel, target=link, error=str(error)))
        else:
            try: scan_text(gzip.decompress(path.read_bytes()).decode(), rel)
            except (OSError, UnicodeError, ValueError) as error: problems.append(dict(kind='GZIP_CONTENT', path=rel, error=str(error)))
    readme = output / 'README.md'
    if not readme.is_file(): problems.append(dict(kind='MISSING_RESEARCH_README'))
    else:
        embedded = [resolve_local(link, readme, repo_root) for link in links_in(readme)[1]]
        for name in REQUIRED_FIGURES:
            figure = output / 'figures' / name
            if not figure.is_file() or figure.resolve() not in embedded: problems.append(dict(kind='README_SUMMARY_FIGURE_NOT_EMBEDDED', path='figures/' + name))
        if not any(p and p.is_file() and ('regression' in p.parts or 'details' in p.parts) for p in embedded): problems.append(dict(kind='README_B_GEOMETRY_PANEL_NOT_EMBEDDED'))
    configured = index_files or [p for p in ('_docs/README.md', '_docs/experiments/README.md', '_docs/experiments/INDEX.md', '_docs/INDEX.md') if (repo_root / p).is_file()]
    pointers = []
    for name in configured:
        path = repo_root / source_rel(name)
        if not path.is_file(): problems.append(dict(kind='MISSING_DOCS_INDEX', path=str(name))); continue
        for link in links_in(path)[0]:
            if NAMESPACE not in link: continue
            local = resolve_local(link, path, repo_root); checked_links += 1
            if local and (local.is_file() or (local / 'README.md').is_file()): pointers.append(dict(index=str(name), target=str(local.relative_to(repo_root))))
            else: problems.append(dict(kind='BROKEN_DOCS_INDEX_POINTER', path=str(name), target=link))
    if not pointers: problems.append(dict(kind='MISSING_EXISTING_DOCS_INDEX_POINTER'))
    gallery_path = output / 'gallery/GALLERY_MANIFEST.json'
    gallery_summary = None; rights_chain = dict(status='NOT_APPLICABLE_GEOMETRY_ONLY')
    if not gallery_path.is_file(): problems.append(dict(kind='MISSING_GALLERY_MANIFEST'))
    else:
        gm = read(gallery_path); rows = gm.get('records', []); ids = [r['id'] for r in rows]
        if gm.get('private_RGB_review') or any(r.get('RGB_publication') == 'PRIVATE_REVIEW_ONLY' for r in rows):
            problems.append(dict(kind='PRIVATE_GALLERY_IN_PUBLIC_NAMESPACE'))
        if gm.get('rows') != 192 or len(ids) != 192 or len(set(ids)) != 192: problems.append(dict(kind='GALLERY_FULL192_COVERAGE'))
        metric_path = output / 'ROW_METRICS.csv.gz'
        primary_ids = {r['id'] for r in csv_rows(metric_path) if r['partition'] != 'CAL'} if metric_path.exists() else set()
        if set(ids) != primary_ids: problems.append(dict(kind='GALLERY_SCALAR_LEDGER_ID_MISMATCH'))
        detailed = [r['id'] for r in rows if r.get('detail')]
        if len(detailed) != gm.get('detailed_unique_panels') or len(detailed) > 12 or set(detailed) != {r['selected_id'] for r in gm.get('selection_records', [])}:
            problems.append(dict(kind='GALLERY_DETAIL_SELECTION_OR_CAP'))
        regression_ids = {r['id'] for r in gm.get('regression_records', [])}
        if len(gm.get('regression_records', [])) != 3 or regression_ids != {'wood_day_01:005249','plastic_day_01:012545','wood_184309:000432'}:
            problems.append(dict(kind='THREE_FIXED_REGRESSIONS_REQUIRED'))
        if gm.get('public_RGB'):
            rights_path = output / 'RIGHTS_RECEIPT.json'; matches = [r for r in manifest['derivatives'] if r['target'] == 'RIGHTS_RECEIPT.json']
            if not rights_path.is_file() or len(matches) != 1:
                problems.append(dict(kind='PUBLIC_RGB_RIGHTS_RECEIPT_NOT_BOUND'))
            else:
                rights = read(rights_path); original_hash = matches[0]['original_sha256']
                valid = rights.get('all_192_rgb_publication_rights_confirmed') is True and gm.get('rights_receipt_sha256') == original_hash and sha(rights_path) == matches[0]['public_sha256']
                if not valid: problems.append(dict(kind='PUBLIC_RGB_RIGHTS_CHAIN_MISMATCH'))
                rights_chain = dict(status='PASS' if valid else 'FAIL', public_receipt_sha256=sha(rights_path), original_rights_receipt_sha256=original_hash,
                    gallery_rights_receipt_sha256=gm.get('rights_receipt_sha256'), authorization_scope='explicit PRIMARY192 result panels; other populations need their own scope or geometry-only views')
        expected_arms = set(read(output / 'PLAN_AND_CONFIG.json')['arms'])
        complete_method_pages = 0
        for row in rows:
            thumb = gallery_path.parent / row['thumbnail']; page = gallery_path.parent / row['page']
            if not thumb.is_file() or sha(thumb) != row['thumbnail_sha256'] or not page.is_file() or sha(page) != row.get('page_sha256'): problems.append(dict(kind='GALLERY_IMAGE_PAGE_BINDING', id=row['id']))
            if row.get('detail'):
                detail = gallery_path.parent / source_rel(row['detail'])
                if not detail.is_file() or sha(detail) != row.get('detail_sha256'): problems.append(dict(kind='GALLERY_DETAIL_HASH', id=row['id']))
            if page.is_file():
                parser = Links(); parser.feed(page.read_text())
                if len(parser.arms) != EXPECTED_ARMS or set(parser.arms) != expected_arms:
                    problems.append(dict(kind='GALLERY_PAGE_METHOD37_COVERAGE', id=row['id'], rendered_method_rows=len(parser.arms)))
                else: complete_method_pages += 1
        if gm.get('public_RGB'):
            privacy_path = output / 'DISPLAY_PRIVACY_PLAN.json'; binding_path = output / 'RIGHTS_SCOPE_BINDING.json'
            privacy_entries = [r for r in manifest['derivatives'] if r['target'] == 'DISPLAY_PRIVACY_PLAN.json']
            if not privacy_path.is_file() or not binding_path.is_file() or len(privacy_entries) != 1:
                problems.append(dict(kind='PUBLIC_RGB_DISPLAY_PRIVACY_NOT_BOUND'))
            else:
                privacy = read(privacy_path); binding = read(binding_path); privacy_hash = privacy_entries[0]['original_sha256']
                privacy_valid = (set(privacy.get('frames', {})) == set(ids)
                    and privacy.get('measurement_or_inference_changes') is False
                    and gm.get('display_privacy_plan_sha256') == privacy_hash
                    and binding.get('display_privacy_plan_sha256') == privacy_hash
                    and binding.get('rights_receipt_sha256') == gm.get('rights_receipt_sha256')
                    and set(binding.get('approved_ids', [])) == set(ids)
                    and sha(privacy_path) == privacy_entries[0]['public_sha256'])
                for row in rows:
                    item = privacy.get('frames', {}).get(row['id'], {})
                    if item.get('original_image_sha256') != row.get('original_image_sha256') or item.get('raw_hw') != row.get('raw_hw'): privacy_valid = False
                if not privacy_valid: problems.append(dict(kind='PUBLIC_RGB_DISPLAY_PRIVACY_CHAIN_MISMATCH'))
                rights_chain['display_privacy_status'] = 'PASS' if privacy_valid else 'FAIL'
                rights_chain['original_display_privacy_plan_sha256'] = privacy_hash
                rights_chain['actual_render_pixel_review'] = 'not performed by this verifier; consult separate VISUAL_REVIEW/PRIVACY_REVIEW receipts'
        gallery_summary = dict(rows=len(rows), detailed_panels=gm.get('detailed_unique_panels'), public_RGB=gm.get('public_RGB'),
            complete_37_method_pages=complete_method_pages, visual_review_separate=True)
        for row in gm.get('regression_records', []):
            panel = gallery_path.parent / source_rel(row['path'])
            if not panel.is_file() or sha(panel) != row.get('panel_sha256'): problems.append(dict(kind='REGRESSION_PANEL_HASH', id=row['id']))
            if row.get('partition') == 'CAL' and row.get('RGB_display'):
                rights = read(output / 'RIGHTS_RECEIPT.json') if (output / 'RIGHTS_RECEIPT.json').is_file() else {}
                allowed_ids = set(rights.get('additional_RGB_frame_ids', []))
                if row['id'] not in allowed_ids and not rights.get('all_238_rgb_publication_rights_confirmed', False): problems.append(dict(kind='CAL_REGRESSION_RGB_OUTSIDE192_RIGHTS_SCOPE', id=row['id']))
    numbers = None
    try: numbers = verify_numbers(output)
    except (OSError, ValueError, KeyError) as error: problems.append(dict(kind='PUBLIC_NUMERIC_LEDGER', error=str(error)))
    if gallery_summary is not None and numbers is not None:
        primary_ids = {r['id'] for r in csv_rows(output / 'ROW_METRICS.csv.gz') if r['partition'] != 'CAL'}
        if set(ids) != primary_ids: problems.append(dict(kind='GALLERY192_IDS_DIFFER_FROM_PUBLIC_SCALAR_LEDGER'))
    total_bytes = sum(p.stat().st_size for p in all_paths)
    if total_bytes > 250 * 1024 ** 2: problems.append(dict(kind='PUBLIC_PACKAGE_OVER250MiB', bytes=total_bytes))
    check = dict(schema='OFFLINE_PUBLICATION_CHECK_V1', status='PASS' if not problems else 'FAIL', problems=problems,
        checked_local_links=checked_links, existing_docs_index_pointers=pointers, images=images, gallery=gallery_summary, numeric_ledgers=numbers, rights_chain=rights_chain,
        public_artifact_bytes=total_bytes, manifest_sha256=sha(output / 'PUBLICATION_MANIFEST.json'), source_sha256=sha(__file__),
        scope='local files, checksums, privacy text, extensions, EXIF, scalar denominators, transitions and links only',
        pixel_visual_review_performed=False, remote_main_verified=False, GT_or_pose_scoring_performed=False,
        separate_requirements='Root must inspect actual image pixels and verify the actual remote main commit after push.', wall_seconds=time.monotonic() - started)
    (output / 'PUBLICATION_CHECK.json').write_bytes(encoded(check)); return check


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'verify']); p.add_argument('--run-dir', type=Path)
    p.add_argument('--repo-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--aliases', type=Path); p.add_argument('--extra-files', '--extra-json', dest='extra_files', type=Path); p.add_argument('--index-file', action='append'); p.add_argument('--publication-GO', action='store_true'); args = p.parse_args()
    if args.stage == 'prepare':
        if args.run_dir is None: p.error('--run-dir is required for prepare')
        result = prepare(args.run_dir, args.repo_root, args.output, aliases_path=args.aliases, extra_json=args.extra_files, GO=args.publication_GO)
        print(json.dumps(dict(stage='PUBLICATION_PREPARED', derivatives=len(result['derivatives']), preserved_artifacts=len(result['artifacts']))))
    else:
        result = verify(args.repo_root, args.output, index_files=args.index_file, GO=args.publication_GO)
        print(json.dumps(dict(stage='PUBLICATION_CHECK', status=result['status'], problems=len(result['problems']))))
        if result['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
