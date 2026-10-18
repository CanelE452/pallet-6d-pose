"""Independent offline publication audit of root-prepared small derivatives.

This tool writes only three publication metadata files. It does not publish,
score poses, certify pixels as viewed, or infer remote Git verification.
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

from scripts.research.pallet_risk_information_v1 import publish_pack as OLD
from .registry import arms, EXPECTED_N

NAMESPACE = 'pallet_observation_contract_h1_v1'
METADATA = {'PUBLIC_PACK_MANIFEST.json', 'PUBLIC_PACK_AUDIT.json', 'PUBLICATION_RECEIPT.json'}
ALLOWED = {'.md', '.json', '.jsonl', '.jsonl.gz', '.csv', '.csv.gz', '.png', '.jpg', '.jpeg', '.html', '.css', '.js', '.svg'}
CAP = 16 * 1024 * 1024
STREAM_CAP = 128 * 1024 * 1024
CELL_CAP = 64 * 1024
STREAMED_CSV_NAMES = {'B_QUERY_FACTORIAL.csv.gz', 'B_MASK_QUERY_ROWS.csv.gz',
                     'B_MASK_TRANSITION_BOUNDS.csv.gz', 'TAIL_TRANSITIONS.csv.gz'}
WINDOWS = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"\'<>`]*')
PRIVATE = re.compile(r'/(?:home/[^/\s\"\'<>`]+|Users/[^/\s\"\'<>`]+|tmp|root|mnt|media|private|var/tmp|run)/[^\s\"\'<>`]*')
SECRET = re.compile(r'(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"\']?[A-Za-z0-9_+/.=-]{24,}')
BULK = OLD.BULK_KEYS | {'references', 'ground_truth', 'reference_poses', 'logits', 'action_logits', 'action_scores', 'risk_scores', 'candidate_poses', 'feature_bank'}
POSE_FIELDS = {'actual_pose', 'selected_candidate_id', 'selected_source_aliases', 'selected_candidate_aliases',
               'action_role', 'fallback', 'technical_failure', 'prediction_status', 'status', 'reason', 'original_selected_record_sha256'}
FORBIDDEN_NAMES = {'FIXED_REFERENCES.json', 'SYNTH_GT_LABELS.jsonl.gz', 'HEAD_FEATURES.jsonl.gz', 'ACTION_TARGETS.jsonl.gz'}


def sha(path):return OLD.sha(path)
def read(path):return OLD.read(path)
def suffix(path):return OLD.suffix(path)


def relative(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Explicit relative publication path required')
    return p


def decoded(text):
    for _ in range(4):
        new = html.unescape(unquote(text))
        new = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), new)
        new = new.replace('\\/', '/').replace('\\\\', '\\')
        if new == text:return text
        text = new
    return text


def scan_text(text, label):
    value = decoded(text)
    OLD.scan_text(value, label)
    if PRIVATE.search(value) or WINDOWS.search(value):raise ValueError('Decoded private path in ' + label)
    if SECRET.search(value):raise ValueError('Credential-like assignment in ' + label)


def no_bulk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in BULK and isinstance(item, (dict, list)):
                reference_file_binding=(key=='references' and isinstance(item,dict) and set(item)=={'path','sha256'}
                    and isinstance(item['path'],str) and bool(item['path'])
                    and isinstance(item['sha256'],str) and re.fullmatch('[0-9a-f]{64}',item['sha256']))
                if not (reference_file_binding or (isinstance(item, dict) and set(item) == {'publication', 'items', 'canonical_payload_sha256'}
                        and item['publication'] == 'PRIVATE_BULK_NOT_COPIED'
                        and isinstance(item['items'], int) and re.fullmatch('[0-9a-f]{64}', str(item['canonical_payload_sha256'])))):
                    raise ValueError('Public raw bank/reference/score payload: ' + key)
            no_bulk(item)
    elif isinstance(value, list):
        for item in value:no_bulk(item)
    elif isinstance(value, str) and value.strip().startswith(('{','[')):
        try:parsed=json.loads(value)
        except (ValueError,TypeError):return
        no_bulk(parsed)


def raw_bytes(path):
    raw = Path(path).read_bytes()
    if len(raw) > CAP:raise ValueError('Public encoded file exceeds16MiB cap')
    data = gzip.decompress(raw) if suffix(path) in ('.csv.gz', '.jsonl.gz') else raw
    if len(data) > CAP:raise ValueError('Public decoded file exceeds16MiB cap')
    return data


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links = []
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value and key in ('href', 'src', 'xlink:href'):self.links.append((value, key == 'src' or tag in ('img', 'image')))


def check_link(source, value, repo, image=False):
    url = urlsplit(html.unescape(value.strip()))
    if url.scheme:
        if url.scheme not in ('http', 'https', 'mailto') or image:raise ValueError('Unsafe/external embedded link')
        return
    if url.netloc:raise ValueError('Protocol-relative link outside offline bundle')
    if not url.path:return
    target = unquote(url.path)
    if target.startswith('/') or WINDOWS.match(target):raise ValueError('Absolute link outside offline bundle')
    target = re.sub(r':\d+$', '', target)
    dest = (source.parent / target).resolve()
    if not dest.is_relative_to(repo.resolve()) or not dest.exists():raise ValueError('Missing or escaping relative link: ' + target)


def check_links(path, repo):
    check_link_text(path,path.read_text(),repo)


def check_link_text(path,text,repo):
    scan_text(text, path.name)
    if path.suffix in ('.html', '.svg'):
        parser = Links();parser.feed(text)
        for value, image in parser.links:check_link(path, value, repo, image)
    elif path.suffix == '.md':
        for m in re.finditer(r'(!?)\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)', text):
            check_link(path, m[2].strip('<>'), repo, bool(m[1]))
        for m in re.finditer(r'(?m)^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)', text):check_link(path, m[1].strip('<>'), repo)
    for m in re.finditer(r'url\(\s*[\"\']?([^\s)\"\']+)', text, flags=re.I):check_link(path, m[1], repo, True)


def tree_inventory(root, *, source=False):
    output = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if source and ('__pycache__' in rel.parts or path.suffix in ('.pyc', '.pyo')):continue
        if path.is_symlink():raise ValueError('Symlink is forbidden in checked publication/source tree')
        if path.is_file():output[str(rel)] = sha(path)
    return output


def source_inventory(repo):
    root = repo / 'scripts/research' / NAMESPACE
    inventory = tree_inventory(root, source=True)
    if not inventory:raise ValueError('New source namespace is empty')
    for name in inventory:
        path = root / name
        if path.stat().st_size > CAP:raise ValueError('Unexpected source bulk file')
        scan_text(path.read_text(), str(Path('scripts/research') / NAMESPACE / name))
    return {str(Path('scripts/research') / NAMESPACE / name): value for name, value in inventory.items()}


def seals(run, evaluation):
    pm = read(run / 'PREDICTION_MANIFEST.json');ev = read(evaluation / 'EVALUATION_SEAL.json')
    registered = list(arms(bool(pm['augmentation_executed'])))
    if pm['status'] != 'ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING' or pm['real_reference_inputs'] is not False:
        raise ValueError('All actual outputs must be sealed before publication')
    if pm['arms'] != registered or len(pm['ordered_ids']) != 238 or len(set(pm['ordered_ids'])) != 238:raise ValueError('Registered full238 outputs required')
    if pm['experiment_plan_sha256'] != sha(run / 'EXPERIMENT_PLAN.json') or not str(read(run/'EXPERIMENT_PLAN.json').get('status','')).startswith('FROZEN'):raise ValueError('Frozen plan changed')
    if pm['artifacts']['PREDICTIONS.jsonl.gz'] != sha(run / 'PREDICTIONS.jsonl.gz'):raise ValueError('Original sealed predictions changed')
    if ev['status'] != 'COMPLETE' or ev['prediction_manifest_sha256'] != sha(run / 'PREDICTION_MANIFEST.json'):raise ValueError('Completed evaluated output binding required')
    for name, expected in ev['artifacts'].items():
        if sha(evaluation / relative(name)) != expected:raise ValueError('Original sealed evaluation artifact changed')
    result = read(evaluation / 'EVALUATION.json')
    if result['arms'] != registered:raise ValueError('Evaluated registry changed')
    for pop, count in EXPECTED_N.items():
        methods = result['populations'][pop]
        if set(methods) != set(registered) or any(m['denominator'] != count for m in methods.values()):raise ValueError('Missing full population denominator')
    return pm, ev, registered


def inventory_rgb(path):
    source = OLD.rgb_inventory(path)
    raw_byte_hashes = set();raw_pixel_hashes = set()
    for record in source.values():
        info = OLD.image_check(record['path'], public=False)
        if [info['height'], info['width']] != record['raw_hw']:raise ValueError('Raw RGB canvas changed')
        raw_byte_hashes.add(record['sha256']);raw_pixel_hashes.add(info['pixel_sha256'])
    return source, raw_byte_hashes, raw_pixel_hashes


def derivative_map(path, run, public):
    data = read(path);entries = data['derivatives'] if isinstance(data, dict) else data
    if not isinstance(entries, list):raise ValueError('Explicit root derivative list required')
    seen = set();proof = []
    for row in entries:
        target = relative(row['public_path'])
        if str(target) in seen or str(target) in METADATA:raise ValueError('Duplicate/self-referential derivative target')
        seen.add(str(target));src = Path(row['original_path']);dst = public / target
        if not src.is_absolute():src = run / relative(row['original_path'])
        if not src.is_file() or src.resolve() == dst.resolve():raise ValueError('Non-tautological original derivative source required')
        if sha(src) != row['original_sha256'] or sha(dst) != row['public_sha256']:raise ValueError('Original/public derivative hash mismatch: ' + str(target))
        transform = row['transform']
        if transform == 'byte_exact':expected = src.read_bytes()
        elif transform in ('JSONsafe', 'CSVsafe'):
            expected, _ = OLD.derivative_bytes(src, 'summary' if transform == 'JSONsafe' else 'csv')
        else:raise ValueError('Unregistered derivative transform')
        if expected != dst.read_bytes():raise ValueError('Exact declared transform differs: ' + str(target))
        proof.append(dict(public_path=str(target), original_basename=src.name, original_path_sha256=hashlib.sha256(str(src).encode()).hexdigest(),
            original_sha256=row['original_sha256'], public_sha256=row['public_sha256'], transform=transform))
    return proof


def selected_check(run, public, pm, registered):
    original = list(OLD.rows(run / 'PREDICTIONS.jsonl.gz'));selected = list(OLD.rows(public / 'SELECTED_POSES.jsonl.gz'))
    if [r['id'] for r in original] != pm['ordered_ids'] or [r['id'] for r in selected] != pm['ordered_ids']:raise ValueError('Selected238 identity/order changed')
    count = 0
    for old, new in zip(original, selected):
        if set(new) - {'id', 'partition', 'session', 'cluster_id', 'predictions'} or not {'id', 'partition', 'predictions'} <= set(new):raise ValueError('Selected metadata whitelist violated')
        for key in set(new) - {'predictions'}:
            if new[key] != old[key]:raise ValueError('Selected metadata differs from actual output')
        if set(new['predictions']) != set(registered):raise ValueError('Selected arm denominator changed')
        for arm, value in new['predictions'].items():
            source = old['predictions'][arm]
            if set(value) - POSE_FIELDS or 'actual_pose' not in value:raise ValueError('Selected output leaked non-whitelisted candidate/bank fields')
            if value['actual_pose'] != source['actual_pose']:raise ValueError('Whole actual selected pose is not exact')
            for key, field in value.items():
                if key == 'actual_pose':continue
                no_bulk(field)
                if key == 'original_selected_record_sha256':
                    if field != hashlib.sha256(OLD.canonical(source)).hexdigest():raise ValueError('Original selected record binding changed')
                elif field != source.get(key):raise ValueError('Selected source/status/alias metadata changed')
            count += 1
    if count != 238 * len(registered):raise ValueError('Selected actual pose coverage changed')
    return dict(rows=238, arms=len(registered), actual_selected_poses_exact=count, private_candidate_bank_payloads=0, private_logits_payloads=0, reference_payloads=0)


def stream_sources(map_path, run, public):
    data=read(map_path);entries=data['derivatives'] if isinstance(data,dict) else data;result={}
    for row in entries:
        target=relative(row['public_path'])
        if target.name not in STREAMED_CSV_NAMES:continue
        source=Path(row['original_path'])
        if not source.is_absolute():source=run/relative(row['original_path'])
        if row['transform']!='byte_exact' or source.name!=target.name:
            raise ValueError('Only named byte-exact original CSV ledgers permit streaming expansion')
        if sha(source)!=row['original_sha256'] or sha(public/target)!=row['public_sha256'] or row['original_sha256']!=row['public_sha256']:
            raise ValueError('Streaming CSV original/public exact bytes are not bound')
        result[str(target)]=source
    return result


def stream_csv(path, original):
    """Bounded decoder; all rows/cells are scanned without a whole decoded copy."""
    with gzip.open(original,'rb')as stream:
        header=stream.readline(CELL_CAP+1)
    if len(header)>CELL_CAP or not header.endswith(b'\n'):raise ValueError('Original streamed CSV header is oversized')
    original_fields=next(csv.reader(io.StringIO(header.decode('utf-8'))))
    if not original_fields or len(original_fields)>128 or len(original_fields)!=len(set(original_fields)) or set(original_fields)&BULK:
        raise ValueError('Original streamed CSV has unsafe or duplicate columns')
    size=0;count=0
    def lines():
        nonlocal size
        with gzip.open(path,'rb')as stream:
            while True:
                line=stream.readline(CELL_CAP*128+1)
                if not line:break
                size+=len(line)
                if len(line)>CELL_CAP*128 or size>STREAM_CAP:raise ValueError('Named streamed CSV exceeds bounded decode/line cap')
                yield line.decode('utf-8')
    old_limit=csv.field_size_limit();csv.field_size_limit(CELL_CAP)
    try:
        reader=csv.DictReader(lines())
        if reader.fieldnames!=original_fields:raise ValueError('Streamed public/source headers differ')
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('Streamed CSV row width mismatch')
            if any(len(v.encode('utf-8'))>CELL_CAP for v in row.values()):raise ValueError('Streamed CSV cell exceeds64KiB')
            scan_text('\t'.join(row.values()),path.name)
            for value in row.values():no_bulk(value)
            count+=1
    finally:csv.field_size_limit(old_limit)
    return dict(kind='streamed_csv',decoded_bytes=size,rows=count,columns=len(original_fields),
        original_header_identical=True,header_sha256=hashlib.sha256(header).hexdigest(),
        physical_limit_bytes=CAP,decoded_limit_bytes=STREAM_CAP,cell_limit_bytes=CELL_CAP)


def scan_asset(path, public, repo, raw_hashes, raw_pixels, *, metadata=False, streamed_sources=None):
    name = str(path.relative_to(public));ext = suffix(path)
    if path.is_symlink() or not path.resolve().is_relative_to(public.resolve()):raise ValueError('Public symlink escape')
    if ext not in ALLOWED or path.name in FORBIDDEN_NAMES:raise ValueError('Forbidden publication extension/raw payload filename')
    if path.stat().st_size > CAP:raise ValueError('Public file exceeds16MiB')
    if sha(path) in raw_hashes:raise ValueError('Original raw RGB bytes published')
    if ext in ('.png', '.jpg', '.jpeg'):
        info = OLD.public_raster(path, raw_hashes, raw_pixels)
        from PIL import Image
        with Image.open(path) as im:
            for value in im.info.values():
                if isinstance(value, str):scan_text(value, name + ' image metadata')
        return dict(kind='raster', **info)
    if path.name in STREAMED_CSV_NAMES:
        if not streamed_sources or name not in streamed_sources:raise ValueError('Named streamed CSV requires explicit exact source mapping')
        return stream_csv(path,streamed_sources[name])
    data = raw_bytes(path);text = data.decode('utf-8');scan_text(text, name)
    if ext == '.json':
        value = read(path);OLD.canonical(value);no_bulk(value)
    elif ext in ('.jsonl', '.jsonl.gz'):
        if name not in ('SELECTED_POSES.jsonl.gz', 'RUN_LOG.jsonl'):raise ValueError('Raw JSONL/bank export is forbidden')
        for row in OLD.rows(path):
            OLD.canonical(row)
            if name != 'SELECTED_POSES.jsonl.gz':no_bulk(row)
    elif ext in ('.csv', '.csv.gz'):
        reader = csv.DictReader(io.StringIO(text))
        blocked=set(reader.fieldnames or[]) & BULK
        scalar_action_count=name=='SYNTHETIC_COVERAGE.csv' and blocked=={'actions'}
        if not reader.fieldnames or len(reader.fieldnames)!=len(set(reader.fieldnames)) or (blocked and not scalar_action_count):
            raise ValueError('CSV contains unsafe/duplicate columns: '+name+' '+','.join(sorted(blocked)))
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('CSV row width mismatch')
            if scalar_action_count and not re.fullmatch(r'(?:0|[1-9][0-9]*)',row['actions']):
                raise ValueError('Synthetic coverage actions must be a nonnegative integer count, never a bank')
            for value in row.values():no_bulk(value)
    if ext in ('.md', '.html', '.svg', '.css'):check_links(path, repo)
    return dict(kind='metadata' if metadata else 'decoded_text')


def write_metadata(path, value, run, replace=False):
    safe = OLD.sanitized(value, Counter())
    data = OLD.encoded(safe)
    # Error messages may carry an OS path; do not leak it through an audit.
    text = data.decode();text = WINDOWS.sub('${PRIVATE_PATH}', text);text = PRIVATE.sub('${PRIVATE_PATH}', text);data = text.encode()
    if path.exists() and path.read_bytes() != data:
        if not replace:raise FileExistsError('Preserve existing publication metadata; explicit replacement required')
        backup = run / 'public_pack_prior' / sha(path) / path.name;backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():shutil.copyfile(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(data)


def context(args):
    if not args.publication_GO:raise PermissionError('Explicit root PUBLIC_READY publication GO required')
    run, repo, public, evaluation = map(lambda p: Path(p).resolve(), (args.run_dir, args.repo, args.public_root, args.evaluation))
    if public != repo / '_docs/experiments' / NAMESPACE:raise ValueError('Only the new public namespace is checked')
    pm, ev, registered = seals(run, evaluation)
    inv, raw_hashes, raw_pixels = inventory_rgb(args.rgb_inventory)
    original = list(OLD.rows(run / 'PREDICTIONS.jsonl.gz'))
    if set(inv) != {r['id'] for r in original if r['partition'] != 'CAL'}:raise ValueError('RawRGB/evaluated primary192 metadata join differs')
    return run, repo, public, evaluation, pm, ev, registered, inv, raw_hashes, raw_pixels


def baseline_readme_pointer(run,repo):
    """Certify only the six new lines; old Git bytes are preserved, not waived."""
    base=read(run/'EXPERIMENT_PLAN.json')['base_main']
    if not re.fullmatch('[0-9a-f]{40}',base) or read(run/'INPUT_MANIFEST.json')['base_main']!=base:
        raise ValueError('Frozen original main commit identity required')
    saved=run/'ROOT_README_BASE_BYTES'
    if saved.is_symlink():raise ValueError('Explicit original README byte copy required')
    blob=subprocess.run(['git','-C',str(repo),'show',base+':readme.md'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
    if not blob or saved.read_bytes()!=blob:raise ValueError('Private original README copy is not exact frozen Git blob')
    path=repo/'readme.md';current=path.read_bytes()
    if not current.endswith(blob):raise ValueError('Original README suffix was changed')
    prefix=current[:-len(blob)]
    if len(prefix.splitlines())!=6 or not prefix.endswith(b'\n\n'):raise ValueError('Only the declared six-line README introduction is in scope')
    check_link_text(path,prefix.decode('utf-8'),repo)
    return dict(whole_current_file_sha256=sha(path),scope='ONLY_NEW_SIX_LINE_PREFIX_SECURITY_AND_LINK_REVIEW',
        base_main=base,original_git_blob_sha1=hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest(),
        original_base_sha256=hashlib.sha256(blob).hexdigest(),private_base_copy_sha256=sha(saved),
        original_suffix_bytes=len(blob),original_suffix_exact=True,introduced_prefix_bytes=len(prefix),
        introduced_prefix_lines=6,introduced_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        preexisting_baseline_content_review='NOT_RESCANNED_OR_CERTIFIED; exact historical bytes preserved per instruction')


def pointers(args, repo):
    result = {}
    for name in args.pointer:
        path = repo / relative(name)
        if path.suffix not in ('.md', '.html'):raise ValueError('Explicit documentation pointers only')
        if str(relative(name))=='readme.md':
            result['readme.md']=baseline_readme_pointer(Path(args.run_dir),repo)
        else:
            check_links(path, repo);result[str(relative(name))] = sha(path)
    return result


def prepare(args):
    run, repo, public, evaluation, pm, ev, registered, inv, raw_hashes, raw_pixels = context(args)
    sources = source_inventory(repo);proof = derivative_map(args.extra_map, run, public)
    streamed=stream_sources(args.extra_map,run,public)
    selected = selected_check(run, public, pm, registered)
    inventory = tree_inventory(public)
    for name in inventory:
        scan_asset(public / name, public, repo, raw_hashes, raw_pixels, metadata=name in METADATA,streamed_sources=streamed)
    value = dict(schema='OBSERVATION_H1_PUBLIC_PACK_MANIFEST_V1', status='PREPARED_OFFLINE_AUDIT_REQUIRED',
        artifacts={n: h for n, h in inventory.items() if n not in METADATA}, source_files=sources,
        pointer_files=pointers(args, repo), derivatives=proof, extra_map_sha256=sha(args.extra_map),
        prediction_manifest_sha256=sha(run / 'PREDICTION_MANIFEST.json'), evaluation_seal_sha256=sha(evaluation / 'EVALUATION_SEAL.json'),
        rgb_inventory_sha256=sha(args.rgb_inventory), registered_arms=registered, expected_denominators=EXPECTED_N,
        selected_output_check=selected, actual_raw_RGB_images_checked=len(inv), copied_raw_RGB=False,
        metadata_excluded_from_payload_hashes=sorted(METADATA), source_bytecode_excluded=['__pycache__', '.pyc', '.pyo'],
        file_size_policy=dict(physical_cap_bytes=CAP,general_decoded_cap_bytes=CAP,named_byte_exact_streamed_csvs=sorted(STREAMED_CSV_NAMES),streamed_decoded_cap_bytes=STREAM_CAP,streamed_cell_cap_bytes=CELL_CAP),
        actual_pixel_review_performed_by_this_tool=False, human_review=False, physical_owner_truth='NOT_CERTIFIED',
        remote_verification_by_this_tool=False, self_hash_policy='Only the three named root metadata files are excluded from payload hashes; every existing metadata file still passes decoded safe-content scan. Late remote receipt is external evidence, never inferred here.')
    write_metadata(public / 'PUBLIC_PACK_MANIFEST.json', value, run, args.replace_metadata)
    return value


def verify(args):
    started = time.monotonic()
    run, repo, public, evaluation, pm, ev, registered, inv, raw_hashes, raw_pixels = context(args)
    manifest = read(public / 'PUBLIC_PACK_MANIFEST.json');issues = [];checks = Counter();rasters = {};streamed_info={}
    def checked(name, callback):
        checks[name] += 1
        try:return callback()
        except Exception as error:
            issues.append(dict(check=name, error_type=type(error).__name__, reason=str(error)));return None
    audit_path = public / 'PUBLIC_PACK_AUDIT.json'
    if not audit_path.exists():
        write_metadata(audit_path, dict(status='NOT_YET_VERIFIED', actual_verification_started=True,
            explanation='Actual checker output path created before link scan; this is not a PASS or a remote-publication claim'), run)
    def equality(a, b, reason):
        if a != b:raise ValueError(reason)
    checked('manifest_registry_denominators_and_metadata_policy', lambda: equality(
        (manifest['registered_arms'],manifest['expected_denominators'],manifest['metadata_excluded_from_payload_hashes']),
        (registered,EXPECTED_N,sorted(METADATA)),'Registered outputs, whole denominators or three-file selfhash policy changed'))
    current = checked('current_public_tree', lambda: tree_inventory(public))
    if current is not None:checked('exact_payload_inventory_and_hashes', lambda: equality({n: h for n, h in current.items() if n not in METADATA}, manifest['artifacts'], 'Payload set or bytes changed after prepare'))
    checked('dynamic_source_inventory_and_hashes', lambda: equality(source_inventory(repo), manifest['source_files'], 'Namespace source set or bytes changed'))
    checked('pointer_files', lambda: equality(pointers(args, repo), manifest['pointer_files'], 'Documentation pointer bytes changed'))
    checked('derivative_map_input_sha', lambda: equality(sha(args.extra_map), manifest['extra_map_sha256'], 'Root original/public derivative map changed'))
    checked('all_derivative_original_bytes_and_transforms', lambda: equality(derivative_map(args.extra_map, run, public), manifest['derivatives'], 'Original derivative bindings changed'))
    streamed=checked('named_streamed_csv_exact_mapping',lambda:stream_sources(args.extra_map,run,public)) or {}
    checked('selected_actual_poses_full238', lambda: equality(selected_check(run, public, pm, registered), manifest['selected_output_check'], 'Selected pose coverage changed'))
    for key, actual in (('prediction_manifest_sha256', sha(run / 'PREDICTION_MANIFEST.json')), ('evaluation_seal_sha256', sha(evaluation / 'EVALUATION_SEAL.json')), ('rgb_inventory_sha256', sha(args.rgb_inventory))):
        checked(key, lambda key=key, actual=actual: equality(actual, manifest[key], 'Original seal changed'))
    for name in current or {}:
        info = checked('asset:' + name, lambda name=name: scan_asset(public / name, public, repo, raw_hashes, raw_pixels, metadata=name in METADATA,streamed_sources=streamed))
        if info and info['kind'] == 'raster':rasters[name] = info
        if info and info['kind'] == 'streamed_csv':
            streamed_info[name]=info;checks['streamed_CSV_rows']+=info['rows'];checks['streamed_CSV_decoded_bytes']+=info['decoded_bytes']
    checked('required_final_documents', lambda: equality(all((public / n).is_file() for n in ('README.md', 'REPORT_KO.md', 'METHOD_AND_SOURCES.md', 'REPRODUCE.md')), True, 'Required final report document missing'))
    value = dict(schema='OBSERVATION_H1_PUBLIC_PACK_AUDIT_V1', status='PASS' if not issues else 'FAIL',
        failure_count=len(issues), problems=issues, checks=dict(checks), public_artifacts=len(manifest['artifacts']),
        dynamic_source_files=len(manifest['source_files']), raster_checks=rasters,streamed_CSV_checks=streamed_info, source_sha256=sha(__file__),
        manifest_sha256=sha(public / 'PUBLIC_PACK_MANIFEST.json'), actual_raw_RGB_images_checked=len(inv),
        selected_pose_check=manifest['selected_output_check'], wall_seconds=time.monotonic() - started,
        decoded_metadata_also_scanned=True, actual_pixel_review_performed_by_this_tool=False, human_review=False,
        independent_physical_truth='NOT_CERTIFIED', remote_verification_by_this_tool=False, git_mutations=0)
    write_metadata(audit_path, value, run, args.replace_metadata or read(audit_path).get('status') == 'NOT_YET_VERIFIED')
    if issues:raise ValueError('Publication offline audit failed: ' + str(len(issues)))
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('command', choices=('prepare', 'verify'))
    for name in ('run-dir', 'repo', 'public-root', 'evaluation', 'rgb-inventory', 'extra-map'):p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--publication-GO', action='store_true');p.add_argument('--replace-metadata', action='store_true')
    p.add_argument('--pointer', action='append', default=[])
    return p


if __name__ == '__main__':
    args = parser().parse_args();value = prepare(args) if args.command == 'prepare' else verify(args)
    print(json.dumps(dict(status=value['status'], failure_count=value.get('failure_count', 0))))
