"""독립 package byte/출처 압축/README 링크 검산. 원 SOURCE 경로는 따르지 않는다."""
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from urllib.parse import unquote, urlsplit

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
PACKAGE = R / 'V239_PUBLICATION_PACKAGE'
OUT = R / 'V239_PUBLICATION_VERIFY_C'
EXPECTED = '2ecde64665b1e85d353b40bd64c26d1bc745148b055a1ee842281d8bb244754a'
HISTORY = '../V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/artifacts/V235_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md'

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def guarded(path):
    text = str(path)
    require('v112' not in text.lower() and '__pycache__' not in text.lower()
            and not text.lower().endswith('.pyc'), 'forbidden lexical path')
    for parent in reversed(path.parents):
        require(not stat.S_ISLNK(os.lstat(parent).st_mode), 'symlink parent')
    require(stat.S_ISREG(os.lstat(path).st_mode), 'not regular leaf')
    return path

def leaf(relative):
    p = PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in relative,
            'unsafe package relative path')
    return guarded(PACKAGE / relative)

def open_leaf(relative):
    return os.fdopen(os.open(leaf(relative), os.O_RDONLY | os.O_NOFOLLOW), 'rb')

def digest(stream):
    h, size = hashlib.sha256(), 0
    while True:
        b = stream.read(1024 * 1024)
        if not b:
            break
        h.update(b)
        size += len(b)
    return h.hexdigest(), size

def data(relative):
    require(relative in declared or relative == 'MANIFEST.json', 'undeclared read')
    with open_leaf(relative) as f:
        return f.read()

def metadata(relative):
    return json.loads(data(relative))

declared = {}
with open_leaf('MANIFEST.json') as f:
    manifest_bytes = f.read()
require(hashlib.sha256(manifest_bytes).hexdigest() == EXPECTED, 'manifest pin')
m = json.loads(manifest_bytes)
for entry in m['files']:
    require(entry['path'] not in declared, 'duplicate declaration')
    declared[entry['path']] = entry
require(len(declared) == m['file_count'] == 177, 'declared count')
verified, total = [], 0
for relative, entry in declared.items():
    with open_leaf(relative) as f:
        sha, size = digest(f)
    require((sha, size) == (entry['sha256'], entry['bytes']), 'leaf bytes ' + relative)
    verified.append({'path': relative, 'sha256': sha, 'size_bytes': size})
    total += size
require(total == m['total_bytes'] == 16816979, 'declared total bytes')

source_map = metadata('PUBLICATION_SOURCE_MAP.json')
require(len(source_map['files']) == 171, 'source map count')
mapped, gzip_verified = set(), []
for entry in source_map['files']:
    relative = entry['path']
    require(relative in declared and relative not in mapped, 'source map declaration')
    mapped.add(relative)
    if entry['encoding'] == 'identity':
        actual = (declared[relative]['sha256'], declared[relative]['bytes'])
    else:
        require(entry['encoding'] == 'gzip-lossless' and relative.endswith('.gz'), 'encoding')
        with open_leaf(relative) as f, gzip.GzipFile(fileobj=f) as decoded:
            actual = digest(decoded)
        gzip_verified.append({'path': relative, 'decoded_sha256': actual[0], 'decoded_bytes': actual[1]})
    require(actual == (entry['source_sha256'], entry['source_bytes']), 'source byte provenance')
require(len(gzip_verified) == 4 and {x['path'] for x in gzip_verified} ==
        {x for x in declared if x.endswith('.gz')}, 'gzip closure')

readme = data('README_KO.md').decode('utf-8')
links, inert = [], []
for destination in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', readme):
    if destination == HISTORY:
        inert.append(destination)
        continue
    split = urlsplit(destination)
    if split.scheme or split.netloc:
        inert.append(destination)
        continue
    relative = unquote(split.path)
    require(relative in declared, 'README link not declared: ' + relative)
    links.append(relative)
require(inert == [HISTORY], 'exact inert history link')

plan = metadata('PUBLICATION_PLAN.json')
require(plan['scientific_scope'] == m['scope'], 'scope equality')
require(m['source_gate_pass'] is False and m['current_admission'] is False
        and m['new_current_or_held_payload'] == 0 and m['publication_is_not_model_success'] is True,
        'bounded publication scope')
require(m['scope']['model_adopted'] is False and m['scope']['tolerance_relaxed'] is False
        and m['scope']['new_current_or_held_payload'] == 0, 'no adoption or tolerance change')
v237 = metadata('artifacts/V237_ROOT_SOURCE_CONTEXT_INPUTS/original_source_inputs_v1/COMPLETE.json')
for key, expected in {'completed': True, 'PASS': True, 'groups': 1767,
    'original_rows_retained': 15040, 'masked_target_minus1_retained': 7423,
    'known7617_original_RGB96_and_valid_SHA_exact': 7617,
    'all15040_original_input_and_crop_availability_exact': True,
    'FIT_DEV_group_overlap': 0, 'raw_assets_decoded': 1140,
    'SYN_RGB_uint8_members_only': 440, 'YCB_PNG_RGB_only': 700,
    'other_NPZ_array_members_decoded': 0, 'new_current_or_held_payload': 0,
    'model_forward': 0, 'training': 0}.items():
    require(v237[key] == expected, 'V237 status ' + key)
failed = metadata('artifacts/V238_ROOT_CONTEXT_RESIDUAL/context_residual_source_fit_v1/FAILED_EXECUTION.json')
require(failed['completed'] is False and failed['PASS'] is False
        and failed['automatic_retry'] is False, 'preserved pre-optimizer failure')
diag = metadata('artifacts/V238_ALL_FIT_DESCRIPTOR_DIAG/DIAGNOSIS.json')
for key, expected in {'FIT_rows': 5876, 'first_failing_batch': 4,
    'exceed_1e_6_rows': 66, 'max_abs': 1.5515834093093872e-6,
    'new_head_or_optimizer_updates': 0, 'DEV_or_current_numeric_feature_or_logits': 0,
    'tolerance_or_threshold_changed': False}.items():
    require(diag[key] == expected, 'V238 diagnosis ' + key)
for lane in ['V239_CANONICAL_FEATURE_PRODUCER_B', 'V239_CANONICAL_PAIRED_TRAINER_A']:
    ready = metadata('artifacts/' + lane + '/FINAL_READY.json')
    require(ready['completed'] is True and ready['PASS'] is True
            and ready['prepared_only'] is True and ready['actual_entry_approved'] is False,
            'V239 preparation status')
    if lane.endswith('_B'):
        require(ready['actual_SOURCE_images_NPZ_weights_features_or_DINO_model'] == 0
                and ready['actual_classifier_optimizer_fit'] == 0
                and ready['independent_C_CODE_review'].startswith('PENDING'), 'producer unexecuted')
    else:
        require(ready['actual_SOURCE_feature_geometry_modelweight_image_or_fit'] == 0
                and ready['independent_CODE_review_pending'] is True, 'trainer unexecuted')
for text in ['새 모델의 개선 결과는 아직 없습니다', '새 optimizer 0회', '새 DEV 출력 0개',
             '실제 실행 전', '허용 오차 1e-6을 완화하지 않았습니다', '새로운 독립 평가셋이 아닙니다']:
    require(text in readme, 'README bounded claim ' + text)

code_bytes = guarded(Path(__file__).absolute()).read_bytes()
receipt = {'completed': True, 'PASS': True, 'manifest_sha256': EXPECTED,
    'package': str(PACKAGE), 'verifier': {'path': str(Path(__file__).absolute()),
    'sha256': hashlib.sha256(code_bytes).hexdigest(), 'size_bytes': len(code_bytes)},
    'declared_files_checked': len(verified), 'total_bytes_excluding_manifest': total,
    'manifest_size_bytes': len(manifest_bytes), 'source_map_files_checked': len(mapped),
    'gzip_decoded_verified': gzip_verified, 'README_local_link_references': len(links),
    'README_unique_local_targets': len(set(links)), 'inert_links_not_followed': inert,
    'bounded_status_QA': {'V237': 'completed metadata and prior independent QA retained',
      'V238': 'feature-route stop before optimizer and new DEV output; not classifier performance',
      'V239': 'prepared/unexecuted; full producer/trainer CODE review remains pending'},
    'prior_inspection_note': 'Schema inspection ended with IndexError searching encoding==gzip; actual enum is gzip-lossless. Read-only inspection only; no package change or verifier execution occurred.',
    'limitations': ['Opaque hashes only for arrays/models/images; no numeric decode or image view.',
      'gzip decoded bytes hashed only; JSONL contents not interpreted.',
      'No source-pointer follow, external history access, Git or network.',
      'This receipt checks packaged bytes/status; it does not certify fresh model performance.'],
    'verified_files': verified}
receipt_path = OUT / 'REVIEW.json'
with receipt_path.open('x', encoding='utf-8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps({'completed': True, 'PASS': True, 'receipt': str(receipt_path),
    'receipt_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    'receipt_bytes': receipt_path.stat().st_size, 'verifier': receipt['verifier'],
    'files': len(verified), 'bytes': total, 'gzip_files': len(gzip_verified),
    'README_links': len(links)}, ensure_ascii=False))
