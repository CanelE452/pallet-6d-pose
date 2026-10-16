"""Final read-only package QA; no pose, mask, cost or selection recomputation.

The independent arithmetic receipt verifies stored numerical data. This layer
checks its frozen inputs, output identities, protected originals and publication
links. Model visual categories remain unvalidated physical interpretations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import unquote, urlsplit

from .audit_guard import SupportAuditOnly

IDS = [
    'plastic_night_01:038630', 'wood_day_01:005286',
    'wood_night_01:030698', 'plastic_day_01:020955',
    'plastic_day_01:014704', 'wood_night_01:030898',
]
CATEGORIES = ('PALLET_OUTER_SUPPORT', 'OTHER_OR_CONSTRUCTED_BOUNDARY', 'UNRESOLVED')
BASE = '61343a24130c553f063e0f6ade732d5e810e6e55'
ORIGINAL = 'c56a4e92027a8e56f6704773f24d7c3da78082f1'
SOURCE_NAMESPACE = 'scripts/research/pallet_outline_physical_support_audit_v1/'
PUBLIC_NAMESPACE = '_docs/experiments/pallet_outline_physical_support_audit_v1/'
REQUIRED = (
    'REPORT_KO.md', 'INPUT_BINDINGS.json', 'SUPPORT_REVIEW.jsonl',
    'SUPPORT_REVIEW.csv', 'COST_SUPPORT_SUMMARY.csv', 'gallery/index.html',
    'NEXT_CHANGE_PROPOSAL_KO.md', 'REPRODUCE.md', 'COST.json',
    'REVIEW_REQUIRED.md', 'PACKAGING_BINDINGS.json',
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def rows(path):
    path = Path(path)
    with (gzip.open(path, 'rt', newline='') if path.suffix == '.gz'
          else path.open(newline='')) as stream:
        return list(csv.DictReader(stream))


def check_hash(path, expected):
    require(Path(path).is_file(), 'Missing original bound file: ' + str(path))
    require(sha(path) == expected, 'Original file SHA changed: ' + str(path))


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])


def check_public_links(public, allow_pending_validation_outputs=False):
    checked = 0
    for path in sorted(public.rglob('*')):
        if path.suffix not in ('.md', '.html'):
            continue
        text = path.read_text()
        if path.suffix == '.html':
            parser = HTMLLinks()
            parser.feed(text)
            links = parser.links
        else:
            # Shell and Python examples do not declare published file links.
            text = re.sub(r'```[\s\S]*?```', '', text)
            links = re.findall(r'!?\[[^\]\n]*\]\(([^\s)]+)(?:\s+[^)]*)?\)', text)
        for link in links:
            link = link.strip('<>')
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if allow_pending_validation_outputs and target in {
                    (public / 'VALIDATION.json').resolve(), (public / 'VALIDATION.md').resolve()}:
                # The same process writes these after all input checks pass, then
                # repeats link validation without any deferred targets.
                checked += 1
                continue
            require(target.exists(), 'Broken public relative link: ' + str(path) + ' -> ' + link)
            if url.fragment and target.suffix == '.html':
                dest = HTMLLinks()
                dest.feed(target.read_text())
                require(unquote(url.fragment) in dest.ids, 'Missing HTML anchor: ' + link)
            checked += 1
    return checked


def privacy_scan(public):
    files, compressed, decoded_bytes = 0, 0, 0
    for path in sorted(public.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix == '.gz':
            with gzip.open(path, 'rb') as stream:
                data = stream.read()  # Full read also checks gzip CRC and truncation.
            compressed += 1
        elif path.suffix.lower() in ('.md', '.html', '.json', '.jsonl', '.csv', '.txt', '.py', '.css', '.js'):
            data = path.read_bytes()
        else:
            continue
        content = data.decode('utf-8')
        require(not re.search(r'(?:/home/|/Users/|/tmp/|file://)', content),
                'Private absolute path disclosed in: ' + str(path))
        files += 1
        decoded_bytes += len(data)
    return dict(text_files_scanned=files, compressed_text_files_CRC_checked=compressed,
                uncompressed_bytes_scanned=decoded_bytes, private_absolute_path_matches=0)


def guard_check(receipt, label):
    guard = receipt.get('runtime_guard', receipt)
    require(guard.get('forbidden_function_bodies_executed') == 0,
            'Forbidden execution receipt: ' + label)
    require(not guard.get('forbidden_blocked_attempts'), 'Blocked call attempted: ' + label)
    require(guard.get('no_solver_or_model_monkeypatch') is True,
            'Guard monkeypatch contract differs: ' + label)
    return dict(receipt=label, profiled_python_calls=guard['profiled_python_calls'],
                forbidden_function_bodies_executed=0, forbidden_blocked_attempts={})


def close(left, right, description):
    require(abs(float(left) - float(right)) <= 1e-9, description)


def _validate(args, guard):
    started, cpu = time.monotonic(), time.process_time()
    source, asset, score, state, private, public, output = map(Path, (
        args.source_root, args.asset_root, args.score_private, args.state_private,
        args.analysis_private, args.public, args.output_private))
    require(not output.exists(), 'Preserve previous final receipt; choose a new output-private')
    aliases = sorted(((str(p), alias) for p, alias in (
        (private, 'PRIVATE_AUDIT'), (score, 'PRIVATE_SCORE'), (state, 'PRIVATE_STATE'),
        (source, 'SOURCE_ROOT'), (asset, 'ASSET_ROOT'))), key=lambda x: len(x[0]), reverse=True)

    def sanitize(value):
        if isinstance(value, str):
            for prefix, alias in aliases:
                value = value.replace(prefix, alias)
            return value
        if isinstance(value, dict):
            return {sanitize(k): sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        return value

    for name in REQUIRED:
        require((public / name).is_file(), 'Required public result missing: ' + name)
    binding = read(public / 'INPUT_BINDINGS.json')
    original_binding = read(private / 'bindings_v3/INPUT_BINDINGS.json')
    require(binding['representative_ids'] == IDS and binding['representative_count'] == 6
            and binding['candidate_count'] == 12, 'Fixed six case/twelve candidate denominator changed')
    require(binding['source_result_commit'] == ORIGINAL, 'Source result commit changed')
    old = source / '_docs/experiments/pallet_selection_score_diagnosis_v1'
    old_witness = read(old / 'gallery/outline_witness/WITNESS_RECEIPT.json')['examples']
    require([r['id'] for r in old_witness] == IDS, 'Original witness order changed')
    pointer_count = 0
    for number, (current, original, witness) in enumerate(zip(binding['cases'], original_binding['cases'], old_witness), 1):
        require(current == sanitize(original), 'Published case values changed beyond path aliases')
        require(current['id'] == IDS[number - 1] and current['ordinal'] == number,
                'Case identity/order changed')
        for name in ('RGB', 'mask', 'reference'):
            check_hash(original[name]['path'], original[name]['sha256'])
        check_hash(original['original_observation_frame']['path'], original['original_observation_frame']['sha256'])
        require([r['original_dense_pointer'] for r in current['candidates']] == witness['original_dense_pointers'],
                'Original twelve fixed candidate pointers changed')
        require(all(c['stage'] == 'G13' for c in current['candidates']), 'Fixed original G13 stage changed')
        pointer_count += len(current['candidates'])
    require(pointer_count == 12, 'Wrong candidate pointer count')
    direct = binding['direct_selected_sample_bindings']
    require([r['id'] for r in direct] == IDS, 'Direct sample binding order changed')
    for row in direct:
        check_hash(source / row['witness_csv']['path'], row['witness_csv']['sha256'])
        dense = row['selected_dense_frame']
        check_hash(score / dense['path'].removeprefix('PRIVATE_SCORE/'), dense['sha256'])
    check_hash(score / 'cost_v1/COST_SEAL.json', binding['cost_seal']['sha256'])

    # Exact byte snapshots of original public and legacy numerical dependencies.
    required_count = 0
    for record in original_binding['original_required_artifacts']:
        path = source / record['path']
        check_hash(path, record['sha256'])
        blob = git(source, 'show', ORIGINAL + ':' + record['path'])
        require(hashlib.sha256(blob).hexdigest() == record['sha256'], 'Original Git artifact differs')
        required_count += 1
    upstream = read(old / 'SOURCE_BINDINGS.json')
    for path, expected in upstream['legacy_source_sha256'].items():
        check_hash(source / path, expected)
    for path, record in upstream['private_parent_artifacts'].items():
        check_hash(state / path, record['sha256'])
    score_seal = read(state / 'score_v2/SCORE_SEAL.json')
    for path, expected in score_seal['annotation_sha256'].items():
        check_hash(source / path, expected)
    for record in score_seal['reference_bindings'].values():
        check_hash(source / record['path'], record['sha256'])
    # A new source file after packaging requires rebinding, never silent drift.
    current_sources = {str(p.relative_to(source)): sha(p) for p in sorted((source / SOURCE_NAMESPACE).glob('*.py'))}
    require(binding['audit_sources_SHA256'] == current_sources, 'Packaged new source SHA set is stale')

    independent_path = private / 'INDEPENDENT_REVIEW_CHECK.json'
    independent = read(independent_path)
    require(independent['passed'] is True and independent['parent_rows'] == 127
            and independent['observed_rows'] == 3030 and independent['fixed_candidates'] == 12,
            'Independent stored-data arithmetic receipt incomplete')
    require(independent['category_parent_counts']['UNRESOLVED'] == 25
            and independent['unresolved_samples'] == 686, 'Unknown denominator changed')
    require(independent['model_visual_labels_verified_as_physical_GT'] is False
            and independent['reviewer_kind'] == 'model' and independent['independent_blind_review'] is False,
            'Model interpretation promoted to physical/human truth')
    for path, expected in independent['input_sha256'].items():
        check_hash(path, expected)
    require(max(independent['max_absolute_differences'].values()) <= 1e-9,
            'Independent sum tolerance failure')
    require(independent['root_review_module_imports'] == 0 and independent['new_cost_evaluations'] == 0
            and independent['new_pose_solves'] == 0, 'Independent verification used prohibited calculations')
    require(sha(private / 'MASK_FIRST_SUPPORT_DECISIONS.json') == sha(public / 'SUPPORT_DECISIONS.json'),
            'Model decision input bytes changed')

    reviews = [json.loads(line) for line in (public / 'SUPPORT_REVIEW.jsonl').read_text().splitlines() if line.strip()]
    samples = rows(public / 'OBSERVED_SAMPLE_REVIEW.csv.gz')
    categories = rows(public / 'COST_SUPPORT_SUMMARY.csv')
    require(len(reviews) == 127 and len({(r['id'], r['segment_id']) for r in reviews}) == 127,
            'Parent denominator/uniqueness differs')
    require(len(samples) == 3030 and len({(r['id'], int(r['sample_index'])) for r in samples}) == 3030,
            'Observed sample denominator/uniqueness differs')
    require(len(categories) == 18, 'All category rows including empty categories are required')
    require(sum(r['support_category'] == 'UNRESOLVED' for r in reviews) == 25
            and sum(r['support_category'] == 'UNRESOLVED' for r in samples) == 686,
            'Unresolved rows lost')
    require(all(r['reviewer_kind'] == 'model' and r['reviewed'] is True
                and r['diagnostic_only_not_GT_or_training_target'] is True for r in reviews),
            'Model-review diagnostic limitation differs')
    category_by_key = {(r['id'], r['support_category']): r for r in categories}
    sample_case_counts = Counter(r['id'] for r in samples)
    for ident in IDS:
        own = [r for r in samples if r['id'] == ident]
        for category in CATEGORIES:
            row = category_by_key[(ident, category)]
            subset = [r for r in own if r['support_category'] == category]
            require(int(row['observed_sample_count']) == len(subset)
                    and int(row['unreviewed_observed_sample_count']) == 0
                    and int(row['denominator_observed_samples']) == len(own),
                    'Category counts/full denominator differ')
            close(row['signed_Delta'], math.fsum(float(r['Delta']) for r in subset), 'Category signed sum differs')
            close(row['absolute_sample_Delta_sum'], math.fsum(abs(float(r['Delta'])) for r in subset),
                  'Category absolute sum differs')

    lineage = read(private / 'mask_lineage_v2/MASK_LINEAGE_SUMMARY.json')
    require(lineage['all_mask_outline_checks_pass'] is True
            and lineage['used_parent_segments'] == 127 and lineage['observed_samples'] == 3030,
            'Original mask-to-stored-outline reproduction incomplete')
    for frame in lineage['frames']:
        value = read(private / 'mask_lineage_v2' / frame['path'])
        require(all(value['checks'].values()), 'A mask lineage parity check failed')
        require(sample_case_counts[value['id']] == len(value['sample_parent_segment_ids']), 'Parent mapping sample count differs')
        for path, expected in value['source_sha256'].items():
            check_hash(source / path, expected)
    # The initial historical QA intentionally keeps its then-missing report link.
    # Final QA is a separate receipt after the report exists.
    gallery = read(public / 'evidence/GALLERY_FINAL_QA.json')
    require(gallery['passed'] is True and gallery['cases'] == 6
            and gallery['original_candidate_panels_checked'] == 12
            and gallery['original_observed_samples_preserved'] == 3030
            and gallery['parent_segments_preserved'] == 127
            and gallery['exact_original_RGB_crop_pixel_checks'] == 6,
            'Gallery coordinate/original-pixel QA incomplete')
    require(gallery['root_report_links_pending_only'] == [], 'Final gallery still defers report link verification')
    require(gallery['independent_homogeneous_projection_max_abs_px'] <= 1e-9
            and gallery['model_sample_indices_never_paired'] is True,
            'Fixed projection or separate model sample QA failed')

    guarded = []
    for label, value in (
        ('bindings_v3/INPUT_BINDINGS.json', original_binding),
        ('mask_lineage_v2/MASK_LINEAGE_SUMMARY.json', lineage),
        ('REGISTRATION_RUNTIME_GUARD_V2.json', read(private / 'REGISTRATION_RUNTIME_GUARD_V2.json')),
        ('REVIEW_AGGREGATION_RECEIPT.json', read(private / 'REVIEW_AGGREGATION_RECEIPT.json')),
        ('evidence/GALLERY_FINAL_QA.json', gallery),
        ('gallery/INITIAL_GALLERY_RECEIPT.json', read(public / 'gallery/INITIAL_GALLERY_RECEIPT.json')),
        ('gallery/GALLERY_RECEIPT.json', read(public / 'gallery/GALLERY_RECEIPT.json')),
    ):
        guarded.append(guard_check(value, label))
    cost = read(public / 'COST.json')
    zero_fields = ('GPU_model_jobs', 'new_training', 'new_model_forwards', 'SAM_reruns',
                   'new_line_detections', 'PnP_or_pose_optimization', 'new_candidates',
                   'new_cost_evaluations', 'new_selection_or_reranking', 'full_952_replay',
                   'full_11603_cost_recomputation', 'human_review_seconds')
    require(all(cost[k] == 0 for k in zero_fields), 'A forbidden execution count is nonzero')
    require(cost['maximum_concurrent_heavy_CPU_jobs'] <= 2
            and cost['process_CPU_sum_seconds'] <= cost['CPU_analysis_cap_seconds'] == 7200,
            'CPU resource cap exceeded')
    require(cost['human_review_performed'] is False and cost['deployment_inference_latency_or_FPS_measured'] is False,
            'Model review/diagnostic CPU time misrepresented')
    close(cost['elapsed_sum_seconds'], math.fsum(r['elapsed_seconds'] for r in cost['measured_stages']),
          'Measured elapsed stage sum differs')
    close(cost['process_CPU_sum_seconds'], math.fsum(r['process_cpu_seconds'] or 0 for r in cost['measured_stages']),
          'Measured process CPU stage sum differs')
    for record in read(public / 'PACKAGING_BINDINGS.json')['records']:
        check_hash(public / record['public_path'], record['public_sha256'])
        check_hash(private / record['original_alias'].removeprefix('PRIVATE_AUDIT/'), record['original_sha256'])

    # Preserve the original user's dirty workspace exactly, not just its six files.
    start = read(private / 'START_BEFORE_FETCH.json')
    require(git(asset, 'rev-parse', 'HEAD').decode().strip() == start['original_HEAD'], 'Original workspace HEAD changed')
    require(git(asset, 'branch', '--show-current').decode().strip() == start['original_branch'], 'Original workspace branch changed')
    tracked = git(asset, 'status', '--porcelain', '-z', '--untracked-files=no')
    all_status = git(asset, 'status', '--porcelain', '-z')
    require(hashlib.sha256(tracked).hexdigest() == start['original_tracked_status_SHA256'], 'Original tracked NUL status changed')
    require(hashlib.sha256(all_status).hexdigest() == start['original_all_status_SHA256'], 'Original full NUL status changed')
    require(all_status == (private / 'ORIGINAL_STATUS_NUL.bin').read_bytes(), 'Original NUL status bytes changed')
    for path, expected in start['protected_files_SHA256'].items():
        check_hash(asset / path, expected)
    changed = git(source, 'diff', '--name-only', BASE, '--').decode().splitlines()
    untracked = git(source, 'ls-files', '--others', '--exclude-standard').decode().splitlines()
    asset_symlinks = ('data', '_docs/experiments/pallet_visibility_curriculum_20261013_v1')
    for relative in asset_symlinks:
        link = source / relative
        require(link.is_symlink() and link.resolve() == (asset / relative).resolve(),
                'Intentional asset link must be an exact symlink to ASSET_ROOT: ' + relative)
    unexpected = [p for p in changed + untracked
                  if not p.startswith((SOURCE_NAMESPACE, PUBLIC_NAMESPACE)) and p not in asset_symlinks]
    require(not unexpected, 'New worktree changes outside approved namespaces: ' + repr(unexpected))
    links = check_public_links(public, allow_pending_validation_outputs=True)
    privacy = privacy_scan(public)

    result = dict(schema='OUTLINE_PHYSICAL_SUPPORT_FINAL_VALIDATION_V1', passed=True,
        source_result_commit=ORIGINAL, execution_base_commit=BASE,
        representative_ids=IDS, fixed_cases=6, original_fixed_candidate_pointers=12,
        used_parent_segments=127, shared_observed_samples=3030,
        unresolved_parent_segments=25, unresolved_observed_samples=686,
        duplicate_or_missing_original_sample_rows=0, unreviewed_observed_samples=0,
        category_signed_and_absolute_sums_pass=True, zero_count_categories_preserved=True,
        model_labels_verified_as_physical_truth=False, human_review_claimed=False,
        independent_arithmetic_receipt=dict(path='PRIVATE_AUDIT/INDEPENDENT_REVIEW_CHECK.json', sha256=sha(independent_path)),
        independent_max_absolute_differences=independent['max_absolute_differences'],
        exact_original_public_artifacts_checked=required_count,
        original_legacy_sources_SHA_checked=len(upstream['legacy_source_sha256']),
        original_private_prediction_candidate_reference_artifacts_SHA_checked=len(upstream['private_parent_artifacts']),
        unchanged_annotation_SHA_checked=len(score_seal['annotation_sha256']),
        unchanged_reference_source_SHA_checked=len(score_seal['reference_bindings']),
        protected_dirty_workspace_files_SHA_checked=len(start['protected_files_SHA256']),
        original_workspace_HEAD_branch_NUL_status_exact=True,
        intentional_asset_symlink_targets_verified=list(asset_symlinks),
        asset_symlink_write_protection_is_usage_policy_not_OS_enforcement=True,
        worktree_changes_outside_new_namespaces=[], operational_defaults_papers_predictions_references_edited=False,
        fixed_native_physical_Q_dimensions_and_K_preserved=True,
        original_mask_transform_crop_and_observed_sample_order_pass=True,
        model_to_observed_own_samples_never_paired=True,
        public_relative_links_checked=links, privacy=privacy,
        runtime_guard_receipts=guarded,
        no_new_training_forward_mask_line_pose_cost_candidate_selection=True,
        new_performance_result=False, final_validator_scientific_recomputation=False,
        measured_stages_CPU_within_7200_seconds=True,
        artifact_sha256={str(p.relative_to(public)): sha(p) for p in sorted(public.rglob('*'))
                         if p.is_file() and p.name not in ('VALIDATION.json', 'VALIDATION.md')},
        validator_source_SHA256=sha(Path(__file__)),
        elapsed_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time() - cpu,
        runtime_guard=guard.receipt())
    guard_check(result, 'final validator')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (public / 'VALIDATION.json').write_text(json.dumps(sanitize(result), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (public / 'VALIDATION.md').write_text(
        '# 최종 검산\n\n'
        'PASS. 원 대표 6장·고정 후보 12개, 부모 선분 127개·공유 관측 표본 3,030개를 보존했습니다. '
        '미확정 25선분·686표본과 빈 범주도 전체 분모에 남아 있습니다. '
        '범주별 signed/absolute 기여 합은 기존 표본 비용과 허용 오차 1e-9 안에서 일치합니다.\n\n'
        '원 RGB crop 픽셀·후보별 Q/등록 치수·원 마스크 변환·표본 순서·방향별 비용은 저장된 독립 검산을 통과했습니다. '
        '관측→모델 표본만 대응시키며 모델→관측 표본은 후보별로 구분합니다. '
        '분류는 모델의 영상 해석이고 물리 정답·사람 검수·독립 블라인드 검증이 아닙니다.\n\n'
        '원 소스·예측·참조·어노테이션 SHA와 기존 dirty workspace의 HEAD·branch·NUL 상태 및 보호 파일이 일치합니다. '
        '새 namespace 밖 변경은 없고, 금지 호출·새 추론·마스크·자세·비용·후보·재선택·성능 산출은 0회입니다. '
        '공개 상대 링크와 압축 텍스트 CRC·비공개 절대경로 검사를 통과했습니다.\n\n'
        '상세 수치·SHA·실측 시간: [VALIDATION.json](VALIDATION.json). '
        '검토 근거: [보고서](REPORT_KO.md), [독립 집계 receipt](evidence/INDEPENDENT_REVIEW_CHECK.json), '
        '[원영상 갤러리](gallery/index.html).\n')
    # Include the two final validation files in the privacy/link check too.
    check_public_links(public)
    privacy_scan(public)
    return result


def validate(args):
    with SupportAuditOnly() as guard:
        return _validate(args, guard)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('source-root', 'asset-root', 'score-private', 'state-private',
                'analysis-private', 'public', 'output-private'):
        parser.add_argument('--' + key, required=True)
    result = validate(parser.parse_args())
    print(json.dumps({k: result[k] for k in ('passed', 'fixed_cases', 'used_parent_segments',
                                           'shared_observed_samples', 'elapsed_seconds', 'process_cpu_seconds')}))
