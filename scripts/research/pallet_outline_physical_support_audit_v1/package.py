"""Publish small audit receipts with explicit path aliases; never rerun science.

Historical attempt receipts are optional for a clean reproduction. Their private
original SHA is retained when a public copy needs path redaction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def package(source_root, asset_root, score_private, state_private, analysis_private, output):
    started, cpu = time.monotonic(), time.process_time()
    source, asset, score, state, analysis, public = map(Path, (
        source_root, asset_root, score_private, state_private, analysis_private, output))
    # Longest prefixes first: do not disclose local account or temporary paths.
    aliases = sorted(((str(p), name) for p, name in (
        (analysis, 'PRIVATE_AUDIT'), (score, 'PRIVATE_SCORE'),
        (state, 'PRIVATE_STATE'), (source, 'SOURCE_ROOT'), (asset, 'ASSET_ROOT'))),
        key=lambda item: len(item[0]), reverse=True)

    def sanitize(value):
        if isinstance(value, str):
            for path, alias in aliases:
                value = value.replace(path, alias)
            if '/home/' in value or '/tmp/' in value:
                raise ValueError('Unbound private path in public receipt')
            return value
        if isinstance(value, list):
            return [sanitize(v) for v in value]
        if isinstance(value, dict):
            return {sanitize(k): sanitize(v) for k, v in value.items()}
        return value

    copied = []

    def copy_receipt(relative, destination, required=True):
        path = analysis / relative
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return
        value = sanitize(read(path))
        write(public / destination, value)
        copied.append(dict(original_alias='PRIVATE_AUDIT/' + relative,
                           original_sha256=sha(path), public_path=destination,
                           public_sha256=sha(public / destination),
                           derivation='JSON serialization plus literal local-path alias replacement only'))

    # Exact timestamps and semantic decisions remain unchanged. SUPPORT_DECISIONS
    # itself is kept byte-for-byte by review.py, not rewritten by this packager.
    for name in ('REVIEW_PROTOCOL.json', 'RAW_RGB_INITIAL_REVIEW.json',
                 'MASK_FIRST_SUPPORT_DECISIONS.json', 'FINAL_VISUAL_REVIEW.json'):
        copy_receipt(name, 'evidence/review/' + name)
    for name in ('REVIEW_AGGREGATION_RECEIPT.json', 'INDEPENDENT_REVIEW_CHECK.json',
                 'REGISTRATION_AUDIT_V2.json', 'REGISTRATION_RUNTIME_GUARD_V2.json',
                 'OCCLUDER_PRECEDENT_SOURCE_AUDIT.json', 'ROOT_DOCUMENT_AND_LEDGER_READ.json',
                 'FETCH_AND_INSTRUCTION_CHECK.json', 'START_BEFORE_FETCH.json',
                 'GALLERY_FINAL_QA.json'):
        copy_receipt(name, 'evidence/' + name, required=name not in (
            'ROOT_DOCUMENT_AND_LEDGER_READ.json', 'FETCH_AND_INSTRUCTION_CHECK.json',
            'START_BEFORE_FETCH.json', 'GALLERY_FINAL_QA.json'))
    for path in sorted((analysis / 'mask_lineage_v2').glob('*.json')):
        copy_receipt('mask_lineage_v2/' + path.name, 'evidence/mask_lineage/' + path.name)
    attempts = (
        'bindings_v1/INPUT_BINDINGS.json', 'bindings_v2/INPUT_BINDINGS.json',
        'bindings_v3/INPUT_BINDINGS.json', 'mask_lineage_v1/MASK_LINEAGE_SUMMARY.json',
        'mask_lineage_v2/MASK_LINEAGE_SUMMARY.json', 'REGISTRATION_AUDIT.json',
        'REGISTRATION_AUDIT_V2.json', 'GALLERY_COST_AND_ATTEMPTS.json')
    for relative in attempts:
        copy_receipt(relative, 'evidence/attempts/' + relative.replace('/', '__'), required=False)

    bindings_path = analysis / 'bindings_v3/INPUT_BINDINGS.json'
    bindings = sanitize(read(bindings_path))
    seal_path = score / 'cost_v1/COST_SEAL.json'
    seal = read(seal_path)
    dense_entries = {v['path']: v for v in seal['dense_frame_journals']}
    direct = []
    for case in bindings['cases']:
        witness = source / '_docs/experiments/pallet_selection_score_diagnosis_v1/gallery/outline_witness' / f"witness_{case['ordinal']:02d}_samples.csv.gz"
        dense_rel = case['candidates'][0]['original_dense_pointer']['path']
        dense = score / 'cost_v1' / dense_rel
        if sha(dense) != dense_entries[dense_rel]['sha256']:
            raise ValueError('Selected fixed dense frame changed')
        direct.append(dict(id=case['id'], witness_csv=dict(
            path=str(witness.relative_to(source)), sha256=sha(witness)),
            selected_dense_frame=dict(path='PRIVATE_SCORE/cost_v1/' + dense_rel,
                                      sha256=sha(dense)),
            candidate_pointers=[v['original_dense_pointer'] for v in case['candidates']]))
    bindings['public_derivation'] = dict(original_private_snapshot_sha256=sha(bindings_path),
        local_path_aliases=['SOURCE_ROOT', 'ASSET_ROOT', 'PRIVATE_SCORE', 'PRIVATE_STATE', 'PRIVATE_AUDIT'],
        absolute_paths_not_published=True, original_snapshot_unchanged=True)
    bindings['direct_selected_sample_bindings'] = direct
    bindings['cost_seal'] = dict(path='PRIVATE_SCORE/cost_v1/COST_SEAL.json', sha256=sha(seal_path))
    bindings['execution_base_commit'] = '61343a24130c553f063e0f6ade732d5e810e6e55'
    bindings['instruction_SHA256'] = '0c826955ebea3006f804dd4c3ac59cee9d48ebbc900d43543a6de66f5cf1e93b'
    bindings['audit_sources_SHA256'] = {str(p.relative_to(source)): sha(p) for p in sorted(
        (source / 'scripts/research/pallet_outline_physical_support_audit_v1').glob('*.py'))}
    write(public / 'INPUT_BINDINGS.json', bindings)
    measured = []
    for relative in attempts[:-1] + ('REVIEW_AGGREGATION_RECEIPT.json', 'INDEPENDENT_REVIEW_CHECK.json'):
        path = analysis / relative
        if path.exists():
            value = read(path)
            if 'elapsed_seconds' in value:
                measured.append(dict(receipt='PRIVATE_AUDIT/' + relative,
                    elapsed_seconds=value['elapsed_seconds'],
                    process_cpu_seconds=value.get('process_cpu_seconds')))
    # Include all four measured render attempts and the existing gallery QA once.
    gallery_cost = analysis / 'GALLERY_COST_AND_ATTEMPTS.json'
    if gallery_cost.exists():
        measured.extend(sanitize(read(gallery_cost))['measured_stages'])
    else:
        # A new reproduction need not possess the original failed render history.
        for path, alias in ((analysis / 'initial_gallery/INITIAL_GALLERY_RECEIPT.json',
                             'PRIVATE_AUDIT/initial_gallery/INITIAL_GALLERY_RECEIPT.json'),
                            (public / 'gallery/GALLERY_RECEIPT.json',
                             'PUBLIC_AUDIT/gallery/GALLERY_RECEIPT.json')):
            if path.exists():
                value = read(path)
                measured.append(dict(receipt=alias, elapsed_seconds=value['elapsed_seconds'],
                    process_cpu_seconds=value['process_cpu_seconds']))
    # Fresh reproduction has just its own receipts; historical attempts optional.
    for filename in ('GALLERY_FINAL_QA.json',):
        path = analysis / filename
        if path.exists():
            value = read(path)
            measured.append(dict(receipt='PRIVATE_AUDIT/' + filename,
                elapsed_seconds=value['elapsed_seconds'], process_cpu_seconds=value['process_cpu_seconds']))
    cost = dict(schema='PHYSICAL_SUPPORT_AUDIT_ACTUAL_COST_V1', measured_stages=measured,
        elapsed_sum_seconds=sum(v['elapsed_seconds'] for v in measured),
        process_CPU_sum_seconds=sum(v['process_cpu_seconds'] or 0 for v in measured),
        CPU_analysis_cap_seconds=7200, GPU_model_jobs=0, maximum_concurrent_heavy_CPU_jobs=2,
        new_training=0, new_model_forwards=0, SAM_reruns=0, new_line_detections=0,
        PnP_or_pose_optimization=0, new_candidates=0, new_cost_evaluations=0,
        new_selection_or_reranking=0, full_952_replay=0, full_11603_cost_recomputation=0,
        human_review_seconds=0, human_review_performed=False,
        model_visual_review='Actual view_image review; exact model attention time not instrumented separately',
        file_downloads='No dataset/model download; existing local originals and receipts reused; git fetch separately uninstrumented',
        analysis_includes_read_IO_and_display=True, measured_wait_seconds=None,
        editing_git_network_model_reasoning_time='Not added as CPU stage time; task wall includes these',
        measurement_cutoff='Measured stage sum through packaging; subsequent final-validator CPU/wall is recorded in VALIDATION.json separately to avoid self-referential output hashes.',
        final_validation_cost_receipt='VALIDATION.json elapsed_seconds / process_cpu_seconds',
        deployment_inference_latency_or_FPS_measured=False,
        historical_attempt_receipts_present=gallery_cost.exists(),
        superseded_attempts='Original execution retained all attempts privately; fresh reproductions are not required to fabricate that history.',
        historical_notes=dict(
            initial_binding_attempt_no_runtime_guard='Original v1 was metadata reads only; v2/v3 added explicit runtime guard. Not presented as a guarded run.',
            attempt_changes='Original execution: crop top cone omitted in case04, uniform display-only ROI expansion; font redraw changed labels only; actual observer lineage audit added; previous files preserved.'))
    start_path = analysis / 'START_BEFORE_FETCH.json'
    if start_path.exists():
        cost['task_wall_seconds_at_packaging'] = time.time() - read(start_path)['start_epoch']
    cost['packaging_elapsed_seconds'] = time.monotonic() - started
    cost['packaging_process_CPU_seconds'] = time.process_time() - cpu
    write(public / 'COST.json', cost)
    write(public / 'PACKAGING_BINDINGS.json', dict(schema='PHYSICAL_SUPPORT_PUBLIC_RECEIPTS_V1',
        records=copied, source_namespace_unchanged=True, private_NPZ_dense_weights_not_copied=True))
    return dict(copied_receipts=len(copied), measured_elapsed_sum_seconds=cost['elapsed_sum_seconds'],
                measured_process_CPU_sum_seconds=cost['process_CPU_sum_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source-root', 'asset-root', 'score-private', 'state-private', 'analysis-private', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source_root, args.asset_root, args.score_private,
        args.state_private, args.analysis_private, args.output)))
