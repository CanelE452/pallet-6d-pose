"""Seal the current SOURCE checkpoint; do not run feature production or fitting."""
from pathlib import Path
import hashlib
import json
import os

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')


def safe(p):
    p = Path(p)
    s = str(p).lower()
    assert not any(x in s for x in ('v112', '__pycache__', '.pyc'))
    assert not p.is_symlink()
    return p


def binding(p):
    p = safe(p)
    h = hashlib.sha256()
    n = 0
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
            n += len(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': n}


def main():
    top = [
        'V237_CONTEXT_COMPOSITION_A', 'V237_RELATIONAL_QUERY_DESIGN_B',
        'V237_SOURCE_INPUT_MANIFEST_C', 'V237_ROOT_SOURCE_CONTEXT_INPUTS',
        'V237_INPUT_REBUILD_REVIEW_C', 'V238_ROOT_CONTEXT_RESIDUAL',
        'V238_CONTEXT_RESIDUAL_TRAINER_A', 'V238_TRAINER_REVIEW_B',
        'V238_TRAIN_COMPLETION_QA_B', 'V238_SOURCE_REPORT_C',
        'V238_SOURCE_GALLERY_A', 'V238_FEATURE_ROUTE_DIAG',
        'V238_ALL_FIT_DESCRIPTOR_DIAG', 'V239_ROOT_CANONICAL_CONTEXT',
        'V239_CANONICAL_FEATURE_PRODUCER_B', 'V239_CANONICAL_PAIRED_TRAINER_A',
        'V239_CANONICAL_SOURCE_QA_C',
    ]
    excluded = [
        'V237_ROOT_SOURCE_CONTEXT_INPUTS/original_source_inputs_v1/' + n
        for n in ('NODE_RGB96.npy', 'NODE_VALID96.npy',
                  'CONTEXT_RGB224_F32.npy', 'CONTEXT_VALID224.npy')
    ] + [
        'V238_ROOT_CONTEXT_RESIDUAL/context_residual_source_fit_v1/' + n
        for n in ('NODE_CLS.npy', 'CONTEXT_CLS.npy')
    ]
    # Bind omitted cache bytes so the publication describes exactly what is local.
    exclusions = [{'relative_path': rel, **binding(R / rel)} for rel in excluded]
    extra_names = [
        'build_v237_source_context_inputs.py', 'render_v237_fixed_source_inputs.py',
        'assemble_v238_actual_entry.py', 'diagnose_v238_feature_routes.py',
        'diagnose_v238_all_fit_prefix.py', 'verify_v238_saved_head_cpu.py',
        'build_v238_publication.py', 'create_v239_publication_plan.py',
    ]
    completed = [
        binding(R / 'V237_ROOT_SOURCE_CONTEXT_INPUTS/original_source_inputs_v1/COMPLETE.json'),
        binding(R / 'V237_INPUT_REBUILD_REVIEW_C/REVIEW.json'),
        binding(R / 'V238_FEATURE_ROUTE_DIAG/DIAGNOSIS.json'),
        binding(R / 'V238_ALL_FIT_DESCRIPTOR_DIAG/DIAGNOSIS.json'),
    ]
    # Each is a completion of a specifically scoped check, never model success.
    for b in completed:
        d = json.loads(Path(b['path']).read_bytes())
        assert d['completed'] is True and d['PASS'] is True
    fail = json.loads((R / 'V238_ROOT_CONTEXT_RESIDUAL/context_residual_source_fit_v1/FAILED_EXECUTION.json').read_bytes())
    assert fail['completed'] is False and fail['PASS'] is False
    protocol = binding(R / 'V239_ROOT_CANONICAL_CONTEXT/PROTOCOL_BEFORE_CANONICAL_FEATURES_OR_PAIRED_FIT.json')
    assert protocol['sha256'] == '4cbbd1ede0420e1cc6dfe83446b2204141bd99ac29781836b9663c49c77fa021'
    plan = {
        'schema': 'V239_PREPARED_NOT_TRAINED_PUBLICATION_PLAN',
        'approved_publication': True, 'authorization': 'Human requested current work on main and continuous publication.',
        'source_only': True, 'versions': ['V237', 'V238', 'V239'],
        'top_directories': top, 'exclude_relative_paths': excluded,
        'excluded_large_cache_bindings': exclusions,
        'required_completed_receipts': completed,
        'extra_files': [{'storage': 'tools/' + n, 'source': binding(R / n)} for n in extra_names],
        'readme': binding(R / 'V239_PUBLICATION_README_KO.md'),
        'third_party_notice': binding(R / 'V239_PUBLICATION_NOTICES.md'),
        'source_gate_pass': False,
        'scientific_scope': {
            'V237': 'Completed original SOURCE RGB reconstruction and independent input QA; fixed four-case visualization.',
            'V238': 'Actual attempt failed active-feature alignment before optimizer; diagnostic completion is not classifier performance.',
            'V239': 'New canonical producer and two-arm trainer prepared only. No actual feature production, optimizer updates or new DEV predictions.',
            'model_adopted': False, 'new_current_or_held_payload': 0,
            'new_PnP': False, 'tolerance_relaxed': False,
        },
        'V239_protocol': protocol,
        'publication_limits': [
            'Absolute source paths in preserved artifacts are provenance pointers, not portable files. No pointer targets are followed by package verification.',
            'Large local RGB/valid and failed intermediate feature caches omitted with SHA/size and rebuild code; no claim that all local bytes are in Git.',
            'No new trained V238/V239 classifier or performance result exists. Copied V238 CONTROL is the historical V233 model.',
            'Prepared CODE/toy PASS is not actual SOURCE execution. Actual fit requires a separate verified entry.',
        ],
    }
    out = R / 'V239_PUBLICATION_PLAN.json'
    with out.open('x') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'completed': True, 'plan': binding(out), 'excluded_caches': len(exclusions)}))


if __name__ == '__main__':
    main()
