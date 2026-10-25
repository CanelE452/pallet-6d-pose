"""Seal the actual canonical SOURCE comparison and its reproducible saved outputs."""
from pathlib import Path
import hashlib
import json

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')


def safe(p):
    p = Path(p)
    assert not any(s in str(p).lower() for s in ('v112', '__pycache__', '.pyc'))
    assert not p.is_symlink()
    return p


def binding(p):
    p = safe(p); h = hashlib.sha256(); n = 0
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block); n += len(block)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': n}


def main():
    tops = ['V239_ROOT_CANONICAL_CONTEXT', 'V239_CANONICAL_FEATURE_PRODUCER_B',
            'V239_CANONICAL_PAIRED_TRAINER_A', 'V239_PRODUCER_CODE_REVIEW_A',
            'V239_TRAINER_CODE_REVIEW_B', 'V239_PAIR_COMPLETION_QA_B',
            'V239_CPU_REPLAY_CODE_REVIEW_B', 'V239_FEATURE_PACKET_QA_C',
            'V239_SOURCE_GALLERY_A', 'V240_SOURCE_REPORT_C',
            'V240_ROOT_COMPLETION_QA_B', 'V240_ROOT_SOURCE_GALLERY_A',
            'V240_SOURCE_ERROR_JOIN_B']
    feature = json.loads((R / 'V239_ROOT_CANONICAL_CONTEXT/canonical_source_features_v1/COMPLETE.json').read_bytes())
    omitted = 'V239_ROOT_CANONICAL_CONTEXT/canonical_source_features_v1/NODE_PIXEL_FEATURES.npy'
    omitted_binding = feature['features']['NODE_PIXEL_FEATURES.npy']
    assert omitted_binding['path'] == str(R / omitted)
    assert feature['completed'] and feature['PASS']
    req = ['V239_ROOT_CANONICAL_CONTEXT/canonical_source_features_v1/COMPLETE.json',
           'V239_ROOT_CANONICAL_CONTEXT/canonical_pair_source_fit_v1/COMPLETE.json',
           'V239_FEATURE_PACKET_QA_C/actual_feature_review_v1/REVIEW.json',
           'V240_ROOT_COMPLETION_QA_B/actual_review_v1/REVIEW.json',
           'V240_SOURCE_REPORT_C/results/COMPLETE.json',
           'V239_ROOT_CANONICAL_CONTEXT/CPU_REPLAY_CONTROL.json',
           'V239_ROOT_CANONICAL_CONTEXT/CPU_REPLAY_TREATMENT.json',
           'V239_SOURCE_GALLERY_A/gallery_v1/COMPLETE.json',
           'V240_SOURCE_ERROR_JOIN_B/REVIEW.json']
    receipts = [binding(R / n) for n in req]
    for b in receipts:
        d = json.loads(Path(b['path']).read_bytes())
        assert d['completed'] is True and d['PASS'] is True
    report = json.loads((R / 'V240_SOURCE_REPORT_C/results/SUMMARY.json').read_bytes())
    assert report['gate']['numeric_gate_pass'] is False
    extras = [
        ('assemble_v239_actual_entries.py', 'code/assemble_v239_actual_entries.py'),
        ('assemble_v239_saved_feature_qa.py', 'code/assemble_v239_saved_feature_qa.py'),
        ('verify_v239_saved_head_cpu.py', 'code/verify_v239_saved_head_cpu.py'),
        ('V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py', 'code/relational_query_head.py'),
        ('V237_ROOT_SOURCE_CONTEXT_INPUTS/original_source_inputs_v1/INPUT_GEOMETRY.npz', 'replay/INPUT_GEOMETRY.npz'),
        ('V233_ROOT_CROP_ADAPTATION/crop_samples_v1/samples.npz', 'replay/ORIGINAL_SAMPLES.npz'),
        ('V240_PUBLICATION_STATUS.json', 'STATUS.json'),
        ('create_v240_publication_plan.py', 'code/create_v240_publication_plan.py'),
    ]
    plan = {'schema': 'V240_ACTUAL_CANONICAL_SOURCE_PUBLICATION_PLAN',
            'approved_publication': True, 'authorization': 'Human authorized continuous main publication of attempts, results, code and visualizations.',
            'source_only': True, 'versions': ['V239', 'V240'],
            'top_directories': tops, 'exclude_relative_paths': [omitted],
            'excluded_large_cache_bindings': [{'relative_path': omitted, **omitted_binding,
                'reason': 'Duplicate pixel1154 cache. Six public classifier input arrays are sufficient for both frozen-head CPU replays. Original seven-array QA was completed locally.'}],
            'required_completed_receipts': receipts,
            'extra_files': [{'storage': dst, 'source': binding(R / src)} for src, dst in extras],
            'readme': binding(R / 'V240_PUBLICATION_README_KO.md'),
            'third_party_notice': binding(R / 'V240_PUBLICATION_NOTICES.md'),
            'source_gate_pass': False,
            'scientific_scope': {'canonical_features_complete': True, 'both_models_completed1000': True,
                                 'independent_QA_PASS': True, 'SOURCE_gate_PASS': False,
                                 'model_adopted': False, 'new_current_comparison': False,
                                 'new_current_or_held_payload': 0, 'new_PnP': False,
                                 'macro_gain_pp': report['gate']['macro_gain_pp'],
                                 'YCB_DEV_E': {'CONTROL_correct': 0, 'TREAT_correct': 1, 'truth': 57},
                                 'q_L_native_U_and_split_unchanged': True},
            'limits': ['SOURCE internal DEV is exposed/reused; not fresh blind or current pallet performance.',
                       'YCB associated-r AUX/nativeUNKNOWN is not pallet physical native gold.',
                       'Package/QA completion does not mean model success; source gate failed.',
                       'No next CF generation or next model training result exists.',
                       'Absolute metadata source pointers are provenance only and are not followed by verification.']}
    p = R / 'V240_PUBLICATION_PLAN.json'
    with p.open('x') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'completed': True, 'plan': binding(p), 'required_scoped_completions': len(receipts)}))


if __name__ == '__main__':
    main()
