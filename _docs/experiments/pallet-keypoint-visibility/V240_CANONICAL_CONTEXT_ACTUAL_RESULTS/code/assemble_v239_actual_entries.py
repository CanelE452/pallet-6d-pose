"""Authorize exact SOURCE entries after scoped independent QA; no array/model reads."""
from pathlib import Path
import argparse
import hashlib
import json
import datetime

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
ROOT = R / 'V239_ROOT_CANONICAL_CONTEXT'
PROTOCOL_SHA = '4cbbd1ede0420e1cc6dfe83446b2204141bd99ac29781836b9663c49c77fa021'


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(x in str(p).lower() for x in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation'))
    for q in reversed((p, *p.parents)):
        assert not q.is_symlink()
    return p


def bind(p):
    p = safe(p)
    b = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(b).hexdigest(), 'size_bytes': len(b)}


def read_binding(b):
    p = safe(b['path'])
    actual = bind(p)
    assert actual['sha256'] == b['sha256']
    if 'size_bytes' in b:
        assert actual['size_bytes'] == b['size_bytes']
    return json.loads(p.read_bytes())


def write(p, obj):
    with safe(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return bind(p)


def append_binding(items, b):
    same = [v for v in items if v['path'] == b['path']]
    if same:
        assert all(v['sha256'] == b['sha256'] for v in same)
    else:
        items.append(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['feature', 'train'], required=True)
    ap.add_argument('--code-review', required=True)
    ap.add_argument('--code-review-sha256', required=True)
    ap.add_argument('--feature-qa')
    ap.add_argument('--feature-qa-sha256')
    a = ap.parse_args()
    review = bind(a.code_review)
    assert review['sha256'] == a.code_review_sha256
    rd = read_binding(review)
    assert rd['completed'] is True and rd['PASS'] is True
    protocol = bind(ROOT / 'PROTOCOL_BEFORE_CANONICAL_FEATURES_OR_PAIRED_FIT.json')
    assert protocol['sha256'] == PROTOCOL_SHA
    input_qa = bind(R / 'V237_INPUT_REBUILD_REVIEW_C/REVIEW.json')
    iq = read_binding(input_qa)
    assert iq['completed'] and iq['PASS']
    scope = {'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_only': True, 'automatic_retry': False, 'stage': a.stage,
             'root_protocol': protocol, 'independent_code_review': review,
             'V237_independent_input_QA': input_qa,
             'original_q_L_native_U_and_internal_FIT_DEV_preserved': True,
             'new_current_or_held_payload': 0, 'PnP_or_coordinate_regression': False,
             'source_DEV_pretrained_features_or_fixed_output_only': True,
             'current_admission': False, 'model_adopted': False}
    if a.stage == 'feature':
        template = R / 'V239_CANONICAL_FEATURE_PRODUCER_B/ROOT_FEATURE_ENTRY_FALSE_TEMPLATE.json'
        e = json.loads(template.read_bytes())
        assert e['approved_canonical_SOURCE_feature_build'] is False
        e['approved_canonical_SOURCE_feature_build'] = True
        scope.update({'classifier_or_optimizer_updates': 0,
                      'old_prefix_or_old_final_CONTROL_payload_allowed': False,
                      'feature_scope': 'Original SOURCE known7617/all15040 nodes and context; same official frozen DINO, no head or statistics fit.',
                      'masked_S_U_context': 'Input-only support; target -1 retained, no semantic masking.',
                      'canonical_sharing': 'One full96 forward output supplies both local first384 and NODE_CLS. All known7617 byteexact, no replacement of old mismatches.'})
        sb = write(ROOT / 'ACTUAL_CANONICAL_FEATURE_SCOPE.json', scope)
        e['root_scope'] = sb
        for b in (sb, review, input_qa):
            append_binding(e['metadata_whitelist'], b)
        out = ROOT / 'ACTUAL_CANONICAL_FEATURE_ENTRY.json'
    else:
        assert a.feature_qa and a.feature_qa_sha256
        fq = bind(a.feature_qa)
        assert fq['sha256'] == a.feature_qa_sha256
        qd = read_binding(fq)
        assert qd['completed'] is True and qd['PASS'] is True
        complete = bind(ROOT / 'canonical_source_features_v1/COMPLETE.json')
        packet = read_binding(complete)
        assert packet['completed'] and packet['PASS']
        assert packet['root_protocol']['sha256'] == PROTOCOL_SHA
        scope.update({'canonical_features': complete, 'independent_feature_QA': fq,
                      'new_CONTROL_updates': 1000, 'new_TREAT_updates': 1000,
                      'same_initial_base_seed17_and_PCG17_draws': True,
                      'images_DINO_oldprefix_oldCONTROL_or_private_GT_payload_allowed': False,
                      'both_finals_and_global_sharing_QA_before_any_DEV_head_output': True,
                      'threshold_logit': 0})
        sb = write(ROOT / 'ACTUAL_CANONICAL_PAIR_TRAIN_SCOPE.json', scope)
        template = R / 'V239_CANONICAL_PAIRED_TRAINER_A/ENTRY_FALSE_TEMPLATE.json'
        e = json.loads(template.read_bytes())
        assert e['approved_canonical_source_pair_training'] is False
        e['approved_canonical_source_pair_training'] = True
        e['root_scope'] = sb
        e['feature_complete'] = complete
        for name in e['features']:
            e['features'][name] = packet['features'][name]
        for name in ('samples_npz', 'geometry_npz'):
            e[name] = packet[name]
        e['payload_whitelist'] = list(e['features'].values()) + [e['samples_npz'], e['geometry_npz']]
        for b in (sb, review, input_qa, fq, complete):
            append_binding(e['metadata_whitelist'], b)
        e['output_directory'] = str(ROOT / 'canonical_pair_source_fit_v1')
        out = ROOT / 'ACTUAL_CANONICAL_PAIR_TRAIN_ENTRY.json'
    for b in e['code_whitelist']:
        assert bind(b['path'])['sha256'] == b['sha256']
    assert not safe(e['output_directory']).exists()
    eb = write(out, e)
    print(json.dumps({'completed': True, 'actual_entry': eb, 'scope': sb, 'actual_features_or_fit_launched': False}))


if __name__ == '__main__':
    main()
