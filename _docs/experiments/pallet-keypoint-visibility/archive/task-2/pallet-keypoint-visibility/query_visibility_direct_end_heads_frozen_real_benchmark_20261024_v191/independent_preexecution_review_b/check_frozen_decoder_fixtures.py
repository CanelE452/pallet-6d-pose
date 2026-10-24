"""Independent invented-only static review fixtures; never a production runner."""
from pathlib import Path
import ast
import hashlib
import json
import os
import stat
import numpy as np

V = Path(__file__).absolute().parent.parent
PRED = V / 'prediction_preparation_a/infer_both_frozen_direct_end_heads.py'
EVAL = V / 'evaluation_preparation_a/evaluate_saved_direct_end_real.py'
META = ('row_id', 'original_global_index', 'native_slot', 'query_xy', 'L', 'original_query_available')


def pure(path, names, namespace):
    assert all('v112' not in part.casefold() for part in path.parts)
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        last = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    assert stat.S_ISREG(os.fstat(last).st_mode)
    with os.fdopen(last, 'r') as stream:
        text = stream.read()
    nodes = [node for node in ast.parse(text).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<review-pure>', 'exec'), namespace)
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = dict(np=np)
    ps = pure(PRED, {'require', 'decode', 'available_batches'}, p)
    e = dict(np=np, META=META, SUPPORT_ORDER=['visible_end_support', 'foreign_front_support'],
             OWNER_ORDER=['TARGET', 'FOREIGN', 'BACKGROUND'])
    es = pure(EVAL, {'require', 'decode', 'direct_decode', 'joint', 'equal', 'packet_schema'}, e)
    checks = []

    def check(name, ok):
        assert bool(ok), name
        checks.append(dict(name=name, PASS=True))

    # The scalar reference uses Python comparisons, not either production decoder.
    tiny = np.nextafter(np.float32(0), np.float32(1))
    raw = np.array([-tiny, -0., 0., tiny, -np.inf, np.inf, np.nan, np.nan], np.float32)
    available = np.array([True] * 7 + [False])
    expected = np.array([(-1 if not a or not np.isfinite(v) else 1 if float(v) >= 0. else 0)
                         for v, a in zip(raw, available)], np.int8)
    support = np.array([[0, 0], [1, -1], [-1, 1], [1, 1], [-1, -1], [0, 1], [1, 0], [np.nan, np.nan]], np.float32)
    owner = np.array([[0, 0, 0], [1, 2, 2], [3, 2, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [np.nan] * 3], np.float32)
    sp, op, dp, proxy = p['decode'](support, owner, raw, available)
    esp, eop, ep = e['decode'](support, owner, available)
    check('new_direct_decoders_match_independent_scalar_at_signed_zero_subnormal_nonfinite_U',
          np.array_equal(dp, expected) and np.array_equal(e['direct_decode'](raw, available), expected))
    check('old_support_owner_proxy_identical_between_predictor_evaluator',
          np.array_equal(sp, esp) and np.array_equal(op, eop) and np.array_equal(proxy, ep))
    check('direct_not_gated_by_support_NONE_BOTH_or_owner_class',
          dp[:4].tolist() == [0, 1, 1, 1] and proxy[:4].tolist() == [-1, 0, 1, 1] and op[1] == 1)
    check('joint_support_states_separate_from_direct_native', e['joint'](sp).tolist() == [0, 1, 2, 3, 0, 2, 1, 4])
    meta = dict(row_id=np.array(['invented:' + str(i) for i in range(8)]),
                original_global_index=np.arange(8, dtype=np.int64),
                native_slot=np.array([7, 2, 0, 5, 1, 6, 3, 4], np.int8),
                query_xy=np.ones((8, 2), np.float64), L=np.ones(8, np.float64),
                original_query_available=available)
    old = dict(meta, raw_support_logits=support, raw_owner_logits=owner,
               support_prediction=sp, owner_prediction=op, regional_support_proxy_D0_E1_Uminus1=proxy,
               support_order=np.array(e['SUPPORT_ORDER']), owner_order=np.array(e['OWNER_ORDER']))
    new = dict(old, raw_direct_end_external_logit=raw, direct_end_prediction_D0_E1_Uminus1=dp,
               direct_end_order=np.array(['VISIBLE_END0', 'EXTERNAL_BLOCKED_END1', 'UNKNOWNminus1']))
    e['packet_schema'](old, 8, False)
    e['packet_schema'](new, 8, True)
    check('same_actual_schema_helper_accepts_distinct_old13_and_new16_with_shuffled_native_metadata', True)
    rejected = 0
    bads = [(old, True), (new, False), (dict(new, direct_end_prediction_D0_E1_Uminus1=np.zeros(8, np.int8)), True),
            (dict(new, query_xy=meta['query_xy'].astype(np.float32)), True),
            (dict(new, row_id=np.array(['duplicate'] * 8)), True)]
    for packet, direct in bads:
        try:
            e['packet_schema'](packet, 8, direct)
        except (ValueError, AssertionError):
            rejected += 1
    check('wrong_old_new_schema_dtype_row_identity_or_direct_decoder_rejected', rejected == len(bads))
    mask = np.array([i % 3 != 0 for i in range(1000)], bool)
    batches = p['available_batches'](mask)
    check('fixed192_batches_cover_each_available_exactly_once_without_U',
          np.array_equal(np.concatenate(batches), np.flatnonzero(mask)) and max(map(len, batches)) == 192)
    print(json.dumps(dict(completed=True, PASS=True, invented_only=True,
                         code_sha256=dict(predictor=ps, evaluator=es), checks=checks, count=len(checks),
                         actual_model_array_weight_annotation_or_score_reads=0)))


if __name__ == '__main__':
    main()
