"""V219 independent CODE-only review. Reads exact text, exercises invented NumPy only."""
import ast
import builtins
import hashlib
import json
import os
import stat
import symtable
from pathlib import Path

import numpy as np

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
H = P / 'publication_root_preparation/V219_TRAINING_PREPARATION_C'
OUT = Path(__file__).parent
EXPECTED = {
    str(H/'FINAL_READY.json'): '832e04ebc781ada27dc0d01cb9a4e60cec05bd3e9f3cdff1945fd199555c707a',
    str(H/'train_strong_plus_expected_part_weak.py'): '11c40cb6edfb5260310e2370ff06f4ac04a30d3d69e3f892f3bcd9709696a06d',
    str(H/'source_training_contract.py'): 'fc0a161e8a4c1a413d528ce1e66eed1b2addf2d4593c632506280f10732451d7',
    str(H/'PROTOCOL_BEFORE_ANY_V219_SOURCE_PAYLOAD_OR_FIT.json'): 'f3b769daf7f5419db9ff77a28718db49a804fecc152a7691bbe3f8eb18a39540',
    str(H/'ROOT_ACTUAL_V219_SOURCE_PAIR_ENTRY_FALSE_TEMPLATE.json'): '6fd89e1e9bc2bde06e8051a1c7d9a6d38fd9e769752cb20d098c6fee6e835cab',
    str(P/'query_visibility_expected_part_weak_paired_source_training_20261025_v219/ROOT_PROSPECTIVE_SOURCE_ONLY_WEAK_PAIRED_FIT_SCOPE.json'): '1c7e15525713f981410828c604dea47ce9c25226a482bd62babdb3b21bd1f3f7',
    str(P/'query_visibility_query_jitter_paired_source_training_20261024_v183/source_fit_preparation/train_jitter_paired_source.py'): '80004a31dce09a9b8fc00e78b93defa1b60d3b8cc35aa8d910c2a776f0ed70b6',
    str(P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/source_target_contract.py'): 'd8925908d00f4f120bd80bae7a691a1a3846c4e9096e2e111d71984843e0b738',
}

def read(path):
    s = str(path)
    assert s in EXPECTED and 'v112' not in s.lower() and '..' not in Path(s).parts
    p = Path(s)
    fd = os.open(p.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        leaf = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(leaf, 'rb') as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
        data = f.read()
    assert hashlib.sha256(data).hexdigest() == EXPECTED[s]
    return data.decode()

def require(ok, message):
    if not bool(ok):
        raise AssertionError(message)

def rejected(call):
    try:
        call()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Expected rejection')

def main():
    text = {p: read(p) for p in EXPECTED}
    csrc = text[str(H/'source_training_contract.py')]
    c = {'__name__': 'invented_v219_contract'}
    exec(compile(csrc, 'BOUND_PURE_CONTRACT', 'exec'), c)
    oldpath = str(P/'query_visibility_query_jitter_paired_source_training_20261024_v183/source_fit_preparation/train_jitter_paired_source.py')
    oldsrc = text[oldpath]
    oldtree = ast.parse(oldsrc)
    names = {'validate_targets', 'draw_step', 'five_losses'}
    nodes = [n for n in oldtree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    symbols = symtable.symtable(oldsrc, oldpath, 'exec')
    globals_by_function = {}
    for child in symbols.get_children():
        if child.get_name() in names:
            globals_by_function[child.get_name()] = sorted(s.get_name() for s in child.get_symbols() if s.is_referenced() and s.is_global() and not hasattr(builtins, s.get_name()))
    assert globals_by_function == {'validate_targets': ['np', 'require'], 'draw_step': ['np', 'require'], 'five_losses': ['F', 'require', 'torch']}
    old = dict(np=np, require=require)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'BOUND_OLD_FUNCTIONS', 'exec'), old)

    # Actual old validator and new pool helper; all values below are invented.
    n = 12
    available = np.ones(n, bool)
    available[-2:] = False
    support = np.column_stack(((np.arange(n) % 2 == 0), (np.arange(n)//2 % 2 == 0))).astype(np.int8)
    support[~available] = -1
    mass = np.tile(np.asarray([.1, .2, .3], np.float64), (n, 1))
    mass[1] = [0., 1., 0.]
    ambiguous = np.full(n, .2, np.float64)
    invalid = np.full(n, .2, np.float64)
    ambiguous[1] = invalid[1] = 0.
    mass[~available] = np.nan
    ambiguous[~available] = invalid[~available] = np.nan
    y = dict(support_target=support, owner_soft=mass, owner_target=np.full(n, -1, np.int8), owner_ambiguous_mass=ambiguous, owner_invalid_mass=invalid)
    assert old['validate_targets'](y, available) is y
    dev = 'd'*64
    metadata = [dict(component_role='TRAIN', nominal_fit_eligible=True, component_id='a'*64, lineage_bound=True) for _ in range(n)]
    metadata[1]['component_id'] = dev
    metadata[3]['nominal_fit_eligible'] = False
    metadata[5]['component_role'] = 'HOLDOUT'
    metadata[7]['lineage_bound'] = False
    mask = c['fit_mask'](metadata, {dev})
    assert not mask[[1, 3, 5, 7]].any()

    def hard_cache(classes):
        a = np.ones(n, bool)
        target = (np.arange(n) % classes).astype(np.int8)
        return dict(x=dict(original_query_available=a), y=dict(support_target=np.full((n, 2), -1, np.int8), owner_target=target))
    domains = c['DOMAINS']
    caches = {domains[0]: dict(x=dict(original_query_available=available), y=y), domains[1]: hard_cache(3), domains[2]: hard_cache(2)}
    masks = {d: mask.copy() for d in domains}
    pools = c['old_pools'](caches, masks)
    assert set(pools) == {'support', 'conditional', 'public', 'CF'}
    for j, probability in enumerate(pools['conditional']):
        expected = np.where(mask & available, np.nan_to_num(mass[:, j]), 0.)
        expected /= expected.sum(dtype=np.float64)
        assert np.array_equal(probability, expected)
        assert not probability[~(mask & available)].any()
    for j in range(2):
        for label, ids in zip((1, 0), pools['support'][j]):
            assert (mask[ids] & available[ids]).all() and (support[ids, j] == label).all()

    # Execute original draw kernel with newly masked pools, paired old/strong RNGs.
    pt = (np.arange(n) % 2).astype(np.int8)
    pp = c['balanced_pools'](pt, mask & available)
    draws = []
    for with_weak in (False, True):
        rng, point_rng, weak_rng = (np.random.default_rng(i) for i in (1810, 1811, 1812))
        arm = []
        for _ in range(16):
            d = old['draw_step'](rng, pools)
            pi = c['draw_balanced'](point_rng, pp, 16)
            if with_weak:
                c['draw_balanced'](weak_rng, [np.array([0, 2]), np.array([1, 3])], 8)
            assert list(d) == ['visible', 'foreign', 'synthetic_owner', 'public_owner', 'CF_owner']
            assert [len(v) for v in d.values()] == [32, 32, 48, 48, 32]
            for k, ids in d.items():
                dom = domains[1] if k == 'public_owner' else domains[2] if k == 'CF_owner' else domains[0]
                assert masks[dom][ids].all()
                if dom == domains[0]:
                    assert available[ids].all()
            assert mask[pi].all() and available[pi].all()
            assert np.array_equal(pt[pi], np.asarray([0]*16+[1]*16, np.int8))
            arm.append({k: v.tolist() for k, v in d.items()} | {'point32': pi.tolist()})
        draws.append(arm)
    assert draws[0] == draws[1]

    # Real source-contract pure context and point authority, invented values only.
    q8 = [[float(j)*10, float(j)*-5] for j in range(8)]
    cc = c['context32'](q8[2], 20., 2, True, q8, [20.]*8, [True]*8)
    assert cc.dtype == np.float32 and cc.shape == (32,) and cc[2] == 1 and cc[:8].sum() == 1
    assert cc[8+2] == 0 and (cc[16+4:16+6] == 0).all() and cc[8:16].sum() == 7
    av8 = [True]*8; av8[2] = False
    cu = c['context32'](None, None, 2, False, q8, [20.]*8, av8)
    assert cu[2] == 1 and not cu[8:].any()
    pcpath = str(P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/source_target_contract.py')
    pc = {'__name__': 'invented_point_contract'}
    exec(compile(text[pcpath], 'BOUND_POINT_CONTRACT', 'exec'), pc)
    r = dict(component_role='TRAIN', mapping_status='EXACT_ORIGINAL_IMAGE_OWN_SCENE', family='SELF_SCENE15', state_name='NOMINAL', state_index=0, original_query_available=True, query_xy=[10.,20.], L=96., saved_projected_reference_xy=[10.,20.], point0_target_alone_state='D', point0_combined_state='D', reference_locality_for_actual_input=dict(inside_original_frame_canvas_extent=True,inside_crop_canvas_extent=True,raw96_reference_index_xy=[47.5,47.5]))
    assert pc['point_loss_target'](r)[0] == 0
    assert pc['point_loss_target'](r | {'point0_combined_state':'E'})[0] == 1
    assert pc['point_loss_target'](r | {'point0_combined_state':'S'})[0] == -1
    assert pc['point_loss_target'](r | {'original_query_available':False})[0] == -1

    # AST checks preserve differentiable batch boundaries; no torch/model/forward executed.
    objective = next(n for n in ast.parse(csrc).body if isinstance(n, ast.FunctionDef) and n.name == 'objective')
    slices = [ast.unparse(n.slice) for n in ast.walk(objective) if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice)]
    assert ':192' in slices and '192:224' in slices and '224:' in slices
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('detach', 'numpy', 'item') for n in ast.walk(objective))
    five = next(n for n in nodes if n.name == 'five_losses')
    fs = ast.unparse(five)
    for snippet in ['support[:32, 0]', 'support[32:64, 1]', 'owner[64:112]', 'owner[112:160]', 'owner[160:192]', 'torch.stack(losses).mean()']:
        assert snippet in fs, snippet
    runner = text[str(H/'train_strong_plus_expected_part_weak.py')]
    assert runner.index("if e['mode']=='THROUGHPUT_PROBE':") < runner.index("freeze=write(out/'ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json'")
    assert runner.index("freeze=write(out/'ALL2_FINAL_FROZEN_BEFORE_ANY_DEV_OR_REAL_OUTPUT.json'") < runner.index('if i not in weakpixels:weakpixels[i]=weak_RGB')
    assert 'steps<=32' in runner and 'old_checkpoint_loaded=False' in runner and 'checkpoint_saved=False' in runner
    entry = json.loads(text[str(H/'ROOT_ACTUAL_V219_SOURCE_PAIR_ENTRY_FALSE_TEMPLATE.json')])
    assert entry['approved_source_pair_fit_or_probe'] is False and entry['warmstart_checkpoint'] is None and not entry['binary_whitelist']
    evidence = dict(completed=True, PASS=True, groups4=[
        'exact old function free globals and actual validator/new masked pool schema',
        'masked owner renormalization and actual paired old192/point32 draws with separate weak RNG',
        'pure native/self/U condition32 and strong own-image D/E vs S/U authority',
        'differentiable old192/point32/weak16 AST slices, probe and both-final barrier'],
        old_function_nonbuiltin_globals=globals_by_function,
        old_draw_sizes=[32,32,48,48,32], paired_invented_steps_each=16,
        source_payload_or_model_or_torch_forward_calls=0,
        note='Gradient routing is a static differentiability/slice review, not neural loss/gradient replay.',
        input_code_bindings=[dict(path=p,sha256=h) for p,h in EXPECTED.items()])
    target = OUT/'INVENTED_NUMPY_AND_STATIC_CHECKS.json'
    with target.open('x') as f:
        json.dump(evidence,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,groups=4,receipt=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())))

if __name__ == '__main__':
    main()
