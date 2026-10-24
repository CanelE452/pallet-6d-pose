"""PUBLIC fixed-composition query extension. No learned modules/current data."""
import ast
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

V = Path(__file__).resolve().parent
P = V.parent
OLD = P / 'query_visibility_flat_pallet_cue_source_20261023_v122/actual_corpus'
OUT = V / 'actual_corpus'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def bind(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}

def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s]

def write(p, d):
    with Path(p).open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def jsonl(p, rs):
    with Path(p).open('x') as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')

def pure(p, names):
    tree = ast.parse(Path(p).read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in funcs} == set(names)
    ns = {'np': np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(p)+'::pure_functions', 'exec'), ns)
    return ns

def main():
    policy_path = V/'ROOT_GEOMETRY_LOCK_BEFORE_CONSTRUCTION.json'
    policy = json.loads(policy_path.read_text())
    assert policy['normalized_grid_each_axis'] == [-0.25, 0, 0.25]
    assert policy['source_RGB_construction_authorized'] and not policy['current_allowed']
    assert not policy['fit_allowed_yet'] and not policy['V112_allowed']
    assert not OUT.exists(), 'Preserve any completed/partial construction'
    old_rows = rows(OLD/'OCCURRENCES4896_INPUT_ONLY.jsonl')
    old_gt = rows(OLD/'ORIGINAL_WEAK_AND_COMPOSITE_LABELS4896_GT_ONLY.jsonl')
    assert len(old_rows) == len(old_gt) == 4896
    for b in policy['inputs'].values():
        assert sha(b['path']) == b['sha256']
    kernel_path = P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
    recon_path = P/'query_visibility_flat_pallet_cue_source_20261023_v122/composite_preparation/build_flat_cue_corpus.py'
    target_path = P/'query_visibility_target_context_rgb_20261023_v107/target_context_math.py'
    crop = pure(kernel_path, ('crop',))['crop']
    recon = pure(recon_path, ('require', 'patch_slices', 'reconstruct_full_RGB'))['reconstruct_full_RGB']
    target_code = pure(target_path, ('valid_box', 'target_context_grid', 'sample_target_context'))
    sample_target = target_code['sample_target_context']
    OUT.mkdir()
    write(OUT/'STARTED.json', {'policy': bind(policy_path), 'code': bind(__file__), 'fit_model_current_V112_PnP': 0})
    with np.load(OLD/'INPUT_CROPS4896_INPUT_ONLY.npz', allow_pickle=False) as f:
        old_rgb = f['rgb96']
        old_valid = f['valid96']
    n = 32*9*17
    rgb96 = np.empty((n,3,96,96), np.uint8)
    valid96 = np.empty((n,1,96,96), bool)
    new_rows = [None]*n
    labels = [None]*n
    original_cache = {}
    patch_cache = {}
    context_valid_cache = {}
    alpha_counts = defaultdict(Counter)
    input_counts = Counter()
    centre_parity = 0
    changed_q = 0
    for cue in range(32):
        base = 153*cue + 17*4
        before = old_rows[base]
        r0 = np.array(before['query_xy'], np.float64)
        L0 = float(before['L0'])
        assert before['offset_xy'] == [0,0]
        assert before['cue_index'] == cue
        recipes = []
        native_images = []
        for role in range(17):
            parent = old_rows[base+role]
            z = parent['full_RGB_recipe']
            ob = z['original_full_RGB']
            if ob['path'] not in original_cache:
                assert sha(ob['path']) == ob['sha256']
                native = np.load(ob['path'], allow_pickle=False)
                assert ash(native) == parent['decoded_RGB_sha256']
                original_cache[ob['path']] = native
            original = original_cache[ob['path']]
            if z['kind'] == 'ORIGINAL':
                native = original
            else:
                pb = z['patch']
                if pb['path'] not in patch_cache:
                    assert sha(pb['path']) == pb['sha256']
                    with np.load(pb['path'], allow_pickle=False) as f:
                        patch_cache[pb['path']] = (f['rgb'].copy(), f['alpha'].copy())
                pr, pa = patch_cache[pb['path']]
                native = recon(original, pr, pa, z['placement_xy'])
            assert ash(native) == z['decoded_RGB_sha256']
            recipes.append(z)
            native_images.append(native)
        for j in range(9):
            ax = policy['normalized_grid_each_axis'][j%3]
            ay = policy['normalized_grid_each_axis'][j//3]
            offset = np.array([ax,ay], np.float64)*L0
            q = r0+offset
            points = np.array(before['initial_points'], np.float64)
            points[0] = q
            good = np.isfinite(points).all(1) & ~(points == [-1,-1]).all(1)
            bbox = np.r_[points[good].min(0),points[good].max(0)].tolist()
            L = float(np.clip(.25*np.max(np.ptp(points[good],axis=0)),16,max(before['raw_hw'])))
            hw = before['raw_hw']
            H,W = hw
            cont = bool(np.isfinite(q).all() and 0 <= q[0] < W and 0 <= q[1] < H)
            qx,qy = map(int,np.floor(q+.5))
            nearest = bool(0 <= qx < W and 0 <= qy < H)
            context_key = json.dumps([hw,bbox])
            if context_key not in context_valid_cache:
                _, vt = sample_target(native_images[0], bbox)
                context_valid_cache[context_key] = bool(vt.any())
            context = context_valid_cache[context_key]
            for role in range(17):
                idx = 153*cue+17*j+role
                parent_idx = base+role
                parent = old_rows[parent_idx]
                r = copy.deepcopy(parent)
                r.update(query_index=9*cue+j, query_id=f'V125_CUE_{cue:02d}::J{j}',
                         jitter_index=j, offset_xy=offset.tolist(),
                         query_grid_normalized_offset=[ax,ay], query_xy=q.tolist(),
                         L=L, initial_points=points.tolist(), context_box_xyxy=bbox,
                         q_native_continuous_valid=cont, q_nearest_valid=nearest,
                         q_native_valid=cont and nearest, emulation_grid='V125_NORMALIZED_QUARTER_L0',
                         context_Q_inside_geometry=True,
                         occurrence_index=idx, rgb96_row=idx,
                         occurrence_id=f'V125:{cue:02d}:{j}:{parent["variant"]}:{parent["donor_rank"]}',
                         parent_recipe_occurrence_index=parent_idx,
                         full_RGB_recipe=copy.deepcopy(recipes[role]))
                a,v = crop(native_images[role], q, L, 96)
                factual = bool(cont and nearest and v[:,47:49,47:49].all())
                available = factual and context
                r.update(rgb96_sha256=ash(a), valid96_sha256=ash(v),
                         factual_query_available=factual, context_available=context, query_available=available)
                rgb96[idx] = a
                valid96[idx] = v
                if j == 4:
                    r['parent_occurrence_index'] = parent_idx
                    assert r['query_xy'] == parent['query_xy'] and r['L'] == parent['L']
                    assert r['context_box_xyxy'] == parent['context_box_xyxy']
                    assert np.array_equal(a,old_rgb[parent_idx]) and np.array_equal(v,old_valid[parent_idx])
                    centre_parity += 1
                else:
                    changed_q += 1
                g = copy.deepcopy(old_gt[parent_idx])
                g.update(occurrence_index=idx,occurrence_id=r['occurrence_id'],jitter_index=j,
                         inherited_label_from_parent_occurrence=parent_idx,
                         uncertainty_support_unchanged=True,
                         same_expected_weak_cue_identity_under_query_shift=True,
                         public_query_emulation_not_real_initialization=True)
                z = recipes[role]
                inside = None
                if z['kind'] == 'OPAQUE_PATCH':
                    _, alpha = patch_cache[z['patch']['path']]
                    px,py = z['placement_xy']
                    lx,ly = qx-px,qy-py
                    inside = bool(nearest and 0 <= lx < alpha.shape[1] and 0 <= ly < alpha.shape[0] and alpha[ly,lx])
                g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'] = inside
                g['nearest_q_in_image_GT_diagnostic_only'] = nearest
                g['alpha_outside_does_not_certify_target_pixel'] = True
                key = r['source']+'::'+r['raw_split']
                if g['candidate_aux_hidden_label_GT_only'] in (0,1) and inside is not None:
                    alpha_counts[key][r['variant']+('::q_inside' if inside else '::q_outside')] += 1
                input_counts['available' if available else 'input_U'] += 1
                new_rows[idx] = r
                labels[idx] = g
        print(json.dumps({'cue_complete':cue+1,'total_cues':32}),flush=True)
    assert centre_parity == 544 and changed_q == 4352
    assert all(r is not None for r in new_rows)
    assert all(r['occurrence_index'] == i for i,r in enumerate(new_rows))
    assert all(g['occurrence_index'] == i for i,g in enumerate(labels))
    rf = OUT/'OCCURRENCES_INPUT_ONLY.jsonl'
    gf = OUT/'LABELS_GT_ONLY.jsonl'
    cf = OUT/'STORED_RGB96_INPUT_ONLY.npz'
    jsonl(rf,new_rows)
    jsonl(gf,labels)
    np.savez_compressed(cf,rgb96=rgb96,valid96=valid96,occurrence_index=np.arange(n,dtype=np.int32))
    summary = {'completed':True,'PASS':True,'policy':bind(policy_path),'code':bind(__file__),
               'input_manifest':bind(rf),'stored_crops':bind(cf),'weak_evidence':bind(gf),
               'outputs':{'input_manifest':bind(rf),'stored_crops':bind(cf),'labels':bind(gf)},
               'parent_corpus_seal':bind(OLD/'CORPUS_SEAL.json'),
               'occurrences':n,'query_conditions':288,'original_cues':32,'original_RGBs':len(original_cache),
               'changed_q_occurrences':changed_q,'centre544_exact_old_crop_valid_q_L_context':centre_parity,
               'availability':dict(input_counts),'states':dict(Counter(g['candidate_aux_hidden_label_GT_only'] for g in labels)),
               'alpha_by_source_rawsplit':{k:dict(v) for k,v in alpha_counts.items()},
               'original_physical_compositions_weak_uncertainty_and_fold_unchanged':True,
               'model_encoder_fit_current_V112_PnP':0,
               'limits':['Normalized q offsets are public emulations; not actual initial predicted coordinates.',
                         'Existing recorded-cue weak visibility labels are preserved; no exact native or per-pixel target gold.',
                         'Outside pasted alpha does not certify target pixels.',
                         'Same scenes, donors and repeated offsets are correlated; this branch does not increase original diversity.']}
    write(OUT/'CORPUS_COMPLETE.json',summary)
    print(json.dumps({'completed':True,'report':bind(OUT/'CORPUS_COMPLETE.json'),'alpha':summary['alpha_by_source_rawsplit']},ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
