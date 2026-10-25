"""별도 Root 승인 뒤 SOURCE canonical packet만 독립 검산. encoder/head 실행 없음."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
from bound_io import need, identity, verify, safe, read_raw, load_json, binding, write_json, open_entry

LAYOUT = {'ORIGINAL_ALL_FEATURES.npy': ('float32', (7617, 1186)),
    'NODE_CLS.npy': ('float32', (1767, 8, 384)),
    'CONTEXT_CLS.npy': ('float32', (1767, 384)),
    'SAMPLE_GROUP_INDEX.npy': ('int64', (7617,)),
    'SAMPLE_NODE_INDEX.npy': ('int64', (7617,)),
    'RELATIVE_XY8.npy': ('float32', (7617, 8, 2)),
    'NODE_PIXEL_FEATURES.npy': ('float32', (1767, 8, 1154))}
COUNTS = dict(groups=1767, supported_nodes=13801, known=7617, FIT=5876, DEV=1741,
              all_original_rows=15040, masked_target_minus1=7423, context_supported_groups=1760)
SAMPLE_KEYS = {'full_view', 'central_view', 'fractions', 'central4', 'condition',
               'target', 'family', 'fit', 'available', 'row_id'}
GEO_KEYS = {'node_available', 'node_fraction', 'context_fraction', 'q8', 'L8',
            'original_available8', 'box_xyxy'}

def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def relative(g, group, slot):
    q = g['q8'][group]; active = q[np.arange(len(slot)), slot]; L = g['L8'][group, slot]
    avail = g['original_available8'][group]
    need(np.isfinite(active).all() and np.isfinite(L).all() and (L > 0).all()
         and np.isfinite(q[avail]).all(), '원 q/L finite')
    out = np.zeros(q.shape, np.float64)
    np.divide(q - active[:, None], L[:, None, None], out=out, where=avail[:, :, None])
    out = np.clip(out, -8, 8).astype(np.float32)
    out[~avail] = 0; out[np.arange(len(slot)), slot] = 0
    return out

def check_arrays(f, d, g, rows, expected=COUNTS):
    n = len(d['target']); groups = len(g['q8'])
    for k in ('fit', 'available', 'central4'):
        need(d[k].dtype == bool and d[k].shape == (n,), '원 sample bool')
    need(d['target'].dtype == np.int8 and np.isin(d['target'], [0, 1]).all()
         and d['family'].dtype == np.int8 and np.isin(d['family'], [0, 1]).all(), '분리된 D/E family')
    need(d['row_id'].dtype.kind == 'U' and len(set(d['row_id'])) == n, '원 row ID')
    need(same(d['full_view'], np.arange(n, dtype=np.int64)*2) and
         same(d['central_view'], np.arange(n, dtype=np.int64)*2+1), '원 view index')
    need(d['condition'].dtype == np.float32 and d['condition'].shape == (n,32)
         and np.isfinite(d['condition']).all() and d['fractions'].dtype == np.float32
         and d['fractions'].shape == (n,2), '원 condition/fraction schema')
    for k in ('original_available8', 'node_available'):
        need(g[k].dtype == bool and g[k].shape == (groups,8), '원 support bool8')
    for k, shape in [('q8',(groups,8,2)),('L8',(groups,8)),('box_xyxy',(groups,4))]:
        need(g[k].dtype == np.float64 and g[k].shape == shape, '원 F64 geometry')
    for k, shape in [('node_fraction',(groups,8)),('context_fraction',(groups,))]:
        need(g[k].dtype == np.float32 and g[k].shape == shape and np.isfinite(g[k]).all()
             and ((g[k] >= 0)&(g[k] <= 1)).all(), '원 valid fraction')
    need(not (g['node_available'] & ~g['original_available8']).any(), '원 U 승격 금지')
    group = f['SAMPLE_GROUP_INDEX.npy']; slot = f['SAMPLE_NODE_INDEX.npy']
    need(((group >= 0)&(group < groups)).all() and ((slot >= 0)&(slot < 8)).all(), 'group/slot 범위')
    for value in f.values():
        need(np.isfinite(value).all(), 'finite feature')
    pixel, node, x = f['NODE_PIXEL_FEATURES.npy'], f['NODE_CLS.npy'], f['ORIGINAL_ALL_FEATURES.npy']
    need(same(node, np.ascontiguousarray(pixel[:,:,:384])), '모든 NODE CLS 공통 pixel byte')
    need(same(x[:,:1154], pixel[group,slot]), '모든 known pixel1154 byte')
    need(same(x[:,:384], node[group,slot]), '모든7617 first384 byteexact')
    need(same(x[:,1154:], d['condition']) and same(x[:,1152:1154], d['fractions']), '원 condition/fractions byteexact')
    need(not np.count_nonzero(pixel[~g['node_available']]) and
         not np.count_nonzero(f['CONTEXT_CLS.npy'][g['context_fraction'] == 0]), 'unsupported zero')
    need(np.array_equal(g['node_available'][group,slot], d['available']) and d['available'].all()
         and d['central4'].all(), '원 known 지원 유지')
    need(same(f['RELATIVE_XY8.npy'], relative(g,group,slot)), 'F64 상대좌표 independent 계산')
    syn = d['family'] == 0
    need(not np.count_nonzero(d['condition'][~syn]) and
         np.array_equal(slot[syn],d['condition'][syn,:8].argmax(1)) and
         same(f['RELATIVE_XY8.npy'][syn],d['condition'][syn,16:].reshape(-1,8,2)), '원 SYN native / YCB nativeUNKNOWN')
    mapped, ids, roles, masked = set(), set(), {}, 0
    for r in rows:
        need(r['row_id'] not in ids, 'full map 원 ID 중복'); ids.add(r['row_id'])
        gi, si = int(r['group_index']), int(r['active_slot'])
        need(0 <= gi < groups and 0 <= si < 8, 'full map group/slot')
        fit = bool(r['fit']); need(gi not in roles or roles[gi] == fit, 'full map FIT/DEV group 중복')
        roles[gi] = fit
        q = np.asarray([np.nan,np.nan] if r['q'] is None else r['q'], np.float64)
        L = np.float64(np.nan if r['L'] is None else r['L'])
        need(np.array_equal(q,g['q8'][gi,si],equal_nan=True) and
             np.array_equal(L,g['L8'][gi,si],equal_nan=True), '원 q/L identity')
        need(bool(r['original_query_available']) == bool(g['original_available8'][gi,si])
             and bool(r['crop_available']) == bool(g['node_available'][gi,si]), 'full map input/support 별도 유지')
        i = r['sample_index']
        if i is None:
            need(int(r['target']) == -1, 'masked target-1 유지'); masked += 1; continue
        i = int(i); need(0 <= i < n and i not in mapped, '원 sample 단일 mapping'); mapped.add(i)
        need((gi,si) == (group[i],slot[i]) and r['row_id'] == str(d['row_id'][i])
             and int(r['target']) == int(d['target'][i]) and int(r['family_code']) == int(d['family'][i])
             and fit == bool(d['fit'][i]) and bool(r['crop_available']) == bool(d['available'][i]), '원 sample metadata identity')
    need(mapped == set(range(n)), '모든 known sample 유지')
    need(not (set(group[d['fit']]) & set(group[~d['fit']])), 'known FIT/DEV overlap')
    actual = dict(groups=groups,supported_nodes=int(g['node_available'].sum()),known=n,
        FIT=int(d['fit'].sum()),DEV=int((~d['fit']).sum()),all_original_rows=len(rows),
        masked_target_minus1=masked,context_supported_groups=int((g['context_fraction']>0).sum()))
    need(actual == expected, '고정 SOURCE counts')
    return dict(counts=actual,all_known_active_NODE_first384_byte_exact=True,
        all_NODE_CLS_pixel_byte_exact=True,all_known_pixel1154_byte_exact=True,
        original_conditions_fractions_relative_group_slot_q_L_exact=True,FIT_DEV_group_overlap=0,
        active_bytes_sha256=hashlib.sha256(x[:,:384].tobytes()).hexdigest(),tolerance_used=0,features_overwritten=0)

def npy(b, white):
    path = verify(b,white)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        return np.load(stream,allow_pickle=False)

def npz(b, white, keys):
    path = verify(b,white)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream, np.load(stream,allow_pickle=False) as z:
        need(set(z.files) == keys, '정확 nonpixel NPZ members')
        return {k:z[k] for k in keys}

def run(entry_path):
    e,_ = open_entry(entry_path,'approved_independent_SOURCE_feature_packet_QA',__file__)
    meta,payload = e['metadata_whitelist'],e['payload_whitelist']
    c = load_json(e['feature_complete'],meta)
    need(c['completed'] is True and c['PASS'] is True and c['counts'] == COUNTS, 'producer terminal COMPLETE')
    need(c['code']['sha256'] == '29a890a308f232743f65bc525e5aaee346b8702ae14c5db8a60b15d5bd367ff4'
         and identity(c['root_protocol']) == identity(e['root_protocol']), 'producer pin/protocol')
    need(set(c['features']) == set(LAYOUT) and c['features'] == c['outputs'], 'exact7 output closure')
    for k in ('samples_npz','geometry_npz','sample_map'):
        need(identity(e[k]) == identity(c[k]), '원 nonpixel binding closure')
    need(e['samples_npz']['sha256'] == '837bbe7ff406e27c6ed2981493a83c2adab830cd5b2de498e4a48de456456e99'
         and e['geometry_npz']['sha256'] == '994dd3a2a183a9ed5b4b53745daaca892516ec4507fca79cf280f0e78c514a83'
         and e['sample_map']['sha256'] == '82d4ca430e038e6567538e7dc373069f621913a480fc3a69c17d32634a1e12c6', '원 input pins')
    need(set(e['features']) == set(LAYOUT) and all(identity(e['features'][k]) ==
         identity(c['features'][k]) for k in LAYOUT), 'entry exact7 closure')
    for b in e['features'].values(): verify(b,payload)
    out = safe(e['output_directory']); need(not out.exists(), 'fresh QA output'); out.mkdir()
    try:
        f = {k:npy(b,payload) for k,b in e['features'].items()}
        headers = {}
        for k,(dtype,shape) in LAYOUT.items():
            need(str(f[k].dtype) == dtype and f[k].shape == shape, 'exact NPY header '+k)
            headers[k] = dict(dtype=dtype,shape=list(shape),binding=e['features'][k])
        d,g = npz(e['samples_npz'],payload,SAMPLE_KEYS),npz(e['geometry_npz'],payload,GEO_KEYS)
        rows = [json.loads(x) for x in read_raw(verify(e['sample_map'],meta)).decode().splitlines()]
        proof = check_arrays(f,d,g,rows)
        qa = load_json(c['canonical_sharing_QA_binding'],meta)
        need(qa == c['canonical_sharing_QA'] and qa['completed'] is True and qa['PASS'] is True
             and qa['active_first384_bytes_sha256'] == proof['active_bytes_sha256']
             and qa['node_selected_bytes_sha256'] == proof['active_bytes_sha256']
             and qa['fresh_construction_from_one_common_output'] is True
             and qa['mismatching_features_overwritten'] == 0 and qa['tolerance_relaxed'] is False, 'producer sharing receipt 비교')
        need(c['base_state_sha256_before'] == c['base_state_sha256_after'] == c['base_state_sha256_before_after']
             and c['classifier_optimizer_or_training'] == 0 and c['new_current_or_held_payload'] == 0
             and c['no_forward_GT_refs_pose_owner_target_identity'] is True, 'metadata frozen/no GT forward flags')
        result = dict(completed=True,PASS=True,feature_complete=e['feature_complete'],entry=binding(entry_path),
            code=binding(__file__),headers=headers,independent_numeric_proof=proof,current_admission=False,
            raw_RGB_DINO_model_head_or_protected_reads=0,
            limits='저장 특징/원 nonpixel metadata 독립 검산. DINO forward·RGB sampling·teacher gold 재인증 아님.')
        receipt = write_json(out/'REVIEW.json',result); print(json.dumps(receipt))
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',dict(completed=False,PASS=False,automatic_retry=False,traceback=traceback.format_exc())); raise

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
