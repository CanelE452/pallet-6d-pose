"""Independent saved V255 feature/target algebra and fixed crop replay, no model."""
import argparse
import ast
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import traceback
import numpy as np

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R/'V255_FEATURE_TARGET_QA_C'
FEATURE_KEYS = ('ORIGINAL_ALL_FEATURES.npy', 'NODE_CLS.npy', 'CONTEXT_CLS.npy', 'SAMPLE_GROUP_INDEX.npy', 'SAMPLE_NODE_INDEX.npy', 'RELATIVE_XY8.npy')
GEOM_KEYS = {'q8', 'L8', 'original_available8', 'node_available', 'node_fraction', 'context_fraction', 'box_xyxy'}
SAMPLE_KEYS = {'full_view', 'central_view', 'fractions', 'central4', 'condition', 'target', 'family', 'fit', 'available', 'row_id'}
STATE = 'bb4c7a80aa873bf3d34c3c1d44a28cee331503d7ccf52aba85f2d23edb65c3c3'


def safe(path, missing=False):
    p = Path(path)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'oldval', 'holdout'))
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            assert missing
            continue
        assert not stat.S_ISLNK(mode), str(a)
    return p


def ident(b):
    return {k: b[k] for k in ('path', 'sha256', 'size_bytes')}


def hash_file(path):
    h, n = hashlib.sha256(), 0
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part); n += len(part)
    return {'path': str(Path(path)), 'sha256': h.hexdigest(), 'size_bytes': n}


def read(b):
    with os.fdopen(os.open(safe(b['path']), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        raw = f.read()
    assert len(raw) == b['size_bytes'] and hashlib.sha256(raw).hexdigest() == b['sha256']
    return raw


def js(b):
    return json.loads(read(b))


def fp(x):
    if x is None: return b'N'
    if isinstance(x, bool): return b'B'+bytes([x])
    if isinstance(x, list): return b'L'+struct.pack('<Q', len(x))+b''.join(map(fp, x))
    assert type(x) in (float, int)
    return b'F'+struct.pack('<d', x)


def put(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()
    with os.fdopen(os.open(safe(path, True), os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o644), 'wb') as f:
        f.write(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def finite_pair(q):
    return isinstance(q, list) and len(q) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in q)


def independent_target(q, L, av, record, hw):
    def state(key):
        v = record.get(key)
        return v[0] if isinstance(v, list) and v and v[0] in ('D', 'E', 'S', 'U') else 'UNKNOWN'
    a, c = state('target_witness_states'), state('combined_witness_states')
    ref = record.get('projected_reference_xy')
    if not finite_pair(ref): ref = None
    loc = {'original_input_available': av, 'original_input_U_preserved': not av,
        'raw96_reference_index_xy': None, 'inside_sample_center_extent': None,
        'inside_crop_canvas_extent': None, 'inside_original_frame_canvas_extent': None,
        'not_coordinate_error_correction_or_pixel_visibility_proof': True}
    valid_q = finite_pair(q) and type(L) in (int, float) and math.isfinite(L) and L > 0
    if ref is not None:
        loc['inside_original_frame_canvas_extent'] = -.5 <= ref[0] < hw[1]-.5 and -.5 <= ref[1] < hw[0]-.5
        if valid_q:
            uv = [47.5+96*(ref[j]-q[j])/L for j in range(2)]
            loc.update(raw96_reference_index_xy=uv, inside_sample_center_extent=all(0 <= x <= 95 for x in uv), inside_crop_canvas_extent=all(-.5 <= x < 95.5 for x in uv))
    reasons = []
    if av is not True: reasons.append('ORIGINAL_INPUT_U')
    if not valid_q: reasons.append('MISSING_Q_OR_L')
    if a != 'D': reasons.append('TARGET_ALONE_NOT_D')
    if c not in ('D', 'E'): reasons.append('COMBINED_NOT_DE')
    if ref is None: reasons.append('MISSING_REFERENCE')
    if loc['inside_original_frame_canvas_extent'] is not True: reasons.append('REFERENCE_OUTSIDE_FRAME')
    if loc['inside_crop_canvas_extent'] is not True: reasons.append('REFERENCE_OUTSIDE_ACTUAL_INPUT_CROP')
    return (-1 if reasons else int(c == 'E')), sorted(set(reasons)), a, c, ref, loc


def load_numpy_kernels(binding, expected_sha, names):
    assert binding['sha256'] == expected_sha
    tree = ast.parse(read(binding))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    ns = {'np': np, 'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), binding['path'], 'exec'), ns)
    return ns


def unit_or_zero(x, supported, allow_supported_zero=False):
    assert np.count_nonzero(x[~supported]) == 0
    norms = np.linalg.norm(np.asarray(x[supported], np.float64), axis=-1)
    good = np.abs(norms-1) <= 2e-5
    if allow_supported_zero: good |= norms == 0
    assert good.all()
    return int((norms == 0).sum())


def run(entry_path):
    e = json.loads(safe(entry_path).read_bytes())
    assert e.get('approved_saved_feature_target_QA') is True
    assert e['code'] == hash_file(__file__)
    scope = js(e['read_scope'])
    assert scope['approved'] is True and scope['entry_contract'] == {k:v for k,v in e.items() if k != 'read_scope'}
    out = safe(e['output_dir'], True)
    assert out == H/'actual_review_v1' and not out.exists()
    out.mkdir()
    try:
        c = js(e['feature_complete'])
        source_entry = js(e['producer_entry']); source_scope = js(source_entry['root_scope'])
        assert source_scope['approved'] is True and source_scope['entry_contract'] == {k:v for k,v in source_entry.items() if k != 'root_scope'}
        assert c['completed'] is True and c['PASS'] is True
        assert c['entry'] == e['producer_entry'] and c['root_scope'] == source_entry['root_scope']
        assert c['root_policy'] == e['root_policy'] == source_entry['root_policy']
        assert c['root_policy']['sha256'] == '0c0bc375c5e07cf7640bcb76bfd8efeda09e5a3d58f0c40c78094ba6dbc373a4'
        assert c['stage2_complete'] == e['stage2_complete'] and c['stage2_QA'] == e['stage2_QA']
        assert c['stage1_parent_manifest'] == source_entry['stage1_parent_manifest']
        assert c['code'] == source_entry['code'] and c['code']['sha256'] == '17b4db314eaba22c6a0045e55c7ce73d18e2a5a68ccfd4461dc8a8e8f42bbaad'
        q2, d2 = js(e['stage2_QA']), js(e['stage2_complete'])
        assert q2['completed'] is True and q2['PASS'] is True and q2['dataset_complete'] == e['stage2_complete']
        assert q2['ray_subset'] == e['ray_subset']
        ray = js(e['ray_subset'])['rows']
        root = Path(e['feature_complete']['path']).parent
        assert root == Path(source_entry['output_dir']) and Path(e['feature_complete']['path']) == root/'COMPLETE.json'
        expected_outputs = {'NODE_RGB96.npy','NODE_VALID96.npy','CONTEXT_RGB224_F32.npy','CONTEXT_VALID224.npy',
            'INPUT_GEOMETRY.npz','samples.npz','ALL_NEW_SOURCE_ROWS.jsonl','INPUT_SCENE_BINDINGS.json',
            'TARGET_COMPILER_QA.json','CANONICAL_BYTE_SHARING_AND_FROZEN_ENCODER_QA.json','CENSUS.json',*FEATURE_KEYS}
        assert set(c['outputs']) == expected_outputs and set(c['features']) == set(FEATURE_KEYS)
        assert c['samples_npz'] == c['outputs']['samples.npz'] and c['geometry_npz'] == c['outputs']['INPUT_GEOMETRY.npz']
        assert c['sample_map'] == c['outputs']['ALL_NEW_SOURCE_ROWS.jsonl']
        assert c['target_compiler_QA'] == c['outputs']['TARGET_COMPILER_QA.json']
        assert c['canonical_sharing_QA_binding'] == c['outputs']['CANONICAL_BYTE_SHARING_AND_FROZEN_ENCODER_QA.json']
        manifest = js(c['manifest'])
        assert len(manifest['files']) == len(c['outputs'])
        assert {b['path']:ident(b) for b in manifest['files']} == {b['path']:ident(b) for b in c['outputs'].values()}
        for name, b in c['outputs'].items():
            assert Path(b['path']) == root/name and hash_file(b['path']) == ident(b)
        def output(name): return c['outputs'][name]
        with np.load(io.BytesIO(read(output('samples.npz'))), allow_pickle=False) as z:
            assert set(z.files) == SAMPLE_KEYS
            samples = {k:z[k].copy() for k in z.files}
        with np.load(io.BytesIO(read(output('INPUT_GEOMETRY.npz'))), allow_pickle=False) as z:
            assert set(z.files) == GEOM_KEYS
            g = {k:z[k].copy() for k in z.files}
        rows = [json.loads(x) for x in read(output('ALL_NEW_SOURCE_ROWS.jsonl')).splitlines() if x]
        scenes = [json.loads(x) for x in read(d2['outputs']['SCENES.jsonl']).splitlines() if x]
        bindings = js(output('INPUT_SCENE_BINDINGS.json'))['rows']
        parents = {p['base_index']:p for p in js(c['stage1_parent_manifest'])['parents']}
        G, N = len(scenes), len(samples['target'])
        assert len(rows) == 8*G and len(bindings) == G and len({r['row_id'] for r in rows}) == 8*G
        assert g['q8'].shape == (G,8,2) and g['q8'].dtype == np.float64
        assert g['L8'].shape == (G,8) and g['L8'].dtype == np.float64
        for key in ('original_available8','node_available'): assert g[key].shape == (G,8) and g[key].dtype == bool
        assert g['node_fraction'].shape == (G,8) and g['node_fraction'].dtype == np.float32
        assert g['context_fraction'].shape == (G,) and g['context_fraction'].dtype == np.float32
        assert g['box_xyxy'].shape == (G,4) and g['box_xyxy'].dtype == np.float64
        raw = {k:np.load(safe(output(name)['path']), allow_pickle=False, mmap_mode='r') for k,name in (
            ('rgb','NODE_RGB96.npy'),('valid','NODE_VALID96.npy'),('context','CONTEXT_RGB224_F32.npy'),('cv','CONTEXT_VALID224.npy'))}
        assert raw['rgb'].shape == (G,8,3,96,96) and raw['rgb'].dtype == np.uint8
        assert raw['valid'].shape == (G,8,96,96) and raw['valid'].dtype == bool
        assert raw['context'].shape == (G,3,224,224) and raw['context'].dtype == np.float32
        assert raw['cv'].shape == (G,224,224) and raw['cv'].dtype == bool
        central = np.zeros((G,8),bool); center_fraction = np.zeros((G,8),np.float32)
        for start in range(0,G,32):
            v,cv = np.asarray(raw['valid'][start:start+32]),np.asarray(raw['cv'][start:start+32])
            central[start:start+len(v)] = v[:,:,47:49,47:49].all((2,3))
            center_fraction[start:start+len(v)] = v[:,:,32:64,32:64].mean((2,3)).astype(np.float32)
            assert np.array_equal(v.mean((2,3)).astype(np.float32),g['node_fraction'][start:start+len(v)])
            assert np.array_equal(cv.mean((1,2)).astype(np.float32),g['context_fraction'][start:start+len(v)])
            assert np.array_equal(g['original_available8'][start:start+len(v)] & central[start:start+len(v)] & v.any((2,3)),g['node_available'][start:start+len(v)])
        group,slot,labels,ids,conditions = [],[],[],[],[]
        reason_counts,point_counts,loss_counts = collections.Counter(),collections.Counter(),collections.Counter()
        original_u = 0
        scene_map = {}
        for gi,s in enumerate(scenes):
            assert s['scene_index'] == gi and s['source_role'] == 'FIT'
            meta = js(s['metadata']); p = parents[s['base_index']]; original = p['before_input']; before = meta['original_before_input']
            assert p['identity_PASS'] is True and p['source_base_role'] == 'FIT'
            for key in ('q8','L8','original_available8','box_xyxy','raw_hw'):
                assert fp(before[key]) == fp(original[key])
            assert np.array_equal(g['q8'][gi],np.asarray(before['q8'],np.float64),equal_nan=True)
            assert np.array_equal(g['L8'][gi],np.asarray(before['L8'],np.float64),equal_nan=True)
            assert np.array_equal(g['original_available8'][gi],before['original_available8'])
            box = np.full(4,np.nan) if before['box_xyxy'] is None else np.asarray(before['box_xyxy'],np.float64)
            assert np.array_equal(g['box_xyxy'][gi],box,equal_nan=True)
            expected_binding = {'group_index':gi,'scene_id':s['scene_id'],'image':s['image'],'metadata':s['metadata'],'RGB_sha256_HWC':s['RGB_sha256_HWC'],
                'original_parent_base_index':s['base_index'],'target_native_slot_teacher_only':s['target_native_slot']}
            assert bindings[gi] == expected_binding
            image_input = {k:s['image'][k] for k in ('path','sha256','format','member')}
            gid = hashlib.sha256(json.dumps([image_input,before['q8'],before['L8'],before['original_available8']],ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            for j in range(8):
                rec,r = meta['fresh_records_CF'][j],rows[8*gi+j]
                assert rec['CF_index'] == j and rec['canonical_index'] == rec['native_canonical_index'] == meta['permCF_to_canonical'][j]
                q,L,av = before['q8'][j],before['L8'][j],before['original_available8'][j]
                y,reasons,a,state,ref,loc = independent_target(q,L,av,rec,meta['RGB_shape'][:2])
                before_support = y; available = bool(g['node_available'][gi,j] and av)
                if not available: y=-1;reasons=sorted(set(reasons+['RAW96_INPUT_SUPPORT_UNAVAILABLE']))
                rid = 'V255:%s:NATIVE%d' % (s['scene_id'],j)
                assert r['row_id'] == rid and r['scene_id'] == s['scene_id'] and r['new_scene_index'] == r['group_index'] == gi
                assert r['group_id'] == gid
                assert r['native_slot'] == r['active_slot'] == r['active_native_slot'] == j and r['logical_aux_slot'] is None
                assert r['intervention_target_native_slot'] == s['target_native_slot'] and r['variant'] == s['variant']
                assert r['base_index'] == s['base_index'] and r['source_role'] == 'FIT' and r['fit'] is True and r['family_code'] == 0
                assert r['original_component_role'] == 'TRAIN' and r['internal_source_role'] == 'INTERNAL_SOURCE_FIT'
                assert r['generation_family'] == 'V255_NEW_OWN_SCENE_ORIGINAL_BEFORE_INPUT'
                assert r['own_image_record_identity_PASS'] is True and r['own_image_mapping_reasons'] == []
                assert r['new_teacher_key'] == {'scene_id':s['scene_id'],'RGB_sha256_HWC':s['RGB_sha256_HWC'],'active_CF_index':j}
                assert r['original_query_xy'] == r['q'] == q and r['original_L'] == r['L'] == L and r['original_query_available'] == av
                assert r['point0_target_alone_state'] == a and r['point0_combined_state'] == r['original_state'] == state
                assert r['projected_reference_xy_teacher_only'] == ref and r['reference_locality_teacher_only'] == loc
                assert r['declared_canonical_index_teacher_only'] == meta['permCF_to_canonical'][j]
                assert r['point_loss_target_before_local_input_support'] == before_support and r['target'] == y and r['target_reasons'] == reasons
                assert r['crop_available'] == available and r['loss_mask'] == (y in (0,1))
                assert r['compatibility_namespace_only'] == {'applied_after_own_image_proof':True,'family':'BEFORE_ANCHOR4','state_name':'NOMINAL','state_index':0,'not_historical_source_row_or_4base_variant_key':True}
                assert r['no_q_replacement_or_variant_label_inference'] is True
                point_counts[state]+=1;loss_counts[str(y)]+=1;reason_counts.update(reasons);original_u+=not av
                if y in (0,1):
                    assert r['sample_index'] == len(group)
                    group.append(gi);slot.append(j);labels.append(y);ids.append(rid)
                    condition=np.zeros(32,np.float32);condition[j]=1
                    for other in range(8):
                        if other != j and before['original_available8'][other]:
                            condition[8+other]=1
                            condition[16+2*other:18+2*other]=np.clip((np.asarray(before['q8'][other],np.float64)-np.asarray(q,np.float64))/np.float64(L),-8,8).astype(np.float32)
                    conditions.append(condition)
                else: assert r['sample_index'] is None
            scene_map[(s['base_index'],s['target_native_slot'],s['variant'])]=(gi,s,meta)
        group,slot=np.asarray(group,np.int64),np.asarray(slot,np.int64)
        assert len(group)==N and samples['target'].dtype==np.int8 and np.array_equal(samples['target'],labels)
        assert samples['row_id'].dtype.kind=='U' and samples['row_id'].tolist()==ids
        assert samples['fit'].dtype==samples['available'].dtype==bool and samples['fit'].all() and samples['available'].all()
        assert samples['family'].dtype==np.int8 and not samples['family'].any()
        assert samples['full_view'].dtype==samples['central_view'].dtype==np.int64
        assert np.array_equal(samples['full_view'],2*np.arange(N)) and np.array_equal(samples['central_view'],2*np.arange(N)+1)
        condition=np.stack(conditions);fraction=np.stack((g['node_fraction'][group,slot],center_fraction[group,slot]),1)
        assert samples['condition'].dtype==samples['fractions'].dtype==np.float32
        assert samples['condition'].tobytes()==condition.tobytes() and samples['fractions'].tobytes()==fraction.tobytes()
        assert samples['central4'].dtype==bool and np.array_equal(samples['central4'],central[group,slot])
        f={name:np.load(safe(c['features'][name]['path']),allow_pickle=False,mmap_mode='r') for name in FEATURE_KEYS}
        shapes=((N,1186),(G,8,384),(G,384),(N,),(N,),(N,8,2))
        for name,shape in zip(FEATURE_KEYS,shapes):
            arr=f[name];assert arr.shape==shape and arr.dtype==(np.int64 if name in FEATURE_KEYS[3:5] else np.float32)
            assert np.isfinite(arr).all() and ident(c['features'][name])==ident(output(name))
        assert np.array_equal(f['SAMPLE_GROUP_INDEX.npy'],group) and np.array_equal(f['SAMPLE_NODE_INDEX.npy'],slot)
        base,node,context=f[FEATURE_KEYS[0]],f[FEATURE_KEYS[1]],f[FEATURE_KEYS[2]]
        assert np.ascontiguousarray(base[:,:384]).tobytes()==np.ascontiguousarray(node[group,slot]).tobytes()
        assert np.ascontiguousarray(base[:,-32:]).tobytes()==condition.tobytes() and np.ascontiguousarray(base[:,1152:1154]).tobytes()==fraction.tobytes()
        relative=np.zeros((N,8,2),np.float64)
        np.divide(g['q8'][group]-g['q8'][group,slot,None,:],g['L8'][group,slot,None,None],out=relative,where=g['original_available8'][group,:,None])
        relative=np.clip(relative,-8,8).astype(np.float32);relative[~g['original_available8'][group]]=0;relative[np.arange(N),slot]=0
        assert f['RELATIVE_XY8.npy'].tobytes()==relative.tobytes()
        unit_or_zero(node,g['node_available']);unit_or_zero(context,g['context_fraction']>0)
        unit_or_zero(base[:,:384],np.ones(N,bool));unit_or_zero(base[:,768:1152],fraction[:,1]>0)
        zero_q=unit_or_zero(base[:,384:768],samples['central4'],True)
        cq=js(c['canonical_sharing_QA_binding']);compiler=js(c['target_compiler_QA'])
        assert cq==c['canonical_sharing_QA'] and cq['completed'] is True
        assert cq['active_first384_byte_exact'] is True and cq['condition32_byte_exact'] is True and cq['first_node_batch_repeat_byte_exact'] is True
        assert cq['frozen_encoder_state_sha256_before']==cq['frozen_encoder_state_sha256_after']==c['base_state_sha256_before']==c['base_state_sha256_after']==STATE
        assert c['official_base']['sha256'] == c['asset_identity']['sha256'] == 'f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb'
        assert c['asset_identity']['vendor_revision'] == '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
        assert cq['supported_nodes']==int(g['node_available'].sum()) and cq['context_supported']==int((g['context_fraction']>0).sum())
        assert cq['known']==N and cq['groups']==G and cq['new_classifier_or_optimizer_updates']==0 and cq['teacher_fields_forwarded'] is False
        assert compiler['completed'] is True and compiler['PASS'] is True and compiler['record_identity_failures']==0
        assert compiler['scenes']==G and compiler['all_native_rows']==8*G and compiler['known_supported_rows']==N and compiler['masked_rows']==8*G-N
        assert compiler['point0_states']==dict(point_counts) and compiler['loss_targets']==dict(loss_counts) and compiler['reasons']==dict(reason_counts) and compiler['original_U_rows_preserved']==original_u
        census=js(output('CENSUS.json'))
        assert census==c['counts'] and census['new_scenes']==G and census['all_new_native_rows']==8*G and census['known_rows']==N and census['masked_rows']==8*G-N
        # Same predeclared sixteen template/native cases as Stage2 independent QA.
        requested=[]
        for case in ray:
            if case['base_index'] is None: continue
            requested += [(case['base_index'],None,'BEFORE')]+[(case['base_index'],case['native_slot'],role) for role in ('COVER','NEAR_SHAM','FAR_SHAM')]
        local=load_numpy_kernels(source_entry['code_bindings']['local_crop'],'6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b',('crop',))
        ctx=load_numpy_kernels(source_entry['code_bindings']['context_crop'],'aa7cd8e2568c8b7d30827e8681b4b87899944bd23cfa1b27948ceda5b55d07f3',('valid_box','target_context_grid','sample_target_context'))
        for key in sorted(set(requested),key=lambda x:(x[0],-1 if x[1] is None else x[1],x[2])):
            gi,s,meta=scene_map[key]
            with np.load(io.BytesIO(read(s['image'])),allow_pickle=False) as z:
                assert z.files==['RGB_uint8_HWC'];rgb=z['RGB_uint8_HWC']
            assert hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()==s['RGB_sha256_HWC']
            before=meta['original_before_input']
            for j in range(8):
                q,L=before['q8'][j],before['L8'][j]
                expected_rgb=np.zeros((3,96,96),np.uint8);expected_valid=np.zeros((96,96),bool)
                if finite_pair(q) and type(L) in (int,float) and math.isfinite(L) and L>0:
                    expected_rgb,v=local['crop'](rgb,q,float(L),96);expected_valid=v[0]
                assert expected_rgb.tobytes()==np.ascontiguousarray(raw['rgb'][gi,j]).tobytes() and np.array_equal(expected_valid,raw['valid'][gi,j])
            expected_context=np.zeros((3,224,224),np.float32);expected_cv=np.zeros((224,224),bool)
            if ctx['valid_box'](before['box_xyxy']):
                x,expected_cv=ctx['sample_target_context'](rgb,before['box_xyxy'],(224,224));expected_context=np.ascontiguousarray(x.transpose(2,0,1))
            assert expected_context.tobytes()==np.ascontiguousarray(raw['context'][gi]).tobytes() and np.array_equal(expected_cv,raw['cv'][gi])
        review={'completed':True,'PASS':True,'feature_complete':e['feature_complete'],'code':e['code'],'entry':hash_file(entry_path),'read_scope':e['read_scope'],
            'all_native_rows':8*G,'new_scenes':G,'known_rows':N,'masked_rows':8*G-N,'known_D':labels.count(0),'known_E':labels.count(1),'original_U_rows_preserved':original_u,
            'all_feature_dtype_shape_finite_input_support_PASS':True,'all_N_active_NODE_first384_bytes_exact':True,'condition_fractions_relative_bytes_exact':True,
            'independent_own_scene_j_k_identity_targets_locality_reasons_PASS':True,'supported_zero_Q_rows':zero_q,
            'crop_replay':{'fixed_cases':len(ray),'scene_slots':len(requested),'unique_scenes':len(set(requested)),'all8_local_and_context_bytes_exact':True,'source_ray_subset':e['ray_subset']},
            'encoder_claim_scope':'Producer frozen CODE/metadata asserts official state and first repeat; independent QA here verifies saved feature algebra, not encoder/weight recomputation.',
            'producer_import_or_call':False,'new_encoder_or_model_forward':0,'official_weight_body_or_old_Source_or_current_held_access':0,'model_data_architecture_threshold_choice':False,
            'training_admission_by_this_QA':False,'physical_native_literalq_wholepart_gold_claim':False}
        b=put(out/'REVIEW.json',review)
        print(json.dumps({'completed':True,'PASS':True,'review':b,'all_rows':8*G,'known_rows':N,'D':labels.count(0),'E':labels.count(1),'masked':8*G-N,'original_U':original_u,'crop_unique_scenes':len(set(requested))},sort_keys=True))
    except BaseException as exc:
        put(out/'FAILURE.json',{'completed':False,'PASS':False,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--entry',required=True);run(p.parse_args().entry)
