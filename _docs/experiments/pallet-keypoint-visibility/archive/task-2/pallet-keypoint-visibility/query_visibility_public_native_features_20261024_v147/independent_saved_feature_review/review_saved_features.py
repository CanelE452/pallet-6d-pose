"""Independent source-only saved feature QA. No torch/model/GT imports or I/O."""
from pathlib import Path
import ast, collections, hashlib, json
import numpy as np

H=Path(__file__).resolve().parent
V=H.parent
COMPLETE=V/'public_native_DINO512_v1/FEATURE_COMPLETE.json'
EXPECTED='81881cc015e0d6a59df4beaf93c4932c2e70abd99c3da40cc1225f2d3653ff2f'
READS=[]

def sha(path):
    path=Path(path);READS.append(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()
def binding(path): return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(b):
    assert sha(b['path'])==b['sha256'],b['path']
    return Path(b['path'])
def read(path):
    READS.append(str(path));return json.loads(Path(path).read_text())
def lines(path):
    READS.append(str(path));return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def pure(path,names):
    READS.append(str(path));tree=ast.parse(Path(path).read_text())
    ns={'np':np};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::input_math_only','exec'),ns)
    return ns

def main():
    assert sha(COMPLETE)==EXPECTED
    c=read(COMPLETE);assert c['completed'] and c['PASS']
    for k in ('code','protocol','policy','entry','inputs','input_seal','features','inputfacts'):check(c[k])
    p=read(c['protocol']['path']);packet=read(c['inputs']['path']);frames=packet['frames']
    assert len(frames)==64 and p['rows']==512 and p['context_box_mode']==c['context_box_mode']=='FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX'
    seal=read(c['input_seal']['path']);assert seal['completed'] and seal['PASS'] and seal['inputs']==c['inputs']
    for b in c['source_input_only'].values():check(b)
    original=packet['original_input_bindings']
    for b in original.values():check(b)
    props=lines(original['proposals']['path']);queries=lines(original['queries']['path'])
    assert len(props)==64 and len(queries)==512
    READS.append(c['features']['path'])
    with np.load(c['features']['path'],allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    READS.append(original['query_npz']['path'])
    with np.load(original['query_npz']['path'],allow_pickle=False) as f:q0={k:f[k].copy() for k in ('query_xy','L','continuous_valid','nearest_valid','query_id','global_index','frame_index','native_slot','frame_id','split')}
    facts=lines(c['inputfacts']['path']);assert len(facts)==512
    x=a['features'];assert x.shape==(512,1928) and x.dtype==np.float32 and np.isfinite(x).all()
    expected_keys={'features','global_index','frame_index','native_slot','query_id','frame_id','source_id','split','query_xy','L','selected_index','query_available','native_continuous_valid','nearest_valid','local_central4_available','context_available','support_fractions','unit_Q_inside','occurrence_unit'}
    assert set(a)==expected_keys
    assert exact(a['global_index'],np.arange(512,dtype=np.int32))
    assert exact(a['occurrence_unit'],np.arange(512,dtype=np.int32))
    assert exact(a['frame_index'],np.repeat(np.arange(64,dtype=np.int32),8))
    assert exact(a['native_slot'],np.tile(np.arange(8,dtype=np.int8),64))
    for k in ('query_xy','L','query_id','global_index','frame_index','native_slot','frame_id','split'):
        assert exact(a[k],q0[k]),('original R0 metadata bytes',k)
    assert exact(a['query_available'],q0['continuous_valid'])
    assert exact(a['native_continuous_valid'],q0['continuous_valid'])
    assert exact(a['nearest_valid'],q0['nearest_valid'])
    assert a['query_available'].dtype==bool and int(a['query_available'].sum())==463
    assert int((~a['query_available']).sum())==49
    target=pure(check(p['target_crop_code']),('valid_box','target_context_grid','sample_target_context'))
    crop=pure(check(p['local_crop_code']),('crop',))['crop']
    expected_inside=[];expected_c4=[];expected_context=[];expected_fractions=[]
    for frame,(r,s) in enumerate(zip(frames,props)):
        assert r['frame_index']==s['frame_index']==frame
        selected=int(np.argmax([z['score'] for z in s['candidates']]))
        assert r['selected_index']==s['selected_index']==selected
        q9=np.asarray(s['candidates'][selected]['keypoints_xy'],np.float64);q=q9[:8]
        assert q.shape==(8,2) and np.isfinite(q).all()
        good=np.isfinite(q).all(1)&~(q==[-1.,-1.]).all(1)
        lo=q[good].min(0);hi=q[good].max(0);box=np.r_[lo,hi]
        assert r['context_box_xyxy']==box.tolist() and r['initial_points']==q.tolist()
        # Native side excludes centroid and is independently recomputed.
        L=float(np.clip(.25*np.max(np.ptp(q[good],axis=0)),16,max(r['raw_hw'])))
        assert r['L']==L
        sl=slice(frame*8,frame*8+8)
        assert exact(a['query_xy'][sl],q) and np.all(a['L'][sl]==L)
        assert a['source_id'][sl].tolist()==[r['source_id']]*8
        assert np.all(a['selected_index'][sl]==selected)
        rgb=np.load(check(r['canonical_RGB_NPY']),allow_pickle=False)
        assert rgb.dtype==np.uint8 and list(rgb.shape)==r['raw_hw']+[3] and ah(rgb)==r['decoded_RGB_sha256']
        with np.load(check(r['crop_packet']),allow_pickle=False) as f:
            assert set(f.files)=={'rgb96','valid96','global_index'}
            hwc=f['rgb96'].copy();valid=f['valid96'].copy()
            assert exact(f['global_index'],np.arange(frame*8,frame*8+8,dtype=np.int32))
        assert hwc.shape==(8,96,96,3) and hwc.dtype==np.uint8 and valid.shape==(8,96,96) and valid.dtype==bool
        chw=np.ascontiguousarray(hwc.transpose(0,3,1,2));t,vt=target['sample_target_context'](rgb,box)
        assert t.dtype==np.float32 and t.shape==(224,224,3) and vt.dtype==bool and vt.shape==(224,224)
        for slot in range(8):
            i=frame*8+slot;f=facts[i];qr=queries[i]
            for key,value in dict(global_index=i,frame_index=frame,frame_id=r['id'],source_id=r['source_id'],split=r['split'],native_slot=slot,query_id=r['query_ids'][slot],query_xy=q[slot].tolist(),L=L,selected_index=selected,context_box_xyxy=box.tolist()).items():assert f[key]==value,(i,key)
            assert qr['query_id']==f['query_id'] and qr['query_xy']==f['query_xy'] and qr['L']==f['L']
            ca,cv=crop(rgb,q[slot],L,96)
            assert exact(ca,chw[slot]) and exact(cv,valid[slot][None])
            hashes=dict(RGB96_HWC_sha256=ah(hwc[slot]),RGB96_CHW_sha256=ah(chw[slot]),valid96_sha256=ah(valid[slot]),target224_F32_HWC_sha256=ah(t),target_valid_sha256=ah(vt))
            for key,value in hashes.items():assert f[key]==value,(i,key)
            fractions=np.array([vt.mean(),valid[slot].mean(),valid[slot,32:64,32:64].mean()],np.float32)
            c4=bool(valid[slot,47:49,47:49].all());ctx=bool(vt.any())
            u=(q[slot]-lo)/(hi-lo);inside=bool(ctx and (u>=0).all() and (u<=1).all())
            assert exact(a['support_fractions'][i],fractions)
            assert f['fractions']==fractions.tolist()
            for key,expected in [('Q_inside',inside),('local_central4_available',c4),('context_available',ctx),('native_continuous_valid',bool(q0['continuous_valid'][i])),('nearest_valid',bool(q0['nearest_valid'][i]))]:assert f[key] is expected,(i,key)
            expected_inside.append(inside);expected_c4.append(c4);expected_context.append(ctx);expected_fractions.append(fractions)
    fr=np.array(expected_fractions,np.float32);inside=np.array(expected_inside,bool);c4=np.array(expected_c4,bool);context=np.array(expected_context,bool)
    assert exact(a['unit_Q_inside'],inside) and exact(a['local_central4_available'],c4) and exact(a['context_available'],context)
    assert exact(a['support_fractions'],fr) and exact(x[:,1924:1927].copy(),fr)
    assert exact(x[:,1927].copy(),inside.astype(np.float32))
    support=[fr[:,0]>0,inside,fr[:,1]>0,fr[:,2]>0,c4]
    block_names=['T','Q','F','C','P'];blocks=[];norms={}
    for j,(name,mask) in enumerate(zip(block_names,support)):
        b=x[:,j*384:(j+1)*384];blocks.append(b)
        assert not b[~mask].any(),(name,'unsupported must be zero')
        norm=np.linalg.norm(b.astype(np.float64),axis=1)
        maxerr=float(np.max(np.abs(norm[mask]-1))) if mask.any() else 0.
        assert maxerr<1e-6,(name,maxerr)
        norms[name]=dict(supported=int(mask.sum()),unsupported=int((~mask).sum()),max_supported_unit_norm_abs_error=maxerr)
    # Independent float64 cosine recomputation from saved normalized blocks.
    pairs=[(0,3),(0,2),(1,3),(1,4)];max_cos=0.
    for index,(j,k) in enumerate(pairs):
        u=blocks[j].astype(np.float64);v=blocks[k].astype(np.float64)
        expected=np.sum(u*v,axis=1)/(np.maximum(np.linalg.norm(u,axis=1),1e-12)*np.maximum(np.linalg.norm(v,axis=1),1e-12))
        err=float(np.max(np.abs(expected-x[:,1920+index].astype(np.float64))))
        assert err<1e-6,(index,err);max_cos=max(max_cos,err)
    assert c['primary_encoder_calls']==64 and c['repeat_first_calls']==1 and c['total_encoder_calls']==65 and c['views_per_call']==17
    assert c['first_repeat_raw_tokens_bitexact'] and c['encoder_state_unchanged'] and c['weight_and_inputs_unchanged']
    runtime=read(check(p['asset_runtime']))
    assert c['frozen_encoder_state_before']==c['frozen_encoder_state_after']==p['expected_encoder_state']==runtime['state_before_SHA256']
    assert c['classifier_R0_P3_fit_private_PnP_calls']==0 and c['labels_pose_depth_reference_body_reads']==0
    # No source semantics/pose/depth filenames or model weights were opened.
    assert p['weight']['path'] not in READS
    assert not any('ALL512_CROP_FACTS_AND_UNREVIEWED_TEMPLATE' in path for path in READS)
    review=dict(completed=True,PASS=True,code=binding(__file__),feature_complete=binding(COMPLETE),features=c['features'],inputfacts=c['inputfacts'],inputs=c['inputs'],
        all512_rows_recomputed=True,frames=64,rows=512,available=463,finite_OOB=49,
        folds=dict(collections.Counter(a['split'].tolist())),
        original_R0_ID_q_L_fold_slot_and_availability_bytes_exact=True,all_firstargmax_selections_exact=True,
        independent_q8_minmax_and_native_L_exact=True,saved64_RGB96_HWC_to_CHW_valid_and_bilinear_crop_parity=True,
        target64_F32_RGB_and_valid_hashes_exact=True,all512_component_flags_fractions_exact=True,
        embedding_norm_and_zero_rules=norms,cosine_max_abs_error=max_cos,
        support_counts=dict(central4=int(c4.sum()),context=int(context.sum()),Qinside=int(inside.sum()),nearest=int(a['nearest_valid'].sum())),
        stored_scalar4fractions_and_Qinside_exact=True,
        state_claim_validation='Compared completed65call/repeat/state/file claims to sealed protocol/runtime receipts only; did not instantiate or forward any model. Raw encoder tokens are not independently reencoded.',
        model_or_encoder_or_classifier_forward=0,model_weight_body_reads=0,semantic_labels_GT_pose_depth_body_reads=0,
        current_private_body_reads=0,source_RGB_and_saved_crop_math_only=True,
        dataflow_limit='Norm/scalar checks verify saved feature algebra and input lineage; they do not independently recreate frozen DINO embeddings.',
        files_read=sorted(set(READS)))
    output=H/'REVIEW_COMPLETE.json'
    with output.open('x') as f:json.dump(review,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(review=binding(output),support_counts=review['support_counts'],norms=norms,cosine_max_abs_error=max_cos)))

if __name__=='__main__':main()
