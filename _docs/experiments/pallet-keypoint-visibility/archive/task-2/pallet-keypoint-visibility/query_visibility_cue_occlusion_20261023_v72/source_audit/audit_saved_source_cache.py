"""Source-only input identity/native9-valid audit; no model import or forward."""
from pathlib import Path
from collections import Counter
import ast, hashlib, json
import numpy as np

R=Path(__file__).resolve().parents[1];P=R.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
    out=R/'source_audit/INDEPENDENT_SOURCE_CACHE_AUDIT.json';assert not out.exists()
    code=R/'prepare_source_and_features.py';receipt=load(R/'source_cache/FEATURE_RECEIPT.json');prepared=load(R/'source_cache/PREPARED.json')
    assert sha(code)==receipt['code_sha256']
    for b in prepared['source_bindings']+prepared['outputs']:assert sha(b['path'])==b['sha256']
    for name,key in [('MODEL_INPUT_ONLY.npz','input_sha256'),('INPUT_INDEX_ONLY.jsonl','index_sha256'),('P3_ROIS.npz','feature_sha256')]:assert sha(R/'source_cache'/name)==receipt[key]
    s=P/'query_visibility_reference_jitter_20261023_v71/data/synth_reference_jitter/BUILD_V2';y=R/'data/ycb_reference_cover'
    original_s=lines(s/'ROWS.jsonl');original_y=lines(y/'INPUT_INDEX_LINEAGE_ONLY.jsonl');index=lines(R/'source_cache/INPUT_INDEX_ONLY.jsonl')
    assert len(original_s)==260 and len(original_y)==3645 and len(index)==3905
    for n,(actual,expected) in enumerate(zip(index,original_s+original_y)):
        assert actual['input_index']==n
        if n<260:
            assert actual['row_id']=='SYNTH:'+expected['row_id']
            assert actual['image']==expected['model_inputs']['rgb']['path'] and actual['image_sha256']==expected['model_inputs']['rgb']['sha256']
        else:
            assert actual['row_id']=='YCB:'+expected['row_id']
            assert actual['image']==expected['image'] and actual['image_sha256']==expected['image_sha256']
        assert set(actual)=={'input_index','row_id','image','image_sha256'}
    assert len({r['row_id'] for r in index})==3905
    with np.load(R/'source_cache/MODEL_INPUT_ONLY.npz',allow_pickle=False) as z:inp={k:z[k] for k in z.files}
    assert set(inp)=={'rgb96','crop_valid96','query_valid','query_xy','box_xyxy','L'}
    checks={}
    with np.load(s/'MODEL_INPUTS_ONLY.npz',allow_pickle=False) as a:
        expected=dict(rgb96=a['rgb'].transpose(0,3,1,2),crop_valid96=a['valid'][:,None].astype(bool),query_valid=np.isfinite(a['initial_query_xy']).all(1),query_xy=a['initial_query_xy'],box_xyxy=a['target_box_xyxy'],L=a['input_crop_L'])
        for k,v in expected.items():assert np.array_equal(inp[k][:260],v,equal_nan=True);checks['SYNTH:'+k]='exact'
    with np.load(y/'MODEL_INPUT_ONLY.npz',allow_pickle=False) as b:
        for k in b.files:assert np.array_equal(inp[k][260:],b[k],equal_nan=True);checks['YCB:'+k]='exact'
    assert len(inp['L'])==3905 and np.isfinite(inp['query_xy']).all() and np.isfinite(inp['L']).all() and (inp['L']>0).all()
    groups={}
    for n,r in enumerate(index):groups.setdefault(r['image_sha256'],[]).append(n)
    assert len(groups)==receipt['whole_RGB_forwards']==prepared['unique_RGB']==62
    records={r['image_sha256']:r for r in receipt['RGB_records']};assert set(records)==set(groups)
    hw=np.zeros((3905,2),np.int32);gain=np.zeros(3905);off=np.zeros((3905,2));affine=[]
    for h,nn in groups.items():
        rr=records[h];assert rr['rows']==len(nn) and rr['image']==index[nn[0]]['image']
        raw_h,raw_w=rr['raw_hw'];ih,iw=rr['input_shape'];g=min(640/(raw_h+200),640/(raw_w+200))
        expected=[round((iw-round((raw_w+200)*g))/2-.1),round((ih-round((raw_h+200)*g))/2-.1)]
        assert rr['gain']==g and rr['offset']==expected and max(ih,iw)==640 and ih%32==iw%32==0
        hw[nn]=rr['raw_hw'];gain[nn]=g;off[nn]=expected;affine.append(rr)
    # Reconstruct the exact float32 sequence used by V72 Torch elementwise ops.
    axis=(np.arange(9,dtype=np.float32)-np.float32(4))/np.float32(9)
    xx,yy=np.meshgrid(axis,axis);offset=np.stack((xx,yy),-1)
    grid=inp['query_xy'].astype(np.float32)[:,None,None,:]+offset[None]*inp['L'].astype(np.float32)[:,None,None,None]
    valid=np.isfinite(grid).all(-1)&(grid[...,0]>=0)&(grid[...,0]<hw[:,None,None,1])&(grid[...,1]>=0)&(grid[...,1]<hw[:,None,None,0])
    with np.load(R/'source_cache/P3_ROIS.npz',allow_pickle=False) as z:
        assert z.files==['x'];x=z['x']
    assert x.shape==(3905,65,9,9) and x.dtype==np.float16 and np.isfinite(x).all()
    assert np.array_equal(x[:,64],valid.astype(np.float16))
    assert np.all(x[:,:64][np.broadcast_to(~valid[:,None],x[:,:64].shape)]==0)
    assert np.array_equal(grid[:,4,4],inp['query_xy'].astype(np.float32))
    # Prior V60 arithmetic: same phase, double native grid before norm->float32.
    axis64=((np.arange(9,dtype=np.float64)+.5)/9-.5)
    xx64,yy64=np.meshgrid(axis64,axis64)
    grid64=inp['query_xy'][:,None,None,:]+np.stack((xx64,yy64),-1)[None]*inp['L'][:,None,None,None]
    valid64=np.isfinite(grid64).all(-1)&(grid64[...,0]>=0)&(grid64[...,0]<hw[:,None,None,1])&(grid64[...,1]>=0)&(grid64[...,1]<hw[:,None,None,0])
    norm32=2*((((grid+np.float32(100))*gain.astype(np.float32)[:,None,None,None]+off.astype(np.float32)[:,None,None,:])/np.float32(8))+np.float32(.5))/np.float32(80)-np.float32(1)
    norm64=(2*((((grid64+100)*gain[:,None,None,None]+off[:,None,None,:])/8)+.5)/80-1).astype(np.float32)
    old=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
    old_text=old.read_text();assert 'index=network/8.;norm=2*(index+.5)/80-1' in old_text and "align_corners=False" in old_text
    tree=ast.parse(code.read_text());features=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='features')
    feature_text=ast.unparse(features)
    assert "GT_ONLY" not in feature_text and "reference_xy" not in feature_text and "uncertainty" not in feature_text
    assert 'data[\'query_xy\']' in feature_text and 'data[\'L\']' in feature_text
    assert 'grad' in feature_text
    result=dict(schema='V72_INDEPENDENT_SAVED_SOURCE_INPUT_AND_NATIVE9_CACHE_AUDIT',PASS=True,
      N=3905,SYNTH_prefix=260,YCB_prefix=3645,unique_row_ids=3905,unique_full_RGB=62,
      all_source_prefix_rowid_RGB_mapping_exact=True,all_original_q_L_box_and_RGB96_valid_arrays_unchanged=checks,
      only_model_input_keys=sorted(inp),GT_fields_in_model_inputs=False,
      feature_code_GT_not_read_or_passed=True,source_annotation_guided_query_or_box_lineage_not_erased='These are preregistered source proposal/teacher-derived inputs, not deployment initializer queries; GT is absent from actual model tensors but source sampler provenance is not GT-independent.',
      native9=dict(shape=list(x.shape),dtype=str(x.dtype),finite=True,valid_cells=int(valid.sum()),invalid_cells=int((~valid).sum()),total_cells=3905*81,
        all_saved_P3_valid_channels_match_native_grid=True,invalid_feature_cells_exact_zero=True,center_float32_q_exact=True,query_valid_true=int(inp['query_valid'].sum()),center_native_valid=int(valid[:,4,4].sum()),
        phase='(i-4)L/9; ((native+100)*gain+integer_letterbox_offset)/8; norm=2*(feature_index+.5)/80-1; grid_sample align_corners=False/zeros',
        prior_V60_same_phase=True,exact_receptive_field_center_claim=False,
        float32_native_grid_vs_V60_float64_maxdiff_px=float(np.max(np.abs(grid-grid64))),
        float32_norm_vs_V60_norm_to_float32_maxdiff=float(np.max(np.abs(norm32-norm64))),
        native_valid_disagreements_vs_float64=int(np.count_nonzero(valid!=valid64))),
      affine_62_records_independently_match=True,receipt_no_fit_current_PnP=True,
      limits=['No fresh P3/backbone replay or learned model forward. Valid-mask and input identity were recalculated from saved source arrays; first64 feature channels checked finite and invalidzero only.','V72 float32 native arithmetic is phase-equivalent to prior V60 but not promised bit-identical to V60 float64-then-float32 normalization.','Source q/L/box heldfixed; physical/native8 corner semantics and visibility teacher quality are outside this numerical cache audit.'],
      current223_RGB_GT_scores_features_model_weights_newforward_fit_PnP=0,
      bindings=[bind(z) for z in [code,old,R/'source_cache/PREPARED.json',R/'source_cache/FEATURE_RECEIPT.json',R/'source_cache/INPUT_INDEX_ONLY.jsonl',R/'source_cache/MODEL_INPUT_ONLY.npz',R/'source_cache/P3_ROIS.npz',s/'MODEL_INPUTS_ONLY.npz',s/'ROWS.jsonl',y/'MODEL_INPUT_ONLY.npz',y/'INPUT_INDEX_LINEAGE_ONLY.jsonl']],audit_code=bind(Path(__file__)))
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(PASS=True,audit=bind(out),native9=result['native9']),ensure_ascii=False))
if __name__=='__main__':main()
