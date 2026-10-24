"""Independent saved-output audit; no RGB, GT, model imports or forward calls."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,time,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'source_audit';OUT=ROOT/'source_outputs/SAM21_BASE_PLUS'
SEAL_SHA='1ee0ca1fe2a1575aaae12d6b090977f5e3a2ab8ca48028345e1070ce8e01ebd0'
HASHES={}
def sha(p):
    p=Path(p)
    if str(p)not in HASHES:
        h=hashlib.sha256()
        with p.open('rb')as f:
            while b:=f.read(8*1024*1024):h.update(b)
        HASHES[str(p)]=h.hexdigest()
    return HASHES[str(p)]
def ref(p):return dict(path=str(p),sha256=sha(p))
def bound(b):
    assert sha(b['path'])==b['sha256'],('SHA changed',b['path'])
    if 'bytes'in b:assert Path(b['path']).stat().st_size==b['bytes']
def equals(a,b):return np.array_equal(np.asarray(a),np.asarray(b))
def scalar_equal(a,b):return(math.isnan(a)and math.isnan(b))or a==b
def write(path,j):
    assert not path.exists();path.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')

def main():
    report_path=HERE/'INDEPENDENT_EXTRACTION_OUTPUT_AUDIT.json';assert not report_path.exists();start=time.monotonic()
    sp=OUT/'EXTRACTION_SEAL.json';assert sha(sp)==SEAL_SHA;s=json.loads(sp.read_text());assert s['completed']and s['source_only']
    manifest_path=ROOT/'source_inputs/INPUTS_RGB_ONLY.jsonl';manifest=[json.loads(x)for x in manifest_path.read_text().splitlines()]
    with(ROOT/'source_inputs/QUERY_ID_MAP.csv').open()as f:mapping=list(csv.DictReader(f))
    selected=set()
    for source in sorted(set(r['source']for r in manifest)):
        ix=[i for i,r in enumerate(manifest)if r['source']==source]
        selected.update([ix[0],ix[len(ix)//2],ix[-1]])
    for hw in sorted(set(tuple(r['raw_hw'])for r in manifest if r['source']=='SYNTH')):
        selected.add(next(i for i,r in enumerate(manifest)if r['source']=='SYNTH'and tuple(r['raw_hw'])==hw))
    selected.update(i for i,r in enumerate(manifest)if any(not q['native_input_valid']for q in r['queries']))
    selected=sorted(selected)
    protocol=dict(schema='INDEPENDENT_V68_EXTRACTION_OUTPUT_AUDIT_FIXED_SAMPLE_V1',source_extraction_seal=ref(sp),
        sampling='Each source first/upper-middle/last manifest target;first SYNTH target pernativeHW;all targets containing input-manifest-invalid queries. No quality/model scores or GT select this subset.',
        selected_manifest_indices=selected,selected_targets=[dict(index=i,id=manifest[i]['id'],source=manifest[i]['source'],raw_hw=manifest[i]['raw_hw'])for i in selected],
        scope='All artifact hashes/metadata/branch argmax/query mappings;fixed native full-map float32 scalar exact/r2 replay. No extractor helper imported.',
        preliminary_schema_read='Mask000000 was inspected for schema before this protocol and is already the deterministic first SYNTH sample.',
        forbidden=['RGBbytes','GT/masks/annotations','current data or results','model construction/forward/training','checkpoint deserialization'],code=ref(Path(__file__)))
    protocol['auditor_revision']='V1 failed on np.array_equal(equal_nan=True) applied to string identity arrays. V1 code/protocol preserved; only audit equality helper corrected, no source/output/model artifact changed.'
    pp=HERE/'INDEPENDENT_EXTRACTION_OUTPUT_AUDIT_PROTOCOL_V2.json';write(pp,protocol)
    binding_results=[]
    for role,b in s['bindings'].items():bound(b);binding_results.append(dict(role=role,**b,PASS=True))
    entry=json.loads((ROOT/'ENTRYLOCK.json').read_text());assert entry['approved']and entry['source_only']and not entry['current223_allowed']
    for b in entry['inputs'].values():bound(b)
    for b in entry['authorizing_evidence']:bound(b)
    runtime=json.loads(Path(s['bindings']['runtime_seal']['path']).read_text());assert runtime['READY']
    for b in runtime['bindings']:bound(b)
    bound(runtime['checkpoint'])
    for b in s['outputs']:bound(b)
    start_record=json.loads((OUT/'RUN_STARTED.json').read_text());assert start_record['bindings']==s['bindings']
    assert s['model_state_sha256']==s['final_model_state_sha256']and len(s['model_state_sha256'])==64
    assert s['model_weights_unchanged']and s['checkpoint_unchanged']
    assert s['training_calls']==s['optimizer_updates']==s['PnP_calls']==0 and not s['current_inference_allowed']
    targets=json.loads((OUT/'TARGET_OUTPUTS.json').read_text());assert len(targets)==len(manifest)==1091
    with(OUT/'PER_QUERY.csv').open()as f:queries=list(csv.DictReader(f))
    assert len(queries)==len(mapping)==8590 and[int(r['joined_query_index'])for r in queries]==list(range(8590))
    with np.load(OUT/'SOURCE_QUERY_OUTPUTS.npz',allow_pickle=False)as z:
        expected={'sample_id','source','group_id','global_id','native_valid','exact_logit','radius2_logit','exact_visible','radius2_visible','selected_mask_index'}
        assert set(z.files)==expected;qp={k:z[k]for k in z.files}
    assert all(a.shape==(8590,)for a in qp.values())and qp['native_valid'].dtype==bool
    for k in ['exact_logit','radius2_logit']:assert qp[k].dtype==np.float32
    for k in ['exact_visible','radius2_visible','selected_mask_index']:assert qp[k].dtype==np.int8
    source_queries=Counter();source_targets=Counter();invalid_reasons=Counter();invalid_ids=[];ties=[];samples=[]
    mask_schema={'chosen_logit','quality_scores','selected_index','original_prompt_box','predictor_prompt_box','raw_hw','sample_id','joined_query_index','query','pixel','native_valid','exact_logit','radius2_logit','exact_visible','radius2_visible'}
    mask_paths={x['path']:x for x in s['mask_outputs']};assert len(mask_paths)==1091
    assert set(mask_paths)==set(str(p)for p in(OUT/'masks').glob('*.npz'))
    queried_indices=[];sample_query_n=sample_invalid_n=mask_bytes=0
    for ti,(m,t)in enumerate(zip(manifest,targets)):
        assert t['manifest_index']==ti and t['joined_target_index']==m['joined_target_index']
        for k in ['global_id','source','group_id','image','image_sha256','raw_hw','box','side']:assert t[k]==m[k]
        h,w=m['raw_hw'];assert t['query_count']==len(m['queries']);source_targets[m['source']]+=1
        b=t['mask_output'];assert b==mask_paths[b['path']];bound(b);mask_bytes+=b['bytes']
        with zipfile.ZipFile(b['path'])as archive:
            with archive.open('chosen_logit.npy')as f:
                version=np.lib.format.read_magic(f)
                shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f)if version==(1,0)else np.lib.format.read_array_header_2_0(f))
                assert shape==(h,w)and dtype==np.float32 and not fortran
        with np.load(b['path'],allow_pickle=False)as z:
            assert set(z.files)==mask_schema
            arrays={k:z[k]for k in z.files if k!='chosen_logit'}
            native=z['chosen_logit']if ti in selected else None
        quality=arrays['quality_scores'];selected_index=int(arrays['selected_index'])
        assert quality.dtype==np.float32 and quality.shape==(3,)and np.isfinite(quality).all()
        assert equals(quality,np.asarray(t['predicted_quality'],np.float32))
        maxima=[j for j,v in enumerate(quality)if v==quality.max()];assert selected_index==maxima[0]==t['selected_mask_index']
        if len(maxima)>1:ties.append(dict(manifest_index=ti,global_id=m['id'],quality=quality.tolist(),tied_max_indices=maxima,selected=selected_index))
        assert equals(arrays['original_prompt_box'],m['box'])and arrays['original_prompt_box'].dtype==np.float64
        assert equals(arrays['predictor_prompt_box'],np.asarray(m['box'],np.float32))
        assert equals(arrays['predictor_prompt_box'],t['predictor_box_float32'])and equals(arrays['raw_hw'],[h,w])
        assert equals(arrays['sample_id'],[q['sample_id']for q in m['queries']])
        assert equals(arrays['joined_query_index'],[q['joined_query_index']for q in m['queries']])
        assert equals(arrays['query'],[q['query']for q in m['queries']])and equals(arrays['pixel'],[q['pixel']for q in m['queries']])
        assert equals(arrays['native_valid'],[q['native_input_valid']for q in m['queries']])
        if native is not None:assert native.dtype==np.float32 and native.shape==(h,w)and np.isfinite(native).all()
        st=dict(manifest_index=ti,global_id=m['id'],source=m['source'],raw_hw=[h,w],queries=len(m['queries']),invalid=0,
            native_exact_radius2_float32_scalar_replay=True,mask_sha256=b['sha256'])
        if native is not None:st.update(native_pixels=int(native.size),strict_positive_pixels=int((native>0).sum()),exact_zero_pixels=int((native==0).sum()),
            strict_positive_mask_SHA256=hashlib.sha256(np.ascontiguousarray(native>0).tobytes()).hexdigest())
        for j,q in enumerate(m['queries']):
            ix=q['joined_query_index'];queried_indices.append(ix);a=queries[ix];mm=mapping[ix]
            for k in ['sample_id','global_id','source','group_id','image_sha256']:assert a[k]==mm[k]
            assert int(a['joined_target_index'])==m['joined_target_index']and int(a['native_corner'])==q['native_corner']
            assert [float(a['query_x']),float(a['query_y'])]==q['query']and [int(a['pixel_x']),int(a['pixel_y'])]==q['pixel']
            assert [int(a['native_H']),int(a['native_W'])]==[h,w]and float(a['side'])==m['side']
            x,y=q['query'];px,py=math.floor(x+.5),math.floor(y+.5)
            valid=0<=x<w and 0<=y<h and 0<=px<w and 0<=py<h
            reason='CONTINUOUS_OUTSIDE_NATIVE'if not(0<=x<w and 0<=y<h)else'ROUNDED_OUTSIDE_NATIVE'if not valid else''
            assert valid==q['native_input_valid']==(a['native_valid']=='True')==bool(qp['native_valid'][ix])
            assert a['status']==('VALID'if valid else'U')and a['invalid_reason']==reason
            assert int(a['selected_mask_index'])==selected_index==int(qp['selected_mask_index'][ix])
            for k in range(3):assert float(a[f'predicted_quality_{k}'])==float(quality[k])
            for k in ['sample_id','global_id','source','group_id']:assert a[k]==str(qp[k][ix])
            for k in ['exact_logit','radius2_logit']:
                value=float(arrays[k][j]);assert scalar_equal(value,float(a[k]))and scalar_equal(value,float(qp[k][ix]))
                assert math.isfinite(value)if valid else math.isnan(value)
            for k in ['exact_visible','radius2_visible']:
                value=int(arrays[k][j]);assert value==int(a[k])==int(qp[k][ix])
            source_queries[m['source']]+=1
            if not valid:
                invalid_reasons[reason]+=1;invalid_ids.append(q['sample_id']);st['invalid']+=1
                assert int(a['exact_visible'])==int(a['radius2_visible'])==-1 and int(a['radius2_window_pixels'])==0 and a['radius2_clipped']=='False'
            else:
                assert int(a['exact_visible'])==int(float(a['exact_logit'])>0)and int(a['radius2_visible'])==int(float(a['radius2_logit'])>0)
                window=[(yy,xx)for yy in range(max(0,py-2),min(h,py+3))for xx in range(max(0,px-2),min(w,px+3))]
                assert int(a['radius2_window_pixels'])==len(window)and(a['radius2_clipped']=='True')==(len(window)!=25)
            if native is not None:
                sample_query_n+=1
                if valid:
                    exact=float(native[py,px]);near=max(float(native[yy,xx])for yy,xx in window)
                    assert exact==float(a['exact_logit'])and near==float(a['radius2_logit'])
                    assert int(exact>0)==int(a['exact_visible'])and int(near>0)==int(a['radius2_visible'])
                else:sample_invalid_n+=1
        if native is not None:samples.append(st)
        if(ti+1)%300==0:print('INDEPENDENT_MASKS_VERIFIED',ti+1,'seconds',time.monotonic()-start,flush=True)
    assert sorted(queried_indices)==list(range(8590))and len(invalid_ids)==6 and sample_invalid_n==6
    assert dict(source_queries)==s['source_query_counts']and dict(source_targets)==s['source_target_counts']
    assert mask_bytes==s['output_mask_bytes']
    assert int(qp['native_valid'].sum())==8584 and (~qp['native_valid']).sum()==6
    for k in ['exact_logit','radius2_logit']:assert np.isfinite(qp[k][qp['native_valid']]).all()and np.isnan(qp[k][~qp['native_valid']]).all()
    result=dict(PASS=True,schema='V68_INDEPENDENT_SAVED_EXTRACTION_OUTPUT_AUDIT_V1',extraction_seal=ref(sp),protocol=ref(pp),code=ref(Path(__file__)),
        all_seal_bindings_rehashed=binding_results,runtime_binding_files_rehashed=len(runtime['bindings']),entry_authorizing_evidence_rehashed=len(entry['authorizing_evidence']),
        output_artifact_hashes_verified=len(s['outputs']),native_mask_archive_hashes_and_headers_verified=1091,mask_bytes=mask_bytes,
        targets=1091,queries=8590,native_valid=8584,native_invalid=6,source_query_counts=dict(source_queries),source_target_counts=dict(source_targets),
        all1091mask_metadata_and8590CSV_NPZ_manifest_mapping_PASS=True,
        selected_max_quality_all1091=True,exact_max_quality_ties=len(ties),tie_examples=ties[:12],tie_break='firstindex of all exactmax quality ties',
        fixed_native_fullmap_replay=dict(targets=len(samples),queries=sample_query_n,invalid_queries=sample_invalid_n,exact_or_radius2_logit_mismatches=0,strict_greater_zero_decision_mismatches=0,samples=samples),
        invalid_ids=invalid_ids,invalid_reason_counts=dict(invalid_reasons),invalid_semantics='All6 retain NaN rawlogits/-1visible/U, not a negative prediction.',
        saved_query_exact_zero_logits=int((qp['exact_logit'][qp['native_valid']]==0).sum()),saved_query_radius2_zero_logits=int((qp['radius2_logit'][qp['native_valid']]==0).sum()),
        model_immutability=dict(checkpoint_file_stream_SHA_verified=s['bindings']['checkpoint']['sha256'],
            recorded_initial_state_SHA=s['model_state_sha256'],recorded_final_state_SHA=s['final_model_state_sha256'],equal=True,
            sealed_extractor_state_digest_and_before_after_assertion_reviewed=True,
            limitation='Checkpoint bytes were rehashed without deserialization. In-memory extraction state no longer exists; before/afterstate equality is verified from its immutable recordedseal plus the sealedguardcode, not a newly reconstructed state or newforward.'),
        limitations=['Only the chosen full-native map is retained. All quality/selected-index records and sealedselect_mask code agree with maxquality/firsttie; alternative fullmap tensors cannot be replayed without a forbidden newforward.',
          'All1091archive hashes/headers and per-mask metadata were verified. Fullmap finite-value/scalar sampling was independently recomputed only for the fixedsubset above.',
          'This validates source extraction mechanics, not segmentation accuracy, targetownership GT, external/self cause, or full semanticvisibility success.'],
        RGB_GT_current_feature_arrays_or_model_weights_deserialized=0,new_forward_training_GPU_or_PnP=0,existing_artifacts_mutated=False,seconds=time.monotonic()-start)
    write(report_path,result);print('INDEPENDENT_EXTRACTION_PASS',sha(report_path),'sampletargets',len(samples),'samplequeries',sample_query_n,'ties',len(ties),'seconds',result['seconds'],flush=True)

if __name__=='__main__':main()
