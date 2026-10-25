"""Independent saved SOURCE packet/metadata check; original payloads stay inert."""
import ast
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import numpy as np

H=Path(__file__).parent
R=H.parent
P=R.parent
ROOT=R/'V233_ROOT_CROP_ADAPTATION'
OUT=ROOT/'crop_samples_v1'
COMPLETE=OUT/'COMPLETE.json'
ENTRY=ROOT/'ACTUAL_CROP_ASSEMBLY_ENTRY.json'
allowed={str(COMPLETE),str(ENTRY)}
access={}

def read(path):
    path=Path(path)
    assert str(path) in allowed and 'v112' not in str(path).casefold()
    cursor=Path('/')
    for part in path.parts[1:]:
        cursor/=part
        assert not stat.S_ISLNK(os.lstat(cursor).st_mode)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb')as f:
        raw=f.read()
    access[str(path)]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}
    return raw

def bound(binding):
    raw=read(binding['path'])
    actual=access[binding['path']]
    assert len(binding['sha256'])==64 and actual['sha256']==binding['sha256']
    if 'size_bytes'in binding:assert actual['size_bytes']==binding['size_bytes']
    return raw

def rows(raw):return [json.loads(line)for line in raw.splitlines()if line]
def sha_array(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

entryraw=read(ENTRY)
assert access[str(ENTRY)]['sha256']=='63c90f5b9a08ee57db2c0b8ea602b603dd48d3a57a0b30afddda105a392b6e5e'
entry=json.loads(entryraw)
complete=json.loads(read(COMPLETE))
assert access[str(COMPLETE)]['sha256']=='45b294ea7e89ff1a9e7b1b4765a24c88b2b0b402a79aaa3e5e77b65f4ac80535'
assert complete['PASS']and complete['completed']and complete['entry']['sha256']==access[str(ENTRY)]['sha256']
assert complete['inputs']==entry['inputs']
assert len(complete['outputs'])==6
allowed.update(b['path']for b in complete['outputs'])
expected_names={'samples.npz','raw_rgb96.npy','raw_valid96.npy','ALL15040_ROWS.jsonl','SAMPLE_DIAGNOSTICS.jsonl','SUMMARY.json'}
assert {Path(b['path']).name for b in complete['outputs']}==expected_names
assert all(Path(b['path']).parent==OUT for b in complete['outputs'])
data={Path(b['path']).name:bound(b)for b in complete['outputs']}
for key in ['syn_metadata','point_teacher','point_contract','condition_contract']:
    allowed.add(entry['inputs'][key]['path'])
allowed.add(complete['code']['path'])
assembler=bound(complete['code']).decode()
pointcode=bound(entry['inputs']['point_contract']).decode()
conditioncode=bound(entry['inputs']['condition_contract']).decode()
# Only explicit pure contract functions run; no producer/trainer import/run.
def pure(code,names):
    ns={'np':np,'math':math}
    nodes=[n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<bound-pure-contract>','exec'),ns)
    return ns
pt=pure(pointcode,{'need','finite_pair','point_loss_target'})['point_loss_target']
contract_context=pure(conditioncode,{'need','context32'})['context32']
meta=rows(bound(entry['inputs']['syn_metadata']))
teachers=rows(bound(entry['inputs']['point_teacher']))
assert len(meta)==len(teachers)==7040
teacher={r['row_id']:r for r in teachers}
assert len(teacher)==7040 and set(teacher)=={r['row_id']for r in meta}
ledger=rows(data['ALL15040_ROWS.jsonl'])
diagnostics=rows(data['SAMPLE_DIAGNOSTICS.jsonl'])
summary=json.loads(data['SUMMARY.json'])
assert len(ledger)==15040 and len({r['row_id']for r in ledger})==15040
assert len(diagnostics)==7617
with np.load(io.BytesIO(data['samples.npz']),allow_pickle=False)as packet:
    arrays={k:packet[k]for k in packet.files}
assert set(arrays)=={'full_view','central_view','fractions','central4','condition','target','family','fit','available','row_id'}
assert np.array_equal(arrays['full_view'],np.arange(7617,dtype=np.int64)*2)
assert np.array_equal(arrays['central_view'],np.arange(7617,dtype=np.int64)*2+1)
types={'full_view':np.dtype('int64'),'central_view':np.dtype('int64'),'fractions':np.dtype('float32'),'central4':np.dtype(bool),'condition':np.dtype('float32'),'target':np.dtype('int8'),'family':np.dtype('int8'),'fit':np.dtype(bool),'available':np.dtype(bool)}
for k,t in types.items():assert arrays[k].dtype==t
assert arrays['row_id'].dtype.kind=='U'
for k,a in arrays.items():
    assert a.shape==((7617,32)if k=='condition'else(7617,2)if k=='fractions'else(7617,))
    if a.dtype.kind=='f':assert np.isfinite(a).all()
rgb=np.load(io.BytesIO(data['raw_rgb96.npy']),allow_pickle=False)
valid=np.load(io.BytesIO(data['raw_valid96.npy']),allow_pickle=False)
assert rgb.shape==(7617,3,96,96)and rgb.dtype==np.uint8
assert valid.shape==(7617,1,96,96)and valid.dtype==bool

def context(m):
    k=m['native_slot'];out=np.zeros(32,np.float32);out[k]=1
    assert m['original_query_available']==m['context_available8'][k]
    if m['original_query_available']:
        q=np.asarray(m['query_xy'],np.float64);scale=np.float64(m['L'])
        for other in range(8):
            if other==k or not m['context_available8'][other]:continue
            out[8+other]=1
            out[16+2*other:18+2*other]=np.clip((np.asarray(m['context_q8'][other],np.float64)-q)/scale,-8,8).astype(np.float32)
    return out

def geom(q,L):
    a=(np.arange(96,dtype=np.float64)+.5)/96-.5
    xx,yy=np.meshgrid(q[0]+a*L,q[1]+a*L)
    return(xx>=-.5)&(xx<639.5)&(yy>=-.5)&(yy<479.5)

sample_index=0
full_counts=collections.Counter();sample_counts=collections.Counter();role_counts=collections.Counter()
original_unavailable=collections.Counter();all_unavailable=collections.Counter();known_unavailable=0
component_roles={};ycb_frame_roles={};source_rgb={}
syn_hwc=0;syn_valid=0;cond_all=0
for index,row in enumerate(ledger):
    target=row['target'];assert target in(-1,0,1)
    if index<7040:
        m=meta[index];t=teacher[m['row_id']]
        assert row['family_code']==0 and row['family']=='SYN_NATIVE_WITNESS0'and row['source_index']==index
        assert row['row_id']==m['row_id']
        for key in ['original_global_index','native_slot','query_xy','L','original_query_available','component_id']:
            assert row[key]==m[key]and t[key]==m[key]
        assert m['component_role']=='TRAIN'and m['nominal_fit_eligible']and m['lineage_bound']
        assert t['source_teacher_key']==[m['source_image_original_global_index']//8,m['native_slot']]
        expected_target,reasons=pt(t)
        assert target==expected_target and row['target_exclusion_reasons']==reasons
        assert row['original_state']==t['point0_combined_state']and row['target_alone_state']==t['point0_target_alone_state']
        assert row['region_state_context_only']==t['region_state_context_only']
        assert row['reference_xy_GT_diagnostic_only']==t['saved_projected_reference_xy']
        assert row['reference_locality_diagnostic_only']==t['reference_locality_for_actual_input']
        q=row['query_xy'];ref=row['reference_xy_GT_diagnostic_only']
        distance=math.dist(q,ref)if q is not None and ref is not None and all(math.isfinite(x)for x in q+ref)else None
        assert distance==row['q_reference_distance_px_diagnostic_only']
        fit=hashlib.sha256(('V233:'+m['component_id']).encode()).digest()[0]%5!=0
        cid=m['component_id'];assert cid not in component_roles or component_roles[cid]==fit;component_roles[cid]=fit
        cond=context(m)
        assert np.array_equal(cond,contract_context(m['query_xy'],m['L'],m['native_slot'],m['original_query_available'],m['context_q8'],m['context_L8'],m['context_available8']))
        assert sha_array(cond)==m['context_FP32_native_mask_xy_sha256'];cond_all+=1
    else:
        i=index-7040;n,k=divmod(i,8)
        assert row['family_code']==1 and row['family']=='YCB_ASSOCIATED_R'and row['source_index']==i
        assert row['input_id']==f'Y109_SOURCE_{n:06d}'and row['row_id']==row['input_id']+':aux'+str(k)
        assert row['native_slot']is None and row['native_mapping']=='UNKNOWN'and row['logical_aux_slot']==k
        scene=int(row['scene_id']);assert scene in(0,1,2,3,4,6,7);fit=scene<5
        frame=(scene,row['source_frame_id']);assert frame not in ycb_frame_roles or ycb_frame_roles[frame]==fit;ycb_frame_roles[frame]=fit
        assert target==({'D':0,'E':1}.get(row['original_state'],-1))
        assert row['reference_xy_GT_diagnostic_only']is None and row['q_reference_distance_px_diagnostic_only']is None
        q=row['query_xy'];v=geom(q,row['L']);near=[math.floor(x+.5)for x in q]
        available_original=0<=q[0]<640 and 0<=q[1]<480 and 0<=near[0]<640 and 0<=near[1]<480 and q!=[-1,-1]
        assert row['original_query_available']==available_original
        assert row['central4']==bool(v[47:49,47:49].all())
        assert row['valid_fractions']==[float(v.mean()),float(v[32:64,32:64].mean())]
        assert row['available']==bool(available_original and row['central4']and v.any())
        cond=np.zeros(32,np.float32)
    role='INTERNAL_SOURCE_FIT'if fit else'INTERNAL_SOURCE_DEV'
    assert row['fit']==fit and row['internal_source_role']==role and row['original_component_role']=='TRAIN'
    full_counts[(row['family'],role,row['original_state'],target)]+=1
    role_counts[(row['family'],role)]+=1
    if not row['original_query_available']:original_unavailable[row['family']]+=1
    if not row['available']:all_unavailable[row['family']]+=1
    if target==-1:
        assert row['sample_index']is None
        continue
    j=sample_index;sample_index+=1
    assert row['sample_index']==j and row==diagnostics[j]
    assert arrays['row_id'][j]==row['row_id']and arrays['target'][j]==target
    assert arrays['family'][j]==row['family_code']and arrays['fit'][j]==fit and arrays['available'][j]==row['available']
    vv=valid[j,0]
    frac=[float(vv.mean()),float(vv[32:64,32:64].mean())]
    assert row['valid_fractions']==frac and arrays['central4'][j]==bool(vv[47:49,47:49].all())==row['central4']
    assert np.array_equal(arrays['fractions'][j],np.asarray(frac,np.float32))
    assert row['available']==bool(row['original_query_available']and row['central4']and vv.any())
    assert np.array_equal(arrays['condition'][j],cond)
    assert sha_array(rgb[j])==row['pixel_sha256']and sha_array(vv)==row['valid_sha256']
    if index<7040:
        assert sha_array(rgb[j].transpose(1,2,0))==t['actual_rgb96_sha256'];syn_hwc+=1
        assert sha_array(vv)==t['actual_valid96_sha256'];syn_valid+=1
    else:
        assert np.array_equal(vv,v)
        source=row['source_RGB_binding'];source_rgb[source['path']]=source
    known_unavailable+=int(not row['available'])
    sample_counts[(row['family_code'],fit,target,row['available'])]+=1
assert sample_index==7617 and syn_hwc==syn_valid==6096 and cond_all==7040
assert {str(k):v for k,v in sorted(full_counts.items())}==summary['full_counts']
assert {str(k):v for k,v in sorted(sample_counts.items())}==summary['sample_counts']
assert summary['known_unavailable_rows_deleted']==0 and known_unavailable==0
assert summary['original_rows15040']==15040 and summary['known_samples7617']==7617
assert sum(row['target']==-1 for row in ledger)==summary['unknown_target944plus6479']==7423
assert len(source_rgb)==len(summary['actual_RGB_reads'])==656
assert {b['path']:b for b in summary['actual_RGB_reads']}==source_rgb
allowed_rgb={b['path']:b for b in entry['ycb_rgb_read_allowlist']}
assert source_rgb==allowed_rgb
assert complete['training_performed']is False
# Relevant byte-axis contract from pinned assembler: input HWC -> output CHW.
assert "x['rgb96'][i].transpose(2,0,1)"in assembler
review={'completed':True,'PASS':True,'scope':'Independent saved crop arrays/metadata only; no original RGB/NPZ/raw GT/weights/model execution.',
 'bindings':list(access.values()),'checks':{'six_outputs_SHA_size_schema':True,'samples7617_and_all15040_retained_exact_order':True,'full2j_central2j_plus1':True,'condition32_all7040_and_samples6096_bitexact_YCB_zero':True,'source_point_target_role_and_immutable_fields':True,'SYN_HWC_originalhash_vs_new_CHW_all6096':True,'sample_CHW_RGB_and_bool_valid_diaghash_all7617':True,'YCB_valid_geometry_all8000':True},
 'known_counts':{str(k):v for k,v in sorted(sample_counts.items())},'all_row_counts':{str(k):v for k,v in sorted(role_counts.items())},
 'all_rows':15040,'physical_known_samples':7617,'unknown_target_rows':7423,'SYN_known_pixel_hash_matches':syn_hwc,'SYN_known_valid_hash_matches':syn_valid,
 'original_unavailable_retained':dict(original_unavailable),'all_unavailable_retained':dict(all_unavailable),'known_unavailable_deleted':0,
 'SYN_components_by_role':dict(collections.Counter(component_roles.values())),'YCB_frames_by_role':dict(collections.Counter(ycb_frame_roles.values())),
 'actual_producer_RGB656_claim_matches_saved_sample_bindings':True,'B_original_RGB_reads':0,
 'limits':['YCB original image bytes were not replayed; pixel SHA checked against new diagnostics, source path/SHA metadata checked against entry.','SYN known6096 new crop pixels additionally match independently inherited teacher HWC RGB/valid hashes after explicit axis conversion; original selected NPZ remained unopened.','SYN unknown944 crop-valid bitmap is not redecoded; all row metadata and original U are retained, with stored validity claims checked internally only.','YCB original row/target population was independently closed in V232 MANIFEST_QA_B; this task uses new ledger and prior result rather than reopening original YCB manifest bodies.','No pixel appearance interpretation, new labels, physical gold, independence or model choice/admission. Cross-family provenance remains UNKNOWN.'],
 'access':{'new_output_files':6,'approved_SYN_metadata_JSONL':2,'pure_CODE_files':3,'original_NPZ_RGB_raw_GT_weight_or_protected_payload_FS':0,'V112_or_forbidden_cache_FS':0,'model_forward_training_or_PnP':0}}
raw=(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
fd=os.open(H/'REVIEW.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
with os.fdopen(fd,'wb')as f:f.write(raw)
print(json.dumps({'completed':True,'PASS':True,'path':str(H/'REVIEW.json'),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),'samples':7617,'allrows':15040,'SYN_hash_matches':syn_hwc,'all_unavailable_retained':dict(all_unavailable)}))
