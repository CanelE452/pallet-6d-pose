"""Independent V152 static R0 review and completed RGB-only transport replay.
No torch, model, checkpoint, pose/K/mask/depth or current/private payload access.
"""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
import cv2
from PIL import Image
V=Path(__file__).resolve().parents[1]; H=Path(__file__).resolve().parent; P=V.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def ck(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

assert Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve()
assert cv2.__version__=='4.9.0' and np.__version__=='1.26.4';cv2.setNumThreads(1)
prep=V/'initializer_preparation'; protocol=read(prep/'PROTOCOL_BEFORE_SOURCE_FORWARD.json')
for k in ('code','policy','selection','generated','inputs','reference_code','reference_protocol','backbone_code','native_grid','original_run','original_protocol','helper_code','canonical_transport_seal','canonical_manifest','effective_run','run_diff'):ck(protocol[k])
assert protocol['frames']==10 and protocol['native_slots']==80 and not protocol['actual_forward']
assert protocol['checkpoint']['sha256']=='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7' # metadata only; no weight hash/read
seal=read(ck(protocol['canonical_transport_seal']));assert seal['completed'] and seal['PASS'] and seal['frames']==10 and seal['lossless']
for k in ('code','policy','protocol','selection','source_download_seal','source_inputs','manifest','model_inputs','diagnostic','decoder_trace'):ck(seal[k])
assert seal['policy']==protocol['policy'] and seal['selection']==protocol['selection']
manifest=read(ck(seal['manifest'])); rows=manifest['rows'];assert rows==lines(ck(protocol['inputs']))==lines(ck(seal['model_inputs']))
selection=read(ck(seal['selection']))['selected_frames'];assert len(rows)==len(selection)==10
rawrows=lines(ck(seal['source_inputs'])); diagnostic=read(ck(seal['diagnostic']));assert diagnostic['frames']==10
keys={'frame_index','id','source','source_id','split','image','image_sha256','decoded_RGB_sha256','raw_hw','decoder','canonical_RGB_NPY'}
checks=[]
for i,(r,s,original) in enumerate(zip(rows,selection,rawrows)):
 assert set(r)==keys and r['frame_index']==i and r['source']=='MR6D' and r['split']=='SOURCE_DEVELOPMENT'
 assert s['split']=='val' and s['scene'] in ('000000','000002','000005','000006')
 assert r['source_id']==f"val/{s['scene']}:{s['frame_key']}" and r['id']==f"V152_MR6D_val_{s['scene']}_{s['frame_filename_stem']}"
 assert r['image'].endswith(s['source_files']['RGB']) and r['image_sha256']==sha(r['image'])
 assert {k:r[k] for k in original}==original
 assert r['decoder']=='PINNED_CV2_IMREAD_COLOR_BGR_TO_RGB'
 bgr=cv2.imread(r['image'],cv2.IMREAD_COLOR);rgb=np.ascontiguousarray(bgr[...,::-1]);npy=np.load(ck(r['canonical_RGB_NPY']),allow_pickle=False)
 pil=np.array(Image.open(r['image']).convert('RGB'),np.uint8,copy=True)
 assert npy.dtype==rgb.dtype==np.uint8 and list(rgb.shape)==r['raw_hw']+[3] and rgb.tobytes()==npy.tobytes()
 assert sha(r['canonical_RGB_NPY']['path'])==r['canonical_RGB_NPY']['sha256'] and hashlib.sha256(rgb.tobytes()).hexdigest()==r['decoded_RGB_sha256']
 assert np.ascontiguousarray(npy[...,::-1]).tobytes()==bgr.tobytes()
 eq=rgb.tobytes()==pil.tobytes();d=diagnostic['rows'][i];assert d['frame_index']==i and d['id']==r['id'] and d['exact_same_RGB']==eq
 assert eq and d['maximum_absolute_channel_difference']==0 and d['changed_pixels']==0 and d['changed_channel_values']==0
 checks.append(dict(frame_index=i,id=r['id'],source_id=r['source_id'],raw_JPEG_SHA=True,original_cv2_RGB_matches_canonical_NPY=True,canonical_RGB_to_BGR_roundtrip=True,pinned_PIL_diagnostic_same=True,raw_hw=r['raw_hw']))
assert len({r['source_id'] for r in rows})==10 and not any('/000001:' in r['source_id'] for r in rows)

# Independent structural normalization of the saved old/new loops. Numerical
# constants in these exact positions are inventory sizes, never network scales.
a=ast.parse(Path(protocol['original_run']['path']).read_text());c=ast.parse(Path(protocol['effective_run']['path']).read_text())
oldnodes=list(ast.walk(a));newnodes=list(ast.walk(c));assert len(oldnodes)==len(newnodes)
changes=[]
for x,y in zip(oldnodes,newnodes):
 assert type(x) is type(y)
 for field in x._fields:
  av,bv=getattr(x,field),getattr(y,field)
  if isinstance(av,ast.AST) or isinstance(av,list):continue
  if av!=bv:
   if isinstance(x,ast.Constant) and type(av) is int:assert (av,bv) in ((64,10),(512,80),(66,12))
   elif isinstance(x,ast.Constant) and isinstance(av,str):assert bv==av.replace('ALL64','ALL10').replace('QUERY512','QUERY80').replace('v146_exact_R0_factory','v152_exact_R0_factory')
   elif isinstance(x,ast.keyword):assert (av,bv) in (('all64_and512_retained','all10_and80_retained'),('all64_frames512_native_slots_preserved','all10_frames80_native_slots_preserved'))
   else:raise AssertionError((type(x).__name__,field,av,bv))
   changes.append(dict(node=type(x).__name__,field=field,before=av,after=bv))
assert sum(z['before']==64 for z in changes)==6 and sum(z['before']==512 for z in changes)==4 and sum(z['before']==66 for z in changes)==1

# Execute only extracted NumPy query helper and native-grid math on invented
# inputs. The original and new loops call this exact unchanged helper.
base=ast.parse(Path(protocol['helper_code']['path']).read_text()); roi=ast.parse(Path(protocol['native_grid']['path']).read_text())
ng=[n for n in roi.body if isinstance(n,ast.FunctionDef) and n.name=='native_grid'];ns={'np':np}
exec(compile(ast.Module(body=ng,type_ignores=[]),'native_grid_extracted','exec'),ns)
ns['pure_native']=lambda:ns['native_grid']
qn=[n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='query_arrays'];exec(compile(ast.Module(body=qn,type_ignores=[]),'query_helper_extracted','exec'),ns)
q=np.array([[40.+i*13,50.+i*5] for i in range(9)],np.float64)
def case(x):return dict(candidates=[dict(score=.7,keypoints_xy=x),dict(score=.7,keypoints_xy=x+50)],selected_index=0)
out=ns['query_arrays'](case(q),[720,1280]);assert out['query_xy'].tobytes()==q[:8].tobytes()
expected=max(16,.25*max(np.ptp(q[:8],axis=0)));assert (out['L']==expected).all()
q2=q.copy();q2[8]=[1e8,-1e8];other=ns['query_arrays'](case(q2),[720,1280]);assert all(np.array_equal(out[k],other[k],equal_nan=True) for k in out)
no=ns['query_arrays'](dict(candidates=[],selected_index=None),[720,1280]);assert no['query_xy'].shape==(8,2) and np.isnan(no['query_xy']).all() and np.isnan(no['L']).all() and not no['native_grid_valid'].any()
for xy in ((-.25,10.),(-.5,0.),(1279.49,719.49),(1279.5,719.5),(1280.,720.)):
 t=q.copy();t[0]=xy;o=ns['query_arrays'](case(t),[720,1280]);assert np.array_equal(o['nearest_valid'],o['RGBcentre_valid'])
 assert o['continuous_valid'][0]==(0<=xy[0]<1280 and 0<=xy[1]<720)
 assert np.array_equal(o['query_xy'][0],xy)

result=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(prep/'PROTOCOL_BEFORE_SOURCE_FORWARD.json'),initializer_code=protocol['code'],effective_run=protocol['effective_run'],canonical_transport_seal=protocol['canonical_transport_seal'],frames=10,slots_preserved_by_reviewed_operator=80,
 original_decoder_callsite_cv2_BGR_verified=True,canonical_JPEG_NPY_roundtrip_all10_exact=True,pinned_PIL_diagnostic_all10_exact=True,frame_order_and_official_source_ids_exact=True,
 input_whitelist_has_no_GT=True,source_RGB_checks=checks,AST_inventory_only_changes=changes,original_factory_native_grid_firstargmax_raw9_native8_unchanged=True,
 invented_boundary_noquery_centroid_tie_math_checks=True,frozen_FP32_TF32false_threads1_and_repeatfirst_state_guards_reviewed=True,actual_R0_forward_authorization_required=True,
 candidate_selection_and_GT_instance_association_distinct=True,actual_model_or_checkpoint_weight_reads_hashes_or_calls=0,GT_pose_K_masks_depth_or_labels_read=0,current_private_V112_access=0,
 limitations=['This is static initializer review plus completed CPU decoder transport replay, not an actual learned R0 numerical execution audit.', 'Pinned PIL and cv2 happen to match all ten JPEGs; this is not a universal JPEG decoder equivalence claim.', 'No source semantic D/E or native-role/entity association is established by transport or model-code review.'])
write(H/'REVIEW.json',result);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),frames=10,model_calls=0)))
