from pathlib import Path
from collections import Counter
import ast,hashlib,json,datetime,sys
import numpy as np
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_flat_pallet_cue_bridge_20261023_v92')
O=R/'independent_pre_rgb_review'
FIXED_FAILED_MANIFEST=O/'provenance/FAILED_STATIC_DESCRIPTION_STRING_V1/FAILED_ATTEMPT_MANIFEST.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def check(b): assert sha(b['path'])==b['sha256'],b['path']
def pure(path,name,extra=None):
 t=ast.parse(Path(path).read_text()); ns=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name==name];assert len(ns)==1
 env={'np':np};env.update(extra or {});exec(compile(ast.Module(body=ns,type_ignores=[]),str(path),'exec'),env);return env[name]
p=read(R/'RGB96_PRODUCER_PROTOCOL_BEFORE_PIXELS.json');policy=read(p['policy']['path']);code=Path(p['code']['path']);source=code.read_text();tree=ast.parse(source)
for k in ['policy','code','generated','crop_code','native_grid_code','input_manifest']:check(p[k])
assert p['policy']['sha256']=='d94f9d4281c6b0e4c817920fbad8097e4a70c390a763e6c32b72c4b330126a75'
assert p['code']['sha256']=='133f894a2e587b1eabf70e5074f1b823f72a110a188bea6b7172f91d8b5713fb'
assert sha(R/'RGB96_PRODUCER_PROTOCOL_BEFORE_PIXELS.json')=='cab7d8fa656baa9af7d6bd1022d6018c8d8bd7d00f7ddb61f56f96c4ef7032c5'
assert read(p['generated']['path'])['PASS']
for b in policy['bindings'].values():check(b)
inputs=lines(policy['bindings']['saved_input_metadata']['path']);props=lines(policy['bindings']['saved_proposals']['path']);manifest=lines(p['input_manifest']['path'])
assert len(inputs)==len(props)==len(manifest)==20 and len({r['id'] for r in inputs})==20
meta_counts=Counter(); nqueries=0; ncenter=0; no_detect=[]; max_qerr=0.;max_Lerr=0.
for i,(r,pr,m) in enumerate(zip(inputs,props,manifest)):
 assert pr['input_id']==r['id']==m['input_id'] and pr['image_sha256']==r['image_sha256']==m['image_sha256']
 for k in ('image','RGB_pixels_sha256','decoded_RGB_input','raw_hw','source_subset','officialpath_lineage'):assert m[k]==r[k],(i,k)
 assert m['frame_index']==i and m['old_initializer_group_state']=='UNASSIGNED_GROUP_HOLD' and m['native8_semantic_mapping']=='UNKNOWN' and m['GT_label_created']is False
 assert len(r['raw_hw'])==2 and all(isinstance(x,int)and x>0 for x in r['raw_hw'])
 assert all(len(r[k])==64 and set(r[k])<=set('0123456789abcdef') for k in ('image_sha256','RGB_pixels_sha256'))
 split='VALIDATION'if int(r['source_subset']) in (1,4)else'TRAIN';assert m['split']==split;meta_counts[split]+=1
 c=pr['candidates'];assert len(c)==m['candidate_count']
 if not c:
  assert pr['selected_index'] is None and pr['no_detection']is True
  assert all(m[k]is None for k in ('selected_index','selected_score','selected_predictor_box_xyxy','query8_xy','L'))
  no_detect.append(i);continue
 scores=np.asarray([c0['score'] for c0 in c],dtype=np.float64);assert np.isfinite(scores).all()
 # Python index of the maximum is deliberately independent of NumPy argmax.
 idx=list(scores).index(max(scores));assert idx==pr['selected_index']==m['selected_index'] and pr['no_detection']is False
 q9=np.asarray(c[idx]['keypoints_xy'],np.float64);assert q9.shape==(9,2)and np.isfinite(q9).all();q=q9[:8]
 assert np.array_equal(np.asarray(m['query8_xy'],np.float64),q)
 assert m['selected_predictor_box_xyxy']==c[idx]['box_xyxy']and m['selected_score']==c[idx]['score']
 keep=[v for v in q if np.isfinite(v).all() and not(v[0]==-1 and v[1]==-1)]
 L=16. if len(keep)<2 else min(max(.25*max(max(v[a] for v in keep)-min(v[a] for v in keep) for a in (0,1)),16.),max(r['raw_hw']))
 assert float(m['L'])==L
 h,w=r['raw_hw'];rn=np.floor(q+.5);cont=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h);nn=(rn[:,0]>=0)&(rn[:,0]<w)&(rn[:,1]>=0)&(rn[:,1]<h)
 ncenter+=int((cont&nn).sum());nqueries+=8
assert meta_counts=={'TRAIN':12,'VALIDATION':8} and len(no_detect)==2 and nqueries==144 and ncenter==122
# Read only function ASTs. This does not import old cache modules or open source arrays.
crop=pure(p['crop_code']['path'],'crop');make=pure(code,'make_frame_arrays')
def scalar_pixel(rgb,q,L,iy,ix):
 h,w=rgb.shape[:2];x=float(q[0])+((ix+.5)/96-.5)*L;y=float(q[1])+((iy+.5)/96-.5)*L
 valid=(-.5<=x<w-.5)and(-.5<=y<h-.5)
 if not valid:return np.zeros(3,np.uint8),False
 x0=int(np.floor(x));y0=int(np.floor(y));dx=x-x0;dy=y-y0;acc=[0.,0.,0.]
 for ax,ay,weight in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):
  xx=x0+ax;yy=y0+ay
  if 0<=xx<w and 0<=yy<h:
   for ch in range(3):acc[ch]+=float(rgb[yy,xx,ch])*weight
 return np.asarray([int(np.floor(min(255.,max(0.,v))+.5)) for v in acc],np.uint8),True
rng=np.random.default_rng(20261023);generated_pixels=0;generated_queries=0
for hw,L in (((7,11),16.),((31,47),31.5)):
 rgb=rng.integers(0,256,(*hw,3),dtype=np.uint8);h,w=hw
 q=np.asarray([[0,0],[w-1,h-1],[-.5,2],[w-.5,2],[-1,-1],[w+2,h+2],[3.125,4.25],[w-.50001,h-.50001]],np.float64)
 arr=make(rgb,{'raw_hw':list(hw),'selected_index':0,'query8_xy':q.tolist(),'L':L},crop)
 assert np.array_equal(arr['query_xy'],q)and np.all(arr['L']==L)and arr['query_exists'].all()
 rn=np.floor(q+.5);ct=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h);nv=(rn[:,0]>=0)&(rn[:,0]<w)&(rn[:,1]>=0)&(rn[:,1]<h)
 assert np.array_equal(arr['query_continuous_valid'],ct)and np.array_equal(arr['query_nearest_valid'],nv)and np.array_equal(arr['query_center_valid'],ct&nv)
 for k in range(8):
  for iy in range(96):
   for ix in range(96):
    px,v=scalar_pixel(rgb,q[k],L,iy,ix)
    assert v==arr['crop_valid96'][k,0,iy,ix]and np.array_equal(px,arr['rgb96'][k,:,iy,ix]),(hw,k,iy,ix)
    generated_pixels+=1
  generated_queries+=1
 assert np.array_equal(arr['crop_anyvalid'],arr['crop_valid96'].reshape(8,-1).any(1))
 assert not arr['rgb96'].transpose(0,2,3,1)[~arr['crop_valid96'][:,0]].any()
 missing=make(rgb,{'raw_hw':list(hw),'selected_index':None},crop)
 assert np.isnan(missing['query_xy']).all()and np.isnan(missing['L']).all()
 for k in ('rgb96','crop_valid96','query_exists','query_continuous_valid','query_nearest_valid','query_center_valid','crop_anyvalid'):assert not missing[k].any()
# Independent test tie-first and no threshold with pure validation function.
const={}
for node in tree.body:
 if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id.endswith('_KEYS'):const[node.targets[0].id]=ast.literal_eval(node.value)
validate=pure(code,'validate_selected',const)
r={'id':'GENERATED','image':'DO_NOT_OPEN','image_sha256':'a'*64,'RGB_pixels_sha256':'b'*64,'raw_hw':[7,11],'source_subset':1,'officialpath_lineage':'generated','decoded_RGB_input':{'path':'DO_NOT_OPEN','sha256':'c'*64}}
def cand(v,idx):return {'candidate_index':idx,'score':v,'box_xyxy':[-9,-4,20,30],'keypoints_xy':np.arange(18).reshape(9,2).tolist(),'keypoints_conf':[.001]*9}
pr={'input_id':'GENERATED','image_sha256':'a'*64,'candidates':[cand(.0001,0),cand(.0001,1)],'selected_index':0,'no_detection':False,'input_shape':[544,640],'canvas_shape':[680,840],'added_border':100,'forward_hooks':1}
s=validate(r,pr);assert s['selected_index']==0 and s['predictor_box_xyxy'][0]==-9 and np.array_equal(s['query8_xy'],np.arange(18).reshape(9,2)[:8])
try:validate(r,dict(pr,selected_index=1))
except AssertionError:pass
else:raise AssertionError('alternate same-score candidate accepted')
assert validate(r,dict(pr,candidates=[],selected_index=None,no_detection=True))is None
# Static path review: only run contains actual source pixel np.load/JPEG sha/frame rendering.
funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
run=funcs['run'];first=run.body[0];assert isinstance(first,ast.Assign)and isinstance(first.value,ast.Call)and isinstance(first.value.func,ast.Name)and first.value.func.id=='authorize'
np_load=[]
for name,node in funcs.items():
 for n in ast.walk(node):
  if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='np'and n.func.attr=='load':np_load.append(name)
assert np_load==['run']
imports=[ast.unparse(n)for n in ast.walk(tree)if isinstance(n,(ast.Import,ast.ImportFrom))]
assert not any(any(x in s for x in ('torch','ultralytics','cv2','transformers'))for s in imports)
assert "out.mkdir(exist_ok=False)"in source and "allow_pickle=False"in source
assert "semantic_state='UNLABELED_UNKNOWN'"in source and "native8_semantic_mapping='UNKNOWN'"in source
# Description strings are not file accesses. Operational file-read paths were reviewed explicitly.
assert {n.func.value.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load' and isinstance(n.func.value,ast.Name)}=={'np'}
assert 'rgb=np.load(frame['decoded_RGB_input']['path'],allow_pickle=False)' in source
assert 'sha(frame['image'])==frame['image_sha256']' in source
# Explicit reviews of root gate code, based on operational source lines and template.
auth=ast.get_source_segment(source,funcs['authorize']);assert "e['approved']is True"in auth
for k in ('model_forward_allowed','training_allowed','current_allowed','PnP_allowed'):assert k in auth
assert "read(e['independent_review']['path'])['PASS']"in auth
assert "frames==lines(p['input_manifest']['path'])"in auth
review={'schema':'V92_INDEPENDENT_PRE_RGB_CODE_METADATA_GENERATED_REVIEW_V1','completed':True,'PASS':True,'blockers':[],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'bindings':{'scope':binding(O/'SCOPE_BEFORE_METADATA_REPLAY.json'),'static_guard_repair_addendum':binding(O/'STATIC_GUARD_V2_REPAIR_ADDENDUM.json'),'reviewer_code':binding(__file__),'policy':p['policy'],'producer_protocol':binding(R/'RGB96_PRODUCER_PROTOCOL_BEFORE_PIXELS.json'),'producer_code':p['code'],'producer_generated':p['generated'],'input_manifest':p['input_manifest'],'entry_template':binding(R/'RGB96_ROOT_ENTRYLOCK_TEMPLATE.json'),'crop_code':p['crop_code'],'native_grid_code':p['native_grid_code']},
 'metadata_replay':{'frames':20,'slots':160,'selected_frames':18,'predicted_query_slots':nqueries,'no_detection_frames':2,'no_detection_slots_U':16,'no_detection_frame_indices':no_detect,'factual_query_center_valid':ncenter,'split_counts':dict(meta_counts),'all_original_order_id_image_SHA_raw_HW_canonical_RGB_pixel_SHA_metadata_exact':True,'first_argmax_tie_first_exact':True,'selected_q9_first8_unchanged':True,'L_independent_scalar_span_clip_exact':True,'predictor_box_unchanged_no_GT_choice':True,'old_group_HOLD_preserved':True,'native8_semantic_mapping':'UNKNOWN'},
 'generated_independent_replay':{'RGB96_pixels_allchannels_exact':generated_pixels,'queries_full96x96':generated_queries,'source':'generated random uint8 only','boundary_cases':['halfopen(-.5,W-.5)','q_continuous0..W vs nearest floor(q+.5) separated','OOB q unchanged','(-1,-1) remains supplied/OOB query, no invented semantic negative','NaN q/L for no detection','bilinear zero exterior floor(clip+.5) uint8'],'invalid_crop_RGB_zero':True,'all96_valid_masks_exact':True,'first_argmax_lowconfidence_tie_and_alternate_rejection':True,'centroid_slot8_excluded':True},
 'code_review':{'entry_approved_hash_bound_PASS_before_source_pixel_access':True,'source_pixels_only_in_run_after_authorize':True,'canonical_RGB_NPY_SHA_and_pixelSHA_nativeHW_uint8_guard_present':True,'original_JPEG_bytes_SHA_only_no_redecode':True,'crop_AST_exact_V74_reuse':True,'L_exact_V60_function_reuse':True,'GT_bbox_mask_depth_pose_cue_labels_not_read':True,'no_P3_model_weights_head_forward_training_current_PnP':True,'fresh_output_directory_no_silent_restart':True,'budget_bytes':p['output_budget_bytes'],'all20_full_and8_crops_rendered_no_scorefilter':True,'no_detection_U_and_OOB_U_not_D_E_S':True},
 'failed_attempt_preserved':binding(FIXED_FAILED_MANIFEST),'한국어_판정':'사전 코드·메타데이터·생성 산술 검토 PASS. 원20장 순서·예측 firstargmax·q8/L·split과 미검출16U 보존을 확인했습니다. 실제 source 픽셀 및 실행 결과의 검증은 아직 수행하지 않았습니다.',
 'limitations':['실제 JPEG/NPY 픽셀, 저장 crop/valid 배열, source pixelSHA는 이번 리뷰에서 읽거나 경험 검증하지 않음; 실제 root 실행 후 별도 saved-output QA 필요.','생성 검사 및 코드 검토를 실제 source crop 일치 검증으로 확대 해석하지 않음.','TRAIN2/3/5 및 VAL1/4는 동결된 공식 subset 분할이며 물리 촬영장 독립성 증명 아님.','선택된 predictor instance와 실제 팔레트 correspondence 및 native8 의미는 검증하지 않았음. no_detection/OOB/invalid input은 physical E/S negative가 아님.','실제 source run은 별도 root approved entry가 필요하며 이 리뷰는 모델실행/학습/평가/current read를 허용하지 않음.'],
 'actual_source_RGB_JPEG_NPY_pixels_read':0,'source_P3_or_inputtensor_array_read':0,'GT_bbox_or_labels_read':0,'current_read':0,'model_or_weights_parameter_read':0,'forward':0,'training':0,'PnP':0,'model_training_admission':False,'fully_solved':False,'goal_status':'active'}
with (O/'REVIEW.json').open('x')as f:json.dump(review,f,ensure_ascii=False,indent=2);f.write('\n')
seal={'completed':True,'PASS':True,'scope':binding(O/'SCOPE_BEFORE_METADATA_REPLAY.json'),'review':binding(O/'REVIEW.json'),'code':binding(__file__),'generated_producer_proof':p['generated'],'actual_source_pixels_read':0,'forward':0,'training':0,'current':0,'PnP':0}
with(O/'SEAL.json').open('x')as f:json.dump(seal,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'review':binding(O/'REVIEW.json'),'seal':binding(O/'SEAL.json'),'PASS':True,'generated_pixels':generated_pixels,'metadata_frames':20},ensure_ascii=False))
