"""Fixed SOURCE TRAIN interventions. No training and no protected score/image access."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,ast,contextlib,csv,hashlib,io,json,math,os,stat,sys,types,zipfile,traceback
import numpy as np
H=Path(__file__).parent;V=H.parent;P=V.parent;OUT=V/'source_train_modality_diagnosis_v2'
SOURCE=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212'
FIT=SOURCE/'runs/C62_C63_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIXED4000/COMPLETE.json'
FPROTO=SOURCE/'fit_preparation_a/PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json'
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_TRAIN_MODALITY_DIAGNOSIS_SCOPE.json'
CLAR=V/'ROOT_DIAGNOSTIC_CONDITION_AND_TRAIN_ROW_IO_CLARIFICATION.json'
MODES=('FACTUAL_UNCHANGED','CONSTANT_RGB127_KEEP_VALID_XY_CONTEXT','ZERO_OTHER_Q8_XY_ONLY','ZERO_OTHER_Q8_MASK_AND_XY','ZERO_NATIVE_ONEHOT_BRANCH_ONLY','ZERO_ALL_CONDITION_BRANCHES')
ARMS=('C62_DENSE_POINT_RAW_Q8','C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
PAYLOAD={}
def need(v,m):
 if not bool(v):raise AssertionError(m)
def safe(p):
 need('v112'not in str(p).casefold(),'Lexical forbidden name before FS');p=Path(p);need(p.is_absolute()and P in p.parents and '..'not in p.parts,'Exact project path');return p
def opened(p):
 p=safe(p);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for c in p.parts[1:-1]:n=os.open(c,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
  n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file');return os.fdopen(n,'rb')
def raw(p):
 need(Path(p).suffix in('.json','.jsonl','.py')or(Path(p).parent==OUT and Path(p).suffix=='.csv'),'Text metadata/code only')
 with opened(p)as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(raw(p)).hexdigest())
def checked(b):
 z=raw(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'Exact metadata/code binding');return z
def get(b):return json.loads(checked(b))
def lines(b):return[json.loads(x)for x in checked(b).splitlines()if x.strip()]
def write(p,obj):
 safe(p);need(p.parent in(H,OUT),'Own outputs only')
 with p.open('x',encoding='utf8')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def jsonl(p,rows):
 with p.open('x',encoding='utf8')as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def payload_identity(binding):
 need(isinstance(binding,dict)and isinstance(binding.get('path'),str)and isinstance(binding.get('sha256'),str),'Payload path/SHA required')
 safe(binding['path']);need(len(binding['sha256'])==64 and all(c in '0123456789abcdef'for c in binding['sha256']),'Exact SHA256 identity')
 return binding['path'],binding['sha256']
def payload_file(b):
 need(b['path']in PAYLOAD and payload_identity(PAYLOAD[b['path']])==payload_identity(b),'Exact root path/SHA identity before header/hash');h=hashlib.sha256()
 with opened(b['path'])as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 need(h.hexdigest()==b['sha256'],'Exact opaque payload SHA');return opened(b['path'])
def pure_context(binding):
 s=checked(binding).decode();tree=ast.parse(s);names={'require','native_index','bool_flag','q64','build_context_vectors'};ns={'np':np}
 for n in tree.body:
  if isinstance(n,ast.FunctionDef)and n.name in names:exec(compile(ast.Module(body=[n],type_ignores=[]),binding['path'],'exec'),ns)
 need(names<=set(ns),'Exact pure q8 context dependencies');return ns['build_context_vectors']
def module(binding,name):
 z=checked(binding);m=types.ModuleType(name);m.__file__=binding['path'];sys.modules[name]=m;exec(compile(z,binding['path'],'exec'),m.__dict__);return m

def npy_selected(stream,take,expected_shape,expected_dtype):
 """Parse header; discard unselected bytes opaquely. Interpret selected TRAIN rows only."""
 version=np.lib.format.read_magic(stream);need(version in((1,0),(2,0)),'Declared standard numeric NPY')
 shape,fortran,dtype=(np.lib.format.read_array_header_1_0(stream)if version==(1,0)else np.lib.format.read_array_header_2_0(stream))
 need(tuple(shape)==tuple(expected_shape)and (dtype.kind=='U'if expected_dtype is None else np.dtype(dtype)==np.dtype(expected_dtype))and not fortran and not dtype.hasobject,'Exact C-order numeric/Unicode header, no pickle')
 take=np.asarray(take,np.int64);need(take.ndim==1 and len(set(take.tolist()))==len(take)and ((take>=0)&(take<shape[0])).all(),'Unique selected original TRAIN indices')
 order=np.argsort(take);sorted_take=take[order];rowbytes=int(np.prod(shape[1:],dtype=np.int64))*dtype.itemsize;out=np.empty((len(take),*shape[1:]),dtype=dtype);current=0
 def consume(n):
  while n:
   z=stream.read(min(n,1<<20));need(bool(z),'No truncated NPY discard');n-=len(z)
 for position,index in zip(order,sorted_take):
  consume((int(index)-current)*rowbytes);z=stream.read(rowbytes);need(len(z)==rowbytes,'Selected row exact byte count');out[position]=np.frombuffer(z,dtype=dtype).reshape(shape[1:]);current=int(index)+1
 return out
def selected_source(binding,take,all_point_rows):
 with payload_file(binding)as f,zipfile.ZipFile(f)as z:
  out={}
  for key,shape,dtype in [('rgb96',(65536,96,96,3),np.uint8),('valid96',(65536,96,96),bool)]:
   with z.open(key+'.npy')as stream:out[key]=npy_selected(stream,take,shape,dtype)
  point_take=[r['cache_index']for r in all_point_rows]
  for key,shape,dtype in [('row_id',(65536,),None),('original_global_index',(65536,),np.int64),('native_slot',(65536,),np.int8),('query_xy',(65536,2),np.float64),('L',(65536,),np.float64),('original_query_available',(65536,),bool)]:
   with z.open(key+'.npy')as stream:a=npy_selected(stream,point_take,shape,dtype)
   for i,r in enumerate(all_point_rows):need(meta_equal(a[i],r[key],key),'Selected TRAIN-only original source META '+key)
 return out

def meta_equal(stored,original,key):
 if key=='query_xy':return np.array_equal(stored,np.asarray([np.nan,np.nan]if original is None else original,np.float64),equal_nan=True)
 if key=='L':return np.array_equal(stored,np.asarray(np.nan if original is None else original,np.float64),equal_nan=True)
 return bool(stored==original)

def hook_slice(mode):
 return {'ZERO_OTHER_Q8_XY_ONLY':(80,96),'ZERO_OTHER_Q8_MASK_AND_XY':(72,96),'ZERO_NATIVE_ONEHOT_BRANCH_ONLY':(64,72),'ZERO_ALL_CONDITION_BRANCHES':(64,96)}.get(mode)
@contextlib.contextmanager
def condition_intervention(model,mode):
 need(mode in MODES,'Fixed diagnostic mode');sl=hook_slice(mode);handle=None
 if sl is not None:
  def zero(module,args):
   x=args[0];need(x.ndim==2 and x.shape[1]==96,'Exact residual first-linear condition layout');y=x.clone();y[:,sl[0]:sl[1]]=0;need(__import__('torch').equal(x[:,:64],y[:,:64]),'Shared features64 unchanged');return(y,)
  handle=model.spatial_residual[0].register_forward_pre_hook(zero)
 try:yield
 finally:
  if handle is not None:handle.remove()
def input_intervention(x,mode):
 if mode!='CONSTANT_RGB127_KEEP_VALID_XY_CONTEXT':return x
 y=x.clone();y[:,:3]=127./255.;need(__import__('torch').equal(y[:,3:],x[:,3:]),'valid/fixedXY/zeroSAM unchanged');return y

def decoded(logits):return np.where(np.isfinite(logits),logits>=0,-1).astype(np.int8)
def confusion(truth,pred):
 cm=np.zeros((2,3),np.int64)
 for t,p in zip(truth,pred):need(t in(0,1)and p in(-1,0,1),'Known D/E and D/E/U');cm[int(t),int(p)if p>=0 else 2]+=1
 def div(a,b):return float(a/b)if b else None
 return dict(N=len(truth),D=int(cm[0].sum()),E=int(cm[1].sum()),CM_D_E_by_D_E_U=cm.tolist(),D_precision=div(cm[0,0],cm[:,0].sum()),D_recall=div(cm[0,0],cm[0].sum()),E_precision=div(cm[1,1],cm[:,1].sum()),E_recall=div(cm[1,1],cm[1].sum()),U=int(cm[:,2].sum()),U_rate=div(cm[:,2].sum(),cm.sum()))
def deltas(factual,changed,truth):
 a=decoded(factual);b=decoded(changed);valid=np.isfinite(factual)&np.isfinite(changed);d=changed[valid].astype(np.float64)-factual[valid].astype(np.float64);tr=np.zeros((3,3),np.int64)
 for x,y in zip(a,b):tr[x if x>=0 else 2,y if y>=0 else 2]+=1
 return dict(N=len(a),finite_paired=int(valid.sum()),mean_delta=float(d.mean())if len(d)else None,mean_abs_delta=float(np.abs(d).mean())if len(d)else None,median_abs_delta=float(np.median(np.abs(d)))if len(d)else None,p95_abs_delta=float(np.quantile(np.abs(d),.95))if len(d)else None,max_abs_delta=float(np.abs(d).max())if len(d)else None,sign_flips=int((a!=b).sum()),transition_D_E_U=tr.tolist(),factual_correct_changed_wrong=int(((a==truth)&(b!=truth)).sum()),factual_wrong_changed_correct=int(((a!=truth)&(b==truth)).sum()))
def tensor_state_hash(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

def prepare():
 fp=get(bind(FPROTO));fit=get(bind(FIT));teacher=get(fp['teacher_complete']);mapped=lines(teacher['outputs']['ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl']);weak=get(fp['weak_metadata']);aug=get(fp['old_cache_bindings']['augmentation']);rr=lines(fp['context_input_bindings']['SYNTH_AUG65536']);roles=lines(fp['context_input_bindings']['SYNTH_AUG65536_roles']);role={r['row_id']:r for r in roles};byid={r['row_id']:(i,r)for i,r in enumerate(rr)}
 need(len(rr)==65536 and len(byid)==65536,'Original source row identity');scenes=lines(fp['context_input_bindings']['original_scenes']);desired={r['anchor_original_global_index']//8 for r in mapped};scene8={s['scene_index']:s['original_input_metadata8']for s in scenes if s['scene_index']in desired};context=pure_context(fp['source_bindings']['context_adapter']);contract=module(fp['target_contract'],'v214_pure_target_contract')
 point=[]
 for r in mapped:
  i,m=byid[r['row_id']];z=role[r['row_id']];need(z['role']=='TRAIN'and r['component_role']=='TRAIN','Immutable TRAIN before any source pixel interpretation')
  for k in META:need(r[k]==m[k]==z[k],'Exact original input '+k)
  need(m['source_image_original_global_index']==r['source_teacher_key'][0]*8+r['native_slot'],'Own actual image teacher')
  label,why=contract.point_loss_target(r);eight=scene8[m['anchor_original_global_index']//8];need([o['native_slot']for o in eight]==list(range(8)),'Same original anchor8')
  c=context(m['query_xy'],m['L'],m['native_slot'],m['original_query_available'],[o['original_query_xy']for o in eight],[o['original_L']for o in eight],[o['original_input_flags']['query_available']for o in eight])
  point.append(dict(**{k:m[k]for k in META},cache_index=i,component_role='TRAIN',component_id=z['component_id'],base_index=m['base_index'],family=m['family'],variant=m['variant'],source_teacher_key=r['source_teacher_key'],point0_combined_state=r['point0_combined_state'],point0_target_alone_state=r['point0_target_alone_state'],target=label,excluded_reasons=why,factors=z['factors'],other_mask8=c['other_mask8'].tolist(),other_dxdy16=c['other_dxdy16'].tolist()))
 need(len(point)==7040 and Counter(r['target']for r in point)=={0:5853,1:243,-1:944},'Original admitted full7040/6096; unknown retained')
 wr=[]
 for r in weak['rows']:
  need(r['component_role']=='TRAIN'and r['original_query_available']is True,'Weak TRAIN only');eight=weak['frames'][r['frame_q8_key']]['metadata8'];active=eight[r['native_slot']]
  for k in('query_xy','L','original_query_available','native_slot'):need(r[k]==active[k],'Weak original sameframe8')
  c=context(r['query_xy'],r['L'],r['native_slot'],True,[o['query_xy']for o in eight],[o['L']for o in eight],[o['original_query_available']for o in eight]);wr.append(dict(r,target=r['proposed_weak_target'],other_mask8=c['other_mask8'].tolist(),other_dxdy16=c['other_dxdy16'].tolist()))
 need(len(wr)==44 and Counter(r['target']for r in wr)=={0:26,1:18},'Original44weak authority separate')
 selected=[]
 for pop,label,rs in [('POINT',0,point),('POINT',1,point),('WEAK_FACTUAL',0,[r for r in wr if r['variant']=='FACTUAL']),('WEAK_COVER',1,[r for r in wr if r['target']==1])]:
  r=next(r for r in rs if r['target']==label);selected.append(dict(population='POINT'if pop=='POINT'else'WEAK',selection_reason=pop+'_FIRST_ORIGINAL_TARGET_'+str(label),row_id=r['row_id']))
 plan=dict(completed=True,PASS=True,point_rows=point,weak_rows=wr,point_known=6096,point_unknown=944,weak_known=44,selected4_before_pixel_or_score_reads=selected,RGB_valid_or_score_reads=0)
 write(H/'SOURCE_TRAIN_SELECTION_AND_CONTEXT_METADATA.json',plan)
 bindings=[aug['inputs'],*weak['payload_whitelist'],*[fit['arms'][a]['checkpoint']for a in ARMS],*[fit['arms'][a]['outputs'][pop]for a in ARMS for pop in('POINT7040','WEAK44')]];need(len({b['path']for b in bindings})==59,'Exact1 sharedSOURCE input+52weakPNG+2ownedcheckpoint+4TRAINpred packets')
 write(H/'FUTURE_PAYLOAD59_WHITELIST_NOT_OPENED.json',dict(completed=True,payload_whitelist=bindings,SOURCE_raw_input=aug['inputs'],SOURCE_TRAIN_only_predictions={a:{p:fit['arms'][a]['outputs'][p]for p in('POINT7040','WEAK44')}for a in ARMS},source_payload_header_stat_hash_reads=0))
 print(json.dumps(dict(completed=True,PASS=True,point7040=7040,known6096=6096,unknown944=944,weak44=44,selected4=selected)))

def weak_arrays(rows):
 from PIL import Image
 cache={};rgb=[];valid=[]
 for r in rows:
  for key,kind,wanted_sha in [('RGB_binding','RGB',r['rgb96_sha256_expected']),('valid_binding','L',r['valid96_sha256_expected'])]:
   b=r[key]
   if b['path']not in cache:
    with payload_file(b)as f,Image.open(f)as im:
     need(im.format=='PNG'and im.mode==kind and im.size==(96,96),'Exact weak PNG schema')
     a=np.asarray(im).copy()
    if kind=='L':need(np.isin(a,[0,255]).all(),'Original validity');a=a==255
    cache[b['path']]=a
   a=cache[b['path']];need(hashlib.sha256(a.tobytes()).hexdigest()==wanted_sha,'Original weak decoded pixels')
  rgb.append(cache[r['RGB_binding']['path']]);valid.append(cache[r['valid_binding']['path']])
 return dict(rgb96=np.stack(rgb),valid96=np.stack(valid))
def saved_factual(binding,rows):
 with payload_file(binding)as f,np.load(f,allow_pickle=False)as z:
  ids=z['row_id'];need(len(ids)==len(set(ids.tolist())),'Unique saved TRAIN-only row IDs');by={str(r):i for i,r in enumerate(ids)};raw=z['raw_direct_end_external_logit'];pred=z['direct_end_prediction_D0_E1_Uminus1'];meta={k:z[k]for k in META if k!='row_id'}
  out=[]
  for r in rows:
   need(r['row_id']in by,'Saved factual exact row join');i=by[r['row_id']]
   for k in meta:need(meta_equal(meta[k][i],r[k],k),'Saved original q/L/native/U '+k)
   v=float(raw[i]);need(math.isfinite(v)and int(pred[i])==int(v>=0),'Saved finite fixed decoder');out.append(v)
 return np.asarray(out,np.float32)
def verify_replay(saved,actual):
 need(saved.shape==actual.shape and np.isfinite(saved).all()and np.isfinite(actual).all(),'Finite exact factual replay vectors')
 maximum=float(np.max(np.abs(saved.astype(np.float64)-actual.astype(np.float64))))
 need(maximum<=1e-4 and np.array_equal(decoded(saved),decoded(actual)),'Predeclared replay maxabs1e-4 and signs exact; never relax')
 return dict(max_abs_logit_difference=maximum,signs_exact=True,rows=len(saved))
def context_match(left,right):
 return left['query_xy']==right['query_xy']and left['L']==right['L']and left['native_slot']==right['native_slot']and left['other_mask8']==right['other_mask8']and left['other_dxdy16']==right['other_dxdy16']
def pair_plan(populations):
 out=[]
 rows=populations['POINT']['rows'];bygroup=defaultdict(dict)
 for i,r in enumerate(rows):
  key=(r['component_id'],r['base_index'],r['native_slot'],r['family']);need(r['variant']not in bygroup[key],'Unique original point variant group');bygroup[key][r['variant']]=i
 for key,variants in sorted(bygroup.items()):
  for variant in('COVER','NEAR_SHAM','FAR_SHAM'):
   if 'BEFORE'in variants and variant in variants:out.append(dict(population='POINT',group=list(key),left_index=variants['BEFORE'],right_index=variants[variant],left_variant='BEFORE',right_variant=variant))
 wr=populations['WEAK']['rows'];group=defaultdict(dict)
 for i,r in enumerate(wr):
  if r.get('attempt_id')is not None:group[r['attempt_id']][r['variant']]=i
 for key,vs in sorted(group.items()):
  cover=[i for v,i in vs.items()if wr[i]['target']==1];sham=[i for v,i in vs.items()if wr[i]['target']==0]
  need(len(cover)==len(sham)==1,'Original weak COVER/SHAM pair, not variant-name native state')
  out.append(dict(population='WEAK',group=key,left_index=sham[0],right_index=cover[0],left_variant=wr[sham[0]]['variant'],right_variant=wr[cover[0]]['variant']))
 for pair in out:
  p=populations[pair['population']];i,j=pair['left_index'],pair['right_index'];a,b=p['rows'][i],p['rows'][j]
  pair.update(left_row_id=a['row_id'],right_row_id=b['row_id'],left_target=a['target'],right_target=b['target'],same_target=a['target']==b['target'],exact_qL_native_othercontext=context_match(a,b),exact_validity=np.array_equal(p['x']['valid96'][i],p['x']['valid96'][j]),exact_RGB=np.array_equal(p['x']['rgb96'][i],p['x']['rgb96'][j]))
  pair['exact_nonRGB_model_inputs_matched']=pair['exact_qL_native_othercontext']and pair['exact_validity']
 return out

def run(entry):
 e=get(bind(entry));need(e.get('approved_SOURCE_TRAIN_modality_diagnostic_forward')is True and e.get('automatic_retry')is False,'Separate bounded root actual entry')
 p=get(e['protocol']);need(e['code']==bind(__file__)==p['code']and e['protocol']==bind(H/'PROTOCOL_BEFORE_ANY_SOURCE_PAYLOAD_OR_MODEL.json'),'Exact frozen diagnostic code')
 need(e['scope']==p['scope']==bind(SCOPE)and e['clarification']==p['clarification']==bind(CLAR),'Exact prospective input intervention scope');need(get(e['clarification'])['conditions']==list(MODES),'Fixed6 conditions')
 fit=get(e['fit_complete']);need(e['fit_complete']==p['fit_complete']and fit['completed']and fit['PASS']and fit['all2_frozen_before_any_SOURCE_outputs']and fit['steps_each']==4000 and fit['parameters_each']==118359,'Both fixed source finals')
 review=get(e['source_fit_review']);need(review['completed']and review['PASS']and review['fit_complete']==e['fit_complete'],'Independent exact fit PASS')
 barrier=get(fit['all2_freeze']);need(barrier['completed']and barrier['source_predictions_emitted']is False and barrier['checkpoints']=={a:fit['arms'][a]['checkpoint']for a in ARMS},'Both final4000 before source predictions')
 whitelist=get(p['payload_whitelist']);need(e['payload_whitelist']==whitelist['payload_whitelist'],'Exact root payload59');PAYLOAD.update({b['path']:b for b in e['payload_whitelist']});need(len(PAYLOAD)==59,'No inputs/weights outside fixed SOURCE whitelist')
 need(e['allow_only_selected_TRAIN_NPY_rows_decode']is True and e['no_HOLD_or_protected_image_score_decode']is True,'TRAIN selective decoding policy')
 plan=get(p['selection']);need(e['output_directory']==str(OUT)and not OUT.exists(),'Fresh exclusive output');OUT.mkdir()
 point=[r for r in plan['point_rows']if r['target']in(0,1)];weak=plan['weak_rows'];need(len(point)==6096 and len(weak)==44 and all(r['component_role']=='TRAIN'and r['original_query_available']for r in point+weak),'Only fixed known available TRAIN forwards;7040 unknown separately retained')
 populations={'POINT':dict(rows=point,x=selected_source(whitelist['SOURCE_raw_input'],[r['cache_index']for r in point],plan['point_rows'])),'WEAK':dict(rows=weak,x=weak_arrays(weak))}
 # Input-pair exactness is measured before any checkpoint or saved prediction decoding.
 pairs=pair_plan(populations);write(OUT/'PAIR_INPUT_MATCHES_BEFORE_SCORE_READS.json',dict(completed=True,pairs=pairs,counts={pop:dict(all_pairs=sum(r['population']==pop for r in pairs),exact_nonRGB_matched=sum(r['population']==pop and r['exact_nonRGB_model_inputs_matched']for r in pairs))for pop in populations},checkpoint_or_prediction_bodies_read_so_far=False))
 from PIL import Image
 image_manifest=[]
 for n,s in enumerate(plan['selected4_before_pixel_or_score_reads']):
  pop=populations[s['population']];i=next(i for i,r in enumerate(pop['rows'])if r['row_id']==s['row_id'])
  for mode,a in [('FACTUAL_UNCHANGED',pop['x']['rgb96'][i]),('CONSTANT_RGB127_KEEP_VALID_XY_CONTEXT',np.full_like(pop['x']['rgb96'][i],127))]:
   path=OUT/('FIXED_%02d_'%n+('FACTUAL'if mode==MODES[0]else'RGB127')+'_RAW96.png');Image.fromarray(a).save(path);
   with opened(path)as encoded:image_sha=hashlib.sha256(encoded.read()).hexdigest()
   image_manifest.append(dict(path=str(path),sha256=image_sha,population=s['population'],row_id=s['row_id'],mode=mode,display_only_not_new_training_input=True))
 write(OUT/'FIXED8_DISPLAY_IMAGE_MANIFEST.json',dict(completed=True,images=image_manifest,selection=p['selection'],selection_before_model_scores=True))
 import torch
 need(sys.executable==p['python_executable']and torch.__version__=='2.1.1+cu118'and np.__version__=='1.26.4'and torch.cuda.is_available(),'Pinned runtime');need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Deterministic CUBLAS');torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 fp=get(p['fit_protocol']);head=module(fp['source_bindings']['head'],'v214_head');candidate=module(fp['source_bindings']['candidate'],'v214_candidate');spatial=module(fp['source_bindings']['spatial_head'],'v214_spatial');cls=spatial.make_spatial_q8_head_class(head,candidate);outputs={};replay={};state_proofs={}
 for arm in ARMS:
  f=fit['arms'][arm]
  with payload_file(f['checkpoint'])as file:cp=torch.load(file,map_location='cpu',weights_only=True)
  need(cp['update']==4000 and cp['seed']==1810 and cp['arm']==arm and cp['use_sam']is False and cp['spatial_q8']is True and cp['weak_bridge']==(arm==ARMS[1])and cp['weak_loss_weight']==(.25 if arm==ARMS[1]else 0)and cp['protocol']==fit['protocol'],'Exact fixed source checkpoint metadata')
  model=cls(spatial_q8=True).float().cuda().eval().requires_grad_(False);model.load_state_dict(cp['state_dict'],strict=True);del cp;need(sum(v.numel()for v in model.parameters())==118359 and len(model.state_dict())==36,'Same C61 structure');before=tensor_state_hash(model);outputs[arm]={};replay[arm]={}
  for pop,info in populations.items():
   rs=info['rows'];x=info['x'];outputs[arm][pop]={}
   for mode in MODES:
    result=[]
    with torch.inference_mode(),condition_intervention(model,mode):
     for start in range(0,len(rs),96):
      part=rs[start:start+96];take=slice(start,start+len(part));inp=head.assemble_local_input(torch.as_tensor(x['rgb96'][take],device='cuda'),torch.as_tensor(x['valid96'][take],device='cuda'),use_sam=False);inp=input_intervention(inp,mode)
      native=torch.as_tensor([r['native_slot']for r in part],dtype=torch.int64,device='cuda');mask=torch.as_tensor([r['other_mask8']for r in part],dtype=torch.float32,device='cuda');xy=torch.as_tensor([r['other_dxdy16']for r in part],dtype=torch.float32,device='cuda');result.extend(model(inp,native,mask,xy)['direct_end_external_logit'].cpu().numpy().tolist())
    outputs[arm][pop][mode]=np.asarray(result,np.float32);need(tensor_state_hash(model)==before,'Frozen state unchanged per condition')
    if mode==MODES[0]:
     saved=saved_factual(whitelist['SOURCE_TRAIN_only_predictions'][arm]['POINT7040'if pop=='POINT'else'WEAK44'],rs);actual=outputs[arm][pop][MODES[0]];replay[arm][pop]=verify_replay(saved,actual)
  need(tensor_state_hash(model)==before and not model.training and all(not v.requires_grad for v in model.parameters()),'No state/gradient/fit change');state_proofs[arm]=before;del model
 summary=dict(completed=True,PASS=True,conditions=list(MODES),models={},factual_replay=replay,unchanged_model_state_sha=state_proofs,all7040_point_metadata_retained=True,point_known6096=True,point_unknown944_not_forwarded=True,point_originalU64_not_forwarded=True,weak44_separate=True,OOD_not_generalization_or_protected_failure_cause=True,selected_model=None,adopted=False,fully_solved=False)
 scalar_rows=[];metric_rows=[];native_rows=[]
 for arm in ARMS:
  summary['models'][arm]={}
  for pop,info in populations.items():
   rs=info['rows'];truth=np.asarray([r['target']for r in rs],np.int8);modes=outputs[arm][pop];summary['models'][arm][pop]={}
   for mode,logit in modes.items():
    d=decoded(logit);by_native={str(k):confusion(truth[np.array([r['native_slot']==k for r in rs])],d[np.array([r['native_slot']==k for r in rs])])for k in range(8)}
    z=confusion(truth,d);z.update(deltas(modes[MODES[0]],logit,truth),by_native=by_native);summary['models'][arm][pop][mode]=z;metric_rows.append(dict(arm=arm,population=pop,condition=mode,**{k:json.dumps(v)if isinstance(v,(list,dict))else v for k,v in z.items()}))
    for native,metrics in by_native.items():native_rows.append(dict(arm=arm,population=pop,condition=mode,native_slot=int(native),**{k:json.dumps(v)if isinstance(v,(list,dict))else v for k,v in metrics.items()}))
    for i,r in enumerate(rs):scalar_rows.append(dict(arm=arm,population=pop,condition=mode,row_id=r['row_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],query_xy=r['query_xy'],L=r['L'],original_query_available=r['original_query_available'],component_role='TRAIN',known_target_D0_E1=int(truth[i]),raw_logit=float(logit[i])if np.isfinite(logit[i])else None,prediction_D0_E1_Uminus1=int(d[i]),factual_raw_logit=float(modes[MODES[0]][i])if np.isfinite(modes[MODES[0]][i])else None,delta_from_factual=float(logit[i]-modes[MODES[0]][i])if np.isfinite(logit[i])and np.isfinite(modes[MODES[0]][i])else None))
 pair_scores=[]
 for pair in pairs:
  for arm in ARMS:
   for mode in MODES:
    vals=outputs[arm][pair['population']][mode];a=float(vals[pair['left_index']]);b=float(vals[pair['right_index']]);pair_scores.append(dict(pair,arm=arm,condition=mode,left_logit=a if math.isfinite(a)else None,right_logit=b if math.isfinite(b)else None,right_minus_left=b-a if math.isfinite(a)and math.isfinite(b)else None,source_RGB_dependence_evidence=mode==MODES[0]and pair['exact_nonRGB_model_inputs_matched']and not pair['exact_RGB']and math.isfinite(a)and math.isfinite(b)and abs(b-a)>1e-4))
 jsonl(OUT/'ALL73680_SOURCE_TRAIN_CONDITION_SCALARS.jsonl',scalar_rows);jsonl(OUT/'PAIRED_SOURCE_VARIANT_CONDITION_CONTRASTS.jsonl',pair_scores);jsonl(OUT/'ALL7040_ORIGINAL_POINT_TARGET_STATUS_METADATA.jsonl',plan['point_rows'])
 with(OUT/'FULL_AND_NATIVE8_METRICS.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(metric_rows[0]));w.writeheader();w.writerows(metric_rows)
 with(OUT/'NATIVE8_METRICS.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(native_rows[0]));w.writeheader();w.writerows(native_rows)
 write(OUT/'SUMMARY.json',summary)
 write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=e['code'],protocol=e['protocol'],entry=bind(entry),fit_complete=e['fit_complete'],source_fit_review=e['source_fit_review'],selection=p['selection'],payload_whitelist=p['payload_whitelist'],factual_replay=replay,scalar_rows=len(scalar_rows),pair_contrast_rows=len(pair_scores),NPY_interpreted_SOURCE_HOLDOUT_or_protected_rows=0,model_optimizer_training_calls=0,new_checkpoint_or_input_cache_saved=False,outputs={n:bind(OUT/n)for n in('SUMMARY.json','ALL73680_SOURCE_TRAIN_CONDITION_SCALARS.jsonl','PAIRED_SOURCE_VARIANT_CONDITION_CONTRASTS.jsonl','ALL7040_ORIGINAL_POINT_TARGET_STATUS_METADATA.jsonl','PAIR_INPUT_MATCHES_BEFORE_SCORE_READS.json','FIXED8_DISPLAY_IMAGE_MANIFEST.json','FULL_AND_NATIVE8_METRICS.csv','NATIVE8_METRICS.csv')}))
 print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'),scalar_rows=len(scalar_rows))))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-metadata',action='store_true');parser.add_argument('--entry');a=parser.parse_args()
 try:prepare()if a.prepare_metadata else run(a.entry)
 except BaseException:
  if OUT.is_dir()and not(OUT/'FAILURE.json').exists():write(OUT/'FAILURE.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False))
  raise
