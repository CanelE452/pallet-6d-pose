"""V88 bounded independent code + generated math + small GT schema audit.
Never imports trainer/model or opens images/features/checkpoints. No model forward.
"""
from pathlib import Path
import ast,hashlib,json,datetime,zipfile,collections,math
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent

def bind(p):
 p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']
P=R/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json';T=R/'train_external_once.py';M=R/'external_ray_head.py'
expected={'protocol':'b8fc78174e168a7f5abdde50bc195af5bb1d25c685317d02bd471cbaeb28f72e','trainer':'6b3a49913d1e189cff43715f20a1e07d4aebfab0be7f4cab9624f7b69413580d','model':'648be1dffa2a6a024688ed1d69e16728b690dfe37443e44ada5514f94379ad2b'}
for key,path in [('protocol',P),('trainer',T),('model',M)]:assert bind(path)['sha256']==expected[key]
p=read(P);scope=read(O/'REVIEW_SCOPE_BEFORE_GT_SCHEMA.json');assert scope['protocol']==bind(P)
generated=read(R/'GENERATED_TESTS.json');assert generated['PASS'] and generated['trainer']==bind(T) and generated['protocol']==bind(P) and generated['model_code']==bind(M)
check(p['bindings']['base_architecture']);check(p['bindings']['source_input_bindings']);check(p['bindings']['teacher_protocol']);check(p['bindings']['teacher_seal']);check(p['bindings']['independent_teacher_audit'])
basepath=Path(p['bindings']['base_architecture']['path']);base=ast.parse(basepath.read_text());tree=ast.parse(T.read_text());head=ast.parse(M.read_text())
# Only pure helpers are compiled; top-level imports, protocol(), loading and run() are not executed.
allowed=('admission','objective','sample','metric','confusion','report')
ns={'np':np,'torch':torch,'F':F}
funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in allowed]
assert len(funcs)==len(allowed)
exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),str(T)+'::AST_PURE_HELPERS_ONLY','exec'),ns)
torch.set_num_threads(1)
checks={};details={}
# Independent parameter arithmetic, no module construction/weights/forward.
counts=[65*24*9+24,4*8*9+8,8*12*9+12,12*16*9+16,4*8*25+8,8*16*9+16,16*24*9+24,24*24*9+24,42*16+16,16*16+16,85*32+32,32*24*9+24,90*32+32,32+1]
assert sum(counts)==p['parameter_count']==41237;checks['independent_parameter_arithmetic']=True
fields=['p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42']
externalclass=next(n for n in head.body if isinstance(n,ast.ClassDef) and n.name=='ExternalRayHead');ef=next(n for n in externalclass.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
baseclass=next(n for n in base.body if isinstance(n,ast.ClassDef) and n.name=='NativeContextSupportHead');bf=next(n for n in baseclass.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
assert [x.arg for x in ef.args.args][1:]==[x.arg for x in bf.args.args][1:]==fields
loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_source')
gpuassign=next(n for n in loader.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gpu' for t in n.targets))
assert isinstance(gpuassign.value,ast.Call) and [k.arg for k in gpuassign.value.keywords]==fields
fwd=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='forward');assert 'frame = ix // 8' in ast.unparse(fwd)
checks['six_input_whitelist_order_and_frame_major_target_index_static']=True
# Stable BCE and gradients vs an independent scalar formula, no model forward.
rng=np.random.default_rng(822026);loss_cases=0
for n in (1,7,96,257):
 for _ in range(20):
  x=rng.normal(0,20,n);y=rng.choice([-1,0,1],n);mask=rng.random(n)>.4;known=mask&np.isin(y,[0,1])
  z=torch.tensor(x,dtype=torch.float64,requires_grad=True);v=ns['objective'](z,torch.tensor(y),torch.tensor(mask));ref=float(np.mean(np.logaddexp(0,x[known])-y[known]*x[known])) if known.any() else 0.
  assert math.isclose(float(v.detach()),ref,rel_tol=1e-13,abs_tol=1e-13)
  v.backward();grad=np.zeros(n);grad[known]=(1/(1+np.exp(-x[known]))-y[known])/known.sum() if known.any() else 0.
  assert np.allclose(z.grad.numpy(),grad,rtol=1e-12,atol=1e-14);assert np.count_nonzero(z.grad.numpy()[~known])==0;loss_cases+=1
z=torch.tensor([-1000.,1000.,-1000.,1000.],dtype=torch.float64,requires_grad=True);v=ns['objective'](z,torch.tensor([0,1,1,0]),torch.ones(4,dtype=torch.bool));assert torch.isfinite(v) and float(v.detach())==500.;v.backward();assert torch.isfinite(z.grad).all();loss_cases+=1
z=torch.tensor([3.,-4.],requires_grad=True);v=ns['objective'](z,torch.tensor([-1,-1]),torch.ones(2,dtype=torch.bool));v.backward();assert float(v)==0 and torch.equal(z.grad,torch.zeros_like(z));loss_cases+=1
checks['stable_BCE_unknown_invalid_miss_zero_gradient_and_graphzero']=True;details['generated_loss_cases']=loss_cases
# Exhaustive Boolean factual admission combinations + half-open reference geometry.
admission_cases=0
for ycode in (-1,0,1):
 for bits in range(16):
  for dx,dy in [(-.5,0),(.5,0),(0,-.5),(0,.5),(0,0),(.5000000001,0),(-.5000000001,0),(np.nan,0),(0,np.inf)]:
   q=np.zeros((1,8,2));r=q.copy();r[:]=[dx,dy];qvalid=np.full((1,8),bool(bits&1));roi=np.full((1,8),bool(bits&2));local=np.full((1,8),bool(bits&4));target=np.array([bool(bits&8)])
   admit,square,offset,available=ns['admission'](np.full((1,8),ycode),qvalid,roi,local,target,r,q,np.ones(1))
   expected_square=bool(np.isfinite([dx,dy]).all() and -.5<=dx<.5 and -.5<=dy<.5)
   assert np.all(square==expected_square);assert np.all(admit==(ycode in (0,1) and bits==15 and expected_square));admission_cases+=1
checks['half_open_ref_inside_only_loss_no_U_training']=True;details['generated_admission_cases']=admission_cases
# Fixed sampler equality and source-independent deterministic RNG; no source tensors read.
pools={0:np.arange(17),1:np.arange(17,20)};r1=np.random.default_rng(20261023);r2=np.random.default_rng(20261023)
for _ in range(2000):
 ix=ns['sample'](pools,r1);assert np.array_equal(ix,ns['sample'](pools,r2));assert ix.shape==(96,) and np.sum(ix<17)==48 and np.sum(ix>=17)==48
checks['fixed_2000_updates_sampler_48_clear_48_external_with_replacement']=True
# Admission misses never trigger prediction U; factual invalidity does. All-known and conditional differ.
y=np.array([0,1,0,1,0,1,-1,-1]);qvalid=np.array([1,1,1,1,0,1,1,1],bool);available=np.array([1,1,1,1,1,0,1,1],bool)
meta=dict(y=y,qvalid=qvalid,train=np.array([1,1,1,1,0,0,0,0],bool),val=np.array([0,0,0,0,1,1,1,1],bool),primary=np.array([1,1,0,0,0,0,0,0],bool),square=np.array([1,1,0,0,1,1,1,1],bool),self_visible=np.zeros(8,int))
pred=dict(logits=np.array([0.,0.,-1.,1.,-1.,1.,1000.,-1000.]),available=available,nll=np.ones(8))
known=np.isin(y,[0,1]);cm=ns['confusion'](pred,known,meta)
assert cm['confusion_rows_GT_CLEAR_EXTERNAL_cols_pred_CLEAR_EXTERNAL_U']==[[1,1,1],[0,2,1]]
old=meta['square'].copy();meta['square']=~old;assert ns['confusion'](pred,known,meta)==cm;meta['square']=old
rep=ns['report'](pred,meta);assert rep['train']['all_known_fixed_gate_metrics']['N']==4 and rep['train']['conditional_admitted_metrics']['N']==2 and rep['val']['teacher_U']==2
checks['all_known_gate_misses_retained_factual_U_only_teacherU_not_accuracy']=True
meta['valpools']={0:np.array([2,4]),1:np.array([3,5])};nll=np.array([.1,.2,.3,.6,.9,1.4,.5,.7]);score,per=ns['metric']({'nll':nll},np.arange(8),meta)
assert math.isclose(per['0'],.6) and math.isclose(per['1'],1.0) and math.isclose(score,.8)
checks['selector_equal_class_mean_BCE_FP64_reduction']=True
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run');runtext=ast.unparse(run)
assert 'range(1, 2001)' in runtext and 'step % 100 == 0' in runtext and 'saved = score < best' in runtext
assert "assert not RUN.exists()" in runtext and "'FIRST_UPDATE.json'" in runtext and "'MODEL_FREEZE.json'" in runtext
assert "selection_eligible=False" in runtext and "current_evaluation_admission=False" in runtext
assert 'load_state_dict(torch.load(RUN /' in runtext and 'BEST_EXTERNAL_HEAD.pt' in runtext
checks['one_fit_restart_guard_update0_ineligible_earliest_strict_min_snapshot_freeze_static']=True
initial=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='initialize_from_proxy');it=ast.unparse(initial)
assert "key.startswith('reference_localizer.')" in it and "key.startswith('classifier.2.')" in it
assert 'nn.init.zeros_(model.classifier[2].weight)' in it and 'nn.init.zeros_(model.classifier[2].bias)' in it
assert "['classifier.2.bias', 'classifier.2.weight', 'reference_localizer.bias', 'reference_localizer.weight']" in runtext
checks['old_DES_output_localizer_excluded_new_binary_zero_init_static']=True
# All source pack inspection below is JSON/JSONL schema/lineage only; no chunk arrays opened.
source=read(p['bindings']['source_input_bindings']['path']);pack=next(x for x in source['packs'] if x['source']=='YCB_PUBLIC_REAL')
for k in ('manifest','seal','lineage'):check(pack[k])
manifest=read(pack['manifest']['path']);assert manifest['completed'] and manifest['frame_count']==4200 and manifest['query_count']==33600
last=0
for c in manifest['chunks']:
 assert c['start']==last and c['end']-c['start']==c['instances'];last=c['end'];n=c['instances'];sc=c['schema']
 for name,shape,dtype in [('p3_roi',[n,8,65,9,9],'float16'),('rgb96',[n,8,3,96,96],'uint8'),('crop_valid96',[n,8,1,96,96],'bool'),('target_rgb128',[n,3,128,128],'uint8'),('target_valid128',[n,1,128,128],'bool'),('context42',[n,8,42],'float32'),('query8_xy',[n,8,2],'float64'),('query_center_valid',[n,8],'bool'),('ROI_anyvalid',[n,8],'bool'),('L',[n],'float64')]:assert sc[name]['shape']==shape and sc[name]['dtype']==dtype
assert last==4200
lines=[json.loads(x) for x in Path(pack['lineage']['path']).read_text().splitlines()];assert len(lines)==4200 and len(set(x['id'] for x in lines))==4200
assert all(x['source']=='YCB_PUBLIC_REAL' and x['input_index']==i for i,x in enumerate(lines))
checks['immutable_public_pack_metadata_all4200_identity_order_schema_no_chunk_reads']=True
# Only the small, explicitly admitted teacher NPZ is opened, with scalar labels/identity fields read.
check(p['bindings']['teacher_GT']);gtpath=Path(p['bindings']['teacher_GT']['path']);schema={}
with zipfile.ZipFile(gtpath) as zf:
 for name in zf.namelist():
  with zf.open(name) as f:
   version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version)
  schema[name.removesuffix('.npy')]={'shape':list(shape),'dtype':str(dtype),'fortran':fortran}
with np.load(gtpath,allow_pickle=False) as z:
 gt={key:z[key] for key in ('id','split','label8','reason8','human_semantic_gold8')}
assert gt['label8'].shape==(4200,8) and str(gt['label8'].dtype)=='int8' and np.isin(gt['label8'],[-1,0,1]).all()
assert np.all(gt['human_semantic_gold8']==-1)
assert all(x['id']==str(gt['id'][i]) and x['split']==str(gt['split'][i]) for i,x in enumerate(lines))
assert np.all((gt['label8']==1)==(gt['reason8']=='EXTERNAL_OTHER_FIRST_HIT_PROXY'))
assert np.all((gt['label8']==0)==(gt['reason8']=='CLEAR_TO_VIRTUAL_ENDPOINT_PROXY'))
counts={}
for split,frames,exp in [('train',3400,{-1:14809,0:9508,1:2883}),('val',800,{-1:3324,0:2172,1:904})]:
 mask=gt['split']==split;assert int(mask.sum())==frames
 c={code:int(np.sum(gt['label8'][mask]==code)) for code in (-1,0,1)};assert c==exp
 counts[split]={'frames':frames,'slots':frames*8,'label_counts':{str(k):v for k,v in c.items()},'reason_counts':dict(collections.Counter(gt['reason8'][mask].reshape(-1).tolist()))}
for key,shape in [('reference_xy8',[4200,8,2]),('convex_self_visible_proxy8_separate',[4200,8]),('label8',[4200,8]),('human_semantic_gold8',[4200,8])]:assert schema[key]['shape']==shape
checks['reviewed_teacher_codes_reasons_U_gold0_and_actual_metadata_counts_match']=True
metadata={'completed':True,'PASS':True,'teacher':p['bindings']['teacher_GT'],'teacher_schema':schema,'read_array_bodies_only':list(gt),'counts':counts,'public_pack':pack,'chunks_N':len(manifest['chunks']),'chunk_arrays_opened':0,'all4200_GT_input_ids_and_splits_exact':True,'input_metadata_query_counts':manifest['query_counts'],'source_RGB_masks_depth_weights_features_current_reads':0,'new_model_forward_fit_PnP':0}
write(O/'TEACHER_SCHEMA_COUNTS_AND_PACK_METADATA_AUDIT.json',metadata)
write(O/'GENERATED_INDEPENDENT_LOSS_ADMISSION_SAMPLER_METRIC_TESTS.json',dict(completed=True,PASS=True,checks=checks,details=details,code=bind(__file__),trainer_AST_functions_only=list(allowed),new_model_forward_calls=0,optimizer_updates=0,source_tensor_reads=0,current_PnP=0))
notes=[
 '출력1은 EXTERNAL_OTHER_FIRST_HIT_PROXY,0은 CLEAR_TO_VIRTUAL_ENDPOINT_PROXY이며 clear는 physicalcueD가 아니다. U/ownfront/self는 외부 negative로 변환되지 않는다. Human semantic gold는 전부-1이다.',
 '학습은 YCB만이며 V75 synthetic checkpoint는 고정 shared feature/first classifier linear 초기화다. 이전 DES final/localizer는 제외하고 새 binary final을 zero-init한다. Nine-ray SYNTH targets는 새 single-ray supervision에 혼합하지 않는다.',
 'rinsideq는 loss/selector cohort에만 사용된다. All-known VAL gate와 실제 factualU는 qcenter+모델 input_available만 사용하므로 misses도 제외되지 않는다. Generated counterexample에서 square만 바꾸어도 all-known confusion은 동일했다.',
 '정확한 6개 input tensor와 기존 frame-major8/q/L/P3/RGB recipe를 재사용한다. Q는 V76 public emulation이지 실제 YCB detector native8 성공을 뜻하지 않는다. Public sourceVAL는 반복 개발자료이고 blind/final benchmark가 아니다.',
 '새 raw score는 balanced training 결과로서 calibrated real deployment likelihood가 아니다. 통과해도 별도 동결 public physicalcue 평가 준비만 가능하고 current223 평가를 자동 승인하지 않는다.',
 'Saved NLL_FP32는 teacherU에도 clamp(0,1)한 diagnostic 값이 저장되므로 UNKNOWN 행의 NLL은 실제 label NLL로 해석하면 안 된다. 본 selector/metrics는 known/admitted만 사용하여 그 값이 선택에 들어가지 않는다.',
 '기존 GENERATED_TESTS의 source_arrays_checkpoint_reads=0은 deserialization/array loading0으로 해석해야 한다. generated()->protocol()은 고정 NPZ/checkpoint의 SHA byte-read를 수행한다. 이 검토는 protocol()/model_module()/generated()/run()을 호출하지 않았고 checkpoint bytes도 읽지 않았다.',
 'Runtime seed/CUBLAS/CUDNN/TF32 설정은 코드에 명시되어 있지만 torch.use_deterministic_algorithms(True)를 강제하지 않는다. 프로토콜이 exact GPU backward bit claim을 False로 명시한 범위에서 검토했다.',
 '이 PASS는 코드/생성 math/작은 teacher metadata에 한정한다. 원 feature chunk/가중치나 실제 source forward를 검증했다고 주장하지 않으며 실제 source GPU preflight는 root entry후 별도로 수행되어야 한다.'
]
report=dict(completed=True,PASS=True,blockers=[],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=bind(P),trainer=bind(T),model=bind(M),base_architecture=p['bindings']['base_architecture'],scope=bind(O/'REVIEW_SCOPE_BEFORE_GT_SCHEMA.json'),generated=bind(O/'GENERATED_INDEPENDENT_LOSS_ADMISSION_SAMPLER_METRIC_TESTS.json'),existing_generated_proof=bind(R/'GENERATED_TESTS.json'),teacher_metadata=bind(O/'TEACHER_SCHEMA_COUNTS_AND_PACK_METADATA_AUDIT.json'),review_code=bind(__file__),checks=checks,notes=notes,source_RGB_masks_depth_modelweights_feature_chunks_current_reads=0,model_construction_forward_calls=0,optimizer_updates=0,PnP=0,scientific_status='SOURCE_ONLY_EXTERNAL_RAY_PROXY_COMPONENT_PREFIT_REVIEW; no semanticgold/currenteval admission')
write(O/'INDEPENDENT_PREFIT_REVIEW.json',report)
write(O/'REVIEW_SEAL.json',dict(completed=True,PASS=True,review=bind(O/'INDEPENDENT_PREFIT_REVIEW.json'),files=[bind(x) for x in sorted(O.iterdir()) if x.is_file()],root_code_protocol_unchanged=True,new_model_forward_fit_current_PnP=0))
print(json.dumps({'review':bind(O/'INDEPENDENT_PREFIT_REVIEW.json'),'seal':bind(O/'REVIEW_SEAL.json'),'PASS':True,'blockers':[],'generated_loss_cases':loss_cases,'generated_admission_cases':admission_cases,'known_counts':counts},ensure_ascii=False))
