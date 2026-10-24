"""B independent CPU invented checks. Exact code only; no production data/imported trainer."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,stat,types
import numpy as np
import torch

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_native_conditioned_grouped_source_training_20261024_v199'
A=V/'fit_preparation_a';H=Path(__file__).absolute().parent
OLD=P/'query_visibility_source_group_split_direct_end_training_20261024_v194'
ALLOWED={
 A/'native_conditioned_head.py':'b32395816d60b698d9b445f497c3609b317d399879731cc1256354a1c90dc83d',
 A/'train_native_conditioned_grouped_source.py':'c9334d39425b73b0da8fde0a5765e8189f44380ee554672765f5a437cd7c31e8',
 P/'query_visibility_local_support_raw_head_20261024_v181/head_preparation/local_support_head.py':'14a40c4dd14d416c1648e62b408fd7acdf74f06d4113c69785bd737b45aa848b',
 P/'query_visibility_sampled_domain_direct_end_20261024_v189/direct_end_training_preparation_c/synthetic_direct_end_candidate.py':'1668550225434f609ed70d2d62093e35bea6bd8471b9c1f5692f7cfb0b358341',
 OLD/'source_component_split_preparation_b/trainer_component_mask_adapter.py':'8ac3763c8d7a376d1bd826f08516c292c3f274afb51827d530a9b003128102bb',
 OLD/'grouped_fit_preparation_b/train_grouped_direct_end.py':'6cc8c09e74cf1c72b8209709e78d7d37d67fa2b4c8b0c5245fda74fe7e695ec8'}

def read(p):
 assert p in ALLOWED and 'v112' not in str(p).lower() and '..' not in p.parts
 cur=Path(p.anchor)
 for x in p.parts[1:]:
  cur/=x;assert not stat.S_ISLNK(os.lstat(cur).st_mode)
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==ALLOWED[p];return b

def module(p,name):
 read(p);spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.detach().contiguous().numpy().tobytes()==b.detach().contiguous().numpy().tobytes()
def need(v,msg):
 if not bool(v):raise AssertionError(msg)
def rejects(fn):
 try:fn()
 except (ValueError,AssertionError):return True
 return False

def run():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 assert not torch.cuda.is_initialized()
 head=module(next(p for p in ALLOWED if p.name=='local_support_head.py'),'B_original_head')
 candidate=module(next(p for p in ALLOWED if p.name=='synthetic_direct_end_candidate.py'),'B_direct_candidate')
 native=module(A/'native_conditioned_head.py','B_native_head')
 adapter=module(OLD/'source_component_split_preparation_b/trainer_component_mask_adapter.py','B_mask')
 source=read(A/'train_native_conditioned_grouped_source.py').decode();tree=ast.parse(source);funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 extract=('native_indices','gather_native','decode','train_native_class_priors','train_raw_identity_collisions')
 ns=dict(np=np,hashlib=hashlib,c=types.SimpleNamespace(need=need))
 exec(compile(ast.Module(body=[funcs[k] for k in extract],type_ignores=[]),'<exact_AST_pure_helpers>','exec'),ns)
 checks={};details={}
 cls=native.make_native_conditioned_head_class(head,candidate);oldcls=candidate.make_candidate_head_class(head)
 torch.manual_seed(1810);old=oldcls().float()
 torch.manual_seed(1810);zero=cls(native_condition=False).float()
 torch.manual_seed(1810);one=cls(native_condition=True).float()
 assert sum(p.numel() for p in old.parameters())==115222
 assert sum(p.numel() for p in zero.parameters())==sum(p.numel() for p in one.parameters())==116407
 assert all(eq(t,zero.state_dict()[k]) for k,t in old.state_dict().items())
 assert all(eq(t,one.state_dict()[k]) for k,t in zero.state_dict().items())
 checks['parameter_and_original_and_paired_initial_state_bytes']=True
 rgb=torch.arange(4*96*96*3,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(4,96,96,3)
 valid=torch.ones((4,96,96),dtype=torch.bool);valid[0,0,:]=False
 x=head.assemble_local_input(rgb,valid,use_sam=False);ids=torch.tensor([0,3,6,7],dtype=torch.int64);saved=x.clone()
 assert not x[:,6:].any()
 with torch.no_grad():a=old(x);b=zero(x,ids);c=one(x,ids)
 assert all(eq(a[k],b[k]) and eq(a[k],c[k]) for k in a) and eq(x,saved)
 assert (zero.native_residual[2].weight==0).all() and (zero.native_residual[2].bias==0).all()
 checks['same_RAW_tensor_and_initial_all_outputs_base_equivalent']=True
 for bad in [ids.float(),ids.to(torch.int8),ids.bool(),ids[:,None],ids[:3],torch.tensor([-1,0,1,2]),torch.tensor([0,1,2,8])]:assert rejects(lambda bad=bad:one(x,bad))
 assert rejects(lambda:cls(native_condition=1))
 checks['strict_native_int64_shape_range_and_bool_flag']=True
 target=torch.tensor([0.,1.,0.,1.])
 for model in (zero,one):
  model.zero_grad(set_to_none=True);torch.nn.functional.binary_cross_entropy_with_logits(model(x,ids)['direct_end_external_logit'],target).backward()
 for k,p in zero.named_parameters():
  if k.startswith('native_residual'):continue
  q=dict(one.named_parameters())[k]
  assert (p.grad is None and q.grad is None) or eq(p.grad,q.grad)
 checks['initial_base_gradient_same_under_same_direct32_analogue']=True
 with torch.no_grad():
  for model in (zero,one):
   for p in model.native_residual.parameters():p.zero_()
   model.native_residual[0].weight[0,0]=1.;model.native_residual[0].weight[0,64]=1.;model.native_residual[2].weight[0,0]=1.
 z=torch.zeros((2,64),dtype=torch.float32);z[1,0]=1.;n0=torch.zeros(2,dtype=torch.int64);n1=torch.ones(2,dtype=torch.int64)
 d=one.residual_from_features(z,n0)-one.residual_from_features(z,n1)
 assert abs(float(d[0]-d[1]))>.01
 assert eq(zero.residual_from_features(z,n0),zero.residual_from_features(z,n1))
 for model in (zero,one):model.zero_grad(set_to_none=True);model.residual_from_features(z,n0).sum().backward()
 assert (zero.native_residual[0].weight.grad[:,64:]==0).all() and (one.native_residual[0].weight.grad[:,64:]!=0).any()
 with torch.no_grad():a=one(x,ids);b=one(x,ids.flip(0))
 assert all(eq(a[k],b[k]) for k in ('support_logits','owner_logits'))
 checks['nonlinear_interaction_zero_control_and_condition_gradient_boundary']=True
 # Scalar expected ordering independently enumerates the task layout.
 names=('visible','foreign','synthetic_owner','public_owner','CF_owner');sizes=(32,32,48,48,32);domains=('SYNTH_AUG65536',)*3+('PUBLIC_TRAIN1480','CF3816')
 oldcache={d:{'x':{'native_slot':((np.arange(17)+j)%8).astype(np.int8)}}for j,d in enumerate(dict.fromkeys(domains))}
 direct={'x':{'native_slot':((np.arange(19)+5)%8).astype(np.int8)}}
 draw={name:((np.arange(size)*3+i)%17).astype(np.int64)for i,(name,size)in enumerate(zip(names,sizes))};di=((np.arange(32)*2)%19).astype(np.int64)
 got=ns['gather_native'](oldcache,direct,draw,di)
 expected=[]
 for name,domain in zip(names,domains):
  for idx in draw[name]:expected.append(int(oldcache[domain]['x']['native_slot'][int(idx)]))
 for idx in di:expected.append(int(direct['x']['native_slot'][int(idx)]))
 assert got.dtype==np.int64 and got.tolist()==expected and len(got)==224
 assert rejects(lambda:ns['native_indices'](np.array([1.],np.float64))) and rejects(lambda:ns['native_indices'](np.array([8],np.int8)))
 checks['exact224_native_alignment_and_no_numeric_coercion']=True
 # These are invented booleans, not any real split/HOLDOUT row.
 av=np.array([True,True,True,False,True,True]);y=np.array([0,1,-1,-1,1,0],np.int8);mask=np.array([True,True,True,True,False,False])
 keepy=y.copy();keepav=av.copy();pool=adapter.masked_direct_pools(av,y,mask)
 assert pool['feasible'] and [a.tolist()for a in pool['pools']]==[[0],[1]]
 assert np.array_equal(y,keepy) and np.array_equal(av,keepav)
 assert rejects(lambda:adapter.assert_draws_train({},np.array([4]),{'DIRECT_CONTROL':mask}))
 checks['TRAIN_mask_U_UNKNOWN_preserve_and_forbidden_draw_reject']=True
 tiny=np.nextafter(np.float32(0),np.float32(-np.inf));raw=np.array([tiny,0,1],np.float32)
 decoded=ns['decode'](np.zeros((3,2),np.float32),np.zeros((3,3),np.float32),raw,np.array([True,True,False]))
 assert decoded[2].tolist()==[0,1,-1]
 checks['raw_ge0_boundary_and_original_U_precedence']=True
 nslt=np.repeat(np.array([0,2,5,7],np.int8),3);yt=np.tile(np.array([0,1,0],np.int8),4);tm=np.arange(12)<9
 cache={'x':{'native_slot':nslt,'original_query_available':np.ones(12,bool)},'y':{'direct_end_external_target':yt},'metadata':[{'domain':'REAL'if i<3 else'SYNTH'}for i in range(12)]}
 prior=ns['train_native_class_priors'](cache,tm)
 assert prior['TRAIN_known_class_totals']=={'D':6,'E':3}
 for row in prior['rows']:
  if row['direct_D']+row['direct_E']:assert row['raw_known_E_fraction']==1/3 and row['balanced_draw_expected_E_given_domain_native']==.5
  if row['native_slot'] in (6,7):assert row['direct_D']==row['direct_E']==0
 checks['TRAIN_native_prior_balanced_half_and_no_6_7_imputation']=True
 rgb2=np.zeros((6,96,96,3),np.uint8);val2=np.ones((6,96,96),bool);val2[2,0,0]=False
 coll={'x':dict(rgb96=rgb2,valid96=val2,native_slot=np.array([0,1,0,0,6,7],np.int8),original_query_available=np.array([True]*5+[False]),row_id=np.array(['B%d'%i for i in range(6)])), 'y':{'direct_end_external_target':np.array([0,1,1,1,-1,-1],np.int8)}}
 cmask=np.array([True,True,True,False,True,True]);answer=ns['train_raw_identity_collisions'](coll,cmask)
 assert answer['known_available_TRAIN_rows']==3 and answer['excluded_TRAIN_inputU']==1 and answer['excluded_TRAIN_available_UNKNOWN']==1
 assert answer['RAW_only']['unique_identities']==2 and answer['RAW_only']['conflicting_identities']==1 and answer['RAW_plus_native']['conflicting_identities']==0
 checks['collision_diagnostic_RAW_valid_NATIVE_and_TRAIN_known_only']=True
 oldsource=read(OLD/'grouped_fit_preparation_b/train_grouped_direct_end.py').decode();oldf={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(oldsource).body if isinstance(n,ast.FunctionDef)}
 inherited=('pure_training','metadata_matches','load_old','load_direct','gather','assemble','decode','objective','join_direct_split','split_masks','cohort_counts','save_cohorts','fresh_run')
 assert all(oldf[k]==ast.dump(funcs[k],include_attributes=False)for k in inherited)
 checks['13_inherited_loss_mask_join_U_functions_AST_exact']=True
 # No actual trainer import. Check control-flow source order and RAW-only arm literals.
 fit=ast.get_source_segment(source,funcs['fit'])
 assert fit.index("barrier=out/")<fit.index("full=save_outputs(")
 assert fit.index("for arm,use_sam,native_condition in ARMS:")<fit.index("barrier=out/")
 assert "for update in range(1,8001)"in fit and 'manual_seed(1810)'in fit and 'default_rng(1811)'in fit
 assert "ARMS=(('C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL',False,False),('C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT',False,True))"in source
 assert "model(inp,native)"in ast.get_source_segment(source,funcs['save_outputs'])
 checks['both_final8000_before_source_predictions_RAW_flags_and_saved_native']=True
 assert all(checks.values()) and not torch.cuda.is_initialized()
 return dict(completed=True,PASS=True,invented_CPU_only=True,checks=checks,count=len(checks),torch_version=torch.__version__,actual_source_metadata_arrays_pixels_checkpoints_scores_reads=0,actual_fit_or_source_forward=0,CUDA_initialized=False,notes='No actual trainer import; pure helpers AST-extracted. Only fresh invented tensors/scalars and explicit source code files are read.')

if __name__=='__main__':
 result=run();result['test_code']={'path':str(Path(__file__).absolute()),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with (H/'GENERATED_INDEPENDENT_INVENTED_CHECKS.json').open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(result,ensure_ascii=False))
