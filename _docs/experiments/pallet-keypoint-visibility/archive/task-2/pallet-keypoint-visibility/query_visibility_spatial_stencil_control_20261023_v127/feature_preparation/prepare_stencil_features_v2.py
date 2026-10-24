"""Root-only local stencil extraction. No Torch/model/source I/O at import."""
import os,sys
os.environ['XFORMERS_DISABLED']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,importlib.util,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
s=importlib.util.spec_from_file_location('v127_common',V/'stencil_common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
PROTOCOL=H/'FEATURE_PROTOCOL_V2_BEFORE_FORWARD.json';OUT=V/'stencil_features_v2';PARENT=P/'query_visibility_boundary_query_source_20261023_v125/feature_baseline_preparation/FEATURE_BASELINE_PROTOCOL_BEFORE_FORWARD.json'

def authorize(entry):
 e=c.read(entry);assert e['approved_source_stencil_feature_extraction']is True and e['source_only']is True
 assert all(e[k]is False for k in('fit_allowed','current_allowed','V112_allowed','PnP_allowed'))
 for k in('code','common_code','protocol','policy','precision_addendum','generated'):c.check(e[k])
 p=c.read(PROTOCOL);root=c.policy();assert e['code']==p['code']==c.bind(__file__)and e['common_code']==p['common_code']==c.bind(V/'stencil_common.py')and e['policy']==p['policy']==c.bind(c.POLICY)and e['precision_addendum']==p['precision_addendum']==c.bind(c.ADDENDUM)and e['generated']==p['generated']
 proof=c.read(e['generated']['path']);assert proof['PASS']and proof['feature_code']==e['code']and proof['common_code']==e['common_code']
 for k in('parent_protocol','input_occurrences','base_features','base_feature_receipt','stored_crops','C8_preprocess','C11_producer','asset_runtime','asset_plan','weight'):c.check(p[k])
 for b in p['official_source_bindings']:c.check(b)
 assert p['input_occurrences']==root['bindings']['input_occurrences']and p['base_features']==root['bindings']['base_features']
 return e,p

def run(entry):
 e,p=authorize(entry);assert not OUT.exists(),'Preserve previous failure, no silent restart.'
 import torch
 from torch.nn import functional as F
 assert torch.__version__==p['runtime']['torch']=='2.1.1+cu118'and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 OUT.mkdir();c.write(OUT/'STARTED.json',dict(entry=c.bind(entry),source_only=True,model_calls=0));start=time.monotonic()
 try:
  rs=c.rows(p['input_occurrences']['path']);assert len(rs)==c.N and all(r['occurrence_index']==r['rgb96_row']==i for i,r in enumerate(rs))
  with np.load(p['base_features']['path'],allow_pickle=False)as f:
   base={k:np.array(f[k],copy=True)for k in('features','occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index','unit_Q_inside')}
  with np.load(p['stored_crops']['path'],allow_pickle=False)as f:rgb=f['rgb96'];valid=f['valid96'];indices=f['occurrence_index']
  assert base['features'].shape==(c.U,1928)and base['features'].dtype==np.float32 and np.isfinite(base['features']).all()
  um=base['occurrence_unit'];assert um.shape==(c.N,)and np.array_equal(np.unique(um),np.arange(c.U))and np.array_equal(base['global_occurrence_index'],np.arange(c.N))and np.array_equal(indices,np.arange(c.N))
  assert rgb.shape==(c.N,3,96,96)and rgb.dtype==np.uint8 and valid.shape==(c.N,1,96,96)and valid.dtype==bool
  reps=np.full(c.U,-1,np.int32)
  for i,r in enumerate(rs):
   assert c.ash(rgb[i])==r['rgb96_sha256']and c.ash(valid[i])==r['valid96_sha256']
   u=int(um[i]);j=int(reps[u])
   if j<0:reps[u]=i
   else:assert np.array_equal(rgb[i],rgb[j])and np.array_equal(valid[i],valid[j])
   for k in('query_available','factual_query_available','context_available'):assert bool(base[k][i])==r[k]
  assert (reps>=0).all()and np.array_equal(base['query_available'],base['factual_query_available']&base['context_available'])and int((~base['query_available']).sum())==51
  ns=c.pure_functions(p['C8_preprocess']['path'],('preprocess_views',),dict(torch=torch,F=F,MEAN=(.485,.456,.406),STD=(.229,.224,.225)))
  preprocess=ns['preprocess_views']
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
  sys.path.insert(0,p['official_repo']);from dinov2.hub.backbones import dinov2_vits14_reg
  encoder=dinov2_vits14_reg(pretrained=False);state=torch.load(p['weight']['path'],map_location='cpu',weights_only=True);encoder.load_state_dict(state,strict=True);del state;encoder=encoder.cuda().float().eval().requires_grad_(False)
  before=c.state_sha(encoder);assert before==c.read(p['asset_runtime']['path'])['state_before_SHA256'];raws=[];pools=[];ordered=[];calls=0;repeated=False;parity=[]
  for start_u in range(0,c.U,8):
   uu=np.arange(start_u,min(start_u+8,c.U));ix=reps[uu];n=len(uu)
   rr=torch.as_tensor(np.ascontiguousarray(rgb[ix]),device='cuda');vv=torch.as_tensor(np.ascontiguousarray(valid[ix]),device='cuda')
   local,frac=preprocess(rr,vv);assert local.shape==(2*n,3,224,224)
   # Keep original3n shape/full96 row positions without native context reads.
   # Firstn rows are deterministic discarded full96 copies, never source outputs.
   views=torch.cat((local[:n],local),0);assert views.shape==(3*n,3,224,224)
   with torch.inference_mode():
    out=encoder.forward_features(views);calls+=1
    if not repeated:
     again=encoder.forward_features(views);calls+=1;assert all(torch.equal(out[k],again[k])for k in('x_norm_clstoken','x_norm_patchtokens'));repeated=True
    cls=out['x_norm_clstoken'][n:2*n];patch=out['x_norm_patchtokens'][n:2*n];assert patch.shape==(n,256,384)
    grid=patch.reshape(n,16,16,384)
    # Fixed half-index bilinear nodes have equal quarter weights. Canonical mean
    # order includes exactly C11's rawP expression at the central node.
    nodes=[grid[:,y:y+2,x:x+2].mean((1,2))for y in c.AXIS for x in c.AXIS]
    rawp=patch.reshape(n,16,16,384)[:,7:9,7:9].mean((1,2));assert torch.equal(nodes[4],rawp)
    full=F.normalize(cls.float(),p=2,dim=1,eps=1e-12);full=torch.where((frac[:,0]>0)[:,None],full,torch.zeros_like(full))
    c4=vv[:,0,47:49,47:49].flatten(1).all(1);pv=F.normalize(rawp.float(),p=2,dim=1,eps=1e-12);pv=torch.where(c4[:,None],pv,torch.zeros_like(pv))
    expectedF=torch.as_tensor(base['features'][uu,768:1152],device='cuda');expectedP=torch.as_tensor(base['features'][uu,1536:1920],device='cuda')
    fe=torch.equal(full,expectedF);pe=torch.equal(pv,expectedP)
    if not(fe and pe):
     c.write(OUT/'PARITY_FAILURE.json',dict(units=uu.tolist(),F_bitexact=fe,P_bitexact=pe,max_F_drift=float((full-expectedF).abs().max()),max_P_drift=float((pv-expectedP).abs().max()),no_tolerance_relaxation=True));raise AssertionError('CachedF/P exact parity failed; no retry or tolerance change.')
    raw=torch.stack(nodes,1);pool=F.normalize(raw.mean(1).float(),p=2,dim=1,eps=1e-12);pool8=pool[:,None,:].expand(-1,8,-1).reshape(n,3072)
    outer=torch.stack([F.normalize(nodes[k].float(),p=2,dim=1,eps=1e-12)for k in c.OUTER],1).reshape(n,3072)
   raws.append(raw.cpu().numpy());pools.append(pool8.cpu().numpy());ordered.append(outer.cpu().numpy());parity.append(dict(first_unit=int(start_u),units=n,views=3*n,F_bitexact=True,P_bitexact=True))
   if(start_u//8+1)%40==0:print(c.read(OUT/'STARTED.json')['source_only'],start_u+n,c.U,calls,flush=True)
  assert c.state_sha(encoder)==before and repeated and calls==(c.U+7)//8+1 and all(not t.requires_grad for t in encoder.parameters());del encoder
  raw=np.concatenate(raws);pool8=np.concatenate(pools);outer=np.concatenate(ordered);assert raw.shape==(c.U,9,384)and pool8.shape==outer.shape==(c.U,3072)and raw.dtype==pool8.dtype==outer.dtype==np.float32 and all(np.isfinite(x).all()for x in(raw,pool8,outer))
  output=OUT/'STENCIL_FEATURES_INPUT_ONLY.npz';np.savez_compressed(output,raw_nodes=raw,pooled9_repeat8=pool8,ordered_outer8=outer,representative_occurrence_index=reps,occurrence_unit=um,query_available=base['query_available'],factual_query_available=base['factual_query_available'],context_available=base['context_available'],global_occurrence_index=base['global_occurrence_index'],unit_Q_inside=base['unit_Q_inside'])
  c.check(p['base_features']);c.write(OUT/'FEATURE_COMPLETE.json',dict(completed=True,PASS=True,code=c.bind(__file__),common_code=c.bind(V/'stencil_common.py'),protocol=e['protocol'],entry=c.bind(entry),policy=e['policy'],precision_addendum=e['precision_addendum'],output=c.bind(output),base_features=p['base_features'],input_occurrences=p['input_occurrences'],stored_crops=p['stored_crops'],occurrences=c.N,unique_inputs=c.U,extra_shape=[c.U,3072],node_shape=[c.U,9,384],base1928_rewritten=False,all_F_P_bitexact=True,parity_batches=parity,encoder_calls=calls,primary_calls=(c.U+7)//8,repeat_calls=1,first_batch_repeat_exact=True,encoder_state_before=before,encoder_state_after=before,model_state_unchanged=True,all_source_mapping_and_availability_exact=True,discarded_views='firstn full96 copies per3n call',seconds=time.monotonic()-start,labels_GT_body_or_hash_read=False,fit_current_V112_PnP=0));print(c.read(OUT/'FEATURE_COMPLETE.json'),flush=True)
 except BaseException:
  c.write(OUT/'FAILED_EXECUTION.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

def prepare():
 root=c.policy();parent=c.read(PARENT);cc=c.read(PARENT.parents[1]/'actual_corpus/CORPUS_COMPLETE.json');proof=H/'GENERATED_FEATURE_TESTS_V2.json';g=c.read(proof);assert g['PASS']and g['feature_code']==c.bind(__file__)and g['common_code']==c.bind(V/'stencil_common.py')
 keys=('asset_runtime','asset_plan','weight','official_repo','official_revision','official_source_bindings','runtime','C8_preprocess','C11_producer')
 p={k:parent[k]for k in keys};p.update(completed=True,code=c.bind(__file__),common_code=c.bind(V/'stencil_common.py'),policy=c.bind(c.POLICY),precision_addendum=c.bind(c.ADDENDUM),generated=c.bind(proof),parent_protocol=c.bind(PARENT),input_occurrences=root['bindings']['input_occurrences'],base_features=root['bindings']['base_features'],base_feature_receipt=root['bindings']['base_feature_receipt'],stored_crops=cc['stored_crops'],output=str(OUT),batch_units=8,batch_layout='discardedfulln,usedfulln,centraln; exact3n, lastn4',nodes='rowmajor k+.5=(3.5,7.5,11.5), canonical2x2 raw mean at3:5/7:9/11:13',outer_order=list(c.OUTER),normalization='POOL all9rawmean thenL2 repeat8; ORDERED raw outer nodeL2each',strict_parity='C11 maskedF/P bitexact; stop without tolerance widening',actual_source_pixel_feature_weight_model_reads_or_calls=0)
 c.write(PROTOCOL,p);c.write(H/'ROOT_FEATURE_ENTRY_TEMPLATE_V2.json',dict(approved_source_stencil_feature_extraction=False,source_only=True,fit_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,code=c.bind(__file__),common_code=p['common_code'],protocol=c.bind(PROTOCOL),policy=p['policy'],precision_addendum=p['precision_addendum'],generated=p['generated']))
 print(dict(code=c.bind(__file__),protocol=c.bind(PROTOCOL),template=c.bind(H/'ROOT_FEATURE_ENTRY_TEMPLATE_V2.json'),command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json',actual_forward=0),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_EXACT_ENTRY.json')
