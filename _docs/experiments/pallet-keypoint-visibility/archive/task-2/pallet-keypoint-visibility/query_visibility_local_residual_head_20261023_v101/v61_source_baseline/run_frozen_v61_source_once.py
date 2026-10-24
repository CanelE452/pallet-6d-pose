"""Frozen original v61 source baseline. No backbone, fit, private or current reader."""
from pathlib import Path
import ast,argparse,hashlib,importlib.util,json,sys,time
import numpy as np
import torch
from torch import nn
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
ADAPTER=V/'source_fit_preparation/data_adapter.py'
CLASS=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
CKPT=P.parent/'target_presence_classifier_20261022_v61/BEST_PRESENCE_HEAD.pt'
CKPT_SHA='c75e04c41b8097ff63c2f134f76e12ec560603e7bcfe585446fe6042e31e4387'
PROTOCOL=H/'BASELINE_PROTOCOL_BEFORE_SOURCE_FORWARD.json';OUT=H/'outputs'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def proxy_class():
 node=next(x for x in ast.parse(CLASS.read_text()).body if isinstance(x,ast.ClassDef)and x.name=='ProxyHead');ns={'nn':nn}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(CLASS),'exec'),ns)
 return ns['ProxyHead'],hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
 return h.hexdigest()
def decode(z,a):return np.where(a,np.where(z>=0,0,1),-1).astype(np.int8)
def metric(pred,y,mask):
 known=y>=0;cm=np.array([[np.sum(mask&known&(y==c)&(pred==v))for v in(0,1,-1)]for c in(0,1)])
 ratio=lambda a,b:float(a/b)if b else None
 return dict(CM=cm.tolist(),rows=['AUX_D','AUX_hidden_E_or_S'],columns=['D','hidden','U'],known_N=int(cm.sum()),all_N=int(mask.sum()),teacher_U_N=int((mask&~known).sum()),D_precision=ratio(cm[0,0],cm[:,0].sum()),D_recall=ratio(cm[0,0],cm[0].sum()),hidden_precision=ratio(cm[1,1],cm[:,1].sum()),hidden_recall=ratio(cm[1,1],cm[1].sum()),known_prediction_U_rate=ratio(cm[:,2].sum(),cm.sum()),teacher_U_predictions={str(v):int((mask&~known&(pred==v)).sum())for v in(0,1,-1)})
def pair_metric(pred,pairs,split,splits):
 pp=pairs[splits[pairs[:,0]]==split];n=len(pp);good=(pred[pp]==np.array([0,1,0])).all(1)
 return dict(N=n,all3_correct=int(good.sum()),all3_correct_rate=float(good.mean())if n else None,before_D=int((pred[pp[:,0]]==0).sum()),cover_hidden=int((pred[pp[:,1]]==1).sum()),sham_D=int((pred[pp[:,2]]==0).sum()))
def report(z,a,m):
 pred=decode(z,a);y=m['labels'];out={}
 for dom in('V96_REAL','V99_SYNTH'):
  out[dom]={}
  for split in('TRAIN','VALIDATION'):
   mask=(m['domain']==dom)&(m['split']==split)
   out[dom][split]=dict(all_known=metric(pred,y,mask),by_variant={v:metric(pred,y,mask&(m['variant']==v))for v in np.unique(m['variant'][mask])},by_typed_state={t:metric(pred,y,mask&(m['typed_state']==t))for t in('D','E','S','U')},D_vs_E=metric(pred,y,mask&np.isin(m['typed_state'],['D','E'])),D_vs_S=metric(pred,y,mask&np.isin(m['typed_state'],['D','S'])),by_appearance={v:metric(pred,y,mask&(m['appearance']==v))for v in np.unique(m['appearance'][mask])},by_aspect={v:metric(pred,y,mask&(m['aspect']==v))for v in np.unique(m['aspect'][mask])},paired=pair_metric(pred,m['old_pairs']if dom=='V96_REAL'else m['new_pairs'],split,m['split']))
 return out

def freeze(bindings_path):
 assert not OUT.exists()and sha(CKPT)==CKPT_SHA
 b=read(bindings_path)
 for v in b.values():check(v)
 d=module(ADAPTER,'v61_freeze_exact_C5_adapter');m=d.load_metadata(b);_,astsha=proxy_class()
 g=H/'GENERATED_BASELINE_TESTS.json';proof=read(g);assert proof['PASS']and proof['code']==bind(__file__)
 assert sys.version_info[:2]==(3,10)and torch.__version__=='2.1.1+cu118'
 write(PROTOCOL,dict(completed=True,root_delegated_source_only_authorization=True,code=bind(__file__),generated=bind(g),data_adapter=bind(ADAPTER),old_inert_helper=bind(d.OLD_CODE),original_class_code=bind(CLASS),ProxyHead_AST_SHA=astsha,checkpoint=bind(CKPT),bindings_file=bind(bindings_path),bindings=b,source_counts=m['counts'],expected_occurrences=33984,unique_units=3652,parameter_count=28033,batch=128,device='cuda:0',runtime=dict(python=str(Path(sys.executable).resolve()),torch=torch.__version__,numpy=np.__version__,TF32=False,autocast=False,threads=1),decoder='Original visible logit >=0 -> D(0), <0 -> hidden(1); factual unavailable -> U(-1). Signed zero is D. No rounded probability decoding.',GT_targets='Exact C5 adapter D0,EorS1,U-1; typed E/S separate. Physical-cue AUX differs from original v61 M4-centre presence; bound V90 matched feature recipe. No original synthetic evaluation claim.',evaluation='All source splits/variants/typedstates/appearance/aspect; exact194old+96new selectedtriples. Teacher unknown separate. No gate/candidate/threshold selection.',no_new_backbone_fit_current_private_PnP=True,training_admission=False))
 print(json.dumps(bind(PROTOCOL)),flush=True)
def run():
 p=read(PROTOCOL)
 for k in('code','generated','data_adapter','old_inert_helper','original_class_code','checkpoint','bindings_file'):check(p[k])
 for b in p['bindings'].values():check(b)
 assert p['root_delegated_source_only_authorization']and p['code']==bind(__file__)and sha(CKPT)==CKPT_SHA and not OUT.exists()
 assert read(p['generated']['path'])['PASS']and read(p['generated']['path'])['code']==p['code']
 assert torch.__version__=='2.1.1+cu118'and str(Path(sys.executable).resolve())==p['runtime']['python']
 torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 OUT.mkdir();write(OUT/'STARTED.json',dict(protocol=bind(PROTOCOL),no_silent_restart=True));begin=time.monotonic()
 try:
  d=module(check(p['data_adapter']),'v61_actual_C5_adapter');m=d.load_metadata(p['bindings']);assert m['counts']==p['source_counts']
  data,u,available=d.load_inputs(p['bindings'],m);assert len(u)==33984 and len(data['p3_roi'])==3652
  C,astsha=proxy_class();assert astsha==p['ProxyHead_AST_SHA'];model=C();assert sum(x.numel()for x in model.parameters())==28033
  model.load_state_dict(torch.load(CKPT,map_location='cpu'),strict=True);model.requires_grad_(False).eval().cuda();before=state_sha(model);zz=[]
  with torch.inference_mode():
   for lo in range(0,len(data['p3_roi']),128):
    x=torch.from_numpy(np.ascontiguousarray(data['p3_roi'][lo:lo+128])).float().cuda();z=model(x).float().cpu().numpy();zz.append(z)
    if lo==0:assert np.array_equal(z,model(x).float().cpu().numpy())
  unitz=np.concatenate(zz);z=unitz[u];assert np.isfinite(z).all()and state_sha(model)==before and sha(CKPT)==CKPT_SHA
  pred=decode(z,available);line=m['lineage'];ids=np.array([r.get('occurrence_id',r.get('image_id','')+':AUX'+str(r.get('aux_corner_index',-1)))for r in line])
  np.savez_compressed(OUT/'SOURCE_SCORES.npz',raw_visible_logit=z,hidden_logit=-z,decoded_aux_prediction=pred,available=available,label=m['labels'],typed_state=m['typed_state'],domain=m['domain'],source=m['source'],split=m['split'],original_split=m['original_split'],variant=m['variant'],appearance=m['appearance'],aspect=m['aspect'],occurrence_id=ids,occurrence_index=np.arange(len(z)),input_unit_index=u,query_xy=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line],np.float64),L=np.array([np.nan if r['L']is None else r['L']for r in line],np.float64),unit_visible_logit=unitz,old_pairs=m['old_pairs'],new_pairs=m['new_pairs'])
  write(OUT/'SOURCE_RESULTS.json',dict(completed=True,source_only=True,protocol=bind(PROTOCOL),results=report(z,available,m),all33984_retained=True,dedup_units=3652,model_parameters=28033,model_state_before=before,model_state_after=state_sha(model),first_batch_repeat_exact=True,inference_batches=len(zz),repeat_forward_batches=1,new_backbone_fit_current_private_PnP=0,source_GT_class_or_mask_forwarded=False,native8_mapping='UNKNOWN',human_gold=False,limits=p['GT_targets'],seconds=time.monotonic()-begin))
  write(OUT/'BASELINE_SEAL.json',dict(completed=True,PASS_execution=True,protocol=bind(PROTOCOL),checkpoint=p['checkpoint'],scores=bind(OUT/'SOURCE_SCORES.npz'),results=bind(OUT/'SOURCE_RESULTS.json'),first_batch_repeat_exact=True,frozen_model_unchanged=True,no_new_backbone_fit_private_current_PnP=True))
  print(json.dumps(dict(completed=True,seal=bind(OUT/'BASELINE_SEAL.json'),seconds=time.monotonic()-begin))),
 except BaseException as exc:
  write(OUT/'FAILED.json',dict(exception_type=type(exc).__name__,exception=str(exc),no_automatic_restart=True));raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze-bindings',type=Path);a.add_argument('--run',action='store_true');args=a.parse_args()
 if args.freeze_bindings:freeze(args.freeze_bindings)
 elif args.run:run()
 else:raise SystemExit('choose --freeze-bindings or --run')
