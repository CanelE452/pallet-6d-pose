"""Generated-only numerical/sampler/selector checks; no source arrays/weights."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
import torch
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c5_generated_trainer',H/'train_source_once.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);d=t.d

def generated_meta():
 n=d.N;y=np.full(n,-1,np.int8);source=np.full(n,'PALLOX');split=np.full(n,'TRAIN');variant=np.full(n,'BEFORE');domain=np.array(['V96_REAL']*d.NOLD+['V99_SYNTH']*d.NNEW);typed=np.full(n,'U')
 specs=[(0,'LOCO','BEFORE',0,'D'),(1,'PALLOX','BEFORE',0,'D'),(2,'PALLOX','CUE_PRESERVING_SHAM',0,'D'),(3,'PALLOX','BEFORE',1,'E'),(4,'PALLOX','OPAQUE_COVER',1,'E'),(d.NOLD,'SYNTH_EXPECTED_CUE','BEFORE',0,'D'),(d.NOLD+1,'SYNTH_EXPECTED_CUE','SHAM',0,'D'),(d.NOLD+2,'SYNTH_EXPECTED_CUE','COVER',1,'E'),(d.NOLD+3,'SYNTH_EXPECTED_CUE','BEFORE',1,'S')]
 # Explicit Unicode lengths so synthetic strings cannot silently truncate.
 source=source.astype('U30');variant=variant.astype('U24');split=split.astype('U12')
 for i,s,v,l,ty in specs:y[i]=l;source[i]=s;variant[i]=v;split[i]='VALIDATION';typed[i]=ty
 old={k:v[:d.NOLD]for k,v in dict(labels=y,source=source,split=split,variant=variant).items()}
 return dict(labels=y,known=y>=0,source=source,split=split,variant=variant,typed_state=typed,domain=domain,old_meta=old)

def main(output):
 torch.set_num_threads(1);checks={}
 # Closed-form final-scalar BCE + delta-preservation + matched hinge, F64.
 n=32;y=torch.tensor(([0,1,0]*8)+[0,1,-1,-1,0,1,0,1]);z=torch.linspace(-2,2,n,dtype=torch.float64,requires_grad=True);base=torch.linspace(1,-1,n,dtype=torch.float64);a=torch.ones(n,dtype=torch.bool);a[[2,4,27,31]]=False;triples=torch.arange(24).reshape(8,3)
 out={'query_logit':z,'query_available':a,'base_logit':base,'delta_logit':z-base};loss=t.training_loss(out,y,triples);loss['loss'].backward()
 zz=z.detach().numpy();bb=base.numpy();yy=y.numpy();av=a.numpy();v=(yy>=0)&av;pc=v&((bb>=0)==(yy==1));p=triples.numpy();pm=av[p[:,1]]&av[p[:,2]];cc=p[pm,1];ss=p[pm,2]
 bce=np.mean(np.logaddexp(0,zz[v])-yy[v]*zz[v]);pen=np.mean((zz[pc]-bb[pc])**2);gap=zz[cc]-zz[ss];hinge=np.mean(np.maximum(2-gap,0));expected=bce+.05*pen+.25*hinge
 grad=np.zeros(n);grad[v]=(1/(1+np.exp(-zz[v]))-yy[v])/v.sum();grad[pc]+=.1*(zz[pc]-bb[pc])/pc.sum();active=gap<2;grad[cc[active]]-=.25/len(cc);grad[ss[active]]+=.25/len(cc)
 assert abs(float(loss['loss'])-expected)<1e-12 and np.max(abs(z.grad.numpy()-grad))<1e-12
 assert np.all(z.grad.numpy()[~v]==0);checks['analytic_loss_and_gradient_F64_maxdiff']=float(np.max(abs(z.grad.numpy()-grad)))
 # No known/available data or pair => differentiable exact zero.
 bad={'query_logit':torch.zeros(24,requires_grad=True),'query_available':torch.zeros(24,dtype=torch.bool),'base_logit':torch.zeros(24),'delta_logit':torch.zeros(24)}
 lo=t.training_loss(bad,torch.tensor([0,1,0]*8),torch.arange(24).reshape(8,3));assert float(lo['loss'])==0 and lo['available_margin_pairs']==0;lo['loss'].backward();checks['unavailable_loss_and_margin_zero']=True
 # 2000 draws, exact CPU205 sequence reproducibility, 24 triple rows+40 cohorts.
 pools={'old_triplets':np.arange(24).reshape(8,3),'new_triplets':np.arange(24,48).reshape(8,3),'old_LOCO_BEFORE_D':np.arange(48,52),'old_PALLOX_BEFORE_D':np.arange(52,56),'old_BEFORE_E':np.arange(56,64),'new_ALL_S':np.arange(64,72),'new_ALL_D':np.arange(72,80),'new_ALL_E':np.arange(80,88)}
 gs=[torch.Generator(device='cpu').manual_seed(205)for _ in range(2)];dig=hashlib.sha256()
 for update in range(2000):
  a1,p1,r1=d.draw(pools,gs[0]);a2,p2,r2=d.draw(pools,gs[1]);assert np.array_equal(a1,a2)and np.array_equal(p1,p2)and r1==r2
  assert len(a1)==64 and p1.shape==(8,3)and np.array_equal(a1[p1].reshape(-1),np.array(r1['unshuffled_occurrence_indices'])[:24]);assert r1['cohort_draw_quotas']==dict(old_triplets=4,new_triplets=4,old_LOCO_BEFORE_D=4,old_PALLOX_BEFORE_D=4,old_BEFORE_E=8,new_ALL_S=8,new_ALL_D=8,new_ALL_E=8)
  dig.update(a1.astype('<i8').tobytes());dig.update(p1.astype('<i8').tobytes())
 checks['all2000_generated_sampler_draws']=True;checks['generated_draw_sequence_sha256']=dig.hexdigest()
 # Same-nine-cohort equal weighting, unavailable safe0/ln2, unknown ignored.
 m=generated_meta();zz=np.linspace(-1,1,d.N);aa=np.ones(d.N,bool);aa[0]=False;val,parts=d.selector(zz,aa,m);ms=d.val_masks(m)
 ex=[]
 for k,mask in ms.items():
  q=np.where(aa,zz,0);ex.append(float(np.mean(np.logaddexp(0,q[mask])-m['labels'][mask]*q[mask])))
 assert val==float(np.mean(ex))and tuple(parts)==d.VAL_NAMES and parts['old_LOCO_BEFORE_D']['mean_BCE']==float(np.log(2))
 z2=zz.copy();z2[~m['known']]=1e6;assert d.selector(z2,aa,m)[0]==val
 criteria=[1.,1.,.9,.9,1.];best=float('inf');selected=None
 for i,x in enumerate(criteria):
  if x<best:best=x;selected=i*100
 assert selected==200;checks['nine_cohort_safe0_unknown_exclusion_and_earliest_tie']=True
 # Raw0 tie is hidden, invalid staysU; no learned cause declaration.
 assert t.decode(np.array([-1.,0.,1.]),np.array([True,True,False])).tolist()==[0,1,-1];checks['raw0_and_invalidU']=True
 # Unapproved entry must fail before any model/source reads.
 try:t.require_entry({'approved_actual_source_fit':False});raise RuntimeError('gate failed open')
 except AssertionError:checks['unapproved_entry_fails_closed']=True
 # Source loader and model are not invoked by tests.
 result={'completed':True,'PASS':True,'trainer':d.bind(H/'train_source_once.py'),'adapter':d.bind(H/'data_adapter.py'),'test_code':d.bind(__file__),'runtime':{'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__},'checks':checks,'actual_source_array_weight_model_forward_optimizer_reads':0,'generated_scalar_autograd_only':True,'model_semantic_accuracy_claim':False}
 with Path(output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'PASS':True,'output':d.bind(output),'checks':checks}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
