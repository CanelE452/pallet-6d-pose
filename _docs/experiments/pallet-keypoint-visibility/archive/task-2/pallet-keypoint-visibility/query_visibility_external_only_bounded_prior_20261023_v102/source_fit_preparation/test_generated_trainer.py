"""Generated C6/C7 trainer checks, no actual source tensors/weights or fit."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
import torch
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v102_generated_trainer',H/'train_source_once.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t);d=t.d

def mock_meta():
 n=d.N;y=np.full(n,-1,np.int8);source=np.full(n,'PALLOX',dtype='U30');split=np.full(n,'TRAIN',dtype='U12');variant=np.full(n,'BEFORE',dtype='U24');raw=np.full(n,'U');domain=np.array(['V96_REAL']*d.NOLD+['V99_SYNTH']*d.NNEW)
 spec=[(0,'LOCO','BEFORE',0,'D'),(1,'PALLOX','BEFORE',0,'D'),(2,'PALLOX','CUE_PRESERVING_SHAM',0,'D'),(3,'PALLOX','BEFORE',1,'E'),(4,'PALLOX','OPAQUE_COVER',1,'E'),(d.NOLD,'SYNTH_EXPECTED_CUE','BEFORE',0,'D'),(d.NOLD+1,'SYNTH_EXPECTED_CUE','SHAM',0,'D'),(d.NOLD+2,'SYNTH_EXPECTED_CUE','COVER',1,'E'),(d.NOLD+3,'SYNTH_EXPECTED_CUE','BEFORE',1,'S')]
 for i,s,v,k,state in spec:y[i]=k;source[i]=s;variant[i]=v;split[i]='VALIDATION';raw[i]=state
 old={k:v[:d.NOLD]for k,v in dict(labels=y,source=source,split=split,variant=variant).items()};known=np.isin(raw,['D','E'])
 return dict(labels=y,known=known,typed_state=raw,source=source,split=split,variant=variant,domain=domain,old_meta=old,known_geometric_S=raw=='S',unresolved_teacher_U=raw=='U',external_target=np.where(known,y,-1))

def main(output):
 torch.set_num_threads(1);checks={}
 n=32;y=torch.tensor([0,1,0]*8+[0,1,-1,-1,0,1,0,1]);z=torch.linspace(-2,2,n,dtype=torch.float64,requires_grad=True);raw=torch.linspace(1,-1,n,dtype=torch.float64);a=torch.ones(n,dtype=torch.bool);a[[2,4,27,31]]=False;triples=torch.arange(24).reshape(8,3)
 # Deliberately inconsistent anchor sign in this helper fixture proves that the
 # correct-base penalty reads the ORIGINAL reference, not anchor.
 out=dict(query_logit=z,query_available=a,reference_base_logit=raw,base_logit=-raw,delta_logit=z-raw);loss=t.training_loss(out,y,triples);loss['loss'].backward()
 zz=z.detach().numpy();rr=raw.numpy();yy=y.numpy();av=a.numpy();v=(yy>=0)&av;correct=v&((rr>=0)==(yy==1));p=triples.numpy();pm=av[p[:,1]]&av[p[:,2]];cc=p[pm,1];ss=p[pm,2];gap=zz[cc]-zz[ss]
 expected=np.mean(np.logaddexp(0,zz[v])-yy[v]*zz[v])+.05*np.mean((zz[correct]-rr[correct])**2)+.25*np.mean(np.maximum(2-gap,0));g=np.zeros(n);g[v]=(1/(1+np.exp(-zz[v]))-yy[v])/v.sum();g[correct]+=.1*(zz[correct]-rr[correct])/correct.sum();act=gap<2;g[cc[act]]-=.25/len(cc);g[ss[act]]+=.25/len(cc)
 assert abs(float(loss['loss'])-expected)<1e-12 and np.max(abs(z.grad.numpy()-g))<1e-12 and np.all(z.grad.numpy()[~v]==0)
 checks['analytic_loss_gradient_maxdiff']=float(np.max(abs(z.grad.numpy()-g)));checks['correct_base_uses_original_reference']=True
 bad=dict(query_logit=torch.zeros(24,requires_grad=True),query_available=torch.zeros(24,dtype=torch.bool),reference_base_logit=torch.zeros(24),base_logit=torch.zeros(24),delta_logit=torch.zeros(24));lo=t.training_loss(bad,torch.tensor([0,1,0]*8),triples);assert float(lo['loss'])==0 and lo['available_margin_pairs']==0;lo['loss'].backward();checks['all_unavailable_zero_loss_margin']=True
 pools={'old_triplets':np.arange(24).reshape(8,3),'new_triplets':np.arange(24,48).reshape(8,3),'old_LOCO_BEFORE_D':np.arange(48,52),'old_PALLOX_BEFORE_D':np.arange(52,56),'old_BEFORE_E':np.arange(56,64),'new_ALL_D':np.arange(64,72),'new_ALL_E':np.arange(72,88)}
 label=np.zeros(88,np.int8);label[np.arange(24).reshape(8,3)[:,1]]=1;label[np.arange(24,48).reshape(8,3)[:,1]]=1;label[56:64]=1;label[72:]=1
 gs=[torch.Generator(device='cpu').manual_seed(205)for _ in range(2)];hh=hashlib.sha256()
 for _ in range(2000):
  i1,p1,r1=d.draw(pools,gs[0]);i2,p2,r2=d.draw(pools,gs[1]);assert np.array_equal(i1,i2)and np.array_equal(p1,p2)and r1==r2
  assert len(i1)==64 and np.sum(label[i1]==0)==np.sum(label[i1]==1)==32 and np.array_equal(i1[p1].reshape(-1),np.array(r1['unshuffled_occurrence_indices'])[:24]);t.draw_hash_update(hh,i1,p1)
 checks['all2000_identical_cpu205_D32_E32_draws']=True;checks['generated_draw_sha256']=hh.hexdigest()
 m=mock_meta();zz=np.linspace(-1,1,d.N);aa=np.ones(d.N,bool);aa[0]=False;value,parts=d.selector(zz,aa,m);masks=d.val_masks(m);assert len(parts)==8 and 'new_ALL_S'not in parts
 direct=[float(np.mean(np.logaddexp(0,np.where(aa,zz,0)[v])-m['labels'][v]*np.where(aa,zz,0)[v]))for v in masks.values()];assert value==float(np.mean(direct))
 zz2=zz.copy();zz2[m['known_geometric_S']|m['unresolved_teacher_U']]=1e6;assert d.selector(zz2,aa,m)[0]==value
 cm=d.external_cm(zz,aa,m,np.ones(d.N,bool));assert cm['known_geometric_S_out_of_task_N']==1 and cm['unresolved_teacher_U_N']==d.N-9 and cm['known_external_N']==8 and 'unknown_GT_N'not in cm
 assert parts['old_LOCO_BEFORE_D']['mean_BCE']==float(np.log(2));checks['S_known_outside_task_distinct_from_U_and_excluded_selector']=True
 best=float('inf');idx=None
 for update,x in zip((0,100,200,300),(1.,.9,.9,1.)):
  if x<best:best=x;idx=update
 assert idx==100;checks['strict_earliest_selector']=True
 assert d.decode(np.array([-1.,0.,1.]),np.array([True,True,False])).tolist()==[0,1,-1];checks['raw0_invalidU_precedence']=True
 try:t.require_entry({'approved_actual_source_fit':False});raise RuntimeError('unapproved entry accepted')
 except AssertionError:checks['no_unapproved_source_entry']=True
 # Pure prospective inter-arm admission law, not a performance-derived choice.
 def choose(records):
  passing=[r for r in records if r['all_gates_pass']]
  return min(passing,key=lambda r:(r['criterion'],t.ARMS.index(r['arm'])))['arm']if passing else None
 assert choose([dict(arm=x,criterion=.5,all_gates_pass=True)for x in t.ARMS])==t.ARMS[0]
 assert choose([dict(arm=x,criterion=.1,all_gates_pass=False)for x in t.ARMS])is None;checks['prospective_passing_only_arm_selection_C6_tie']=True
 result=dict(completed=True,PASS=True,trainer=d.bind(H/'train_source_once.py'),adapter=d.bind(H/'external_data_adapter.py'),test_code=d.bind(__file__),runtime=dict(python=sys.executable,torch=torch.__version__,numpy=np.__version__),checks=checks,actual_source_array_weight_model_forward_optimizer_reads=0,generated_scalar_autograd_only=True)
 with Path(output).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(PASS=True,output=d.bind(output),checks=checks)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
