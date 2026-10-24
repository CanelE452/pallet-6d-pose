"""C17 invented-only bounded checks. No actual features/checkpoints/source model calls."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json
from unittest.mock import patch
import numpy as np
import torch
H=Path(__file__).resolve().parent;CODE=H/'train_source_once.py'
s=importlib.util.spec_from_file_location('C17_generated',CODE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def main(out):
 checks={}
 w=dict(labels=np.array([1,1,0,0,0,1,0,-1,1],np.int8),split=np.array(['TRAIN']*5+['VALIDATION','TRAIN','TRAIN','TRAIN']),variant=np.array(['OPAQUE_COVER','OPAQUE_COVER','CUE_PRESERVING_SHAM','CUE_PRESERVING_SHAM','BEFORE','OPAQUE_COVER','BEFORE','OPAQUE_COVER','OPAQUE_COVER']),pair_ready=np.array([1,1,1,1,0,1,0,0,1],bool),alpha=np.array([0,1,1,0,-1,0,-1,-1,0],np.int8))
 av=np.array([1,1,1,1,1,1,0,1,0],bool);pools=m.wide_pools(w,av)
 checks['strata_known_available_TRAIN_only']=all(np.array_equal(pools[k],np.array([m.BASE+i]))for i,(k,_)in enumerate(m.STRATA))
 g=torch.Generator().manual_seed(220);g2=torch.Generator().manual_seed(220);sha=hashlib.sha256()
 for step in range(1000):
  ix,audit=m.draw_wide64(pools,g);ref,_=m.draw_wide64(pools,g2);assert np.array_equal(ix,ref)
  labels=w['labels'][ix-m.BASE];assert np.array_equal(np.bincount(labels,minlength=2),[32,32])
  pos=0
  for k,n in m.STRATA:
   assert np.isin(ix[pos:pos+n],pools[k]).all();pos+=n
  sha.update(ix.astype('<i8').tobytes())
 checks['1000_wide64_seed220_quota32D32E_exact']=True
 # Literal objective: mean128 + .25 hinge24 + mean64 + mean64, never pooled256.
 z=torch.linspace(-3,4,256,requires_grad=True);y=torch.tensor(([1.,0.]*64)+([1.]*32+[0.]*32)+([1.]*32+[0.]*32));a=torch.ones(256,dtype=torch.bool);pairs=torch.arange(48).reshape(24,2)
 total,ob,hinge,fb,wb=m.loss256(z,y,a,pairs)
 expected=torch.nn.functional.binary_cross_entropy_with_logits(z[:128],y[:128])+.25*torch.relu(2-(z[pairs[:,0]]-z[pairs[:,1]])).mean()+torch.nn.functional.binary_cross_entropy_with_logits(z[128:192],y[128:192])+torch.nn.functional.binary_cross_entropy_with_logits(z[192:],y[192:])
 checks['exact_equal_domain_loss_and_only_old24hinge']=torch.equal(total,expected)
 checks['loss_not_pooled256']=not torch.isclose(total,torch.nn.functional.binary_cross_entropy_with_logits(z,y))
 z2=torch.linspace(-3,4,256,requires_grad=True);y2=y.clone();a2=a.clone();a2[254]=False;y2[255]=-1
 loss,*_=m.loss256(z2,y2,a2,pairs);loss.backward();checks['new_unavailable_and_teacherU_grad0']=bool(z2.grad[254]==0 and z2.grad[255]==0 and z2.grad[253]!=0)
 # New selector parts independently expected as four means, equal weight with old14.
 wz=np.array([-2,1,-1,2,999],np.float32);wa=np.array([1,1,1,0,1],bool)
 wide=dict(labels=np.array([0,1,0,1,-1]),source=np.array(['LOCO','LOCO','TOMIE','TOMIE','LOCO']),split=np.full(5,'VALIDATION'),typed_state=np.array(list('DEDEU')))
 oldparts={f'OLD{i}':dict(count=i+1,mean_BCE=float(i)/10,unavailable=0)for i in range(14)}
 with patch.object(m.c15,'selector14',return_value=(0.,oldparts.copy())):
  val,parts=m.selector18(None,np.r_[np.zeros(m.BASE,np.float32),wz],np.r_[np.ones(m.BASE,bool),wa],None,None,wide,None)
 expected_parts=[float(np.logaddexp(0,-2)),float(np.logaddexp(0,1)-1),float(np.logaddexp(0,-1)),float(np.log(2))]
 checks['18_equal_cohort_selector_U_safe0']=len(parts)==18 and np.isclose(val,np.mean([i/10 for i in range(14)]+expected_parts),rtol=0,atol=1e-15)
 checks['selector_known_unavailable_counted']=parts['WIDE_TOMIE_DEVELOPMENT_E']['unavailable']==1 and parts['WIDE_LOCO_VALIDATION_D']['count']==1
 hist=[dict(update=i,criterion_eighteen_VAL_cohorts_BCE=v)for i,v in((0,2.),(100,1.),(200,1.),(300,1.5))];checks['earliest_strict_source_min']=m.earliest_history(hist)==(100,1.)
 # All-row vs factual/available denominator, including a known inputU and available teacherU.
 yy=np.array([0,1,-1,0,1,1],np.int8);predz=np.array([-1,0,4,1,1,1],np.float32);aa=np.array([1,0,1,1,1,1],bool)
 ww=dict(labels=yy,lineage=[dict(source='LOCO'if i<3 else'TOMIE',raw_split='VALIDATION'if i<3 else'TOMIE_DEVELOPMENT_STRESS',offset_xy=[0,0])for i in range(6)],variant=np.array(['CUE_PRESERVING_SHAM','OPAQUE_COVER','OPAQUE_COVER','CUE_PRESERVING_SHAM','OPAQUE_COVER','OPAQUE_COVER']),alpha=np.array([1,0,-1,1,1,0]),pair_ready=np.array([1,1,0,1,1,1],bool),nearest_q_in_image=np.ones(6,bool))
 report=m.wide_report(predz,aa,aa,ww);gg=report['groups']['LOCO::VALIDATION']
 checks['allrows_and_factual_denominators_distinct']=gg['all_planned']['teacher_E']==1 and gg['all_planned']['CM_D_E_rows_D_E_U_columns'][1]==[0,0,1]and gg['factual_available']['teacher_E']==0
 checks['teacherU_available_distinct']=gg['teacher_U_available']==1 and gg['known_input_U']==1
 checks['10_prospective_diagnostics_not_old_gates']=len(report['prospective_checks'])==10 and report['prospective_checks']['LOCO::VALIDATION::E_recall']['denominator']==1
 src={'state':torch.tensor([3.]),'optimizer':{'step':torch.tensor(10.)},'rng':[torch.tensor([5],dtype=torch.uint8)]};saved=m.clone_cpu(src);src['state'].add_(9);src['optimizer']['step'].add_(2)
 checks['selected_resume_deepcopy']=saved['state'].item()==3 and saved['optimizer']['step'].item()==10
 tree=ast.parse(CODE.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');text=ast.unparse(run)
 checks['exact_old192_helper_and_seeds_preserved']='c15.draw192(c12, c10, pools, mp, woodp, flatp, gens, flatgen)'in text and '(208, 214, 215, 216, 217)'in text and 'manual_seed(218)'in text and 'manual_seed(220)'in text and 'torch.manual_seed(219)'in text
 checks['separate3_predict_matrices']='torch.cat((model(xo), model(xf), model(xw)))'in text
 checks['1000_updates_fresh_optimizer_resume_saved']='range(1, 1001)'in text and 'torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)'in text and 'optimizer_state=clone_cpu(optimizer.state_dict())'in text and 'RNG_states=capture_rng(gens, flatgen, widegen)'in text
 assert all(checks.values()),checks
 result=dict(completed=True,PASS=True,code=dict(path=str(CODE),sha256=hashlib.sha256(CODE.read_bytes()).hexdigest()),runtime=dict(torch=torch.__version__,device='CPU invented tensors only'),checks={k:bool(v)for k,v in checks.items()},wide1000_generated_trace_sha256=sha.hexdigest(),actual_source_features_weights_model_forward_fit_current_V112=0)
 with Path(out).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);main(ap.parse_args().output)
