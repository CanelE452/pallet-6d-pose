"""C15 invented metadata/RNG/loss checks only; no model or source tensor calls."""
from pathlib import Path
import argparse, hashlib, math, tempfile
import numpy as np
import torch
import train_source_once as c


def fixture():
 inputs=[];gt=[]
 for ci in range(32):
  source='LOCO'if ci<10 else'TOMIE';split='TRAIN'if 4<=ci<10 else'VALIDATION'
  raw=split if source=='LOCO'else'TOMIE_DEVELOPMENT_STRESS'
  for j in range(9):
   qi=9*ci+j
   for role in range(17):
    i=17*qi+role;rank=None if role==0 else(role-1)//2
    variant='BEFORE'if role==0 else('OPAQUE_COVER'if role%2 else'CUE_PRESERVING_SHAM')
    ready=role>0 and not(ci==4 and rank in(0,1,2,3))and not(ci>=10 and rank==0)
    r=dict(occurrence_index=i,occurrence_id='GENERATED_'+str(i),cue_index=ci,jitter_index=j,query_index=qi,donor_rank=rank,variant=variant,source=source,split=split,raw_split=raw,source_frame_group='GEN_'+str(ci),query_xy=[10.+j,20.],L=32.,initial_points=[[10.,20.]]*8,context_box_xyxy=[0.,0.,30.,30.])
    g={k:r[k]for k in('occurrence_index','occurrence_id','cue_index','jitter_index','donor_rank','variant')}
    g.update(candidate_aux_hidden_label_GT_only=0 if not role else(1 if role%2 else 0)if ready else-1,geometry_pair_ready=bool(ready),human_gold=False,native_mapping='UNKNOWN')
    inputs.append(r);gt.append(g)
 return inputs,gt


def reject(fn):
 try:fn()
 except(AssertionError,ValueError,IndexError,KeyError):return
 raise AssertionError('Expected rejection missing')


def test():
 torch.set_num_threads(1)
 inputs,gt=fixture();new=c.join_flat(inputs,gt);available=np.ones(c.NEW,bool)
 assert {s:int((new['typed_state']==s).sum())for s in'DEU'}==dict(D=2358,E=2070,U=468)
 for source,split,expect in(('LOCO','TRAIN',[450,396]),('LOCO','VALIDATION',[324,288]),('TOMIE','VALIDATION',[1584,1386])):
  mask=(new['source']==source)&(new['split']==split)&(new['labels']>=0)
  assert np.array_equal(np.bincount(new['labels'][mask],minlength=2),expect)
 poolsnew=c.flat_pools(new,available)
 bad=[dict(r)for r in inputs];bad[1]['query_xy']=[999.,999.];reject(lambda:c.join_flat(bad,gt))
 badgt=[dict(r)for r in gt];badgt[1]['candidate_aux_hidden_label_GT_only']=-1;reject(lambda:c.join_flat(inputs,badgt))
 bad=[dict(r)for r in inputs];bad[1530]['split']='TRAIN';reject(lambda:c.join_flat(bad,gt))
 c12,_,c10,_,_=c.old.source_modules()
 pools={'old_triplets':np.arange(12).reshape(4,3),'synth_triplets':np.arange(12,24).reshape(4,3),'old_LOCO_D':np.arange(24,30),'old_PALLOX_D':np.arange(30,36),'synth_D':np.arange(36,44),'YCB_D':np.arange(44,60),'old_natural_E':np.arange(80,86),'synth_E':np.arange(86,102),'YCB_E':np.arange(102,126)}
 mp={'D':np.arange(41984,42024),'E':np.arange(42024,42064)};wp=np.arange(c.old.BASE,c.old.BASE+124).reshape(62,2)
 y=np.full(c.N,-1,np.int8);y[:160]=0;y[[1,4,7,10,13,16,19,22]]=1;y[80:160]=1;y[mp['D']]=0;y[mp['E']]=1;y[wp[:,0]]=1;y[wp[:,1]]=0;y[c.BASE:]=new['labels']
 gens=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)];refs=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)]
 ng=torch.Generator().manual_seed(218);refnew=torch.Generator().manual_seed(218);prefix=hashlib.sha256();extra_hash=hashlib.sha256()
 for step in range(1000):
  ix,pairs,audit,ix128,*_=c.draw192(c12,c10,pools,mp,wp,poolsnew,gens,ng)
  reference=c.old.draw128(c12,c10,pools,mp,wp,*refs)
  assert np.array_equal(ix[:128],reference[0])and np.array_equal(pairs,reference[1])and audit['old128']==reference[2]
  # Independent literal hierarchical draw: cue then member, E32 then independent BEFORE16.
  er=[];br=[]
  for _ in range(32):
   cue=int(poolsnew['cues'][torch.randint(6,(1,),generator=refnew).item()]);rr=poolsnew['cover'][cue];er.append(int(rr[torch.randint(len(rr),(1,),generator=refnew).item()]))
  for _ in range(16):
   cue=int(poolsnew['cues'][torch.randint(6,(1,),generator=refnew).item()]);rr=poolsnew['before'][cue];br.append(int(rr[torch.randint(9,(1,),generator=refnew).item()]))
  assert np.array_equal(ix[128:],np.r_[er,br,np.array(er[:16])+1])
  assert np.array_equal(np.bincount(y[ix],minlength=2),[96,96])and pairs.shape==(24,2)and pairs.max()<128
  assert np.array_equal(y[ix[pairs]],np.tile([1,0],(24,1)))
  assert (new['split'][ix[128:]-c.BASE]=='TRAIN').all()and (new['source'][ix[128:]-c.BASE]=='LOCO').all()
  assert np.array_equal(ix[176:],ix[128:144]+1)
  c.old.hash_draw(prefix,ix128,pairs);extra_hash.update(ix[128:].astype('<i8').tobytes())
 # Two equal domain means plus ONLY old24 hinge, no pooled192 mean or new hinge.
 z=torch.linspace(-3,3,192,requires_grad=True);target=torch.as_tensor(y[ix],dtype=torch.float32);a=torch.ones(192,dtype=torch.bool);pp=torch.as_tensor(pairs)
 loss,ob,oh,nb=c.loss192(z,target,a,pp)
 refob=torch.nn.functional.binary_cross_entropy_with_logits(z[:128],target[:128]);refnb=torch.nn.functional.binary_cross_entropy_with_logits(z[128:],target[128:]);refh=torch.relu(2-(z[pp[:,0]]-z[pp[:,1]])).mean()
 assert torch.equal(loss,refob+.25*refh+refnb)and torch.equal(ob,refob)and torch.equal(nb,refnb)
 assert not torch.isclose(loss,torch.nn.functional.binary_cross_entropy_with_logits(z,target)+.25*refh)
 a[[0,128]]=False;target[[1,129]]=-1
 loss,*_=c.loss192(z,target,a,pp);loss.backward();assert not z.grad[[0,1,128,129]].any()
 # Four new selector cohorts preserve unavailable-known denominators with exact ln2.
 original_selector=c.old.selector10
 c.old.selector10=lambda *_:(1.,{f'OLD{i}':dict(count=1,mean_BCE=1.,unavailable=0)for i in range(10)})
 try:
  zz=np.zeros(c.N,np.float32);aa=np.ones(c.N,bool);aa[c.BASE:]=False
  value,parts=c.selector14(None,zz,aa,{},new,None)
  assert len(parts)==14 and abs(value-(10+4*math.log(2))/14)<1e-15
  assert [parts[k]['count']for k in('LOCO_VALIDATION_D','LOCO_VALIDATION_E','TOMIE_DEVELOPMENT_D','TOMIE_DEVELOPMENT_E')]==[324,288,1584,1386]
  assert all(parts[k]['count']==parts[k]['unavailable']for k in parts if not k.startswith('OLD'))
 finally:c.old.selector10=original_selector
 history=[dict(update=u,criterion_fourteen_VAL_cohorts_BCE=v)for u,v in((0,.5),(100,.4),(200,.4),(300,.6))]
 assert c.earliest_history(history)==(100,.4)
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'denied.json';c.write(p,dict(approved_source_fit=False));reject(lambda:c.validate_entry(p))
 design=c.read(c.DESIGN);assert c.sha(c.DESIGN)==c.DESIGN_SHA and design['sampling']['new_generator_seed']==218 and design['optimizer']['lr']==1e-4
 source=Path(c.__file__).read_text();assert 'model(xo),model(xn)'in source and 'assert not initial.any()'not in source
 return dict(completed=True,PASS=True,code=c.bind(c.__file__),test_code=c.bind(__file__),design=c.bind(c.DESIGN),runtime=torch.__version__,all4896_joined=True,combined_occurrences=51004,combined_units=19432,all1000_invented_old128_draws_exact=True,new218_hierarchical_cue_then_member_exact=True,E32_BEFORE16_matchedSHAM16=True,no192_shuffle=True,only_old24_hinge=True,equal_old_new_BCE_domain_weights=True,TRAIN_LOCO_only_no_S_U_loss=True,U_unavailable_zero_gradient=True,selector14_safe0_denominators=True,earliest_strict_minimum=True,unapproved_entry_rejected_before_payloads=True,warmstart_two_shapes_static_check=True,generated_old128_prefix_sha256=prefix.hexdigest(),generated_new64_sha256=extra_hash.hexdigest(),actual_source_features_pixels_parameters_models_forward_fit_current_V112_reads_or_calls=0)


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();result=test();c.write(a.output,result);print(result)
