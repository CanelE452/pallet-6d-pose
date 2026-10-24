"""Invented tensors/metadata only; no actual source arrays/checkpoints/current."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,tempfile
import numpy as np
import torch
import train_source_once as m

def fixture924():
 inputs=[];gt=[];accepted=set(range(48,110))
 for b in range(28):
  for off in range(33):
   j=33*b+off;rank=None if not off else(off-1)//2;variant='BEFORE'if not off else('OPAQUE_COVER'if off%2 else'CUE_PRESERVING_SHAM');ready=off>0 and b*16+rank in accepted
   r=dict(occurrence_index=j,occurrence_id='GENERATED_'+str(j),base_index=b,original_base_row_index=b+100,slot_index=b+100,donor_rank=rank,variant=variant,source='LOCO'if b<3 else'PALLOX',split='TRAIN',raw_split='TRAIN'if b<3 else'TRAIN_LAB',query_xy=[float(b)+.25,float(b)+.5],L=64.,native_slot_logical_only=b%8,frozen_context_box_xyxy=[0.,0.,128.,128.],full_rgb_changed=bool(ready))
   g={k:r[k]for k in('occurrence_index','occurrence_id','base_index','original_base_row_index','donor_rank','variant')};g.update(candidate_aux_hidden_label_GT_only=0 if off==0 else(1 if off%2 else 0)if ready else-1,pair_ready=bool(ready),native_mapping='UNKNOWN',human_gold=False)
   inputs.append(r);gt.append(g)
 return inputs,gt

def rejected(fn):
 try:fn()
 except(AssertionError,ValueError,IndexError,KeyError):return
 raise AssertionError('Expected rejection missing')

def test():
 torch.set_num_threads(1);c12,c11,c10,c9,d=m.source_modules()
 inputs,gt=fixture924();joined=m.join_new(inputs,gt)
 assert joined['pairs'].shape==(62,2)and {s:int((joined['typed_state']==s).sum())for s in'DEU'}=={'D':90,'E':62,'U':772}
 assert(joined['split']=='TRAIN').all()and(joined['labels'][joined['typed_state']=='U']==-1).all()
 mapping=m.expected_new_units(inputs);assert mapping.shape==(924,)and len(np.unique(mapping))==152 and np.array_equal(mapping[::33],np.arange(28))
 bad=[dict(r)for r in inputs];bad[103]['query_xy']=[999.,999.];rejected(lambda:m.join_new(bad,gt))
 badgt=[dict(r)for r in gt];badgt[1]['candidate_aux_hidden_label_GT_only']=1;rejected(lambda:m.join_new(inputs,badgt))
 badgt=[dict(r)for r in gt];badgt[100]['pair_ready']=not badgt[100]['pair_ready'];rejected(lambda:m.join_new(inputs,badgt))
 bad=[dict(r)for r in inputs];bad[1]['raw_split']='VALIDATION';rejected(lambda:m.join_new(bad,gt))
 # Original old96 draws have independent generators; added32 and shuffle128 cannot perturb them.
 pools={'old_triplets':np.arange(12).reshape(4,3),'synth_triplets':np.arange(12,24).reshape(4,3),'old_LOCO_D':np.arange(24,30),'old_PALLOX_D':np.arange(30,36),'synth_D':np.arange(36,44),'YCB_D':np.arange(44,60),'old_natural_E':np.arange(80,86),'synth_E':np.arange(86,102),'YCB_E':np.arange(102,126)}
 y=np.full(m.N,-1,np.int8);y[:160]=0;y[[1,4,7,10,13,16,19,22]]=1;y[80:160]=1
 mp={'D':np.arange(41984,42024),'E':np.arange(42024,42064)};y[mp['D']]=0;y[mp['E']]=1;y[m.BASE:]=joined['labels']
 wp=joined['pairs'];gens=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)];refs=[torch.Generator().manual_seed(s)for s in(208,214,215)];hashes=[hashlib.sha256()for _ in range(3)]
 for step in range(2000):
  ix,pair,audit,i64,t64,i96,t96=m.draw128(c12,c10,pools,mp,wp,*gens)
  ri,rt,ra,r64,rt64=c12.draw96(c10,pools,mp,*refs)
  assert np.array_equal(i96,ri)and np.array_equal(t96,rt)and np.array_equal(i64,r64)and np.array_equal(t64,rt64)
  assert len(ix)==128 and len(set(audit['final128_permutation']))==128 and pair.shape==(24,2)
  assert np.array_equal(np.bincount(y[ix],minlength=2),[64,64])and np.array_equal(y[ix[pair]],np.tile([1,0],(24,1)))
  assert np.array_equal(ix[pair[:8]],ri[rt[:,1:3]])and np.array_equal(ix[pair[8:]],np.array(audit['new_cover_sham_occurrence_indices']))
  assert(audit['base96']==ra)and all(j in wp[:,0]for j in ix[pair[8:,0]])
  for h,ii,pp in zip(hashes,(i64,i96,ix),(t64,t96,pair)):m.hash_draw(h,ii,pp)
 # True2-column pair math, absolute BCE and unknown/unavailable zero gradient.
 raw=torch.randn(128,requires_grad=True);target=torch.tensor(y[ix],dtype=torch.float32);available=torch.ones(128,dtype=torch.bool);pairs=torch.as_tensor(pair)
 loss,bce,hinge=m.supervised_loss(raw,target,available,pairs)
 expected=torch.nn.functional.binary_cross_entropy_with_logits(raw,target)+.25*torch.relu(2-(raw[pairs[:,0]]-raw[pairs[:,1]])).mean()
 assert torch.equal(loss,expected);loss.backward();assert torch.isfinite(raw.grad).all()
 rejected(lambda:m.supervised_loss(raw,target,available,torch.zeros((2,3),dtype=torch.long)))
 z=torch.tensor([-2.,-5.,8.,-8.],requires_grad=True);t=torch.tensor([1.,0.,-1.,1.]);aa=torch.tensor([1,1,1,0],dtype=torch.bool);loss,bc,h=m.supervised_loss(z,t,aa,torch.tensor([[0,1]]));assert h==0 and bc>1;loss.backward();assert z.grad[0]<0 and z.grad[2]==z.grad[3]==0
 za=torch.zeros(4,requires_grad=True);lz,_,_=m.supervised_loss(za,torch.full((4,),-1.),torch.zeros(4,dtype=torch.bool),torch.empty((0,2),dtype=torch.long));lz.backward();assert not za.grad.any()
 assert d.decode(np.array([-1.,-0.,0.,1.],np.float32),np.array([1,1,1,0],bool)).tolist()==[0,1,1,-1]
 # Same head/seed, fresh generated weights; no trained checkpoint is loaded.
 torch.manual_seed(211);head=c11.SpatialTargetQueryHead();torch.manual_seed(211);head2=c11.SpatialTargetQueryHead()
 assert sum(p.numel()for p in head.parameters())==221825 and all(torch.equal(v,head2.state_dict()[k])for k,v in head.state_dict().items())
 xx=torch.randn(128,1928);out=head(xx);assert not out.any();ll,_,_=m.supervised_loss(out,target,available,pairs);ll.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in head.parameters())
 cuda_generated=False
 if torch.cuda.is_available():
  torch.use_deterministic_algorithms(True);torch.cuda.manual_seed_all(211);zg=torch.tensor([-2.,-5.,1.,-1.],device='cuda',requires_grad=True);tg=torch.tensor([1.,0.,1.,0.],device='cuda');ag=torch.ones(4,device='cuda',dtype=torch.bool);pg=torch.tensor([[0,1],[2,3]],device='cuda');a1,_,_=m.supervised_loss(zg,tg,ag,pg);a1.backward();g1=zg.grad.clone();zg.grad=None;a2,_,_=m.supervised_loss(zg,tg,ag,pg);a2.backward();assert torch.equal(a1,a2)and torch.equal(g1,zg.grad);cuda_generated=True
 # Selector is the exact parent10 function; appended TRAIN924 cannot affect it.
 oldparts={'OLD_'+str(i):dict(count=3,mean_BCE=.2+i*.03,unavailable=0)for i in range(8)}
 fake=SimpleNamespace(selector=lambda *_:(0.,dict(oldparts)))
 parentmeta=dict(old={},known=np.ones(m.BASE,bool),external_target=np.zeros(m.BASE,np.int8),domain=np.full(m.BASE,'OLD',dtype='U32'),split=np.full(m.BASE,'TRAIN',dtype='U16'),typed_state=np.full(m.BASE,'D'))
 parentmeta['domain'][-4:]=c12.MATERIAL;parentmeta['split'][-4:]='VALIDATION';parentmeta['typed_state'][-2:]='E';parentmeta['external_target'][-2:]=1
 zz=np.full(m.N,-2.,np.float32);zz[m.BASE-2:m.BASE]=2.;aa=np.ones(m.N,bool);aa[m.BASE-4]=False;meta={'parent_meta':parentmeta}
 value,parts=m.selector10(c12,zz,aa,meta,fake);expect=c12.selector10(zz[:m.BASE],aa[:m.BASE],parentmeta,fake);assert (value,parts)==expect and len(parts)==10
 zz[m.BASE:]=100.;aa[m.BASE:]=False;assert m.selector10(c12,zz,aa,meta,fake)==expect
 assert abs(parts['material_VALIDATION_D']['mean_BCE']-(np.log(2)+np.logaddexp(0,-2))/2)<1e-15
 assert m.earliest_history([{'update':0,'criterion_ten_VAL_cohorts_BCE':2.},{'update':100,'criterion_ten_VAL_cohorts_BCE':1.},{'update':200,'criterion_ten_VAL_cohorts_BCE':1.}])==(100,1.)
 design=m.read(m.DESIGN);parent=m.read(design['parent_C12_design']['path']);assert design['source_gates']==parent['source_gates']and len(design['source_gates'])==15 and design['selector']==parent['selector']
 # Entry rejection happens before source/feature/model bodies.
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'unapproved.json';m.write(p,{'approved_source_fit':False});rejected(lambda:m.validate_entry(p))
 return dict(completed=True,PASS=True,code=m.bind(m.__file__),test_code=m.bind(__file__),design=m.bind(m.DESIGN),parent_C12_trainer=m.bind(m.C12),runtime=torch.__version__,generated924_join_D90_E62_U772=True,all_unknowns_and_rawfolds_preserved=True,negative_q_label_pair_split_fixtures_rejected=True,generated152_storage_units_mapping=True,generated2000_old96_draws_independent_exact=True,generated_draw_hashes=[h.hexdigest()for h in hashes],actual_required_old64_SHA=m.OLD64_SHA,actual_required_old96_SHA=m.OLD96_SHA,batch128_64D64E_pairs24_true2columns=True,absolute_BCE_not_replaced_by_margin=True,unknown_unavailable_zero_gradient=True,fresh_exact_head221825_seed211=True,generated_CUDA_loss_backward_exact=cuda_generated,exact_parent_selector10_and15gates=True,appendTRAIN_no_selector_influence=True,unavailable_selection_ln2=True,earliest_strict_tie=True,unapproved_entry_before_source_bodies=True,actual_source_arrays_weights_model_forward_fit_current_V112_reads=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=test();m.write(a.output,r);print(r)
