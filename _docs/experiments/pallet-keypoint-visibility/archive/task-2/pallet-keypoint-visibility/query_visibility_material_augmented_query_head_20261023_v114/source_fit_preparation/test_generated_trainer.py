"""Generated-only C12 joins, sampling, selector, loss and head API."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib
import numpy as np
import torch
import train_source_once as m

def test():
 torch.set_num_threads(1);c11,c10,c9,d=m.source_modules();torch.manual_seed(211);a=c11.SpatialTargetQueryHead();torch.manual_seed(211);b=c11.SpatialTargetQueryHead()
 assert sum(p.numel()for p in a.parameters())==221825 and all(torch.equal(v,b.state_dict()[k])for k,v in a.state_dict().items())
 x=torch.randn(96,1928);assert not a(x).count_nonzero()
 try:a({'features':x,'reference':torch.zeros(96,2)})
 except ValueError:pass
 else:raise AssertionError('GT dictionary input accepted')
 # Old draw uses its own unchanged generator; extras/final permutation cannot perturb it.
 y=np.zeros(m.N,np.int8);y[[1,4,7,10,13,16,19,22]]=1;y[80:160]=1
 pools={'old_triplets':np.arange(12).reshape(4,3),'synth_triplets':np.arange(12,24).reshape(4,3),'old_LOCO_D':np.arange(24,30),'old_PALLOX_D':np.arange(30,36),'synth_D':np.arange(36,44),'YCB_D':np.arange(44,60),'old_natural_E':np.arange(80,86),'synth_E':np.arange(86,102),'YCB_E':np.arange(102,126)}
 newpools={'D':np.arange(m.BASE,m.BASE+40),'E':np.arange(m.BASE+40,m.BASE+80)};y[newpools['E']]=1
 oldgen=torch.Generator().manual_seed(208);referencegen=torch.Generator().manual_seed(208);mg=torch.Generator().manual_seed(214);sg=torch.Generator().manual_seed(215);oldsha=hashlib.sha256();allsha=hashlib.sha256()
 for _ in range(2000):
  ix,tr,audit,old,ot=m.draw96(c10,pools,newpools,oldgen,mg,sg);expected,et,_=c10.draw(pools,referencegen)
  assert np.array_equal(old,expected)and np.array_equal(ot,et)and np.array_equal(ix[tr],old[ot])
  assert np.array_equal(np.bincount(y[ix],minlength=2),[48,48])and np.array_equal(y[ix][tr],np.tile([0,1,0],(8,1)))
  assert len(audit['material_D_occurrence_indices'])==len(audit['material_E_occurrence_indices'])==16 and len(audit['final96_permutation'])==len(set(audit['final96_permutation']))==96
  for h,ii,tt in ((oldsha,old,ot),(allsha,ix,tr)):h.update(ii.astype('<i8').tobytes());h.update(tt.astype('<i8').tobytes())
 # Full400x8 invented input/ref-free join, both original folds, S/U never hidden labels.
 inputs=[];line=[]
 for i in range(400):
  split='TRAIN'if i<300 else'VALIDATION';original='train'if i<300 else'val';q=[[i+k,k]for k in range(8)]
  r=dict(input_id='GENERATED_%04d'%i,instance_row_index=i,plan_row_index=i,initial_points=q,roi_side=16.,query_box_xyxy=[i,0,i+7,7],image={'path':'INVENTED','sha256':'0'*64},raw_hw=[480,640],split=split,original_split=original,source='GENERATED_MATERIAL',source_emulation=True,actual_initializer_query=False,native_mapping='UNKNOWN');inputs.append(r)
  for k in range(8):
   j=8*i+k;line.append(dict(local_occurrence_index=j,local_input_unit_index=j,global_occurrence_index=m.BASE+j,global_input_unit_index=m.OLDUNITS+j,instance_row_index=i,plan_row_index=i,logical_aux_slot=k,input_id=r['input_id'],query_id=r['input_id']+'::q'+str(k),query_xy=q[k],L=16.,query_box_xyxy=r['query_box_xyxy'],image=r['image'],raw_hw=r['raw_hw'],split=split,original_split=original,source=r['source']))
 state=np.tile(list('DESUDESU'),(400,1));target=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8);joined=m.join_material(inputs,line,state,target,np.array([r['initial_points']for r in inputs]),np.full(400,16.))
 assert len(joined['labels'])==3200 and(joined['labels'][np.isin(joined['typed_state'],['S','U'])]==-1).all()and(joined['split']=='TRAIN').sum()==2400
 bad=[dict(r)for r in line];bad[0]['global_input_unit_index']+=1
 try:m.join_material(inputs,bad,state,target,np.array([r['initial_points']for r in inputs]),np.full(400,16.))
 except AssertionError:pass
 else:raise AssertionError('Wrong global unit accepted')
 meta={k:np.concatenate((np.full(m.BASE,'U'if k=='typed_state'else -1 if k=='labels'else'OLD'),joined[k]))for k in('typed_state','labels','split','domain','source','variant')}
 meta.update(known=np.isin(meta['typed_state'],['D','E']),external_target=np.where(np.isin(meta['typed_state'],['D','E']),meta['labels'],-1).astype(np.int8),known_geometric_S=meta['typed_state']=='S',unresolved_teacher_U=meta['typed_state']=='U',old={})
 avail=np.ones(m.N,bool);avail[m.BASE]=False;mp=m.material_pools(meta,avail);assert m.BASE not in mp['D']and(meta['split'][mp['D']]=='TRAIN').all()
 oldparts={'OLD_%d'%i:dict(count=3,mean_BCE=.2+i*.03,unavailable=0)for i in range(8)};fake=SimpleNamespace(selector=lambda *_:(float(np.mean([r['mean_BCE']for r in oldparts.values()])),dict(oldparts)))
 z=np.where(meta['external_target']==0,-2.,2.).astype(np.float32);valD=np.flatnonzero((meta['domain']==m.MATERIAL)&(meta['split']=='VALIDATION')&(meta['typed_state']=='D'));avail[valD[0]]=False
 value,parts=m.selector10(z,avail,meta,fake);assert len(parts)==10 and parts['material_VALIDATION_D']['unavailable']==1
 safe=np.where(avail,z,0).astype(np.float64);loss=np.logaddexp(0,safe)-np.where(meta['known'],meta['external_target'],0)*safe
 expected=[r['mean_BCE']for r in oldparts.values()]+[float(loss[(meta['domain']==m.MATERIAL)&(meta['split']=='VALIDATION')&(meta['typed_state']==s)].mean())for s in('D','E')]
 assert value==float(np.mean(expected))and m.earliest_history([{'update':0,'criterion_ten_VAL_cohorts_BCE':2.},{'update':100,'criterion_ten_VAL_cohorts_BCE':1.},{'update':200,'criterion_ten_VAL_cohorts_BCE':1.}])==(100,1.)
 design=m.read(m.DESIGN);material,gates=m.material_report(d,z,avail,meta,design);assert set(gates)=={'material_VAL_D_recall_min','material_VAL_E_recall_min','material_VAL_known_U_rate_max'}and len(set(design['source_gates'])-set(gates))==12
 assert material['VALIDATION']['known_geometric_S_out_of_task_N']==200 and material['VALIDATION']['unresolved_teacher_U_N']==200
 # Unknown/unavailable loss0 and original8 matched hinge only; mean over eligible96.
 raw=torch.randn(96,requires_grad=True);tar=torch.as_tensor(y[ix],dtype=torch.float32);aa=torch.ones(96,dtype=torch.bool);pairs=torch.as_tensor(tr)
 loss,bce,hinge=c10.supervised_loss(raw,tar,aa,pairs);assert torch.equal(loss,bce+.25*hinge);loss.backward();assert torch.isfinite(raw.grad).all()
 raw2=torch.tensor([-2.,9.,-9.,3.],requires_grad=True);tt=torch.tensor([0.,1.,-1.,1.]);aa2=torch.tensor([1,0,1,1],dtype=torch.bool);l,_,_=c10.supervised_loss(raw2,tt,aa2,torch.empty((0,3),dtype=torch.long));l.backward();assert raw2.grad[1]==raw2.grad[2]==0 and raw2.grad[0]!=0 and raw2.grad[3]!=0
 assert d.decode(np.array([-1.,-0.,0.,1.],np.float32),np.array([1,1,1,0],bool)).tolist()==[0,1,1,-1]
 # A root entry must exist before any actual metadata/feature body can load.
 import tempfile
 with tempfile.TemporaryDirectory()as td:
  entry=Path(td)/'UNAPPROVED.json';m.write(entry,{'approved_source_fit':False})
  try:m.run(entry)
  except(AssertionError,KeyError):pass
  else:raise AssertionError('Unapproved entry accepted')
 return dict(completed=True,PASS=True,code=m.bind(m.__file__),test_code=m.bind(__file__),design=m.bind(m.DESIGN),head_class_code=m.bind(m.C11),runtime=torch.__version__,parameters=221825,exact_C11_class_seed211_lastzero=True,generated2000_base64_independent_rng_exact=True,generated_base64_SHA=oldsha.hexdigest(),generated_full96_SHA=allsha.hexdigest(),expected_actual_base64_SHA=m.EXPECTED_OLD_DRAW_SHA,generated_batch96_48D48E_and_triplet_remap_exact=True,material_rng214_finalshuffle215=True,generated400x8_join_and_rawfolds=True,global_unit_corruption_rejected=True,selector_equal10_and_unavailable_ln2=True,strict_earliest_tie=True,old12_plus_material3_gate_names_exact=True,S_U_loss0_and_no_extra_pairs=True,rawsign0_and_U=True,unapproved_entry_before_body_rejected=True,actual_source_features_checkpoints_pixels_model_fit_current_V112_reads=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=test();m.write(a.output,result);print(result)
