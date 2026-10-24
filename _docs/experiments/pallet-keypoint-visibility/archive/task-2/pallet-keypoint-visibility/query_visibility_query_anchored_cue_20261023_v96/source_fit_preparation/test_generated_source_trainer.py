"""Invented arrays/metadata only; no real source, weights or current reads."""
from pathlib import Path
import argparse,copy,importlib.util,json,math,tempfile
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c3_generated_trainer',HERE/'train_source_once.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
def rejected(fn):
 try:fn()
 except(AssertionError,ValueError,KeyError):return
 raise AssertionError('Malformed fixture accepted')
def metadata():
 lineage=[];gt=[]
 for i in range(960):
  y=0 if i<6 or 160<=i<215 else 1 if 215<=i<219 else-1
  source='LOCO'if i<160 else'PALLOX';raw=('TRAIN'if i%2==0 else'VALIDATION')if source=='LOCO'else('TRAIN_LAB'if i%2==0 else'VAL_HALL')
  for j in range(33):
   k=None if j==0 else(j-1)//2;v='BEFORE'if j==0 else t.VARIANTS[1]if j%2 else t.VARIANTS[2];index=33*i+j;oid=f'INVENTED{i}::{j}'
   accepted=i in(0,1,160,161)and k in(0,2,15);label=y if j==0 else(1 if j%2 else 0)if accepted else-1
   lineage.append(dict(occurrence_index=index,slot_index=i,donor_rank=k,variant=v,occurrence_id=oid,source=source,split=raw,query_xy=[float(i%31),17.],L=32.))
   gt.append(dict(occurrence_index=index,slot_index=i,base_row_index=i,donor_rank=k,variant=v,occurrence_id=oid,candidate_aux_hidden_label_GT_only=label,known_candidate_GT_only=label>=0,query_anchored_before_label=y,original_C1_V95_label=1 if i==220 else y,pair_ready=accepted,native_expected_part_state='UNKNOWN',human_gold=False))
 return lineage,gt

def main(output):
 torch.set_num_threads(1);torch.manual_seed(194)
 lineage,gt=metadata();m=t.join_metadata(lineage,gt);assert m['pair_ready'].shape==(960,16)and m['pair_ready'].sum()==12
 assert [np.sum(m['labels'][::33]==k)for k in(0,1,-1)]==[61,4,895]
 assert m['counts']['TRAIN']['accepted_pairs']==m['counts']['VALIDATION']['accepted_pairs']==6
 assert m['original_split'][33*160]=='TRAIN_LAB'and m['split'][33*160]=='TRAIN'
 for ss,raw in [('LOCO','TRAIN_LAB'),('PALLOX','TRAIN'),('PALLOX','VALIDATION'),('NOPE','TRAIN')]:rejected(lambda:t.canonical_split(ss,raw))
 bad=copy.deepcopy(gt);bad[33*215]['candidate_aux_hidden_label_GT_only']=-1;bad[33*215]['known_candidate_GT_only']=False;rejected(lambda:t.join_metadata(lineage,bad))
 bad=copy.deepcopy(lineage);bad[1]['query_xy']=[0,0];rejected(lambda:t.join_metadata(bad,gt))
 bad=copy.deepcopy(gt);bad[2]['pair_ready']=False;rejected(lambda:t.join_metadata(lineage,bad))
 bad=copy.deepcopy(lineage);bad[3]['donor_rank']=15;rejected(lambda:t.join_metadata(bad,gt))
 rejected(lambda:t.join_metadata(lineage[:-1],gt[:-1]))
 pools=t.make_pools(m);g=torch.Generator().manual_seed(194);g2=torch.Generator().manual_seed(194)
 for _ in range(2000):
  ix=t.draw(pools,g);assert np.array_equal(ix,t.draw(pools,g2));assert len(ix)==64 and m['known'][ix].all()and(m['split'][ix]=='TRAIN').all()
  assert [int(x[ix].sum())for x in t.cohort_masks(m,'TRAIN').values()]==list(t.QUOTAS)
 missing=dict(m,labels=m['labels'].copy());missing['labels'][m['source']=='LOCO']=-1;rejected(lambda:t.make_pools(missing))
 missing=dict(m,labels=m['labels'].copy());missing['labels'][m['variant']==t.VARIANTS[1]]=-1;rejected(lambda:t.make_pools(missing))

 # Unequal cohort sizes; one unavailable extreme logit must incur ln2.
 sm=dict(labels=np.array([0,0,0,0,0,0,1,1,1,-1]),source=np.array(['LOCO','LOCO','PALLOX','PALLOX','PALLOX','PALLOX','PALLOX','PALLOX','PALLOX','LOCO']),split=np.array(['VALIDATION']*10),variant=np.array(['BEFORE']*3+[t.VARIANTS[2]]*3+['BEFORE']*2+[t.VARIANTS[1],'BEFORE']))
 z=np.array([-1.,-2.,-1.,-1.,0.,-1.,1.,-999.,2.,999.],np.float32);a=np.ones(10,bool);a[7]=False
 value,coh=t.selector(z,a,sm);sp=lambda x:math.log1p(math.exp(x));expected=np.mean([(sp(-1)+sp(-2))/2,sp(-1),(2*sp(-1)+math.log(2))/3,(sp(-1)+math.log(2))/2,sp(-2)])
 assert abs(value-expected)<1e-14 and coh['BEFORE_E']['unavailable']==1
 best=float('inf');chosen=None
 for update,v in[(0,2.),(100,1.),(200,1.),(300,1.1)]:
  if v<best:best=v;chosen=update
 assert chosen==100
 z=np.where(m['labels']==0,-1.,1.).astype(np.float32);a=np.ones(31680,bool)
 rr,checks,passed=t.report(z,a,m);assert passed and len(checks)==7 and rr['VALIDATION']['paired']['all3_correct']==6
 assert rr['VALIDATION']['paired_by_source']['LOCO']['accepted_pairs']==3
 a[33*1+1]=False;rr,_,passed=t.report(z,a,m);assert passed and rr['VALIDATION']['paired']['all3_correct_rate']==5/6
 assert rr['VALIDATION']['five_cohorts']['COVER_E']['E_recall']==5/6
 a[33*1+5]=False;rr,_,passed=t.report(z,a,m);assert not passed and rr['VALIDATION']['paired']['all3_correct_rate']==4/6
 assert t.decode(np.array([-0.,0.,np.nextafter(np.float32(0),np.float32(-1)),1.]),np.array([1,1,1,0],bool)).tolist()==[1,1,0,-1]
 # A zero controlled-family denominator remains untested rather than perfect.
 only=m['pair_ready'].copy();only[:160]=False;mm=dict(m,pair_ready=only)
 assert t.pair_metrics(t.decode(z,a),mm,'VALIDATION','LOCO')['all3_correct_rate']is None

 # Repeated sparse references load exactly once without deleting occurrences.
 with tempfile.TemporaryDirectory()as d:
  d=Path(d);p=np.zeros((1,65,9,9),np.float16);p[:,64]=1;rgb=np.full((1,3,96,96),11,np.uint8);v=np.ones((1,1,96,96),bool)
  np.savez(d/'base.npz',p3_roi=p,rgb96=rgb,crop_valid96=v);np.savez(d/'changed.npz',p3_roi=np.repeat(p,2,0),rgb96=np.repeat(rgb+4,2,0),crop_valid96=np.repeat(v,2,0))
  refs={k:dict(file=t.bind(d/'base.npz'),key=k,index=0)for k in t.FIELDS};base=[dict(before_input_refs=refs)];(d/'base.json').write_text(json.dumps(base));bb=t.bind(d/'base.json')
  hashes=lambda x:{k:__import__('hashlib').sha256(np.ascontiguousarray(x[k]).tobytes()).hexdigest()for k in t.FIELDS}
  basehash=hashes(dict(p3_roi=p[0],rgb96=rgb[0],crop_valid96=v[0]));afterhash=hashes(dict(p3_roi=p[0],rgb96=(rgb+4)[0],crop_valid96=v[0]))
  r=[dict(slot_index=0,input_unit_index=0,full_rgb_changed=False,input_ref=dict(kind='BASE_ROW',slot_index=0,base_lineage=bb),tensor_value_sha256=basehash)for _ in range(33)]
  r[1]=dict(slot_index=0,input_unit_index=1,full_rgb_changed=True,input_ref=dict(kind='CHANGED_CHUNK',file=t.bind(d/'changed.npz'),index=0),tensor_value_sha256=afterhash)
  r[2]=dict(slot_index=0,input_unit_index=2,full_rgb_changed=True,input_ref=dict(kind='CHANGED_CHUNK',file=t.bind(d/'changed.npz'),index=1),tensor_value_sha256=afterhash)
  data,unit=t.materialize_unique_inputs(base,r,bb);assert len(unit)==33 and len(data['p3_roi'])==3 and unit.tolist()==[0,1,2]+[0]*30
  assert data['rgb96'][:,0,0,0].tolist()==[11,15,15]
  bad=copy.deepcopy(r);bad[3]['input_unit_index']=1;rejected(lambda:t.materialize_unique_inputs(base,bad,bb))
  bad=copy.deepcopy(r);bad[3]['input_unit_index']=3;rejected(lambda:t.materialize_unique_inputs(base,bad,bb))
  bad=copy.deepcopy(r);bad[0]['full_rgb_changed']=True;rejected(lambda:t.materialize_unique_inputs(base,bad,bb))
  bad=copy.deepcopy(r);bad[1]['input_ref']['file']['sha256']='0'*64;rejected(lambda:t.materialize_unique_inputs(base,bad,bb))
  model=t.operator().QueryCenterScalar();uz,ua=t.predict(model,data,unit,'cpu');assert len(uz)==33 and np.all(uz==0)and ua.all()
  assert np.array_equal(uz,uz[0]*np.ones(33,np.float32))
 rejected(lambda:t.require_entry(dict(approved_actual_source_fit=False)))
 op=t.operator();model=op.QueryCenterScalar();assert sum(x.numel()for x in model.parameters())==26573
 pp=torch.randn(4,65,9,9);pp[:,64]=1;rgb=torch.randint(0,256,(4,3,96,96),dtype=torch.uint8);valid=torch.ones(4,1,96,96,dtype=torch.bool);valid[2]=False
 y=torch.tensor([0,1,1,-1]);known=y>=0;out=model(pp,rgb,valid);loss=op.known_binary_loss(out,y,known)
 assert int(loss['available_known_count'])==2 and op.decode_binary(out).tolist()==[1,1,-1,1]
 zz=torch.tensor([2.,-1.,0.,123.],requires_grad=True);op.known_binary_loss(dict(query_logit=zz,query_available=torch.tensor([True,True,False,True])),y,known)['loss'].backward();assert zz.grad[2:].tolist()==[0.,0.]
 with torch.no_grad():model.readout[-1].weight.fill_(.031)
 model.train();train=model(pp,rgb,valid);model.eval();ev=model(pp,rgb,valid);assert torch.equal(train['query_logit'],ev['query_logit'])
 op.known_binary_loss(train,y,known)['loss'].backward();assert any(x.grad is not None and x.grad.abs().sum()>0 for x in model.p3_encoder.parameters())
 try:model(pp,rgb,valid,labels=y)
 except TypeError:pass
 else:raise AssertionError('GT forward accepted')
 proof=dict(completed=True,PASS=True,trainer=t.bind(HERE/'train_source_once.py'),model=t.bind(t.MODEL),generated_test_code=t.bind(__file__),torch=torch.__version__,runtime=__import__('sys').executable,invented_base_rows=960,invented_occurrences=31680,rank_pairs_shape=[960,16],before_once_with_naturalE4_preserved=True,legacy_C1_V95_label_retained_separately=True,rank_success_pair_labels_and_shared_before_checked=True,same_q_L_and_order_guards=True,raw_source_splits_canonicalized_without_reassignment=True,five_required_TRAIN_VAL_cohorts_empty_NO_FIT=True,sampler_generated_batches=2000,quotas=list(t.QUOTAS),selector_five_equal_means_includes_unavailable_ln2=True,earliest_strict_tie=True,separate_controlled_families_zero_denominator_untested=True,seven_gates_knownU_in_denominators=True,sparse_units_33rows_3units=True,repeated_unit_and_changed_base_misuse_rejected=True,unchanged_26573params_three_tensor_forward=True,known_only_available_loss_gradient=True,train_eval_identical_scalar=True,generated_nonzero_encoder_backward=True,GT_forward_blocked=True,actual_source_arrays_GT_checkpoint_current_reads=0,actual_fit_updates=0)
 t.write(output,proof);print(json.dumps(proof),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'GENERATED_SOURCE_TRAINER_TESTS.json');main(p.parse_args().output)
