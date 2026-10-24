"""Invented labels and saved arrays only, no actual source/model reader."""
from pathlib import Path
import hashlib,importlib.util,json,tempfile,math
import numpy as np
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('own_saved_auditor',H/'audit_completed_source_fit.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def main():
 assert a.same_binding(dict(path='/invented/checkpoint.pt',sha256='a'*64,bytes=123),dict(path='/invented/checkpoint.pt',sha256='a'*64))
 assert not a.same_binding(dict(path='/invented/checkpoint.pt',sha256='a'*64),dict(path='/invented/checkpoint.pt',sha256='b'*64))
 line=[];gt=[]
 for i in range(960):
  source='LOCO'if i<160 else'PALLOX';raw=('TRAIN'if i%2==0 else'VALIDATION')if source=='LOCO'else('TRAIN_LAB'if i%2==0 else'VAL_HALL')
  y=0 if i<6 or 160<=i<215 else 1 if 215<=i<219 else -1
  for j in range(33):
   k=None if j==0 else(j-1)//2;v='BEFORE'if j==0 else'OPAQUE_COVER'if j%2 else'CUE_PRESERVING_SHAM';oi=33*i+j;ready=i in[0,1,160,161]and k in[0,2,15];label=y if j==0 else(1 if j%2 else 0)if ready else-1
   line.append(dict(occurrence_index=oi,occurrence_id=f'GENERATED:{oi}',slot_index=i,donor_rank=k,variant=v,source=source,split=raw,query_xy=[32.,32.],L=16.,input_unit_index=i,input_id=f'FRAME{i//8}',native_slot_logical_only=i%8,image_sha256='a'*64,original_image_sha256='a'*64,input_facts=dict(operator_query_available=True)))
   gt.append(dict(occurrence_index=oi,occurrence_id=f'GENERATED:{oi}',slot_index=i,base_row_index=i,donor_rank=k,variant=v,candidate_aux_hidden_label_GT_only=label,known_candidate_GT_only=label>=0,query_anchored_before_label=y,original_C1_V95_label=-1,pair_ready=ready,native_expected_part_state='UNKNOWN',human_gold=False))
 m=a.metadata(line,gt);assert a.counts(m)['TRAIN']['accepted_pairs']==a.counts(m)['VALIDATION']['accepted_pairs']==6
 z=np.where(m['y']==0,-1.,1.).astype(np.float32);av=np.ones(31680,bool)
 result,gates,passed=a.all_metrics(z,av,m);assert passed and len(gates)==7 and result['VALIDATION']['paired']['all3_correct']==6
 av[34]=False;line[34]['input_facts']['operator_query_available']=False;z[34]=0
 result,gates,passed=a.all_metrics(z,av,m);assert passed and result['VALIDATION']['paired']['all3_correct']==5 and result['VALIDATION']['five_cohorts']['COVER_E']['matrix']==[[0,0,0],[0,5,1]]
 nll=(np.logaddexp(np.float32(0),z)-np.where(m['y']>=0,m['y'],0)*z).astype(np.float32)
 assert a.predictions(np.array([-0.,0.,-1e-7,1e-7]),np.array([1,1,1,0],bool)).tolist()==[1,1,0,-1]
 with tempfile.TemporaryDirectory()as td:
  path=Path(td)/'generated.npz'
  np.savez_compressed(path,query_logit=z,query_available=av,prediction=a.predictions(z,av),scalar_target_D0_E1_Uminus1=m['y'],candidate_known=m['known'],NLL_FP32=nll,source=m['source'],split=m['split'],original_split=m['raw'],variant=m['variant'],occurrence_index=np.arange(31680),base_row_index=np.arange(31680)//33,donor_rank=np.array([-1 if r['donor_rank']is None else r['donor_rank']for r in line],np.int8),input_unit_index=np.array([r['input_unit_index']for r in line]),occurrence_id=np.array([r['occurrence_id']for r in line]),input_id=np.array([r['input_id']for r in line]),native_slot_logical_only=np.array([r['native_slot_logical_only']for r in line]),image_sha256=np.array([r['image_sha256']for r in line]),original_image_sha256=np.array([r['original_image_sha256']for r in line]),query_xy_raw=np.tile([32.,32.],(31680,1)),L_raw=np.full(31680,16.),accepted_pair=m['ready'])
  r,_,_=a.audit_scores(path,line,m);a.equal(r['metrics'],result)
  assert r['unknown_NLL_excluded']==int((~m['known']).sum())and r['unavailable']==1
 # Explicit2x3/unknown oracle, independent of the full fixture structure.
 mm=dict(y=np.array([0,0,0,1,1,1,-1]),known=np.array([1,1,1,1,1,1,0],bool));p=np.array([0,1,-1,0,1,-1,1])
 c=a.cm(p,mm,np.ones(7,bool));assert c['matrix']==[[1,1,1],[1,1,1]]and c['D_recall']==c['E_recall']==1/3 and c['known_U_rate']==2/6 and c['unknown_GT_N']==1
 history=[dict(update=u,value=.6931471805599453)for u in range(0,2001,100)];assert min(history,key=lambda x:x['value'])['update']==0
 proof=dict(completed=True,PASS=True,auditor=a.bind(H/'audit_completed_source_fit.py'),generated_code=a.bind(__file__),all31680_generated_saved_array_replay=True,naturalE4_and_old_labels_separate=True,raw0_and_invalidU=True,unknownNLLexcluded=True,perrank_U_fails_pair=True,all7_gates_analytic=True,checkpoint21_exacttie_earliest0=True,no_actual_source_or_model_outputs_read=True,new_model_optimizer_forward_current_PnP=0)
 with(H/'GENERATED_AUDITOR_TESTS.json').open('x')as f:json.dump(proof,f,indent=2);f.write('\n')
 print(json.dumps(proof))
if __name__=='__main__':main()
