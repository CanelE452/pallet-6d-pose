"""Invented saved-feature packet and score metadata only, no actual current I/O."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,tempfile
import numpy as np
import torch
import frozen_head_current as m

def fixture(core):
 p=m.C11/'current_evaluation_preparation/test_generated_saved_evaluator.py'
 fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='fixture')
 ns=dict(np=np,c=SimpleNamespace(BOX_MODE='FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX'));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p)+'::generated_fixture_only','exec'),ns)
 return ns['fixture'](core)
def csvstyle(rs):return[{k:(''if v is None else str(v))for k,v in r.items()}for r in rs]

def test():
 torch.set_num_threads(1);core,extra=m.pure_core();prior,truth,manual,_,old,raw=fixture(core)
 raw['valid'][100]=raw['query_available'][100]=raw['context_available'][100]=True
 x=np.zeros((1784,1928),np.float32);x[:,0]=raw['query_logit'];x[:,1927]=raw['context_Q_inside']
 packet=dict(features1928=x,query_available=raw['query_available'],prior_valid=raw['prior_valid'],valid=raw['valid'],context_available=raw['context_available'],local_central4_available=raw['local_central4_available'],context_Q_inside=raw['context_Q_inside'],query_xy=raw['original_query_xy'].astype(np.float32),side_L=raw['original_side_L'].astype(np.float32),original_query_xy=raw['original_query_xy'],original_side_L=raw['original_side_L'],id=raw['id'],native_corner=raw['native_corner'])
 with tempfile.TemporaryDirectory()as td:
  path=Path(td)/'generated_only.npz';np.savez_compressed(path,**packet,forbidden_old_logits=np.array([{}],dtype=object))
  a=m.load_saved_inputs(path);assert set(a)==set(m.WHITELIST)and(~a['valid']).sum()==45
  bad=dict(packet);bad['context_available']=a['context_available'].copy();bad['context_available'][100]=False
  bad['valid']=a['valid'].copy();bad['valid'][100]=False;bad['query_available']=bad['valid'];bp=Path(td)/'extraU.npz';np.savez_compressed(bp,**bad)
  try:m.load_saved_inputs(bp)
  except ValueError:pass
  else:raise AssertionError('Extra U accepted')
 class Stub:
  def __init__(self):self.calls=0
  def __call__(self,tensor):self.calls+=1;assert tensor.shape==(1784,1928);return tensor[:,0]
 stub=Stub();unmasked,z,pv=m.predict_generated_or_actual(stub,a['features1928'],a['valid'],torch,'cpu');assert stub.calls==2 and z.shape==(1784,)
 newrows=csvstyle(core.decode_rows(prior,z,pv,a['valid']));c11rows=csvstyle(core.decode_rows(prior,-z,pv,a['valid']))
 assert newrows[0]['authoritative_decision']=='D'and newrows[0]['p_visible']=='0.5'
 assert newrows[1]['authoritative_decision']==newrows[2]['authoritative_decision']=='E'
 assert all(r['authoritative_decision']=='UNKNOWN'and r['geometry_self_hidden_if_valid']==''for r in newrows if r['valid']=='False')
 base=m.module(m.METRIC,'generated_c12_metric');tables=m.calculate_tables(core,extra,base,truth,prior,newrows,old,c11rows,manual)
 assert tables['availability']['new_total_U']==45 and tables['availability']['extra_operator_U']==0
 assert {r['N']for r in tables['manual_near_D']}=={153,223}and len(tables['D_vs_external_E_2x3'])==32
 # C11 is resolved through saved baseline rows, not silently sent to missing probabilities.
 for r in tables['metrics']:
  if r['model']==m.BASE:assert r['other_unresolved_visible']+r['other_unresolved_occluded']<=45
 hist=csvstyle([r for r in tables['metrics']if r['model']in('v61','legacy',m.OLD,m.OLD_GEO,m.BASE,m.BASE_GEO)])
 assert core.historical_replay(tables['metrics'],hist,{'v61','legacy',m.OLD,m.OLD_GEO,m.BASE,m.BASE_GEO})==60
 # The exact source guard rejects failed gates/reviews before current file hashing.
 desc=lambda s:dict(path='GENERATED_'+s,sha256='0'*64)
 e=dict(checkpoint=desc('cp'),source_trainer=desc('trainer'),source_freeze=desc('freeze'),source_fit_complete=desc('fit'),source_review=desc('review'))
 freeze=dict(completed=True,threshold_logit=0,selected_update=1200,checkpoint=e['checkpoint'],trainer=e['source_trainer'],head_class_code=m.bind(m.HEAD),current_used_for_fit_or_selection=False,PnP=0)
 fit=dict(completed=True,optimizer_updates=2000,model_freeze=e['source_freeze']);gates=dict(completed=True,all_gates_pass=True,gates={str(i):dict(PASS=True)for i in range(15)})
 review=dict(completed=True,PASS=True,fit=e['source_fit_complete']);ad=dict(completed=True,frozen_model_admitted_for_once_current_comparison=True,source_all15_gates_pass=True,source_model_freeze=e['source_freeze'],source_saved_review=e['source_review'])
 m.source_guard(e,freeze,fit,gates,review,ad);rejections=0
 for i in range(15):
  bad=dict(gates,gates={k:dict(v)for k,v in gates['gates'].items()});bad['gates'][str(i)]['PASS']=False
  try:m.source_guard(e,freeze,fit,bad,review,ad)
  except ValueError:rejections+=1
  else:raise AssertionError('Failed source gate accepted')
 try:m.source_guard(e,freeze,fit,gates,dict(review,PASS=False),ad)
 except ValueError:rejections+=1
 else:raise AssertionError('Failed independent review accepted')
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'UNAPPROVED.json';m.write(p,{'approved':False})
  try:m.authorize(p,'INFER')
  except ValueError:rejections+=1
  else:raise AssertionError('Unapproved entry accepted')
 return dict(completed=True,PASS=True,code=m.bind(m.__file__),test_code=m.bind(__file__),runtime=torch.__version__,invented_slots=1784,whitelist=list(m.WHITELIST),forbidden_old_logits_not_deserialized=True,features_only_stub2calls=True,prior45U_and_extra0=True,extraU_rejected=True,rawsign_tiny_signedzero_probabilityhalf=True,geometry_valid_only_and_U_first=True,exact_pinned_cohorts_manual153_223_and_baselines=True,historical_rows_replayed=60,source_gate_entry_rejections=rejections,new_encoder_RGB_initializer_calls=0,actual_source_current_packet_score_GT_checkpoint_or_model_reads=0,actual_head_forward_fit_V112_PnP=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=test();m.write(a.output,result);print(result)
