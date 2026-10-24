"""Actual HEAD_ONLY header schema with invented1784 values only; no real current I/O."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,tempfile
import numpy as np
import torch
import frozen_head_current as m

def csvstyle(rs):return [{k:(''if v is None else str(v))for k,v in r.items()}for r in rs]
def fixture(core):
 p=m.C11/'current_evaluation_preparation/test_generated_saved_evaluator.py';fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='fixture')
 ns=dict(np=np,csv_style=csvstyle,c=SimpleNamespace(BOX_MODE='FROZEN_INITIAL_Q8_UNCLIPPED_UNEXPANDED_MINMAX'));exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p)+'::fixture_only','exec'),ns);return ns['fixture'](core)
def minimal(rows):return [{k:r[k]for k in m.MINIMAL_SCORE_FIELDS}for r in rows]
def test():
 torch.set_num_threads(1);core,extra=m.pure_core();prior,truth,manual,_,old,raw=fixture(core)
 raw['valid'][100]=raw['query_available'][100]=raw['context_available'][100]=True
 x=np.zeros((1784,1928),np.float32);x[:,0]=raw['query_logit'];x[:,1927]=raw['context_Q_inside']
 packet=dict(features1928=x,query_available=raw['query_available'],prior_valid=raw['prior_valid'],valid=raw['valid'],context_available=raw['context_available'],local_central4_available=raw['local_central4_available'],context_Q_inside=raw['context_Q_inside'],query_xy=raw['original_query_xy'].astype(np.float32),side_L=raw['original_side_L'].astype(np.float32),original_query_xy=raw['original_query_xy'],original_side_L=raw['original_side_L'],id=raw['id'],native_corner=raw['native_corner'])
 infer=next(n for n in ast.parse(Path(m.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='infer')
 header=next(tuple(k.arg for k in n.keywords)for n in ast.walk(infer)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='dict'and n.keywords and n.keywords[0].arg=='id')
 assert header==m.MINIMAL_SCORE_FIELDS and len(header)==9
 with tempfile.TemporaryDirectory()as td:
  path=Path(td)/'invented.npz';np.savez_compressed(path,**packet,forbidden_logits=np.array([{}],dtype=object));a=m.load_saved_inputs(path)
  assert set(a)==set(m.WHITELIST)and(~a['valid']).sum()==45
  bad=dict(packet);bad['context_available']=a['context_available'].copy();bad['context_available'][100]=False;bad['valid']=a['valid'].copy();bad['valid'][100]=False;bad['query_available']=bad['valid'];bp=Path(td)/'extraU.npz';np.savez_compressed(bp,**bad)
  try:m.load_saved_inputs(bp)
  except ValueError:pass
  else:raise AssertionError('ExtraU accepted')
  class Stub:
   def __init__(self):self.calls=0
   def __call__(self,t):self.calls+=1;return t[:,0]
  stub=Stub();_,z,pv=m.predict_generated_or_actual(stub,a['features1928'],a['valid'],torch,'cpu');assert stub.calls==2
  new=csvstyle(core.decode_rows(prior,z,pv,a['valid']));c11=csvstyle(core.decode_rows(prior,-z,pv,a['valid']))
  c14_expected=csvstyle(core.decode_rows(prior,z,pv,a['valid']));c15pv=torch.sigmoid(torch.as_tensor(z)).numpy();c15_expected=csvstyle(core.decode_rows(prior,-z,c15pv,a['valid']))
  c14p=Path(td)/'C14_HEAD_ONLY_SCORES.csv';c15p=Path(td)/'C15_HEAD_ONLY_SCORES.csv';core.writecsv(c14p,minimal(c14_expected));core.writecsv(c15p,minimal(c15_expected));c14=core.readcsv(c14p);c15=core.readcsv(c15p)
  assert tuple(c14[0])==tuple(c15[0])==header and all('geometry_self_hidden'not in r for r in c14+c15)
  assert m.enrich_minimal_baseline(core,prior,c14)==c14_expected and m.enrich_minimal_baseline(core,prior,c15)==c15_expected
  base=m.module(m.METRIC,'generated_c16_metric');tables=m.calculate_tables(core,extra,base,truth,prior,new,old,c11,manual,c14,c15)
  assert tables['availability']['new_total_U']==45 and tables['availability']['extra_operator_U']==0 and {r['N']for r in tables['manual_near_D']}=={153,223}
  assert len(tables['D_vs_external_E_2x3'])==48
  # Equal saved raw values must yield equal C14/C16 metrics; C15 remains independently decoded.
  metrickey=lambda r:(r['cohort'],r['group'],r['task'])
  nidx={metrickey(r):r for r in tables['metrics']if r['model']==m.NEW}
  for r in tables['metrics']:
   if r['model']==m.C14:
    for k in ('TP','FN_decided','FP','TN','U_visible','U_occluded'):assert r[k]==nidx[metrickey(r)][k]
  hist=csvstyle([r for r in tables['metrics']if r['model']not in(m.NEW,m.NEW_GEO)])
  assert core.historical_replay(tables['metrics'],hist,set(m.MODELS)-{m.NEW,m.NEW_GEO})==100
  assert new[0]['authoritative_decision']=='D'and new[0]['p_visible']=='0.5'and new[1]['authoritative_decision']==new[2]['authoritative_decision']=='E'
  assert all(r['authoritative_decision']=='UNKNOWN'and r['geometry_self_hidden_if_valid']==''for r in new if r['valid']=='False')
  bad=[dict(r)for r in c14];bad[0]['query_x']='9999'
  try:m.enrich_minimal_baseline(core,prior,bad)
  except ValueError:pass
  else:raise AssertionError('Changed baselineq accepted')
 desc=lambda s:dict(path='GENERATED_'+s,sha256='0'*64)
 e=dict(checkpoint=desc('cp'),source_trainer=desc('trainer'),source_freeze=desc('freeze'),source_fit_complete=desc('fit'),source_review=desc('review'),source_replay_certificate=desc('replay'),descriptive_evaluation_only=True,model_promotion_authorized=False)
 freeze=dict(completed=True,design=m.bind(m.DESIGN),threshold_logit=0,selected_update=8000,checkpoint=e['checkpoint'],trainer=e['source_trainer'],head_class_code=m.bind(m.HEAD),current_used_for_fit_or_selection=False,PnP=0)
 fit=dict(completed=True,optimizer_updates=8000,total_updates=8000,replayed_C15_updates=1000,continued_updates=7000,replay_certificate=e['source_replay_certificate'],model_freeze=e['source_freeze'])
 gates=dict(completed=True,all_gates_pass=True,gates={str(i):dict(PASS=True)for i in range(15)});review=dict(completed=True,PASS=True,fit=e['source_fit_complete'])
 replay=dict(completed=True,PASS=True,replayed_updates=1000,**{k:True for k in ('batches_exact','losses_exact','step1000_parameters_exact','step1000_scores_exact','step1000_fourteen_BCE_exact','optimizer_and_RNG_not_reset_at1000','history0_to1000_exact')})
 m.source_guard(e,freeze,fit,gates,review,replay);m.source_guard(e,freeze,fit,dict(gates,all_gates_pass=False,gates={k:dict(PASS=False)for k in gates['gates']}),review,replay)
 rejects=0
 for kw in ['batches_exact','losses_exact','step1000_parameters_exact','step1000_scores_exact','step1000_fourteen_BCE_exact','optimizer_and_RNG_not_reset_at1000','history0_to1000_exact']:
  try:m.source_guard(e,freeze,fit,gates,review,dict(replay,**{kw:False}))
  except ValueError:rejects+=1
  else:raise AssertionError('Inexact replay accepted')
 for args in [(e,freeze,dict(fit,continued_updates=6000),gates,review,replay),(e,dict(freeze,selected_update=8100),fit,gates,review,replay),(dict(e,model_promotion_authorized=True),freeze,fit,gates,review,replay)]:
  try:m.source_guard(*args)
  except ValueError:rejects+=1
  else:raise AssertionError('Bad duration/promotion accepted')
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'UNAPPROVED.json';m.write(p,dict(approved=False))
  try:m.authorize(p,'INFER')
  except ValueError:rejects+=1
  else:raise AssertionError('Unapproved entry accepted')
 return dict(completed=True,PASS=True,code=m.bind(m.__file__),test_code=m.bind(__file__),runtime=torch.__version__,actual_minimal_header_from_infer_AST=list(header),invented_slots=1784,C14_C15_minimal_CSV_roundtrip=True,both_geometry_restored_before_score_index=True,baseline_z_pv_qL_availability_unchanged=True,original45U_extra0=True,rawsign_tiny_signedzero_probabilityhalf=True,historical_rows_replayed=100,manual_N=[153,223],source_replay_duration_entry_rejections=rejects,failed15_descriptive_only_accepted=True,actual_source_current_body_weight_GT_or_model_reads=0,new_encoder_RGB_initializer_fit_PnP=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=test();m.write(a.output,r);print(r)
