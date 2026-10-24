"""Invented minimal C14 schema only. No current/source packet/weights/model forward."""
from pathlib import Path
import argparse,json
import numpy as np
import frozen_head_current as m
import evaluate_saved_C14_schema_repair as repair
from test_generated_current import fixture,csvstyle

def test():
 core,extra=m.pure_core();prior,truth,manual,_,old,raw=fixture(core)
 raw['valid'][100]=raw['query_available'][100]=raw['context_available'][100]=True
 z=np.array(raw['query_logit'],np.float32);pv=np.asarray(1/(1+np.exp(-(-z.astype(np.float64)))),np.float32)
 expected=csvstyle(core.decode_rows(prior,z,pv,raw['query_available']))
 keys=('id','native_corner','query_x','query_y','roi_side','valid','raw_query_aux_hidden_logit','raw_p_visible_FP32','authoritative_decision')
 minimal=[{k:r[k]for k in keys}for r in expected]
 reconstructed=repair.enrich_C14_minimal(core,prior,minimal)
 assert reconstructed==expected,'Original geometry/output restoration differs'
 base=m.module(m.METRIC,'generated_schema_repair_math')
 reproduced=False
 try:m.calculate_tables(core,extra,base,truth,prior,expected,old,expected,manual,minimal)
 except KeyError as e:assert e.args==('geometry_self_hidden',);reproduced=True
 assert reproduced
 a=m.calculate_tables(core,extra,base,truth,prior,expected,old,expected,manual,expected)
 b=m.calculate_tables(core,extra,base,truth,prior,expected,old,expected,manual,reconstructed)
 assert a==b
 c14hist=csvstyle([r for r in a['metrics']if r['model']in(m.C14,m.C14_GEO)])
 assert core.historical_replay(b['metrics'],c14hist,{m.C14,m.C14_GEO})==20
 assert b['availability']['new_total_U']==45 and b['availability']['extra_operator_U']==0
 rejected=0
 for key,value in [('id','BAD'),('native_corner','99'),('query_x','9999'),('roi_side','9999'),('authoritative_decision','BAD')]:
  bad=[dict(r)for r in minimal];bad[0][key]=value
  try:repair.enrich_C14_minimal(core,prior,bad)
  except ValueError:rejected+=1
  else:raise AssertionError('Bad minimal identity/sign accepted')
 return dict(completed=True,PASS=True,adapter=m.bind(repair.__file__),test=m.bind(__file__),original_operator=m.bind(m.__file__),invented_slots=1784,minimal_C14_schema_reproduced_original_geometry_KeyError=True,all_original_C14_z_probability_q_L_availability_preserved=True,enriched_geometry_raw_geo_and_U_exact=True,all_tables_equal_to_pre_enriched_fixture=True,C14_historical_rows_exact=20,original45U_extra0=True,bad_identity_q_L_sign_rejections=rejected,actual_current_source_arrays_GT_weights_or_model_forward=0,new_infer_fit_PnP=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=test();m.write(a.output,r);print(json.dumps(r))
