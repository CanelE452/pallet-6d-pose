"""Actual V2 metadata-comparison helper, authored scalars/arrays only."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
H=Path(__file__).parent
sp=importlib.util.spec_from_file_location('v2',H/'evaluate_saved_source_point_holdout.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
checks=[]
for name,stored,original,key,wanted in [
 ('nullq_pair',np.array([np.nan,np.nan],np.float64),None,'query_xy',True),
 ('nullq_not_scalar',np.array(np.nan,np.float64),None,'query_xy',False),
 ('nullL_scalar',np.array(np.nan,np.float64),None,'L',True),
 ('nullL_not_vector',np.array([np.nan],np.float64),None,'L',False),
 ('finiteq_exact',np.array([10.,20.],np.float64),[10.,20.],'query_xy',True),
 ('finiteq_mismatch',np.array([10.,20.],np.float64),[10.,20.0000000001],'query_xy',False),
 ('finiteL_exact',np.array(96.,np.float64),96.,'L',True),
 ('finiteL_mismatch',np.array(96.,np.float64),96.0000000001,'L',False),
 ('unavailable_kept',np.bool_(False),False,'original_query_available',True),
 ('unavailable_not_true',np.bool_(False),True,'original_query_available',False),
 ('native_exact',np.int8(7),7,'native_slot',True),
 ('native_mismatch',np.int8(7),6,'native_slot',False)]:
 m.need(bool(m.original_metadata_value_equal(stored,original,key))is wanted,name);checks.append(name)
# Full original unavailable META shape exercises the helper used in the production loop.
original=dict(row_id='TOY:native7',original_global_index=7,native_slot=7,query_xy=None,L=None,original_query_available=False)
stored=dict(row_id=np.str_('TOY:native7'),original_global_index=np.int64(7),native_slot=np.int8(7),query_xy=np.array([np.nan,np.nan],np.float64),L=np.array(np.nan,np.float64),original_query_available=np.bool_(False))
m.need(all(m.original_metadata_value_equal(stored[k],v,k)for k,v in original.items()),'All6 unavailable original metadata exactly preserved');checks.append('allMETA6_unavailable_nulls')
result=dict(completed=True,PASS=True,code_sha256=hashlib.sha256((H/'evaluate_saved_source_point_holdout.py').read_bytes()).hexdigest(),checks=checks,actual_GT_prediction_payload_or_weight_reads=0)
with(H/'NULL_IDENTITY_INVENTED_CHECKS.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
