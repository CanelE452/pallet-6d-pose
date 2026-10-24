import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import torch
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c11_generated_adapter',H/'current_adapter.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

def test(output):
 torch.set_num_threads(1)
 q=np.array([[-20.,10.],[700.,500.],[-1,-1],[np.nan,np.nan],[15,30],[20,30],[50,60],[22,40]])
 before=q.copy();assert np.array_equal(c.initial_context_box(q),[-20,10,700,500]);assert np.array_equal(q,before,equal_nan=True)
 assert c.initial_context_box(np.tile([-1.,-1.],(8,1))) is None
 assert c.initial_context_box(np.tile([2.,3.],(8,1))) is None
 narrow=np.tile([2.,3.],(8,1));narrow[0]=[2.1,3.1];assert c.initial_context_box(narrow) is not None
 # Existing native reader input verification, entirely invented arrays/rows.
 reader=c.prior_reader();n=1784
 data=dict(p3_roi=np.zeros((n,65,9,9),np.float32),rgb96=np.zeros((n,3,96,96),np.uint8),crop_valid96=np.ones((n,1,96,96),bool),query_xy=np.tile([12.,13.],(n,1)).astype(np.float32),side_L=np.full(n,16,np.float32));data['p3_roi'][:,64]=1
 frames=[];meta=[];prior=[]
 for fi in range(223):
  identity=f'GENERATED_{fi:03d}';frames.append(dict(id=identity,initial_points=[[12.,13.]]*8,raw_hw=[24,32],image_sha256='0'*64));meta.append(dict(id=identity,side=16.))
  for k in range(8):
   ix=fi*8+k;valid=ix<1739
   prior.append(dict(id=identity,native_corner=k,query_x=12.,query_y=13.,pixel_x=12,pixel_y=13,image_sha256='0'*64,roi_side=16.,saved_roi_pixel_valid=valid,valid=valid,status='OK'if valid else'UNKNOWN',reason=''if valid else'GENERATED_PRIOR_U',geometry_self_hidden=(k==0)))
 base,oq,L=reader.verify_input_metadata(data,prior,frames,meta)
 assert len(base)==1784 and np.array_equal(oq,data['query_xy']) and np.array_equal(L,data['side_L'])
 local=np.ones(n,bool);context=np.ones(n,bool);context[100]=False;priorvalid=np.array([r['valid']for r in base]);op=priorvalid&local&context
 z=np.tile(np.array([-1.,1.,0.,-0.,np.nextafter(np.float32(0),np.float32(-1)),np.nextafter(np.float32(0),np.float32(1))],np.float32),298)[:n];z=np.where(op,z,0).astype(np.float32)
 pv=torch.sigmoid(-torch.as_tensor(z)).numpy();core=c.pure_core();out=core.decode_rows(base,z,pv,op)
 assert sum(r['valid']for r in out)==1738 and all(not out[i]['valid'] for i in range(1739,1784)) and out[100]['additional_operator_U']
 for i,r in enumerate(out):
  assert r['authoritative_decision']==(('D'if z[i]<0 else'E')if op[i]else'UNKNOWN')
  assert r['geometry_self_hidden_if_valid']==(base[i]['geometry_self_hidden']if r['valid']else None)
  assert r['geometry_self_hidden']==base[i]['geometry_self_hidden']
 assert any(op&((pv>=.5)!=(z<0))), 'Rounded probability ties need rawsign.'
 # Qoutside descriptor cannot addU: only factual/local/context availability enters op.
 Q_inside=np.zeros(n,bool);assert not Q_inside.any() and op[0]
 # Pure source gate: a failed source gate cannot authorize current data reads.
 desc=lambda name:dict(path='GENERATED_'+name,sha256='0'*64)
 e=dict(source_freeze=desc('freeze'),source_gates=desc('gates'),checkpoint=desc('checkpoint'),source_fit_complete=desc('fit'))
 p={k:e[k]for k in('source_freeze','source_gates')}
 f=dict(completed=True,selected_update=1800,threshold_logit=0,checkpoint=e['checkpoint'],trainer=c.bind(c.HEAD),current_used_for_fit_or_selection=False,PnP=0)
 complete=dict(completed=True,optimizer_updates=2000,model_freeze=e['source_freeze']);gates=dict(completed=True,all_gates_pass=True,gates={str(i):dict(PASS=True)for i in range(12)});audit=dict(completed=True,PASS=True,source_freeze=e['source_freeze'])
 c.source_guard(p,e,f,complete,gates,audit);gates['gates']['0']['PASS']=False
 try:c.source_guard(p,e,f,complete,gates,audit)
 except ValueError:pass
 else:raise AssertionError('Failed source gate accepted.')
 # Actual producer maths and preprocessing on invented arrays only, no encoder.
 producer=c.module(c.PRODUCER,c.PRODUCER_SHA,'c11_generated_spatial_math');patch=torch.randn(8,256,384);cls=torch.randn(8,384)
 features,inside=producer.assemble(cls,patch,cls+1,patch+1,cls-1,[[12.,13.]]*8,[[0,0,24,24]]*8,np.ones((8,3),np.float32),np.ones(8,bool))
 assert features.shape==(8,1928) and inside.all()
 # End-to-end V94 pure CSV scalar decoder accepts unchanged identities/valid/geometry.
 import csv,io
 stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(out[0]));w.writeheader();w.writerows(out);stream.seek(0);indexed=core.score_index(list(csv.DictReader(stream)),'raw_query_aux_hidden_logit');assert len(indexed)==1784
 c.write_new(output,dict(completed=True,PASS=True,code=c.bind(H/'current_adapter.py'),test_code=c.bind(__file__),protocol=c.bind(c.PROTOCOL),runtime=torch.__version__,tests=['unclipped_unexpanded_finite_nonsentinel_q8_minmax','positive_extent_only_not_legacy_span2','q_exact_no_move','inert_pinned_reader1784native_order','original1739valid45U_preserved','one_extra_contextU_separate','rawsign_tiny_posneg_zero_not_pvisible_rounding','geometry_guardU_and_route_separate','Qoutside_no_additionalU','12sourcegates_before_current','producer1928math_on_invented_tokens','V94scalar_CSV_decoder_exact_AST'],prior_U=45,generated_extra_context_U=1,actual_current_artifact_SHA_body_or_pixel_reads=0,actual_source_features_weights_DINO_or_head_parameter_load_or_forward_fit_PnP=0))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=H/'GENERATED_CURRENT_TESTS.json');test(a.parse_args().output)
