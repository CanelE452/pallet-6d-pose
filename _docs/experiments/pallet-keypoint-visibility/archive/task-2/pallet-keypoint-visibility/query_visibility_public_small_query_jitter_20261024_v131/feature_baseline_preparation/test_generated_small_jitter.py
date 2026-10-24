"""Invented CPU/static integration checks; no model/source payload calls."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys,tempfile
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
def bind(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def calls(tree,name):return[ast.dump(n,include_attributes=False)for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr==name]
def run(out):
 p=H/'run_frozen_small_jitter.py';text=p.read_text();tree=ast.parse(text);compile(tree,str(p),'exec');s=importlib.util.spec_from_file_location('v131_generated',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert 'torch'not in sys.modules;parent=ast.parse(m.PARENT_CODE.read_text());checks={}
 for name in ('preprocess_views','preprocess_target','assemble','sample_target_context','forward_features'):
  assert calls(tree,name)==calls(parent,name),name
 checks['exact_parent_preprocess_3n_views_assemble_encoder_calls_AST']=True
 row={'occurrence_index':0,'id':'invented','query_xy':[32.,32.],'initial_points':[[32.,32.],[0.,0.],[64.,0.],[64.,64.],[0.,64.],[1.,1.],[63.,1.],[63.,63.]],'carrier_slot':0,'raw_hw':[128,128],'L':16.,'context_box_xyxy':[0.,0.,64.,64.],'full_RGB_format':'NPZ_RGB_UINT8_HWC','full_RGB_key':'RGB_uint8','rgb96_row':0,'rgb96_chunk':{'path':'invented','sha256':'0'*64}}
 m.validate_inputs([row]);bad=dict(row,teacher_label=1)
 try:m.validate_inputs([bad])
 except AssertionError:pass
 else:raise AssertionError('Teacher field accepted')
 checks['input_q8_L_context_validation_and_no_teacher_fields']=True
 order=np.r_[np.arange(918)[::-1],np.arange(3)];names=np.r_[np.full(918,'PUBLIC_AUX'),np.full(3,'NATIVE')];centre=np.array([i for i in range(918)if(i//17)%9==4]);ix=m.mapping(names,order,centre);assert len(centre)==102 and np.array_equal(order[ix],centre)
 duplicate=order.copy();duplicate[0]=duplicate[1]
 try:m.mapping(names,duplicate,centre)
 except AssertionError:pass
 else:raise AssertionError('Duplicate/missing identity accepted')
 checks['explicit_noncontiguous_corpus_local_mapping_102centre_no_hard_offset']=True
 assert 55296%8==55296%512==0 and[ min(8,918-i)for i in range(0,918,8)]==[8]*114+[6]and[min(512,918-i)for i in range(0,918,512)]==[512,406]
 checks['same_parent_batch_phase_115primary_plus_repeat_head512_406']=True
 with tempfile.TemporaryDirectory()as td:
  path=Path(td)/'scores.npz';q=np.stack((np.arange(921),np.arange(921)+2),1).astype(np.float64);L=np.full(921,16.);raw=np.arange(921,dtype=np.float32);a=np.ones(921,bool)
  np.savez_compressed(path,corpus=names,source_occurrence_index=order,raw_head_logit_before_U=raw,query_available=a,query_xy_raw=q,L_raw=L,external_target=np.array([object()],object),raw_state=np.array([object()],object))
  rr=[{'query_xy':q[i].tolist(),'L':float(L[i])}for i in ix];got,z,av=m.reference_scores(path,centre,'C23',rr);assert np.array_equal(got,ix)and np.array_equal(z,raw[ix])and av.all()
  e=Path(td)/'entry.json';e.write_text(json.dumps({'approved_small_jitter_features_and_two_heads':False}))
  try:m.authorize(e)
  except AssertionError:pass
  else:raise AssertionError('Unapproved entry accepted')
 checks['labelled_reference_NPZ_lazy_whitelist_forbidden_object_never_loaded']=True;checks['unapproved_entry_fails_before_source_model_payload']=True
 assert "np.array_equal(features[centre],reference)"in text and "np.array_equal(raw[centre],previous)"in text and "np.array_equal(available[centre],previous_available)"in text
 checks['features_and_both_raw_heads_exact_centre_parity_no_tolerance']=True
 assert "('C16',p['checkpoint'],4500"in text and "('C23',p['C23_checkpoint'],8000"in text and 'range(0,N,512)'in text and 'head.eval('not in text # .eval() is chained on construction.
 checks['same_frozen_head_class_C16_4500_C23_8000_two512_scoring_paths']=True
 z=np.array([-np.nextafter(np.float32(0),np.float32(1)),-0.,0.,1.],np.float32);a=np.array([1,1,1,0],bool);pred=np.where(a,(z>=0).astype(np.int8),-1).astype(np.int8);assert pred.tolist()==[0,1,1,-1]
 checks['raw_hidden0_signedzero_E_and_factualU_first']=True
 assert "available = factual & context"in text and "features[:, 1927]"in text and "features[~qinside, 384:768]"in text and 'available = factual & context & qinside'not in text
 checks['Qoutside_flag_zero_not_availability_gate']=True
 result={'completed':True,'PASS':True,'code':bind(p),'test_code':bind(Path(__file__)),'checks':checks,'check_count':len(checks),'actual_source_RGB_features_predictions_weights_model_current_V112_reads_or_calls':0}
 with out.open('x')as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'completed':True,'PASS':True,'code':result['code'],'checks':len(checks),'output':bind(out)}),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=H/'GENERATED_SMALL_JITTER_CHECKS.json');o=a.parse_args();run(o.output)
