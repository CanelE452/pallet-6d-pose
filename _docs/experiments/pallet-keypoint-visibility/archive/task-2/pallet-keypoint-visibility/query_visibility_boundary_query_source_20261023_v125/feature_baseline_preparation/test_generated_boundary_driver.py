"""Invented metadata/arrays only; no source files, Torch, weights or model calls."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,tempfile
import numpy as np
H=Path(__file__).resolve().parent;CODE=H/'run_frozen_boundary_source.py'
s=importlib.util.spec_from_file_location('generated_v125_driver',CODE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def sample(qi,delta,role,rank=None):
 q8=np.array([[10.,10.],[35.,5.],[35.,30.],[5.,30.],[20.,5.],[20.,30.],[5.,17.5],[35.,17.5]])
 q8[0]+=delta;box=np.r_[q8.min(0),q8.max(0)].tolist();L=float(np.clip(.25*np.max(np.ptp(q8,axis=0)),16,50))
 variant=m.VARIANTS[role];recipe=dict(kind='ORIGINAL',original_full_RGB=dict(path='/invented/native.npy',sha256='0'*64),original_format='NPY',raw_hw=[40,50],decoded_RGB_sha256=str(role)*64)
 if role:recipe.update(kind='OPAQUE_PATCH',patch=dict(path='/invented/shared_patch.npz',sha256='a'*64),placement_xy=[3+role,4])
 return dict(occurrence_index=0,rgb96_row=0,occurrence_id='pending',query_index=qi,query_id='q'+str(qi),cue_index=0,cue_id='generated_cue',jitter_index=qi,offset_xy=list(delta),source='LOCO',split='TRAIN',raw_split='TRAIN',source_frame_group='generated_scene',query_xy=q8[0].tolist(),L=L,initial_points=q8.tolist(),context_box_xyxy=box,source_emulated_not_initializer=True,original_initializer_touched=False,fixed_carrier_slot=0,variant=variant,donor_rank=None if not role else rank,raw_hw=[40,50],full_RGB=recipe['original_full_RGB'],full_RGB_format='NPY',decoded_RGB_sha256='0'*64,target_bbox_xyxy=[5.,5.,35.,30.],full_RGB_recipe=recipe,rgb96_sha256='e'*64,valid96_sha256='f'*64)

def rejected(fn):
 try:fn()
 except (AssertionError,ValueError,KeyError):return True
 raise AssertionError('Expected fail-closed rejection')

def run(out):
 parent=[sample(0,[0,0],role,7)for role in range(3)]
 for i,r in enumerate(parent):r.update(occurrence_index=i,rgb96_row=i,occurrence_id='parent'+str(i))
 rs=[]
 # Two query conditions and one rank intentionally differ from 9 jitters/8 ranks/4896 rows.
 for qi,delta in enumerate(([0,0],[8,0])):
  for role in range(3):
   r=sample(qi,delta,role,7);i=len(rs);r.update(occurrence_index=i,rgb96_row=i,occurrence_id='new'+str(i),parent_recipe_occurrence_index=role)
   if qi==0:r['parent_occurrence_index']=role
   rs.append(r)
 checks={};groups=m.validate_occurrences(rs,parent);checks['dynamic6_rows2_queries1_rank']=len(groups)==2
 bad=copy.deepcopy(rs);bad[4]['L']+=1;checks['qL_mismatch_rejected']=rejected(lambda:m.validate_occurrences(bad,parent))
 bad=copy.deepcopy(rs);bad[4]['full_RGB_recipe']['placement_xy'][0]+=1;checks['fixed_native_recipe_change_rejected']=rejected(lambda:m.validate_occurrences(bad,parent))
 bad=copy.deepcopy(rs);bad[4]['candidate_aux_hidden_label_GT_only']=1;checks['GT_feature_field_rejected']=rejected(lambda:m.validate_occurrences(bad,parent))
 bad=copy.deepcopy(rs);bad[4]['split']='VALIDATION';checks['fold_reassignment_rejected']=rejected(lambda:m.validate_occurrences(bad,parent))
 bad=copy.deepcopy(rs);bad[0]['rgb96_sha256']='9'*64;checks['centre_parent_crop_mismatch_rejected']=rejected(lambda:m.validate_occurrences(bad,parent))
 bad=copy.deepcopy(rs);bad.pop();checks['missing_sham_rejected']=rejected(lambda:m.validate_occurrences(bad,parent))
 a=copy.deepcopy(rs[0]);b=copy.deepcopy(a);b['variant']='CUE_PRESERVING_SHAM';checks['dedup_input_only_not_variant']=m.unit_key(a)==m.unit_key(b)
 v=np.ones((1,96,96),bool);tv=np.ones((224,224),bool)
 checks['valid_availability']=m.factual_availability([20.,20.],[40,50],v,tv)==(True,True,True,True)
 checks['finite_OOB_not_available']=m.factual_availability([-2.,20.],[40,50],v,tv)==(False,True,False,True)
 v[:,47,47]=False;checks['literal_central4_mask_gate']=m.factual_availability([20.,20.],[40,50],v,tv)==(False,True,False,False)
 v[:]=True;checks['contextU_distinct']=m.factual_availability([20.,20.],[40,50],v,np.zeros_like(tv))==(True,False,False,True)
 checks['knownU_denominator']=m.metrics(np.array([0,1,-1]),np.array([0,-1,1]))['CM_D_E_rows_D_E_U_columns']==[[1,0,0],[0,0,1]]
 recon=m.pure(m.CORPUS_CODE,('require','patch_slices','reconstruct_full_RGB'))['reconstruct_full_RGB'];im=np.zeros((5,6,3),np.uint8);pat=np.full((3,4,3),77,np.uint8);alpha=np.ones((3,4),bool);actual=recon(im,pat,alpha,[-1,3]);expected=im.copy();expected[3:5,0:3]=77
 checks['exact_existing_native_reconstruction_generated']=np.array_equal(actual,expected)and not im.any()
 with tempfile.TemporaryDirectory()as td:
  ep=Path(td)/'entry.json';ep.write_text(json.dumps({'approved_PUBLIC_source_feature_and_frozen_baseline':False}))
  checks['unapproved_entry_rejected_before_payload_IO']=rejected(lambda:m.authorize(ep))
 assert all(checks.values())
 result=dict(completed=True,PASS=True,code=dict(path=str(CODE),sha256=hashlib.sha256(CODE.read_bytes()).hexdigest()),checks=checks,source_pixels_features_weights_models_current_V112_reads_or_calls=0)
 with Path(out).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);run(ap.parse_args().output)
