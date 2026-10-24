"""Authored functional fixtures only. No GT, source arrays, scores or model."""
import importlib.util,copy,json,math,hashlib,tempfile
from pathlib import Path
H=Path(__file__).parent
sp=importlib.util.spec_from_file_location('holdout',H/'evaluate_saved_source_point_holdout.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
checks=[]
def ok(name,fn):fn();checks.append(name)
def rejects(fn):
 try:fn()
 except (AssertionError,OSError):return
 raise AssertionError('Expected fail-closed rejection')
def row():
 r=dict(component_role='HOLDOUT',mapping_status='EXACT_ORIGINAL_IMAGE_OWN_SCENE',family='SELF_SCENE15',state_name='NOMINAL',state_index=0,original_query_available=True,query_xy=[100.,100.],L=96.,saved_projected_reference_xy=[100.,100.],point0_target_alone_state='D',point0_combined_state='E')
 r['reference_locality_for_actual_input']=m.locality(r['query_xy'],r['L'],True,r['saved_projected_reference_xy'],[512,512]);return r
ok('HOLDOUT_DE_valid_and_role_unchanged',lambda: (m.need(m.evaluation_target(row())[0]==1,'E valid'),m.need(row()['component_role']=='HOLDOUT','Role unchanged')))
def role():
 r=row();r['component_role']='TRAIN';m.need(m.evaluation_target(r)[0]==-1,'TRAIN cannot masquerade HOLDOUT')
ok('TRAIN_not_HOLDOUT_by_role_complement_or_rewrite',role)
def unknown():
 for key,value in [('original_query_available',False),('point0_combined_state','S'),('point0_combined_state','U'),('point0_combined_state','UNKNOWN'),('point0_target_alone_state','S'),('saved_projected_reference_xy',None)]:
  r=row();r[key]=value;m.need(m.evaluation_target(r)[0]==-1,'Unknown/excluded kept '+key)
ok('OriginalU_SELF_missing_never_external',unknown)
def bounds():
 for x,want in [(52.,1),(148.,-1),(52.5,1)]:
  r=row();r['saved_projected_reference_xy']=[x,100.];r['reference_locality_for_actual_input']=m.locality(r['query_xy'],96.,True,r['saved_projected_reference_xy'],[512,512]);m.need(m.evaluation_target(r)[0]==want,'Exact half-open canvas')
 r=row();r['reference_locality_for_actual_input']['inside_original_frame_canvas_extent']=False;m.need(m.evaluation_target(r)[0]==-1,'Frame outside remains excluded')
ok('Halfopen_canvas_no_samplecenter_gate',bounds)
def distinct():
 r=m.extract(dict(target_witness_states=['D','S'],combined_witness_states=['E','D'],state='U',target_alone_region_state='S',projected_reference_xy=[1.,2.]));m.need(r['point0_combined_state']=='E'and r['region_state_context_only']=='U','Point0 separate from region/variant');m.need(m.extract({'state':'D'})['point0_combined_state']=='UNKNOWN','No region fallback')
ok('Saved_point0_NEAR_SHAM_E_and_region_separate',distinct)
def parser():
 scene=dict(scene_index=4,base_index=1,variant='NEAR_SHAM',native_rows=[dict(native_slot=k,declared_CF_to_canonical=k)for k in range(8)])
 own=dict(scene_index=4,base_index=1,variant='NEAR_SHAM',permCF_to_canonical=list(range(8)),records_CF=[dict(CF_index=k,canonical_index=k)for k in range(8)])
 rr,why=m.select_scene({'rows':[own]},scene);m.need(why is None and len(rr)==8,'Exact scene/perm')
 bad=copy.deepcopy(own);bad['records_CF'][0]['canonical_index']=1;rejects(lambda:m.select_scene({'rows':[bad]},scene))
 m.need(m.select_scene({'rows':[]},scene)[1]=='UNKNOWN_MISSING_EXACT_OWN_SCENE','Missing no fallback')
ok('Exact_CF_permutation_missing_preserved',parser)
def mapping():
 def scene(si,variant):
  return dict(scene_index=si,base_index=1,variant=variant,source_RGB_binding={'path':'INERT'},original_input_metadata8=[dict(original_global_index=si*8+k,original_occurrence_id=f's{si}:n{k}',original_query_xy=[10.+k,20.],original_L=96.,original_input_flags={'query_available':True},declared_CF_to_canonical=k)for k in range(8)])
 sm={4:scene(4,'BEFORE'),5:scene(5,'NEAR_SHAM')};r=dict(native_slot=0,source_image_original_global_index=40,anchor_original_global_index=32,scene_index=5,base_index=1,variant='NEAR_SHAM',anchor_scene_index=4,query_xy=[10.,20.],L=96.,original_query_available=True,source_image_original_query_id='s5:n0',anchor_original_query_id='s4:n0',original_flags_ref=[4,0],image_flags_ref=[5,0],family='BEFORE_ANCHOR4',state_index=0,state_name='NOMINAL')
 z=m.map_input(r,sm);m.need(z['component_role']=='HOLDOUT'and z['source_teacher_key']==[5,0]and z['anchor_scene_index']==4,'Actual image teacher, anchor q retained')
 bad=copy.deepcopy(r);bad['query_xy']=[11.,20.];rejects(lambda:m.map_input(bad,sm))
ok('Image_anchor_not_interchangeable_HOLDrole_preserved',mapping)
ok('Fixed_zero_decoder_and_inputU',lambda:m.need([m.decode(x,a)for x,a in [(-1.,True),(0.,True),(1.,True),(math.nan,True),(1.,False)]]==[0,1,1,-1,-1],'Zero/U decoder'))
def metrics():
 rs=[{'evaluation_target_D0_E1_UNKNOWNminus1':t,'predictions':{'a':{'prediction':p}}}for t,p in[(0,0),(0,1),(1,-1),(-1,1)]];z=m.metric(rs,'a');m.need(z['confusion_D_E_by_D_E_U']==[[1,1,0],[0,0,1]]and z['unknown_truth']==1 and z['U']==1,'Unknown retained in denominator')
ok('Confusion_unknown_U_denominators',metrics)
def gates():
 fit=dict(completed=True,PASS=True,source_only=True,from_scratch=True,parameters_each=118359,steps_each=4000,total_updates=8000,all2_frozen_before_any_SOURCE_outputs=True,protected_real_COMMON_VAL_access=0,arms={})
 entry=dict(fit_complete={'path':'INERT','sha256':'x'},prediction_packets={});barrier=dict(completed=True,checkpoints={});review=dict(completed=True,PASS=True,fit_complete=entry['fit_complete'])
 for a in m.ARMS:
  cp={'path':'INERT_'+a};pred={'path':'INERT_PRED_'+a};fit['arms'][a]=dict(updates=4000,use_sam=False,spatial_q8=True,weak_bridge=a==m.ARMS[1],weak_loss_weight=.25 if a==m.ARMS[1]else 0,checkpoint=cp,outputs={'SYNTH_AUG65536':pred});barrier['checkpoints'][a]=cp;entry['prediction_packets'][a]=pred
 m.guard_fit(fit,barrier,review,entry)
 for field,value in [('updates',3999),('use_sam',True),('weak_loss_weight',0.)]:
  bad=copy.deepcopy(fit);bad['arms'][m.ARMS[1]][field]=value;rejects(lambda:m.guard_fit(bad,barrier,review,entry))
 bad=copy.deepcopy(review);bad['fit_complete']={'path':'WRONG'};rejects(lambda:m.guard_fit(fit,barrier,bad,entry))
ok('Both_final4000_exact_fitreview_RAW_weak_flags',gates)
ok('Forbidden_lexical_before_FS',lambda:rejects(lambda:m.safe(str(m.P/'not_present_v112/file.json'))))
def symlink():
 with tempfile.TemporaryDirectory(dir=H,prefix='invented_only_')as temp:
  d=Path(temp);(d/'a.json').write_text('{}');(d/'link.json').symlink_to(d/'a.json');rejects(lambda:m.raw(d/'link.json'))
ok('Symlink_refused',symlink)
result=dict(completed=True,PASS=True,code_sha256=hashlib.sha256((H/'evaluate_saved_source_point_holdout.py').read_bytes()).hexdigest(),checks=checks,actual_GT_payload_or_prediction_array_reads=0,operational_model_forward_fit=0)
with (H/'INVENTED_CHECKS.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
