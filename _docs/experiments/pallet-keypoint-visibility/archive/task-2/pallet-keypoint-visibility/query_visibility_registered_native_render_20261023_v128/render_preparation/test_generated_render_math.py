"""Small invented-array/static checks; no registered mesh, RGB source or render I/O."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,tempfile
import numpy as np
H=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def run(out):
 code=H/'render_registered_six_once.py';text=code.read_text();compile(ast.parse(text),str(code),'exec');m=module(code,'invented_renderer_math');checks={}
 assert 'torch'not in sys.modules and 'PIL'not in sys.modules;checks['inert_import_no_models_or_images']=True
 camera=dict(K=np.array([[4.,0,.5],[0,4.,.5],[0,0,1.]]),R=np.eye(3),t=np.zeros(3),eye_world=np.zeros(3))
 depth=np.full((2,2),2.);world=m.world_points(depth,camera);expected=np.array([[[-.25,-.25,2],[.25,-.25,2]],[[-.25,.25,2],[.25,.25,2]]]);assert np.array_equal(world,expected);checks['integer_pixel_centres_inverse_projection']=True
 vertex=np.array([[-2.,-2.,2.],[2.,-2.,2.],[0.,2.,2.]]);triangle=np.array([[0,1,2]],np.int32);buf=dict(depth=depth.copy(),part_id=np.zeros((2,2),np.int32),face_id=np.zeros((2,2),np.int32))
 rgb=m.shade(buf,vertex,triangle,'WOOD_BOARD_BLOCK',camera);other=dict(buf,part_id=np.full((2,2),123,np.int32));assert np.array_equal(rgb,m.shade(other,vertex,triangle,'WOOD_BOARD_BLOCK',camera));checks['RGB_has_no_target_part_ID_palette']=True
 changed=dict(buf,part_id=np.array([[999,0],[0,0]],np.int32));after=m.shade(changed,vertex,triangle,'WOOD_BOARD_BLOCK',camera);assert np.array_equal(rgb[1:],after[1:])and np.array_equal(rgb[0,1],after[0,1])and not np.array_equal(rgb[0,0],after[0,0]);checks['same_target_surface_RGB_bitexact_under_foreign_change']=True
 polymer=m.shade(buf,vertex,triangle,'POLYMER_RIB_LEG',camera);assert rgb.dtype==polymer.dtype==np.uint8 and rgb.mean()>polymer.mean()>25;checks['explicit_sRGB_makes_dark_polymer_nonblack']=True
 bg=dict(buf,depth=np.full((2,2),np.nan),part_id=np.full((2,2),-1,np.int32),face_id=np.full((2,2),-1,np.int32));assert np.array_equal(m.shade(bg,vertex,triangle,'WOOD_BOARD_BLOCK',camera),np.full((2,2,3),[225,226,228],np.uint8));checks['background_invalid_depth_has_explicit_RGB']=True
 xy=np.array([[0.,0.],[.49,.49],[.51,.51],[255.,255.],[255.6,0.],[-.6,0.]])
 ix,inside=m.nearest_pixels(xy,256);assert ix.tolist()==[[0,0],[0,0],[1,1],[255,255],[256,0],[-1,0]]and inside.tolist()==[True,True,True,True,False,False];checks['nearest_aliasing_and_OOB_retained_without_clipping']=True
 raster_path=H.parent.parent/'query_visibility_synthetic_part_teacher_20261023_v93/raster_preparation/cpu_part_raster.py';r=module(raster_path,'invented_pinned_raster_math');d=np.array([[[2.,2.]],[[2.,3.]]]);f=np.array([[[0,0]],[[1,1]]],np.int32);old=m.digest(d,f);base=r.resolve_layers(d,f)
 assert base['ambiguity'].tolist()==[[True,False]]and base['tie_part_count'].tolist()==[[2,1]];checks['different_part_ties_explicit_ambiguity']=True
 foreign=(np.array([[1.,2.]]),np.array([[2,2]],np.int32),999);combined=r.resolve_layers(d,f,foreign=foreign);assert combined['part_id'].tolist()==[[999,0]]and combined['ambiguity'].tolist()==[[False,True]]and m.digest(d,f)==old;checks['immutable_target_layers_plus_real_foreign_depth_resolve']=True
 class Teacher:
  @staticmethod
  def first_hit(eye,point,tri,owners):return dict(status='HIT',owner=0,distance=float(np.linalg.norm(point-eye)),face_ids=[0])
 fake_data=dict(native_witness_points=np.broadcast_to([0.,0.,2.],(8,37,3)),native_patch_quads=np.broadcast_to([0.,0.,2.],(8,3,4,3)))
 fake_buffer=dict(bg,ambiguity=np.zeros((2,2),bool));records=[dict(active_witness_indices=[0],state='D')for _ in range(8)]
 arrays,summaries=m.witness_diagnostics(fake_data,camera,r,Teacher,fake_buffer,vertex,triangle,np.array([0]),records)
 assert all(s['nearest_background']==s['nearest_nonfinite_depth']==s['certain_owner_disagreements']==37 and s['finite_depth_comparable']==0 and s['max_abs_depth_difference_between_different_rays']is None for s in summaries);json.dumps(summaries,allow_nan=False)
 checks['certain_background_owner_mismatch_retained_depth_null_serializable']=True
 assert "raster.rasterize_parts(fc,ft"in text and "foreign=(fd[0],ff[0],999)"in text and "combined_tri=np.concatenate([tri,ft+len(vertices)])"in text;checks['foreign_uses_3D_triangles_not_RGB_alpha_overlay']=True
 assert "records==item['records'][variant]"in text and 'nearest_pixel_does_not_relabel_continuous_teacher=True'in text and 'projected_patch_spans_xy'in text;checks['continuous_teacher_exact_and_pixel_discrepancies_separate']=True
 assert "eye_world=np.array(design['camera_eye'],np.float64)"in text and 'no_query_crops_created=True'in text and 'target_rasters=2,foreign_rasters=4'in text;checks['literal_saved_eye_6_scenes_no_crops']=True
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'entry.json';p.write_text(json.dumps(dict(approved_source_render=False,model_forward=False,fit=False,current223=False,V112=False,PnP=False,automatic_retry=False)))
  try:m.authorize(p)
  except AssertionError:pass
  else:raise AssertionError('Unapproved source render passed')
 checks['unapproved_entry_fails_before_mesh_payloads']=True
 assert all(checks.values())and 'torch'not in sys.modules and 'PIL'not in sys.modules
 receipt=dict(completed=True,PASS=True,code=bind(code),test_code=bind(Path(__file__)),checks=checks,check_count=len(checks),actual_registered_mesh_source_RGB_model_weight_render_fit_current_V112_reads_or_calls=0)
 with out.open('x')as fp:json.dump(receipt,fp,indent=2,allow_nan=False);fp.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,code=receipt['code'],checks=len(checks),output=bind(out))),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'GENERATED_RENDER_TESTS.json');a=ap.parse_args();run(a.output)
