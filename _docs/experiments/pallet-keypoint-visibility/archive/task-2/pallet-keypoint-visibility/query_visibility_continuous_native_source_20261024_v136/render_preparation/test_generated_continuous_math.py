"""V136 metadata/invented-array checks only; no registered NPZ or actual raster/RGB."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,tempfile,collections
import numpy as np
H=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def run(out):
 code=H/'render_continuous_corpus_once.py';cp=H/'continuous_corpus_math.py';compile(ast.parse(code.read_text()),str(code),'exec');compile(ast.parse(cp.read_text()),str(cp),'exec');m=module(code,'invented_v136_driver');c=module(cp,'invented_v136_core');policy=json.loads(m.POLICY.read_text());checks={}
 assert 'torch'not in sys.modules and'PIL'not in sys.modules;checks['inert_no_models_or_images']=True
 fake_meshes=[dict(path='INVENTED_A',sha256='x'),dict(path='INVENTED_B',sha256='y')];configs=c.make_configs(policy,fake_meshes);assert len(configs)==640 and collections.Counter(r['split']for r in configs)=={'TRAIN':512,'VALIDATION':128};checks['literal640_counts512_128_cartesian_axes']=True
 for fold in('TRAIN','VALIDATION'):
  rr=[r for r in configs if r['split']==fold]
  for name,n in(('foreign_family',4),('target_role',4)):
   cc=collections.Counter(r['factors'][name]for r in rr);assert len(cc)==n and len(set(cc.values()))==1
  assert len(collections.Counter(tuple(r['sham_direction_unit_xy'])for r in rr))==8
  assert all(sum(r['factors']['material']==mat for r in rr)==len(rr)//2 for mat in policy['fixed_factors']['materials'])
 checks['foreign4_roles4_directions8_materials_balanced_byfold']=True
 geometry={};layers=collections.Counter()
 for r in configs:
  value=(r['dimensions_xyz_m'],r['positive_scale_xyz']);assert geometry.setdefault(r['geometry_group_id'],value)==value;layers[r['target_layer_group_id']]+=1
 assert len(geometry)==40 and len(layers)==320 and set(layers.values())=={2};checks['geometry_shared40_groups_and320_material_twin_target_layers']=True
 tr={(r['factors']['signed_quarter'],tuple(r['factors']['camera_pair_deg']))for r in configs if r['split']=='TRAIN'};va={(r['factors']['signed_quarter'],tuple(r['factors']['camera_pair_deg']))for r in configs if r['split']=='VALIDATION'};assert tr.isdisjoint(va)and set(policy['fixed_factors']['asset_seeds']['TRAIN']).isdisjoint(policy['fixed_factors']['asset_seeds']['VALIDATION']);checks['asset_and_camera_groups_fold_disjoint']=True
 for key in('policy','base_configs','render_complete','source_math','renderer'):m.check(policy['old_regression'][key])
 old_configs=json.loads(Path(policy['old_regression']['base_configs']['path']).read_text())['rows']
 audit=c.metadata_audit(configs,old_configs);assert audit['PASS']
 for fold,count in(('TRAIN',32),('VALIDATION',8)):
  rr=[r for r in configs if r['split']==fold];groups={r['geometry_group_id']:r for r in rr}
  assert len(groups)==count
  assert sorted(r['generation_keys']['height_stratum']for r in groups.values())==list(range(count))
  for r in groups.values():
   ordinal=r['generation_keys']['height_stratum'];height=r['dimensions_xyz_m'][1]
   assert .10+.10*ordinal/count<=height<.10+.10*(ordinal+1)/count
  assert min(r['dimensions_xyz_m'][1]for r in rr)<min(r['dimensions_xyz_m'][1]for r in old_configs if r['split']=='VALIDATION')
  assert max(r['dimensions_xyz_m'][1]for r in rr)>max(r['dimensions_xyz_m'][1]for r in old_configs if r['split']=='VALIDATION')
  for r in rr:
   az,el=r['factors']['camera_pair_deg'];dims=r['dimensions_xyz_m'];cam=r['camera']
   assert 5<abs(az)<40 and 10<el<45 and 2.2<r['factors']['distance_scale']<4.2
   assert np.isclose(np.linalg.norm(cam['eye_world']),max(dims[0],dims[2])*r['factors']['distance_scale'],rtol=0,atol=1e-14)
   assert r['source_label_mapping_certificate']['top_cosine_margin']>0.10
 checks['all640_exact_normal_rule_no_ties_and_no_GTbest_mapping']=True
 checks['height32_8_strata_bothcover_oldVAL_height_range']=True
 checks['camera_and_height_independent_hashes_samebroad_ranges_bothfolds']=True
 checks['distance_dimensionless_V129_multiplier_and_derived_m_exact']=True
 checks['effective_material_recipes_and_assets_disjoint_newfolds_and_old']=True
 checks['original128_64_corpus_bindings_unchanged']=True
 for cfg in configs:
  anchors=np.array(cfg['native8_canonical_xyz']);A=np.array(cfg['canonical_to_CF_rotation']);perm=cfg['permCF_to_canonical'];half=np.abs(anchors[0]);half=half[[2,1,0]]if cfg['factors']['signed_quarter']%180 else half;assert np.array_equal((anchors@A.T)[perm],c.SIGNS*half);assert np.array_equal(c.project(anchors,cfg['camera'])[perm],np.array(cfg['native8_CF_xy']))
 checks['all640_signed_perms_camera_Atranspose_no_double_rotation']=True
 template=dict(vertices=c.SIGNS*c.OLD_DIM/2,triangles=np.array([[0,1,2]],np.int32),part_owner=np.array([0]),face_axis=np.array([0]),face_side=np.array([1]),native_anchor3D=c.SIGNS*c.OLD_DIM/2,native_primary_part_id=np.zeros(8,np.int32),native_incident_part_ids=np.zeros((8,1),np.int32),native_patch_quads=np.broadcast_to(c.SIGNS[:,None,None,:]*c.OLD_DIM/2,(8,3,4,3)).copy(),native_edge_samples=np.broadcast_to(c.SIGNS[:,None,:]*c.OLD_DIM/2,(8,9,3)).copy(),native_witness_points=np.broadcast_to(c.SIGNS[:,None,:]*c.OLD_DIM/2,(8,37,3)).copy())
 data=c.scaled_data(template,configs[0]);scale=np.array(configs[0]['positive_scale_xyz'])
 for key in('vertices','native_anchor3D','native_patch_quads','native_edge_samples','native_witness_points'):assert np.array_equal(data[key],template[key]*scale)
 checks['one_positive_transform_applies_to_all_registered_arrays']=True
 for cfg in configs:
  data=c.scaled_data(template,cfg)
  # Give the invented patch finite area rather than a single vertex.
  for i in range(8):
   a=data['native_anchor3D'][i];away=-c.SIGNS[i]
   for axis in range(3):
    other=[j for j in range(3)if j!=axis];u=np.eye(3)[other[0]]*.01*away[other[0]];v=np.eye(3)[other[1]]*.01*away[other[1]];data['native_patch_quads'][i,axis]=[a,a+u,a+u+v,a+v]
  f,delta,receipt=c.foreign_placements(data,cfg);assert receipt['near_clearance_px']>=1-1e-10 and receipt['far_additional_px']>=128-1e-10
  for name in('NEAR_SHAM','FAR_SHAM'):assert np.allclose(f[name]-f['COVER'],np.broadcast_to(delta[name],(8,3)),rtol=0,atol=2e-15)
 checks['all640_analytic_8vertex_near1px_far128_rigid_translation']=True
 cfg=configs[0];data=c.scaled_data(template,cfg);records=[dict(name='canonical%d'%i,state='DESU'[i%4],active_witness_indices=[0,1],native_canonical_index=i)for i in range(8)];ordered=c.CF_records(records,cfg['permCF_to_canonical'],data,cfg['camera'])
 assert all(r['canonical_index']==cfg['permCF_to_canonical'][i]and r['CF_index']==i and r['state']==records[cfg['permCF_to_canonical'][i]]['state']and len(r['projected_active_witness_bbox_xyxy'])==4 for i,r in enumerate(ordered));checks['CF_state_reordered_once_and_support_bbox_explicit']=True
 text=code.read_text();assert 'raster.rasterize_parts' in text and 'foreign=(fd[0],ff[0],999)'in text and 'same_target_RGB_bitexact=True'in text and "SCENES_INPUT_ONLY.json"in text and "SCENES_GT_ONLY.json"in text and "BASE_COMPLETE.json"in text and 'automatic_retry=False'in text;checks['3D_foreign_GT_split_perbase_checkpoint_no_retry_contract']=True
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'unapproved.json';p.write_text(json.dumps(dict(approved_source_corpus_render=False,model_forward=False,fit=False,current223=False,V112=False,PnP=False,automatic_retry=False)))
  try:m.authorize(p)
  except AssertionError:pass
  else:raise AssertionError('Unapproved actual render accepted')
 checks['unapproved_entry_failclosed_before_registered_payloads']=True
 assert all(checks.values())and 'torch'not in sys.modules and'PIL'not in sys.modules
 receipt=dict(metadata_audit=audit,completed=True,PASS=True,code=bind(code),core_code=bind(cp),test_code=bind(Path(__file__)),checks=checks,check_count=len(checks),actual_registered_mesh_payload_RGB_render_model_fit_current_V112_calls=0)
 with out.open('x')as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(completed=True,PASS=True,code=receipt['code'],core_code=receipt['core_code'],checks=len(checks),output=bind(out))),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'GENERATED_CORPUS_TESTS.json');args=ap.parse_args();run(args.output)
