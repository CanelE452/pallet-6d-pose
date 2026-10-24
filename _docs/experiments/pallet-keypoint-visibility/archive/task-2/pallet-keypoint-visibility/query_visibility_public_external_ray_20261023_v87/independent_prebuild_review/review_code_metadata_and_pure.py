from pathlib import Path
from collections import defaultdict,Counter
import ast,json,hashlib,datetime,math
import numpy as np
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_external_ray_20261023_v87');D=R/'independent_prebuild_review'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
policy=D/'REVIEW_POLICY_BEFORE_METADATA_AND_PURE_TESTS.json';assert policy.exists();code=R/'build_public_external_ray_teacher.py';codebind=b(code);s=code.read_text();tree=ast.parse(s);wanted={'clean','sample_ray','generated_tests','geometry'};functions=[z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name in wanted];assert len(functions)==4
ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),'<pure AST subset>','exec'),ns);sample=ns['sample_ray'];official_tests=ns['generated_tests']();assert official_tests['PASS'] and len(official_tests['tests'])==12
args=[a.arg for a in next(x for x in functions if x.name=='sample_ray').args.args];assert args==['r','camera_xyz','K','depth','scale','owner_count','owner_id','target_id'];assert not any('M0' in a or 'mask' in a.lower() for a in args)
# Independent scalar oracle: analytic intrinsics, range interval, ownership decision.
rng=np.random.default_rng(87023);counts=Counter();mismatches=[];cases=1000
for j in range(cases):
 raw=int(rng.integers(0,65536)) if j%7 else 0;scale=float(rng.choice([.1,1.,2.5]));px=int(rng.integers(0,5));py=int(rng.integers(0,3));fx=float(rng.choice([.5,1.,500.]));fy=float(rng.choice([.75,2.,450.]));cx=float(rng.choice([-1.,0.,2.]));cy=float(rng.choice([-1.,0.,1.]));K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]);gamma=math.sqrt(1+((px-cx)/fx)**2+((py-cy)/fy)**2);z=max(.1,raw*scale+float(rng.uniform(-50,50)));cnt=int(rng.integers(0,4));owner=0 if j%2==0 else 3
 dep=np.full((3,5),raw,np.uint16);cc=np.full((3,5),cnt,np.uint8);oi=np.full((3,5),owner,np.int16);actual=sample((px,py),[0,0,z],K,dep,scale,cc,oi,0)
 if raw==0:expected=(-1,'INVALID_DEPTH')
 else:
  lo=max(0,(raw-.5)*scale)*gamma;hi=(raw+.5)*scale*gamma;rr=z*gamma
  if rr-hi>15:
   if cnt==1 and owner!=0:expected=(1,'EXTERNAL_OTHER_FIRST_HIT_PROXY')
   else:expected=(-1,'FRONT_UNASSIGNED_U' if cnt==0 else 'FRONT_OVERLAP_U' if cnt>1 else 'FRONT_OWN_U')
  elif lo-rr>15:expected=(0,'CLEAR_TO_VIRTUAL_ENDPOINT_PROXY')
  else:expected=(-1,'RANGE_BAND_U')
 observed=(actual['label'],actual['reason']);counts[actual['reason']]+=1
 if observed!=expected:mismatches.append({'case':j,'actual':observed,'expected':expected})
assert not mismatches,mismatches[:5]
# Named independent edge cases (no source pixels).
K=np.eye(3);dep=np.full((3,3),500,np.uint16);cc=np.ones((3,3),np.uint8);oi=np.ones((3,3),np.int16)
def run(z=1016.,d=dep,scale=2.,k=K,c=cc,o=oi,r=(0.,0.)):return sample(r,[0,0,z],k,d,scale,c,o,0)
edge={}
for name,res,expected in [
 ('front_exact_margin_with_quantization',run(),(-1,'RANGE_BAND_U')),
 ('front_just_above_margin',run(z=1016.000001),(1,'EXTERNAL_OTHER_FIRST_HIT_PROXY')),
 ('clear_exact_margin_with_quantization',run(z=984.),(-1,'RANGE_BAND_U')),
 ('clear_just_above_margin',run(z=983.999999),(0,'CLEAR_TO_VIRTUAL_ENDPOINT_PROXY')),
 ('clear_owner_same_target_still_clear',run(z=980.,o=np.zeros_like(oi)),(0,'CLEAR_TO_VIRTUAL_ENDPOINT_PROXY')),
 ('own_front_stays_unknown',run(z=1020.,o=np.zeros_like(oi)),(-1,'FRONT_OWN_U')),
 ('missing_even_unique_other',run(d=np.zeros_like(dep)),(-1,'INVALID_DEPTH')),
 ('missing_even_unique_own',run(d=np.zeros_like(dep),o=np.zeros_like(oi)),(-1,'INVALID_DEPTH')),
 ('invalid_scale_zero',run(scale=0.),(-1,'INVALID_DEPTH')),
 ('invalid_scale_nan',run(scale=float('nan')),(-1,'INVALID_DEPTH')),
 ('singular_camera',run(k=np.array([[1.,0,0],[0,1,0],[0,0,0]])),(-1,'INVALID_CAMERA')),
 ('continuous_outside_even_nearest_inside',run(r=(-.1,0)),(-1,'INVALID_REFERENCE_RASTER')),
 ('continuous_inside_but_nearest_outside',run(r=(2.8,0)),(-1,'INVALID_REFERENCE_RASTER'))]:
 assert (res['label'],res['reason'])==expected,(name,res);edge[name]=ns['clean'](res)
# Generated geometry integration fixture tests R/t/K units and same-permutation application.
class H:
 FIXED_ZUP_SIGNS=np.array([[-1,-1,1],[1,-1,1],[1,-1,-1],[-1,-1,-1],[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]],dtype=float)
 @staticmethod
 def normal_facing_permutation(fixed,projected,cam):
  H.args=(fixed.copy(),projected.copy(),cam.copy());return np.array([4,5,6,7,0,1,2,3]),None
 @staticmethod
 def convex_self_visible_proxy(dynamic,cam):
  H.selfargs=(dynamic.copy(),cam.copy());return np.array([1,0,1,0,1,0,1,0])
rot=np.array([[0,-1,0],[1,0,0],[0,0,1]]);t=np.array([10,20,100]);kk=np.array([[500,0,320],[0,450,240],[0,0,1]])
row={'cam_R_m2c':rot.ravel().tolist(),'cam_t_m2c':t.tolist(),'K':kk.ravel().tolist()};model={'min_x':-1,'min_y':-2,'min_z':-3,'size_x':2,'size_y':4,'size_z':6};r,xyz,perm,sf,k=ns['geometry'](row,model,H)
fixed=H.FIXED_ZUP_SIGNS*np.array([1,2,3]);expected_cam=np.stack([-fixed[:,1]+10,fixed[:,0]+20,fixed[:,2]+100],axis=1);expected_proj=np.stack([500*expected_cam[:,0]/expected_cam[:,2]+320,450*expected_cam[:,1]/expected_cam[:,2]+240],axis=1)
np.testing.assert_allclose(xyz,expected_cam[perm],rtol=0,atol=1e-12);np.testing.assert_allclose(r,expected_proj[perm],rtol=0,atol=1e-12);np.testing.assert_array_equal(H.args[0],fixed);np.testing.assert_array_equal(H.selfargs[0],fixed[perm]);np.testing.assert_array_equal(H.args[2],np.array([-20,10,-100]));np.testing.assert_array_equal(sf,[1,0,1,0,1,0,1,0])
# Source METADATA only. No image decoding, actualbuild, NPZ array access or model import.
prot=R/'SOURCE_EXTERNAL_RAY_PROTOCOL_BEFORE_DEPTH_PIXELS.json';assert b(prot)['sha256']=='a5e7d481b7d72bf0799cd513c1f92c848714d5e921b9bc264beb3f8bd07463cd';p=json.loads(prot.read_text());bindings={}
for name,item in p['bindings'].items():actual=b(Path(item['path']));assert actual['sha256']==item['sha256'];bindings[name]=actual
rows=[json.loads(l) for l in Path(p['bindings']['manifest']['path']).read_text().splitlines()];assert len(rows)==4200;assert Counter(z['split'] for z in rows)=={'train':3400,'val':800};models=json.loads(Path(p['bindings']['models_info']['path']).read_text());assert len(models)==21
G=defaultdict(list);scene={};mb=[]
for z in rows:G[z['scene_id'],z['frame_id']].append(z)
assert len(G)==1000
for (sid,fid),rs in G.items():
 folder=Path(rs[0]['depth']).parent.parent
 if sid not in scene:
  gp=folder/'scene_gt.json';kp=folder/'scene_camera.json';scene[sid]=(json.loads(gp.read_text()),json.loads(kp.read_text()));mb.extend([b(gp),b(kp)])
 gt,ca=scene[sid][0][str(fid)],scene[sid][1][str(fid)];assert sorted(z['instance_id'] for z in rs)==list(range(len(gt)));assert len({z['depth'] for z in rs})==1;assert len({(z['height'],z['width']) for z in rs})==1
 for z in rs:
  iid=z['instance_id'];assert z['object_id']==gt[iid]['obj_id'];assert z['cam_R_m2c']==gt[iid]['cam_R_m2c'] and z['cam_t_m2c']==gt[iid]['cam_t_m2c'] and z['K']==ca['cam_K'] and z['depth_scale']==ca['depth_scale']>0;assert z['split']==('train' if sid<8 else 'val');assert Path(z['depth']).name==f'{fid:06d}.png';assert Path(z['mask_vis']).parent.name=='mask_visib' and Path(z['mask_vis']).name==f'{fid:06d}_{iid:06d}.png';assert Path(z['mask_vis']).parent.parent==folder;assert str(z['object_id']) in models
  assert (np.array([models[str(z['object_id'])]['size_'+v] for v in 'xyz'])>0).all()
max_instances=max(map(len,G.values()));assert max_instances<256 and max(z['instance_id'] for z in rows)<32767
checks={
 'range_quantization_and_units':'Depthraw0 invalid. Zinterval=(raw+/-.5)*positive_scene_scale_mm; gamma on rounded pixel ray converts both endpoint and observation to range; fixed15mm compared strictly outside interval. Generated analytic oracle matches1000cases.',
 'invalid_priority':'Invaliddepth returnsU before owner can imply positive/negative; continuous+nearest framebounds required. Own-front, unassigned-front and overlap-front allU; clear0 ignoresowner andM0 as prospectivecontract says.',
 'owner_all_instances':'Metadata4200/1000 complete ordinals; rowobjectid/pose/K/scale and mask_vis filenames match every sceneGTinstance. uint8 count cannot overflow currentcohort max'+str(max_instances)+'. UniqueOTHER is ordinal unequalto target, not objectcategory, so repeated same-category instances remain foreign.',
 'reference_perm':'geometry applies same perm once to projected and cameraxyz; nativehelper is hashbound. Builder requires exact oldid/order plus array_equal(r,oldr),(perm,oldperm),(sf,oldself) all4200 before storing labels. This review tests pure fixture only, NOT actual4200r equality.',
 'no_M0_gate':'sample_ray has no M0/maskfull input; all actual Image.open paths are depth/mask_vis. OldcenterM0/M4/localsupport arrays are read from old sealedNPZ solely as separately named diagnostics. No source support/mesh membership requirement.',
 'read_execution_boundary':'Only main() contains actualpixel reads; ASTsubset executed only pure funcs. No RGB, actualbuilderrun, actualdepth/maskdecode, model, current orPnP.'}
report={'schema':'V87_INDEPENDENT_PREBUILD_CODE_METADATA_REVIEW_V1','completed':True,'PASS':True,'blockers':[],'policy':b(policy),'protocol':b(prot),'builder_code':codebind,'code_unchanged_after_review':b(code)==codebind,'source_binding_hashes':bindings,'checks':checks,'pure_tests':{'original_generated12':official_tests,'independent_random_analytic_oracle':{'N':cases,'seed':87023,'mismatches':0,'reason_counts':dict(counts)},'independent_named_edge_cases':edge,'geometry_fixture':{'PASS':True,'camera_transform_units_mm':True,'dynamic_perm_applied_once_to_xy_and_xyz':True,'self_visible_flag_exported_separate':True,'actual_source_geometry_equality_executed':False}},'metadata':{'rows':4200,'frames':1000,'splits':dict(Counter(z['split'] for z in rows)),'scene_count':len(scene),'complete_instance_ordinals':True,'max_instances_per_frame':max_instances,'depth_scales':sorted(set(z['depth_scale'] for z in rows)),'mask_filename_and_instance_identity_mismatches':0,'GTpose_K_scale_metadata_mismatches':0,'scene_metadata_bindings':mb,'actual_GT_npz_arrays_loaded':False,'actual_depth_mask_RGB_pixels_decoded':False},'limitations':['Raster rounding changes the fractional reference ray; endpoint uses originalz on the rounded pixel ray. This is disclosed approximation, not exact renderer/mesh first-hit visibility.','M4 ownership is weak observed-raster evidence. Unannotated/environment foreground remainsU; own foreground hides what lies behind and remainsU.','CLEAR means clear to a virtual endpoint, not semanticD or observed physicalcue; selfvisible proxy remains separate and is not semanticS.','15mmrange margin and quantization interval do not calibrate GTpose/sensor alignment/noise.','Actualbuild remains root-only after review; all4200reference parity and pixel schemas are runtime failclosed assertions, not established by this metadata review.'],'nonblocking_notes':['Metadata/read failures before the pixel loop are outside BUILD_FAILURE_PRESERVED try-block; root terminal log and existing outputdirectory must still be preserved if they occur. No data/label repair requested.','Futureinstance counts>=256 would overflow uint8 owner_count; the bound currentcohort max is explicitly verified below256, so no current blocker.'],'scope':{'actualbuild':0,'depth_mask_RGB_pixels':0,'newforward_fit_current_PnP':0,'V85_annotation_files_modified':0,'model_weights_loaded':0,'opaque_GT_NPZ_SHA_only':True},'review_code':b(Path(__file__)),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
f=D/'INDEPENDENT_PREBUILD_CODE_METADATA_REVIEW.json';assert not f.exists();f.write_text(json.dumps(ns['clean'](report),ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'review':b(f),'PASS':True,'blockers':[],'maxinstances':max_instances},indent=2))
