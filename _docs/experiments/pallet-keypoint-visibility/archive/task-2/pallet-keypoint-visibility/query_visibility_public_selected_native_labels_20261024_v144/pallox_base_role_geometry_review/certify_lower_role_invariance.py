"""Known-source geometry only: lower roles invariant to unspecified upper height.

No initializer/prediction/query errors, model, optimization or part-cut selection.
"""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');V=P/'query_visibility_public_selected_native_labels_20261024_v144';H=V/'pallox_base_role_geometry_review';OLD=P/'query_visibility_public_native_reference_feasibility_20261024_v140'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
policy=V/'ROOT_SOURCE_NATIVE_ALIGNMENT_SCOPE.json';assert bind(policy)['sha256']=='d2d356ca7325aff8918c0839b6ca3657bb4dd8c4a1adf96d119a52661af161ca'
factspath=OLD/'pallox_reference_diagnostics_v1/SOURCE_GEOMETRY_FEASIBILITY.json';facts=read(factspath)
axispath=OLD/'pallox_reference_preparation/AXIS_REGISTRATION_PROPOSAL_BEFORE_NAMED_NATIVE_LABELS.json';axis=read(axispath)
namedpath=OLD/'pallox_named_virtual_reference_v1/NAMED_CF32_REFERENCES.json';named=read(namedpath)
contract=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py';assert bind(contract)['sha256']=='4ddd134aa26da1d676628bd6987a161ab36f09328dfdaa89bedd213e397755ab'
names={'FIXED_ZUP_SIGNS','VERTICAL_FIXED_FACES','outward_normals','normal_facing_permutation'};nodes=[]
for n in ast.parse(contract.read_text()).body:
 if isinstance(n,ast.FunctionDef)and n.name in names:nodes.append(n)
 elif isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in names for t in n.targets):nodes.append(n)
ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(contract)+'::original_pure_geometry_only','exec'),ns)
lo=np.array(facts['mesh']['bounds_min']);hi=np.array(facts['mesh']['bounds_max']);C=np.array(facts['mesh']['AABB_center']);B=np.array(axis['source_specific_registration']['world_from_centered_object_rotation']);original=np.array(named['registered_fixed_object_corners_xyz_m'])
assert np.allclose(B@B.T,np.eye(3))and np.linalg.det(B)==1
world=lambda q:(q-C)@B.T
fixedlower=np.flatnonzero(ns['FIXED_ZUP_SIGNS'][:,2]<0);fixedupper=np.flatnonzero(ns['FIXED_ZUP_SIGNS'][:,2]>0)
bottom_Y=float(original[fixedlower[0],1])
assert np.array_equal(original[fixedlower,1],np.full(4,bottom_Y)) and abs(bottom_Y-hi[1])<1e-15
def project(q,K,T):
 cam=q@T[:3,:3].T+T[:3,3];uv=cam@K.T;return uv[:,:2]/uv[:,2,None],cam[:,2]
frames=[];panels=[];font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
for frame in facts['fixed_four_records']:
 T=np.array(frame['supplied_object_to_camera_T']);K=np.array(frame['K']);camera=(-T[:3,:3].T@T[:3,3]-C)@B.T
 uv,depth=project(original,K,T);perm,cos=ns['normal_facing_permutation'](world(original),uv,camera)
 oldframe=next(r for r in named['frames']if r['id']==frame['id']);assert perm.tolist()==oldframe['permutation_dynamic_to_registered_fixed']
 lower=original[perm[[2,3]]].copy();assert np.array_equal(lower[:,1],np.full(2,bottom_Y))
 endpoints=[]
 for t in (lo[1],bottom_Y):
  q=original.copy();q[fixedupper,1]=t;pp,zz=project(q,K,T);endpoints.append(dict(raw_upper_Y_m=float(t),front_top_X_gap=float(pp[perm[1],0]-pp[perm[0],0]),all8_positive_camera_Z=bool((zz>0).all()),min_camera_Z=float(zz.min())))
 assert all(r['front_top_X_gap']>0 and r['all8_positive_camera_Z']for r in endpoints)
 # Independent polynomial proof: projected X difference numerator has no t².
 # For two fixed-XZ top endpoints: u(t)=(alpha*t+beta)/(gamma*t+delta).
 a,b=original[perm[0]].copy(),original[perm[1]].copy();a[1]=b[1]=0
 ca=a@T[:3,:3].T+T[:3,3];cb=b@T[:3,:3].T+T[:3,3]
 alpha=float(K[0]@T[:3,1]);gamma=float(T[2,1]);ba=float(K[0]@ca);bb=float(K[0]@cb);da=float(ca[2]);db=float(cb[2])
 slope=alpha*(da-db)+gamma*(bb-ba);intercept=bb*da-ba*db
 nend=[float(slope*t+intercept)for t in (lo[1],bottom_Y)];assert min(nend)>0
 for t in np.linspace(lo[1],bottom_Y,67,endpoint=False):
  q=original.copy();q[fixedupper,1]=t;pp,zz=project(q,K,T);per,c=ns['normal_facing_permutation'](world(q),pp,camera)
  assert np.array_equal(per,perm)and np.array_equal(q[per[[2,3]]],lower)
  assert np.allclose(pp[per[[2,3]]],uv[perm[[2,3]]],rtol=0,atol=0)
  assert abs((pp[perm[1],0]-pp[perm[0],0])-(slope*t+intercept)/((gamma*t+da)*(gamma*t+db)))<1e-10
 rows=[]
 for k in (2,3):
  old=next(r for r in named['rows']if r['source_frame_id']==frame['id']and r['native_slot']==k)
  assert np.allclose(old['reference_object_xyz_m'],original[perm[k]],rtol=0,atol=0)
  assert np.allclose(old['reference_xy'],uv[perm[k]],rtol=0,atol=0)
  rows.append(dict(native_slot=k,role='whole_assembly_front_lower_'+('right'if k==2 else'left')+'_virtual_footprint',original_unchanged_reference_object_xyz_m=original[perm[k]].tolist(),original_unchanged_reference_xy=uv[perm[k]].tolist(),positive_Z=bool(depth[perm[k]]>0),continuous_inframe=bool(0<=uv[perm[k],0]<640 and 0<=uv[perm[k],1]<480),geometry_role_invariant_to_all_upper_heights=True,physical_vertex_membership_required=False,standalone_palletbase_horizontal_extent_certificate=False,semantic_visibility_label='SEPARATE_SOURCE_RGB_ADJUDICATION'))
 frames.append(dict(id=frame['id'],original_split=frame['original_split'],original_source_pose_file=frame['input_files_read']['pose'],original_K_file=frame['input_files_read']['K'],original_named_frame_role_status=oldframe['role_status'],front_face_argmax_height_invariant_reason='vertical normals have zero registered-Z component; upper-height changes only common positive cosine denominator',front_order_top_X_continuous_interval_proof=dict(upper_raw_Y_interval_m=[float(lo[1]),bottom_Y],nondegenerate_interval='lower endpoint inclusive; upper endpoint excluded, degenerate zeroheight only algebraic limit',positive_depth_all_endpoints=True,gap_endpoints=endpoints,projective_difference_numerator_linear_slope=slope,projective_difference_numerator_intercept=intercept,numerator_at_interval_endpoints=nend,strict_positive_everywhere=True),additional_67_sample_pure_math_parity=True,lower_roles=rows))
 # Whole-frame display only: raw RGB untouched beside copied lower-role overlay.
 rawpath=Path(frame['raw_RGB_copy']['path']);assert bind(rawpath)==frame['raw_RGB_copy'];rgb=Image.open(rawpath).convert('RGB');assert hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()==frame['RGB_decoded_sha256']
 overlay=rgb.copy();draw=ImageDraw.Draw(overlay)
 for row in rows:
  x,y=row['original_unchanged_reference_xy'];draw.ellipse((x-4,y-4,x+4,y+4),outline=(0,255,255),width=2);draw.text((x+6,y-17),str(row['native_slot']),font=font(20),fill=(0,255,255))
 canvas=Image.new('RGB',(1280,560),'white');d=ImageDraw.Draw(canvas);d.text((10,8),frame['id']+' | unchanged whole-assembly LOWER virtual roles2/3',font=font(20),fill='black');canvas.paste(rgb,(0,42));canvas.paste(overlay,(640,42));d.text((10,531),'Source role evidence only; no query/GTerror. No selected baseupper, exact surface-vertex or native D/E gold claim.',font=font(16),fill='black');path=H/(frame['id'].replace(':','_')+'_LOWER_ROLES_SOURCE_ONLY.png');canvas.save(path);panels.append(bind(path))
report=dict(completed=True,PASS=True,policy=bind(policy),code=bind(Path(__file__)),source_facts=bind(factspath),original_axis_proposal=bind(axispath),original_named32=bind(namedpath),original_CF_rule=bind(contract),no_initializer_query_or_GTerror_body_read=True,no_upper_boundary_or_point_snapping_or_optimization=True,
 certified_contract='Conditional on existing UP=-OBJY and FULL ASSEMBLY X/Z AABB footprint, the original native2/3 lower virtual references and original_CF role identities are invariant for EVERY possible upper height above fixed bottom plane, in all4 original cameras.',
 not_certified='This is not a recovered standalone palletbase horizontal footprint, original detector target ontology certificate, exact physical corner/surface point, or independent human visibility gold. Fullcrate native0/1 cannot be called baseupper without independent part boundary evidence.',
 fullassembly_raw_XZ_bounds_m=[lo[[0,2]].tolist(),hi[[0,2]].tolist()],fixed_raw_bottom_Y_m=bottom_Y,raw_mesh_maxY_m=float(hi[1]),all4_frames8_lower_roles_preserved=True,frames=frames,figures=panels,wholecrate_original32_coordinates_labels_files_unchanged=True,
 usable_partial_bridge='Retain wholecrate source references. Permit independently reviewed LAB native2/3 as whole-assembly lower virtual footprint role weak supervision after entity/role association is accepted from original fullframe structural evidence. Keep native0/1 and uncertified roles unavailable for base-specific supervision. Do not require chosen virtual point to lie exactly on scan mesh.',
 no_new_model_or_pose_fit_PnP_current223_V112_access=True)
with (H/'LOWER_ROLE_INVARIANCE_CERTIFICATE.json').open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(completed=True,PASS=True,certificate=bind(H/'LOWER_ROLE_INVARIANCE_CERTIFICATE.json'),gaps=[[r['front_top_X_gap']for r in f['front_order_top_X_continuous_interval_proof']['gap_endpoints']]for f in frames],figures=panels)))
