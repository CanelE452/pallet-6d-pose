"""Provided-pose, model-free MR6D source references using exact original V74 functions."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont,__version__ as PIL_VERSION
V=Path(__file__).resolve().parents[1];P=V.parent;H=Path(__file__).resolve().parent;O=H/'provided_reference_v1'
SRC=P/'query_visibility_mr6d_public_native_intake_20261024_v151';F=SRC/'factual_sample_v1';D=SRC/'provided_metadata'
V74=P/'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
POLICY='a941442d93e0a49554be86716fd324a4ad3bed01d629abb2d2ed77212bce7270'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ast_contract():
 text=V74.read_text();tree=ast.parse(text);wanted={'FIXED_ZUP_SIGNS','VERTICAL_FIXED_FACES','ALL_DYNAMIC_FACES','NATIVE_NAMES'};funcs={'outward_normals','normal_facing_permutation','convex_self_visible_proxy'};nodes=[]
 for n in tree.body:
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in wanted for t in n.targets):nodes.append(n)
  if isinstance(n,ast.FunctionDef)and n.name in funcs:nodes.append(n)
 module=ast.Module(body=nodes,type_ignores=[]);ns={'np':np};exec(compile(module,str(V74),'exec'),ns)
 assert funcs.issubset(ns)and wanted.issubset(ns)
 return ns,dict(original_code=bind(V74),pure_selected_AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),functions=sorted(funcs),constants=sorted(wanted),exact_original_AST_no_math_edits=True)
def project(points,R,t,K):
 cam=np.asarray(points,np.float64)@R.T+t;ip=cam@K.T;return ip[:,:2]/ip[:,2:],cam

def dashed(draw,a,b,fill,width=2):
 a=np.asarray(a,float);b=np.asarray(b,float);dist=np.linalg.norm(b-a)
 if not math.isfinite(dist)or dist==0:return
 for t in np.arange(0,dist,13):
  x=a+(b-a)*t/dist;y=a+(b-a)*min(t+7,dist)/dist;draw.line((tuple(x),tuple(y)),fill=fill,width=width)
def draw_instance(raw,instances,full_only=None):
 im=raw.copy();dr=ImageDraw.Draw(im);W,H=im.size;font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
 colors=[(0,205,255),(255,143,30),(245,70,195),(85,235,85)]
 for pos,x in enumerate(instances):
  if full_only is not None and x['gt_instance_index']!=full_only:continue
  col=colors[pos%len(colors)];p=np.asarray(x['dynamic_projected8']);valid=np.asarray(x['inframe8']);i=x['gt_instance_index']
  for a,b in[(0,1),(1,2),(2,3),(3,0),(0,4),(1,5),(2,6),(3,7)]:
   if valid[a]and valid[b]:dr.line((tuple(p[a]),tuple(p[b])),fill=col,width=2)
  for a,b in[(4,5),(5,6),(6,7),(7,4)]:
   if valid[a]and valid[b]:dashed(dr,p[a],p[b],col)
  for k,(xy,inside)in enumerate(zip(p,valid)):
   if not inside:continue
   xx,yy=map(float,xy);dr.ellipse((xx-5,yy-5,xx+5,yy+5),fill=col,outline=(20,20,20),width=1)
   label=f'i{i}:n{k}';box=dr.textbbox((0,0),label,font=font);tw=box[2]-box[0]+7;th=box[3]-box[1]+7
   # Marker at reference; label placement affects only a separate visualization copy.
   dx=8 if k in [0,3,4,7] else -tw-8;dy=-th-7 if k in [0,1,4,5] else 8
   tx=max(0,min(W-tw-1,xx+dx));ty=max(0,min(H-th-1,yy+dy));dr.rectangle((tx,ty,tx+tw,ty+th),fill=(10,18,26));dr.text((tx+3,ty-2),label,font=font,fill=col)
  c=x['centroid_projected'];
  if 0<=c[0]<W and 0<=c[1]<H:
   xx,yy=c;dr.line((xx-7,yy,xx+7,yy),fill=col,width=2);dr.line((xx,yy-7,xx,yy+7),fill=col,width=2);dr.text((min(W-100,xx+8),max(0,yy-20)),f'i{i}:centroid',font=small,fill=col,stroke_width=1,stroke_fill='black')
 return im

def main():
 assert sha(V/'ROOT_POLICY_BEFORE_MODEL_OR_REFERENCE_BRIDGE.json')==POLICY
 selection=read(F/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json');download=read(F/'TINY_SOURCE_DOWNLOAD_COMPLETE.json');validation=read(F/'SOURCE_GEOMETRY_AND_INSTANCE_MASK_VALIDATION.json');cad=read(SRC/'native_readiness_review/actual_tiny_audit/RESULTS.json')
 assert download['completed']and download['PASS']and len(selection['selected_frames'])==10 and len(validation['frames'])==10
 assert len(cad['mesh']['canonical8_geometry_support'])==8 and all(x['nearest_triangle_surface_distance']==0 for x in cad['mesh']['canonical8_geometry_support'])
 ns,proof=ast_contract();signs=ns['FIXED_ZUP_SIGNS'];mi=read(D/'models/models_info.json')['1'];dims=np.asarray([mi['size_x'],mi['size_y'],mi['size_z']],np.float64);center=np.asarray([mi['min_x'],mi['min_y'],mi['min_z']],np.float64)+dims/2;fixed=signs*dims/2+center
 assert np.array_equal(center,np.zeros(3))and np.array_equal(dims,[800,1200,144]);assert not O.exists();O.mkdir()
 refs=[];centroids=[];frames=[];all_instances=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
 for fi,(s,v)in enumerate(zip(selection['selected_frames'],validation['frames'])):
  assert(fi,v['scene'],v['frame_key'])==(v['selection_index'],s['scene'],s['frame_key'])
  source_id=f"{s['split']}/{s['scene']}:{s['frame_key']}";frameid=f"V152_MR6D_{s['split']}_{s['scene']}_{int(s['frame_key']):06d}";gtpath=D/s['split']/s['scene']/'scene_gt.json';capath=D/s['split']/s['scene']/'scene_camera.json';gt=read(gtpath)[s['frame_key']];ca=read(capath)[s['frame_key']];K=np.asarray(ca['cam_K'],np.float64).reshape(3,3)
  rgbpath=Path(v['RGB']['path']);assert sha(rgbpath)==v['RGB']['sha256'];raw=Image.open(rgbpath).convert('RGB');arr=np.array(raw);assert hashlib.sha256(arr.tobytes()).hexdigest()==v['decoded_RGB_sha256'];W,HH=raw.size;instances=[]
  for ix in s['obj1_gt_instance_indices']:
   g=gt[ix];assert g['obj_id']==1;R=np.asarray(g['cam_R_m2c'],np.float64).reshape(3,3);t=np.asarray(g['cam_t_m2c'],np.float64).reshape(3);xy,cam=project(fixed,R,t,K);cxy,ccam=project(center[None],R,t,K);camera_obj=np.linalg.solve(R,-t)
   perm,cos=ns['normal_facing_permutation'](fixed,xy,camera_obj);assert sorted(perm.tolist())==list(range(8));dynamic=xy[perm];d3=fixed[perm];dcam=cam[perm];selfproxy=ns['convex_self_visible_proxy'](d3,camera_obj);marg=float(np.sort(cos)[-1]-np.sort(cos)[-2]);inside=(dcam[:,2]>0)&(dynamic[:,0]>=0)&(dynamic[:,0]<W)&(dynamic[:,1]>=0)&(dynamic[:,1]<HH);facing=int(np.argmax(cos));lr=float(dynamic[1,0]-dynamic[0,0]);assert lr>=0
   instance={'frame_index':fi,'id':frameid,'source':'MR6D','source_id':source_id,'split':'SOURCE_DEVELOPMENT','official_split':s['split'],'scene':s['scene'],'frame_key':s['frame_key'],'gt_instance_index':ix,'obj_id':1,
    'provided_R_m2c':R.tolist(),'provided_t_m2c_mm':t.tolist(),'provided_K':K.tolist(),'camera_in_object_coordinates_exact_linear_inverse':camera_obj.tolist(),'camera_Rtranspose_inverse_difference_mm':float(np.linalg.norm(camera_obj-(-R.T@t))),
    'canonical_fixed_corners_mm':fixed.tolist(),'canonical_fixed_projected8':xy.tolist(),'dynamic_to_fixed_permutation':perm.tolist(),'normal_face_scores':cos.tolist(),'chosen_face_index':facing,'chosen_face_fixed_indices':ns['VERTICAL_FIXED_FACES'][facing].tolist(),'front_runner_margin':marg,'front_top_imageX_separation_px':lr,
    'exact_tie_policy_original_writer_uncertified':bool(marg<=1e-10),'dynamic_projected8':dynamic.tolist(),'dynamic_camera_Z8_mm':dcam[:,2].tolist(),'inframe8':inside.tolist(),'convex_self_visible_proxy8_only':selfproxy.tolist(),'centroid_projected':cxy[0].tolist(),'centroid_camera_Z_mm':float(ccam[0,2]),
    'structural_reference_uncertainty':'Nominal CAD corner lies on supplied mesh, but actual dimensions and supplied pose are low-centimeter approximate. Reference_xy is deterministic nominal projection, not certified <=3px physical gold. Upright configurations retain source deck +Z independent of image gravity.',
    'source_visible_masks_nonexclusive':bool(any(x['visible_mask_overlap_pixels']>0 for x in v['instance_pair_mask_overlaps'])),'semantic_states_initial':['U_UNREVIEWED']*8,'RGB':v['RGB']}
   for k in range(8):
    refs.append({'frame_index':fi,'id':frameid,'source':'MR6D','source_id':source_id,'split':'SOURCE_DEVELOPMENT','official_split':s['split'],'scene':s['scene'],'frame_key':s['frame_key'],'gt_instance_index':ix,'obj_id':1,'native_slot':k,'native_name':ns['NATIVE_NAMES'][k],
     'fixed_canonical_index':int(perm[k]),'reference_model_XYZ_mm':d3[k].tolist(),'reference_camera_XYZ_mm':dcam[k].tolist(),'reference_xy':dynamic[k].tolist(),'continuous_inframe':bool(inside[k]),'positive_camera_Z':bool(dcam[k,2]>0),
     'front_face_scores':cos.tolist(),'front_runner_margin':marg,'front_top_imageX_separation_px':lr,'dynamic_to_fixed_permutation':perm.tolist(),'convex_self_visible_proxy_only':bool(selfproxy[k]),'semantic_state':'U_UNREVIEWED','human_gold':False,'train_ready':False,
     'reference_point_uncertainty':instance['structural_reference_uncertainty'],'source_visible_mask_nonexclusive':instance['source_visible_masks_nonexclusive']})
   centroids.append({'frame_index':fi,'id':frameid,'source_id':source_id,'gt_instance_index':ix,'centroid_model_XYZ_mm':center.tolist(),'centroid_reference_xy':cxy[0].tolist(),'camera_Z_mm':float(ccam[0,2]),'separately_projected_3D_centroid_not_mean_2D_corners':True})
   instances.append(instance);all_instances.append(instance)
  marked=draw_instance(raw,instances);bp=O/(f'{fi:02d}_{s["scene"]}_{int(s["frame_key"]):06d}_ALL_INSTANCES_NATIVE.png');marked.save(bp)
  # Keep the original RGB file intact, with separate raw-left/marked-right diagnostic panel.
  canvas=Image.new('RGB',(W*2,HH+80),'#edf2f7');dr=ImageDraw.Draw(canvas);dr.text((18,7),f"{fi:02d} {source_id} | PROVIDED POSE/K ONLY | raw left / native references right",font=font,fill='#19334b');dr.text((18,35),'nominal CAD / cm-scale pose uncertainty | native semantic D/E/S unassigned | no model / q / fit',font=font,fill='#52697f');canvas.paste(raw,(0,80));canvas.paste(marked,(W,80));panel=O/(f'{fi:02d}_{s["scene"]}_{int(s["frame_key"]):06d}_RAW_AND_NATIVE.png');canvas.save(panel)
  solo=[]
  for x in instances:
   sp=O/(f'{fi:02d}_{s["scene"]}_{int(s["frame_key"]):06d}_INSTANCE_{x["gt_instance_index"]:02d}.png');draw_instance(raw,instances,x['gt_instance_index']).save(sp);solo.append(bind(sp))
  frames.append({'frame_index':fi,'id':frameid,'source_id':source_id,'official_split':s['split'],'scene':s['scene'],'frame_key':s['frame_key'],'raw_RGB':v['RGB'],'decoded_RGB_sha256':v['decoded_RGB_sha256'],'RGB_decoder':'PIL'+PIL_VERSION+' diagnostic only; not certified R0 canonical input','GT':bind(gtpath),'camera':bind(capath),'instances':instances,'all_instance_overlay':bind(bp),'raw_and_reference_panel':bind(panel),'single_instance_overlays':solo})
 assert len(refs)==104 and len(centroids)==13 and len(frames)==10
 for name,rows in [('ALL104_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl',refs),('ALL13_PROJECTED_CENTROIDS_GT_ONLY.jsonl',centroids)]:
  with(O/name).open('x')as f:
   for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 write(O/'FRAME10_INSTANCE13_GEOMETRY_AND_OVERLAYS.json',{'completed':True,'frames':frames,'native_reference_rows':104,'instances':13,'centroids':13})
 result={'completed':True,'PASS':True,'UTC':datetime.now(timezone.utc).isoformat(),'code':bind(Path(__file__).resolve()),'root_policy':bind(V/'ROOT_POLICY_BEFORE_MODEL_OR_REFERENCE_BRIDGE.json'),'selection':bind(F/'FIXED_SOURCE_SAMPLE_SELECTION_BEFORE_PIXELS.json'),'download':bind(F/'TINY_SOURCE_DOWNLOAD_COMPLETE.json'),'CAD_evidence':bind(SRC/'native_readiness_review/actual_tiny_audit/RESULTS.json'),'exact_original_V74_AST':proof,
 'original_order_contract':bind(P/'query_visibility_loco_native_context_20261023_v74/source_contract/NATIVE_ORDER_AND_INPUT_CONTRACT.json'),
 'references':bind(O/'ALL104_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl'),'centroids':bind(O/'ALL13_PROJECTED_CENTROIDS_GT_ONLY.jsonl'),'frames_and_geometry':bind(O/'FRAME10_INSTANCE13_GEOMETRY_AND_OVERLAYS.json'),
 'counts':{'frames':10,'instances':13,'native_references':104,'centroids':13,'nominal_inframe':sum(r['continuous_inframe']for r in refs),'nominal_OOF':sum(not r['continuous_inframe']for r in refs)},
 'minimum_front_runner_margin':min(x['front_runner_margin']for x in all_instances),'minimum_front_top_LR_px':min(x['front_top_imageX_separation_px']for x in all_instances),
 'all_source_refs_retained':True,'provided_R_T_K_or_pixels_changed':False,'query_or_model_outputs_read':False,'native_semantic_labels_created':0,'new_model_fit_PnP_download_private_access':0,
 'limits':['Projection is nominal supplied-pose evidence, not physical pixel gold.','PositiveZ deck refers source-asset anatomy, not image-up; upright pallet is retained.','Source full/visible masks are not exclusive owner masks or native semantic teachers.','Original writer tie policy remains uncertified although recorded per-instance margins may be positive.'],
 'actual_visual_review_pending':True}
 write(O/'REFERENCE_PROJECTION_COMPLETE.json',result);print(json.dumps({'projection_complete':bind(O/'REFERENCE_PROJECTION_COMPLETE.json'),'counts':result['counts'],'min_margin':result['minimum_front_runner_margin'],'min_LR':result['minimum_front_top_LR_px']}))
if __name__=='__main__':main()
