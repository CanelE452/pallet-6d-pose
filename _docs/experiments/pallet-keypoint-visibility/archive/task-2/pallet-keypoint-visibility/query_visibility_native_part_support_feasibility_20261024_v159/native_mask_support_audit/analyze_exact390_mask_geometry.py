"""All390 TRAIN original-query/nominal-reference mask geometry; no RGB/model/feature reads."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math,traceback
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
O=Path(__file__).resolve().parent;V=O.parent;S=P/'query_visibility_mr6d_full_public_source_20261024_v153'
BINDINGS={};CHECKS=[]
def guard(p,mask=False):
 p=Path(p);s=str(p).lower();assert p.is_absolute() and str(p).startswith(str(P)+'/')
 assert not any(t in s for t in ('v112','current223','/private','/val/000001/'))
 if mask:assert str(p).startswith(str(S/'full_source_v1/original')+'/') and ('/mask/' in s or '/mask_visib/' in s) and p.suffix=='.png'
 else:assert p.suffix in ('.json','.jsonl','.py')
 return p
def bind(p,expected=None,mask=False):
 p=guard(p,mask);b=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 if expected:assert b['sha256']==expected,str(p)
 BINDINGS[str(p)]=b;return b
def read(p):bind(p);return json.loads(guard(p).read_text())
def readb(b):bind(b['path'],b['sha256']);return json.loads(Path(b['path']).read_text())
def lineb(b):bind(b['path'],b['sha256']);return [json.loads(s) for s in Path(b['path']).read_text().splitlines() if s]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def clean(x):
 if isinstance(x,np.generic):x=x.item()
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 return x
def write(p,x):
 with Path(p).open('x') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def writejl(p,rr):
 with Path(p).open('x') as f:
  for r in rr:f.write(json.dumps(clean(r),ensure_ascii=False,allow_nan=False)+'\n')
def desc(xs):
 v=np.asarray([x for x in xs if x is not None],float);v=v[np.isfinite(v)]
 return dict(n=len(v),**({k:float(x) for k,x in zip(('min','p25','median','p75','max'),np.quantile(v,[0,.25,.5,.75,1]))} if len(v) else {}))
def signed_distance(mask):
 if not mask.any() or mask.all():return None
 return np.where(mask,distance_transform_edt(mask),-distance_transform_edt(~mask))
def point(xy,masks,selected,distances):
 h,w=next(iter(masks.values()))['full'].shape;x,y=map(float,xy)
 inside=math.isfinite(x+y) and 0<=x<=w-1 and 0<=y<=h-1
 result=dict(xy=[x,y],continuous_in_image=inside,nearest_pixel=None,selected_full=None,selected_visible=None,
   other_full_count=None,other_visible_count=None,selected_and_other_visible_overlap=None,selected_and_other_full_overlap=None,
   selected_full_signed_opposite_pixel_distance=None,selected_visible_signed_opposite_pixel_distance=None)
 if not inside:return result
 ix,iy=int(math.floor(x+.5)),int(math.floor(y+.5));result['nearest_pixel']=[ix,iy]
 full=bool(masks[selected]['full'][iy,ix]);vis=bool(masks[selected]['visible'][iy,ix])
 of=sum(bool(a['full'][iy,ix]) for j,a in masks.items() if j!=selected);ov=sum(bool(a['visible'][iy,ix]) for j,a in masks.items() if j!=selected)
 result.update(selected_full=full,selected_visible=vis,other_full_count=of,other_visible_count=ov,
   selected_and_other_visible_overlap=vis and ov>0,selected_and_other_full_overlap=full and of>0)
 for kind in ('full','visible'):
  d=distances[(selected,kind)];result['selected_'+kind+'_signed_opposite_pixel_distance']=None if d is None else float(d[iy,ix])
 # Nonzero-weight originalimage bilinear footprint,without altering source coordinates.
 x0,y0=math.floor(x),math.floor(y);fx,fy=x-x0,y-y0;foot=[]
 for xx,wx in ((x0,1-fx),(x0+1,fx)):
  for yy,wy in ((y0,1-fy),(y0+1,fy)):
   weight=wx*wy
   if weight>0:
    assert 0<=xx<w and 0<=yy<h
    foot.append(dict(pixel=[xx,yy],weight=weight,selected_full=bool(masks[selected]['full'][yy,xx]),selected_visible=bool(masks[selected]['visible'][yy,xx]),
      other_full_count=sum(bool(a['full'][yy,xx]) for j,a in masks.items() if j!=selected),
      other_visible_count=sum(bool(a['visible'][yy,xx]) for j,a in masks.items() if j!=selected)))
 result['bilinear_footprint']=foot
 result['bilinear_selected_full_weight']=sum(z['weight']*z['selected_full'] for z in foot)
 result['bilinear_selected_visible_weight']=sum(z['weight']*z['selected_visible'] for z in foot)
 result['bilinear_any_other_visible_weight']=sum(z['weight']*(z['other_visible_count']>0) for z in foot)
 result['bilinear_selected_visible_any_touch']=any(z['selected_visible'] for z in foot)
 result['bilinear_selected_visible_all_touch']=all(z['selected_visible'] for z in foot)
 return result
def window(p,side,masks,selected):
 if p['nearest_pixel'] is None:return dict(side=side,available=False)
 ix,iy=p['nearest_pixel'];h,w=masks[selected]['full'].shape;r=side//2;x0,x1=max(0,ix-r),min(w,ix+r+1);y0,y1=max(0,iy-r),min(h,iy+r+1)
 sf=masks[selected]['full'][y0:y1,x0:x1];sv=masks[selected]['visible'][y0:y1,x0:x1]
 others=[a['visible'][y0:y1,x0:x1] for j,a in masks.items() if j!=selected];ov=np.logical_or.reduce(others) if others else np.zeros_like(sv)
 return dict(side=side,available=True,valid_pixels=int(sf.size),selected_full_pixels=int(sf.sum()),selected_visible_pixels=int(sv.sum()),other_visible_union_pixels=int(ov.sum()),
  selected_and_other_visible_overlap_pixels=int((sv&ov).sum()),selected_full_fraction=float(sf.mean()),selected_visible_fraction=float(sv.mean()),other_visible_union_fraction=float(ov.mean()))

assert not (O/'MASK_ANALYSIS_STARTED.json').exists(),'Preserve existing attempt; no blind retry.'
seal=read(O/'CASE_AND_CALCULATION_SEAL_BEFORE_MASK_BODIES.json');rules=readb(seal['rules'])
mapping_complete=read(O/'EXACT390_MAPPING_COMPLETE_BEFORE_MASK_BODIES.json');mapping=lineb(mapping_complete['mapping']);frames=readb(mapping_complete['frame_geometry'])['frames']
assert len(mapping)==390 and len(frames)==88 and Counter(r['original_role_state'] for r in mapping)==Counter(D=341,E=49)
assert all(r['split']=='SOURCE_DEVELOPMENT' for r in mapping+frames)
write(O/'MASK_ANALYSIS_STARTED.json',dict(completed=False,code=bind(__file__),case_and_calculation_seal=bind(O/'CASE_AND_CALCULATION_SEAL_BEFORE_MASK_BODIES.json'),mapping=mapping_complete['mapping'],automatic_retry=False))
try:
 refcomplete=read(S/'native_reference_preparation/provided_reference_v1/REFERENCE_PROJECTION_COMPLETE.json')
 bind(refcomplete['code']['path'],refcomplete['code']['sha256'])
 cad=readb(refcomplete['CAD_evidence']);nominal_mesh={r['fixed_Zup_index']:r for r in cad['mesh']['canonical8_geometry_support']}
 assert len(nominal_mesh)==8 and all(r['nearest_mesh_vertex_distance']==r['nearest_triangle_surface_distance']==0 for r in nominal_mesh.values())
 valid_path=P/'query_visibility_train_query_owner_witness_20261024_v158/review_evidence_preparation/CENTER4_VALIDITY_INPUT_ONLY_SIDECAR.json'
 valid=read(valid_path);savedvalid={r['global_index']:r for r in valid['cases']}
 byframe=defaultdict(list)
 for r in mapping:byframe[r['frame_index']].append(r)
 rows=[];maskfacts=[];framefacts=[]
 for fr in frames:
  fi=fr['frame_index'];assert fr['split']=='SOURCE_DEVELOPMENT';hw=fr['raw_hw'];masks={};distances={}
  facts={r['gt_instance_index']:r for r in fr['factual_summary']['instances']}
  for mb in fr['mask_bindings']:
   inst=mb['gt_instance_index'];masks[inst]={}
   for kind in ('full','visible'):
    b=mb[kind+'_mask'];bind(b['path'],b['sha256'],mask=True)
    a=np.asarray(Image.open(b['path']));assert list(a.shape)==hw and a.dtype==np.uint8 and set(np.unique(a))<={0,255}
    mask=a!=0;masks[inst][kind]=mask
    assert int(mask.sum())==facts[inst][kind+'_pixels']
    maskfacts.append(dict(frame_index=fi,gt_instance_index=inst,mask_kind=kind,original_file_binding=b,decoded_shape=list(a.shape),decoded_dtype=str(a.dtype),
       unique_values=np.unique(a).tolist(),decoded_uint8_sha256=ash(a),support_pixels=int(mask.sum()),original_factual_total_exact=True))
    distances[(inst,kind)]=signed_distance(mask)
   assert int((masks[inst]['visible']&~masks[inst]['full']).sum())==facts[inst]['visible_outside_full']
  pairs=[]
  for pair in fr['factual_summary']['instance_pair_overlaps']:
   ia,ib=pair['GT_indices'];fv=int((masks[ia]['full']&masks[ib]['full']).sum());vv=int((masks[ia]['visible']&masks[ib]['visible']).sum())
   assert fv==pair['full_overlap'] and vv==pair['visible_overlap'];pairs.append(dict(instance_indices=[ia,ib],full_overlap_pixels=fv,visible_overlap_pixels=vv))
  framefacts.append(dict(frame_index=fi,source_id=fr['source_id'],instances=len(masks),all_instances_obj_id=1,pair_overlaps=pairs,
    nonexclusive_visible_overlap_present=any(p['visible_overlap_pixels']>0 for p in pairs),all_original_factual_counts_and_overlaps_exact=True))
  for r in byframe[fi]:
   gt=r['selected_gt_instance_index'];q=np.asarray(r['query_xy'],float);ref=np.asarray(r['reference_xy'],float);L=r['L']
   qp=point(q,masks,gt,distances);rp=point(ref,masks,gt,distances);center=[]
   for iy in (47,48):
    for ix in (47,48):
     xy=q+(np.array([ix+.5,iy+.5])/96-.5)*L;p=point(xy,masks,gt,distances)
     p['RGB96_index_yx']=[iy,ix];p['saved_crop_valid']=savedvalid[r['global_index']]['valid2x2_bits'][iy-47][ix-47];center.append(p)
   segment=[point(q+(ref-q)*t,masks,gt,distances) for t in np.linspace(0,1,33)]
   sv=[z for z in segment if z['continuous_in_image']]
   dx,dy=ref-q;cheb=max(abs(dx),abs(dy))/L;mesh=nominal_mesh[r['fixed_canonical_index']]
   row={k:r[k] for k in ('global_index','query_id','frame_index','source_id','split','native_slot','query_xy','L','original_role_state','original_role_target','weak_visible_owner_support','center4_valid','selected_gt_instance_index','native_name','fixed_canonical_index','reference_xy','reference_model_XYZ_mm','reference_continuous_inframe','reference_point_uncertainty')}
   row.update(query_mask_geometry=qp,nominal_reference_mask_geometry=rp,
     query_windows=[window(qp,k,masks,gt) for k in (5,17)],nominal_reference_windows=[window(rp,k,masks,gt) for k in (5,17)],
     center4_samples=center,center4_saved_valid_bits=savedvalid[r['global_index']]['valid2x2_bits'],center4_saved_valid_count=savedvalid[r['global_index']]['valid_count'],
     center4_selected_full_nearest_count=sum(z['selected_full'] is True for z in center),center4_selected_visible_nearest_count=sum(z['selected_visible'] is True for z in center),
     center4_other_visible_nearest_count=sum((z['other_visible_count'] or 0)>0 for z in center),center4_selected_and_other_visible_overlap_count=sum(z['selected_and_other_visible_overlap'] is True for z in center),
     nominal_reference_dx=float(dx),nominal_reference_dy=float(dy),nominal_reference_distance=float(np.linalg.norm(ref-q)),nominal_reference_Chebyshev_over_L=float(cheb),
     nominal_reference_inside96=cheb<=.5,nominal_reference_inside32=cheb<=1/6,
     segment33=dict(valid_samples=len(sv),selected_full_count=sum(z['selected_full'] is True for z in sv),selected_visible_count=sum(z['selected_visible'] is True for z in sv),other_visible_count=sum((z['other_visible_count'] or 0)>0 for z in sv),
       selected_and_other_visible_overlap_count=sum(z['selected_and_other_visible_overlap'] is True for z in sv)),
     nominal_model_support=dict(definition='provided models_info AABB extrema + unchanged original dynamic native permutation',
       nearest_nominal_mesh_vertex_distance_mm=mesh['nearest_mesh_vertex_distance'],nearest_nominal_mesh_surface_distance_mm=mesh['nearest_triangle_surface_distance'],
       point_is_vertex_on_provided_nominal_mesh=True,photo_physical_native_material_certification='UNKNOWN'),
     source_masks_nonexclusive=True,mask_instances_cover_obj1_pallets_only=True,actual_native_or_physical_visibility_gold=False,
     point_quality_target=-1,original_q_L_native_role_split_unchanged=True,training_admitted=False)
   rows.append(row)
 rows.sort(key=lambda r:r['global_index']);assert len(rows)==390 and [r['global_index'] for r in rows]==seal['global_indices']
 # Geometry-only preparation file deliberately excludes prior role and owner observations.
 prep=[dict(case_id=f"Q{r['global_index']:04d}",**{k:r[k] for k in ('global_index','query_id','frame_index','source_id','split','native_slot','native_name','query_xy','L','selected_gt_instance_index','reference_xy','reference_model_XYZ_mm')},
  selected_instance_mask_bindings=mapping_by['selected_instance_mask_bindings'],other_instance_mask_bindings=mapping_by['other_instance_mask_bindings'],
  reference_is_nominal_not_pixel_gold=True,source_masks_nonexclusive=True,foreign_cargo_masks_not_available=True,native_material_witness_status='UNKNOWN_PENDING_SOURCE_VISUAL_EVIDENCE')
  for r in rows for mapping_by in [next(z for z in mapping if z['global_index']==r['global_index'])]]
 allrows=O/'ALL390_ORIGINAL_QUERY_NOMINAL_REFERENCE_MASK_GEOMETRY.jsonl';writejl(allrows,rows)
 writejl(O/'ALL390_NATIVE_MATERIAL_WITNESS_GEOMETRY_INPUTS.jsonl',prep)
 flat=[]
 for r in rows:
  q=r['query_mask_geometry'];p=r['nominal_reference_mask_geometry'];f={k:r[k] for k in ('global_index','query_id','frame_index','source_id','split','native_slot','query_xy','L','original_role_state','original_role_target','weak_visible_owner_support','selected_gt_instance_index','reference_xy','nominal_reference_inside96','nominal_reference_inside32','center4_saved_valid_count','center4_selected_full_nearest_count','center4_selected_visible_nearest_count','center4_other_visible_nearest_count','center4_selected_and_other_visible_overlap_count')}
  for prefix,a in [('q',q),('ref',p)]:
   for k in ('continuous_in_image','selected_full','selected_visible','other_full_count','other_visible_count','selected_and_other_visible_overlap','selected_and_other_full_overlap','selected_full_signed_opposite_pixel_distance','selected_visible_signed_opposite_pixel_distance','bilinear_selected_full_weight','bilinear_selected_visible_weight'):f[prefix+'_'+k]=a.get(k)
  flat.append(f)
 with (O/'ALL390_MASK_GEOMETRY_SUMMARY.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in flat)
 groupings=defaultdict(list)
 for r in rows:
  for k in ('ALL','ROLE_'+r['original_role_state'],'NATIVE_'+str(r['native_slot']),'ROLE_'+r['original_role_state']+'::NATIVE_'+str(r['native_slot']),'OWNER_'+r['weak_visible_owner_support']):groupings[k].append(r)
 aggregates={}
 for name,rr in groupings.items():
  a=dict(rows=len(rr),frames=len({r['frame_index'] for r in rr}),nominal_ref_inside96=sum(r['nominal_reference_inside96'] for r in rr),nominal_ref_inside32=sum(r['nominal_reference_inside32'] for r in rr))
  for prefix,key in [('q','query_mask_geometry'),('ref','nominal_reference_mask_geometry')]:
   pp=[r[key] for r in rr]
   a[prefix]=dict(in_image=sum(p['continuous_in_image'] for p in pp),selected_full_nearest=sum(p['selected_full'] is True for p in pp),selected_visible_nearest=sum(p['selected_visible'] is True for p in pp),
    any_other_full_nearest=sum((p['other_full_count'] or 0)>0 for p in pp),any_other_visible_nearest=sum((p['other_visible_count'] or 0)>0 for p in pp),
    selected_and_other_full_overlap=sum(p['selected_and_other_full_overlap'] is True for p in pp),selected_and_other_visible_overlap=sum(p['selected_and_other_visible_overlap'] is True for p in pp),
    selected_visible_bilinear_any_touch=sum(p.get('bilinear_selected_visible_any_touch',False) for p in pp),
    selected_visible_bilinear_all_touch=sum(p.get('bilinear_selected_visible_all_touch',False) for p in pp),
    selected_full_signed_distance=desc([p['selected_full_signed_opposite_pixel_distance'] for p in pp]),selected_visible_signed_distance=desc([p['selected_visible_signed_opposite_pixel_distance'] for p in pp]))
  a['center4_selected_visible_support_count_histogram']=dict(Counter(r['center4_selected_visible_nearest_count'] for r in rr));a['center4_saved_valid_count_histogram']=dict(Counter(r['center4_saved_valid_count'] for r in rr))
  a['both_q_and_ref_selected_visible_nearest']=sum(r['query_mask_geometry']['selected_visible'] is True and r['nominal_reference_mask_geometry']['selected_visible'] is True for r in rr)
  a['q_not_selected_visible_ref_is_selected_visible']=sum(r['query_mask_geometry']['selected_visible'] is False and r['nominal_reference_mask_geometry']['selected_visible'] is True for r in rr)
  aggregates[name]=a
 write(O/'MASK_GEOMETRY_AGGREGATES.json',aggregates);write(O/'TRAIN88_MASK_TOTALS_AND_OVERLAPS.json',dict(frames=framefacts,mask_files=maskfacts))
 CHECKS=[dict(name='All390 original identity,q/L,native,split,weakD341/E49 fixed',PASS=True),dict(name='Only88selectedTRAIN frames mask files read;noRGB/depth/features/weights/heldscene1 files',PASS=True),
   dict(name='AlloriginalmaskfileSHA,binaryuint8shape,supporttotals,visibleoutsidefull andpair overlaps exact',PASS=True),
   dict(name='Exactsource selectedGT andsame-native reference used;no optimizedinstance/native reassignment',PASS=True),
   dict(name='ProvidednominalCAD all8corners aremeshvertices/surface;physicalphotocorrespondencestaysUNKNOWN',PASS=True)]
 limits=['Fullandvisibleinstance masks are nonexclusive;their overlap is explicitlyretained.They cannot provide exclusivepixelownership/nativeD/E gold.',
   'Maskscover obj_id1 palletinstances only.Foreigncargo/KLT/tub/background classes are notexhaustivelymasked;outsideallpalletmasks doesnotimplybackground.',
   'Provided800x1200x144nominalmesh bboxcorners coincide withmeshvertices,butactualpalletdimensions/pose arecmuncertain.Masktouch/non-touch cannot certify a physicalnativepart/point.',
   'No originalq/L/native/weakD/E/ownerlabel/split changed.No physical,pixel orcoordinatequalitygold andnoautomaticfitadmissioncreated.',
   'Nearestandbilinear masksamples,localwindow fractions,binarymaskdistance andsegmentcounts are descriptive rastergeometry,not physicalocclusion orcausal evidence.']
 outputs=[bind(allrows),bind(O/'ALL390_MASK_GEOMETRY_SUMMARY.csv'),bind(O/'ALL390_NATIVE_MATERIAL_WITNESS_GEOMETRY_INPUTS.jsonl'),bind(O/'MASK_GEOMETRY_AGGREGATES.json'),bind(O/'TRAIN88_MASK_TOTALS_AND_OVERLAPS.json')]
 complete=dict(completed=True,PASS=True,source_only=True,rows=390,frames=88,original_roles={'D':341,'E':49},all390_retained=True,
  selected_instance_and_native_unchanged=True,original_qL_split_roles_and_weakowner_unchanged=True,source_bindings=list(BINDINGS.values()),
  code=bind(__file__),rules=seal['rules'],case_seal=bind(O/'CASE_AND_CALCULATION_SEAL_BEFORE_MASK_BODIES.json'),exact_mapping=mapping_complete['mapping'],
  mask_files_read=len(maskfacts),RGB_or_depth_or_fullfeature_or_weight_bodies_read=0,held_scene1_mask_or_RGB_or_label_records_analyzed=0,
  counts=aggregates,nonexclusive_mask_frames=sum(r['nonexclusive_visible_overlap_present'] for r in framefacts),
  provided_nominal_mesh_corner_support=nominal_mesh,native_physical_material_certification='UNKNOWN_FOR_ALL390_FROM_MASK_GEOMETRY_ALONE',
  checks=CHECKS,semantic_limits=limits,outputs=outputs,point_quality_targets_all_minus1=True,new_native_or_D_E_labels_created=0,
  concrete_preparation_action='Preparedall390exactselected-instance/native geometry-only witnessinputs,includingselected/othermaskbindingsandnominalrefs,forfixedsourcevisualnative-part feasibilityinspection.Nopriorrole/ownerobservationsinthepreparationinputs.',
  new_model_or_fit_or_private_or_V112_or_PnP_qL_calls=0,training_admitted=False,selected_model=None,goal_status='active_unsolved')
 write(O/'COMPLETE.json',complete)
 print(json.dumps(dict(completed=True,complete=bind(O/'COMPLETE.json'),mask_files=len(maskfacts),all=aggregates['ALL'],roles={k:aggregates[k] for k in ('ROLE_D','ROLE_E')}),indent=2))
except Exception as e:
 write(O/'FAILED.json',dict(completed=False,error=str(e),traceback=traceback.format_exc(),automatic_retry=False));raise
