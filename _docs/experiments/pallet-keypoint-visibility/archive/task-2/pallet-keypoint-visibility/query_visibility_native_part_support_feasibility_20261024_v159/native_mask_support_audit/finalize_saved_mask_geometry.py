"""Repair output-CSV packaging only; audit saved calculations without re-reading masks/RGB."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
O=Path(__file__).resolve().parent;V=O.parent;B={};CHECKS=[]
def guard(p):
 p=Path(p);s=str(p).lower();assert p.is_absolute() and str(p).startswith(str(O)+'/')
 assert not any(t in s for t in ('v112','current223','/private'))
 assert p.suffix in ('.json','.jsonl','.csv','.py');return p
def bind(p,expected=None):
 p=guard(p);b=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 if expected:assert b['sha256']==expected
 B[str(p)]=b;return b
def read(p):bind(p);return json.loads(guard(p).read_text())
def lines(p):bind(p);return [json.loads(s) for s in guard(p).read_text().splitlines() if s]
def ck(name,c,count=None):CHECKS.append(dict(name=name,PASS=bool(c),rows=count));assert c,name
def desc(v):
 v=sorted(x for x in v if x is not None and math.isfinite(x));n=len(v)
 def quant(q):
  pos=(n-1)*q;lo=math.floor(pos);hi=math.ceil(pos);return v[lo]+(v[hi]-v[lo])*(pos-lo)
 return dict(n=n,**({k:quant(q) for k,q in zip(('min','p25','median','p75','max'),(0,.25,.5,.75,1))} if n else {}))
assert not (O/'COMPLETE.json').exists()
failure=read(O/'FAILED.json');started=read(O/'MASK_ANALYSIS_STARTED.json')
ck('Failure is outputCSV hash-guard packaging after mask calculations',"outputs=[bind(allrows)" in failure['traceback'] and "assert p.suffix in ('.json','.jsonl','.py')" in failure['traceback'])
bind(started['code']['path'],started['code']['sha256'])
seal=read(O/'CASE_AND_CALCULATION_SEAL_BEFORE_MASK_BODIES.json');mapping_meta=read(O/'EXACT390_MAPPING_COMPLETE_BEFORE_MASK_BODIES.json')
mapping=lines(Path(mapping_meta['mapping']['path']));bind(mapping_meta['mapping']['path'],mapping_meta['mapping']['sha256'])
frames=read(Path(mapping_meta['frame_geometry']['path']))['frames'];bind(mapping_meta['frame_geometry']['path'],mapping_meta['frame_geometry']['sha256'])
rows=lines(O/'ALL390_ORIGINAL_QUERY_NOMINAL_REFERENCE_MASK_GEOMETRY.jsonl');flat=list(csv.DictReader((O/'ALL390_MASK_GEOMETRY_SUMMARY.csv').open()))
prep=lines(O/'ALL390_NATIVE_MATERIAL_WITNESS_GEOMETRY_INPUTS.jsonl');stats=read(O/'MASK_GEOMETRY_AGGREGATES.json');maskfacts=read(O/'TRAIN88_MASK_TOTALS_AND_OVERLAPS.json')
lookup={r['global_index']:r for r in mapping};final={r['global_index']:r for r in rows}
ck('All390 rows exact IDs/order preserved',len(rows)==len(final)==390 and [r['global_index'] for r in rows]==seal['global_indices'] and set(final)==set(lookup),390)
ck('CSV and future geometry-input rows preserve390 identities',len(flat)==len(prep)==390 and [int(r['global_index']) for r in flat]==seal['global_indices'] and [r['global_index'] for r in prep]==seal['global_indices'],390)
ck('All88TRAIN frames retained',len(frames)==len(maskfacts['frames'])==88 and len({r['frame_index'] for r in rows})==88)
ck('Original D341/E49 unchanged',Counter(r['original_role_state'] for r in rows)==Counter(D=341,E=49))
for i,r in final.items():
 m=lookup[i]
 for k in ('query_id','frame_index','source_id','split','native_slot','query_xy','L','original_role_state','original_role_target','weak_visible_owner_support','selected_gt_instance_index','reference_xy','fixed_canonical_index'):
  assert r[k]==m[k]
 assert r['native_slot']==i%8 and r['split']=='SOURCE_DEVELOPMENT'
 assert r['nominal_reference_mask_geometry']['xy']==m['reference_xy'] and r['query_mask_geometry']['xy']==m['query_xy']
 assert m['reference_xy']==m['selected_instance_geometry']['dynamic_projected8'][r['native_slot']]
 assert r['point_quality_target']==-1 and not r['training_admitted'] and r['original_q_L_native_role_split_unchanged']
 assert r['source_masks_nonexclusive'] and r['mask_instances_cover_obj1_pallets_only'] and not r['actual_native_or_physical_visibility_gold']
 assert r['nominal_model_support']['nearest_nominal_mesh_vertex_distance_mm']==r['nominal_model_support']['nearest_nominal_mesh_surface_distance_mm']==0
 assert r['nominal_model_support']['photo_physical_native_material_certification']=='UNKNOWN'
 for p in [r['query_mask_geometry'],r['nominal_reference_mask_geometry']]+r['center4_samples']:
  if p['continuous_in_image']:
   assert p['nearest_pixel']==[math.floor(p['xy'][0]+.5),math.floor(p['xy'][1]+.5)]
   assert abs(sum(z['weight'] for z in p['bilinear_footprint'])-1)<1e-12
   assert abs(p['bilinear_selected_visible_weight']-sum(z['weight']*z['selected_visible'] for z in p['bilinear_footprint']))<1e-12
  else:assert p['nearest_pixel'] is None and p['selected_visible'] is None
 assert r['center4_saved_valid_count']==sum(sum(v) for v in r['center4_saved_valid_bits'])
ck('Exactexisting selectedinstance/same-native,qL/owner/roles/native/split remainfixed',True,390)
ck('Point nearest/bilinear geometry internally consistent;out-of-image not background',True,390)
ck('All390 physical/native/pixel certification UNKNOWN/noqualitygold/noadmission',True,390)
ck('Futuregeometry-only reviewinputs hide priorroles/owner',all('original_role_state' not in r and 'weak_visible_owner_support' not in r for r in prep),390)
expected_masks={b['path']:b for fr in frames for inst in fr['mask_bindings'] for b in (inst['full_mask'],inst['visible_mask'])}
actual_masks={r['original_file_binding']['path']:r for r in maskfacts['mask_files']}
ck('Allselected88TRAIN source maskfile records complete',set(actual_masks)==set(expected_masks))
frame_input={r['frame_index']:r for r in frames}
for p,m in actual_masks.items():
 expected=expected_masks[p];assert m['original_file_binding']==expected and '/val/000001/' not in p
 fr=frame_input[m['frame_index']];inst=next(v for v in fr['factual_summary']['instances'] if v['gt_instance_index']==m['gt_instance_index'])
 assert m['decoded_shape']==fr['raw_hw'] and m['decoded_dtype']=='uint8' and set(m['unique_values'])<={0,255}
 assert m['support_pixels']==inst[m['mask_kind']+'_pixels'] and m['original_factual_total_exact']
for fr in maskfacts['frames']:
 expected=frame_input[fr['frame_index']]['factual_summary']['instance_pair_overlaps']
 assert len(fr['pair_overlaps'])==len(expected)
 for p,q in zip(fr['pair_overlaps'],expected):assert p['instance_indices']==q['GT_indices'] and p['full_overlap_pixels']==q['full_overlap'] and p['visible_overlap_pixels']==q['visible_overlap']
ck('Saved decodedmask dimensions,binaryvalues,totals andpair overlaps matchboundintake',True,len(actual_masks))
groups=defaultdict(list)
for r in rows:
 for k in ('ALL','ROLE_'+r['original_role_state'],'NATIVE_'+str(r['native_slot']),'ROLE_'+r['original_role_state']+'::NATIVE_'+str(r['native_slot']),'OWNER_'+r['weak_visible_owner_support']):groups[k].append(r)
for name,rr in groups.items():
 a=stats[name];assert a['rows']==len(rr) and a['frames']==len({r['frame_index'] for r in rr})
 assert a['nominal_ref_inside96']==sum(r['nominal_reference_inside96'] for r in rr) and a['nominal_ref_inside32']==sum(r['nominal_reference_inside32'] for r in rr)
 for prefix,key in [('q','query_mask_geometry'),('ref','nominal_reference_mask_geometry')]:
  pp=[r[key] for r in rr];z=a[prefix]
  for field,pkey in [('selected_full_nearest','selected_full'),('selected_visible_nearest','selected_visible'),('selected_and_other_full_overlap','selected_and_other_full_overlap'),('selected_and_other_visible_overlap','selected_and_other_visible_overlap')]:assert z[field]==sum(p[pkey] is True for p in pp)
  assert z['in_image']==sum(p['continuous_in_image'] for p in pp)
  assert z['any_other_visible_nearest']==sum((p['other_visible_count'] or 0)>0 for p in pp)
  for k in ('full','visible'):
   expected=desc([p['selected_'+k+'_signed_opposite_pixel_distance'] for p in pp]);actual=z['selected_'+k+'_signed_distance']
   assert expected['n']==actual['n']
   for key in expected:
    if key!='n':assert math.isclose(expected[key],actual[key],rel_tol=1e-12,abs_tol=1e-12)
ck('All role/native/weakowner groupcounts andboundary-distance summaries reproduce',True,len(groups))
invalid=[r for r in rows if r['center4_saved_valid_count']<4]
ck('Singleoriginal padded case retained,weakownerUNKNOWN',len(invalid)==1 and invalid[0]['global_index']==1043 and invalid[0]['weak_visible_owner_support']=='UNKNOWN')
repair=dict(schema='V159_OUTPUT_CSV_BINDING_GUARD_REPAIR',packaging_only=True,
 failure='Completed numerical mask pass stopped when final output binding guard rejected .csv; no numerical or label calculation failed.',
 repaired_action='Separate metadata-only finalization permits outputCSV hashing and audits alreadywritten390rows/88frame maskfacts. No mask/RGB/model re-read or numerical rerun.',
 executed_analysis_code=started['code'],failed_record=bind(O/'FAILED.json'),preserved_failed_code=bind(O/'provenance/BEFORE_OUTPUT_CSV_BINDING_REPAIR/analyze_exact390_mask_geometry.py'),
 finalization_code=bind(__file__),new_mask_or_RGB_reads_in_finalization=0)
write_path=O/'PACKAGING_REPAIR_RECORD.json'
write_path.write_text(json.dumps(repair,ensure_ascii=False,indent=2)+'\n');bind(write_path)
limits=['Selected full/visible masks are nonexclusive instance geometry,not exclusive owner/physicalnative visibility truth. All overlaps retained.',
 'Only obj_id1 palletinstance masks areprovided;KLT/tub/cargo/otherforeignobjects lack exhaustiveinstance masks.Outsidetarget/otherpallet masks doesnot implybackground.',
 'Nativepositions use provided800x1200x144bboxextrema andunchangeddynamicpermutation.All8nominalpoints coincidewithprovidedmeshvertices/surface;actualphysicalphoto-nativecorners remainuncertified undercm pose/dimension uncertainty.',
 'Nominalreferenceincrop ormasktouch gives coverage/context evidence only;it doesnotcertifyactualpointininput/nativeD/E/coordinatequality.No q/L orlabel/crop changes.',
 'This390row source isEuro-wood MR6D obj1;no realSQUARE material/native8 truth iscreated.NativeE supportremainsonlyslots4/5;all8native slots retained.',
 'All390 physical/native/pixelcertification staysUNKNOWN frommaskgeometry alone;weakowner/nativeD/E fields stayseparate.No automatictrainingadmission orfitselection.']
outputs=[bind(O/p) for p in ('ALL390_ORIGINAL_QUERY_NOMINAL_REFERENCE_MASK_GEOMETRY.jsonl','ALL390_MASK_GEOMETRY_SUMMARY.csv','ALL390_NATIVE_MATERIAL_WITNESS_GEOMETRY_INPUTS.jsonl','MASK_GEOMETRY_AGGREGATES.json','TRAIN88_MASK_TOTALS_AND_OVERLAPS.json')]
complete=dict(completed=True,PASS=True,source_only=True,rows=390,frames=88,original_roles={'D':341,'E':49},all390_retained=True,
 original_qL_native_split_roles_weakowner_and_selectedinstance_unchanged=True,executed_analysis_code=started['code'],finalization_code=bind(__file__),
 case_seal=bind(O/'CASE_AND_CALCULATION_SEAL_BEFORE_MASK_BODIES.json'),rules=seal['rules'],exact_mapping=mapping_meta['mapping'],
 source_metadata_bindings_verified_during_preparation=mapping_meta['source_bindings'],actual_mask_read_bindings_recorded=[v['original_file_binding'] for v in maskfacts['mask_files']],
 mask_files_read_during_analysis=len(actual_masks),new_mask_or_RGB_reads_in_finalization=0,RGB_or_depth_or_fullfeature_or_weight_bodies_read=0,
 held_scene1_mask_or_RGB_or_label_records_analyzed=0,counts=stats,nonexclusive_visible_overlap_frames=sum(r['nonexclusive_visible_overlap_present'] for r in maskfacts['frames']),
 native_coverage={str(i):dict(rows=stats['NATIVE_'+str(i)]['rows'],D=sum(r['native_slot']==i and r['original_role_state']=='D' for r in rows),E=sum(r['native_slot']==i and r['original_role_state']=='E' for r in rows)) for i in range(8)},
 material_coverage=dict(public_source='MR6D obj1 nominalEuro-woodmodel800x1200x144',real_square_material_truth_created=0),
 native_physical_material_certification='UNKNOWN_FOR_ALL390_FROM_MASK_GEOMETRY_ALONE',checks=CHECKS,check_count=len(CHECKS),semantic_limits=limits,
 outputs=outputs,metadata_output_bindings=list(B.values()),packaging_failure_repaired_without_reexecuting_masks=True,repair_record=bind(write_path),
 point_quality_targets_all_minus1=True,new_D_E_or_native_semantic_labels_created=0,
 concrete_preparation_action='All390 geometry-only native-material witnessinputs prepared with exactselectedinstance/native,nominalrefs andselected/othermaskbindings.No priorrole/weakownerobservationsinpreparationinputs.',
 new_model_or_fit_or_private_or_V112_or_PnP_qL_calls=0,training_admitted=False,selected_model=None,goal_status='active_unsolved')
path=O/'COMPLETE.json';path.write_text(json.dumps(complete,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(completed=True,complete=bind(path),checks=len(CHECKS),mask_files=len(actual_masks),all=stats['ALL'],by_role={k:stats[k] for k in ('ROLE_D','ROLE_E')}),indent=2))
