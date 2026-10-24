"""Separate nearest-pixel availability from full bilinear pixel-center domain,without q changes."""
from pathlib import Path
import hashlib,json,math
import numpy as np
from PIL import Image
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
assert not (O/'NEAREST_PIXEL_DOMAIN_CLARIFICATION.json').exists()
clarification=dict(schema='V159_NEAREST_VS_BILINEAR_DOMAIN_CLARIFICATION_BEFORE_EDGE_MASK_REREAD',
 original_analysis_domain='Existing query_mask_geometry continuous_in_image means fullbilinear pixel-center domain0<=x<=W-1,0<=y<=H-1;itdoesnotreplaceoriginalqueryavailability.',
 nearest_pixel_domain='Report original q insideimage extent0<=x<W,0<=y<H andwhetherfloor(x+.5),floor(y+.5) isavalidpixel. No clipping/coordinate change.',
 edge_case='OnlyQ1043(originalqy719.185,H720) isoutsidefullbilinearpixel-center domainyetitsnearestpixelindex719exists. ItsoriginalRGB96center4validcountremains2andweakownerUNKNOWN.',
 action='Appendnearest-pixel-only390rows andaggregates;reuse389storedsamples,readonlyexistingQ1043selected/otherinstance maskfilesforlast-rowpixel. Do notaltertheoriginalstrict-domaincalculations,crops,labels orq/L.')
(O/'NEAREST_PIXEL_DOMAIN_RULES_BEFORE_EDGE_REREAD.json').write_text(json.dumps(clarification,ensure_ascii=False,indent=2)+'\n')
old_complete=read(O/'COMPLETE.json');rows=lines(O/'ALL390_ORIGINAL_QUERY_NOMINAL_REFERENCE_MASK_GEOMETRY.jsonl');mapping={r['global_index']:r for r in lines(O/'ALL390_EXACT_SELECTED_NATIVE_REFERENCE_MAPPING.jsonl')}
out=[];rereads=[];edge=[]
for r in rows:
 p=r['query_mask_geometry'];m=mapping[r['global_index']];h,w=m['selected_instance_geometry']['raw_hw'];x,y=m['query_xy'];ix,iy=math.floor(x+.5),math.floor(y+.5)
 available=0<=x<w and 0<=y<h and 0<=ix<w and 0<=iy<h
 q=dict(p)
 if available and p['nearest_pixel'] is None:
  masks={}
  for inst in [m['selected_instance_mask_bindings']]+m['other_instance_mask_bindings']:
   k=inst['gt_instance_index'];masks[k]={}
   for kind in ('full','visible'):
    b=inst[kind+'_mask'];path=Path(b['path']);s=str(path).lower()
    assert str(path).startswith(str(P/'query_visibility_mr6d_full_public_source_20261024_v153/full_source_v1/original')+'/')
    assert '/mask/' in s or '/mask_visib/' in s
    assert not any(t in s for t in ('v112','current223','private','/val/000001/'))
    assert sha(path)==b['sha256'];a=np.asarray(Image.open(path));assert list(a.shape)==[h,w]
    masks[k][kind]=a!=0;rereads.append(b)
  selected=m['selected_gt_instance_index'];full=bool(masks[selected]['full'][iy,ix]);vis=bool(masks[selected]['visible'][iy,ix])
  of=sum(bool(a['full'][iy,ix]) for k,a in masks.items() if k!=selected);ov=sum(bool(a['visible'][iy,ix]) for k,a in masks.items() if k!=selected)
  q.update(nearest_pixel=[ix,iy],selected_full=full,selected_visible=vis,other_full_count=of,other_visible_count=ov,
   selected_and_other_full_overlap=full and of>0,selected_and_other_visible_overlap=vis and ov>0)
  edge.append(dict(global_index=r['global_index'],query_id=r['query_id'],original_q=m['query_xy'],nearest_pixel=[ix,iy],selected_full=full,selected_visible=vis,
    original_center4_saved_valid_count=r['center4_saved_valid_count'],weakowner=r['weak_visible_owner_support']))
 assert available==(q['nearest_pixel'] is not None)
 out.append(dict(global_index=r['global_index'],query_id=r['query_id'],frame_index=r['frame_index'],native_slot=r['native_slot'],split=r['split'],
  original_role_state=r['original_role_state'],query_xy=r['query_xy'],L=r['L'],original_input_available=True,
  original_center4_saved_valid_count=r['center4_saved_valid_count'],nearest_pixel_available=available,
  full_bilinear_pixel_center_domain=p['continuous_in_image'],q_nearest_pixel=q['nearest_pixel'],selected_full=q['selected_full'],selected_visible=q['selected_visible'],
  other_full_count=q['other_full_count'],other_visible_count=q['other_visible_count'],selected_and_other_full_overlap=q['selected_and_other_full_overlap'],selected_and_other_visible_overlap=q['selected_and_other_visible_overlap'],
  source_masks_nonexclusive=True,physical_native_gold=False,coordinate_quality_gold=False,training_admitted=False))
assert len(out)==390 and len(edge)==1 and edge[0]['global_index']==1043
path=O/'ALL390_NEAREST_PIXEL_ONLY_MASK_SUPPORT.jsonl'
with path.open('x') as f:
 for r in out:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
aggregates={}
for name,rr in [('ALL',out)]+[('ROLE_'+s,[r for r in out if r['original_role_state']==s]) for s in ('D','E')]:
 aggregates[name]=dict(rows=len(rr),original_available=sum(r['original_input_available'] for r in rr),nearest_pixel_available=sum(r['nearest_pixel_available'] for r in rr),
  full_bilinear_pixel_center_domain=sum(r['full_bilinear_pixel_center_domain'] for r in rr),selected_full=sum(r['selected_full'] is True for r in rr),selected_visible=sum(r['selected_visible'] is True for r in rr),
  other_full_any=sum((r['other_full_count'] or 0)>0 for r in rr),other_visible_any=sum((r['other_visible_count'] or 0)>0 for r in rr),
  selected_and_other_full_overlap=sum(r['selected_and_other_full_overlap'] is True for r in rr),selected_and_other_visible_overlap=sum(r['selected_and_other_visible_overlap'] is True for r in rr))
record=dict(completed=True,code=bind(__file__),rule=bind(O/'NEAREST_PIXEL_DOMAIN_RULES_BEFORE_EDGE_REREAD.json'),all390_rows=bind(path),edge_case=edge,
 original_masks_reread=rereads,unique_mask_file_inputs_unchanged=True,noRGB_model_labels_qL_or_split_changes=True,nearest_pixel_counts=aggregates)
rp=O/'NEAREST_PIXEL_DOMAIN_CLARIFICATION.json';rp.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
provenance=O/'provenance/BEFORE_NEAREST_PIXEL_DOMAIN_CLARIFICATION';provenance.mkdir(exist_ok=True)
oldpath=provenance/'COMPLETE.json';oldpath.write_bytes((O/'COMPLETE.json').read_bytes())
new=old_complete;new['strict_domain_analysis_complete_preserved']=bind(oldpath);new['nearest_pixel_domain_clarification']=bind(rp)
new['nearest_pixel_only_counts']=aggregates;new['nearest_pixel_sidecar']=bind(path)
new['sampling_domain_limit']='Originalall390availabilityretained.Strictmainpointstatsuse0..W-1/H-1fullbilinearpixel-center domain;nearest-pixel-onlysidecar separatelyreportsfloor(q+.5)maskmembershipandincludesQ1043last-rowpixel. No physical/native/pointgoldinference.'
new['mask_file_rereads_for_boundary_clarification']=len(rereads);new['unique_mask_files_read_total']=old_complete['mask_files_read_during_analysis']
(O/'COMPLETE.json').write_text(json.dumps(new,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(completed=True,complete=bind(O/'COMPLETE.json'),nearest_pixel_counts=aggregates,edge=edge),indent=2))
