"""Bounded independent sourceGT-only unit/weight/coordinate replay. No model."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,datetime,math
import numpy as np
O=Path(__file__).resolve().parent;R=O.parent;P=R.parent;C=R/'teacher_preparation/corpus_build_v1'
def b(p):
 p=Path(p);v=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)}
def read(p):return json.loads(Path(p).read_text())
def jl(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def main():
 seal=read(C/'GT_CORPUS_SEAL.json');assert seal['PASS']and seal['completed']
 for x in seal['files']:assert b(x['path'])==x
 contract=read(R/'teacher_preparation/TEACHER_CONTRACT_BEFORE_BUILD.json')
 for x in contract['bindings'].values():assert b(x['path'])==x
 tiles=jl(contract['bindings']['grid12900']['path']);original=jl(contract['bindings']['cues302']['path']);views=jl(C/'ALL604_CUE_VIEW_TEACHERS_GT_ONLY.jsonl');assert len(tiles)==12900 and len(original)==302 and len(views)==604
 with np.load(C/'ALL77916_TILE_CUE_TRAINING_UNITS_GT_ONLY.npz',allow_pickle=False)as z:arr={k:z[k]for k in z.files}
 groups=contract['seed_duplicate_original_groups'];assert groups==[['V84_A_CUE_00','B01'],['V84_A_CUE_01','B02']]
 half={x for g in groups for x in g};byframe=defaultdict(list)
 for i,t in enumerate(tiles):byframe[t['frame_input_id']].append(i)
 changes=[];raw=Counter();effective=Counter();checks=Counter();mass=[];regions=0
 for vi,v in enumerate(views):
  cue=original[vi//2];view='before'if vi%2==0 else'after';assert v['cue_view_index']==vi and v['cue_index']==vi//2 and v['cue_id']==cue['cue_id']and v['view']==view
  ref=cue['reference_xy'];box=cue['uncertainty_bbox_xyxy_half_open'];assert ref==v['reference_xy_unchanged']and box==v['uncertainty_bbox_xyxy_half_open_unchanged']
  label=-1;secondary=False
  if ref is not None:
   if cue['corpus']=='V85_CANONICAL294':
    source=cue['source_record_unchanged'];secondary=source['proposal_rule_reason']=='SECONDARY_BEFORE_PART_IDENTITY_NOT_SUPPORTED_TRUE'
    if not secondary:
     state=source['proposed_aux_after_state']
     label=0 if view=='before'or state=='APPARENT_SAME_CUE_VISIBLE_AI'else 1 if state=='EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI'else -1
   elif view=='before':label=0
   else:
    vals=list(cue['after_state'].values())
    if len(vals)==2 and vals[0]==vals[1]:label=0 if vals[0]=='APPARENT_SAME_CUE_VISIBLE_AI'else 1 if vals[0]=='EXTERNAL_FOREGROUND_BLOCKER_COVERS_CUE_REGION_AI'else -1
  assert label==v['teacher_label'];raw[str(label)]+=1
  dup=.5 if cue['corpus'].startswith('V84')and cue['slot_or_proposal_id']in half else 1.
  assert dup==v['duplicate_weight'];effective[str(label)]+=dup
  if secondary:assert label==-1;changes.append((cue['cue_id'],view))
  inds=np.flatnonzero(arr['cue_view_index']==vi);assert len(inds)==129
  ti=np.array(byframe[v['frame_input_id']],np.int32);assert np.array_equal(arr['tile_index'][inds],ti)
  assert np.all(arr['teacher_label'][inds]==label)and np.all(arr['duplicate_weight'][inds]==dup)
  full=[];inside=box is not None and 0<=box[0]<box[2]<=640 and 0<=box[1]<box[3]<=480
  for k in ti:
   t=tiles[k];w=t['window_xyxy_half_open']
   area=0 if box is None else (box[2]-box[0])*(box[3]-box[1]);inter=0 if box is None else max(0,min(box[2],w[2])-max(box[0],w[0]))*max(0,min(box[3],w[3])-max(box[1],w[1]))
   full.append(box is not None and inter==area)
  full=np.array(full,bool);assert np.array_equal(arr['full'][inds],full)and np.all(arr['uncertainty_inside_image'][inds]==inside)
  admitted=full & inside & (label!=-1);assert np.array_equal(arr['loss_admitted'][inds],admitted)
  assert np.array_equal(arr['training_label'][inds],np.where(admitted,label,-1))
  sides=np.array([tiles[k]['roi_side']for k in ti]);gw=np.zeros(129,np.float64)
  for L in (128,256):
   m=(sides==L)&full
   if inside and m.any():gw[m]=.5/int(m.sum())
   assert np.array_equal(arr['geometry_tile_weight'][inds][m],gw[m])
  base=gw if label!=-1 else np.zeros(129,np.float64)
  assert np.array_equal(arr['geometry_tile_weight'][inds],gw)and np.array_equal(arr['base_loss_weight'][inds],base)and np.array_equal(arr['loss_weight'][inds],base*dup)
  base_sum=math.fsum(base);eff_sum=math.fsum(base*dup);assert abs(base_sum-(label!=-1))<1e-14 and abs(eff_sum-(dup if label!=-1 else 0))<1e-14
  for L in (128,256):assert abs(math.fsum(base[sides==L])-(.5 if label!=-1 else 0))<1e-14
  mass.append((vi,base_sum,eff_sum));checks['weight_and_admission_units']+=129
  for j,k in enumerate(ti):
   t=tiles[k];q=t['query_xy'];L=t['roi_side'];idx=inds[j]
   er=np.full((2,),np.nan,np.float32);ep=np.full((9,2),np.nan,np.float32)
   if ref is not None:
    er=np.array([2*(ref[d]-q[d])/L for d in range(2)],np.float32)
    pts=[[box[0]+(ix+.5)*(box[2]-box[0])/3,box[1]+(iy+.5)*(box[3]-box[1])/3]for iy in range(3)for ix in range(3)]
    ep=np.array([[2*(pt[d]-q[d])/L for d in range(2)]for pt in pts],np.float32)
   assert np.array_equal(arr['reference_normalized_xy'][idx],er,equal_nan=True)and np.array_equal(arr['uncertainty9_normalized_xy'][idx],ep,equal_nan=True)
   assert np.array_equal(arr['reference_normalized_xy_safe'][idx],np.nan_to_num(er))and np.array_equal(arr['uncertainty9_normalized_xy_safe'][idx],np.nan_to_num(ep));regions+=1
  if label==-1:assert not arr['base_loss_weight'][inds].any()and not arr['loss_weight'][inds].any()
 assert len(changes)==4 and len({q for q,v in changes})==2
 assert {q.rsplit(':',1)[-1]for q,v in changes}=={'BASE_RUNNER_LEFT','BASE_RUNNER_RIGHT'}
 assert raw=={'0':261,'1':37,'-1':306}and effective=={'0':257.,'1':37.,'-1':306.}
 assert regions==77916
 summary=read(C/'GT_CORPUS_SUMMARY_AND_SCHEMA.json');assert summary['raw_view_label_counts']==dict(raw)and summary['duplicate_weighted_view_label_counts']==dict(effective)
 out={'completed':True,'PASS':True,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GT_only':True,'code':b(__file__),'teacher_contract':b(R/'teacher_preparation/TEACHER_CONTRACT_BEFORE_BUILD.json'),'corpus_seal':b(C/'GT_CORPUS_SEAL.json'),'all604_original_coordinate_ID_view_label_replayed':True,'all77916_tilejoins_admission_base_geometry_duplicate_weights_replayed':True,'all77916_reference_and_unc9_F32_coords_and_safezeros_exact':True,'per_known_view_base_mass_one_each_scale_half':True,'seed8_linked6_original_groups_weight_exact':True,'secondary_unsupported_two_both_views_U':changes,'all_unknown_loss_zero':True,'raw_view_labels':dict(raw),'effective_duplicate_weighted_labels':dict(effective),'GTmodelinputs_read':False,'RGB_feature_model_weights_forward_fit_current_PnP':0,'limits':'Metadata/GT math only; does not certify visual annotation truth, targetidentity, mosaic/inference implementation or fit. Cellcollision counts were not independently recomputed in this bounded audit.'}
 with(O/'GT_UNITS_AND_WEIGHTS_REPLAY.json').open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'report':b(O/'GT_UNITS_AND_WEIGHTS_REPLAY.json'),'raw':dict(raw),'effective':dict(effective)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
