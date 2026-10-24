from pathlib import Path
import json,hashlib,numpy as np
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');B=P/'query_visibility_public_cue_audit_20261023_v83/reviewer_b';R=P/'query_visibility_cue_occlusion_20261023_v72/public_pallet_initializer'
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
visual=B/'VISUAL_OBSERVATIONS_BEFORE_INITIALIZER_METADATA.json';assert visual.exists()
cs=json.loads((B/'GEOMETRY_AND_RAWFILE_LOCK_BEFORE_VIEW.json').read_text())['cases']
rows=[json.loads(l) for l in (R/'FROZEN_PROPOSALS.jsonl').read_text().splitlines()];ins=[json.loads(l) for l in (R/'INPUTS_RGB_ONLY.jsonl').read_text().splitlines()]
assert bind(R/'FROZEN_PROPOSALS.jsonl')['sha256']=='0d940f95b86a8a183cea4ce26e9638e3ec56311c7eff04875f13fdc983aef605'
assert len(rows)==len(ins)==48
out=[]
for c in cs:
 match=[z for z in rows if z['image_sha256']==c['raw_rgb']['sha256']];im=[z for z in ins if z['image_sha256']==c['raw_rgb']['sha256']];assert len(match)==len(im)==1
 x=match[0];im=im[0];assert x['input_id']==im['id'] and im['image']==c['image'] and im['raw_hw']==c['raw_hw']
 scores=[z['score'] for z in x['candidates']];expected=int(np.argmax(scores)) if scores else None;assert expected==x['selected_index'];assert x['no_detection']==(not scores)
 z={'case_id':c['case_id'],'input_id':x['input_id'],'image_sha256':x['image_sha256'],'raw_hw':c['raw_hw'],'candidate_count':len(scores),'selected_index_frozen_first_argmax':expected,'no_detection':x['no_detection'],'reference_xy_GT_only':c['reference_xy'],'native8_correspondence':'UNKNOWN','no_GT_based_candidate_or_channel_reselection':True}
 if expected is None:z.update({'selected_raw9':None,'input_native_grid_L':None,'selected_query_local_squares_containing_reference_center':[],'selected_query_local_squares_fully_containing_prior_uncertainty_square':[],'nearest_selected_channel_GT_only_diagnostic':None,'training_crop_availability_for_this_cue':'NO_SELECTED_PROPOSAL_U'})
 else:
  a=x['candidates'][expected];raw9=np.asarray(a['keypoints_xy'],dtype=float);assert raw9.shape==(9,2)
  q=raw9[:8];good=np.isfinite(q).all(-1)&~(q==-1).all(-1);L=float(np.clip(.25*np.max(np.ptp(q[good],axis=0)),16,max(c['raw_hw']))) if sum(good)>=2 else 16.
  d=(np.array(c['reference_xy'])-q)/L;rr=c['prior_uncertainty_radius_px']/L;inside=good&((d>=-.5)&(d<.5)).all(-1);full=good&((d-rr>=-.5)&(d+rr<.5)).all(-1);intersects=good&((d+rr>=-.5)&(d-rr<.5)).all(-1)
  k=int(np.argmin(np.linalg.norm(q-c['reference_xy'],axis=1)))
  z.update({'selected_raw9':raw9.tolist(),'selected_box_xyxy':a['box_xyxy'],'input_native_grid_L':L,'reference_minus_q_over_L_GT_only':d.tolist(),'selected_query_local_squares_containing_reference_center':np.flatnonzero(inside).tolist(),'selected_query_local_squares_fully_containing_prior_uncertainty_square':np.flatnonzero(full).tolist(),'selected_query_local_squares_intersecting_prior_uncertainty_square':np.flatnonzero(intersects).tolist(),'nearest_selected_channel_GT_only_diagnostic':{'index':k,'distance_px':float(np.linalg.norm(q[k]-c['reference_xy'])),'establishes_native_correspondence':False},'training_crop_availability_for_this_cue':'NO_FULL_REFERENCE_CROP_U' if not full.any() else 'FULL_EXISTS_BUT_NATIVE_ASSOCIATION_UNKNOWN'})
 out.append(z)
r={'completed':True,'source_only':True,'visual_observations_frozen_first':bind(visual),'source_bindings':{n:bind(R/n) for n in ['INFERENCE_RESULT.json','PROTOCOL.json','RUN_ENTRYLOCK.json','INPUTS_RGB_ONLY.jsonl','FROZEN_PROPOSALS.jsonl']},'L_recipe':'V60 native_grid unchanged: finite q8 except explicit(-1,-1); clip(.25*max(ptp(qgood)),16,max(H,W)), fewer than2=>16. Local square relative coordinate [-.5,.5).','L_recipe_source':bind(P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'),'cases':out,'scope_limits':['Distances and local-square availability are GT-only diagnostics; nearest channel is not a cue/native8 annotation.','The selected candidate remains the saved first score argmax; no alternative candidate is chosen using a reference.','No proposal/reference coordinate was moved.','No P3/weights/model/current223 read or new inference.'],'family_split':{'coarse_group':'NVIDIA_R2B_STORAGE_GALILEO_FAMILY_UNRESOLVED','split':'UNASSIGNED_GROUP_HOLD','independent_validation_claim':False},'code':bind(Path(__file__))}
p=B/'PUBLIC_INITIALIZER_METADATA_DIAGNOSTIC.json';assert not p.exists();p.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');print(json.dumps(bind(p)))
