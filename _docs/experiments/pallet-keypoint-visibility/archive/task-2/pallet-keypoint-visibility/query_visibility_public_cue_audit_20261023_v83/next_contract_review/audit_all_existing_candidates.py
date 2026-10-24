from pathlib import Path
import json,hashlib,numpy as np,csv
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');D=P/'query_visibility_public_cue_audit_20261023_v83/next_contract_review'
def b(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
policy=D/'READER_POLICY_BEFORE_METADATA_TEST.json';assert policy.exists()
lock=P/'query_visibility_public_cue_audit_20261023_v83/reviewer_b/GEOMETRY_AND_RAWFILE_LOCK_BEFORE_VIEW.json'
props=P/'query_visibility_cue_occlusion_20261023_v72/public_pallet_initializer/FROZEN_PROPOSALS.jsonl'
assert b(props)['sha256']=='0d940f95b86a8a183cea4ce26e9638e3ec56311c7eff04875f13fdc983aef605'
cs=json.loads(lock.read_text())['cases'];src=[json.loads(l) for l in props.read_text().splitlines()];out=[];summary=[]
for c in cs:
 x=next(z for z in src if z['image_sha256']==c['raw_rgb']['sha256']);H,W=c['raw_hw'];r=np.array(c['reference_xy'],dtype=float);rad=c['prior_uncertainty_radius_px'];unc=np.r_[r-rad,r+rad]
 for a in x['candidates']:
  q8=np.asarray(a['keypoints_xy'],dtype=float)[:8];good=np.isfinite(q8).all(-1)&~(q8==-1).all(-1)
  L=float(np.clip(.25*np.max(np.ptp(q8[good],axis=0)),16,max(H,W))) if sum(good)>=2 else 16.
  for k,q in enumerate(q8):
   if not good[k]:status='INVALID';qvalid=False
   else:
    rounded=np.floor(q+.5);qvalid=bool(0<=q[0]<W and 0<=q[1]<H and 0<=rounded[0]<W and 0<=rounded[1]<H)
    crop=np.r_[q-L/2,q+L/2];observed=np.array([max(crop[0],-.5),max(crop[1],-.5),min(crop[2],W-.5),min(crop[3],H-.5)])
    overlap=np.maximum(0,np.minimum(unc[2:],observed[2:])-np.maximum(unc[:2],observed[:2]));full=bool(np.all(observed[2:]>observed[:2]) and np.all(observed[:2]<=unc[:2]) and np.all(observed[2:]>=unc[2:]))
    status='FULL' if full else 'PARTIAL' if np.prod(overlap)>0 else 'MISS'
   out.append({'case_id':c['case_id'],'input_id':x['input_id'],'candidate_index':a['candidate_index'],'is_saved_selected_candidate':a['candidate_index']==x['selected_index'],'native_channel_index':k,'native_correspondence_for_reference':'UNKNOWN','q_x':float(q[0]),'q_y':float(q[1]),'L':L,'query_center_valid':qvalid,'reference_availability_GT_only':status})
 local=[z for z in out if z['case_id']==c['case_id']]
 counts=lambda rows:{s:sum(z['reference_availability_GT_only']==s for z in rows) for s in ['FULL','PARTIAL','MISS','INVALID']}
 summary.append({'case_id':c['case_id'],'saved_candidate_count':len(x['candidates']),'saved_selected_index':x['selected_index'],'all_candidate_q8_rows':len(local),'all_candidate_availability':counts(local),'selected_candidate_availability':counts([z for z in local if z['is_saved_selected_candidate']]),'full_qvalid_rows':[{'candidate_index':z['candidate_index'],'channel':z['native_channel_index']} for z in local if z['reference_availability_GT_only']=='FULL' and z['query_center_valid']],'no_candidate_was_chosen_from_this_GT_only_coverage':True,'native_reference_mapping_known':False})
f=D/'ALL_EXISTING_CANDIDATE_AVAILABILITY.csv'
with f.open('w') as h:w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
r={'completed':True,'metadata_only':True,'policy':b(policy),'source_bindings':[b(lock),b(props)],'rows':len(out),'cases':summary,'limitations':['No alternative detection is selected or admitted for training. All88 returned q8 records are preserved in the diagnostic.','FULL is geometric input availability only. It does not establish intended-target association, native8 correspondence, human visibility gold, or general identifiability.','P15 has no returned candidate; do not silently fill it using the cue.','A GT-free new grid may be separately predeclared, but none is created/selected here.'],'code':b(Path(__file__)),'csv':b(f),'new_RGB_model_fit_composite_current_PnP_reads_or_calls':0}
p=D/'ALL_EXISTING_CANDIDATE_AVAILABILITY.json';assert not p.exists();p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False,indent=2))
