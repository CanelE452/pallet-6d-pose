"""Extract the already-classified thirteen adopted queries; no new metrics/fits.

The output repeats saved literals and source bindings. Physical ownership and
independent pose truth remain UNKNOWN. Original evidence is never overwritten.
"""
import argparse,datetime,gzip,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
METHODS=('HYBRID_STANDARD_PNP','HYBRID_POINT_CONSENSUS_RESTORED')
def bind(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024*1024):h.update(b)
 return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);args=a.parse_args()
output=args.output.absolute()
if output.exists() or output.is_symlink() or any(p.is_symlink()for p in output.parents):raise RuntimeError('output must be a new regular path')
source=DOC/'CORRESPONDENCE_AUDIT_ROWS.jsonl.gz';readout=DOC/'CORRESPONDENCE_AUDIT_READOUT.json'
source_binding=bind(source)
if source_binding['sha256']!='af3ffd627bee403659c0e85b6ffaaf34d01f0085d9a4e932ede5f30eb5f34bdc':raise RuntimeError('frozen audit bytes changed')
old=json.loads(readout.read_text())['bad_adopted_boundary']
if any(old[m+':bad_adopted']!=13 or old[m+':bad_adopted_final_inlier']!=13 for m in METHODS):raise RuntimeError('saved thirteen-query readout changed')
cases=[]
with gzip.open(source,'rt')as f:
 for line in f:
  row=json.loads(line);methods={m['method']:m for m in row['methods']}
  if set(methods)!=set(METHODS):raise RuntimeError('saved method pair changed')
  selected={m:[c for c in methods[m]['boundary_candidates']if c['accepted'] and c['candidate_proxy_quality']=='INCORRECT_OVER8PX']for m in METHODS}
  if [c['id']for c in selected[METHODS[0]]]!=[c['id']for c in selected[METHODS[1]]]:raise RuntimeError('saved q pair differs')
  for c in selected[METHODS[0]]:
   k=c['id'];details={}
   for m in METHODS:
    record=methods[m];query=next(q for q in selected[m]if q['id']==k);corner=next(q for q in record['corners']if q['id']==k)
    if not query['accepted_new_pose_final_inlier']or k not in record['pools']['accepted_final_inliers']['ids']:raise RuntimeError('saved inlier membership differs')
    details[m]={'saved_candidate':query,'saved_corner':corner,'saved_pools':record['pools'],'saved_self_residual_px':record['solver_self_residual_px_by_used_id'][str(k)],'H':record['H'],'reprojected_ids':record['reprojected_ids'],'output_status':record['output_status'],'saved_pose_outcome_vs_N3':record['pose_outcome_vs_N3'],'saved_delta_vs_N3':record['delta_vs_N3'],'physical_boundary_ownership':record['physical_boundary_ownership']}
   cases.append({'frame_id':row['id'],'session':row['session'],'stratum':row['stratum'],'corner_id':k,'q_xy_saved':row['input_points'][k],'N3_xy_saved':row['original_native_N3_points'][k],'proxy_reference_xy_saved':row['saved_evaluation_reference']['native_points_px'][k],'human_state_saved':details[METHODS[0]]['saved_corner']['human_state'],'methods':details,'saved_paired_pose_outcome':row['paired_pose_outcome'],'independent_physical_ownership_certification':'MISSING'})
if len(cases)!=13 or len({(c['frame_id'],c['corner_id'])for c in cases})!=13:raise RuntimeError('literal extraction must match all saved thirteen queries')
result={'schema':'remaining_thirteen_saved_query_case_index_v18','complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_source':source_binding,'saved_readout':bind(readout),'index_code':bind(pathlib.Path(__file__)),'query_count':13,'cases':cases,'selection':'existing accepted=True and saved candidate_proxy_quality=INCORRECT_OVER8PX, unchanged order; no new cutoff or pose calculation','limits':{'new_metric_geometry_projection_PnP_GT_model_training_RGB_timing_calls':0,'saved_literals_and_memberships_only':True,'proxy_error_not_physical_ownership_proof':True,'physical_truth_and_edge_ownership_certified':False,'accuracy_goal_solved':False}}
output.parent.mkdir(parents=True,exist_ok=True)
with output.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'complete':True,'queries':13,'output':str(output),'bytes':output.stat().st_size,'new_numeric_fit_model_calls':0},ensure_ascii=False))
