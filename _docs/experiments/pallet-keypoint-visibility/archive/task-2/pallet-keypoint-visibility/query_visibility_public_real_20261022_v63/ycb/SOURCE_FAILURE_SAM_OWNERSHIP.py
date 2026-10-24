import csv,json,pathlib,numpy as np
from collections import defaultdict
ROOT=pathlib.Path(__file__).resolve().parent.parent;OUT=ROOT/'ycb'
report=json.loads((OUT/'SOURCE_FAILURE_CONTEXT_SWAP.json').read_text());rows=list(csv.DictReader((OUT/'SOURCE_FAILURE_CONTEXT_SWAP_QUERIES.csv').open()));byqid={r['sample_id']:r for r in rows};pairs=[r for r in csv.DictReader((ROOT/'sam/SOURCE_QUERY_SCORED.csv').open())if r['source']=='YCB'and int(r['radius'])==0]
pred={r['id']:r for r in [json.loads(x)for x in (ROOT/'sam/TARGET_PREDICTIONS.jsonl').read_text().splitlines()]if r['source']=='YCB'};masks={}
for fid,r in pred.items():
 p=pathlib.Path(r['mask_path']).resolve();assert p.is_relative_to((ROOT/'sam/masks').resolve())
 with np.load(p)as z:masks[fid]=z['mask'].astype(bool)
counter=[];bytruth=[];headmetrics={run:[]for run in ['MIXED_CONTEXT_V2','MIXED_CONTEXT_RGB_V3']}
for sr in pairs:
 r=byqid[sr['query_id']];px,py=int(sr['pixel_x']),int(sr['pixel_y']);own=bool(masks[r['id']][py,px]);other=bool(masks[r['other_target_id']][py,px]);a=int(r['presence_target']);b=int(r['other_target_y']);counter.append({'sample_id':r['sample_id'],'y_own':a,'y_other':b,'SAM_own':int(own),'SAM_other_box':int(other),'correct_direction':int(own)-int(other)if a>b else int(other)-int(own),'truth_changed':a!=b})
 if a!=b:
  bytruth.append((own==a and other==b,int(own)!=int(other),(int(own)-int(other))*(a-b)>0))
  for run in headmetrics:
   po=float(r[run+'_p']);ps=float(r[run+'_swapped_p']);headmetrics[run].append((po>=.5)==bool(a)and(ps>=.5)==bool(b))
report['paired_SAM_subset']['target_box_swap_diagnostic']={'N':len(pairs),'changed_truth_N':len(bytruth),'definition':'Stored whole-image SAM mask prompted by target A versus target B box, sampled at unchanged original query. No SAM inference or mask selection rerun.','original_correct_AND_other_target_correct_fraction_on_changed_truth':float(np.mean([x[0]for x in bytruth])),'prediction_changes_fraction_on_changed_truth':float(np.mean([x[1]for x in bytruth])),'correct_direction_fraction_on_changed_truth':float(np.mean([x[2]for x in bytruth])),'head_original_correct_AND_swapped_correct_same_rows':{k:float(np.mean(v))for k,v in headmetrics.items()}}
# Add same-denominator state errors for direct baseline comparisons.
report['paired_SAM_subset']['state_false_positive_rates']={}
for state in ['external_proxy','outside_support']:
 rr=[byqid[r['query_id']]for r in pairs if (r['source_support_state']=='EXTERNAL_OCCLUDED_SUPPORT'if state=='external_proxy'else r['source_support_state']=='OUTSIDE_TARGET_SUPPORT')]
 samrr=[r for r in pairs if r['query_id']in {x['sample_id']for x in rr}]
 report['paired_SAM_subset']['state_false_positive_rates'][state]={'N':len(rr),'SAM':float(np.mean([x['pred_visible_support']=='True'for x in samrr])),**{run:float(np.mean([float(x[run+'_p'])>=.5 for x in rr]))for run in ['MIXED_CONTEXT_V2','MIXED_CONTEXT_RGB_V3']}}
(OUT/'SOURCE_FAILURE_CONTEXT_SWAP.json').write_text(json.dumps(report,indent=2))
with (OUT/'SOURCE_FAILURE_SAM_BOX_SWAP.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(counter[0]));w.writeheader();w.writerows(counter)
print(json.dumps(report['paired_SAM_subset'],indent=2))
