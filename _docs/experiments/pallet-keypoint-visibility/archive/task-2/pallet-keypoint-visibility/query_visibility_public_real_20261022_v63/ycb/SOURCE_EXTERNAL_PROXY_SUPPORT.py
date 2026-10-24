import pathlib,json,csv
from collections import defaultdict,Counter
import numpy as np
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parent.parent;OUT=ROOT/'ycb'
instances=[json.loads(x)for x in(OUT/'val_instances.jsonl').read_text().splitlines()];frames=defaultdict(list)
for r in instances:frames[(r['scene_id'],r['frame_id'])].append(r)
rows=list(csv.DictReader((OUT/'SOURCE_FAILURE_CONTEXT_SWAP_QUERIES.csv').open()));qr=defaultdict(list)
for r in rows:qr[(int(r['scene_id']),int(r['frame_id']))].append(r)
sam={r['query_id']:r for r in csv.DictReader((ROOT/'sam/SOURCE_QUERY_SCORED.csv').open())if r['source']=='YCB'and r['radius']=='0'};out=[]
for key,rr in qr.items():
 ir=frames[key];vm={r['instance_id']:np.asarray(Image.open(r['mask_vis']))>0 for r in ir};fm={r['instance_id']:np.asarray(Image.open(r['mask_full']))>0 for r in ir};depth=np.asarray(Image.open(ir[0]['depth']))
 for r in rr:
  if r['external_proxy']!='True':continue
  px,py=np.floor([float(r['query_x'])+.5,float(r['query_y'])+.5]).astype(int);k=int(r['instance_id']);othersvis=[i for i,m in vm.items()if i!=k and m[py,px]];othersfull=[i for i,m in fm.items()if i!=k and m[py,px]]
  support='other_annotated_visible_occluder'if othersvis else'other_annotated_full_only'if othersfull else'no_other_annotated_target_support'
  item=dict(r);item.update(external_support_evidence=support,other_visible_instance_ids=';'.join(map(str,othersvis)),other_full_instance_ids=';'.join(map(str,othersfull)),observed_depth_raw=int(depth[py,px]),missing_depth=not bool(depth[py,px]),SAM_on_subset=r['sample_id']in sam,SAM_false_presence=sam.get(r['sample_id'],{}).get('pred_visible_support')=='True');out.append(item)
counts=Counter(r['external_support_evidence']for r in out);paired={}
for state in counts:
 rr=[r for r in out if r['external_support_evidence']==state and r['SAM_on_subset']];paired[state]={'N':len(rr),'SAM_false_presence':sum(r['SAM_false_presence']for r in rr),'missing_depth':sum(r['missing_depth']for r in rr)}
report={'scope':'Public source YCB VAL external proxy only; no physical relabeling','external_queries':len(out),'other_instance_evidence_counts':dict(counts),'SAM_paired_subset_by_evidence':paired,'missing_depth_total':sum(r['missing_depth']for r in out),'interpretation':'A visible annotated different instance is direct evidence consistent with external occlusion. Lack of such support is not proof of label error: unmodeled occluders, pose/render/depth mismatches or invalid depth can also explain full1/vis0. Preserve raw proxy label and report this uncertainty separately; do not silently turn it into physical external truth.','fixed_earlier_SAM_external_case':[r for r in out if r['sample_id']=='YCB:000008:000001:000000:3'],'new_illustrative_SAM_external_case':[r for r in out if r['sample_id']=='YCB:000008:000967:000002:3']}
(OUT/'SOURCE_EXTERNAL_PROXY_SUPPORT.json').write_text(json.dumps(report,indent=2))
with(OUT/'SOURCE_EXTERNAL_PROXY_SUPPORT.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps({k:v for k,v in report.items()if 'case'not in k},indent=2))
