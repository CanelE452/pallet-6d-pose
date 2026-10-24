"""Saved outputs only. Different targets; no ranking winners or new inference."""
from pathlib import Path
import sys,json,csv,gzip,time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=Path(__file__).resolve().parent;V60=R.parent/'external_ray_proxy_head_20261022_v60';V61=R.parent/'target_presence_classifier_20261022_v61';V50=R.parent/'ocpr_wood_square_transfer_20261022_v50';B=Path('/home/minjae/Documents/pallet-research');SRC=Path('/tmp/pallet-bi-full-execution-20261019-v1');sys.path.insert(0,str(V60));from common import read,rows,sha,utc
def save(n,x):
 with(R/n).open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvsave(n,rr):
 with(R/n).open('x')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
start=time.monotonic();u=read(V60/'PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz');ray=np.load(V60/'REAL_HEAD_OUTPUTS.npz');pre=np.load(V61/'REAL_PRESENCE_OUTPUTS.npz');assert read(V61/'SOURCE_BINDINGS.json')[1]['sha256']==sha(V60/'REAL_ROI_CACHE.npz');assert ray['probability'].shape==pre['probability'].shape==(223,8);tracepath=B/'pallet-statewise-correspondence-diagnosis-20261016-v1/e_trace_v2/E_TRACE.jsonl.gz';tr={x['id']:x for x in rows(tracepath)};allrows=[];sources=[]
for i,x in enumerate(u):
 if x['group']=='WOOD_PRIMARY105':
  old=tr[x['id']];records=old['state_records'];oldq=old['initial_points'];image_match=old['image_sha256']==x['image_sha256'];source=tracepath
 else:
  source=V50/'EXECUTION/SQUARE_YOLO'/x['id'].split(':')[-1]/'OBSERVATION.json.gz';old=read(source);image_match=old['image_sha256']==x['image_sha256'];a=old['_original_source'];records=a['state_observation']['records'];oldq=a['original_raw_points']
 assert image_match;sources.append(dict(id=x['id'],path=str(source),sha256=sha(source)))
 for k,q in enumerate(x['initial_points'][:8]):
  r=records[k];assert int(r['native_id'])==k;rgb=bool(r.get('rgb_evaluated'));same=bool(np.array_equal(oldq[k],q));pold=r.get('p_visible');oldvalid=rgb and same and pold is not None;H,W=x['raw_hw'];query_valid=bool(0<=q[0]<W and 0<=q[1]<H and ray['pixel_valid'][i,k]and pre['pixel_valid'][i,k]);allrows.append(dict(id=x['id'],group=x['group'],native_corner=k,query_x=q[0],query_y=q[1],query_valid=query_valid,ray_blockage_probability=float(ray['probability'][i,k]),ray_blocked_at_fixed_0_5=bool(ray['probability'][i,k]>=.5),presence_probability=float(pre['probability'][i,k]),present_at_fixed_0_5=bool(pre['probability'][i,k]>=.5),legacy_record_exists=True,legacy_RGB_evaluated=rgb,legacy_query_coordinate_equal=same,legacy_direct_score_comparable_location=oldvalid,legacy_p_DIRECT=pold if oldvalid else None,legacy_stored_state=r['state'],legacy_reason=r.get('reason'),real_presence_reference='UNAVAILABLE'))
csvsave('SAME_QUERY_SAVED_OUTPUTS.csv',allrows);save('LEGACY_RECORD_BINDINGS.json',sources);summary=[]
for group in ['WOOD_PRIMARY105','SQUARE118']:
 a=[r for r in allrows if r['group']==group];valid=[r for r in a if r['query_valid']]
 for raypred,prespred in [(False,False),(False,True),(True,False),(True,True)]:summary.append(dict(group=group,all_query_outputs=len(a),valid_queries=len(valid),ray_blocked=raypred,target_present=prespred,n=sum(r['ray_blocked_at_fixed_0_5']==raypred and r['present_at_fixed_0_5']==prespred for r in valid),legacy_RGB_same_coordinate_scores=sum(r['legacy_direct_score_comparable_location']for r in valid),meaning='co-occurring different-target predictions, not an accuracy confusion matrix'))
csvsave('COOCCURRENCE_NOT_ACCURACY.csv',summary)
chosen=[]
for group in ['WOOD_PRIMARY105','SQUARE118']:
 for pair in [(False,False),(True,True)]:
  a=[r for r in allrows if r['group']==group and r['query_valid']and r['ray_blocked_at_fixed_0_5']==pair[0]and r['present_at_fixed_0_5']==pair[1]]
  if a:chosen.append(min(a,key=lambda r:(r['id'],r['native_corner'])))
save('SELECTION.json',dict(rule='posthoc raw ray/presence joint outputs (0,0) and (1,1), lexical ID/native index per group. No new labels, no model choice.',cases=chosen));by={x['id']:x for x in u};fig,axs=plt.subplots(len(chosen),4,figsize=(12,3.0*len(chosen)),layout='constrained');manifest=[]
for row,r in enumerate(chosen):
 x=by[r['id']];assert sha(x['image'])==x['image_sha256'];rgb=np.array(Image.open(x['image']).convert('RGB'));q=np.array([r['query_x'],r['query_y']]);H,W=x['raw_hw'];L=72 if W<=960 else 128;lo=q-L/2;hi=q+L/2
 for col in range(4):
  ax=axs[row,col];ax.set_axis_off()
  if col==0:ax.imshow(rgb);ax.add_patch(Rectangle(lo,L,L,fill=False,color='#00bbbb',lw=1));ax.set_title(r['id']+f' / k{r["native_corner"]}',fontsize=8)
  else:
   ax.add_patch(Rectangle(lo,L,L,facecolor='#888888',edgecolor='none',zorder=-1));ax.imshow(rgb,interpolation='nearest');ax.plot(*q,'+',c='#00d5ed',ms=11,mew=1.5);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
   if col==1:title=f'Legacy human DIRECT/EXTERNAL\np_DIRECT={r["legacy_p_DIRECT"]:.3f}; stored {r["legacy_stored_state"]}'if r['legacy_direct_score_comparable_location']else'Legacy: no RGB score\nstored '+r['legacy_stored_state']
   elif col==2:title=f'v60 external virtual-ray blockage\np_blocked={r["ray_blockage_probability"]:.3f}'
   else:title=f'v61 visible target-pixel presence\np_present={r["presence_probability"]:.3f}'
   ax.set_title(title,fontsize=8)
 manifest.append(dict(id=x['id'],native_corner=r['native_corner'],image=x['image'],image_sha256=x['image_sha256'],query_xy=q.tolist(),crop_xyxy=[*lo.tolist(),*hi.tolist()],stored_predictions=r,real_target_pixel_reference='UNAVAILABLE'))
fig.suptitle('Saved real query predictions: three different targets',fontsize=13);fig.supxlabel('Identical raw RGB/crop/query per row. Cyan + is the initial predicted query, not a reference annotation.\nDo not interpret ray-not-blocked as target-present or compare these probabilities as accuracy.\nLegacy uses real-CAL human labels plus initial geometric gates; v60/v61 heads use different synthetic targets.\nNo real pixel-mask reference, new model inference, retraining, threshold selection or pose evaluation.',fontsize=8)
for ext in ['png','pdf','svg']:fig.savefig(R/('Saved_Real_Three_Targets.'+ext),dpi=160)
plt.close(fig);save('FIGURE_MANIFEST.json',manifest)
desc=SRC/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/state_visibility/bounded_ownership_spatial_v3/MODEL.json';model=read(desc);assert sha(desc)==read(V50/'MODEL_BINDINGS.json')['observations']['state_descriptor']['sha256'];save('EXACT_LEGACY_DESCRIPTOR.json',dict(path=str(desc),sha256=sha(desc),descriptor=model));save('AUDIT_COMPLETE.json',dict(UTC=utc(),frames=len(u),queries=len(allrows),same_frozen_ROI_cache=True,query_coordinates_same_v60_v61=True,legacy_records_frames=len(sources),cases=len(chosen),seconds=time.monotonic()-start,new_fit=0,new_model_forward=0,new_pose=0,new_reference_annotations=0,no_model_superiority_claim=True));print('READ_ONLY_AUDIT_COMPLETE',len(allrows),summary,flush=True)
