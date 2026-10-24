"""Post-freeze independent arithmetic check and honest diagnostic figures."""
from common import *
import sys,time,math,statistics
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
start=time.monotonic();assert sha(ROOT/'BEST_HEAD.pt')==read(ROOT/'MODEL_FREEZE.json')['checkpoint_sha256'];allgaps=[];independent=[];maskstats=[]
def qlinear(a,q):
 a=sorted(a);h=(len(a)-1)*q;i=math.floor(h);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(h-i)
for scope in ['SYNTH','REAL']:
 inputs=read(ROOT/('PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz'if scope=='REAL'else'PREPARED_INPUTS_SYNTH.json.gz'));pack=read(ROOT/f'{scope}_POSE_PREDICTIONS_SEALED.json.gz');byinput={u['id']:u for u in inputs};m={ (r['id'],r['arm']):r for r in csv.DictReader(open(ROOT/f'{scope}_POSE_PER_FRAME.csv'))};pairs=list(csv.DictReader(open(ROOT/f'{scope}_POSE_PAIRED.csv')));summary=list(csv.DictReader(open(ROOT/f'{scope}_POSE_SUMMARY.csv')))
 if scope=='SYNTH':refs={g['id']:dict(R=g['R_fixed'],t=g['t'],order=g['symmetry_order'])for g in rows(B/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/SYNTH_GT_LABELS.jsonl.gz')}
 else:
  w=read(B/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json')['references'];s={g['id']:g['reference']for g in read(ROOT.parent/'ocpr_wood_square_transfer_20261022_v50/EVALUATION_FIXED/PORTABLE_POSE_AUDIT_BUNDLE.json.gz')['frames']if g['group']=='SQUARE_YOLO'};refs={}
  for u in inputs:
   g=w[u['partition']][u['id']]if u['group'].startswith('WOOD')else s[u['id'].split(':')[-1]];refs[u['id']]=dict(R=g['R'],t=g['t'],order=u['symmetry_order'])
 verified=[]
 for p in pack:
  u=byinput[p['id']];assert hashlib.sha256(np.array(u['initial_points'],np.float64).tobytes()).hexdigest()==p['coordinates_sha256'];assert p['original_pose']==u['initial_pose'];assert sum(p['valid9'][:8])==8-p['excluded'];assert not p['fallback']or p['final_pose']==u['initial_pose']
  g=refs[p['id']];G=Rotation.from_matrix(g['R']);angles=[G*Rotation.from_rotvec([0,a,0])for a in np.arange(g['order'])*2*np.pi/g['order']]
  for arm,key in [('N0','original_pose'),('EXTERNAL_RAY_HEAD','final_pose')]:
   pred=p[key];r=m[(p['id'],arm)];T=math.sqrt(math.fsum((x-y)**2 for x,y in zip(pred['centroid'],g['t'])))*100;P=Rotation.from_matrix(pred['R_physical']);Rs=min((a.inv()*P).magnitude()*180/math.pi for a in angles);C=(G.inv()*P).magnitude()*180/math.pi;gap=max(abs(T-float(r['T_cm'])),abs(Rs-float(r['R_deg'])),abs(C-float(r['R_C1_deg'])));allgaps.append(gap);success=T<5 and Rs<5;assert success==(r['success']=='True');verified.append(dict(id=p['id'],group=p['group'],arm=arm,T_cm=T,R_deg=Rs,success=success))
 for s in summary:
  v=[r for r in verified if r['group']==s['group']and r['arm']==s['arm']];assert len(v)==int(s['n'])and sum(r['success']for r in v)==int(s['success'])
  for col in ['T_cm','R_deg']:
   a=[r[col]for r in v]
   for name,f in [('mean',lambda z:math.fsum(z)/len(z)),('median',statistics.median),('P90',lambda z:qlinear(z,.9)),('P95',lambda z:qlinear(z,.95))]:allgaps.append(abs(f(a)-float(s[col+'_'+name])))
 for group in sorted({p['group']for p in pack}):
  pp=[p for p in pack if p['group']==group];j=[r for r in pairs if r['group']==group];assert sum(r['HEAD_success']=='True'for r in j)-sum(r['N0_success']=='True'for r in j)==sum(r['recovery']=='True'for r in j)-sum(r['damage']=='True'for r in j)
  for excluded in sorted({p['excluded']for p in pp}):
   a=[p for p in pp if p['excluded']==excluded];ids={p['id']for p in a};jj=[r for r in j if r['id']in ids];maskstats.append(dict(scope=scope,group=group,excluded=excluded,n=len(a),used_native_points=8-excluded,fallback=sum(p['fallback']for p in a),success=sum(r['HEAD_success']=='True'for r in jj),recovery=sum(r['recovery']=='True'for r in jj),damage=sum(r['damage']=='True'for r in jj),WD_family_changed=sum(p['final_pose']['selected_hypothesis']!=p['original_pose']['selected_hypothesis']for p in a)))
 independent.append(dict(scope=scope,frames=len(pack),metric_rows=len(verified),PASS=True,prediction_seal_unchanged=sha(ROOT/f'{scope}_POSE_PREDICTIONS_SEALED.json.gz')==read(ROOT/f'{scope}_POSE_SEAL.json')['predictions_sha256']))
csvsave('MASK_COUNT_STRATA.csv',maskstats);save('INDEPENDENT_CHECK.json',dict(UTC=utc(),checks=independent,max_pose_metric_difference=max(allgaps),first_numeric_1e_minus7_check_failed_preserved=True,note='No scoring/gate tolerance changed; report actual independent numerical differences, including ill-conditioned trace acos near180deg. All success/recovery/damage counts identical.',method='SciPy SO3 magnitude and independent Python fsum/statistics/linear quantile vs original NumPy trace/aggregation',coordinate_arrays_unchanged=True,exact_cached_fallback_preserved=True,head_unchanged=True,seconds=time.monotonic()-start))
# Cases selected by fixed outcome categories and lexical recording/ID; selection is posthoc illustration.
inputs=read(ROOT/'PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz');uu={u['id']:u for u in inputs};pred={p['id']:p for p in read(ROOT/'REAL_POSE_PREDICTIONS_SEALED.json.gz')};paired=list(csv.DictReader(open(ROOT/'REAL_POSE_PAIRED.csv')));metrics={(r['id'],r['arm']):r for r in csv.DictReader(open(ROOT/'REAL_POSE_PER_FRAME.csv'))};selected=[]
for group in ['WOOD_PRIMARY105','SQUARE118']:
 for category in ['recovery','damage','both_success']:
  pool=[r for r in paired if r['group']==group and (r[category]=='True'if category!='both_success'else r['N0_success']=='True'and r['HEAD_success']=='True')];r=min(pool,key=lambda r:(uu[r['id']]['recording'],r['id']));selected.append(dict(id=r['id'],group=group,purpose=category,selection='posthoc categorical illustration; lexical recording/ID; no model selection'))
save('FIGURE_SELECTION.json',selected);fig,axs=plt.subplots(6,3,figsize=(10.4,15.2),layout='constrained');edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];manifest=[]
for row,c in enumerate(selected):
 u=uu[c['id']];p=pred[c['id']];assert sha(u['image'])==u['image_sha256'];im=np.array(Image.open(u['image']).convert('RGB'));K=np.array(u['K']);initial=u['initial_pose'];Q=np.array(initial['R_cf']).T@np.array(initial['R_physical']);X=np.array(u['native_xyz'])@Q;g=refs[u['id']]
 def project(R,t):
  a=(X@np.array(R).T+np.array(t))@K.T;return a[:,:2]/a[:,2:]
 ref=project(g['R'],g['t']);q=np.array(u['initial_points'])[:8];lo=q.min(0);hi=q.max(0);margin=.15*max(hi-lo);lo=np.maximum([0,0],lo-margin);hi=np.minimum([im.shape[1]-1,im.shape[0]-1],hi+margin);ax=axs[row,0];ax.imshow(im);ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,color='#00b8b8',lw=1.3));ax.set_title(c['group'].split('_')[0]+' | '+c['purpose']+'\n'+u['id'],fontsize=8);ax.set_axis_off();panels=[]
 for col,arm,key in [(1,'N0','original_pose'),(2,'EXTERNAL_RAY_HEAD','final_pose')]:
  ax=axs[row,col];ax.imshow(im);pose=p[key];z=project(pose['R_physical'],pose['centroid'])
  for a,b in edges:ax.plot(ref[[a,b],0],ref[[a,b],1],color='white',ls='--',lw=1.1);ax.plot(z[[a,b],0],z[[a,b],1],color='#e36a00',ls='-',lw=1.1)
  used=np.ones(8,bool)if col==1 else np.array(p['valid9'][:8]);ax.scatter(q[used,0],q[used,1],s=14,marker='o',facecolor='none',edgecolor='#00d3e0',linewidth=.9);ax.scatter(q[~used,0],q[~used,1],s=25,marker='x',c='#ed3c8a',linewidth=1.1)
  if col==2:
   for k in np.flatnonzero(~used):ax.text(q[k,0]+3,q[k,1],f'{k}: {p["probability"][k]:.2f}',fontsize=5.5,color='white',bbox=dict(facecolor='black',alpha=.55,pad=.3))
  ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_xticks([]);ax.set_yticks([]);r=metrics[(u['id'],arm)];ax.set_title('N0'if col==1 else'Frozen ray-proxy head',fontsize=9);ax.set_xlabel(f'T {float(r["T_cm"]):.2f} cm | R C{u["symmetry_order"]} {float(r["R_deg"]):.2f} deg | '+('Pass'if r['success']=='True'else'Fail')+(f'\nUsed {sum(used)}/8; fallback {p["fallback"]}'if col==2 else''),fontsize=7);panels.append(dict(arm=arm,pose=pose,metrics=r))
 manifest.append(dict(**c,image_sha256=u['image_sha256'],same_crop_xyxy=[*lo.tolist(),*hi.tolist()],K=u['K'],physical_XYZ=X.tolist(),raw_observations=q.tolist(),valid9=p['valid9'],probabilities=p['probability'],reference=g,reference_kind='historical reconstructed geometry, not independent physical pose GT',panels=panels))
fig.suptitle('Frozen synthetic external-ray proxy: real recovery, damage and controls',fontsize=13);fig.supxlabel('White dashed: reconstructed reference. Orange solid: pose projection. Cyan circles: original observed coordinates.\nMagenta crosses: excluded original coordinates; projected corners are not new observations. Same RGB/crop per row.\nT/R are whole-pose errors. C2 wood / C4 square historical contracts; repeated development data. No real fitting.\nRay blockage is not physical DIRECT visibility. Both real groups fail the predeclared adoption gate.',fontsize=8)
for ext in ['png','pdf','svg']:fig.savefig(ROOT/('Real_Proxy_Cases.'+ext),dpi=170)
plt.close(fig);save('FIGURE_MANIFEST.json',manifest);print('INDEPENDENT_AND_FIGURES_COMPLETE',len(selected),'metric_gap',max(allgaps),flush=True)
