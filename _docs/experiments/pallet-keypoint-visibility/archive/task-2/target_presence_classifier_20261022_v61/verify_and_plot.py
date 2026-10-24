from pathlib import Path
import sys,csv,json,math,time
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=Path(__file__).resolve().parent;OLD=R.parent/'external_ray_proxy_head_20261022_v60';sys.path.insert(0,str(OLD));from common import read,sha,utc
from head_and_roi import native_grid
from metrics import classification
def save(n,x):
 with(R/n).open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvsave(n,rr):
 with(R/n).open('x')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
start=time.monotonic();rows=list(csv.DictReader(open(R/'SYNTH_QUERY_PREDICTIONS.csv')));inputs=read(OLD/'PREPARED_INPUTS_SYNTH.json.gz');uu={u['id']:u for u in inputs};audit={u['id']:u for u in csv.DictReader(open(R/'MASK_CONTRACT_AUDIT.csv'))};checked=0
by={}
for r in rows:
 if r['valid']=='True':by.setdefault(r['id'],[]).append(r)
for fid,rr in by.items():
 a=audit[fid];assert sha(a['visible_path'])==a['visible_sha256'];im=Image.open(a['visible_path'])
 for r in rr:assert(im.getpixel((int(r['pixel_x']),int(r['pixel_y'])))>127)==(r['presence_target']=='1');checked+=1
verified=[];clear=[]
for split in ['train','validation']:
 a=[r for r in rows if r['split']==split and r['valid']=='True'];y=np.array([int(r['presence_target'])for r in a]);p=np.array([float(r['presence_probability'])for r in a]);tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel();old=next(r for r in csv.DictReader(open(R/'CLASSIFICATION_METRICS.csv'))if r['split']==('TRAIN'if split=='train'else'VAL')and r['arm']=='PRESENCE_HEAD');ap=float(average_precision_score(y,p));auc=float(roc_auc_score(y,p));assert abs(ap-float(old['AUPRC']))<1e-12 and abs(auc-float(old['AUROC']))<1e-12 and[int(old[k])for k in ['TP','FP','FN','TN']]==[tp,fp,fn,tn];verified.append(dict(split=split,n=len(y),AP=ap,AUROC=auc,TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),PASS=True))
 for band in ['CLEAR_INTERIOR_OR_EXTERIOR','UNCERTAIN_BOUNDARY_OR_IMAGE_BORDER']:
  rr=[r for r in a if (r['boundary'].startswith('CLEAR_'))==(band=='CLEAR_INTERIOR_OR_EXTERIOR')];s=classification([int(r['presence_target'])for r in rr],[float(r['presence_probability'])for r in rr]);clear.append(dict(split=split,band=band,**s))
save('INDEPENDENT_VERIFICATION.json',dict(UTC=utc(),PASS=True,mask_center_labels_independently_checked_PIL=checked,metrics=verified,method='PIL getpixel independent label replay; sklearn AP/AUROC/confusion against custom metrics',seconds=time.monotonic()-start));csvsave('CLEAR_VS_UNCERTAIN_METRICS.csv',clear)
va=[r for r in rows if r['split']=='validation'and r['valid']=='True'];selection=[]
for name,target,pred,boundary in [('Clear TP',1,1,'CLEAR_INTERIOR'),('Clear TN',0,0,'CLEAR_EXTERIOR'),('Clear FP',0,1,'CLEAR_EXTERIOR'),('Clear FN',1,0,'CLEAR_INTERIOR'),('Boundary FP',0,1,'BOUNDARY_UNCERTAIN'),('Boundary FN',1,0,'BOUNDARY_UNCERTAIN')]:
 pool=[r for r in va if int(r['presence_target'])==target and(r['predicted_present']=='True')==bool(pred)and r['boundary']==boundary];s=min(pool,key=lambda r:(r['id'],int(r['native_corner'])));selection.append(dict(purpose=name,**s))
save('SYNTH_FIGURE_SELECTION.json',selection);fig,axs=plt.subplots(6,4,figsize=(12,13.7),layout='constrained');manifest=[]
for row,s in enumerate(selection):
 u=uu[s['id']];a=audit[s['id']];rgb=np.array(Image.open(u['image']).convert('RGB'));mask=np.array(Image.open(a['visible_path']))>127;k=int(s['native_corner']);q=np.array(u['initial_points'][k]);_,_,L=native_grid(u['initial_points'],u['raw_hw']);lo=q-L/2;hi=q+L/2;xx,yy=int(s['pixel_x']),int(s['pixel_y']);color='#00bd7f'if s['predicted_present']=='True'else'#dc267f'
 for col in range(4):
  ax=axs[row,col];ax.set_axis_off()
  if col==0:ax.imshow(rgb);ax.add_patch(Rectangle(lo,L,L,fill=False,edgecolor='#00aaaa',lw=1.2));ax.set_title(s['purpose']+' | '+s['id']+f' / k{k}',fontsize=7.5)
  else:
   ax.imshow(mask if col==2 else rgb,cmap='gray',vmin=0,vmax=1)if col==2 else ax.imshow(rgb);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.plot(q[0],q[1],marker='+',c=color if col==3 else'#25bbee',ms=10,mew=1.5);ax.add_patch(Rectangle((xx-.5,yy-.5),1,1,fill=False,edgecolor='yellow',lw=.8));ax.set_title('Native query crop'if col==1 else'M4: white = target'if col==2 else f'p(present)={float(s["presence_probability"]):.3f}',fontsize=8)
  if col==3:ax.text(.5,-.04,f'Center pixel target={s["presence_target"]}; predicted={int(s["predicted_present"]=="True")}\n'+s['boundary'],ha='center',va='top',transform=ax.transAxes,fontsize=7)
 manifest.append(dict(id=s['id'],native_corner=k,purpose=s['purpose'],image_sha256=u['image_sha256'],mask_sha256=a['visible_sha256'],query_xy=q.tolist(),nearest_pixel_xy=[xx,yy],same_crop_xyxy=[*lo.tolist(),*hi.tolist()],label=int(s['presence_target']),probability=float(s['presence_probability']),selection='posthoc errors/controls, fixed lexical ID/corner; not model selection'))
fig.suptitle('Synthetic validation: visible target presence at the initial predicted query',fontsize=13);fig.supxlabel('The single nearest pixel defines presence. Yellow 1-pixel square marks that pixel; + marks the unchanged query.\n3x3 mixed support marks boundary uncertainty only; no patch-vote target. Green predicted present / magenta predicted absent.\nThese are target-pixel labels, not semantic-corner correctness or physical DIRECT visibility. No pose evaluation.',fontsize=8)
for ext in ['png','pdf','svg']:fig.savefig(R/('Synthetic_Presence_Cases.'+ext),dpi=160)
plt.close(fig);save('SYNTH_FIGURE_MANIFEST.json',manifest)
# Real examples: fixed lexical recording/ID within each group, with low/high predicted-support cases.
real=read(OLD/'PREPARED_INPUTS_REAL_VERIFIED_V2.json.gz');rp=np.load(R/'REAL_PRESENCE_OUTPUTS.npz');selections=[]
for group in ['WOOD_PRIMARY105','SQUARE118']:
 pool=[(i,u)for i,u in enumerate(real)if u['group']==group];seen=set()
 for predicate in ['SOME_PREDICTED_ABSENCE','ALL_PREDICTED_PRESENCE']:
  candidates=[(i,u)for i,u in pool if u['id']not in seen and (bool((rp['probability'][i]<.5).any())if predicate=='SOME_PREDICTED_ABSENCE'else bool((rp['probability'][i]>=.5).all()))]
  if not candidates:continue
  i,u=min(candidates,key=lambda t:(t[1]['recording'],t[1]['id']));selections.append((i,u,predicate));seen.add(u['id'])
save('REAL_FIGURE_SELECTION.json',[dict(id=u['id'],group=u['group'],category=c,selection='posthoc predicted category, lexical recording/ID; no target truth available')for _,u,c in selections]);fig,axs=plt.subplots(len(selections),2,figsize=(9.8,3.3*len(selections)),layout='constrained');man=[]
for row,(i,u,c)in enumerate(selections):
 rgb=np.array(Image.open(u['image']).convert('RGB'));q=np.array(u['initial_points'])[:8];H,W=u['raw_hw'];valid=np.array([0<=a<W and 0<=b<H for a,b in q]);good=q[valid];lo=np.maximum([0,0],good.min(0)-25);hi=np.minimum([W-1,H-1],good.max(0)+25)
 for col in range(2):
  ax=axs[row,col];ax.imshow(rgb);ax.set_axis_off();ax.set_title(u['id']+(' | RGB'if col==0 else' | presence predictions; truth UNKNOWN'),fontsize=8)
  if col==0:ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,color='#00aaaa',lw=1.1))
  else:
   ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
   for k,(point,p)in enumerate(zip(q,rp['probability'][i])):
    if not valid[k]:continue
    present=p>=.5;ax.plot(*point,marker='o'if present else'x',c='#00d6a0'if present else'#f23f93',ms=6,mew=1.3,mfc='none');ax.annotate(f'{k}: {p:.2f}',point,xytext=(4,4),textcoords='offset points',fontsize=7,color='white',bbox=dict(facecolor='black',alpha=.65,pad=.5))
 man.append(dict(id=u['id'],image_sha256=u['image_sha256'],queries=q.tolist(),probabilities=rp['probability'][i].tolist(),crop_xyxy=[*lo.tolist(),*hi.tolist()],real_target_mask_truth='UNAVAILABLE',accuracy_computed=False))
fig.suptitle('Frozen synthetic presence head: real query overlays only',fontsize=13);fig.supxlabel('Green circle: predicted target present; magenta x: predicted target absent. Numbers are native query ID and probability.\nNo real target-pixel masks: these overlays are qualitative predictions, not measured real accuracy.\nThe same frozen head, threshold0.5, and cached original coordinates are used; no PnP, fitting or coordinate correction.',fontsize=8)
for ext in ['png','pdf','svg']:fig.savefig(R/('Real_Presence_Overlays.'+ext),dpi=160)
plt.close(fig);save('REAL_FIGURE_MANIFEST.json',man)
v=[r for r in va];y=np.array([int(r['presence_target'])for r in v]);p=np.array([float(r['presence_probability'])for r in v]);pr,re,_=precision_recall_curve(y,p);fig,ax=plt.subplots(figsize=(5.5,4.1),layout='constrained');ax.plot(re,pr,label=f'Head AP={average_precision_score(y,p):.3f}');ax.axhline(y.mean(),ls='--',c='gray',label=f'Prevalence={y.mean():.3f}');ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall: visible target presence',ylabel='Precision',title='Repeated synthetic validation, 1742 queries');ax.legend();fig.savefig(R/'Validation_PR.pdf');fig.savefig(R/'Validation_PR.png',dpi=150);plt.close(fig);print('PRESENCE_VERIFY_FIGURES_COMPLETE',checked,verified,flush=True)
