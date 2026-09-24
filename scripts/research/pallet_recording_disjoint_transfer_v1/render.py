"""Posthoc compact charts and privacy-cropped cases, never used for fitting."""
import hashlib
import html
import numpy as np
import cv2
from . import common as C
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_existing_data_transfer_v1.report import crop, tile
from scripts.research.pallet_clean19_pose_mismatch_v1.render import draw_edges, projection
from scripts.research.pallet_clean19_structured_easyhard_v1.augmentation import apply

COLORS={'R0':'#959ea7','S0':'#397ab1','S1':'#df8c30','TEACHER':'#57a57a'}

def png(name):
    plt.tight_layout();plt.savefig(C.DOC/'figures'/name,dpi=150);plt.close()

def bars(ax,labels,values,title,ylabel):
    x=np.arange(len(labels));arms=list(values);w=.8/len(arms)
    for j,a in enumerate(arms):
        data=values[a];rects=ax.bar(x+(j-(len(arms)-1)/2)*w,data,w,label=a,color=COLORS.get(a))
        if len(labels)<=4:
            ax.bar_label(rects,fmt='%.3f',fontsize=7,padding=2)
    ax.set(xticks=x,xticklabels=labels,title=title,ylabel=ylabel);ax.legend(fontsize=8)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)

def charts():
    groups=C.read(C.DOC/'RESULTS.json')['groups']; split=C.read(C.DOC/'RECORDING_DISJOINT_AUDIT.json')
    names=['CLEAN','MODERATE','SEVERE']; labels=[f'{n.title()} ({groups[n]["S0"]["current"]["frames"]})' for n in names]
    fig,ax=plt.subplots(figsize=(10,4));recs=split['train_recordings']+split['heldout_recordings']
    ax.bar(recs,[split['train_recording_counts'].get(r,0) for r in recs],label='Clean10 train',color=COLORS['S0'])
    ax.bar(recs,[split['recording_counts'].get(r,0) for r in recs],label='Natural HELDOUT128 (reused DEV)',color=COLORS['S1'])
    ax.set(title='Actual recording mapping: zero train/eval overlap',ylabel='Images');ax.legend();png('01_recording_disjoint_split.png')
    import json
    plans=[r for r in map(json.loads,(C.E.C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()) if r['material']=='PLASTIC' and r['real'] and r['plan']['applied']]
    r=min(plans,key=lambda r:hashlib.sha256(f"input-contract:{r['epoch']}:{r['slot']}".encode()).hexdigest())
    with np.load(C.ROOT/r['cache']['path']) as z:img=z['img'];kp=z['keypoints']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    box=np.array(r['plan']['bbox']);x0,y0=np.maximum(0,np.floor(box[:2])).astype(int);x1,y1=np.minimum([640,640],np.ceil(box[2:])).astype(int)
    face=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    for ax,arm in zip(axes,('S0','S1')):
        rgb=apply(img,r['plan'],arm).transpose(1,2,0)[y0:y1,x0:x1].copy()
        for x,y,w,h in face.detectMultiScale(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),1.1,4,minSize=(20,20)):
            rgb[y:y+h,x:x+w]=cv2.resize(cv2.resize(rgb[y:y+h,x:x+w],(4,4)),(w,h),interpolation=cv2.INTER_NEAREST)
        ax.imshow(rgb);ax.set_title(arm+(' | no extra occlusion' if arm=='S0' else ' | same target, random fill'));ax.axis('off')
    fig.suptitle('Only RGB differs: 730/2560 real occurrences; source 2560 unchanged')
    png('02_s0_s1_input_contract.png')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,k in zip(axs,('5','10','20')):
        bars(ax,labels,{a:[groups[n][a]['twoD']['PCK'][k] for n in names] for a in ('S0','S1')},'Natural PCK'+k,'Fraction');ax.set_ylim(0,1.03)
    png('03_natural_pck_by_severity.png')
    for key,name,title in [('current','04_current_pose_by_severity.png','CURRENT production D9'),('oracle','05_oracle_candidate_by_severity.png','ORACLE_WD: POSTHOC, NONDEPLOYABLE')]:
        fig,ax=plt.subplots(figsize=(10,4));bars(ax,labels,{a:[groups[n][a][key]['ADDsym_AUC'] for n in names] for a in ('S0','S1')},title,'ADDsym AUC (higher better)');png(name)
    fig,ax=plt.subplots(figsize=(10,4));bars(ax,labels,{a:[groups[n][a]['selection_loss'] for n in names] for a in ('S0','S1')},'Candidate benefit lost by current selection','ORACLE - CURRENT AUC');png('06_selection_loss.png')
    source=C.read(C.DOC/'SOURCE_PRESERVATION.json')['arms']
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    bars(axs[0],['Natural Clean29','Synthetic256'],{a:[groups['CLEAN'][a]['twoD']['PCK']['10'],source[a]['PCK']['10']['fraction']] for a in ('S0','S1')},'Preservation trade-off','PCK10')
    bars(axs[1],['Clean CURRENT','Clean ORACLE'],{a:[groups['CLEAN'][a][k]['ADDsym_AUC'] for k in ('current','oracle')] for a in ('S0','S1')},'Clean pose; source pose not rerun','ADDsym AUC');png('07_clean_preservation.png')
    anchors=C.read(C.DOC/'VERIFIED_ANCHOR_TRANSFER.json')['groups'];gs=['CLEAN','MODERATE','SEVERE','HARD']
    fig,ax=plt.subplots(figsize=(11,4));bars(ax,[f"{g} ({anchors[g]['S0']['n']} pts)" for g in gs],{a:[anchors[g][a]['PCK']['10']['fraction'] for g in gs] for a in ('R0','S0','S1','TEACHER')},'Final human-verified DIRECT_VISIBLE / fixed ID / small supplement','PCK10');ax.set_ylim(0,1.05);png('08_verified_visible_transfer.png')
    hist=C.read(C.DOC/'HISTORICAL_DIRECTION.json')['groups'];gn=['ALL','CLEAN','MODERATE','SEVERE']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,k,title in zip(axes,('PCK10_delta_pp','ADDsym_delta'),('PCK10 delta (percentage points)','CURRENT AUC delta')):
        for j,(key,label,color) in enumerate([('historical_delta','Historical300','#aaaaaa'),('historical_plastic_delta','Historical plastic184','#397ab1'),('recording_disjoint_delta','Recording-disjoint128','#df8c30')]):
            ax.bar(np.arange(4)+(j-1)*.25,[hist[g][key][k] for g in gn],.25,label=label,color=color)
        ax.axhline(0,color='black',lw=.7);ax.set(xticks=np.arange(4),xticklabels=gn,title=title);ax.legend(fontsize=7)
    fig.suptitle('S1-S0 direction only: different denominators, reused DEV')
    png('09_historical_vs_recording_disjoint.png')
    decision=C.read(C.DOC/'TRANSFER_DECISION.json')
    fig,ax=plt.subplots(figsize=(12,4));ax.axis('off')
    texts=[('CONTROLLED PAIR','Clean10 / synthetic512\n320 updates, same R0\nOnly conditional random fill'),
           ('OTHER RECORDINGS','Clean29 / Moderate21 / Severe78\nZero fitting recording overlap\nReused DEV, NOT final TEST'),
           ('OBSERVED TRADE-OFF','Severe: PCK / candidate / current +\nModerate: candidate + / current -\nClean: PCK / pose -'),
           ('ONE NEXT DESIGN','Bounded role-contract normalization\n2/10 role-mismatch signals\nNO training or automatic relabel')]
    for i,(title,body) in enumerate(texts):
        ax.text(.02+i*.255,.5,title+'\n\n'+body,transform=ax.transAxes,va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.8',fc='#edf2f5',ec='#397ab1'))
    ax.set_title(decision['primary'],fontsize=13,pad=15);png('10_transfer_routing.png')
    role=C.read(C.DOC/'H10_ROLE_PREVALENCE.json');fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,arm in zip(axes,('TEACHER','T1','T2')):
        x=np.arange(10);ax.bar(x-.18,[r['models'][arm]['same_ID_mean_px'] for r in role['rows']],.36,label='Stored same-ID')
        ax.bar(x+.18,[r['models'][arm]['best_C4']['mean_px'] for r in role['rows']],.36,label='Best whole C4 (diagnostic only)')
        ax.set_ylabel(arm+' mean px');ax.legend(fontsize=7)
    axes[-1].set(xticks=np.arange(10),xticklabels=[r['id'].split(':')[-1] for r in role['rows']]);fig.suptitle('Fixed H10 common support: two frames flagged; no GT edits');png('01_h10_role_prevalence.png')

def cases():
    rr=C.records();by={r['id']:r for r in rr};pm=C.read(C.RAW/'POSE_METRICS.json');fm=C.read(C.RAW/'FRAME_METRICS.json')
    poses=C.read(C.RAW/'POSE_PREDICTIONS.json');preds=C.read(C.RAW/'PREDICTIONS.json');truth=C.read(C.E.P.RAW/'TRUTH_FOR_DISPLAY_ONLY.json')
    meta={r['id']:r for r in C.read(C.E.V.RAW/'INFERENCE_METADATA.json')}
    def delta(fid):
        a,b=(pm[arm][fid]['current'] for arm in ('S0','S1'))
        return b['ADDsym_normalized']-a['ADDsym_normalized'] if a['available'] and b['available'] else None
    selected=[]
    for group in ('MODERATE','SEVERE'):
        ids=[r['id'] for r in rr if r['severity']==C.SEVS[group] and delta(r['id']) is not None]
        for label,ok,reverse in [('improved',lambda d:d<0,False),('harmed',lambda d:d>0,True)]:
            pool=[i for i in ids if ok(delta(i))]
            chosen=sorted(pool,key=lambda i:(delta(i),i),reverse=reverse)[:4]
            assert len(chosen)==4
            selected.extend(dict(id=i,group=group,label=label,ADD_delta=delta(i)) for i in chosen)
    random=sorted(by,key=lambda i:hashlib.sha256(('recording-disjoint-v1|'+i).encode()).hexdigest())[:6]
    selected.extend(dict(id=i,group='RANDOM',label='fixed_hash_control',ADD_delta=delta(i)) for i in random)
    for index,row in enumerate(selected,1):
        fid=row['id'];r=by[fid];base,offset,scale=crop(r,truth[fid]['box']);K=np.array(meta[fid]['K']);panels=[]
        for arm in ('RGB',*C.ARMS):
            im=base.copy();lines=[arm+' | '+fid,row['group']+' / '+row['label']]
            if arm=='RGB':
                lines+=['Display-only tight pallet ROI','Yellow: raw predicted keypoints','Red: CURRENT pose projection','Cyan dashed: alternate W/D pose','Green crosses: legacy reference','ORACLE: POSTHOC, NONDEPLOYABLE']
            else:
                q=C.E.D.points(preds[arm][fid]);p=poses[arm][fid];m=pm[arm][fid];f=fm[arm][fid]
                if q is not None:draw_edges(im,(q-offset)*scale,(0,220,255))
                if p['current']['available']:draw_edges(im,(projection(p['current'],K)-offset)*scale,(20,30,255))
                for h in p['hypotheses']:
                    if h['name']!=p['current'].get('selected_hypothesis') and h['pose']['available']:
                        draw_edges(im,(projection(h['pose'],K)-offset)*scale,(255,220,0),True)
                for j in range(8):
                    if truth[fid]['valid'][j]:
                        pt=np.clip((np.array(truth[fid]['gt'][j])-offset)*scale,-10000,10000).astype(int)
                        cv2.drawMarker(im,tuple(pt),(0,255,0),cv2.MARKER_CROSS,11,1)
                def fmt(x):return 'NA' if x is None else f'{x:.4f}'
                lines += [f"PCK10 {sum(e<=10 for e in f['errors'])}/{f['corners']} | matched {f['matched']}",
                    'CURRENT ADDnorm '+fmt(m['current'].get('ADDsym_normalized')),
                    'ORACLE ADDnorm '+fmt(m['oracle'].get('ADDsym_normalized')),
                    'Selected '+str(p['current'].get('selected_hypothesis')),
                    'Axis parity '+str(m['current'].get('axis_correct')),
                    'R/yaw/t(cm): '+' / '.join(fmt(m['current'].get(k)) for k in ('rotation_deg','yaw_deg','translation_cm'))]
            panels.append(tile(im,lines,height=min(560,220+base.shape[0])))
        name=f"case_{index:02d}_{row['group'].lower()}_{row['label']}.jpg"
        assert cv2.imwrite(str(C.DOC/'figures'/name),np.hstack(panels),[cv2.IMWRITE_JPEG_QUALITY,85]);row['figure']='figures/'+name
    C.save(C.DOC/'CASE_MANIFEST.json',dict(cases=selected,selection='Posthoc extrema of S1-S0 CURRENT normalized ADD; NOT2D. Available poses only for extrema.',
        random_rule='First6 lexicographically sorted sha256(recording-disjoint-v1|frame_id), independent of output/GT',random_selected=random,
        privacy='Display-only tight pallet ROI, Haar face pixelation; no full originals, no changed inference inputs',
        representatives_not_population=True))
    cards='\n'.join(f"<h2>{html.escape(r['group']+' '+r['label']+' '+r['id'])}</h2><img src='../../_docs/experiments/{C.NAME}/{r['figure']}'>" for r in selected)
    C.save(C.OUT/'GALLERY.html',"<!doctype html><meta charset='utf-8'><title>S0/S1 recording transfer</title><style>body{background:#15232b;color:#eee;font-family:sans-serif}img{width:100%}</style><h1>R0 / S0 / S1 — frozen recording-heldout128</h1><p>22 posthoc displays. Yellow raw2D, red current pose, cyan alternate, green legacy reference. No model or GT edits.</p>"+cards)
    print('FIGURES_COMPLETE',len(selected),'cases',flush=True)

def main():
    (C.DOC/'figures').mkdir(exist_ok=True);charts();cases()

if __name__=='__main__':main()
