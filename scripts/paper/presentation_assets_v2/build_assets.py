"""Read-only N3 scientific figure builder. No inference, fitting, or optimizer calls.

Run with pallet-yolo26 Python. --probe writes a preview ONLY under /tmp.
The normal command refuses to overwrite any completed figure/manifest.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as effects
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / '_docs/paper/presentation_assets_v2'
CODE = ROOT / 'scripts/research/pallet_dim_conditioned_p_v1'
sys.path.insert(0, str(CODE))
import dcp_env as E
from dev_evaluate import population_metadata, iou
from eval_math import measure

REGISTRY = ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
CONTRACT = E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'
SCORES = ROOT/'_docs/experiments/pallet_final_paper_tables_v1/RESCORED_2D.json'
GREEN = ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json'
BASE = E.LINE/'baseline/FULL_CANDIDATES.json'
OLD = ROOT/'_docs/paper/sensors_submission_v1'
TYPES = ['plastic_standard_110x130x11', 'wood_small_80x59x14', 'plastic_standard_110x110x15']
LABELS = ['Rectangular plastic', 'Rectangular wood', 'Square plastic']
NAMES = ['01_method_pipeline', '02_dimension_conditioning_examples',
         '03_residual_refinement_examples', '04_symmetry_contract_C2_C4',
         '05_local_candidate_refinement']
BLUE, PINK, GREEN_C, INK, MUTED = '#007dcc', '#db267c', '#67ed75', '#172b42', '#50657a'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':13, 'text.color':INK,
                     'pdf.fonttype':42, 'axes.labelcolor':INK, 'savefig.facecolor':'white'})
read = E.read

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def binding(p):
    p = Path(p).resolve()
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)

def rgb(path):
    a = cv2.imread(str(ROOT/path))
    assert a is not None, path
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def dimension_contract():
    rr = {r['object_type']:r for r in read(REGISTRY)['objects']}
    cc = {r['object_type']:r for r in read(CONTRACT)['objects']}
    dims = {}
    for typ, expected, order in zip(TYPES, [[1.1,1.3,.11],[.8,.59,.14],[1.1,1.1,.15]], [2,2,4]):
        d = rr[typ]['physical_dimensions_m']
        dims[typ] = [d['x'], d['z'], d['y']]
        assert dims[typ] == expected and cc[typ]['dimensions_m'] == d
        assert cc[typ]['group_order'] == order
        points = np.array(cc[typ]['corners_centroid'])
        for rot, perm in zip(cc[typ]['rotations'], cc[typ]['permutations']):
            assert perm[8] == 8 and sorted(perm) == list(range(9))
            np.testing.assert_allclose(points@np.array(rot).T, points[perm], atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(rot), 1, atol=1e-12)
    return dims, cc

def load_data():
    dims, groups = dimension_contract()
    pe, population = population_metadata()
    baseline = read(BASE)
    assert baseline['weights_sha256'] == E.R0_SHA == sha(E.R0)
    pp = [E.RAW/f'predictions/REAL_DEV/N3_DIM_SYM_seed{s}.json' for s in (1,2,3)]
    pred = [read(p) for p in pp]
    ids = [i.frame_id for i,_ in population]
    assert len(ids) == len(set(ids)) == 319
    for p in pred:
        assert p['complete'] and p['GT_input'] is False
        assert [r['id'] for r in p['records']] == ids
        assert sha(ROOT/p['checkpoint']['path']) == p['checkpoint']['sha256']
    old = read(SCORES)['rows']
    rows = []
    for j,(item,meta) in enumerate(population):
        t = pe.E._legacy_forbidden_target(item)
        im = rgb(item.image); hw = im.shape[:2]
        cs = baseline['frames'][pe.canonical_key(item.image)]
        idx = int(np.argmax([c['score'] for c in cs])) if cs else None
        assert idx is not None
        b = cs[idx]; r0 = np.array(b['keypoints_xy'], float)
        n3 = []
        for p in pred:
            q = p['records'][j]; assert q['selected_index'] == idx
            assert len(q['candidates']) == len(cs)
            for k,(before,after) in enumerate(zip(cs,q['candidates'])):
                for field in before:
                    if field != 'keypoints_xy':
                        assert np.array_equal(np.asarray(before[field]), np.asarray(after[field]))
                a,bp = np.array(before['keypoints_xy']),np.array(after['keypoints_xy'])
                assert np.array_equal(a[8:],bp[8:])
                if k != idx: assert np.array_equal(a,bp)
            points = np.array(q['candidates'][idx]['keypoints_xy'],float)
            valid = np.isfinite(r0[:8]).all(-1) & ~(r0[:8]==-1).all(-1)
            assert np.linalg.norm(points[:8]-r0[:8],axis=-1)[valid].max() <= .01*math.hypot(*hw)+.0002
            n3.append(points)
        target = np.array(t.keypoints_xy,float); mask = np.array(t.keypoint_supervision_mask,bool)
        metrics = []
        matched = iou(b['box_xyxy'],t.box_xyxy)>=.5
        for name,points in zip(['R0']+[f'N3_DIM_SYM_seed{s}' for s in (1,2,3)], [r0]+n3):
            mm = measure(points,target,mask,groups[meta['object_type']]['permutations'],hw,matched,True)
            expected = old[name][j]; assert expected['id'] == item.frame_id
            for field in mm:
                if isinstance(mm[field],float): assert abs(mm[field]-expected[field])<1e-10, (item.frame_id,name,field)
                else: assert mm[field] == expected[field], (item.frame_id,name,field)
            metrics.append(mm)
        errors = [m['frame_mean_px'] for m in metrics]
        rows.append(dict(id=item.frame_id,image=item.image,annotation=item.label,object_type=meta['object_type'],
                         dimensions=dims[meta['object_type']],hw=list(hw),r0=r0.tolist(),n3=[x.tolist() for x in n3],
                         gt=target.tolist(),valid=mask.tolist(),metrics=metrics,
                         errors=errors,delta=[x-errors[0] for x in errors[1:]],
                         mean_delta=float(np.mean(errors[1:])-errors[0])))
    return rows, pred, dims, groups

def select_cases(rows):
    improved = [r for r in rows if r['mean_delta'] < -1e-9]
    harmed = [r for r in rows if r['mean_delta'] > 1e-9]
    all_improve = [r for r in improved if max(r['delta']) < -1e-9]
    a_target = float(np.quantile([r['mean_delta'] for r in improved],.225))
    b_target = float(np.median([r['mean_delta'] for r in improved]))
    c_target = float(np.median([r['mean_delta'] for r in harmed]))
    nearest = lambda pool,target:min(pool,key=lambda r:(abs(r['mean_delta']-target),r['id']))
    selected = [nearest(all_improve,a_target),nearest(improved,b_target),nearest(harmed,c_target)]
    assert len({r['id'] for r in selected})==3
    return selected, dict(metric='current eval_math.measure frame_mean_px; whole-object contract; corners0..7',
        sign='delta = N3 - R0; negative improves',seed_aggregation='arithmetic mean of seed1/2/3 deltas',
        display_seed=1,seed_selected_by_error=False,improved=len(improved),harmed=len(harmed),all_three_improve=len(all_improve),
        A=dict(pool='all three seeds improve',target_quantile_of_all_negative_deltas=.225,target=a_target),
        B=dict(pool='negative mean deltas',target_quantile=.5,target=b_target),
        C=dict(pool='positive mean deltas',target_quantile=.5,target=c_target),tie_break='lexical frame ID')

def dimension_examples(rows,dims):
    # Readability only: largest visible top-face fraction; never prediction error.
    candidates = [dict(id=r['id'],image=r['image'],annotation=r['annotation'],object_type=r['object_type']) for r in rows]
    for r in read(GREEN)['records']:
        candidates.append(dict(id=r['id'],image=r['image']['path'],annotation=r['annotation']['path'],object_type=TYPES[2]))
    pools = {t:[] for t in TYPES}
    for r in candidates:
        o = read(ROOT/r['annotation'])['objects'][0]
        p = np.array(o['projected_cuboid'],float)
        if p.shape != (8,2): continue
        w,hgt,dep = [o['dimensions_m'][k] for k in ('width','height','depth')]
        # For display, require annotated camera-facing axes already agree with canonical lengths.
        # The dimension INPUT always comes from registry; no target-derived W/D swap.
        if [w,dep,hgt] != dims[r['object_type']]: continue
        im = rgb(r['image']); h,w = im.shape[:2]
        if not np.isfinite(p).all() or not ((p[:,0]>15)&(p[:,0]<w-15)&(p[:,1]>15)&(p[:,1]<h-15)).all(): continue
        score = abs(cv2.contourArea(p[[0,1,5,4]].astype(np.float32)))/(h*w)
        # Require useful top-face and visible height on the illustrated anchors.
        if np.linalg.norm(p[0]-p[3])<12: continue
        pools[r['object_type']].append(dict(**r,dimensions=dims[r['object_type']],anchors=p.tolist(),
             readability_score=score,axis_edges={'W':[0,1],'D':[0,4],'H':[0,3]},hw=[h,w]))
    return [sorted(pools[t],key=lambda r:(-r['readability_score'],r['id']))[0] for t in TYPES]

def fig():
    return plt.figure(figsize=(16,9),dpi=200,facecolor='white')

def canvas(f):
    ax=f.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off');return ax

def box(ax,x,y,w,h,text,fc='#eef4fa',ec='#c4d3e3',size=14,dashed=False):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12,rounding_size=.12',facecolor=fc,edgecolor=ec,
                        linewidth=1.3,linestyle='--' if dashed else '-')
    ax.add_patch(patch);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.45)

def arrow(ax,a,b,color=MUTED,dashed=False,lw=1.7):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=color,lw=lw,
                                              linestyle='--' if dashed else '-',shrinkA=2,shrinkB=2))

def save(f,name):
    for suffix in ('png','pdf'):
        p=OUT/f'{name}.{suffix}';assert not p.exists(),p
        f.savefig(p,dpi=200,metadata={'Creator':'presentation_assets_v2; optimizer updates = 0'} if suffix=='pdf' else None)
    plt.close(f)

def pipeline():
    f=fig();a=canvas(f)
    a.text(.5,8.55,'A  /  INFERENCE',size=15,weight='bold')
    box(a,.5,5.7,1.35,1.15,'RGB\nimage')
    box(a,2.3,5.7,2.2,1.15,'Frozen\nYOLO26n-Pose')
    box(a,5.05,6.6,2.55,.85,'Initial 8 corners',size=13)
    box(a,5.05,5.15,2.55,.85,'P3 / P4 features',size=13)
    box(a,8.25,5.75,2.7,1.2,'Local Keypoint\nRefiner (N3)',fc='#e3f4f0',ec='#58a590')
    box(a,11.6,5.75,1.9,1.2,'Refined\n8 corners')
    box(a,11.6,3.65,1.9,1.05,'Prediction-only\nPnP',size=13)
    box(a,14.05,3.65,1.45,1.05,'6D pose\n(R, t)',size=13)
    box(a,8.15,7.65,2.9,.7,'Known [W, D, H]',fc='#fff2d9',ec='#d8af52',size=13)
    box(a,11.5,7.65,3.9,.7,'Known geometry + camera K',fc='#fff2d9',ec='#d8af52',size=12)
    for st,en in [((1.85,6.27),(2.3,6.27)),((4.5,6.4),(5.05,7)),((4.5,6.15),(5.05,5.6)),
                  ((7.6,7),(8.25,6.55)),((7.6,5.6),(8.25,6.05)),((9.6,7.65),(9.6,6.95)),
                  ((10.95,6.35),(11.6,6.35)),((12.55,5.75),(12.55,4.7)),((13.5,4.18),(14.05,4.18))]: arrow(a,st,en)
    # Route geometry to PnP outside the keypoint path.
    a.plot([15.1,15.6,15.6,13.8],[7.65,7.35,5.05,5.05],color=MUTED,lw=1.5)
    arrow(a,(13.8,5.05),(13.1,4.7))
    a.text(.55,4.15,'Only corner xy 0–7 change.\nBBox, confidence, selected instance and center 8 stay fixed.',size=12,linespacing=1.6)
    a.add_patch(FancyBboxPatch((.45,.9),15.1,2.25,boxstyle='round,pad=.08',facecolor='#f7f5fc',edgecolor='#9c8bb7',ls='--',lw=1.5))
    a.text(.75,2.77,'B  /  TRAINING ONLY',size=13,weight='bold',color='#79638f')
    box(a,.85,1.2,3.45,1.05,'GT keypoints + predefined\nobject/task C2 or C4 contract',fc='white',size=12)
    box(a,5.1,1.2,4.2,1.05,'Whole-object correspondence\n(one valid rotation for all corners)',fc='white',size=12)
    box(a,10.2,1.2,4.2,1.05,'Local residual supervision\n(candidate-distribution targets)',fc='white',size=12)
    arrow(a,(4.3,1.75),(5.1,1.75),dashed=True);arrow(a,(9.3,1.75),(10.2,1.75),dashed=True)
    a.plot([14.75,14.75,9.6],[2.05,3.38,3.38],ls='--',color='#79638f',lw=1.4)
    arrow(a,(9.6,3.38),(9.6,5.7),color='#79638f',dashed=True)
    a.text(10,3.05,'refiner training only',size=10,color='#79638f')
    a.text(.55,.35,'Dimensions condition scores; they do not infer symmetry. No GT assignment or explicit C2/C4 code enters N3 inference.',size=11)
    save(f,NAMES[0])

def exact_scatter(ax,points,**kwargs):
    points=np.asarray(points,float);artist=ax.scatter(points[:,0],points[:,1],**kwargs)
    assert np.array_equal(np.asarray(artist.get_offsets()),points)
    return artist

def dims_text(d): return '['+', '.join(f'{v:.2f}' for v in d)+'] m'

def dimensions_figure(examples):
    f=fig();a=canvas(f);receipts=[]
    for j,r in enumerate(examples):
        left=.035+j*.325
        a.text((left+.012)*16,8.35,f'{chr(65+j)}  {LABELS[j]}',size=15,weight='bold')
        a.text((left+.012)*16,7.86,dims_text(r['dimensions']),size=13)
        ax=f.add_axes([left,.34,.30,.49]);im=rgb(r['image']);ax.imshow(im);ax.axis('off')
        p=np.array(r['anchors']); lo=p.min(0)-45;hi=p.max(0)+45
        ax.set_xlim(max(0,lo[0]),min(im.shape[1],hi[0]));ax.set_ylim(min(im.shape[0],hi[1]),max(0,lo[1]))
        for name,color in [('W','#4dc7ff'),('D','#ff70b8'),('H','#ffe071')]:
            i,k=r['axis_edges'][name];x,y=p[i],p[k]
            patch=ax.annotate('',xy=y,xytext=x,arrowprops=dict(arrowstyle='<->',lw=2.5,color=color,shrinkA=0,shrinkB=0))
            patch.arrow_patch.set_path_effects([effects.Stroke(linewidth=4,foreground='#102333'),effects.Normal()])
            mid=(x+y)/2
            ax.annotate(name,mid,xytext=(6,9),textcoords='offset points',size=16,weight='bold',color=color,
                        bbox=dict(facecolor='#102333',edgecolor='none',alpha=.85,pad=2))
        exact_scatter(ax,p[[0,1,3,4]],s=12,c='white',zorder=5)
        receipts.append(dict(**r,source_coordinate_parity=True,axis_policy='display axes already match canonical registry lengths; no GT W/D swap',
                             image_crop=[float(v) for v in (*ax.get_xlim(),*ax.get_ylim())]))
    box(a,.7,1.4,3.6,.8,'Known canonical [W, D, H]',fc='#fff2d9',size=12)
    box(a,5.3,1.4,4.5,.8,'5-D log/ratio dimension feature',fc='#fff2d9',size=12)
    box(a,10.8,1.4,4.3,.8,'Candidate-score residual',fc='#e3f4f0',size=12)
    arrow(a,(4.3,1.8),(5.3,1.8));arrow(a,(9.8,1.8),(10.8,1.8))
    a.text(.7,.72,'Externally supplied object metadata, not an image-based size estimate. Symmetry is a separate object/task contract.',size=11)
    a.text(.7,.27,'RGB: internal report use; publication rights unverified. Square: input/contract illustration, not an N3 DEV319 result.',size=10,color=MUTED)
    save(f,NAMES[1]);return receipts

def overlay(ax,r):
    im=rgb(r['image']);ax.imshow(im);ax.axis('off')
    p=np.array(r['r0'])[:8];q=np.array(r['n3'][0])[:8];g=np.array(r['gt'])[:8]
    valid=np.array(r['valid'])[:8]&np.isfinite(g).all(-1)&~(g==-1).all(-1)
    exact_scatter(ax,g[valid],s=65,c=GREEN_C,marker='x',linewidths=1.6,zorder=6)
    exact_scatter(ax,p,s=48,facecolors='none',edgecolors=BLUE,linewidths=1.4,zorder=7)
    exact_scatter(ax,q,s=22,c=PINK,marker='D',edgecolors='white',linewidths=.4,zorder=8)
    for x,y in zip(p,q):
        ax.annotate('',xy=y,xytext=x,arrowprops=dict(arrowstyle='->',color='#ffe071',lw=1.2,shrinkA=0,shrinkB=0),zorder=9)
    return p,q,g,valid

def residual_figure(selected,rule):
    f=fig();a=canvas(f);receipts=[]
    legend=[Line2D([],[],marker='x',color=GREEN_C,ls='',label='GT (known corners)'),
            Line2D([],[],marker='o',mfc='none',color=BLUE,ls='',label='R0 initial'),
            Line2D([],[],marker='D',color=PINK,ls='',label='N3 refined (seed 1)'),
            Line2D([],[],color='#ca9900',label='R0 → N3 (true scale)')]
    f.legend(handles=legend,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.52,.994),fontsize=12)
    descriptions=['A  Clear improvement','B  Typical improvement','C  Harmed case']
    for j,(r,desc) in enumerate(zip(selected,descriptions)):
        y=.655-j*.286
        ax=f.add_axes([.025,y,.305,.263]); p,q,g,valid=overlay(ax,r)
        movement=np.linalg.norm(q-p,axis=-1);k=int(np.argmax(np.where(valid,movement,-1)))
        # Native pixel crop: no point/arrow amplification and no rescaling of stored xy.
        center=(p[k]+q[k]+g[k])/3;extent=max(30.,float(np.max(np.abs(np.stack([p[k],q[k],g[k]])-center)))+14)
        x1,x2=center[0]-extent,center[0]+extent;y1,y2=center[1]-extent,center[1]+extent
        ax.add_patch(Rectangle((x1,y1),2*extent,2*extent,fill=False,edgecolor='white',ls='--',lw=.9))
        zz=f.add_axes([.35,y,.225,.263]);overlay(zz,r);zz.set_xlim(x1,x2);zz.set_ylim(y2,y1)
        zz.text(.03,.95,f'Corner {k} zoom',transform=zz.transAxes,va='top',color='white',size=10,
                bbox=dict(facecolor=INK,edgecolor='none',alpha=.8,pad=3))
        a.text(9.7,(y+.236)*9,desc,size=15,weight='bold')
        typ=LABELS[TYPES.index(r['object_type'])]
        a.text(9.7,(y+.19)*9,typ+'  '+dims_text(r['dimensions']),size=11)
        b,n=r['errors'][:2]
        a.text(9.7,(y+.127)*9,f'R0 {b:.2f} px  →  N3 {n:.2f} px\nΔ {n-b:+.2f} px  (displayed seed 1)',size=14,linespacing=1.5)
        a.text(9.7,(y+.056)*9,f'3-seed mean Δ {r["mean_delta"]:+.2f} px\nMaximum corner move {movement.max():.2f} px',size=10,linespacing=1.5,color=MUTED)
        a.text(9.7,(y+.009)*9,r['id'],size=8.7,color=MUTED)
        receipts.append(dict(**r,display_seed=1,zoom_corner=k,zoom_xyxy=[x1,y1,x2,y2],max_move_px=float(movement.max()),
                             plotted_r0=p.tolist(),plotted_n3=q.tolist(),plotted_gt=g[valid].tolist(),plotted_gt_indices=np.flatnonzero(valid).tolist(),
                             plotted_coordinate_parity=True,arrows_scale=1.))
    a.text(.45,.34,'Error: symmetry-aware mean over known corners 0–7. Selection uses 3-seed mean Δ; overlays use fixed seed 1.',size=10)
    a.text(.45,.10,'Max displacement = 1% of original-image diagonal. GT is used for evaluation only. Internal RGB; publication rights unverified.',size=9.5)
    save(f,NAMES[2]);return receipts

def symmetry_figure(groups):
    f=fig();a=canvas(f)
    a.text(.65,8.45,'A  Rectangular pallet · C2 contract',size=16,weight='bold')
    a.text(8.2,8.45,'B  Square pallet · C4 contract',size=16,weight='bold')
    a.text(.65,7.87,'0° ≡ 180°    |    90° is not equivalent',size=14)
    a.text(8.2,7.87,'0° ≡ 90° ≡ 180° ≡ 270°',size=14)
    edges=read(CONTRACT)['edges']; edges=[x if isinstance(x,list) else x['corners'] for x in edges]
    # Oblique projection of the EXACT 3-D contract corners. No free matching.
    projection=np.array([[1.,.45],[0.,-2.5],[.65,-.5]])
    for side,typ in enumerate([TYPES[0],TYPES[2]]):
        c=groups[typ];points=np.array(c['corners_centroid']);base=points@projection
        for k,perm in enumerate(c['permutations']):
            col=k%2;row=k//2
            left=(.055 if side==0 else .535)+col*.225;bottom=.48-row*.245
            ax=f.add_axes([left,bottom,.19,.235]);ax.set_aspect('equal');ax.axis('off')
            for i,j in edges:ax.plot(base[[i,j],0],base[[i,j],1],color='#8ba0b4',lw=1.3)
            # Rotated point i occupies original spatial slot perm[i]. Labels stay attached to point i.
            rotated=base[perm]
            for i,(x,y) in enumerate(rotated[:8]):
                ax.scatter(x,y,s=175,facecolor='#e7f2fc',edgecolor=BLUE,zorder=3)
                ax.text(x,y,str(i),ha='center',va='center',fontsize=10,color=INK,zorder=4)
            ax.scatter(*rotated[8],s=12,c=PINK);ax.text(rotated[8,0],rotated[8,1]+.1,'8',ha='center',fontsize=10,color=PINK)
            ax.margins(.18);ax.text(.5,1.01,f'{k*360//c["group_order"]}°',transform=ax.transAxes,ha='center',size=13)
        a.text(.65+7.55*side,1.52,'slot i → π(i), excluding fixed center 8',size=11,color=MUTED)
        lines=[]
        for k,perm in enumerate(c['permutations']):
            lines.append(f'{k*360//c["group_order"]:3d}°  '+ ' '.join(map(str,perm[:8])))
        a.text(.65+7.55*side,1.30,'\n'.join(lines),family='DejaVu Sans Mono',size=10,va='top',linespacing=1.25)
    a.text(.65,3.4,'One rotation relabels the entire cuboid.\nNo independently reassigned corners.\nCenter 8 is fixed in every rotation.',size=13,linespacing=1.7)
    a.text(.65,.40,'Whole-object valid rotations only; no free pointwise/Hungarian matching.',size=12,weight='bold')
    a.text(.65,.12,'Equivalence is predefined by the object/task contract, not inferred from dimensions.',size=12)
    save(f,NAMES[3])

def candidate_figure(pred):
    ckpath=ROOT/pred[0]['checkpoint']['path'];ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    lattice=ck['model_state_dict']['displacements'].numpy()
    assert lattice.shape==(222,2) and np.array_equal(lattice[-1],[0.,0.])
    assert ck['arm']=='N3_DIM_SYM' and ck['step']==6000
    from generic_point_refiner import GenericPointRefiner
    torch.manual_seed(0);head=GenericPointRefiner(**ck['config'])
    assert np.array_equal(lattice,head.displacements.numpy())
    f=fig();a=canvas(f)
    ax=f.add_axes([.06,.23,.43,.66]);ax.set_aspect('equal')
    exact_scatter(ax,lattice[:-1],s=13,c='#88b7d5',alpha=.95)
    exact_scatter(ax,lattice[-1:],s=80,c=INK,marker='x',linewidths=2,zorder=4)
    ax.add_patch(Circle((0,0),.08,fill=False,color=BLUE,ls='--',lw=1.4))
    # Only demonstration decoding probabilities are synthetic. Lattice is checkpoint-exact.
    logits=-np.sum((lattice-np.array([.039,.023]))**2,axis=-1)/(2*.016**2)
    prob=np.exp(logits-logits.max());prob/=prob.sum();mu=(lattice*prob[:,None]).sum(0)
    # Schematic ratio bbox diagonal/image diagonal=0.5 => cap in plot units=0.02.
    cap=.02;out=mu*min(1,cap/np.linalg.norm(mu))
    ax.add_patch(Circle((0,0),cap,facecolor='#dcf1e8',edgecolor='#389777',alpha=.65,lw=1.3))
    arrow(ax,(0,0),mu,color='#b28b41',lw=2);arrow(ax,(0,0),out,color=PINK,lw=2.8)
    exact_scatter(ax,mu[None],s=55,c='#b28b41',marker='s',zorder=6)
    exact_scatter(ax,out[None],s=75,c=PINK,marker='D',zorder=7)
    ax.set(xlim=(-.092,.098),ylim=(-.092,.098),xlabel='Δx / bbox diagonal',ylabel='Δy / bbox diagonal')
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=10)
    ax.legend(handles=[Line2D([],[],color=INK,marker='x',ls='',label='Initial / null'),
        Line2D([],[],color='#b28b41',marker='s',ls='',label='Probability-weighted mean'),
        Line2D([],[],color=PINK,marker='D',ls='',label='Capped residual')],frameon=False,fontsize=10,loc='lower left')
    a.text(.75,8.4,'A  Exact local candidate lattice',size=15,weight='bold')
    a.text(8.55,8.4,'B  Score → expectation → cap',size=15,weight='bold')
    box(a,8.7,6.8,6.1,.8,'13 directions × 17 radii = 221 + null',size=14)
    box(a,8.7,5.3,6.1,.95,'P3 / P4 local image evidence\n+ dimension-conditioned score residual',fc='#fff2d9',size=13)
    box(a,8.7,3.85,6.1,.85,'Softmax probabilities → weighted mean',size=13)
    box(a,8.7,2.4,6.1,.85,'Cap: 0.01 × original-image diagonal',fc='#e3f4f0',size=13)
    for y in [6.8,5.3,3.85]:arrow(a,(11.75,y),(11.75,y-.55))
    a.text(.8,1.45,'Search extent: 0.08 × bbox diagonal. Search radius and output cap use different diagonals.',size=12)
    a.text(.8,.95,'Broad local evidence → small residual correction. The original N3 decoder is an expectation, not an argmax.',size=12)
    a.text(.8,.45,'Lattice is checkpoint-exact; probability/mean illustration is schematic (not measured logits).',size=10,color=MUTED)
    a.text(.8,.16,'Schematic bbox diagonal = 0.5 × image diagonal; λ = 1. Center 8 is never refined.',size=10,color=MUTED)
    save(f,NAMES[4])
    return dict(checkpoint=pred[0]['checkpoint'],lattice=lattice.tolist(),lattice_checkpoint_exact=True,
                directions=13,radial_levels=17,candidates=222,extent_bbox_diagonal=.08,
                cap_original_image_diagonal=.01,decoder='temperature softmax weighted expectation, lambda=1, radial cap',
                schematic=True,schematic_bbox_image_diagonal_ratio=.5,schematic_logits=logits.tolist(),
                schematic_mean=mu.tolist(),schematic_capped=out.tolist(),stencil_fraction=ck['config']['stencil_fraction'])

def protected_sources(pred):
    paths={REGISTRY,CONTRACT,SCORES,GREEN,BASE,E.R0,ROOT/'challenge/real_gt_v2/manifests/PAPER_EVAL_ALL_POS.json'}
    paths.update(CODE.glob('*.py'))
    paths.update((ROOT/'scripts/research/pallet_final_ml_contribution_test_v1').glob('*.py'))
    paths.update((ROOT/'scripts/research/pallet_line_pose_v1').glob('*.py'))
    paths.update((ROOT/'scripts/research/pallet_sensors_submission_v1').glob('*.py'))
    paths.update((OLD/'figures').glob('*'))
    paths.add(OLD/'FIGURE_MANIFEST.json')
    for p in pred:paths.add(ROOT/p['checkpoint']['path'])
    for s in (1,2,3):paths.add(E.RAW/f'predictions/REAL_DEV/N3_DIM_SYM_seed{s}.json')
    for n in ['DIM_NORMALIZATION_LOCK.json','CALIBRATION_AND_SELECTION.json','PAPER_TRAINING_COMPLETE.json']:
        p=E.DOC/n
        if p.exists():paths.add(p)
    for n in ['TABLES.json','TABLE1_MAIN_2D.md','TABLE2_POSE.md']:
        paths.add(SCORES.parent/n)
    return sorted(paths)

def build(probe=False):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert git('branch','--show-current')=='main'
    rows,pred,dims,groups=load_data();cases,rule=select_cases(rows);examples=dimension_examples(rows,dims)
    print(json.dumps(dict(cases=[{'id':r['id'],'errors':r['errors'],'delta':r['mean_delta']} for r in cases],
                          dimensions=[{'id':r['id'],'image':r['image'],'score':r['readability_score']} for r in examples],rule=rule),indent=2),flush=True)
    if probe:
        f,axes=plt.subplots(2,3,figsize=(15,8))
        for ax,r in zip(axes.flat,examples+cases):ax.imshow(rgb(r['image']));ax.set_title(r['id'],fontsize=8);ax.axis('off')
        f.tight_layout();f.savefig('/tmp/pallet_presentation_v2_probe.png',dpi=130);return
    paths=protected_sources(pred)
    paths+=sorted({ROOT/r[k] for r in rows+examples for k in ('image','annotation')})
    before=[binding(p) for p in sorted(set(paths))]
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'ASSET_MANIFEST.json').exists()
    for n in NAMES:
        assert not (OUT/f'{n}.png').exists() and not (OUT/f'{n}.pdf').exists()
    pipeline();dim_receipts=dimensions_figure(examples);res_receipts=residual_figure(cases,rule)
    symmetry_figure(groups);cand=candidate_figure(pred)
    for b in before:assert sha(ROOT/b['path'])==b['sha256'],b['path']
    common=dict(model='N3_DIM_SYM; frozen R0 YOLO26n-Pose',checkpoints=[p['checkpoint'] for p in pred],
                baseline_checkpoint=binding(E.R0),prediction_sources=[binding(E.RAW/f'predictions/REAL_DEV/N3_DIM_SYM_seed{s}.json') for s in (1,2,3)],
                baseline_predictions=binding(BASE),dimensions_source=binding(REGISTRY),contract_source=binding(CONTRACT),
                metric_source=binding(SCORES),generated_script=str(Path(__file__).relative_to(ROOT)))
    figures={}
    for n in NAMES:
        figures[n]=dict(**common,files=[binding(OUT/f'{n}.{ext}') for ext in ('png','pdf')],
                       source_image_paths=[],GT_sources=[],seed=None,selection_rule='implementation/contract diagram; no performance selection',
                       limitations=['No new training, inference, or model selection.'])
    figures[NAMES[1]].update(examples=dim_receipts,source_image_paths=[r['image'] for r in examples],GT_sources=[r['annotation'] for r in examples],
        selection_rule='Largest in-frame top-face fraction with 15px margins, >=12px height; camera-facing dimensions already match canonical axes; lexical tie.',
        limitations=['Internal RGB only; external redistribution rights UNVERIFIED.',
                     'GT edge anchors are illustration only; canonical input dimensions come solely from registry.',
                     'Square illustrates metadata/contract, not N3 DEV319 performance; some saved anchors are PnP-derived.'])
    figures[NAMES[2]].update(examples=res_receipts,source_image_paths=[r['image'] for r in cases],GT_sources=[r['annotation'] for r in cases],
        seed=1,selection_rule=rule,limitations=['Internal RGB only; external redistribution rights UNVERIFIED.',
        'DEV319 reused development data, not an independent held-out confirmation.',
        'Three-seed selection and fixed seed1 overlay; no best-seed selection.',
        'Small local corrections do not demonstrate large-corner recovery. GT may include geometry-derived reference points.',
        'Native keypoint indices are plotted unchanged; error uses one valid whole-object GT permutation per prediction.'])
    figures[NAMES[3]].update(contracts={t:groups[t] for t in [TYPES[0],TYPES[2]]},
        limitations=['C2 is the paper-declared benchmark convention, not verified physical/visual symmetry.',
                     'C4 contract illustration does not establish C4 coverage in N3 training or DEV319.'])
    figures[NAMES[4]].update(candidate_details=cand,seed=1,limitations=['Lattice exact; illustrated probability distribution and output vectors are schematic.'])
    manifest=dict(schema='presentation_assets_v2',source_HEAD=git('rev-parse','HEAD'),source_origin_main=git('rev-parse','origin/main'),
        branch='main',no_training=True,optimizer_updates=0,new_inference=0,model_seed_hyperparameter_changes=False,
        source_unchanged_before_after=True,protected_sources=before,figures=figures,
        metric_recomputation=dict(frames=319,arms=4,values=1276,all_exact_to_rescored_2d=True,coordinate_preservation_all_seeds=True),
        dimensions_exact=dims,old_figure_audit=dict(pipeline='P-only; dims feed PnP, not refiner; replaced in NEW directory',
          graphical_abstract='P-only; does not describe N3 DIM+SYM',candidate_sampling='13x17+null lattice consistent; new dimension-aware decoder diagram',
          qualitative_coordinates='P1/other historical arms, not N3; not reused as N3 evidence'),
        rights='Repository research RGB used for internal professor report only. External publication redistribution permission UNVERIFIED.',
        code=[binding(HERE/'build_assets.py'),binding(HERE/'verify_assets.py')])
    with (OUT/'ASSET_MANIFEST.json').open('x') as stream:json.dump(manifest,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print('BUILD_COMPLETE; optimizer_updates=0; protected sources unchanged',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');build(parser.parse_args().probe)
