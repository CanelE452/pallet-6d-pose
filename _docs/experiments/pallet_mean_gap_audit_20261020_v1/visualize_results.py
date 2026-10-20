"""Saved-data plots/panels, with existing reviewed privacy pixels only.

No pose solves or inference. Native image coordinates, same registered
physical cuboid, reconstructed GRef explicitly distinguished from observations.
"""
import argparse,csv,json,math,html
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import read,rows,index,sha,write,csvwrite,PF

EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
COLORS={'D1':'#f04b4b','PF':'#a67bff','YOLO':'#459bf5','GRef':'#56d277','B':'#ffa72b','I':'#1ce0dc','partial':'#ff74d4','S':'#d4d4d4'}

def readcsv(p): return list(csv.DictReader(Path(p).open()))
def cuboid(d):
    x,y,z=np.array(d)/2
    return np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])
def project(pose,frame,dims):
    x=cuboid(dims)@np.array(pose['R_physical']).T+np.array(pose['centroid'])
    q=x@np.array(frame['K']).T
    if np.any(x[:,2]<=0): raise ValueError('Display projection behind camera')
    return q[:,:2]/q[:,2:]
def wire(draw,q,color,dashed=False):
    for a,b in EDGES:
        p1,p2=q[a],q[b]
        if dashed:
            n=max(1,int(np.linalg.norm(p2-p1)/7))
            for j in range(0,n,2):
                x=p1+(p2-p1)*(j/n);y=p1+(p2-p1)*(min(j+1,n)/n)
                draw.line([tuple(x),tuple(y)],fill=color,width=2)
        else:draw.line([tuple(p1),tuple(p2)],fill=color,width=3)

def run(data,analysis,out):
    out.mkdir(exist_ok=False,parents=True);(out/'cases').mkdir()
    pair=readcsv(analysis/'image_paired.csv'); seed=[x for x in pair if x['seed']=='20261018'];pf=[x for x in seed if x['comparator']==PF[0]]
    B=data/'pallet-boundary-action-repair-20261018-v1'
    F=data/'pallet-bi-full-execution-20261019-v1'
    refpath=data/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2/FIXED_REFERENCES.json'
    refs=read(refpath)['references']; source=index(rows(F/'P0_BASELINE_REAGGREGATION/SCORED_ROWS.jsonl.gz'))
    ups=rows(B/'d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz');up=index(ups);pos={f['id']:i for i,f in enumerate(ups)}
    privacy=read(B/'DISPLAY_PRIVACY_PLAN.json');rights=read(B/'RIGHTS_SCOPE_BINDING.json');receipt=read(B/'RIGHTS_RECEIPT.json')
    assert receipt['all_192_rgb_publication_rights_confirmed'] is True
    assert sha(B/'RIGHTS_RECEIPT.json')==rights['rights_receipt_sha256']
    assert sha(B/'DISPLAY_PRIVACY_PLAN.json')==rights['display_privacy_plan_sha256'] and privacy['measurement_or_inference_changes'] is False
    native=read(B/'privacy_redacted_native_v2/MANIFEST.json');review=read(B/'PRIVACY_NATIVE_V2_INDEPENDENT_MODEL_REVIEW_L.json')
    assert sha(B/'privacy_redacted_native_v2/MANIFEST.json')==review['manifest_sha256'] and review['actual_pixel_review_performed'] is True
    assert native['source_plan_sha256']==rights['display_privacy_plan_sha256']
    nativeidx={x['id']:x for x in native['frames']}
    reasons={}
    def choose(ident,why):reasons.setdefault(ident,[]).append(why)
    for comp in PF:
        ss=[x for x in seed if x['comparator']==comp]
        for m in ('T_cm','R_deg'):
            for rank,r in enumerate(sorted(ss,key=lambda x:(-float(x['delta_'+m]),x['id']))[:10],1):choose(r['id'],f'{comp}:top{rank}_delta_{m}')
    for rec in sorted({x['recording'] for x in pf}):
        ss=[x for x in pf if x['recording']==rec and x['transition']=='BOTH_SUCCESS']
        if ss:choose(min(ss,key=lambda x:x['id'])['id'],'recording_first_ID_success_control_'+rec)
        else:choose(min([x for x in pf if x['recording']==rec],key=lambda x:x['id'])['id'],'recording_without_joint_success_'+rec)
    for trans in ('RECOVER','DAMAGE'):
        ss=[x for x in pf if x['transition']==trans]
        choose(min(ss,key=lambda x:float(x['delta_T_cm']))['id'],'transition_min_delta_T_'+trans)
    selection=[dict(id=k,reasons=';'.join(v)) for k,v in reasons.items()]
    csvwrite(out/'CASE_SELECTION.csv',selection) # frozen before opening display RGB
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    manifests=[];ann=[];casehtml=[]
    for count,item in enumerate(selection,1):
        ident=item['id'];u=up[ident];r=source[ident];assert u['partition']!='CAL'
        plan=privacy['frames'][ident];assert plan['original_image_sha256']==u['image_sha256'] and plan['raw_hw']==u['raw_hw']
        if plan['redactions']:
            n=nativeidx[ident];p=B/'privacy_redacted_native_v2'/n['file'];expected=n['sha256'];mode='EXISTING_REVIEWED_REDACTED_NATIVE'
            assert n['redactions']==len(plan['redactions']) and n['raw_hw']==u['raw_hw']
        else:
            assert ident not in nativeidx;p=Path(u['image']);expected=u['image_sha256'];mode='EXISTING_APPROVED_EMPTY_REDACTION_SOURCE'
        assert sha(p)==expected
        with Image.open(p) as im:base=im.convert('RGB').copy()
        w,h=base.size;assert [h,w]==u['raw_hw']
        ref=refs[u['partition']][ident];g=dict(R_physical=ref['R'],centroid=ref['t'])
        dims=u['known_physical_dimensions_xyz_m'];qref=project(g,u,dims)
        A='D1_H1_MLP_s20261018';methodnames=[A,*PF]
        qposes={a:project(r['poses'][a],u,dims) for a in methodnames}
        # Compare the saved panel labels to the published per-image CSV before drawing.
        for comp in PF:
            rr=next(x for x in seed if x['id']==ident and x['comparator']==comp)
            for a,prefix in [(A,'method'),(comp,'comparator')]:
                assert float(rr[prefix+'_T_cm'])==r['metrics'][a]['translation_cm']
                assert float(rr[prefix+'_R_deg'])==r['metrics'][a]['rotation_deg']
        obsfile=B/f'd1_real_observations_v1/frames/{pos[ident]:04d}/OBSERVATION.jsonl.gz';obs=rows(obsfile)[0]
        assert obs['id']==ident and obs['image_sha256']==u['image_sha256']
        panels=[];titles=['Original RGB (approved privacy display)','D1 seed18: red; GRef: green dashed','PF-anchor: purple; GRef: green dashed','YOLO-anchor: blue; GRef: green dashed','Reconstructed GRef model projection','Saved image observations (not physical GT)']
        for k in range(6):
            im=base.copy();dr=ImageDraw.Draw(im)
            if k in (1,2,3):
                a=methodnames[k-1];wire(dr,qref,COLORS['GRef'],True);wire(dr,qposes[a],COLORS[['D1','PF','YOLO'][k-1]])
            elif k==4:wire(dr,qref,COLORS['GRef'],True)
            elif k==5:
                partial=obs['partial_boundary_observation']['boundary_observations'];revoked={x['native_corner'] for x in partial}
                for m in obs['frozen_BI_fit']:
                    if m['native_corner'] in revoked or m['state']=='S':continue
                    x,y=m['xy'];c=COLORS.get(m['state'],'white');dr.ellipse([x-4,y-4,x+4,y+4],outline=c,width=2);dr.text((x+5,y-10),str(m['native_corner']),fill=c,font=font)
                for m in partial:
                    for ln in m['lines']:
                        n=np.array(ln['normal_unit_xy']);t=np.array([-n[1],n[0]]);o=-ln['offset_px']*n
                        limits=ln.get('provenance',{}).get('actual_observed_inlier_tangent_interval_px')
                        if limits:dr.line([tuple(o+limits[0]*t),tuple(o+limits[1]*t)],fill=COLORS['partial'],width=3)
            tile=Image.new('RGB',(w,h+60),'#111827');tile.paste(im,(0,60));dd=ImageDraw.Draw(tile);dd.text((8,5),titles[k],fill='white',font=font)
            if k in (1,2,3):
                met=r['metrics'][methodnames[k-1]];label=f"T={met['translation_cm']:.3f} cm | R={met['rotation_deg']:.3f} deg | success={met['success']}"
                dd.text((8,29),label,fill='white',font=font)
                ann.append(dict(id=ident,panel=k,arm=methodnames[k-1],T_cm=met['translation_cm'],R_deg=met['rotation_deg'],success=met['success']))
            else:dd.text((8,29),'Raw pixels: origin top-left; x right, y down; no crop/warp',fill='#cbd5e1',font=font)
            panels.append(tile)
        canvas=Image.new('RGB',(w*3,(h+60)*2+52),'#111827')
        for k,im in enumerate(panels):canvas.paste(im,((k%3)*w,(k//3)*(h+60)+52))
        dd=ImageDraw.Draw(canvas);dd.text((8,6),f'{ident} | {u["cluster_id"]} | saved outputs; fixed geometric reference is NOT independent physical GT',fill='white',font=font)
        dd.text((8,29),'Observation colors: orange B, cyan I, magenta actual line inliers. Revoked2D/S omitted. Physical owner UNKNOWN.',fill='#cbd5e1',font=font)
        name=f'case_{count:02d}.jpg';canvas.save(out/'cases'/name,quality=88)
        manifests.append(dict(id=ident,file='cases/'+name,sha256=sha(out/'cases'/name),privacy_source_sha256=expected,original_image_sha256=u['image_sha256'],privacy_mode=mode,existing_redactions=len(plan['redactions']),observation_sha256=sha(obsfile),raw_hw=u['raw_hw'],selection_reason=item['reasons'],native_pixel_scale=1,whole_figure_width=w*3,whole_figure_height=(h+60)*2+52))
        casehtml.append(f'<h2>{html.escape(ident)}</h2><p>{html.escape(item["reasons"])}</p><a href="cases/{name}"><img loading="lazy" src="cases/{name}" style="width:100%"></a>')
    csvwrite(out/'PANEL_VALUES.csv',ann)
    write(out/'PANEL_MANIFEST.json',dict(cases=manifests,reference_sha256=sha(refpath),rights_scope_sha256=sha(B/'RIGHTS_SCOPE_BINDING.json'),rights_receipt_sha256=sha(B/'RIGHTS_RECEIPT.json'),privacy_plan_sha256=sha(B/'DISPLAY_PRIVACY_PLAN.json'),new_inference_calls=0,new_fits=0,physical_owner_verified=False))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mean gap audit panels</title><style>body{max-width:1600px;margin:24px auto;font:16px sans-serif;background:#111827;color:#eee}a{color:#a5d7ff}</style><h1>PRIMARY192 mean gap audit: representative panels</h1><p>All scalar comparisons use all192 frames. Cases are diagnostic only. Green dashed GRef is a reconstructed geometric reference, never independent physical ground truth. No reference is an inference input. Original RGB is shown only within reviewed result panels; existing privacy redactions preserved.</p>'+''.join(casehtml))
    plt.rcParams.update({'font.size':10,'figure.dpi':130})
    # All192 points and axes retained, no tail clipping.
    fig,ax=plt.subplots(1,2,figsize=(12,4.4))
    for q,key,unit in [(0,'T_cm','cm'),(1,'R_deg','deg')]:
        x=np.array([float(r['comparator_'+key]) for r in pf]);y=np.array([float(r['method_'+key]) for r in pf]);lim=max(x.max(),y.max())*1.03
        ax[q].scatter(x,y,s=15,alpha=.7);ax[q].plot([0,lim],[0,lim],color='gray',ls='--');ax[q].set(xlabel=f'PF-anchor error ({unit})',ylabel=f'D1 seed18 error ({unit})',xlim=(0,lim),ylim=(0,lim),title=f'All192 paired {key}: above line is worse')
    fig.tight_layout();fig.savefig(out/'all192_scatter.png');plt.close(fig)
    conc=readcsv(analysis/'gap_concentration.csv');cc=[r for r in conc if r['seed']=='20261018' and r['comparator']==PF[0] and r['top_k']=='5']
    fig,ax=plt.subplots(1,2,figsize=(12,4.2))
    for j,tag in enumerate(('T','R')):
        r=next(x for x in cc if x['metric']==tag);vals=[float(r['top_positive_sum']),float(r['positive_sum'])-float(r['top_positive_sum']),float(r['negative_sum']),float(r['net_sum'])]
        ax[j].bar(['Top5 positive','Other positive','All negative','Net'],vals,color=['#eb6557','#e8b384','#4c9e8e','#5875aa']);ax[j].axhline(0,color='gray');ax[j].set(title=f'{tag}: signed sum over192; D1 minus PF-anchor',ylabel='cm sum' if tag=='T' else 'degree sum')
    fig.tight_layout();fig.savefig(out/'signed_gap_budget.png');plt.close(fig)
    rec=[r for r in readcsv(analysis/'recording_summary.csv') if r['seed']=='20261018' and r['comparator']==PF[0]]
    fig,ax=plt.subplots(1,2,figsize=(12,4.3))
    for j,key in enumerate(('delta_T_sum','delta_R_sum')):
        v=[float(x[key]) for x in rec];ax[j].bar([f'{x["group"]}\nn={x["n"]}' for x in rec],v,color=['#e76c61' if t>0 else '#4c9e8e' for t in v]);ax[j].axhline(0,color='gray');ax[j].set(title='D1 seed18 minus PF-anchor by recording',ylabel='Translation error sum (cm)' if j==0 else 'Rotation error sum (degrees)');ax[j].tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(out/'recording_contributions.png');plt.close(fig)
    print(json.dumps(dict(panels=len(manifests),panel_values=len(ann),selection_ids=list(reasons)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.data_root,a.analysis,a.out)
