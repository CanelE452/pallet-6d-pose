"""Postseal actual-pixel gallery; displayed projections are never measurements."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_risk_information_v1.common import read, rows, sha, write
from scripts.research.pallet_boundary_action_repair_v1.display_privacy import display_image
from .bank import read as bank_read
from .registry import LEARNED

C='OBS_COMMON_s20261018'; B='OBS_BI_s20261018'
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
COLORS={C:'#e28a00',B:'#087cca','N0_HISTORY':'#555555','E0_HISTORY':'#8e44ad',
        'POSEFIX_PF_ANCHOR_HISTORY':'#11876c','POSEFIX_YOLO_ANCHOR_HISTORY':'#9a6515'}


def token(fid): return hashlib.sha256(fid.encode()).hexdigest()[:16]


def projection(pose,frame):
    if not pose.get('available'): return None,None
    if pose.get('R_cf') is not None and pose.get('cf_extents') is not None:
        from scripts.research.pallet_minimal_bi_ablation_v1 import runtime as RT
        # Native k follows the actual selected WD family, not the unordered
        # physical vertex set of the initial family.
        body=RT.C.cuboid(*pose['cf_extents'])
        camera=body@np.asarray(pose['R_cf']).T+np.asarray(pose['centroid'])
    else:
        p0=frame['initial_pose']; Q=np.asarray(p0['R_cf']).T@np.asarray(p0['R_physical'])
        body=np.asarray(frame['native_xyz'])@Q
        camera=body@np.asarray(pose['R_physical']).T+np.asarray(pose['centroid'])
    v=camera@np.asarray(frame['K']).T
    return v[:,:2]/v[:,2:],camera[:,2]


def background(ax,image):
    ax.imshow(image);ax.set_xlim(-.5,image.width-.5);ax.set_ylim(image.height-.5,-.5)
    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])


def wire(ax,pose,frame,color):
    uv,z=projection(pose,frame)
    if uv is None:return
    for a,b in EDGES:
        if np.isfinite(uv[[a,b]]).all() and min(z[a],z[b])>0:
            ax.plot(uv[[a,b],0],uv[[a,b],1],color=color,lw=1.2)


def partial_marks(ax,partial):
    for line in partial['lines']:
        n=np.asarray(line['normal_unit_xy']);t=np.array([-n[1],n[0]]);d=line['offset_px']
        interval=line.get('provenance',{}).get('actual_observed_inlier_tangent_interval_px')
        if interval is None: continue
        points=-d*n+np.asarray(interval)[:,None]*t
        ax.plot(points[:,0],points[:,1],c='#e83e8c',lw=2,ls='--')


def observed(ax,bank,choices,detail=True):
    cc={c['native_corner']:c for c in choices}
    for corner in bank['corners']:
        k=corner['native_corner']
        if k not in cc: continue
        q=np.asarray(corner['q0']); choice=cc[k]
        if detail:
            pts=np.asarray([c['applied_xy'] for c in corner['candidates']]);ax.scatter(pts[:,0],pts[:,1],s=22,c='#bbbbbb',alpha=.7)
        ax.scatter(*q,marker='x',c='#555555',s=24)
        if choice['partial']:
            partial_marks(ax,choice['typed_partial']);suffix='partial normal'
        else:
            p=choice['applied_xy'];ax.scatter(*p,c='#00bfc4',marker='o',s=30,facecolors='none')
            suffix=f"{choice['automatic_state']} {choice['selected_kind']}"
            ax.plot([q[0],p[0]],[q[1],p[1]],c='#00bfc4',lw=1)
        ax.annotate(str(k)+':'+suffix,q,xytext=(2,-10),textcoords='offset points',fontsize=6,color='#00bfc4',bbox=dict(facecolor='white',alpha=.65,pad=.2,edgecolor='none'))


def metric(m):
    return 'NO POSE' if not m['pose_available'] else f"{m['translation_cm']:.2f} cm / {m['rotation_deg']:.2f} deg / "+('success' if m['success'] else 'failure')


def selected_cases(scored):
    groups={k:[] for k in ('RECOVERY','DAMAGE','BOTH_SUCCESS','BOTH_FAILURE')}
    for row in scored:
        a,b=row['metrics'][C]['success'],row['metrics'][B]['success']
        key='BOTH_SUCCESS' if a and b else 'RECOVERY' if b else 'DAMAGE' if a else 'BOTH_FAILURE'
        groups[key].append(row['id'])
    cases=[]
    for category,ids in groups.items():
        ids=sorted(ids)
        cases.append(dict(category=category,population_count=len(ids),selected_ids=ids[:2]))
    for axis in ('translation_cm','rotation_deg'):
        valid=[r for r in scored if r['metrics'][B][axis] is not None and r['metrics'][C][axis] is not None]
        worst=sorted(valid,key=lambda r:(-r['metrics'][B][axis],r['id']))
        damage=sorted(valid,key=lambda r:(-(r['metrics'][B][axis]-r['metrics'][C][axis]),r['id']))
        cases.append(dict(category='maximum_BI_'+axis,population_count=len(valid),selected_ids=[worst[0]['id']] if worst else []))
        cases.append(dict(category='maximum_positive_BI_minus_COMMON_'+axis,population_count=sum(r['metrics'][B][axis]>r['metrics'][C][axis] for r in valid),
            selected_ids=[damage[0]['id']] if damage and damage[0]['metrics'][B][axis]>damage[0]['metrics'][C][axis] else []))
    return cases


def slim_choice(choice):
    out={k:choice[k] for k in ('native_corner','automatic_state','selected_candidate_id','selected_candidate_index','selected_kind','selected_score','scorer_condition','q0','raw_intersection_xy','blend_alpha','applied_xy','covariance_px2','source_line_ids','physical_source_ids','partial')}
    if choice['partial']:
        out['partial_normals']=[{k:line[k] for k in ('normal_unit_xy','offset_px','normal_variance_px2','physical_source_ids')} for line in choice['typed_partial']['lines']]
    return out


def render(row,frame,bank,pred,image,path):
    fig,axes=plt.subplots(2,4,figsize=(18,10),squeeze=False)
    fig.suptitle(row['id']+' / fixed seed20261018 / whole-pose errors (cm, deg)',fontsize=11)
    for ax in axes.flat:background(ax,image)
    raw=np.asarray(frame['initial_points'])[:8]
    for k,state in enumerate(bank['states']):
        if state in ('B','I'):
            axes.flat[0].scatter(*raw[k],marker='x',c='#00bfc4');axes.flat[0].annotate(str(k)+':'+state,raw[k],fontsize=7)
    axes.flat[0].set_title('1. N0 raw visible queries / automatic states',fontsize=9)
    for ax,arm in ((axes.flat[1],C),(axes.flat[2],B)):
        observed(ax,bank,pred[arm]['choices']);ax.set_title('2. '+arm+'\nall candidates gray / selected full cyan / partial magenta',fontsize=8)
    for ax,arm in zip(list(axes.flat)[3:],(C,B,'E0_HISTORY','POSEFIX_PF_ANCHOR_HISTORY','POSEFIX_YOLO_ANCHOR_HISTORY')):
        wire(ax,row['poses'][arm],frame,COLORS[arm]);ax.set_title(arm+'\n'+metric(row['metrics'][arm]),fontsize=8,pad=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.95,pad=2))
    fig.text(.5,.015,'S/O/T/U receive no direct observation. Solid cuboids are pose projections. Partial lines constrain normal only. Full raw canvas; display privacy edits only.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.015,right=.995,bottom=.065,top=.90,hspace=.35,wspace=.07)
    fig.savefig(path,dpi=110);plt.close(fig)


def crop_panel(frame,bank,pred,image,k,path):
    fig,axes=plt.subplots(1,2,figsize=(12,5));q=np.asarray(bank['corners'][k]['q0']);radius=60
    for ax,arm in zip(axes,(C,B)):
        background(ax,image);observed(ax,dict(bank,corners=[bank['corners'][k]]),pred[arm]['choices'])
        uv,z=projection(pred[arm]['actual_pose'],frame)
        if uv is not None and z[k]>0:ax.scatter(*uv[k],s=60,marker='+',c='#b026b8')
        ax.set_xlim(max(-.5,q[0]-radius),min(image.width-.5,q[0]+radius));ax.set_ylim(min(image.height-.5,q[1]+radius),max(-.5,q[1]-radius))
        ax.set_title(arm+' / auto '+bank['states'][k]+f' corner{k}\ngray all candidates, cyan actual point, + model projection',fontsize=9)
    fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def run(run_dir,upstream,bank_root,privacy_path,output):
    start=time.monotonic(); run_dir,output=Path(run_dir),Path(output)
    scored=[r for r in rows(run_dir/'evaluation/SCORED_ROWS.jsonl.gz') if r['partition']!='CAL']
    ledger={r['id']:r for r in rows(run_dir/'PREDICTIONS.jsonl.gz')}; source=list(rows(upstream));byid={r['id']:(i,r) for i,r in enumerate(source)}
    if len(scored)!=192:raise ValueError('Whole PRIMARY192 required')
    privacy=read(privacy_path);cases=selected_cases(scored);chosen={fid for c in cases for fid in c['selected_ids']}
    output.mkdir(parents=True,exist_ok=False)
    for name in ('thumbs','details','rows','crops'):(output/name).mkdir()
    crops=[];frame_records=[];cards=[]
    try:font=ImageFont.truetype('DejaVuSans.ttf',12)
    except OSError:font=ImageFont.load_default()
    contact=Image.new('RGB',(4*320,6*270),'white');sheets=[]
    for position,row in enumerate(scored):
        fid=row['id'];ordinal,frame=byid[fid];bank=bank_read(Path(bank_root)/'frames/real'/f'{ordinal:04d}'/'BANK.json.gz')
        pred=ledger[fid]['predictions'];image=display_image(frame,privacy);name=token(fid)
        canvas=image.copy();draw=ImageDraw.Draw(canvas)
        for arm in (C,B):
            uv,z=projection(row['poses'][arm],frame)
            if uv is not None:
                for a,b in EDGES:
                    if np.isfinite(uv[[a,b]]).all() and min(z[a],z[b])>0:
                        draw.line([tuple(uv[a]),tuple(uv[b])],fill=COLORS[arm],width=2)
        # Header makes each thumbnail a self-describing derivative.
        draw.rectangle((0,0,min(image.width,620),42),fill='white')
        draw.text((4,2),'COMMON orange '+metric(row['metrics'][C]),fill=COLORS[C],font=font)
        draw.text((4,20),'BI blue '+metric(row['metrics'][B]),fill=COLORS[B],font=font)
        canvas.thumbnail((480,360));canvas.save(output/'thumbs'/f'{name}.jpg',quality=78)
        if fid in chosen:render(row,frame,bank,pred,image,output/'details'/f'{name}.png')
        for state in ('B','I'):
            if not any(c['automatic_state']==state for c in crops):
                eligible=[c for c in bank['corners'] if bank['states'][c['native_corner']]==state and len(c['candidates'])>1]
                # Source order is the frozen whole192 order; choose lexicographic ID in second pass below.
                if eligible:crops.append(dict(automatic_state=state,id=fid,native_corner=eligible[0]['native_corner'],deferred=True))
        public_row=dict(id=fid,partition=row['partition'],states=bank['states'],
            predictions={arm:dict(actual_pose=row['poses'][arm],metrics=row['metrics'][arm],status=row['status'][arm],
                choices=[slim_choice(c) for c in pred[arm]['choices']] if arm in LEARNED else None) for arm in row['poses']})
        write(output/'rows'/f'{name}.json',public_row)
        detail=f'<a href="details/{name}.png">parallel process</a>' if fid in chosen else ''
        cards.append(f'<article data-id="{html.escape(fid)}"><a href="rows/{name}.json"><h3>{position+1}. {html.escape(fid)}</h3></a><a href="thumbs/{name}.jpg"><img loading="lazy" src="thumbs/{name}.jpg"></a><p>COMMON {html.escape(metric(row["metrics"][C]))}<br>BI {html.escape(metric(row["metrics"][B]))}</p>{detail} · <a href="rows/{name}.json">exact selected observations + poses</a></article>')
        j=position%24;x=(j%4)*320;y=(j//4)*270;im=canvas.copy();im.thumbnail((320,240));contact.paste(im,(x,y+25));ImageDraw.Draw(contact).text((x+2,y+2),str(position+1)+' '+fid,fill='#111111',font=font)
        if j==23 or position==191:
            sheet=f'contact_{position//24:02d}.jpg';contact.save(output/sheet,quality=84);sheets.append(sheet);contact=Image.new('RGB',(1280,1620),'white')
        frame_records.append(dict(id=fid,ordinal=ordinal,token=name,original_RGB_sha256=frame['image_sha256'],original_hw=frame['raw_hw'],display_only_redactions=privacy['frames'][fid]['redactions'],thumbnail_sha256=sha(output/'thumbs'/f'{name}.jpg'),row_sha256=sha(output/'rows'/f'{name}.json'),detail=f'{name}.png' if fid in chosen else None))
    # Exact predeclared lexicographic B and I candidate example selection.
    crops=[]
    for state in ('B','I'):
        found=None
        for row in sorted(scored,key=lambda r:r['id']):
            ordinal,frame=byid[row['id']];bank=bank_read(Path(bank_root)/'frames/real'/f'{ordinal:04d}'/'BANK.json.gz')
            eligible=[c for c in bank['corners'] if bank['states'][c['native_corner']]==state and len(c['candidates'])>1]
            if eligible:
                k=eligible[0]['native_corner'];path=output/'crops'/f'{state}.png'
                crop_panel(frame,bank,ledger[row['id']]['predictions'],display_image(frame,privacy),k,path)
                found=dict(automatic_state=state,id=row['id'],native_corner=k,path=f'crops/{state}.png',sha256=sha(path));break
        crops.append(found or dict(automatic_state=state,status='NO_ELIGIBLE_FULL_ALTERNATIVE'))
    top='<h1>COMMON vs BI / fixed seed20261018 / PRIMARY192</h1><p>Whole-pose errors: center cm / C2 rotation deg. Orange COMMON, blue BI. Automatic states are predictions. Solid wireframes are model projections; selected cyan points and magenta normal lines are image observations. S/O/T/U are excluded. Numeric links include all13 systems and all6 learned outputs.</p><p><a href="../REPORT_KO.md">Korean report</a> · <a href="../COMPARISON_ALL_ARMS.csv">all populations table</a></p><input id="query" placeholder="Filter image ID" oninput="for(const a of document.querySelectorAll(\'article\')) a.hidden=!a.dataset.id.includes(this.value)">'
    body='<!doctype html><meta charset="utf-8"><title>BI local observation gallery</title><style>body{font:15px sans-serif;margin:24px;background:#eef3f5;color:#20313d}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}article{background:white;padding:12px;border-radius:8px}img{width:100%;height:auto}h3{font-size:14px}input{padding:10px;width:300px;margin:20px}</style>'+top+'<main>'+''.join(cards)+'</main>'
    (output/'index.html').write_text(body)
    result=dict(schema='BI_ACTUAL_PIXEL_GALLERY_V1',rows=192,source_sha256=sha(__file__),privacy_plan_sha256=sha(privacy_path),
        cases=cases,crops=crops,frames=frame_records,contact_sheets=sheets,raw_RGB_uploaded=False,display_redactions_only=True,
        pixel_decode_status='GENERATED_NOT_YET_ROOT_PIXEL_REVIEW',wall_seconds=time.monotonic()-start)
    write(output/'GALLERY_MANIFEST.json',result);return result


def main():
    p=argparse.ArgumentParser()
    for key in ('run-dir','upstream','bank-root','privacy-path','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();run(a.run_dir,a.upstream,a.bank_root,a.privacy_path,a.output)


if __name__=='__main__':main()
