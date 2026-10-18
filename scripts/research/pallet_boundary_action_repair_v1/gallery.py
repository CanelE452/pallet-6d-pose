"""Actual-coordinate scientific gallery; RGB is public only with confirmed rights.

Public geometry views retain the original image canvas, all numerical errors
and missing-RGB reasons. A parallel private RGB gallery uses exact original
pixels for model review. No render is fed back into inference or measurement.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import resource
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import read, rows, sha, write
from .registry import GALLERY_REPRESENTATIVES, ALL_ARMS
from .figures import label
from .display_privacy import display_image

EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
ARMS=('N0','E0','POSEFIX_SUBPIX_PF_ANCHOR','POSEFIX_SUBPIX_YOLO_ANCHOR')+GALLERY_REPRESENTATIVES
COLORS=('#25904b','#247db4','#c88800','#948400','#e36d20','#ac531b','#c53254','#9f40a1')
REGRESSIONS=(('wood_day_01:005249','LEGACY_B_FULL'),('plastic_day_01:012545','LEGACY_EDGE'),('wood_184309:000432','LEGACY_AB_FULL'))
MISSING_RGB='RGB_NOT_PUBLISHED_RIGHTS_UNCONFIRMED; geometry uses exact original image coordinates'


def project(pose,frame,family='ORIGINAL_WD'):
    if not pose or not pose.get('available'):return None,None
    f=frame['families'][family];xyz=np.asarray(f['native_xyz'],float);Q=np.asarray(f['physical_registration_Q'],float)
    R=np.asarray(pose['R_physical'],float)@Q.T;t=np.asarray(pose['centroid'],float)
    camera=(xyz-xyz.mean(0))@R.T+t;image=camera@np.asarray(frame['packet']['solver_input']['K']).T
    with np.errstate(invalid='ignore',divide='ignore'):uv=image[:,:2]/image[:,2:]
    return uv,camera[:,2]


def wire(ax,uv,depth,color,style='-',numbers=False,marker='o',alpha=1):
    if uv is None:return
    for a,b in EDGES:
        if np.isfinite(uv[[a,b]]).all()and min(depth[a],depth[b])>0:
            ax.plot(uv[[a,b],0],uv[[a,b],1],color=color,ls=style,lw=1.3,alpha=alpha)
    good=np.isfinite(uv).all(1)&(depth>0)
    ax.scatter(uv[good,0],uv[good,1],c=color,s=11,marker=marker,alpha=alpha)
    if numbers:
        for i in np.flatnonzero(good):ax.annotate(str(i),uv[i],xytext=(3,-6),textcoords='offset points',color=color,fontsize=7)


def background(ax,frame,image):
    h,w=frame['trace']['raw_hw']
    if image is not None:ax.imshow(image,extent=(-.5,w-.5,h-.5,-.5))
    else:
        ax.set_facecolor('#edf2f4');ax.text(.5,.93,'RGB withheld: rights unconfirmed',transform=ax.transAxes,ha='center',fontsize=7,color='#59666f')
        ax.grid(alpha=.15)
    ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])


def metric_text(metric):
    if not metric['pose_available']:return 'NO POSE / failure'
    return f"{metric['translation_cm']:.2f} cm / {metric['rotation_deg']:.2f} deg / "+('success'if metric['success']else'failure')


def pose_panel(ax,arm,row,frame,image,color,reference=None,numbers=False):
    background(ax,frame,image);uv,z=project(row['poses'][arm],frame);wire(ax,uv,z,color,numbers=numbers)
    if reference is not None:
        p=dict(available=True,R_physical=reference['R'],centroid=reference['t']);guv,gz=project(p,frame)
        wire(ax,guv,gz,'#199a77','--',marker='s',alpha=.7)
    off=0 if uv is None else sum(not(np.isfinite(p).all()and 0<=p[0]<frame['trace']['raw_hw'][1]and 0<=p[1]<frame['trace']['raw_hw'][0])for p in uv)
    ax.set_title(label(arm)+'\n'+metric_text(row['metrics'][arm])+f' | outside {off}/8',fontsize=8,
        bbox=dict(facecolor='white',edgecolor='none',pad=1.5))


def query_geometry(pose,frame,query,family):
    uv,z=project(pose,frame,family);a=uv[query['native_corner']];b=uv[query['native_neighbor']]
    center=np.asarray(query['center_xy'],float);normal=np.asarray(query['outward_normal_xy'],float);edge=b-a
    result=dict(a=a.tolist(),b=b.tolist(),center=center.tolist(),normal=normal.tolist(),length_px=float(np.linalg.norm(edge)),
        endpoint_depth_m=[float(z[query['native_corner']]),float(z[query['native_neighbor']])],lambda_value=None,d_px=None,intersection=None)
    M=np.column_stack((edge,-normal))
    if np.isfinite(M).all()and abs(np.linalg.det(M))>1e-12:
        lam,d=np.linalg.solve(M,center-a);result.update(lambda_value=float(lam),d_px=float(d),intersection=(a+lam*edge).tolist())
    return result


def boundary_panel(ax,pose,frame,bundle,query,family,image,title,*,autoscale=False):
    background(ax,frame,image)
    if query is None:ax.set_title(title+'\nNo originally supported B profile',fontsize=8,
        bbox=dict(facecolor='white',edgecolor='none',pad=1.5));return None
    g=query_geometry(pose,frame,query,family);a=np.array(g['a']);b=np.array(g['b']);c=np.array(g['center']);n=np.array(g['normal']);q=g['intersection']
    uv,z=project(pose,frame,family);wire(ax,uv,z,'#777777',alpha=.35)
    if np.isfinite([a,b]).all():ax.plot([a[0],b[0]],[a[1],b[1]],color='#d55c17',lw=3,marker='o',markersize=4)
    offsets=np.asarray(query['offset_grid_base_px'],float)*bundle['pixel_scale'];s0=c+n*offsets.min();s1=c+n*offsets.max()
    ax.plot([s0[0],s1[0]],[s0[1],s1[1]],color='#008d99',lw=2);ax.scatter(*c,color='#008d99',marker='+',s=55)
    if q is not None:
        q=np.asarray(q);points=np.array([a,b,q]);d=b-a;length=np.linalg.norm(d)
        if length>0:
            lo=min(0.,g['lambda_value']);hi=max(1.,g['lambda_value']);start=a+(lo-.12*(hi-lo))*d;end=a+(hi+.12*(hi-lo))*d
            ax.plot([start[0],end[0]],[start[1],end[1]],color='#8b59a5',ls='--',lw=1)
        ax.scatter(*q,marker='x',color='#c5213b',s=45)
        if autoscale and np.isfinite(points).all():
            h,w=frame['trace']['raw_hw'];allp=np.vstack((points,[[0,0],[w,h]]));lo=allp.min(0);hi=allp.max(0);pad=np.maximum((hi-lo)*.08,5)
            ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(hi[1]+pad[1],lo[1]-pad[1]);ax.set_xticks([]);ax.set_yticks([])
            ax.plot([0,w,w,0,0],[0,0,h,h,0],color='#222',lw=.7)
    lam='undefined'if g['lambda_value']is None else f"{g['lambda_value']:.6g}"
    ax.set_title(title+f"\nlambda={lam}; length={g['length_px']:.3g}px; edge {query['native_corner']}->{query['native_neighbor']}",fontsize=8,
        bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
    return g


def observations_panel(ax,frame,e_run,image):
    background(ax,frame,image);spec=e_run['packet']['solver_input'];uv,z=project(frame['fallback'],frame)
    wire(ax,uv,z,'#75909f',alpha=.3);states=spec['states'];initial=np.asarray(frame['trace']['initial_points'])
    palette={'B':'#04a6c7','I':'#8c57bb','S':'#b38b00','O':'#e08412','T':'#cf448f','U':'#777777'}
    for j,state in enumerate(states[:8]):
        p=uv[j]if state=='S'else initial[j]
        if np.isfinite(p).all():ax.scatter(*p,color=palette[state],marker='D'if state=='S'else'x',s=25);ax.annotate(str(j)+':'+state,p,xytext=(4,-4),textcoords='offset points',fontsize=7,color=palette[state])
    for obs in spec['visible']:
        p=obs['xy'];ax.scatter(*p,color=palette[obs['state']],marker='o',s=19,facecolors='none')
    for item in spec['boundary_observations']:
        for line in item['lines']:
            record=line.get('provenance',{}).get('line_record',{});p=np.asarray(record.get('observed_points',[]))
            if p.ndim==2 and p.shape[1]==2:ax.plot(p[:,0],p[:,1],'.',color='#e03e91',markersize=2)
    ax.set_title('Actual new D1 states and B/I observations\nx: prediction guide; circle: observed B/I; diamond S: model projection',fontsize=7,
        bbox=dict(facecolor='white',edgecolor='none',pad=1.5))


def choose_details(scored,result,predictions=None):
    primary={r['id']:r for r in scored if r['partition']!='CAL'};candidates=[];categories=[]
    for arm in GALLERY_REPRESENTATIVES:
        p=result['paired']['PRIMARY192']['E0__TO__'+arm]
        for category in ('RECOVERY','DAMAGE','BOTH_FAILURE','BOTH_SUCCESS'):
            ids=p['ids'][category]
            if category=='BOTH_FAILURE':
                ids=[fid for fid in ids if all(primary[fid]['metrics'][a][axis]is not None for a in (arm,'E0')for axis in ('translation_cm','rotation_deg'))
                    and any(primary[fid]['metrics'][arm][axis]>primary[fid]['metrics']['E0'][axis]for axis in ('translation_cm','rotation_deg'))]
                ids=sorted(ids,key=lambda fid:(-(primary[fid]['metrics'][arm]['translation_cm']-primary[fid]['metrics']['E0']['translation_cm']),
                    -(primary[fid]['metrics'][arm]['rotation_deg']-primary[fid]['metrics']['E0']['rotation_deg']),hashlib.sha256(fid.encode()).hexdigest()))
            else:
                if category=='BOTH_SUCCESS'and predictions is not None:
                    if arm.startswith('B_'):
                        ids=[fid for fid in ids if predictions[fid][arm].get('boundary_applied')is True and predictions[fid][arm].get('fallback')is False]
                    else:ids=[fid for fid in ids if bool(predictions[fid][arm].get('actions'))]
                ids=sorted(ids,key=lambda fid:hashlib.sha256(fid.encode()).hexdigest())
            item=dict(arm=arm,category=category if category!='BOTH_FAILURE'else'BOTH_FAILURE_MAX_CONTINUOUS_WORSENING',
                population_count=len(ids),raw_transition_count=len(p['ids'][category]),eligible_category_count=len(ids),selected_id=ids[0]if ids else None)
            categories.append(item)
            if ids:candidates.append(item)
    # Category-round-robin ensures all nonempty types are represented before fill.
    ordered=sorted(candidates,key=lambda r:(['RECOVERY','DAMAGE','BOTH_FAILURE_MAX_CONTINUOUS_WORSENING','BOTH_SUCCESS'].index(r['category']),list(GALLERY_REPRESENTATIVES).index(r['arm'])))
    selected=[];quota=Counter();seen=set()
    # First find one panel covering every nonempty category globally. The
    # category-first naive fill could otherwise consume all twelve slots before
    # success-preservation examples were considered.
    categories_present=[c for c in ('RECOVERY','DAMAGE','BOTH_FAILURE_MAX_CONTINUOUS_WORSENING','BOTH_SUCCESS')if any(r['category']==c for r in ordered)]
    def cover(index,chosen,used,counts):
        if index==len(categories_present):return chosen
        category=categories_present[index]
        choices=sorted((r for r in ordered if r['category']==category),key=lambda r:(hashlib.sha256(r['selected_id'].encode()).hexdigest(),r['arm']))
        for item in choices:
            duplicate=item['selected_id']in used
            if not duplicate and counts[item['arm']]>=3:continue
            nc=counts.copy();nu=used.copy()
            if not duplicate:nc[item['arm']]+=1;nu.add(item['selected_id'])
            found=cover(index+1,chosen+([]if duplicate else[item]),nu,nc)
            if found is not None:return found
        return None
    first=cover(0,[],set(),Counter())
    if first is None:raise ValueError('Fixed detail quota cannot cover nonempty categories; preserve this technical failure')
    for item in first:selected.append(item);seen.add(item['selected_id']);quota[item['arm']]+=1
    for item in ordered:
        if len(seen)>=12:break
        if item['selected_id']in seen:continue
        if quota[item['arm']]>=3:continue
        selected.append(item);seen.add(item['selected_id']);quota[item['arm']]+=1
    return selected,categories


def observed_marks(ax,spec):
    for obs in spec['visible']:
        ax.scatter(*obs['xy'],edgecolors='#00a2b8'if obs['state']=='B'else'#824cb0',facecolors='none',marker='o',s=14)
    for item in spec['boundary_observations']:
        for line in item['lines']:
            p=np.asarray(line.get('provenance',{}).get('line_record',{}).get('observed_points',[]))
            if p.ndim==2 and p.shape[1]==2:ax.plot(p[:,0],p[:,1],'.',color='#df4097',markersize=1.6)


def profile_marks(ax,pose,frame,bundle,query,family):
    if query is None:return
    g=query_geometry(pose,frame,query,family);c=np.asarray(g['center']);n=np.asarray(g['normal'])
    offsets=np.asarray(query['offset_grid_base_px'])*bundle['pixel_scale'];p0=c+n*offsets.min();p1=c+n*offsets.max()
    ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color='#00989d',lw=1.4)
    if g['intersection']is not None:ax.scatter(*g['intersection'],marker='x',color='#d4213d',s=18)


def thumbnail(row,frame,path,image,reference,bundle,e_run,predictions):
    fig,axes=plt.subplots(2,4,figsize=(13.6,5.8));fig.suptitle(row['id']+' / '+row['partition']+' / all errors are whole-pallet pose',fontsize=10)
    queries=sorted((q for g in(bundle or{}).get('groups',[])for q in g['queries']if q['supported']),key=lambda q:q['support_id'])
    query=queries[0]if queries else None
    for ax,arm,color in zip(axes.ravel(),ARMS,COLORS):
        pose_panel(ax,arm,row,frame,image,color,reference)
        if arm in ('B_FINITE_FULL','B_FINITE_LOCAL'):
            profile_marks(ax,row['poses'][arm],frame,bundle,query,predictions[arm].get('selected_family','ORIGINAL_WD'))
        elif arm.startswith('D1_'):observed_marks(ax,e_run['packet']['solver_input'])
        else:observed_marks(ax,frame['packet']['solver_input'])
    fig.text(.5,.01,'Solid/circles: model; green dashed/squares: geometric reference; hollow B/I: image observations; magenta: partial samples; cyan normal/red x: one fixed B query.',ha='center',fontsize=6)
    fig.tight_layout(rect=(0,.03,1,.96),h_pad=4.0,w_pad=1.2);fig.savefig(path,dpi=86,bbox_inches='tight',pil_kwargs=dict(quality=85,optimize=True));plt.close(fig)


def detail(row,frame,b,bundle,e_run,path,image,reference,predictions):
    fig,axes=plt.subplots(3,4,figsize=(18,11.6));fig.suptitle(row['id']+' / '+row['partition']+' / exact raw coordinates; green dashed = geometric reference',fontsize=11)
    for ax,arm,color in zip(axes.ravel()[:8],ARMS,COLORS):pose_panel(ax,arm,row,frame,image,color,reference,numbers=True)
    observations_panel(axes[2,0],frame,e_run,image)
    queries=[q for g in(bundle or{}).get('groups',[])for q in g['queries']if q['supported']]
    old=predictions['LEGACY_B_FULL'];family=old.get('selected_family','ORIGINAL_WD');chosen=None
    if queries:
        def rank(q):
            g=query_geometry(row['poses']['LEGACY_B_FULL'],frame,q,family);v=g['lambda_value'];return -1. if v is None else max(-v,v-1,0)
        chosen=max(queries,key=rank)
    before=boundary_panel(axes[2,1],row['poses']['LEGACY_B_FULL'],frame,bundle,chosen,family,image,'Historical B: finite edge vs infinite extension')
    afterentry=predictions['B_FINITE_FULL'];afterfamily=afterentry.get('selected_family','ORIGINAL_WD')
    after=boundary_panel(axes[2,2],row['poses']['B_FINITE_FULL'],frame,bundle,chosen,afterfamily,image,
        'Finite B selected / '+('fallback N0'if afterentry['fallback']else'eligible pose'))
    ax=axes[2,3];ax.axis('off');lines=['Actual L output / logits are not safety probabilities']
    for arm in ('D0_H0_MLP_s20261018','D0_H1_MLP_s20261018','D1_H0_MLP_s20261018','D1_H1_MLP_s20261018','D1_H1_ROLE_AFFINE_s20261018','D1_H1_PAIR_MLP_s20261018','D1_H1_PAIR_ROLE_AFFINE_s20261018'):
        p=predictions[arm];lines.extend([label(arm),f"  {p.get('selected_candidate_id')} / role {p.get('action_role')}",
            '  '+metric_text(row['metrics'][arm]),f"  eligible {p.get('eligible_candidates')}; score {p.get('selected_score'):.6g}"if p.get('selected_score')is not None else '  score N/A'])
    ax.text(0,1,'\n'.join(lines),transform=ax.transAxes,va='top',fontsize=6.5,wrap=True)
    fig.text(.5,.01,'B geometry: orange solid = finite edge; purple dashed = extension; cyan = fixed normal profile; red x = intersection. S has no image observation.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.96),h_pad=4.0,w_pad=1.2);fig.savefig(path,dpi=140,bbox_inches='tight');plt.close(fig)
    return dict(before=before,after=after,query_support_id=chosen['support_id']if chosen else None,
        display_transform='raw image pixels, no crop; proportional Matplotlib display scaling only',image_hw=frame['trace']['raw_hw'])


def saved_support_text(support):
    """Display archived support schemas without granting historical finite proof."""
    if support is None:
        return 'N/A'
    if isinstance(support, list):
        return f'historical support records {len(support)}; finite audit N/A'
    if isinstance(support, dict):
        if not support:
            return 'N/A'
        return str(support.get('supported_profiles','N/A'))+'/'+str(support.get('fixed_profile_count','N/A'))
    raise TypeError('Unknown archived support display schema')


def page(fid,thumb,detail_name,row,rgb_public,predictions):
    cells=''
    for a in ALL_ARMS:
        p=predictions[a]
        finite=saved_support_text(p.get('support'))
        values=[label(a),metric_text(row['metrics'][a]),row['action_sources'][a]['role'],str(p.get('selected_candidate_id')),
            str(p.get('selected_score','N/A')),str(p.get('archived_candidates','N/A')),str(p.get('eligible_candidates','N/A')),
            str('N0'in row['action_sources'][a]['aliases']),str(p.get('none_selected','N/A')),str(p.get('fallback')),finite]
        cells+='<tr data-arm="'+html.escape(a,quote=True)+'">'+''.join('<td>'+html.escape(v)+'</td>'for v in values)+'</tr>'
    content='<h1>'+html.escape(fid)+'</h1><p>Whole-pallet pose errors; reference is a fixed geometric annotation, not certified physical GT.</p>'
    if not rgb_public:content+='<p>'+MISSING_RGB+'</p>'
    content+='<img class="wide" src="'+thumb+'" alt="Parallel pose outputs"/>'
    if detail_name:content+='<a href="'+detail_name+'"><img class="wide" src="'+detail_name+'" alt="Detailed observations and B/L stages"/></a>'
    content+='<p>Scores are uncalibrated logits. N0 retention is a numerical pose alias; it differs from H0 none or B fallback. B/I and profiles in baseline/B panels are original observations; L panels use new D1 observations. Only one deterministic supported B query is shown in a thumbnail; full support counts are below. Grey privacy rectangles edit the display copy only and are unrelated to O states; all observations were measured on original pixels. Thumbnails use JPEG quality85; detailed scientific panels are PNG.</p>'
    content+='<table><tr><th>Method</th><th>Whole pose / cm, deg</th><th>Actual source</th><th>Selected ID</th><th>Score</th><th>Archived candidates</th><th>Eligible</th><th>N0 retained</th><th>H0 none</th><th>Fallback</th><th>Finite profiles</th></tr>'+cells+'</table>'
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(fid)+'</title><style>body{font:14px system-ui;margin:24px;max-width:1800px;background:#f2f5f7;color:#163042}.wide{width:100%;height:auto;background:white}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd7df;padding:6px}a{color:#1674b6}</style><a href="index.html">All 192</a>'+content+'</html>'


def regressions(scored,meta,preds,old_run,output,*,show_rgb=False,private=False,privacy=None,approved_ids=None):
    directory=output/'regression';directory.mkdir(exist_ok=True);byid={r['id']:r for r in scored};records=[];panels=[]
    for j,(fid,arm)in enumerate(REGRESSIONS):
        row=byid[fid];index,frame=meta[fid]
        path=old_run/'real_batch_all_available_v1/frames'/f'{index:04d}'/'BOUNDARY_BUNDLE.jsonl.gz'
        bundle=next(rows(path))if path.exists()else None;queries=[q for g in(bundle or{}).get('groups',[])for q in g['queries']if q['supported']]
        family=preds[fid][arm].get('selected_family','ORIGINAL_WD')
        def violation(q):
            value=query_geometry(row['poses'][arm],frame,q,family)['lambda_value']
            return -1. if value is None else max(-value,value-1.,0.)
        query=max(queries,key=violation)if queries else None
        after='EDGE_FINITE_FULL'if arm=='LEGACY_EDGE'else'B_FINITE_FULL'
        allowed=private or show_rgb and fid in (approved_ids or set())
        image=np.asarray(display_image(frame,None if private else privacy))if allowed else None
        fig,axes=plt.subplots(1,4,figsize=(20,5.3));pose_panel(axes[0],arm,row,frame,image,'#9059a4',numbers=True)
        before=boundary_panel(axes[1],row['poses'][arm],frame,bundle,query,family,image,'Historical infinite-line intersection / full extent',autoscale=True)
        next_family=preds[fid][after].get('selected_family','ORIGINAL_WD')
        later=boundary_panel(axes[3],row['poses'][after],frame,bundle,query,next_family,image,
            label(after)+' / '+('N0 fallback'if preds[fid][after]['fallback']else'eligible finite pose'),autoscale=True)
        ax=axes[2];profile=None
        if query is not None:
            grid=np.asarray(query['offset_grid_base_px']);prob=np.asarray(query['offset_probability']);ax.plot(grid,prob,color='#008d99')
            notes=[]
            for name,g,color in [('historical',before,'#9059a4'),('finite output',later,'#d55c17')]:
                value=None if g is None or g['d_px']is None else g['d_px']/bundle['pixel_scale']
                if value is not None and grid.min()<=value<=grid.max():ax.axvline(value,color=color,ls='--',label=name)
                notes.append(name+' d='+('undefined'if value is None else f'{value:.5g}')+' base px'+(' (outside sampled interval)'if value is not None and not grid.min()<=value<=grid.max()else''))
            ax.set_xlim(grid.min(),grid.max());ax.set_xlabel('Fixed sampled offset interval / base pixels');ax.set_ylabel('Normalized RGB-derived offset weight')
            ax.set_title('The same observed profile for both poses\nThese are not pose probabilities',fontsize=8);ax.grid(alpha=.2)
            ax.text(.02,.98,'\n'.join(notes),transform=ax.transAxes,va='top',fontsize=7)
            profile=dict(offset_grid_base_px=grid.tolist(),normalized_weights=prob.tolist(),outside_interval_offsets_reported_as_text=True)
        else:ax.axis('off');ax.text(.5,.5,'No originally supported profile',ha='center')
        caption='Regression case, selected before this campaign; not an unbiased performance sample. Orange finite edge / purple infinite extension / cyan normal / red intersection.'
        if arm=='LEGACY_AB_FULL':caption+=' AB is not rerun: this h=0 projection component is shown; the repaired B output is a different registered method.'
        fig.suptitle(fid+' / '+row['partition']+' / whole pose: '+metric_text(row['metrics'][arm]),fontsize=11)
        fig.text(.5,.015,caption,ha='center',fontsize=7,wrap=True);fig.tight_layout(rect=(0,.06,1,.94))
        name=f'{j:02d}_'+hashlib.sha256(fid.encode()).hexdigest()[:10]+'.png';fig.savefig(directory/name,dpi=150,bbox_inches='tight');plt.close(fig)
        records.append(dict(id=fid,partition=row['partition'],historical_arm=arm,new_arm=after,path='regression/'+name,
            panel_sha256=sha(directory/name),
            historical_metric=row['metrics'][arm],new_metric=row['metrics'][after],query_support_id=query['support_id']if query else None,
            old_geometry=before,new_geometry=later,fixed_profile=profile,RGB_display=allowed,raw_hw=frame['trace']['raw_hw'],
            display_transform='raw coordinate projection; second/third panels expand limits to include edge intersection; original frame outline retained; no pose/measurement changed',
            component='h=0; physical material owner unverified'))
        panels.append('<h2>'+html.escape(fid)+'</h2><img style="width:100%" src="'+name+'"/>')
    (directory/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Predeclared regressions</title><a href="../index.html">All192</a><h1>Three historical regression cases</h1>'+''.join(panels)+'</html>')
    write(directory/'REGRESSION_GEOMETRY.json',dict(schema='THREE_HISTORICAL_GEOMETRY_REGRESSIONS_V1',records=records,
        source_sha256=sha(__file__),new_pose_optimization=False,unbiased_performance_sample=False))
    return records


def run(run_dir,old_run,evaluation,original_score,output,*,rgb_public=False,rights_receipt=None,privacy_plan=None,private=False):
    started,cpu=time.monotonic(),time.process_time()
    run_dir=Path(run_dir);old_run=Path(old_run);evaluation=Path(evaluation);output=Path(output)
    if rgb_public and not private:
        if not rights_receipt or not read(rights_receipt).get('all_192_rgb_publication_rights_confirmed'):
            raise ValueError('Explicit rights receipt required for public RGB')
        if not privacy_plan:raise ValueError('Verified display-only privacy plan required for public RGB')
    if output.exists()and any(output.iterdir()):raise FileExistsError('Preserve existing gallery attempt; use a new empty output directory')
    output.mkdir(parents=True,exist_ok=True);(output/'thumbs').mkdir(exist_ok=True);(output/'details').mkdir(exist_ok=True)
    scored=list(rows(evaluation/'SCORED_ROWS.jsonl.gz'));result=read(evaluation/'RESULTS.json')
    refs=read(Path(original_score)/'FIXED_REFERENCES.json')['references']
    frames=list(rows(old_run/'inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz'));meta={r['id']:(i,r)for i,r in enumerate(frames)}
    preds={r['id']:r['predictions']for r in rows(run_dir/'PREDICTIONS.jsonl.gz')}
    selected,categories=choose_details(scored,result,preds);detail_ids={r['selected_id']for r in selected};primary=[r for r in scored if r['partition']!='CAL']
    privacy=read(privacy_plan)if privacy_plan else None;approved_ids={r['id']for r in primary}if rgb_public else set()
    if rgb_public and not private and set(privacy['frames'])!=approved_ids:raise ValueError('Privacy scope must cover exactly PRIMARY192')
    records=[];transforms=[];cards=[]
    for j,row in enumerate(primary):
        fid=row['id'];i,frame=meta[fid];stem=f'{j:03d}_'+hashlib.sha256(fid.encode()).hexdigest()[:10]
        image=np.asarray(display_image(frame,None if private else privacy))if private or rgb_public else None
        thumb='thumbs/'+stem+'.jpg';thumbpath=output/thumb;ref=refs[row['partition']][fid]
        bp=old_run/'real_batch_all_available_v1/frames'/f'{i:04d}'/'BOUNDARY_BUNDLE.jsonl.gz'
        bundle=next(rows(bp))if bp.exists()else None
        e_run=next(rows(run_dir/'d1_real_candidates_v1/frames'/f'{i:04d}'/'E_RUN.jsonl.gz'))
        thumbnail(row,frame,thumbpath,image,ref,bundle,e_run,preds[fid])
        detail_name=None
        if fid in detail_ids:
            detail_name='details/'+stem+'.png';b=next(rows(run_dir/'b_finite_v1/frames'/f'{i:04d}'/'RESULT.jsonl.gz'))
            geom=detail(row,frame,b,bundle,e_run,output/detail_name,image,ref,preds[fid]);transforms.append(dict(id=fid,**geom))
        filename=stem+'.html';(output/filename).write_text(page(fid,thumb,detail_name,row,private or rgb_public,preds[fid]))
        records.append(dict(id=fid,partition=row['partition'],page=filename,thumbnail=thumb,thumbnail_sha256=sha(thumbpath),
            page_sha256=sha(output/filename),detail=detail_name,detail_sha256=sha(output/detail_name)if detail_name else None,
            RGB_publication='PRIVATE_REVIEW_ONLY'if private else'RIGHTS_CONFIRMED'if rgb_public else MISSING_RGB,
            original_image_sha256=frame['image_sha256'],raw_hw=frame['trace']['raw_hw'],display_crop=None,
            thumbnail_encoding='JPEG quality85, optimized; original inference image not changed'))
        cards.append('<a class="card" href="'+filename+'"><h3>'+html.escape(fid)+'</h3><img loading="lazy" src="'+thumb+'"/><p>E0 '+metric_text(row['metrics']['E0'])+'<br/>D1 H1 '+metric_text(row['metrics']['D1_H1_MLP_s20261018'])+'</p></a>')
        if j%16==0:print('GALLERY',j+1,'/192',flush=True)
    title='Private actual RGB review'if private else'Public result gallery'
    body='<!doctype html><html lang="en"><meta charset="utf-8"><title>'+title+'</title><style>body{font:14px system-ui;background:#edf2f5;color:#17334a;margin:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:15px}.card{color:inherit;text-decoration:none;background:white;padding:12px;border-radius:8px}.card img{width:100%}h3{overflow-wrap:anywhere}p{line-height:1.5}</style><h1>'+title+' / PRIMARY192 = TEST49 + REMAINDER143</h1><p>All 192 IDs retained. CAL46 is separate. Click a card for every registered method and detailed panels where selected. Local assets only; GitHub blob does not execute HTML.</p>'
    if not(private or rgb_public):body+='<p>'+MISSING_RGB+'. Private original-RGB review panels are generated separately.</p>'
    regression_records=regressions(scored,meta,preds,old_run,output,show_rgb=rgb_public,private=private,privacy=privacy,approved_ids=approved_ids)
    body+='<p><a href="regression/index.html">Three predeclared historical regression cases (separate tab)</a></p>'
    body+='<div class="grid">'+''.join(cards)+'</div></html>'; (output/'index.html').write_text(body)
    write(output/'GALLERY_MANIFEST.json',dict(schema='FULL192_RESULT_GALLERY_V1',rows=192,generated_thumbnails=len(records),
        detailed_unique_panels=len(detail_ids),detail_cap=12,selection_records=selected,all_category_counts=categories,
        public_RGB=rgb_public and not private,private_RGB_review=private,rights_receipt_sha256=sha(rights_receipt)if rights_receipt else None,
        display_privacy_plan_sha256=sha(privacy_plan)if privacy_plan else None,
        display_processing='solid privacy rectangles on display copy only; original pixels used for every inference and measurement'if rgb_public and not private else'original private RGB'if private else'no RGB',
        records=records,transform_records=transforms,regression_records=regression_records,source_sha256=sha(__file__),
        evaluation_seal_sha256=sha(evaluation/'EVALUATION_SEAL.json'),no_inference_or_measurement_changes=True,
        visual_review='NOT implied by PNG generation; actual pixel review is recorded separately',
        cost=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--old-run',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--original-score',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rgb-public',action='store_true');p.add_argument('--rights-receipt',type=Path);p.add_argument('--privacy-plan',type=Path)
    p.add_argument('--private',action='store_true');a=p.parse_args()
    run(a.run_dir,a.old_run,a.evaluation,a.original_score,a.output,rgb_public=a.rgb_public,rights_receipt=a.rights_receipt,privacy_plan=a.privacy_plan,private=a.private)


if __name__=='__main__':main()
