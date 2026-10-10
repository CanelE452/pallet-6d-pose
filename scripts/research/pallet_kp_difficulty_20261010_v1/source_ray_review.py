"""Pure stored-ray graph illustration and existing source RGB evidence crops.

No detector, head, pose solver, raycaster, renderer or training is invoked.
The assembly is a sensitivity diagnostic; source targets remain unchanged.
"""
from collections import Counter
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .source_ceiling import C,S,DOC,decode_choices


def eligible(q):
    return bool(q['cached_target']=='NONE' and q.get('actual_mesh_point_within_original_tolerance')
        and q.get('in_image') and q.get('supplied_visible_mask_support_3x3')
        and q.get('whole_edge_has_physical_sample'))


def rescued(q):
    return eligible(q) and not q['original_finite'] and q['fixed_normal_offset_rescues_source_depth']


def graph(sources,rows):
    result=[];hist=Counter();inside=Counter()
    for row in rows:
        source=sources[row['index']];choices=np.asarray(source['ideal_choices']).copy();witnesses=[]
        for q in row['queries']:
            if not rescued(q):continue
            position=q['offset_px']+32;lower=int(np.clip(np.floor(position),0,64))
            choices[q['query']]=lower if position-lower<=.5 else min(64,lower+1)
            witnesses.append(q['query'])
        decoded=decode_choices(source['frozen_selected_points'],choices);h,w=row['raw_hw']
        inframe=[q['id'] for q in decoded['corners'] if 0<=q['xy'][0]<w and 0<=q['xy'][1]<h]
        hist[decoded['corner_count']]+=1;inside[len(inframe)]+=1
        result.append(dict(id=row['id'],index=row['index'],source_ceiling_semantic_sha256=C.digest(source),
            source_ray_row_semantic_sha256=C.digest(row),original_ideal_corners=source['ideal']['corner_count'],
            original_in_frame_corners=len(source['before_mask_in_frame_corner_ids']),
            potential_ideal_corners=decoded['corner_count'],potential_in_frame_corners=len(inframe),
            diagnostic_rescued_query_ids=witnesses,potential_corner_ids=[q['id'] for q in decoded['corners']]))
    C.write(DOC/'SOURCE_RAY_DIAGNOSTIC_GRAPH.json',dict(schema='diagnostic_sensitivity_supported_candidate_graph_v1',families=128,
        original_valid_label_ideal_ge4=sum(s['ideal']['corner_count']>=4 for s in sources.values() if s['partition']=='source_test'),
        original_valid_label_in_frame_ideal_ge4=sum(len(s['before_mask_in_frame_corner_ids'])>=4 for s in sources.values() if s['partition']=='source_test'),
        potential_ideal_ge4=sum(v for k,v in hist.items() if k>=4),potential_in_frame_ideal_ge4=sum(v for k,v in inside.items() if k>=4),
        potential_corner_histogram=dict(hist),potential_in_frame_corner_histogram=dict(inside),
        witness_definition='Original NONE, original infinite ray, actual mesh closest point tolerance, in image, delivered mask support, physical edge supported; one fixed +/-0.05px ray hits within original source-depth tolerance',
        interpretation='Mechanical candidate graph of original positive targets plus fixed-offset source-depth witnesses. This is a sensitivity upper-bound illustration, not accepted corrected GT, a label edit, RGB-boundary ownership proof, deployed observation or pose success.',
        position_choice='Single nearest integer bin of the original supplied geometric intersection; lower at exact half, matching stored soft-target MAP; no averaging',
        additional_head_forwards=0,additional_rays=0,additional_PnP_calls=0,
        bindings=[C.binding(DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz'),C.binding(DOC/'SOURCE_CEILING_ROWS.jsonl.gz'),C.binding(Path(__file__))],rows=result))


def choose(rows,category,n=2):
    selected=[]
    for row in rows:
        for q in row['queries']:
            ok=rescued(q) and q['predicted_role']==0 if category=='rescued_boundary' else (
                eligible(q) and q['original_hit_in_front_of_source'] if category=='front_surface' else
                eligible(q) and not q['original_finite'] and not q['fixed_normal_offset_rescues_source_depth'])
            if ok:selected.append((row,q));break
        if len(selected)==n:break
    return selected


def crop_image(paths,q,half=30):
    rgb=cv2.imread(str(paths['rgb']));mask=cv2.imread(str(paths['visible']),0)
    assert rgb is not None and mask is not None
    h,w=mask.shape;x,y=q['position'];cx=int(round(x));cy=int(round(y))
    left=max(0,cx-half);right=min(w,cx+half+1);top=max(0,cy-half);bottom=min(h,cy+half+1)
    return cv2.cvtColor(rgb[top:bottom,left:right],cv2.COLOR_BGR2RGB),mask[top:bottom,left:right],(left,top,right,bottom)


def figure(cases,families,name,title,category):
    fig,axes=plt.subplots(len(cases),3,figsize=(12,3.9*len(cases)),squeeze=False)
    manifests=[]
    for rr,(row,q) in enumerate(cases):
        paths=S.locate(families[row['index']]);rgb,mask,(l,t,r,b)=crop_image(paths,q)
        xy=np.asarray(q['position'])-[l,t];normal=np.asarray(q['normal']);base=xy-q['offset_px']*normal
        for axis in axes[rr,:2]:
            axis.set_xlim(0,r-l-1);axis.set_ylim(b-t-1,0);axis.set_aspect('equal');axis.set_xlabel('Crop x [raw px]');axis.set_ylabel('Crop y [raw px]')
        axes[rr,0].imshow(rgb);axes[rr,0].contour(mask>127,levels=[.5],colors=['white'],linewidths=.8)
        axis=axes[rr,0];line=np.stack([xy-12*normal,xy+12*normal]);axis.plot(line[:,0],line[:,1],color='cyan',lw=1,alpha=.65)
        axis.plot(base[0],base[1],'rx',ms=9,mew=2,label='Initial query center')
        axis.plot(xy[0],xy[1],'o',mfc='none',mec='lime',ms=10,mew=2,label='Mesh-supported source point')
        axis.set_title(f"{row['id']} | q{q['query']} / edge{q['edge']}\nExisting RGB + supplied visible-mask contour",fontsize=10)
        axes[rr,1].imshow(mask,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        axes[rr,1].plot(xy[0],xy[1],'o',mfc='none',mec='lime',ms=10,mew=2)
        axes[rr,1].set_title(f"Actual supplied visible mask, not a hull\nCached label: NONE | role: {'BOUNDARY' if q['predicted_role']==0 else 'INTERNAL'}",fontsize=10)
        axis=axes[rr,2];delta=np.asarray([np.nan if x is None else x*1000 for x in q['ray_depth_minus_source_Z_m']])
        tol=row['depth_tolerance_m']*1000;axis.axhspan(-tol,tol,color='green',alpha=.15,label='Original depth tolerance')
        axis.axhline(0,color='green',lw=1);axis.set_xticks([0,1,2],['Exact edge','+0.05 px','-0.05 px'])
        axis.set_yscale('symlog',linthresh=max(tol,.01));axis.set_xlim(-.5,2.5)
        finite=np.isfinite(delta);axis.scatter(np.flatnonzero(finite),delta[finite],s=55,c=['#ef4444' if i==0 else '#2563eb' for i in np.flatnonzero(finite)],zorder=3)
        span=max(4*tol,float(np.nanmax(np.abs(delta))) if finite.any() else tol);axis.set_ylim(-1.5*span,1.5*span)
        for j in range(3):
            if finite[j]:axis.annotate(f'{delta[j]:+.4g}',(j,delta[j]),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
            else:axis.text(j,.94,'No intersection',transform=axis.get_xaxis_transform(),ha='center',va='top',fontsize=9,color='#b91c1c')
        axis.grid(alpha=.25);axis.set_ylabel('First-surface camera Z minus source Z [mm]')
        axis.set_title(f"Physical closest distance: {q['closest_distance_m']:.2g} m\nSource Z: {q['source_camera_Z']:.4f} m; tolerance: {tol:.3f} mm",fontsize=10)
        manifests.append(dict(id=row['id'],index=row['index'],query_id=q['query'],edge=q['edge'],category=category,
            crop_raw_xyxy=[l,t,r,b],source_RGB=C.binding(paths['rgb']),visible_mask=C.binding(paths['visible']),
            source_annotation=C.binding(paths['label']),source_ray_rows=dict(file='SOURCE_RAY_VALIDATION_ROWS.jsonl.gz',line=row['index']-895,frame_semantic_sha256=C.digest(row)),
            query_semantic_sha256=C.digest(q),query=q,
            green_mark='Source geometric query point confirmed near actual mesh; not an asserted independent visible RGB correspondence',
            mask='Delivered actual visible mask; no cuboid/hull substituted',
            cyan_mark='Original normal search direction, not a physical boundary contour'))
    fig.suptitle(title,fontsize=14,y=.995)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',ncol=2,fontsize=9)
    fig.text(.02,.043,'Fixed normal offsets are review diagnostics. Original labels, RGB, models and training remain unchanged.',fontsize=9)
    fig.tight_layout(rect=[0,.065,1,.965]);out=DOC/'figures'/name;out.parent.mkdir(exist_ok=True)
    fig.savefig(out,dpi=130);plt.close(fig)
    return dict(file=C.binding(out),selection='First source-test frame in retained index order, then first ascending query satisfying category; distinct frame per row',cases=manifests)


def run():
    cv2.setNumThreads(1);sources={r['index']:r for r in C.iter_rows(DOC/'SOURCE_CEILING_ROWS.jsonl.gz')}
    rows=list(C.iter_rows(DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz'));families=S.selected_families();graph(sources,rows)
    rescuedcases=choose(rows,'rescued_boundary')
    controls=choose(rows,'front_surface',n=1)+choose(rows,'unresolved_infinite',n=1)
    figures=[figure(rescuedcases,families,'08_source_ray_sensitivity.png',
        'Actual-mesh boundary rays: fixed subpixel witnesses for two original NONE labels','rescued_boundary'),
        figure(controls,families,'09_source_ray_controls.png',
        'Limits: a real front surface and an unresolved exact-edge miss','front_surface_and_unresolved')]
    C.write(DOC/'SOURCE_RAY_VISUAL_CASES.json',dict(schema='stored_actual_mesh_ray_review_crops_v1',figures=figures,
        actual_existing_RGB_only=True,new_RGB_generated=0,derived_annotated_crop_files=2,
        additional_rays=0,additional_head_forwards=0,additional_detector_forwards=0,additional_PnP_calls=0,
        protocol=C.binding(DOC/'SOURCE_RAY_PROTOCOL.json'),ray_rows=C.binding(DOC/'SOURCE_RAY_VALIDATION_ROWS.jsonl.gz'),
        graph=C.binding(DOC/'SOURCE_RAY_DIAGNOSTIC_GRAPH.json'),code=C.binding(Path(__file__)),
        limitations='A nearby actual-mesh source-depth witness diagnoses edge-ray sensitivity; it does not prove RGB boundary ownership or accept repaired GT. Front-surface and unrescued cases must not be relabeled visible.'))
    print('SOURCE_RAY_REVIEW_COMPLETE',sum(f['file']['bytes'] for f in figures),flush=True)


if __name__=='__main__':run()
