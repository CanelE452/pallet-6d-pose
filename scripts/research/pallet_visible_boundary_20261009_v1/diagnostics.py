"""Post-lock three-case plots and forty reference-assisted diagnostic calls.

References determine the diagnostic starts and visualization only. They never
change any sealed experimental coordinate, route, model, or final pose.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import common as C

CASES=(('photo1','plastic_day_01:007917',(2,3,4,0)),
       ('photo2','eval_pallet07:1778652152626116352',(0,3)),
       ('photo3','eval_pallet07:1778652148628229376',(0,3)))
JITTERS=((0,0),(1,0),(-1,0),(0,1),(0,-1))


def selected_rows(path,ids):
    result={}
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['id'] in ids:result[row['id'],row['method']]=row
    return result


def _line_segment(normal,offset,lo,hi):
    n=np.asarray(normal,float);candidates=[]
    if abs(n[1])>1e-10:
        for x in (lo[0],hi[0]):
            y=-(offset+n[0]*x)/n[1]
            if lo[1]<=y<=hi[1]:candidates.append([x,y])
    if abs(n[0])>1e-10:
        for y in (lo[1],hi[1]):
            x=-(offset+n[1]*y)/n[0]
            if lo[0]<=x<=hi[0]:candidates.append([x,y])
    return np.asarray(candidates[:2],float)


def _plot(image,case,records,path):
    colors=dict(seed='#dfb120',old='#e84246',wide='#b533d6',boundary='#00a9d6',capped='#e88115')
    markers=dict(seed='o',old='o',wide='P',boundary='^',capped='s')
    fig,axes=plt.subplots(len(records),2,figsize=(12,4.8*len(records)+1.0),squeeze=False)
    for pair,record in zip(axes,records):
        k=record['corner'];gt=np.asarray(record['reference_xy']);points=record['coordinates']
        chosen=[np.asarray(points[a]) for a in points]
        bounds=np.asarray([gt,*chosen]);lo=bounds.min(0)-8;hi=bounds.max(0)+8
        for ax,start in zip(pair,('BASE','N3')):
            arms={'seed':start,'old':'SUBPIX' if start=='BASE' else 'N3_SUBPIX',
                  'wide':start+'_WIDE_NATIVE','boundary':start+'_BOUNDARY_NATIVE','capped':start+'_BOUNDARY_CAP1'}
            ax.imshow(image,interpolation='nearest');ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal')
            pair_info=record[start+'_boundary_pair']
            if pair_info['line_supported']:
                segment=_line_segment(pair_info['line_normal'],pair_info['line_offset'],lo,hi)
                if len(segment)==2:
                    ax.plot(segment[:,0],segment[:,1],color=colors['boundary'],lw=1.5,
                            ls='-' if pair_info['accepted'] else '--',alpha=.85,zorder=4)
            for label,arm in arms.items():
                pt=np.asarray(points[arm]);ax.scatter(*pt,s=110,marker=markers[label],facecolors='none',
                    edgecolors=colors[label],lw=2,zorder=5)
            ax.scatter(*gt,s=90,marker='D',facecolors='none',edgecolors='#12e276',lw=2.1,zorder=6)
            errors=record['fixed_identity_error_px']
            pair_label='accepted' if pair_info['accepted'] else 'rejected'
            ax.set_title(f"corner {k} | {start} start | boundary pair {pair_label}\n"
                f"seed {errors[start]:.2f}; oldSP {errors[arms['old']]:.2f}; wide {errors[arms['wide']]:.2f}px\n"
                f"boundary native {errors[arms['boundary']]:.2f}; Base-cap {errors[arms['capped']]:.2f}px",fontsize=10,pad=13)
            ax.tick_params(labelsize=8);ax.grid(False)
    handles=[Line2D([0],[0],ls='none',marker=markers[k],markerfacecolor='none',color=colors[k],
                   label=v) for k,v in [('seed','Base/N3 seed'),('old','old SUBPIX5'),('wide','wide SUBPIX25 native'),
                                       ('boundary','shared boundary native'),('capped','shared boundary Base-CAP1')]]
    handles+=[Line2D([0],[0],ls='none',marker='D',markerfacecolor='none',color='#12e276',label='frozen reference'),
              Line2D([0],[0],color=colors['boundary'],label='accepted pair shared line'),
              Line2D([0],[0],color=colors['boundary'],ls='--',label='fit exists; pair rejected')]
    fig.legend(handles=handles,ncol=4,loc='lower center',bbox_to_anchor=(.5,.014),fontsize=9)
    fig.suptitle(f"{case['name']} | {case['id']} | raw pixels; frozen coordinate source: {case['reference_source']}\n"
                 'Distances use fixed original point identity; reference-assisted jitter calls are diagnostic only',fontsize=13,y=.99)
    fig.subplots_adjust(left=.065,right=.975,bottom=.14 if len(records)==2 else .075,
                        top=.845 if len(records)==2 else .92,hspace=.85,wspace=.23)
    fig.savefig(path,dpi=160);plt.close(fig)


def run():
    out=C.DOC/'DIAGNOSTICS.json'
    assert not out.exists(),'Preserve completed diagnostic calls; no hidden replay'
    marker=C.SCRATCH/'DIAGNOSTIC_STARTED.json'
    assert not marker.exists(),'Preserve partial diagnostic attempt; no hidden replay'
    lock_path=C.DOC/'COORDINATES_LOCK.json';coordinates=C.DOC/'COORDINATES.jsonl.gz'
    lock=C.read(lock_path)
    assert lock['complete'] and lock['prediction_only'] and not lock['scoring_targets_loaded']
    assert C.sha(coordinates)==lock['predictions']['sha256']
    assert C.sha(C.DOC/'PROTOCOL.json')==lock['protocol_sha256']
    ids={fid for _,fid,_ in CASES}
    rows=selected_rows(coordinates,ids)
    old_path=C.ROOT/'_docs/experiments/pallet_n3_subpix_20261008_v1/PREDICTIONS.jsonl.gz'
    controls=selected_rows(old_path,ids)
    assert len(rows)==24 and len(controls)==12
    # All inference coordinates are immutable before any reference access here.
    target_path=C.ROOT/'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json'
    targets=C.read(target_path)
    bounds={r['id']:r for r in C.read(C.DOC/'INPUTS.json')['frames']}
    annotation_path=C.ROOT/'_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/static/LABEL_PROVENANCE_AUDIT.json'
    provenance={r['id']:r for r in C.read(annotation_path)['rows'] if r['population']=='DEV319'}
    report=dict(schema='visible_boundary_postlock_diagnostics_v1',complete=False,
        scope='Reference-assisted image-only diagnostics after all319 coordinates locked; not deployment accuracy or a route-selection rule.',
        protocol_sha256=lock['protocol_sha256'],coordinates_sha256=lock['predictions']['sha256'],
        inference_coordinates_unchanged=True,reference_used_only_in_diagnostics=True,
        inputs=[C.binding(p) for p in (lock_path,coordinates,old_path,target_path,annotation_path)],
        fixed_settings=dict(winSize=[5,5],zeroZone=[-1,-1],maxCount=40,epsilon=.001,
            jitter_xy=[list(j) for j in JITTERS],start='frozen reference + jitter, converted once to float32'),
        new_NN_forwards=0,new_F_calls=0,new_training=0,new_benchmark_calls=0,cases=[],corner_rows=[])
    cv2.setNumThreads(1);calls=0;original_coordinate_sha=C.sha(coordinates)
    C.write(marker,dict(started=True,coordinate_lock_sha256=C.sha(lock_path),cornerSubPix_calls=0))
    for name,fid,corners in CASES:
        prov=provenance[fid];image_path=C.ROOT/prov['image']['path'];ann_path=C.ROOT/prov['annotation']['path']
        assert C.sha(image_path)==prov['image']['sha256']==bounds[fid]['image_sha256']
        assert C.sha(ann_path)==prov['annotation']['sha256']
        image=cv2.imread(str(image_path),cv2.IMREAD_COLOR);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        assert list(gray.shape)==bounds[fid]['raw_hw']
        annotations=C.read(ann_path)['objects'][0]['keypoint_annotations'];gt=np.asarray(targets[fid]['gt'],np.float64)
        sources={str(k):annotations[k]['source'] for k in corners}
        case=dict(name=name,id=fid,image=prov['image'],annotation=prov['annotation'],raw_hw=list(gray.shape),
            requested_corners=list(corners),reference_sources=sources,
            reference_source='manual_click' if name=='photo1' else 'unknown',
            annotation_source_does_not_validate_subpixel_physical_truth=True,
            figure=f'figures/{name.upper()}_REFINEMENT.png')
        records=[]
        for k in corners:
            points={a:np.asarray(controls[fid,a]['native_points'],np.float64)[k] for a in C.CONTROLS}
            points.update({a:np.asarray(rows[fid,a]['native_points'],np.float64)[k] for a in C.NEW_ARMS})
            q0=points['BASE'];record=dict(id=fid,photo=name,corner=k,reference_xy=gt[k],
                reference_source=annotations[k]['source'],coordinates=points,
                fixed_identity_error_px={a:float(np.linalg.norm(q-gt[k])) for a,q in points.items()},
                full_corner_symmetry_selection='not rerun for this diagnostic; distances use unchanged original numbered identity',
                displacement_from_Base_px={a:float(np.linalg.norm(q-q0)) for a,q in points.items()},
                Base_total_cap_px=.01*float(np.hypot(*gray.shape)),near_reference=[])
            for jitter in JITTERS:
                seed=gt[k]+np.asarray(jitter,np.float64)
                C.write(marker,dict(started=True,active_id=fid,active_corner=k,active_jitter=jitter,cornerSubPix_calls_started=calls+1))
                value=cv2.cornerSubPix(gray,seed.astype(np.float32).reshape(1,1,2).copy(),(5,5),(-1,-1),
                    (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
                calls+=1
                if value is None or np.asarray(value).size!=2:raise ValueError('Diagnostic cornerSubPix returned no coordinate')
                value=np.asarray(value,np.float64).reshape(2)
                assert np.isfinite(value).all()
                record['near_reference'].append(dict(jitter_xy=list(jitter),initial_native=seed,
                    actual_cv_final=value,reference_error_px=float(np.linalg.norm(value-gt[k])),
                    move_from_diagnostic_start_px=float(np.linalg.norm(value-seed)),
                    reference_assistance=True,deployment_result=False))
            for start in ('BASE','N3'):
                diag=rows[fid,start+'_BOUNDARY_NATIVE']['correction']['diagnostics']
                pair=next(p for p in diag['paired_records'] if k in p['corners'])
                edge=diag['edge_records'][pair['shared_edge']]
                record[start+'_boundary_pair']=dict(paired_corners=pair['corners'],status=pair['status'],accepted=pair['accepted'],
                    shared_edge=pair['shared_edge'],native_shared_line_residual=pair['shared_line_native_residual_px'],
                    line_supported=edge['supported'],line_normal=edge.get('normal'),line_offset=edge.get('offset'))
            records.append(record);report['corner_rows'].append(record)
        report['cases'].append(case)
        figures=C.DOC/'figures';figures.mkdir(parents=True,exist_ok=True)
        _plot(image[:,:,::-1],case,records,figures/(name.upper()+'_REFINEMENT.png'))
    assert calls==40 and C.sha(coordinates)==original_coordinate_sha
    report.update(complete=True,execution=dict(reference_assisted_cornerSubPix_calls=40,diagnostic_image_decodes=3,
        fixed_reference_corner_count=8,model_forwards=0,F_calls=0,benchmark_calls=0),
        figure_bindings=[C.binding(C.DOC/c['figure']) for c in report['cases']])
    C.write(out,report);C.write(marker,dict(complete=True,cornerSubPix_calls_started=calls,cornerSubPix_calls_complete=calls,
        output=C.binding(out)))
    return report


def render_saved():
    """Redraw completed results without any correction, model, or pose call."""
    out=C.DOC/'DIAGNOSTICS.json';report=C.read(out)
    assert report['complete'] and report['execution']['reference_assisted_cornerSubPix_calls']==40
    coordinates=C.DOC/'COORDINATES.jsonl.gz';coordinate_sha=C.sha(coordinates)
    assert coordinate_sha==report['coordinates_sha256']
    previous_report_sha=C.sha(out)
    for case in report['cases']:
        image_path=C.ROOT/case['image']['path']
        assert C.sha(image_path)==case['image']['sha256']
        image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        records=[r for r in report['corner_rows'] if r['id']==case['id']]
        _plot(image[:,:,::-1],case,records,C.DOC/case['figure'])
    assert C.sha(coordinates)==coordinate_sha
    report['figure_bindings']=[C.binding(C.DOC/c['figure']) for c in report['cases']]
    report.setdefault('figure_only_redraws',[]).append(dict(
        previous_diagnostic_report_sha256=previous_report_sha,image_decodes=3,
        correction_calls=0,cornerSubPix_calls=0,model_forwards=0,F_calls=0,
        reason='Shorter pair-state titles after visual inspection; diagnostic coordinates unchanged.'))
    C.write(out,report)
    marker=C.SCRATCH/'DIAGNOSTIC_STARTED.json';value=C.read(marker)
    value['output']=C.binding(out);C.write(marker,value)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-only',action='store_true',help='Redraw saved diagnostics; zero correction calls.')
    args=parser.parse_args()
    result=render_saved() if args.render_only else run()
    print(json.dumps(dict(complete=result['complete'],execution=result['execution'],
                         figure_only_redraws=result.get('figure_only_redraws',[]))))
