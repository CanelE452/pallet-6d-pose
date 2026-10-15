"""Cached CAL graph statistics and three actual overflow-edge diagnostics.

No production observer, pose, noise model or prior output is changed. The
private function facade captures already measured NMS pixels without mutating
the frozen module. Reference labels and TEST/remaining images are not opened.
"""
from copy import deepcopy
from pathlib import Path
import json
import types
import cv2
import numpy as np

from . import common as C
from . import observe_native_edge_chains_v4_17b as DRIVER
from . import compact_edge_component_dag_v4_18 as GRAPH

OUTPUT=DRIVER.BOUNDARY/'compact_edge_graph_diagnostic_v4_18'
NMS=DRIVER.NMS


def _plot(image,compact,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for ax in axes:ax.imshow(rgb)
    points=[]
    for component in compact['raw_components']:
        xy=np.asarray(component['actual_NMS_points_xy']);points.append(xy)
        axes[1].scatter(xy[:,0],xy[:,1],s=5,c='#ff9c00',alpha=.65)
    for family in compact['whole_component_line_families']:
        xy=np.asarray(family['union_TLS']['observed_points']);good=np.asarray(family['union_TLS']['inlier_mask'],bool)
        axes[1].scatter(xy[good,0],xy[good,1],s=10,c='#00cfff',alpha=.8)
    if points:
        xy=np.vstack(points);lo=xy.min(0)-8;hi=xy.max(0)+8
        for ax in axes:ax.set(xlim=(max(0,lo[0]),min(image.shape[1],hi[0])),ylim=(min(image.shape[0],hi[1]),max(0,lo[1])))
    axes[0].set_title(title+' actual original RGB')
    axes[1].set_title('Orange: all actual raw peaks; cyan: admissible whole-group TLS inliers')
    for ax in axes:ax.set_aspect('equal')
    fig.suptitle('Diagnostic only: no bridge pixels, full keypoint, noise fit, pose fit or default change',fontsize=11)
    fig.savefig(path,dpi=160);plt.close(fig)


def main(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior diagnostics')
    cal=DRIVER.OUTPUT/'CAL_OBSERVATIONS.jsonl.gz';cal_seal=DRIVER._verify_observation_seal(DRIVER.OUTPUT,'CAL')
    rows=list(C.rows(cal));byid={r['id']:r for r in rows};statistics=[]
    for row in rows:
        for ray in row['chain_observation']['edge_records']:
            audit=ray.get('component_linking_audit',{})
            if not audit.get('combinatorial_overflow_abstained'):continue
            summary=GRAPH.graph_summary(audit['links'])
            sub=[]
            for group in summary['connected_groups']:
                links=[x for x in audit['links'] if x['components'][0] in group]
                g=GRAPH.graph_summary(links,group)
                sub.append(dict(nodes=len(group),edges=len(g['edges']),
                    root_paths=g['root_terminal_path_count_upper_bound'],
                    reduced_root_paths=g['transitive_reduced_root_path_count']))
            statistics.append(dict(id=row['id'],native_corner=ray['native_corner'],native_neighbor=ray['native_neighbor'],
                actual_components=audit['actual_components'],graph=summary,connected_group_statistics=sub))
    # Overflow size/structure only. No pose error or reference selects examples.
    ordered=sorted(statistics,key=lambda s:(-max([int(g['root_paths']) for g in s['connected_group_statistics']]+[0]),s['id'],s['native_corner'],s['native_neighbor']))
    selected=[];seen_frames=set()
    for row in ordered:
        if row['id'] in seen_frames:continue
        selected.append(row);seen_frames.add(row['id'])
        if len(selected)==3:break
    source_paths=[Path(__file__),Path(GRAPH.__file__),Path(NMS.__file__),Path(NMS.D.__file__),Path(NMS.F.__file__)]
    sources=[DRIVER._binding(p) for p in source_paths]
    inputs=[DRIVER._binding(p) for p in (cal,DRIVER.OUTPUT/'CAL_OBSERVATION_SEAL.json',DRIVER.OUTPUT/'PROTOCOL.json')]
    output.mkdir(parents=True)
    C.write(output/'PROTOCOL.json',dict(schema='COMPACT_EDGE_DAG_DIAGNOSTIC_PROTOCOL_V4_18',
        sources=sources,inputs=inputs,selection='Three distinct CAL frames with largest cached overflow DAG path bound; no metrics/GT selection.',
        selected_edges=[{k:r[k] for k in ('id','native_corner','native_neighbor')} for r in selected],
        production_observer_changed=False,new_full_observer_run=False,pose_calls=0,NN_calls=0,reference_coordinate_decodes=0))
    cv2.setNumThreads(1);actual=[]
    for selected_row in selected:
        row=byid[selected_row['id']];k=selected_row['native_corner'];mate=selected_row['native_neighbor']
        geometry=row['initial_geometry'];states=row['states'];hidden=geometry['initial_self_hidden']
        if hidden[k] or hidden[mate] or states[k] not in ('B','I') or states[mate]=='S':raise ValueError('State-first query gate')
        visible={r['native_corner']:r for r in row['visible_measurements'] if not hidden[r['native_corner']] and states[r['native_corner']] in ('B','I')}
        q=np.asarray(visible[k]['xy']);other=visible.get(mate)
        projection={i:np.asarray(geometry['initial_projection_xy'][i]) for i in (k,mate)}
        image_path=C.ROOT/row['image'];mask_path=C.ROOT/row['mask_binding']['path']
        if C.sha(image_path)!=row['image_sha256'] or C.sha(mask_path)!=row['mask_binding']['sha256']:raise ValueError('RGB/mask binding')
        image=cv2.imread(str(image_path))
        with np.load(mask_path) as data:mask=data['pallet_mask_clean'].astype(float)
        lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32),(3,3),.8)
        gradients=cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/8.
        cfg=NMS._config(None);captured=[]
        if other is None:
            length=float(np.linalg.norm(projection[mate]-projection[k]));lo=cfg['along_start_px']
            hi=min(cfg['along_end_px'],.8*length,length-cfg['boundary_context_px'])
        else:
            length=float(np.linalg.norm(np.asarray(other['xy'])-q));lo=cfg['along_start_px'];hi=length-lo
        def capture(xy,observed,valid,component,direction,seed_normal,configuration):
            original,audit=NMS._collinear_component_hypotheses(xy,observed,valid,component,direction,seed_normal,configuration)
            gn,_=NMS._image_gradient_normal(NMS._sample(gradients[0],xy),NMS._sample(gradients[1],xy),seed_normal)
            contrast=np.linalg.norm(NMS._sample(lab,observed+gn*cfg['RGB_probe_px'])-NMS._sample(lab,observed-gn*cfg['RGB_probe_px']),axis=-1)
            positions=((xy-q)@direction-lo)/(hi-lo)
            compact=GRAPH.compact_actual_component_groups(xy,observed,valid,component,direction,seed_normal,cfg,
                confidence=np.clip(contrast/40.,.05,1.),positions=positions,links=audit['links'])
            captured.append(compact)
            return original,audit
        function=types.FunctionType(NMS._edge_chains.__code__,{**NMS._edge_chains.__globals__,
            '_collinear_component_hypotheses':capture},NMS._edge_chains.__name__,NMS._edge_chains.__defaults__,NMS._edge_chains.__closure__)
        face=NMS.D._edge_face_role(NMS.D._face_context(geometry),k,mate)
        original=function(image,mask,lab,gradients,k,mate,visible[k],other,projection,face,cfg)
        if len(captured)!=1 or not original['component_linking_audit']['combinatorial_overflow_abstained']:
            raise ValueError('Reproduced edge must exactly retain old overflow outcome')
        compact=captured[0];stem=row['id'].replace(':','__')+f'__edge{k}_{mate}'
        _plot(image,compact,output/(stem+'.png'),f'{row["id"]} edge{k}-{mate}')
        actual.append(dict(id=row['id'],native_corner=k,native_neighbor=mate,image=DRIVER._binding(image_path),
            mask=DRIVER._binding(mask_path),figure=stem+'.png',original_overflow_audit=original['component_linking_audit'],compact=compact))
    summary=[dict(id=r['id'],native_edge=[r['native_corner'],r['native_neighbor']],
        actual_raw_components=len(r['compact']['raw_components']),DAG_edges=len(r['compact']['graph']['edges']),
        old_path_upper_bound=r['compact']['graph']['nontrivial_all_start_path_count_upper_bound'],
        whole_group_line_families=len(r['compact']['whole_component_line_families']),
        unresolved_groups=len(r['compact']['unresolved_groups']),
        whole_group_actual_points=sum(f['actual_point_count'] for f in r['compact']['whole_component_line_families'])) for r in actual]
    C.write(output/'DIAGNOSTIC.json',dict(schema='COMPACT_EDGE_DAG_DIAGNOSTIC_V4_18',cached_CAL_overflow_statistics=statistics,
        actual_small_cases=actual,summary=summary,production_observer_or_input_changes=0,new_CAL_fit=0,pose_calls=0,NN_calls=0,
        reference_coordinate_decodes=0,no_path_multiplicity_as_independent_data=True,
        no_full_observer_rerun=True,new_compact_representation_not_adopted=True))
    for item in sources+inputs:
        if C.sha(C.ROOT/item['path'])!=item['sha256']:raise ValueError('Bound input/source changed')
    C.write(output/'DIAGNOSTIC_SEAL.json',dict(sources=sources,inputs=inputs,
        artifacts=[DRIVER._binding(p) for p in sorted(output.iterdir()) if p.is_file()]))
    print(json.dumps(dict(output=str(output),summary=summary)))


if __name__=='__main__':main()
