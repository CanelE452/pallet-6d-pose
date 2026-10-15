"""Fixed top3 absolute CAL line-normal errors per policy; no method changes."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import calibrate_geometry_guided_edge_noise_v4_29 as D
from . import observe_geometry_guided_edges_v4_29b as OBS

C=D.C
OUTPUT=D.OUTPUT.parent/'CAL_geometry_guided_noise_diagnosis_v4_29'
PLAN=OUTPUT.parent/'CAL_geometry_guided_noise_diagnosis_plan_v4_29/PREDECLARATION.json'
EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def selection(records_by_policy):
    selected=[]
    for policy in D.M.SOURCE_POLICIES:
        candidates=[]
        for row in records_by_policy[policy]:
            if row['partition']!='CAL':raise ValueError('CAL-only diagnosis')
            if not row['compatible']:continue
            for group in row['groups']:
                if len(group['components'])!=1:raise ValueError('One selected component per physical edge')
                component=group['components'][0]
                candidates.append(dict(id=row['id'],policy=policy,session=row['session'],
                    physical_native_edge=group['physical_native_edge'],component=component,
                    absolute_normal_error_px=abs(float(component['normal_error_px']))))
        candidates.sort(key=lambda x:(-x['absolute_normal_error_px'],x['id'],
            tuple(x['physical_native_edge']),x['component']['component_id']))
        selected.extend(dict(x,policy_rank=i+1)for i,x in enumerate(candidates[:3]))
    return selected


def _sources():
    modules=(D,D.M,D.OLD.ALIGN,D.OLD.FACTOR,OBS)
    values={str(Path(m.__file__).resolve().relative_to(C.ROOT)):C.sha(m.__file__)for m in modules}
    for p in (Path(__file__),Path(__file__).with_name('test_diagnose_CAL_geometry_guided_noise_v4_29.py')):
        values[str(p.resolve().relative_to(C.ROOT))]=C.sha(p)
    return values


def declare():
    if PLAN.exists():raise FileExistsError('Preserve fixed diagnostic declaration')
    PLAN.parent.mkdir(parents=True)
    paths=[D.OUTPUT/'MODEL_SEAL.json',D.OUTPUT/'CAL_AUTHENTICATED_INPUTS.jsonl.gz',D.LABELS,
        OBS.OUTPUT/'CAL_ROW_INDEX.json',OBS.OUTPUT/'CAL_OBSERVATION_SEAL.json']
    paths.extend(D.OUTPUT/p/'CAL_ERRORS.jsonl.gz'for p in D.M.SOURCE_POLICIES)
    C.write(PLAN,dict(schema='CAL_ONLY_TOP3_NORMAL_ERROR_DIAGNOSTIC_PREDECLARATION_V29',
        UTC=datetime.now(timezone.utc).isoformat(),source_sha256=_sources(),
        input_sha256={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths},
        selection='per BOTH policies compatible44 CAL groups, top3 descending absolute saved normal_error_px; ties frameID/nativeedge/componentID only; union displayed',
        selection_before_actual_image_read=True,reference='same original fixed geometry-only CAL44 alignment; no correction/pose-error phase selection',
        reference_overlay='explicit CAL diagnostic reference, never runtime measurement or guide',
        panels='original RGB / crop actual inliers+TLS+initial and selected guide+CAL reference / actual target mask / same-axis CAL reference edges',
        fixed_models_runtime_source_and_parameters_unchanged=True,new_posefit_NN_image_search_or_TEST_REMAINDER_labels=0))
    print(dict(plan=str(PLAN),sha256=C.sha(PLAN)))


def segment(component):
    n=np.asarray(component['normal_unit_xy']);t=np.array([-n[1],n[0]])
    actual=np.asarray(component['actual_inlier_xy']);v=actual@t
    return np.asarray([-component['offset_px']*n+float(a)*t for a in (v.min(),v.max())])


def _plot_line(ax,q,color,label,style='-',width=1.7):
    q=np.asarray(q);ax.plot(q[:,0],q[:,1],style,color=color,lw=width,label=label)


def render():
    declaration=C.read(PLAN)
    for key in ('source_sha256','input_sha256'):D._verify(declaration[key])
    if OUTPUT.exists():raise FileExistsError('Preserve diagnostic output')
    records={p:list(C.rows(D.OUTPUT/p/'CAL_ERRORS.jsonl.gz'))for p in D.M.SOURCE_POLICIES}
    chosen=selection(records)
    # Selection is written durably before the first actual RGB/mask decode.
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'SELECTION.json',dict(plan_sha256=C.sha(PLAN),records=chosen,actual_RGB_reads_so_far=0))
    prepared={r['id']:r for r in C.rows(D.OUTPUT/'CAL_AUTHENTICATED_INPUTS.jsonl.gz')}
    if len(prepared)!=46 or any(r['partition']!='CAL'for r in prepared.values()):raise ValueError('CAL46 metadata only')
    labels=C.read(D.LABELS)['rows']
    if set(labels)!=set(prepared):raise ValueError('Isolated CAL46 references only')
    index={x['id']:x for x in C.read(OBS.OUTPUT/'CAL_ROW_INDEX.json')['entries']}
    unique={}
    for item in chosen:
        key=(item['id'],item['component']['component_id'])
        unique.setdefault(key,[]).append(item)
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports=[];bound={}
    for number,((ident,_),membership)in enumerate(unique.items()):
        item=membership[0];component=item['component']['source_component'];k=component['native_corner'];mate=component['native_neighbor']
        row=OBS._read_single(OBS._row_path(OBS.OUTPUT,ident),index[ident])
        g=prepared[ident]['initial_geometry'];alignment=D.OLD.ALIGN._reference_alignment({'runtime':{'initial_geometry':g}},labels[ident])
        if not alignment['compatible']:raise ValueError('Original fixed compatible44 alignment required')
        xyz=np.asarray(g['native_xyz']);K=np.asarray(g['K']);R=np.asarray(alignment['R_reference_native']);t=np.asarray(alignment['t_reference'])
        reference=C.project(xyz,R,t,K);initial=C.project(xyz,np.asarray(g['R_cf']),np.asarray(g['centroid'])-np.asarray(g['R_cf'])@xyz.mean(0),K)
        ref=component['selected_chain_ref'];observation=row['observation'];chain=observation['chain_observation']['edge_records'][ref['ray_index']]['chains'][ref['chain_index']]
        actual=np.asarray(chain['observed_points'])[np.asarray(chain['inlier_mask'],bool)]
        if not np.array_equal(actual,np.asarray(component['actual_inlier_xy'])):raise ValueError('Exact selected actual support required')
        guide_id=ref['guide_reference']['geometry_ref_id'];gg=observation['guide_geometries'][guide_id]
        guide=C.project(np.asarray(gg['native_xyz']),np.asarray(gg['R_cf']),np.asarray(gg['centroid'])-np.asarray(gg['R_cf'])@np.asarray(gg['native_xyz']).mean(0),K)
        n=np.asarray(component['normal_unit_xy']);center=actual.mean(0);anchor=center-n*(n@center+component['offset_px'])
        if not np.allclose(anchor,component['support_anchor_xy'],atol=1e-10,rtol=0):raise ValueError('Unweighted centroid projected onto saved weighted-TLS line')
        p=D.OLD.N.V._pose_vector(R,t,max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3))
        _,_,audit=D.OLD.FACTOR.projected_edge_residual_and_jacobian(p,xyz,K,max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3),component)
        if abs(audit['normal_residual_px']-item['component']['normal_error_px'])>1e-8:raise ValueError('Saved normal residual must replay exactly')
        image_path=C.ROOT/row['image'];mask_path=C.ROOT/row['mask_binding']['path']
        if C.sha(image_path)!=row['image_sha256']or C.sha(mask_path)!=row['mask_binding']['sha256']:raise ValueError('Original actual pixels/mask changed')
        bound[str(image_path.relative_to(C.ROOT))]=C.sha(image_path);bound[str(mask_path.relative_to(C.ROOT))]=C.sha(mask_path)
        image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        with np.load(mask_path,allow_pickle=False)as data:mask=data['pallet_mask_clean'].copy()
        h,w=image.shape[:2];roi=np.vstack([actual,reference[[k,mate]],initial[[k,mate]]])
        lo=np.maximum([0,0],roi.min(0)-45);hi=np.minimum([w-1,h-1],roi.max(0)+45)
        fig,axes=plt.subplots(1,4,figsize=(20,5));measured=segment(component)
        physical=(xyz[mate]-xyz[k])@np.asarray(g['physical_registration_Q']);axis=int(np.argmax(np.abs(physical)))
        same=[]
        for a,b in EDGES:
            v=(xyz[b]-xyz[a])@np.asarray(g['physical_registration_Q'])
            if int(np.argmax(np.abs(v)))==axis:same.append([a,b])
        for j,ax in enumerate(axes):
            ax.imshow(mask if j==2 else image,cmap='gray'if j==2 else None,vmin=0 if j==2 else None,vmax=1 if j==2 else None)
            _plot_line(ax,measured,'#00b8df','actual TLS support extent',width=2.4)
            ax.scatter(actual[:,0],actual[:,1],s=9,c='#ffe600',label='actual saved RGB inliers',zorder=6)
            _plot_line(ax,reference[[k,mate]],'#ce24ac','CAL diagnostic reference edge','--',2.)
            _plot_line(ax,initial[[k,mate]],'#ff8c1b','original initial model edge',':')
            _plot_line(ax,guide[[k,mate]],'#29a45e','selected generating model edge','-.')
            if j==3:
                for pair in same:
                    _plot_line(ax,reference[pair],'#8a56c9',None,':',1.)
                    midpoint=reference[pair].mean(0);ax.text(*midpoint,f'{pair[0]}-{pair[1]}',color='#7a2387',fontsize=9)
            ax.set_xlim((0,w)if j==0 else(lo[0],hi[0]));ax.set_ylim((h,0)if j==0 else(hi[1],lo[1]));ax.set_aspect('equal');ax.set_xlabel('original image x');ax.set_ylabel('original image y')
            ax.set_title(['Original RGB (unchanged brightness)','Actual line / model / CAL reference','Actual target mask (not occluder truth)','Other same-physical-axis reference edges'][j],fontsize=10)
        axes[0].legend(loc='lower left',fontsize=7)
        title=f'{ident} | named edge {k}-{mate} | physical axis {"XYZ"[axis]} | CAL normal {audit["normal_residual_px"]:.3f}px / angle {audit["unoriented_angle_deg"]:.3f}deg'
        fig.suptitle(title,fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);asset=f'case{number}_actual.png';fig.savefig(OUTPUT/asset,dpi=160);plt.close(fig)
        node=next(nod for nod in observation['joint_graph']['line_candidates']if nod['name']==ref['line_name'])
        reports.append(dict(id=ident,policies=[dict(policy=m['policy'],rank=m['policy_rank'])for m in membership],native_edge=[k,mate],physical_axis='XYZ'[axis],figure=asset,
            signed_normal_residual_px=audit['normal_residual_px'],angle_deg=audit['unoriented_angle_deg'],actual_inliers=len(actual),
            measured_line_normal=n.tolist(),offset_px=component['offset_px'],actual_inlier_centroid=center.tolist(),on_line_support_anchor=anchor.tolist(),
            anchor_projection_distance_px=float(np.linalg.norm(anchor-center)),RMS_px=chain['inlier_rms_px'],span_px=component['actual_inlier_span_px'],
            proposal_grade=component['proposal_grade'],source_anchor_state=row['states'][k],mask_context_kind=component['mask_context_kind'],
            global_external_context=node['global_external_context'],selected_guide_id=guide_id,selected_chain_ref=ref,
            actual_points=actual.tolist(),projected_CAL_reference_edge=reference[[k,mate]].tolist(),initial_edge=initial[[k,mate]].tolist(),selected_generating_edge=guide[[k,mate]].tolist(),
            CAL_alignment_phase=alignment['phase'],CAL_geometry_gap_m=alignment['geometry_gap_m'],image=row['image'],mask_binding=row['mask_binding']))
        del row,observation
    C.write(OUTPUT/'DIAGNOSIS_DATA.json',dict(records=reports,selection_count=len(chosen),unique_groups=len(reports),CAL_only=True,current_runtime_or_models_changed=False))
    text='# v29 CAL 선 잡음 진단\n\n각 정책의 호환 CAL44에서 절대 법선 잔차 상위3개를 고정 규칙으로 선정했다. 두 정책의 같은 선은 한 그림으로 합쳤다. 정답 기반 선택은 CAL 진단에만 사용하며 runtime guide/관측·모델·파라미터는 바꾸지 않는다. 마젠타는 **CAL 진단 기준**이고 이미지 측정이나 추론 guide가 아니다.\n\n'
    text+='법선 잔차는 실제 support anchor에서 지정 native 3D edge의 **무한 투영선**까지의 거리다. 이미지 상 실제 모서리를 측정한 값이 아니다. anchor는 unweighted actual-inlier mean을 저장된 weighted-IRLS TLS 선에 정사영한 통계다. 유한 지지/방향 순서는 별도 guard다.\n\n'
    text+='| 사례 | 정책/순위 | edge/축 | normal(px) | 각도(°) | 실제 지지점 | RMS(px) | context |\n|---|---|---|---:|---:|---:|---:|---|\n'
    for item in reports:
        membership=', '.join(p['policy']+'#'+str(p['rank'])for p in item['policies'])
        text+=f'| {item["id"]} | {membership} | {item["native_edge"]}/{item["physical_axis"]} | {item["signed_normal_residual_px"]:.3f} | {item["angle_deg"]:.3f} | {item["actual_inliers"]} | {item["RMS_px"]:.4f} | {item["mask_context_kind"]} |\n'
    for item in reports:text+='\n## '+item['id']+' · edge '+str(item['native_edge'])+'\n\n!['+item['id']+']('+item['figure']+')\n'
    text+='\n실제 선 지지가 있다는 것과 해당 선이 지정된 cuboid 물리 변이라는 것은 다른 주장이다. 다른 평행 구조/슬랫·면 내부 경계·마스크 오류 가능성은 사진·mask와 함께 해석해야 하며 그림만으로 원인을 확정하지 않는다. 이 결과는 새 방법 채택이나 전체192 성능 개선 근거가 아니다.\n'
    (OUTPUT/'REPORT_KO.md').write_text(text)
    for key in ('source_sha256','input_sha256'):D._verify(declaration[key])
    D._verify(bound)
    C.write(OUTPUT/'RENDER_SEAL.json',dict(plan_sha256=C.sha(PLAN),source_sha256=declaration['source_sha256'],input_sha256={**declaration['input_sha256'],**bound},
        assets={p.name:C.sha(p)for p in OUTPUT.iterdir()if p.is_file()},new_posefit_NN_or_runtime_changes=0,primary_labels_decoded=0))
    print(dict(groups=len(reports),selection_count=len(chosen),output=str(OUTPUT),seal_sha256=C.sha(OUTPUT/'RENDER_SEAL.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','render'));args=parser.parse_args()
    declare()if args.stage=='declare'else render()
