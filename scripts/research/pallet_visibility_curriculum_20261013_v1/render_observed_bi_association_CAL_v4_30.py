"""Fixed CAL3 old/new actual-line overlays, using saved pixels and projections.

No new RGB measurement, native pose fit, reference decode, or image inference.
The magenta segment is an already-sealed CAL diagnostic projection only.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

from . import run_observed_bi_line_association_CAL_v4_30b as D

C=D.C
OUTPUT=D.B/'observed_bi_line_association_CAL_process_v4_30'
PLAN=D.B/'observed_bi_line_association_CAL_process_plan_v4_30/PREDECLARATION.json'


def measured_support_segment(node):
    n=np.asarray(node['normal'],float);t=np.array([-n[1],n[0]])
    actual=np.asarray(node['actual_inliers'],float);along=actual@t
    return np.asarray([-float(node['offset'])*n+v*t for v in (along.min(),along.max())])


def selected_actual_lines(graph):
    byname={n['name']:n for n in graph['line_candidates']}
    if len(byname)!=len(graph['line_candidates']):raise ValueError('Unique original node names required')
    result=[byname[name]for name in graph['selected_line_names']]
    if len({tuple(n['native_edge'])for n in result})!=len(result):raise ValueError('Physical edge once, not independent duplicates')
    return result


def verify_actual_origin(node,chain_observation):
    for ref in node['chain_references']:
        line=chain_observation['edge_records'][ref['ray_index']]['chains'][ref['chain_index']]
        actual=np.asarray(line['observed_points'])[np.asarray(line['inlier_mask'],bool)]
        normal=np.asarray(line['normal']);norm=np.linalg.norm(normal)
        if not(np.array_equal(actual,node['actual_inliers'])and np.array_equal(normal/norm,node['normal'])
                and float(line['offset'])/norm==node['offset']):raise ValueError('Saved original TLS/inliers must remain exact')
    return True


def declare():
    if PLAN.exists():raise FileExistsError('Preserve actual-overlay declaration')
    diagnostic=C.read(D.DIAGNOSIS/'DIAGNOSIS_DATA.json');D.fixed_cases([r['id']for r in diagnostic['records']])
    rerank=C.read(D.OUTPUT/'RERANK_SEAL.json');parent=C.read(D.PLAN)
    D.R._verify(rerank['source_sha256']);D.R._verify(rerank['input_sha256'])
    paths=[D.OUTPUT/'RERANK_SEAL.json',D.OUTPUT/'RESULTS.json',D.DIAGNOSIS/'DIAGNOSIS_DATA.json',D.PLAN]
    paths.extend(D.OUTPUT/relative for relative in rerank['assets'])
    for r in diagnostic['records']:paths.extend((C.ROOT/r['image'],C.ROOT/r['mask_binding']['path']))
    paths.extend(C.ROOT/c['spool_relative']for c in parent['cases'])
    sources={str(Path(m.__file__).resolve().relative_to(C.ROOT)):C.sha(m.__file__)for m in(D,D.OBS,D.R,C)}
    for path in(Path(__file__),Path(__file__).with_name('test_render_observed_bi_association_CAL_v4_30.py')):
        sources[str(path.resolve().relative_to(C.ROOT))]=C.sha(path)
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,dict(schema='V30_FIXED_CAL3_CACHED_ACTUAL_OLD_NEW_OVERLAY',UTC=datetime.now(timezone.utc).isoformat(),
        cases=list(D.CASES),source_sha256=sources,input_sha256={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths},
        protected_user_sha256=parent['protected_user_sha256'],fixed_case_selection='same prior CAL diagnostic top3, no new case or success filtering',
        panels='original RGB with saved old selected lines versus saved new selected lines; same mask contour; exact saved named-edge CAL diagnostic projection',
        CAL_reference='already-sealed top3 edge projection only, post-ranking diagnostic; never new label decode, runtime guide or factor',
        draw='actual finite TLS support and original actual inlier pixels; no synthetic bridge, precise corner, S/U point or full pose wire',
        image_brightness_unchanged=True,old_V29_models_or_observations_changed=False,
        new_RGB_measurement_NN_label_decode_or_posefit=0,explicit_root_render_GO_required=True))
    C.write(PLAN.parent/'PLAN_SEAL.json',dict(plan_sha256=C.sha(PLAN)))
    for relative,digest in sources.items():
        target=PLAN.parent/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(target)!=digest:raise ValueError('Exact render source snapshot required')
    print(dict(plan_sha256=C.sha(PLAN),cases=3),flush=True)


def render(*,root_actual_render_GO=False):
    if root_actual_render_GO is not True:raise PermissionError('Root saved-image render GO before IO')
    if OUTPUT.exists():raise FileExistsError('Preserve any actual render attempt')
    plan=C.read(PLAN)
    if C.sha(PLAN)!=C.read(PLAN.parent/'PLAN_SEAL.json')['plan_sha256']:raise ValueError('Exact overlay plan required')
    for key in('source_sha256','input_sha256','protected_user_sha256'):D.R._verify(plan[key])
    diagnostics=C.read(D.DIAGNOSIS/'DIAGNOSIS_DATA.json')['records'];D.fixed_cases([r['id']for r in diagnostics])
    summaries=C.read(D.OUTPUT/'RESULTS.json')['cases'];parents=C.read(D.PLAN)['cases'];OUTPUT.mkdir(parents=True)
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports=[];checks=0
    for i,(diag,summary,parent)in enumerate(zip(diagnostics,summaries,parents)):
        if not diag['id']==summary['id']==parent['id']:raise ValueError('Exact fixed CAL ordering required')
        row=D.OBS._read_single(C.ROOT/parent['spool_relative'],parent)
        bundles=list(C.rows(D.OUTPUT/'graphs'/('case'+str(i)+'.jsonl.gz')))
        if len(bundles)!=1:raise ValueError('One saved rerank bundle only')
        old=row['observation']['joint_graph'];new=bundles[0]['graph']
        if old['selected_line_names']!=summary['old_selected']or new['selected_line_names']!=summary['new_selected']:raise ValueError('Exact selected lines required')
        image_path=C.ROOT/diag['image'];mask_path=C.ROOT/diag['mask_binding']['path']
        if C.sha(image_path)!=row['image_sha256']or C.sha(mask_path)!=diag['mask_binding']['sha256']:raise ValueError('Unchanged original image/mask')
        image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
        with np.load(mask_path,allow_pickle=False)as data:mask=data['pallet_mask_clean'].copy()
        if image.shape[:2]!=mask.shape:raise ValueError('Actual image/mask dimensions required')
        lines=[selected_actual_lines(old),selected_actual_lines(new)]
        for group in lines:
            for node in group:verify_actual_origin(node,row['observation']['chain_observation']);checks+=1
        reference=np.asarray(diag['projected_CAL_reference_edge']);edge=sorted(diag['native_edge'])
        fig,axes=plt.subplots(1,2,figsize=(16,6));h,w=image.shape[:2]
        for j,(ax,group)in enumerate(zip(axes,lines)):
            ax.imshow(image);ax.contour(mask.astype(float),levels=[.5],colors=['white'],linewidths=.45,alpha=.6)
            ax.plot(reference[:,0],reference[:,1],'--',color='#df20c9',lw=1.8,label='fixed CAL diagnostic edge (not measurement)')
            for node in group:
                relevant=node['native_edge']==edge;color=('#ff9824'if j==0 else'#31e874')if relevant else'#00c5e6'
                actual=np.asarray(node['actual_inliers']);support=measured_support_segment(node)
                ax.plot(support[:,0],support[:,1],color=color,lw=2.4 if relevant else 1.,alpha=1. if relevant else .6)
                ax.scatter(actual[:,0],actual[:,1],s=9 if relevant else 2.,color=color,alpha=.95 if relevant else .45)
                mid=support.mean(0);ax.text(*mid,'-'.join(map(str,node['native_edge'])),color='black',fontsize=8,
                    bbox=dict(facecolor=color,alpha=.8,edgecolor='none',pad=1))
            details=summary['old_top_named_edge_fixed_CAL_reference']if j==0 else summary['new_top_named_edge_fixed_CAL_reference']
            value='named edge NOT SELECTED'if not details else f"named edge signed CAL normal {details[0]['signed_normal_residual_px']:.3f}px | angle {details[0]['angle_deg']:.3f}deg"
            ax.set_title(('Stored V29 selected actual lines'if j==0 else'NEW V30 selected actual lines')+'\n'+value,fontsize=11)
            ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_aspect('equal');ax.set_xlabel('original image x');ax.set_ylabel('original image y')
            ax.legend(loc='lower left',fontsize=7)
        fig.suptitle(f"{diag['id']} | named edge {diag['native_edge']} | cyan: other selected actual supports; orange/green: same named edge",fontsize=11)
        fig.text(.5,.015,'Unchanged RGB brightness / white: saved target mask contour / actual finite supports only / no precise KP, new fit or adoption claim',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.035,1,.94]);asset='case'+str(i)+'_old_new_actual.png';fig.savefig(OUTPUT/asset,dpi=160);plt.close(fig)
        def thin(node):return {key:node[key]for key in('name','native_edge','normal','offset','actual_inliers','proposal_grade','chain_references')}
        reports.append(dict(id=diag['id'],figure=asset,image_binding=dict(path=diag['image'],sha256=row['image_sha256']),mask_binding=diag['mask_binding'],
            old_selected=[thin(n)for n in lines[0]],new_selected=[thin(n)for n in lines[1]],
            projected_fixed_CAL_diagnostic_edge=diag['projected_CAL_reference_edge'],native_edge=diag['native_edge'],
            exact_summary=summary,reference_was_used_for_ranking=False,S_U_point_markers=0,
            full_pose_wire_or_precise_corner_adoption=0,raw_actual_support_and_TLS_exact=True))
        del row,bundles,old,new,image,mask
    C.write(OUTPUT/'PROCESS_DATA.json',dict(cases=reports,exact_selected_original_TLS_checks=checks,new_measurement_or_reference_decode=0))
    text='# V30: 원 B/I 점과 실제 선의 대응 gate, 고정 CAL3 사진\n\n기존 실제 선 후보만 다시 순위화했다. 43px는 기존 normal search reach이며, named native 변의 모든 eligible B/I q에서 무한 TLS 선까지 적용한다. 선을 따라 이동하는 거리는 막지 않는다. 초기 S/U 좌표 조회·새 영상 측정·라벨 해독·포즈 적합은 없다. V29 관측·모델은 바꾸지 않았다.\n\n'
    text+='청록은 각 방법이 선택한 다른 실제 선 지지점, 주황/초록은 진단 대상 native 변의 원/새 실제 선이다. 마젠타는 이미 봉인된 **CAL 진단 기준의 해당 변 투영**이며 새 측정이나 순위화 guide가 아니다. 흰 윤곽은 저장 target mask이며 실제 occluder 정답이 아니다. 모든 실제 지지점과 TLS 계수는 원 캐시와 정확히 같음을 대조했다.\n\n'
    text+='| CAL 사례 | unique 후보 원→새 | raw 후보 gate 제외 | 선택 선 원→새 | 해당 변 CAL normal 원→새(px) | 각도 원→새(°) |\n|---|---:|---:|---:|---:|---:|\n'
    for report in reports:
        r=report['exact_summary'];a=r['old_top_named_edge_fixed_CAL_reference'][0];b=r['new_top_named_edge_fixed_CAL_reference']
        normal=f"{a['signed_normal_residual_px']:.3f} → "+(f"{b[0]['signed_normal_residual_px']:.3f}"if b else'미선택')
        angle=f"{a['angle_deg']:.3f} → "+(f"{b[0]['angle_deg']:.3f}"if b else'미선택')
        text+=f"| {r['id']} | {r['old_candidates']}→{r['new_candidates']} | {r['rejected']} | {len(r['old_selected'])}→{len(r['new_selected'])} | {normal} | {angle} |\n"
    for r in reports:text+='\n## '+r['id']+'\n\n!['+r['id']+']('+r['figure']+')\n'
    text+='\n두 목재 사례는 같은 native 변에서 기준에 가까운 새 실제 선을 선택했다. CAD 사례는 문제가 된 변을 **미선택**으로 남겼으며 정밀 복구 성공으로 계산하지 않는다. 세 사례 모두 전체 후보0/전체 선택0은 아니고 joint 대안8개를 보존했다. raw gate 제외 수에는 같은 물리 지지의 여러 원 후보가 포함되므로 unique 후보 감소와 다르다. 모든 새 선택선의 original q/source/양끝 법선 거리와 미선택 장부는 [원 재순위 RESULTS](../observed_bi_line_association_CAL_cases_v4_30/RESULTS.json)에 있다.\n\n'
    text+='원래 큰 CAL 잔차로 선택된 3개 진단 사례이므로 모집단 대표 성능을 뜻하지 않는다. gate 통과나 이 그림만으로 native 정체의 정확성·전체192 개선·관측 독립성·방법 채택을 주장하지 않는다. 별도 CAL46 population/noise 및 uniform 전체 평가가 필요하며 V29 σ/γ는 새 선택 집단의 검증 모델로 재사용하지 않는다.\n'
    (OUTPUT/'REPORT_KO.md').write_text(text)
    for key in('source_sha256','input_sha256','protected_user_sha256'):D.R._verify(plan[key])
    C.write(OUTPUT/'RENDER_SEAL.json',dict(plan_sha256=C.sha(PLAN),source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],
        assets={p.name:C.sha(p)for p in OUTPUT.iterdir()if p.is_file()},actual_TLS_origin_checks=checks,
        new_measurements_NN_labels_posefit_KP_adoption=0,current_V29_unchanged=True))
    print(dict(cases=3,actual_TLS_origin_checks=checks,seal_sha256=C.sha(OUTPUT/'RENDER_SEAL.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','render'))
    parser.add_argument('--root-actual-render-GO',action='store_true');args=parser.parse_args()
    declare()if args.stage=='declare'else render(root_actual_render_GO=args.root_actual_render_GO)
