"""Frozen postseal covariance-budget sensitivity; no solve, fitting, or tuning."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[7]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import joint_ot_jacobian_v2 as A

OUT=Path(__file__).resolve().parent
STUDY=OUT.parent.parent
RUN=STUDY/'evaluation_v3'
AUDIT=OUT.parent/'failure_audit/FAILURE_AUDIT.json'
THRESHOLDS=(3.,4.,6.,9.,12.)
FAILURES=('eval_cad:1778653018878592768','eval_night08:1779449501478488320',
          'eval_pallet07:1778652170735118080','wood_183705:000706')


def distribution(values):
    x=np.asarray(values,float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,
                P90=float(np.quantile(x,.9)) if len(x) else None,max=float(x.max()) if len(x) else None,
                empirical_fraction_le={str(int(t)):float(np.mean(x<=t*np.sqrt(1+1e-7))) if len(x) else None for t in THRESHOLDS})


def singulars(matrix,scale):
    u,s,v=np.linalg.svd(matrix,full_matrices=True)
    tolerance=(s[0]*max(matrix.shape)*np.finfo(float).eps) if len(s) else 0
    rank=int(np.sum(s>tolerance)); full=rank==6
    last=float(s[-1]) if len(s) else 0
    vector=v[-1].copy()
    return dict(shape=list(matrix.shape),singular_values=s.tolist(),rank_machine_epsilon=rank,nullity_6D=6-rank,
        condition_nonzero=float(s[0]/s[rank-1]) if rank else None,
        condition_full_6D=float(s[0]/last) if full else None,rank_deficient=not full,
        weakest_right_singular_vector=vector.tolist(),
        local_unit_residual_weakest_rotation_amplification_deg=float(np.linalg.norm(vector[:3])/last*180/np.pi) if full else None,
        local_unit_residual_weakest_translation_amplification_cm=float(scale*np.linalg.norm(vector[3:])/last*100) if full else None,
        amplification_is_local_linear_response_not_calibrated_uncertainty=True)


def jacobians(row,alignment):
    rt=row['runtime'];xyz=np.asarray(rt['initial_geometry']['native_xyz']);K=np.asarray(row['K']);visible=rt['frozen_visible_measurements']
    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
    result={}
    poses={'INITIAL_COARSE':(row['initial_pose']['R_cf'],row['initial_pose']['centroid']),
           'CURRENT_SAVED':(rt['actual_pose']['R_cf'],rt['actual_pose']['centroid']),
           'REGISTERED_DERIVED_REFERENCE':(alignment['registered_reference_R_cf'],alignment['registered_reference_centroid'])}
    for stage,(R,t) in poses.items():
        p=np.r_[Rotation.from_matrix(R).as_rotvec(),np.asarray(t)/scale]
        residual,J=A.visible_residual_and_jacobian(p,xyz,K,visible,scale)
        q,depth,pixelJ,_=A.projection_and_jacobian(p,xyz,K,scale)
        pixel=np.concatenate([pixelJ[m['native_corner']] for m in visible]) if visible else np.empty((0,6))
        result[stage]=dict(covariance_whitened=singulars(J,scale),unwhitened_native_pixel=singulars(pixel,scale),
                          max_whitened_point_residual=float(np.max(np.linalg.norm(residual.reshape(-1,2),axis=1))) if len(residual) else None)
    return dict(BI_count=len(visible),native_ids=[m['native_corner'] for m in visible],translation_scale_m=scale,
                parameterization='source solver Rodrigues rotation-vector radians; translation/fixed native cuboid diagonal',
                whitened_source='joint_ot_jacobian_v2.visible_residual_and_jacobian',stages=result,
                local_condition_does_not_resolve_discrete_P3P_branches=True)


def curves(points,frames):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for part,color in [('CAL','#3366bb'),('TEST','#c06a19'),('ALL','#333333')]:
        pp=[p['reference_mahalanobis'] for p in points if part=='ALL' or p['partition']==part]
        ff=[r['max_reference_mahalanobis'] for r in frames if part=='ALL' or r['partition']==part]
        for ax,x in zip(axes,(pp,ff)):
            x=np.sort(x);ax.step(x,100*np.arange(1,len(x)+1)/len(x),where='post',color=color,label=part+f' n={len(x)}')
    for ax,title in zip(axes,('Point residual empirical CDF','Frame maximum residual empirical CDF')):
        ax.set_xscale('log');ax.set_xlim(.05,500);ax.set_ylim(0,101);ax.set_xlabel('Mahalanobis distance · frozen covariance');ax.set_ylabel('Empirical fraction (%)')
        for threshold in THRESHOLDS:ax.axvline(threshold,ls=':',lw=.7,color='#aaa')
        ax.axvline(3,color='#aa3333',lw=1.3);ax.text(3.2,8,'current cap 3',rotation=90,color='#aa3333',fontsize=8)
        ax.set_title(title);ax.grid(alpha=.15);ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('Compatible primary 86 · derived reference pose vs frozen B/I measurements',fontsize=11)
    fig.text(.01,.01,'Postseal sensitivity only. No confidence interval, physical coverage guarantee, best TEST cap selection, or new PnP solve.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(OUT/'COVERAGE_CURVE.png',dpi=220);fig.savefig(OUT/'COVERAGE_CURVE.svg');plt.close(fig)
    direct=[p for p in points if p['human_source_state']=='DIRECT_VISIBLE' and p['annotation_mahalanobis'] is not None]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for applied,color in [(False,'#4b6b90'),(True,'#d16d30')]:
        group=[p for p in direct if p['correction_applied']==applied]
        axes[0].scatter([p['annotation_mahalanobis'] for p in group],[p['reference_mahalanobis'] for p in group],s=15,alpha=.55,color=color,label=('corrected' if applied else 'fallback')+f' n={len(group)}')
    axes[0].plot([.01,400],[.01,400],color='#888',lw=.8);axes[0].axvline(3,color='#aaa',ls=':');axes[0].axhline(3,color='#aaa',ls=':')
    axes[0].set_xscale('log');axes[0].set_yscale('log');axes[0].set_xlim(.01,400);axes[0].set_ylim(.01,400)
    axes[0].set_xlabel('Native human DIRECT annotation Mahalanobis');axes[0].set_ylabel('Derived reference projection Mahalanobis');axes[0].legend(fontsize=8);axes[0].grid(alpha=.15)
    for key,color,label in [('annotation_mahalanobis','#26805a','human coordinate'),('reference_mahalanobis','#914c9a','derived pose projection')]:
        x=np.sort([p[key] for p in direct]);axes[1].step(x,100*np.arange(1,len(x)+1)/len(x),where='post',color=color,label=label)
    axes[1].set_xscale('log');axes[1].set_xlim(.03,400);axes[1].set_ylim(0,101);axes[1].axvline(3,color='#aa3333');axes[1].set_xlabel('Mahalanobis distance');axes[1].set_ylabel('Empirical fraction (%)');axes[1].legend(fontsize=9);axes[1].grid(alpha=.15)
    fig.suptitle('Same DIRECT points/covariances · annotation and historical derived pose differ',fontsize=11)
    fig.text(.01,.01,'Neither target supplies independently measured physical uncertainty. Conditional covariance does not certify semantic/model correctness.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(OUT/'DIRECT_ANNOTATION_VS_REFERENCE_MAH.png',dpi=220);fig.savefig(OUT/'DIRECT_ANNOTATION_VS_REFERENCE_MAH.svg');plt.close(fig)


def main():
    if (OUT/'UNCERTAINTY_DIAGNOSTIC.json').exists():raise FileExistsError('Preserve completed diagnostic')
    source_protocol=C.read(RUN/'PROTOCOL.json');sources=source_protocol['source_sha256']
    assert all(C.sha(ROOT/path)==sha for path,sha in sources.items())
    current=C.read(RUN/'PREDICTION_SEAL.json');assert C.sha(RUN/'PREDICTIONS.jsonl.gz')==current['predictions_sha256']
    records=list(C.rows(RUN/'PREDICTIONS.jsonl.gz'));by={r['id']:r for r in records}
    audit=C.read(AUDIT);eligible=[r for r in audit['rows'] if r['cohort']=='PRIMARY_NONSEVERE' and r['alignment']['compatible']];assert len(eligible)==86
    protocol=dict(schema='FROZEN_POSTSEAL_UNCERTAINTY_DIAGNOSTIC_PROTOCOL_V1',thresholds=list(THRESHOLDS),compatible_primary_frames=86,
        source_sha256=sources,diagnostic_source_sha256=C.sha(__file__),input_audit_sha256=C.sha(AUDIT),prediction_cache_sha256=C.sha(RUN/'PREDICTIONS.jsonl.gz'),
        no_model_or_frozen_source_change=True,no_new_PnP_fit=True,no_TEST_parameter_selection=True,thresholds_are_cause_sensitivity_only=True,
        no_bootstrap_CI_or_probability_guarantee=True,geometry_canvas_depth_separate_from_covariance_cap=True,
        metric_target_is_historical_derived_reference_not_independent_physical_truth=True)
    C.write(OUT/'PROTOCOL.json',protocol)
    points=[];frames=[]
    for r in eligible:
        original=by[r['id']];corrections={v['native_corner']:v for v in original['runtime']['visible_correction']['records']}
        pp=[]
        for m in r['BI_measurements']:
            k=m['native_id'];ref=m['registered_reference_pose_support'];annotation=m['annotation_measurement_budget'];meta=corrections[k]
            covariance=np.asarray(meta['covariance_px2']);eigen=np.linalg.eigvalsh(covariance)
            p=dict(frame_id=r['id'],partition=r['partition'],native_id=k,state=m['state'],human_source_state=m['human_source_state'],
                   correction_applied=bool(meta['corrected']),source=m['source'],reference_mahalanobis=ref['mahalanobis'],
                   annotation_mahalanobis=annotation['mahalanobis'] if annotation else None,
                   reference_minus_annotation_mahalanobis=ref['mahalanobis']-annotation['mahalanobis'] if annotation else None,
                   reference_geometry_pass=ref['geometric_visibility_pass'],reference_canvas_pass=ref['inside_closed_canvas'],
                   covariance_eigenvalues_px2=eigen.tolist(),covariance_largest_sigma_px=float(np.sqrt(eigen[-1])),
                   blend_alpha=meta.get('blend_alpha'),line_conditional_largest_sigma_px=meta.get('conditional_covariance',{}).get('largest_std_px'))
            points.append(p);pp.append(p)
        caps={str(int(t)):all(p['reference_mahalanobis']<=t*np.sqrt(1+1e-7) for p in pp) for t in THRESHOLDS}
        frames.append(dict(id=r['id'],partition=r['partition'],BI_count=len(pp),OT_count=r['OT_count'],
            max_reference_mahalanobis=max((p['reference_mahalanobis'] for p in pp),default=0),
            reference_BI_covariance_caps_pass=caps,reference_OT_pass=r['derived_reference_pose_OT_support_all'],
            reference_geometry_pass=all(p['reference_geometry_pass'] for p in pp),reference_canvas_pass=all(p['reference_canvas_pass'] for p in pp),
            reference_depth_pass=min(r['alignment']['reference_depth'])>=1e-5,
            initial_current_metrics=r['metrics'],source_semantic_mismatches=r['semantic_mismatches'],jacobians=jacobians(original,r['alignment'])))
    tables={};groups={}
    for part in ('CAL','TEST','ALL'):
        rows=[r for r in frames if part=='ALL' or r['partition']==part]
        pp=[p for p in points if part=='ALL' or p['partition']==part]
        table=[]
        for t in THRESHOLDS:
            key=str(int(t));both=BIonly=OTonly=neither=full=0
            for r in rows:
                bi=r['reference_BI_covariance_caps_pass'][key];ot=r['reference_OT_pass']
                both+=int(bi and ot);BIonly+=int(bi and not ot);OTonly+=int(ot and not bi);neither+=int(not bi and not ot)
                full+=int(bi and ot and r['reference_geometry_pass'] and r['reference_canvas_pass'] and r['reference_depth_pass'])
            table.append(dict(threshold=t,BI_covariance_and_OT_pass=both,only_BI_covariance_pass=BIonly,only_OT_pass=OTonly,neither_pass=neither,all_constraints_including_geometry_canvas_depth_pass=full))
        tables[part]=dict(frames=len(rows),points=len(pp),frames_with_OT=sum(r['OT_count']>0 for r in rows),frames_without_OT_vacuously_pass_OT=sum(r['OT_count']==0 for r in rows),
                         geometry_pass=sum(r['reference_geometry_pass'] for r in rows),canvas_pass=sum(r['reference_canvas_pass'] for r in rows),depth_pass=sum(r['reference_depth_pass'] for r in rows),threshold_table=table,
                         frame_max_Mah=distribution([r['max_reference_mahalanobis'] for r in rows]),point_reference_Mah=distribution([p['reference_mahalanobis'] for p in pp]))
        grouped={}
        for group in ('B','I','CORRECTED','FALLBACK','HUMAN_DIRECT','HUMAN_SELF','HUMAN_EXTERNAL','HUMAN_UNANNOTATED'):
            selected=[p for p in pp if p['state']==group or group=='CORRECTED' and p['correction_applied'] or group=='FALLBACK' and not p['correction_applied'] or group=='HUMAN_DIRECT' and p['human_source_state']=='DIRECT_VISIBLE' or group=='HUMAN_SELF' and p['human_source_state']=='SELF_OCCLUDED' or group=='HUMAN_EXTERNAL' and p['human_source_state']=='EXTERNAL_OCCLUDED' or group=='HUMAN_UNANNOTATED' and p['human_source_state']=='UNANNOTATED']
            grouped[group]=dict(reference_Mah=distribution([p['reference_mahalanobis'] for p in selected]),annotation_Mah=distribution([p['annotation_mahalanobis'] for p in selected if p['annotation_mahalanobis'] is not None]),
                               largest_sigma_px=distribution([p['covariance_largest_sigma_px'] for p in selected]),reference_minus_annotation_Mah=distribution([p['reference_minus_annotation_mahalanobis'] for p in selected if p['reference_minus_annotation_mahalanobis'] is not None]))
        groups[part]=grouped
    findings=dict(fallback_B_sigma_px=2,fallback_I_sigma_px=3,correction_detector_sigma_assumed_px=2,
        conditional_line_resamples=64,moving_block_length=3,normal_offset_floor_px=1,maximum_blend_alpha=.5,
        covariance_expression='(1-alpha)^2 * 4 I + alpha^2 * conditional_line_intersection_covariance + I',
        alpha_expression='min(0.5, 4/(4+max_eigenvalue(conditional_covariance)))',
        absent_explicit_error_components=['semantic corner/model bias','3D cuboid/rounded physical corner mismatch','systematic detector bias','camera/registration mismatch','raw-detector / line correlation','cross-keypoint shared-line covariance'],
        source_provenance_conditional_covariance_is_calibrated=False,
        source_locations=[dict(path='scripts/research/pallet_visibility_curriculum_20261013_v1/visible_bi_correction_v2.py',lines=[23,27,154,161,164,165,167,181]),dict(path='scripts/research/pallet_foundation_mask_20261013_v1/uncertainty.py',lines=[1,3,33,52,70,75]),dict(path='scripts/research/pallet_visibility_curriculum_20261013_v1/joint_ot_jacobian_v2.py',lines=[83,89,97,98])],
        Mahalanobis_3_is_not_calibrated_physical_99percent_proof=True)
    C.write(OUT/'SOURCE_FACTS.json',findings)
    C.write(OUT/'UNCERTAINTY_DIAGNOSTIC.json',dict(schema='POSTSEAL_FROZEN_BI_UNCERTAINTY_SENSITIVITY_V1',threshold_tables=tables,point_groups=groups,points=points,frames=frames,source_facts=findings,
        caveats=['No cap selected from TEST; thresholds only deterministic constraint sensitivity','Historical derived reference pose and native annotation differ; neither independently measures physical pose uncertainty','Mahalanobis empirical CDF is not a confidence interval or future coverage guarantee','Jacobian uses exactly frozen normalized6D source parameterization; condition is local and does not diagnose discrete P3P branch choice alone']))
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf');plt.rcParams['font.family']=font_manager.FontProperties(fname='/usr/share/fonts/truetype/nanum/NanumGothic.ttf').get_name();plt.rcParams['axes.unicode_minus']=False
    curves(points,frames);readme(tables,groups,frames)
    assert all(C.sha(ROOT/path)==sha for path,sha in sources.items()) and C.sha(AUDIT)==protocol['input_audit_sha256'] and C.sha(RUN/'PREDICTIONS.jsonl.gz')==protocol['prediction_cache_sha256']
    C.write(OUT/'DIAGNOSTIC_SEAL.json',dict(schema='POSTSEAL_UNCERTAINTY_DIAGNOSTIC_SEAL_V1',compatible_primary_frames=len(frames),points=len(points),protocol_sha256=C.sha(OUT/'PROTOCOL.json'),results_sha256=C.sha(OUT/'UNCERTAINTY_DIAGNOSTIC.json'),frozen19_source_hashes_unchanged=True,frozen_failure_audit_unchanged=True,prediction_cache_unchanged=True,new_PnP_fits=0,bootstrap_CI_or_calibration_claim=False,output_files={p.name:C.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='DIAGNOSTIC_SEAL.json'}))
    print(C.finite(tables));print('3BI_FAILURE_JACOBIANS',[(r['id'],r['jacobians']['stages']['CURRENT_SAVED']['covariance_whitened']) for r in frames if r['id'] in FAILURES])


def readme(tables,groups,frames):
    lines=['# Frozen B/I covariance budget 사후 민감도','',
        '현재 방식의 모델 대응이 호환되는 primary 86프레임(CAL44/TEST42)만 검사했습니다. 앞서 봉인한 failure_audit와 source19/모델/예측은 변경하지 않았습니다. 모든 수치는 사후 원인 민감도이며 TEST 기준으로 cap을 선택하거나 새 PnP를 fit하지 않았습니다. 경험 CDF는 신뢰구간, bootstrap CI, 미래 물리 coverage 보증이 아닙니다.', '',
        '현재 cap3의 Mahalanobis budget과 O/T 영역 안에 역사적 derived reference pose가 들어가는지 검사했습니다. 아래 BI는 covariance budget만 뜻합니다. geometry/canvas/depth는 따로 집계한 뒤 마지막 열에서 함께 적용합니다. O/T가 없는 프레임도 OT 통과에 포함되는 자명한 경우를 JSON에 별도 표기했습니다.', '',
        '| 분할 | cap | BI·OT 둘 다 | BI만 | OT만 | 둘 다 실패 | geometry/canvas/depth 포함 전체 통과 |', '|---|---:|---:|---:|---:|---:|---:|']
    for part in ('CAL','TEST','ALL'):
        for r in tables[part]['threshold_table']:
            lines.append(f"| {part} | {r['threshold']:g} | {r['BI_covariance_and_OT_pass']} | {r['only_BI_covariance_pass']} | {r['only_OT_pass']} | {r['neither_pass']} | {r['all_constraints_including_geometry_canvas_depth_pass']} |")
    lines+=['','별도 geometry/canvas/depth 통과: '+', '.join(f"{part} {s['geometry_pass']}/{s['frames']}, {s['canvas_pass']}/{s['frames']}, {s['depth_pass']}/{s['frames']}" for part,s in tables.items() if part!='ALL')+'.', '',
        '![Empirical coverage curve](COVERAGE_CURVE.png)','',
        '실제 covariance는 fallback B σ=2px, I σ=3px를 가정합니다. 보정된 점은 retained image-line의 moving-block resampling 64회와 normal offset 1px floor로 conditional line-intersection Σ를 얻고, α=min(.5,4/(4+λmax(Σconditional))), Σfinal=(1−α)^2·4I+α²Σconditional+I를 씁니다. semantic/model/systematic 오차 항이나 detector–line 상관, 여러 corner가 공유하는 line의 joint covariance는 현재 식에 없습니다. source provenance는 conditional_covariance_is_calibrated=False를 명시합니다.', '',
        '따라서 σ3/Mah≤3는 조건부 입력 budget 통과이지 물리 정답을 99% 포함한다는 증명이 아닙니다. q의 평균 pixel error가 작아져도 작은 Σ로 나누면 reference pose가 budget 밖에 있을 수 있습니다. 주석과 그로부터 얻은 derived pose projection도 서로 다릅니다.', '',
        '| 전체 point group | N | Reference Mah 평균 / P90 | cap3 point fraction |', '|---|---:|---:|---:|']
    for name,v in groups['ALL'].items():
        d=v['reference_Mah']
        if d['n']:lines.append(f"| {name} | {d['n']} | {d['mean']:.3f} / {d['P90']:.3f} | {100*d['empirical_fraction_le']['3']:.1f}% |")
    lines+=['','![Annotation vs derived reference](DIRECT_ANNOTATION_VS_REFERENCE_MAH.png)', '',
        '3개 B/I의 baseline-success→current-failure 사례에서 frozen normalized6D Jacobian을 평가했습니다. parameter는 Rodrigues rotation-vector(rad), translation/native cuboid diagonal입니다. Singular value는 source와 같은 per-point covariance whitening을 사용합니다. 같은 residual 0인 서로 다른 P3P branch가 있을 수 있으므로 local condition만으로 global branch 선택을 설명하거나 인과 효과를 확정하지 않았습니다.', '',
        '| 사례 | Current J singular values (큰→작은) | condition | 최소 singular 방향의 unit whitened residual translation response cm |', '|---|---|---:|---:|']
    for r in frames:
        if r['id'] in FAILURES:
            j=r['jacobians']['stages']['CURRENT_SAVED']['covariance_whitened']
            lines.append('| '+r['id']+' | '+', '.join('%.4g'%s for s in j['singular_values'])+f" | {j['condition_full_6D']:.2f} | {j['local_unit_residual_weakest_translation_amplification_cm']:.2f} |")
    lines+=['','위 response는 unit whitened residual의 **국소 선형 반응**입니다. 확률적 pose 표준편차나 물리 error bound가 아닙니다. 초기·current·registered reference의 whitened/unwhitened J와 최소 singular 방향을 모두 JSON에 보존했습니다. BI2 프레임은 6D nullity가 남는 것도 명시합니다.', '',
        '[전체 JSON](UNCERTAINTY_DIAGNOSTIC.json) · [Source 근거](SOURCE_FACTS.json) · [Protocol](PROTOCOL.json) · [봉인](DIAGNOSTIC_SEAL.json)']
    (OUT/'README_KO.md').write_text('\n'.join(lines)+'\n')
    (OUT/'index.html').write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><title>Frozen covariance sensitivity</title><style>body{font:16px sans-serif;max-width:1250px;margin:25px auto;line-height:1.5}img{width:100%}</style><h1>B/I covariance · 봉인 후 원인 민감도</h1><p>3/4/6/9/12는 민감도 집계입니다. 물리 coverage·신뢰구간·TEST cap 선정 결과가 아닙니다.</p><p><a href='README_KO.md'>한국어 보고서</a> · <a href='UNCERTAINTY_DIAGNOSTIC.json'>전체 JSON</a></p><img src='COVERAGE_CURVE.png'><img src='DIRECT_ANNOTATION_VS_REFERENCE_MAH.png'></html>\n")


if __name__=='__main__':main()
