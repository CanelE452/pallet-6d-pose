"""Postscore explanation of fixed worst-case rules; no numerical pose solve.

Only sealed positive packets, final five-arm predictions/scores, original RGB
and those same fixed evaluation references are read. References appear in the
last separate panel and never guide a query, fit or pose. No missing trajectory
or intermediate PnP/optimization result is reconstructed.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np

from . import common as C
from . import score_clean_fresh_BI_poses_v4_32 as SCORE
from . import render_clean_fresh_BI_actual_runtime_process_CAL_v4_32_v2 as DRAW

M=SCORE.M
OUTPUT=M.B/'clean_fresh_BI_full_result_process_v4_32_r2_v1'
PLAN=M.B/'clean_fresh_BI_full_result_process_plan_v4_32_r2_v1/PREDECLARATION.json'
PRED_SHA='eff80efdb1cb9c17f7eb13df50b953467f82bbe3415c8385b9aa146456948b48'
SCORE_SHA='7bcc9bce3e8b14a87021955c513e3061dcb561cc309fabd61964e06efbcc6c80'
CENTRAL_SHA='c95598253d42f06ace0083d30b5fb46adbb64191fa59ee5b76ce348ef2be10ef'
SOURCE_ARM='CLEAN_FRESH_BI_SOURCE_CONTAINMENT'
COLORS=('#b979e8','#00c7c7','#ffd43b','#39b9ff','#ff8a47')


def choose_cases(primary,test):
    """Postscore illustration only. First score-row order breaks equal maxima."""
    selected={}
    for population,report,arms in(('PRIMARY',primary,M.ARMS[1:]),('TEST',test,(SOURCE_ARM,))):
        if report['count']!=(192 if population=='PRIMARY'else 49):raise ValueError('Full fixed population required')
        for arm in arms:
            def error(row):
                metric=row['metrics'][arm]
                if not metric['pose_available']:return float('inf')
                value=metric['translation_cm']
                if value is None or not np.isfinite(value):raise ValueError('Finite available pose must have scored translation')
                return value
            worst=max(report['rows'],key=error)
            if worst['id']!=report['summary'][arm]['max_translation_id']:raise ValueError('Independent max ID and score summary differ')
            selected.setdefault(worst['id'],[]).append(dict(population=population,arm=arm,
                rule='MAX_TRANSLATION_FIRST_CANONICAL_TIE',translation_cm=worst['metrics'][arm]['translation_cm'],
                missing_pose_is_infinite_failure=not worst['metrics'][arm]['pose_available']))
    return [dict(id=ident,reasons=reasons)for ident,reasons in selected.items()]


def archived_BI(packet):
    source=packet['clean_input'];rows=[]
    for record in source['visible_measurements']:
        k=record['native_corner']
        if source['initial_self_hidden'][k]or source['states'][k]not in('B','I'):continue
        if record['state']!=source['states'][k]:raise ValueError('Archived measurement/state differs')
        rows.append(deepcopy(record))
    return rows


def canonical_model_xyz(packet):
    source=packet['clean_input'];xyz=np.asarray(source['native_xyz']);Q=np.asarray(source['physical_registration_Q'])
    # Center/orientation transport of the physical body only. Final panels do
    # not claim identical native-channel correspondences between WD hypotheses.
    return (Q.T@(xyz-xyz.mean(0)).T).T


def final_projection(packet,entry,arm,legacy):
    pose=SCORE.prediction_pose(entry,arm=arm,legacy_semantics=legacy)
    if pose is None:return None
    return C.project(canonical_model_xyz(packet),np.asarray(pose['R_physical']),
        np.asarray(pose['centroid']),np.asarray(packet['clean_input']['K']))


def image_path(packet,input_map):
    ids=packet['identifiers'];suffix='/sessions/'+ids['session']+'/rgb/'+ids['id'].split(':',1)[1]+'.png'
    matches=[Path(p)for p,h in input_map.items()if p.startswith('data/evaluation/pallet_eval_v1/')
             and p.endswith(suffix)and h==ids['image_sha256']]
    if len(matches)!=1:raise ValueError('Unique original RGB path/SHA metadata required')
    return C.ROOT/matches[0]


def _ranges(ax,packet):
    from matplotlib.patches import Ellipse
    states=packet['clean_input']['states'];hidden=packet['clean_input']['initial_self_hidden']
    for region in packet['physics']['u_regions']:
        k=region['native_corner']
        if hidden[k]or states[k]!='U':continue
        if region.get('shape')=='ellipse':
            ax.add_patch(Ellipse(region['center_xy'],2*region['half_axes_px'][0],2*region['half_axes_px'][1],
                angle=region['angle_deg'],fill=False,edgecolor='#ff975a',linestyle=':',linewidth=1.4))
    for region in packet['regularizers']['o_regions']:
        k=region['native_corner']
        if hidden[k]or states[k]!='O':continue
        basis=np.asarray(region['basis']);axes=region['axes_px']
        ax.add_patch(Ellipse(region['center_xy'],2*axes[0],2*axes[1],
            angle=np.rad2deg(np.arctan2(basis[1,0],basis[0,0])),fill=False,
            edgecolor='#fa5ab5',linestyle=':',linewidth=1.4))
    # Neither O nor U center is marked or promoted to an image point.


def _only_row(path,ident,*,identifiers=False):
    found=None
    for row in C.rows(path):
        current=row['identifiers']['id']if identifiers else row['id']
        if current==ident:
            if found is not None:raise ValueError('Duplicate selected-case row')
            found=row
        else:del row
    if found is None:raise ValueError('Selected case absent, never drop it')
    return found


def _completed():
    if C.sha(M.OUTPUT/'PREDICTION_SEAL.json')!=PRED_SHA or C.sha(M.OUTPUT/'SCORE_SEAL.json')!=SCORE_SHA:
        raise ValueError('Exact approved complete prediction/score seals required')
    protocol=SCORE._prediction_gate(M.OUTPUT,CENTRAL_SHA,PRED_SHA)
    seal=C.read(M.OUTPUT/'SCORE_SEAL.json');scoring=C.read(M.OUTPUT/'SCORING_PROTOCOL.json')
    if (seal['prediction_seal']!=M._binding(M.OUTPUT/'PREDICTION_SEAL.json')
            or seal['scoring_protocol_sha256']!=C.sha(M.OUTPUT/'SCORING_PROTOCOL.json')
            or scoring['prediction_seal_sha256']!=PRED_SHA or scoring['central_plan_sha256']!=CENTRAL_SHA
            or any(C.sha(M.OUTPUT/(p+'_SCORE.json'))!=h for p,h in seal['score_sha256'].items())):
        raise ValueError('Current all-row fixed scoring closure required')
    M._verify(scoring['source_sha256'],scoring['input_sha256'],scoring['protected_user_sha256'])
    return protocol,scoring


def declare(*,root_metadata_GO=False):
    if root_metadata_GO is not True:raise PermissionError('Root renderer metadata GO before IO')
    if PLAN.exists():raise FileExistsError('Preserve each renderer declaration')
    sources={str(Path(p).resolve().relative_to(C.ROOT)):C.sha(p)for p in
        (__file__,C.__file__,SCORE.__file__,M.__file__,DRAW.__file__,SCORE.METRIC.__file__,SCORE.REF.__file__,SCORE.REF.PG.__file__)}
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,dict(schema='FULL_R2_POSTSCORE_WORST_CASE_PROCESS_PLAN_V32',UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,expected_central_sha256=CENTRAL_SHA,expected_prediction_seal_sha256=PRED_SHA,
        expected_score_seal_sha256=SCORE_SHA,case_rule='union of four new-arm PRIMARY192 translation maxima plus TEST49 SOURCE_CONTAINMENT maximum; canonical first tie; no-pose infinity; deduplicate by ID',
        selected_ids_not_decoded_at_declaration=True,maximum_cases=5,
        panels=['original RGB','own coarse + archived N3/SubPix/BI q + U/O approximate ranges','four final-active finite actual line panels',
            'five final predictions in separate panels','last separate fixed-reference wire, omit if R unresolved'],
        storage_disclosure=dict(archived_coarse_BI_ranges_actual_support_final_poses=True,
            native_start_R_t_and_continuous_intermediate_trajectory_not_archived_not_reconstructed=True),
        reference_policy='same SCORING_PROTOCOL fixed bindings only; final separate panel; never query/fit/start/prior',
        no_new_PnP_NN_TLS_features_optimizer_or_observations=True,S_image_points=0,U_O_center_markers=0,
        WD_channels='each active costplan uses its selected WD; final physical-body wires have no corner-channel equivalence claim',
        no_method_routing_or_adoption=True,default='SAME_FROZEN_E unchanged'))
    for rel in sources:
        target=PLAN.parent/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/rel,target)
    C.write(PLAN.with_name('PLAN_SEAL.json'),dict(plan_sha256=C.sha(PLAN),source_sha256=sources))
    return C.sha(PLAN)


def _reference(part,ident,scoring):
    bindings=scoring['reference_bindings'];path=C.ROOT/bindings[part]['path']
    if C.sha(path)!=bindings[part]['sha256']:raise ValueError('Same fixed reference path/SHA')
    if part=='REMAINDER':
        plan=C.read(path);entry=next(a for a in plan['annotations']if a['id']==ident)
        ann=C.ROOT/entry['annotation']
        if C.sha(ann)!=entry['annotation_sha256']:raise ValueError('Fixed annotation changed')
        return SCORE.REF.reference_from_document(C.read(ann)),dict(path=str(ann.relative_to(C.ROOT)),sha256=entry['annotation_sha256'])
    return C.read(path)['rows'][ident]['target']['truth'],deepcopy(bindings[part])


def render(*,root_actual_postscore_render_GO=False):
    if root_actual_postscore_render_GO is not True:raise PermissionError('Root postscore actual render GO before data IO')
    if OUTPUT.exists():raise FileExistsError('Preserve every actual process render')
    plan=C.read(PLAN);pseal=C.read(PLAN.with_name('PLAN_SEAL.json'))
    if C.sha(PLAN)!=pseal['plan_sha256']:raise ValueError('Frozen renderer plan changed')
    M._verify(plan['source_sha256'])
    protocol,scoring=_completed()
    reports={p:C.read(M.OUTPUT/(p+'_SCORE.json'))for p in('PRIMARY','TEST')}
    cases=choose_cases(reports['PRIMARY'],reports['TEST'])
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'CASE_SELECTION.json',dict(plan_sha256=C.sha(PLAN),score_seal_sha256=SCORE_SHA,cases=cases,
        selection_for_explanation_only=True,method_or_parameter_selection=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    index=['# R2 전체 결과: 고정 최대 이동오차 사례의 보관된 과정','',
        'PRIMARY192의 새 네 방법별 최대 이동오차 사례와 TEST49 SOURCE_CONTAINMENT 최대 사례를 합쳐 설명한다. 오차를 본 뒤 고른 진단 사례이며 대표 성공률 표본·방법 선택 기준이 아니다. 기본값은 SAME_FROZEN_E다.','',
        '초기 점은 raw YOLO가 아니라 N3/SubPix/BI 보정 archive다. 실제로 저장된 coarse·관측·유한 선 지지·최종 자세만 그렸다. 전체 예측에 보관되지 않은 native-start 자세나 연속 최적화 중간 궤적을 재계산하거나 만들지 않았다. S는 모델 wire만, O/U는 대략 범위만 사용한다. 고정 참조는 마지막 별도 패널의 평가용이며 탐색·보정·fit·시작점으로 쓰지 않았다.','']
    all_inputs={str(PLAN.relative_to(C.ROOT)):C.sha(PLAN)};case_data=[]
    for ci,case in enumerate(cases):
        ident=case['id'];packet=_only_row(M.OUTPUT/'INPUTS.jsonl.gz',ident,identifiers=True)
        row=_only_row(M.OUTPUT/'PREDICTIONS.jsonl.gz',ident)
        if packet['identifiers']['id']!=row['id']or packet['identifiers']['partition']!=row['partition']:raise ValueError('Case packet/PRED join differs')
        source=packet['clean_input'];physics=packet['physics'];K=np.asarray(source['K']);imgpath=image_path(packet,protocol['input_sha256'])
        if C.sha(imgpath)!=packet['identifiers']['image_sha256']:raise ValueError('Exact original RGB')
        image=np.asarray(Image.open(imgpath).convert('RGB'))
        if tuple(image.shape[:2])!=tuple(physics['image_hw']):raise ValueError('Exact original image canvas')
        folder=OUTPUT/('case'+str(ci));folder.mkdir();shutil.copyfile(imgpath,folder/'source_rgb.png')
        all_inputs[str(imgpath.relative_to(C.ROOT))]=C.sha(imgpath)
        ref,ref_binding=_reference(row['partition'],ident,scoring)
        all_inputs[ref_binding['path']]=ref_binding['sha256']
        metricrow=next(z for z in reports['PRIMARY']['rows']if z['id']==ident)
        bi=archived_BI(packet);coarse=source['initial_coarse_pose']
        coarse_xy=C.project(np.asarray(source['native_xyz']),np.asarray(coarse['R']),np.asarray(coarse['t']),K)
        fig,axes=plt.subplots(3,2,figsize=(16,20),layout='constrained');axes=axes.ravel()
        DRAW._photo(axes[0],image,'1. Actual original RGB')
        DRAW._photo(axes[1],image,'2. Independent coarse + archived B/I + approximate O/U ranges')
        DRAW._wire(axes[1],coarse_xy,color=DRAW.PURPLE,label='predicted own coarse model')
        _ranges(axes[1],packet);DRAW._BI(axes[1],bi);axes[1].legend(loc='upper right',fontsize=8)
        plans={}
        for ax,arm in zip(axes[2:],M.ARMS[1:]):
            entry=row['arms'][arm];DRAW._photo(ax,image,arm.replace('CLEAN_FRESH_BI_',''))
            if entry['pose_available']:
                family=entry['WD_family'];p=entry['new_geometry_edge_cost_plans'][family];plans[arm]=deepcopy(p)
                DRAW._support(ax,p['active_groups'])
                caption=f"active={len(p['active_groups'])}, owners={p['new_owner_ids']}, WD={family}\n"+'saved actual finite support; partial dashed; no new precise KP'
            else:caption='NO POSE — row retained; no selected pose/active plan drawn'
            ax.text(.01,-.15,caption,transform=ax.transAxes,va='top',fontsize=9)
        fig.suptitle(ident+'\nStored observations and final-active finite native lines (no intermediate optimization trajectory)',fontsize=14)
        fig.savefig(folder/'observations_and_active_lines.png',dpi=135);plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(16,20),layout='constrained');axes=axes.ravel()
        final_xy={}
        for ax,arm,color in zip(axes[:5],M.ARMS,COLORS):
            entry=row['arms'][arm];DRAW._photo(ax,image,arm.replace('CLEAN_FRESH_BI_','')+' — PREDICTED')
            xy=final_projection(packet,entry,arm,row['legacy_e_pose_semantics']);final_xy[arm]=None if xy is None else xy.tolist()
            if xy is not None:DRAW._wire(ax,xy,color=color,label='final physical-body prediction',style='-',width=1.8)
            metrics=metricrow['metrics'][arm]
            ax.text(.01,-.15,f"T={metrics['translation_cm']} cm; R={metrics['rotation_deg']} deg\naccepted={entry['accepted']}; pose_available={entry.get('pose_available',True)}; WD={entry.get('WD_family',entry['actual_pose'].get('WD_family')if entry.get('actual_pose')else None)}",
                transform=ax.transAxes,va='top',fontsize=9)
        ax=axes[-1];DRAW._photo(ax,image,'FIXED EVALUATION REFERENCE — separate, never a search guide')
        reference_xy=None
        if ref['R']is not None:
            xyz=C.cuboid(*np.asarray(ref['xyz'],float));reference_xy=C.project(xyz,np.asarray(ref['R']),np.asarray(ref['t']),K)
            DRAW._wire(ax,reference_xy,color='#5bff77',label='fixed evaluation reference only',style='-',width=1.8)
            ax.legend(loc='upper right',fontsize=8)
        else:ax.text(.03,.1,'Reference rotation unresolved: wire omitted.\nSaved translation scores remain in full denominator.',transform=ax.transAxes,fontsize=10,
            bbox=dict(facecolor='white',alpha=.85))
        ax.text(.01,-.15,'No reference pixels/geometry used to query, correct, fit or initialize.\nDerived remainder references are not independent physical metrology.',transform=ax.transAxes,va='top',fontsize=9)
        fig.suptitle(ident+'\nOnly stored final poses. Native channel association differs across WD; no corner-ID equivalence claim.',fontsize=14)
        fig.savefig(folder/'final_predictions_and_fixed_reference.png',dpi=135);plt.close(fig)
        data=dict(id=ident,partition=row['partition'],selection_reasons=case['reasons'],states=source['states'],
            image_binding=dict(path=str(imgpath.relative_to(C.ROOT)),sha256=C.sha(imgpath)),
            original_BI_archives=bi,U_ranges=physics['u_regions'],O_regions=packet['regularizers']['o_regions'],
            original_native_xyz=source['native_xyz'],original_Q=source['physical_registration_Q'],K=source['K'],own_independent_coarse=coarse,
            coarse_projection_xy=coarse_xy.tolist(),active_cost_plans=plans,
            final_predictions={a:dict(actual_pose=row['arms'][a].get('actual_pose'),accepted=row['arms'][a]['accepted'],
                pose_available=row['arms'][a].get('pose_available',True))for a in M.ARMS},final_projection_xy=final_xy,
            saved_metrics=metricrow['metrics'],fixed_reference=ref,fixed_reference_binding=ref_binding,
            fixed_reference_projection_xy=None if reference_xy is None else reference_xy.tolist(),
            stage_inventory=dict(original_RGB=True,archived_BI_and_U_O=True,own_coarse=True,actual_selected_finite_support=True,
                final_pose=True,native_start_pose=False,continuous_intermediate_trajectory=False),
            no_new_PnP_NN_TLS_features_optimizer_or_measurements=True,reference_query_or_posefit_inputs=0,S_observed_points=0,U_O_point_markers=0)
        C.write(folder/'PROCESS_DATA.json',data);case_data.append(dict(id=ident,folder=folder.name,reasons=case['reasons']))
        index+=['## '+ident,'','선택 이유: '+', '.join(z['population']+'/'+z['arm'].replace('CLEAN_FRESH_BI_','')for z in case['reasons']),
            '',f'![실제 관측과 활성 유한 선]({folder.name}/observations_and_active_lines.png)','',
            f'![최종 예측과 별도 고정 참조]({folder.name}/final_predictions_and_fixed_reference.png)','',
            '| 방법 | 저장 T [cm] | 저장 R [°] | accepted |','|---|---:|---:|---|']
        index += [f"| {a} | {metricrow['metrics'][a]['translation_cm']} | {metricrow['metrics'][a]['rotation_deg']} | {row['arms'][a]['accepted']} |"for a in M.ARMS]
        index+=['',f'[보관 값·원천·표시 단계]({folder.name}/PROCESS_DATA.json) · [원본 RGB]({folder.name}/source_rgb.png)','']
        del packet,row,ref,image
    (OUTPUT/'REPORT_KO.md').write_text('\n'.join(index))
    assets={str(p.relative_to(OUTPUT)):C.sha(p)for p in sorted(OUTPUT.rglob('*'))if p.is_file()}
    C.write(OUTPUT/'RENDER_SEAL.json',dict(schema='SEALED_FULL_R2_POSTSCORE_ACTUAL_PROCESS_V32',plan_sha256=C.sha(PLAN),
        prediction_seal_sha256=PRED_SHA,score_seal_sha256=SCORE_SHA,input_sha256=all_inputs,assets=assets,cases=case_data,
        case_selection_for_explanation_only=True,default='SAME_FROZEN_E unchanged',adoption=False,
        numerical_runtime_or_original_artifact_changes=0,reference_only_last_separate_panel=True,
        no_new_PnP_NN_TLS_features_optimizer=True,visual_QA_pending=True))
    print(str(OUTPUT),C.sha(OUTPUT/'RENDER_SEAL.json'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('declare','render'))
    p.add_argument('--root-metadata-GO',action='store_true');p.add_argument('--root-actual-postscore-render-GO',action='store_true')
    a=p.parse_args();print(declare(root_metadata_GO=a.root_metadata_GO)if a.stage=='declare'else render(root_actual_postscore_render_GO=a.root_actual_postscore_render_GO))
