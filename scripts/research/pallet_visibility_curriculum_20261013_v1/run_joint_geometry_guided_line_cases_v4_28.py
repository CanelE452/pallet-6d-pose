"""Two historical-input RGB line-proposal diagnostics; never KP/PnP adoption.

Primary inputs are the original visible_bi_correction_v2 points and original
N3 geometry, before later boundary recovery. Fixed original/swap coarse guides
and BI endpoint proposals only: no later recovered-q proposal bank, e numeric
pose, GT/errors or selected-method winner. Cached masks are actual SAM masks;
there is no independent occluder mask, so unsupported gaps remain unknown.

Frozen legacy12 results are metadata only. Fresh17c43 and geometry-guided43
are separate, same-width actual RGB measurements. No new NN, pose/noise fit,
CAL/TEST/reference scoring, q/state/covariance change or production adoption.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from . import common as C
from . import joint_geometry_guided_lines_v4_28 as JOINT
from . import edge_visibility_v4_28 as VISIBILITY
from . import render_source_preserving_tangent_mask_process_v4_24 as DRAW_SOURCE


B=C.DOC/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
F=B.parent
R=C.DOC/'nonsevere_v1/remainder_after_test49_v1'
OUTPUT=B/'joint_geometry_guided_line_cases_v4_28'
CASES=('wood_night_01:031426','eval_pallet09:1778653661653195264')
TASKS=B/'coarse_prior_containment_eval_v4_25'
FRESH=B/'grazing_native_edge_observations_v4_18b'
ORIGINAL95=F/'fresh_observations_v2'
ORIGINAL143=R/'fresh_observations_v4_14'
GEOMETRY_KEYS=('native_xyz','K','R_cf','centroid','R_physical','physical_registration_Q',
    'cf_extents','initial_projection_xy','initial_self_hidden','initial_roles','image_hw')


def _target_rows(path,ids):
    """Stream text; JSON-decode requested metadata rows only, never score data."""
    ids=tuple(ids);found={};digests={}
    patterns={fid:re.compile(r'"id"\s*:\s*"'+re.escape(fid)+'"')for fid in ids}
    with gzip.open(path,'rt',encoding='utf-8')as stream:
        for line in stream:
            matching=[fid for fid in ids if fid not in found and patterns[fid].search(line)]
            if not matching:continue
            row=json.loads(line)
            if row.get('id') not in matching:continue
            fid=row['id'];found[fid]=row;digests[fid]=hashlib.sha256(line.encode()).hexdigest()
            if len(found)==len(ids):break
    if set(found)!=set(ids):raise ValueError('Both exact requested metadata frames required')
    return found,digests


def _verify(mapping):
    for relative,digest in mapping.items():
        if C.sha(C.ROOT/relative)!=digest:raise ValueError('Bound bytes changed: '+relative)


def _geometry(source):
    return {key:deepcopy(source[key])for key in GEOMETRY_KEYS}


def _visible(records,states,hidden):
    result=[]
    for record in records:
        k=record['native_corner']
        if hidden[k] or states[k] not in ('B','I'):continue
        if record['state']!=states[k]:raise ValueError('Original BI role binding differs')
        result.append(deepcopy(record))
    return result


def _coarse_guides(task,geometry):
    guides=[]
    # ORIGINAL_INITIAL is automatically provided by the pure proposer.
    for name in ('SWAPPED_WD',):
        family=task['families'][name];coarse=family['coarse_pose']
        trial=_geometry(geometry)
        xyz=np.asarray(family['native_xyz']);rotation=np.asarray(coarse['R']);translation=np.asarray(coarse['t'])
        trial.update(native_xyz=xyz.tolist(),R_cf=rotation.tolist(),centroid=translation.tolist(),
            physical_registration_Q=deepcopy(family['physical_registration_Q']),cf_extents=deepcopy(family['cf_extents']),
            R_physical=(rotation@np.asarray(family['physical_registration_Q'])).tolist(),
            initial_projection_xy=C.project(xyz,rotation,translation,np.asarray(geometry['K'])).tolist())
        guides.append(dict(guide_id='INITIAL_'+name,source_kind='ORIGINAL_TASK_COARSE',geometry=trial))
    return guides


def _old_records(correction,states,hidden):
    output=[]
    for record in correction['records']:
        k=record['native_corner']
        if hidden[k] or states[k]not in('B','I'):continue
        output.append(deepcopy(record))
    return output


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every v28 input/partial run')
    paths=[TASKS/'PROTOCOL.json',TASKS/'INPUT_SEAL.json',TASKS/'INPUTS.jsonl.gz',
        FRESH/'OBSERVATION_SEAL.json',FRESH/'OBSERVATIONS.jsonl.gz',
        ORIGINAL95/'OBSERVATION_SEAL.json',ORIGINAL95/'OBSERVATIONS.jsonl.gz',
        ORIGINAL143/'frames/wood_night_01__031426.json',
        B/'image_evidence_v2/eval_pallet09__1778653661653195264.json',
        B/'image_evidence_v2/eval_pallet09__1778653661653195264.npz',
        B/'evaluation_v3/TEST_controls/eval_pallet09__1778653661653195264.json']
    protocol=C.read(TASKS/'PROTOCOL.json');seal=C.read(TASKS/'INPUT_SEAL.json')
    if seal['rows']!=238 or C.sha(TASKS/'INPUTS.jsonl.gz')!=seal['input_sha256']:
        raise ValueError('Original sealed input metadata required')
    sources={str(Path(module.__file__).resolve().relative_to(C.ROOT)):C.sha(module.__file__)
        for module in (JOINT,VISIBILITY,JOINT.NMS,JOINT.NMS.D,JOINT.NMS.F,JOINT.NMS.G,JOINT.NMS.V3,
            DRAW_SOURCE,DRAW_SOURCE.DRAW,C)}
    sources[str(Path(__file__).resolve().relative_to(C.ROOT))]=C.sha(__file__)
    inputs={str(path.relative_to(C.ROOT)):C.sha(path)for path in paths}
    _verify(sources);_verify(inputs);_verify(protocol['protected_user_sha256'])
    output.mkdir(parents=True)
    C.write(output/'PLAN.json',dict(schema='TWO_HISTORICAL_INPUT_GEOMETRY_LINE_DIAGNOSTIC_PLAN_V28',
        created_UTC=datetime.now(timezone.utc).isoformat(),case_ids=list(CASES),
        inputs='original correction-v2 nonS BI points + original N3 geometry/tight actualSAM, before boundary recovery',
        guide_bank='initial ORIGINAL + initial SWAPPED coarse and measured BI endpoint directions only',
        saved_current_recovered_q_proposal_bank_used=False,e_numeric_pose_or_selected_method_winner_used=False,
        controls=['archived old12 first-correction metadata','fresh frozen17c43 actualNMS','new geometry-guided43 actualNMS and joint proposals'],
        widths_not_confounded='fresh base and new geometry use identical43px line configuration; legacy12 is historical only',
        configuration=JOINT.RECIPE,independent_occluder_mask_available=False,
        targetmask_complement_is_not_occluder=True,gap_labels=['observed_support','occluder_supported_gap','unsupported_unknown'],
        unsupported_without_independent_occluder_is_unknown=True,
        S_U_local_anchor_or_coordinate_queries=0,U_intersection_or_precise_point_adoption=False,
        no_new_NN_pose_CAL_fit_or_GT_inputs=True,no_pose_metrics_read=True,no_old_sources_or_defaults_changed=True,
        posthoc_user_selected_cases_not_generalization_or_correction_success=True,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protocol['protected_user_sha256'],
        source_and_plan_before_new_pixel_search=True))
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot differs')
    tasks,td=_target_rows(TASKS/'INPUTS.jsonl.gz',CASES)
    fresh,fd=_target_rows(FRESH/'OBSERVATIONS.jsonl.gz',CASES)
    p09=CASES[1];original95,od=_target_rows(ORIGINAL95/'OBSERVATIONS.jsonl.gz',[p09])
    wood=C.read(ORIGINAL143/'frames/wood_night_01__031426.json')
    initial={CASES[0]:wood['_original_source'],p09:original95[p09]['runtime']}
    rows=[]
    for fid in CASES:
        task,observed,source=tasks[fid],fresh[fid],initial[fid]
        geometry=_geometry(source['initial_geometry']);states=source['states'];hidden=geometry['initial_self_hidden']
        if states!=task['states'] or states!=observed['states'] or hidden!=task['initial_self_hidden']:
            raise ValueError('Both requested cases keep exact original/current state gates')
        for key in GEOMETRY_KEYS:
            if geometry[key]!=observed['initial_geometry'][key]:raise ValueError('Original N3 geometry changed: '+key)
        correction=source['visible_correction'];visible=_visible(correction['visible_measurements'],states,hidden)
        if fid==p09:
            evidence=C.read(B/'image_evidence_v2/eval_pallet09__1778653661653195264.json')
            mask_path=B/'image_evidence_v2'/evidence['mask_npz'];mask_key='pallet_mask'
            if evidence['image_sha256']!=task['image_sha256'] or C.sha(mask_path)!=evidence['mask_file_sha256']:
                raise ValueError('Historical actual SAM bytes differ')
            archived=C.read(B/'evaluation_v3/TEST_controls/eval_pallet09__1778653661653195264.json')['FULL_CAP12']
            if _visible(archived['visible_measurements'],states,hidden)!=visible:
                raise ValueError('Same old B1/B2 fallback and B4/B7 corrected observations required')
        else:
            mask_path=ORIGINAL143/'masks/wood_night_01__031426.npz';mask_key='pallet_mask_tight'
            if C.sha(mask_path)!=wood['mask_evidence']['mask_file_sha256']:raise ValueError('Historical actual SAM bytes differ')
        with np.load(mask_path,allow_pickle=False)as arrays:keys=arrays.files
        image_path=C.ROOT/task['image']
        if C.sha(image_path)!=task['image_sha256']:raise ValueError('Exact original RGB bytes required')
        for path in (mask_path,image_path):inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
        rows.append(dict(id=fid,image=task['image'],image_sha256=task['image_sha256'],states=deepcopy(states),
            geometry=geometry,visible_measurements=visible,old_legacy_records=_old_records(correction,states,hidden),
            old_configuration=deepcopy(correction['configuration']),
            mask_binding=dict(path=str(mask_path.relative_to(C.ROOT)),sha256=C.sha(mask_path),key=mask_key,available_keys=keys),
            pose_guides=_coarse_guides(task,geometry),u_regions=deepcopy(task['u_regions']),
            current_visible_measurements_provenance_only=_visible(task['spec']['visible'],states,hidden),
            current_points_used_as_search_anchors=False,
            source_row_sha256=dict(task=td[fid],fresh=fd[fid],original95=od.get(fid)),
            independent_occluder_mask=None,independent_occluder_binding=None,
            independent_occluder_absence='cached files contain targetSAM/Tiny/raw/clean/alternatives only; no object/occluder mask'))
    _verify(sources);_verify(inputs);_verify(protocol['protected_user_sha256'])
    C.write(output/'INPUTS.json',dict(cases=rows,source_sha256=sources,input_sha256=inputs))
    C.write(output/'INPUT_SEAL.json',dict(rows=2,case_ids=list(CASES),plan_sha256=C.sha(output/'PLAN.json'),
        inputs_sha256=C.sha(output/'INPUTS.json'),source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protocol['protected_user_sha256'],before_new_pixel_queries=True,
        S_U_coordinate_queries=0,no_new_metrics_or_references=True))
    print(dict(prepared_cases=2,old_inputs_only=True,source_sha256=C.sha(__file__),input_seal=C.sha(output/'INPUT_SEAL.json')),flush=True)


def _guard_summary(observation):
    counts=Counter();records=[]
    for ray in observation['edge_records']:
        counts[ray.get('reason','supported' if ray.get('supported') else 'unreported')]+=1
        records.append(dict(native_corner=ray['native_corner'],native_neighbor=ray['native_neighbor'],
            supported=ray.get('supported',False),reason=ray.get('reason'),
            actual_NMS_peaks=ray.get('image_NMS_peak_count'),actual_RGB_context_peaks=ray.get('RGB_context_peak_count'),
            normal_search_radius_px=ray.get('normal_search_radius_px'),along_interval_px=ray.get('along_interval_px'),
            rejected_chains=[{k:r.get(k)for k in ('reason','inlier_count','inlier_fraction','inlier_span_fraction',
                'inlier_rms_px','angle_change_deg','corrected_endpoint_normal_residual_px')}for r in ray.get('rejected_chains',[])]))
    return dict(reason_counts=dict(counts),rays=records)


def _expected_visibility(row,graph,context):
    selected=set(graph['selected_line_names']);seen=set();records=[];counts=Counter()
    for k,state in enumerate(row['states']):
        if row['geometry']['initial_self_hidden'][k]or state not in('B','I'):continue
        for mate in JOINT.NEIGHBORS[k]:
            edge=tuple(sorted([k,mate]))
            if row['geometry']['initial_self_hidden'][mate]or row['states'][mate]=='S'or edge in seen:continue
            seen.add(edge)
            actual=[n for n in graph['line_candidates']if n['name']in selected and tuple(n['native_edge'])==edge]
            support=np.vstack([n['actual_inliers']for n in actual])if actual else np.empty((0,2))
            roles={n['line_record']['mask_context_kind']for n in actual}
            record=context.classify_expected_edge(k,mate,observed_chain_points=support,
                mask_context_kind=next(iter(roles))if len(roles)==1 else None)
            records.append(record);counts.update(record['counts'])
    return dict(records=records,counts=dict(counts),expected_model_is_original_N3_not_measured_edge=True,
        independent_occluder_present=False,target_mask_zero_not_promoted_to_occlusion=True)


def observe(output=OUTPUT):
    output=Path(output).resolve()
    if (output/'OBSERVATION_SEAL.json').exists():raise FileExistsError('Preserve all completed observations')
    seal=C.read(output/'INPUT_SEAL.json');plan=C.read(output/'PLAN.json')
    if C.sha(output/'INPUTS.json')!=seal['inputs_sha256'] or C.sha(output/'PLAN.json')!=seal['plan_sha256']:
        raise ValueError('Frozen two-case inputs/plan required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(seal[key])
    rows=C.read(output/'INPUTS.json')['cases'];started=time.monotonic();paths=[]
    for row in rows:
        path=output/'frames'/(row['id'].replace(':','__')+'.json')
        if path.exists():raise FileExistsError('Preserve every partial observation')
        bgr=cv2.imread(str(C.ROOT/row['image']))
        if bgr is None:raise ValueError('Actual RGB decode failed')
        with np.load(C.ROOT/row['mask_binding']['path'],allow_pickle=False)as arrays:
            mask=arrays[row['mask_binding']['key']].copy()
        image_binding=dict(sha256=row['image_sha256'],image_hw=list(bgr.shape[:2]))
        visibility=VISIBILITY.prepare_edge_visibility(mask,row['geometry'],row['states'],row['visible_measurements'],
            image_binding=image_binding,occluder_mask=None,occluder_binding=None)
        result=JOINT.observe_joint_geometry_guided_lines(bgr,mask,row['geometry'],row['states'],
            row['visible_measurements'],image_binding=image_binding,base_observation=None,
            pose_guides=row['pose_guides'],visibility_context=visibility)
        base=result['base_observation']
        C.write(path,dict(id=row['id'],historical_input=True,base43_guard_summary=_guard_summary(base),
            additional43_guard_summary=_guard_summary(dict(edge_records=result['additional_guided_edge_records'])),
            expected_edge_visibility=_expected_visibility(row,result['joint_graph'],visibility),
            result=result,no_current_recovered_q_proposal_or_anchor_used=True,new_NN_posefit_GT_or_metrics=0))
        paths.append(path);print(dict(observed=row['id'],elapsed_seconds=time.monotonic()-started),flush=True)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(seal[key])
    C.write(output/'OBSERVATION_SEAL.json',dict(rows=2,case_ids=list(CASES),input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),
        observations={str(path.relative_to(output)):C.sha(path)for path in paths},source_sha256=seal['source_sha256'],
        input_sha256=seal['input_sha256'],actual_RGB_NMS_queries_only=True,new_NN_posefit_GT_metrics=0,
        full_KP_or_U_intersection_or_PnP_adoption=False,elapsed_seconds=time.monotonic()-started))


def _wire(ax,projection,color='#9c6fc7'):
    q=np.asarray(projection)
    for k,mate in ((k,m)for k in range(8)for m in JOINT.NEIGHBORS[k]if k<m):
        ax.plot(q[[k,mate],0],q[[k,mate],1],ls='--',lw=1.,c=color,alpha=.8)


def _lines(ax,chains,color,alpha=.65,lw=1.):
    for chain in chains:
        p=np.asarray(chain['observed_points']);good=np.asarray(chain['inlier_mask'],bool)
        p=p[good]
        if len(p)>1:
            n=np.asarray(chain['normal']);t=np.array([-n[1],n[0]])
            p=p[np.argsort(p@t)]
            ax.plot(p[:,0],p[:,1],c=color,alpha=alpha,lw=lw)


def render(output=OUTPUT):
    output=Path(output).resolve()
    if (output/'RENDER_SEAL.json').exists():raise FileExistsError('Preserve every report')
    source=C.read(output/'INPUTS.json');seal=C.read(output/'OBSERVATION_SEAL.json')
    if C.sha(output/'INPUT_SEAL.json')!=seal['input_seal_sha256']:raise ValueError('Input source changed')
    inputseal=C.read(output/'INPUT_SEAL.json')
    if C.sha(output/'INPUTS.json')!=inputseal['inputs_sha256']:raise ValueError('Prepared historical points changed')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):_verify(inputseal[key])
    for relative,digest in seal['observations'].items():
        if C.sha(output/relative)!=digest:raise ValueError('Observed candidates changed')
    font=DRAW_SOURCE.DRAW._font();plt.rcParams['font.family']=font.stem if not font.is_file() else 'sans-serif'
    if font.is_file():
        from matplotlib.font_manager import fontManager,FontProperties
        fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    figures=[];report=[]
    for index,row in enumerate(source['cases']):
        observed=C.read(output/'frames'/(row['id'].replace(':','__')+'.json'));result=observed['result']
        image=cv2.cvtColor(cv2.imread(str(C.ROOT/row['image'])),cv2.COLOR_BGR2RGB)
        fig,axes=plt.subplots(1,4,figsize=(20,6));h,w=image.shape[:2]
        title=('① 초기 KP/PnP 예상 변','② 실제 RGB 선 후보','③ 기하와 함께 평가한 후보','④ 지원과 불확실 구간')
        for ax,text in zip(axes,title):ax.imshow(image);ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_aspect('equal');ax.set_axis_off();ax.set_title(text)
        _wire(axes[0],row['geometry']['initial_projection_xy'])
        for m in row['visible_measurements']:
            k=m['native_corner']
            if row['geometry']['initial_self_hidden'][k]or row['states'][k]not in('B','I'):raise ValueError('BI gate before point display')
            color='#ef9a31'if m['source']=='VISIBLE_BACKBONE_FALLBACK'else'#229c75'
            q=m['xy'];axes[0].scatter(q[0],q[1],marker='x',c=color,s=28);axes[0].text(q[0]+3,q[1],str(k),fontsize=8,color=color)
        axes[0].text(.01,.97,'주황 ×=초기 KP 예측\n초록 ×=기존 RGB 측정',transform=axes[0].transAxes,
            va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9))
        basechains=[c for ray in result['base_observation']['edge_records']for c in
            [*ray.get('chains',[]),*[v for v in ray.get('rejected_chains',[])if v.get('reason')=='existing_corrected_endpoint_normal_guard_failed']]]
        _lines(axes[1],basechains,'#8e969d',lw=1.1)
        graph=result['joint_graph'];nodes=graph['line_candidates'];selected=set(graph['selected_line_names'])
        _lines(axes[2],[n['line_record']for n in nodes],'#8e969d',alpha=.2,lw=.7)
        chosen=[n for n in nodes if n['name']in selected]
        for node in chosen:
            color='#e5b52e'if node.get('production_guard_bypassed_only_for_unadopted_line_candidate')else'#1aa8a0'
            _lines(axes[2],[node['line_record']],color,alpha=.95,lw=1.7)
        for junction in graph.get('rounded_BI_junctions',[]):
            k=junction['native_corner']
            if row['states'][k]not in('B','I')or row['geometry']['initial_self_hidden'][k]:raise ValueError('No U/S intersection display')
            if junction.get('available'):
                q=junction['candidate_extrapolated_intersection_xy']
                axes[2].scatter(q[0],q[1],marker='D',facecolors='none',edgecolors='#d5a22c',s=45,lw=1.)
        for edge in observed['expected_edge_visibility']['records']:
            samples=edge['samples']
            for first,second in zip(samples,samples[1:]):
                label=first['classification']if first['classification']==second['classification']else'unsupported_unknown'
                colors={'observed_support':'#1aa8a0','occluder_supported_gap':'#df8c37','unsupported_unknown':'#9c6fc7'}
                axes[3].plot([first['xy'][0],second['xy'][0]],[first['xy'][1],second['xy'][1]],
                    c=colors[label],ls='-'if label=='observed_support'else'--',lw=1.4)
        _lines(axes[3],[n['line_record']for n in chosen],'#1aa8a0',alpha=.9,lw=1.7)
        for region in row['u_regions']:
            k=region['native_corner']
            if row['states'][k]!='U' or row['geometry']['initial_self_hidden'][k]:raise ValueError('Only old U region, no point marker')
            axes[3].add_patch(Ellipse(region['center_xy'],2*region['half_axes_px'][0],2*region['half_axes_px'][1],
                angle=region.get('angle_deg',0.),facecolor='#d1ab4b',alpha=.12,edgecolor='#b68f31',lw=.8))
        for ax in axes:ax.text(.01,.01,'보라 점선=예측 · 회색/청록=실제 선 지지\n교점/KP/PnP 채택 없음',transform=ax.transAxes,fontsize=7,va='bottom',bbox=dict(facecolor='white',alpha=.85))
        support=observed['expected_edge_visibility']['counts']
        axes[3].text(.01,.97,'독립 가림 마스크 없음\n지원 '+str(support.get('observed_support',0))+' / 미확정 '+str(support.get('unsupported_unknown',0))+' 표본\nU는 기존 넓은 범위만',transform=axes[3].transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9))
        fig.suptitle(row['id']+' · 옛 입력 그대로: 12px 기록 / 새 43px 비교',fontsize=13)
        fig.tight_layout();path=output/('case'+str(index)+'_actual.png');fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig);figures.append(path)
        report.append(dict(id=row['id'],figure=path.name,original_visible_sources=[dict(native_corner=m['native_corner'],source=m['source'],xy=m['xy'])for m in row['visible_measurements']],
            legacy12_records=row['old_legacy_records'],base43_summary=observed['base43_guard_summary'],new43_summary=observed['additional43_guard_summary'],
            joint_candidates=len(nodes),selected_joint_line_names=graph['selected_line_names'],
            expected_edge_visibility=observed['expected_edge_visibility'],
            old_q_changed=False,U_corner_measurement_or_intersection=False,independent_occluder_mask_available=False))
    intro='<p>점선과 보라선은 초기 모델의 <b>예측</b>, 회색·청록은 실제 사진에서 얻은 선 지지입니다. 노랑 마름모는 B/I에서 측정 직선을 연장한 교차 후보일 뿐 실제 모서리 픽셀이나 채택 KP가 아닙니다. 선정된 선도 정답이 아닙니다. 이 페이지는 새 자세·보정 정확도를 평가하지 않습니다.</p><p>원래 B1/B2가 유지되던 옛 입력을 사용했습니다. 기록된 옛 검색 폭12px와 새 기본17c43px를 구분하고, <b>기하 추가 비교는 동일43px끼리</b> 합니다. 이미 나중에 회복된 q나 e 숫자 자세를 새 방법의 증거로 넣지 않았습니다.</p><p>가림 물체의 독립 마스크는 저장되어 있지 않습니다. 타겟 마스크 밖이라는 이유만으로 외부 가림이라고 확정하지 않으며, 지원 없는 부분은 원인 미확정으로 둡니다. U는 기존 넓은 영역만 표시합니다.</p>'
    sections=''.join('<section><h2>'+html.escape(r['id'])+'</h2><img src="'+r['figure']+'" alt="실제 사진의 예측과 관측 구분"><details><summary>입력·후보·거부 사유</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details></section>'for r in report)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>초기 기하와 실제 선 후보</title><style>body{font:16px/1.6 sans-serif;max-width:1600px;margin:24px auto;padding:0 20px;color:#183042}img{width:100%;height:auto}section{margin:30px 0}pre{white-space:pre-wrap;font-size:12px}</style><h1>예측한 변을 단서로 실제 선을 찾기</h1>'+intro+sections+'<p><a href="PROCESS_DATA.json">전체 표시 근거</a></p></html>'
    (output/'index.html').write_text(page,encoding='utf-8');C.write(output/'PROCESS_DATA.json',dict(cases=report,source='saved actual RGB candidate observations only',pose_metrics_or_GT_reads=0,new_KP_adoptions=0))
    C.write(output/'RENDER_SEAL.json',dict(observation_seal_sha256=C.sha(output/'OBSERVATION_SEAL.json'),source_sha256=C.sha(__file__),
        assets={p.name:C.sha(p)for p in [*figures,output/'index.html',output/'PROCESS_DATA.json']},case_ids=list(CASES),
        actual_RGB_only=True,predicted_vs_measured_labels=True,no_GT_wire_S_U_point_or_adopted_KP=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','observe','render'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    dict(prepare=prepare,observe=observe,render=render)[args.command](args.output)
