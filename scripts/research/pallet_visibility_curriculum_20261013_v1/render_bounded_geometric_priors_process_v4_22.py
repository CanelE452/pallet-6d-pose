"""Predeclared actual-image bounded O/S controls; render only after full score GO.

Stored RGB, masks, chain observations and poses only. No reference-pose/corner
reader, inference, fitting, case-best method, source observation mutation or
S/U image-point access. Preparation records the four cases before v22 metrics.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import html
import os
from urllib.parse import quote

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from . import common as C
from . import evaluate_bounded_geometric_priors_v4_22 as EXP
from . import render_o_localization_abstention_process_v4_19 as PREVIOUS

OLD = PREVIOUS.OLD
DRAW = OLD.DRAW
OUTPUT = EXP.OUTPUT.parent/'bounded_geometric_priors_process_v4_22'
PLAN = EXP.OUTPUT.parent/'bounded_geometric_priors_report_plan_v4_22'
CASES = ('wood_night_01:032645', 'wood_night_01:033821',
         'wood_night_01:031344', 'eval_noapril:1775201428535961600')
BASELINE, O_NATIVE, OS_NATIVE = EXP.ARMS[:3]
PARENT_NATIVE = EXP.V.ARMS[1]
METHODS = (BASELINE, PARENT_NATIVE, O_NATIVE, OS_NATIVE)
NAMES = {BASELINE:'기존 e', PARENT_NATIVE:'19 native · O 위치 보류',
         O_NATIVE:'22 native · bounded O', OS_NATIVE:'22 native · bounded O+S'}
COLORS = {BASELINE:DRAW.BLUE, PARENT_NATIVE:DRAW.ORANGE,
          O_NATIVE:'#20a57a', OS_NATIVE:'#8b55bd'}
TITLES = ('① 초기 모델·고정 B/I 기록', '② 실제 마스크·모든 NMS 선 대안',
          '③ 유지된 닫힌 캔버스·U·부분관측', '④ O 경험 영역·초기 S 모델 가정',
          '⑤ 기존 e 저장 자세', '⑥ 19 native 저장 자세',
          '⑦ 22 bounded O native 저장 자세', '⑧ 22 bounded O+S native 저장 자세')
CAPTIONS = (
    '회색 wire는 초기 모델투영. 분홍 ×는 완전 B/I 기록, 회색 +는 부분 B/I의 보관 q이며 정밀 RGB 코너가 아님. S/U 입력 좌표는 읽거나 표시하지 않음.',
    '녹색은 저장 실제 마스크, 회색은 모든 실제 NMS 선 대안. 청록은 O+S 선택 family에서 사용한 물리 edge 그룹이며 동일 실제 픽셀임. 표시용 픽셀 중복 제거는 측정 독립성이나 대안 선택을 뜻하지 않음.',
    'O의 hard domain은 19와 22 모두 원래 닫힌 영상 캔버스. U는 넓은 경험 범위만, 부분 B/I는 선 법선·접선 구간만 사용. 확대창 밖 영역도 허용되며 원래 전체 수치는 JSON에 보존.',
    '주황 O 영역은 raw 예측 중심과 기존 CAL 반경을 보관한 coarse 가정. 실제 가림 외곽이나 정밀 KP가 아니며 22에서만 bounded soft penalty로 사용. 보라 S 표시는 초기 모델투영·가시성 가정이며 S 영상 관측이 아님.',
    '파란 wire는 기존 e의 저장 모델투영. 측정점·참조/GT wire가 아님.',
    '주황 wire는 19 native-only의 저장 선택. O의 위치정보를 보류한 대조군이며 모든 사례에서 같은 방법을 사용.',
    '초록 wire는 동일 관측·hard guards에 bounded O model prior를 더한 22 native-only 저장 선택. O/S 에너지는 독립 영상 정보가 아닌 별도 correlated regularizer.',
    '보라 wire는 동일 bounded O와 초기 S model prior를 더한 저장 선택. 두 prior는 frame별 유한 상한을 가지며 coarse 가정이 틀려도 hard truth로 강제하지 않음. 유일성·정확도 보증 아님.')


def _url(path):
    return quote(os.path.relpath(Path(path).resolve(), OUTPUT), safe='/')


def _binding(path):
    path = Path(path).resolve()
    return str(path.relative_to(C.ROOT)), C.sha(path)


def prepare_plan():
    if PLAN.exists():
        raise FileExistsError('Preserve every declaration')
    if (EXP.OUTPUT/'SCORE_SEAL.json').exists():
        raise ValueError('Declare before new v22 scores, not retrospectively')
    PLAN.mkdir(parents=True)
    source = Path(__file__).resolve()
    (PLAN/source.name).write_bytes(source.read_bytes())
    C.write(PLAN/'PREDECLARATION.json', dict(
        schema='BOUNDED_O_S_REAL_PROCESS_PREDECLARATION_V4_22',
        case_ids=list(CASES), uniform_methods=list(METHODS),
        case_reasons=['기존19 postseal 최대 오차 사례', '기존 O ellipse 불일치 고정 사례',
                      '기존 관측·continuous 진단 사례', '기존 TEST49 회귀 사례'],
        renderer_source_sha256=C.sha(source), new_v22_metrics_read_before_declaration=False,
        requires_complete238_all5_prediction_and_score_seals=True,
        requires_root_audit_complete_memory_GO_before_bulk_render=True,
        no_new_GT_reference_pose_or_corner_read=True, no_GT_wire=True,
        no_new_NN_line_or_pose_fit=True, no_default_adoption=True,
        reviewer_receipt=_binding(EXP.OUTPUT.parent/'independent_bounded_geometric_priors_review_v4_22/PEER_REVIEW_V1.json')))
    print(dict(plan=str(PLAN), cases=list(CASES)), flush=True)


def _load_plan():
    plan = C.read(PLAN/'PREDECLARATION.json')
    if (tuple(plan['case_ids']) != CASES or tuple(plan['uniform_methods']) != METHODS
            or plan['new_v22_metrics_read_before_declaration'] is not False
            or C.sha(Path(__file__)) != plan['renderer_source_sha256']):
        raise ValueError('Exact predeclared cases/methods/source required')
    return plan


def _completed(folder, *, new=False):
    """Verify complete populations; retain only the four selected large rows."""
    protocol = C.read(folder/'PROTOCOL.json')
    predseal, scoreseal = C.read(folder/'PREDICTION_SEAL.json'), C.read(folder/'SCORE_SEAL.json')
    if (predseal['rows'] != 238 or not predseal.get('all3_arms_complete')
            or C.sha(folder/'PREDICTIONS.jsonl.gz') != predseal['predictions_sha256']
            or C.sha(folder/'PROTOCOL.json') != predseal['protocol_sha256']
            or C.sha(folder/'PREDICTION_SEAL.json') != scoreseal['prediction_seal']['sha256']):
        raise ValueError('Complete238 prediction and fixed postseal score required')
    if new and (not predseal.get('all5_arms_complete') or not predseal.get('all_declared_arms_complete')
                or predseal['arms'] != list(EXP.ARMS) or predseal['arm_count'] != 5):
        raise ValueError('All five v22 arms must be complete before rendering')
    for key in ('source_sha256', 'input_sha256', 'protected_user_sha256'):
        EXP.E._verify_sources(protocol[key])
    selected, ordered = {}, []
    for row in C.rows(folder/'PREDICTIONS.jsonl.gz'):
        ordered.append(row['id'])
        if row['id'] in CASES:
            selected[row['id']] = row
    if ordered != protocol['ids'] or len(set(ordered)) != 238 or set(selected) != set(CASES):
        raise ValueError('All unique238 rows and four unreplaced cases required')
    reports, scores = {}, {}
    for partition, count in (('CAL',46), ('TEST',49), ('REMAINDER',143)):
        path = folder/f'{partition}_SCORE.json'
        if C.sha(path) != scoreseal['score_sha256'][partition]:
            raise ValueError('Fixed score binding differs')
        report = C.read(path)
        if len(report['rows']) != count:
            raise ValueError('No fallback, failure or native-incompatible row may be dropped')
        reports[partition] = report
        scores.update({row['id']:row for row in report['rows']})
    if set(scores) != set(ordered):
        raise ValueError('Scores must contain the complete exact population')
    return protocol, selected, scores, reports


def _prior_energies(arm, active):
    if not active:
        return dict(applied=False, O=None, S=None, native=None, total=None)
    solution = arm['solution']
    if not arm['accepted']:
        return dict(applied=True, O=None, S=None, native=None, total=None,
                    no_feasible_selected_candidate=True)
    rows = solution['regularizer_costs']
    if [row['kind'] for row in rows] != list(EXP.P.REGULARIZER_KINDS):
        raise ValueError('Stored separate O/S energies required')
    native, total = solution['native_observation_objective'], solution['total_regularized_objective']
    if not np.isclose(total, native+sum(row['weighted_cost'] for row in rows), atol=1e-10, rtol=1e-12):
        raise ValueError('Displayed prior accounting differs from selected objective')
    return dict(applied=True, O=rows[0]['weighted_cost'], S=rows[1]['weighted_cost'],
                native=native, total=total,
                O_audit=solution['O_coarse_localization_prior_audit'],
                S_audit=solution['S_coarse_visibility_prior_audit'])


def _energy_text(value):
    if not value['applied']:
        return 'O/S prior 미적용'
    if value['O'] is None:
        return 'fallback: 선택된 joint energy 없음'
    return f"영상 {value['native']:.3f} · O prior {value['O']:.3f} · S prior {value['S']:.3f}"


def _pose_panel(ax, q, arm, key, metric, energy, bounds, states):
    if q is None:
        ax.text(.02,.95,'저장 자세 없음',transform=ax.transAxes,va='top')
        return
    DRAW._wire(ax,q,COLORS[key],width=1.8,ls='-' if arm['accepted'] else '--')
    DRAW._points(ax,{k:q[k] for k,state in enumerate(states) if state=='S'},DRAW.PURPLE,'^',prefix='S모델')
    outside = np.any((q < [bounds[0],bounds[3]]) | (q > [bounds[1],bounds[2]]),axis=1)
    text = OLD._metric_text(metric)+' · '+('joint 수용' if arm['accepted'] else 'fallback')+'\n'+_energy_text(energy)
    if outside.any():
        text += '\n확대창 밖 모델 꼭짓점 '+str(int(outside.sum()))+'개 · 전체 좌표 JSON'
    ax.text(.02,.97,text,transform=ax.transAxes,va='top',fontsize=9,
            bbox=dict(facecolor='white',alpha=.89))


def _summary(new, parent):
    rows = []
    for partition, label in (('TEST','기존 TEST49'),('REMAINDER','추가143'),('CAL','별도 CAL46')):
        for key in METHODS:
            value = (parent if key==PARENT_NATIVE else new)[partition]['summary'][key]
            rows.append('<tr><td>'+label+'</td><td>'+NAMES[key]+'</td><td>'+str(value['translation_le5'])+
                '</td><td>'+str(value['successes_5cm_5deg'])+'</td><td>'+f"{value['translation_cm']['mean']:.3f}"+
                '</td><td>'+f"{value['translation_cm']['max']:.3f}"+'</td><td>'+str(value['fallback_count'])+'</td></tr>')
    return '<table><tr><th>집합</th><th>고정 방법</th><th>T≤5cm</th><th>5cm·5°</th><th>평균 cm</th><th>최대 cm</th><th>fallback</th></tr>'+''.join(rows)+'</table>'


def render():
    if OUTPUT.exists():
        raise FileExistsError('Preserve completed and partial reports')
    _load_plan()
    protocol,new,scores,reports = _completed(EXP.OUTPUT,new=True)
    parentprotocol,parent,parentscores,parentreports = _completed(EXP.V.OUTPUT)
    if protocol['ids'] != parentprotocol['ids']:
        raise ValueError('The exact same all238 population is required')
    tasks = {r['id']:r for r in C.rows(EXP.OUTPUT/'INPUTS.jsonl.gz') if r['id'] in CASES}
    observations = {r['id']:r for r in C.rows(OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz') if r['id'] in CASES}
    prior_models = dict(O=C.read(EXP.OCAL.OUTPUT/'MODEL.json'), S=C.read(EXP.SCAL.OUTPUT/'MODEL.json'))
    if set(tasks)!=set(CASES) or set(observations)!=set(CASES):
        raise ValueError('Every predeclared input/observation is required')
    paths = [Path(__file__),Path(PREVIOUS.__file__),Path(OLD.__file__),Path(DRAW.__file__),
             PLAN/'PREDECLARATION.json',OLD.OBS.OUTPUT/'OBSERVATIONS.jsonl.gz',OLD.OBS.OUTPUT/'OBSERVATION_SEAL.json']
    for folder in (EXP.OUTPUT,EXP.V.OUTPUT):
        paths += [folder/name for name in ('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz',
            'PREDICTION_SEAL.json','PREDICTIONS.jsonl.gz','SCORE_SEAL.json','CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json')]
    for folder in (EXP.OCAL.OUTPUT,EXP.SCAL.OUTPUT):
        paths += [folder/name for name in ('MODEL.json','MODEL_SEAL.json')]
    bindings = dict(_binding(p) for p in paths)
    font = DRAW._font()
    if font.is_file():
        bindings[str(font)] = C.sha(font)
    OUTPUT.mkdir(parents=True);assets = OUTPUT/'assets';assets.mkdir()
    cases,pictures = [],[]
    for index,fid in enumerate(CASES):
        task,observed = tasks[fid],observations[fid]
        if (EXP.E._digest(new[fid]['arms'][BASELINE]) != EXP.E._digest(parent[fid]['arms'][BASELINE])
                or scores[fid]['metrics'][BASELINE] != parentscores[fid]['metrics'][BASELINE]
                or observed['states'] != task['states']
                or observed['visible_measurements'] != task['spec']['visible']
                or task['new_chain_observation'] != observed['chain_observation']):
            raise ValueError('Exact baseline and actual frozen observation profile required')
        original_domains = [r['original_domain'] for r in task['O_localization_abstention']['records']]
        if original_domains != [d for d in observed['domains'] if d['state']=='O']:
            raise ValueError('Raw-centered O archive provenance differs')
        image_path,mask_path = C.ROOT/task['image'],C.ROOT/observed['mask_binding']['path']
        if C.sha(image_path)!=task['image_sha256'] or C.sha(mask_path)!=observed['mask_binding']['sha256']:
            raise ValueError('Actual unchanged image/mask binding required')
        bindings[task['image']] = task['image_sha256'];bindings[observed['mask_binding']['path']] = observed['mask_binding']['sha256']
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError('Actual image decode failed')
        image = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        with np.load(mask_path,allow_pickle=False) as arrays:
            mask = arrays['pallet_mask_clean'].copy()
        geometry,K = observed['initial_geometry'],observed['initial_geometry']['K']
        _,_,prepared_S = EXP._prior_inputs(task,prior_models)
        eligible_S = [r['native_corner'] for r in prepared_S['eligible_initial_S']]
        arms = {key:(parent[fid] if key==PARENT_NATIVE else new[fid])['arms'][key] for key in METHODS}
        projected = {key:OLD._model(arm,K) for key,arm in arms.items()}
        metrics = {key:(parentscores if key==PARENT_NATIVE else scores)[fid]['metrics'][key] for key in METHODS}
        energies = {key:_prior_energies(arms[key],key in (O_NATIVE,OS_NATIVE)) for key in METHODS}
        support = [np.asarray(geometry['initial_projection_xy'])]
        support += [np.asarray([m['xy'] for m in task['spec']['visible']]).reshape(-1,2)]
        support += [np.asarray([d['center_xy'] for d in original_domains]).reshape(-1,2)]
        xy = np.vstack(support);h,w = image.shape[:2]
        lo,hi = np.maximum(xy.min(0)-35,0),np.minimum(xy.max(0)+35,[w,h])
        bounds = tuple(map(float,(lo[0],hi[0],hi[1],lo[1])))
        selected = arms[OS_NATIVE]
        bundle = selected['new_image_line_group_authentication'][selected['WD_family']]
        case = dict(id=fid,partition=task['partition'],image=task['image'],image_sha256=task['image_sha256'],
            states=task['states'],initial_geometry=geometry,frozen_BI=task['spec']['visible'],
            original_O_domains=original_domains,hard_O_T_domains=task['spec']['domains'],
            U_regions=task['u_regions'],partial_boundary=task['boundary_merged'],
            actual_NMS_observation=task['new_chain_observation'],mask_binding=observed['mask_binding'],
            original_confident_S_native_ids=eligible_S,
            crop_bounds=bounds,crop_not_based_on_reference_or_selected_pose=True,methods={},stages=[])
        for key,arm in arms.items():
            case['methods'][key] = dict(actual_pose=arm['actual_pose'],accepted=arm['accepted'],WD_family=arm.get('WD_family'),
                model_projection_xy=None if projected[key] is None else projected[key].tolist(),
                metric=metrics[key],separate_model_prior_energy=energies[key],
                selected_endpoint_origin=arm.get('keypoint_readout',{}).get('selected_endpoint_origin'),
                original_O_projection_geometry_only=PREVIOUS._domain_audit(projected[key],original_domains,[h,w]))
        for stage,title in enumerate(TITLES):
            fig,ax = DRAW._base(image,title);ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:])
            if stage==0:
                DRAW._wire(ax,np.asarray(geometry['initial_projection_xy']),'#7e8790',ls='--')
                OLD._frozen_points(ax,task)
            elif stage==1:
                ax.imshow(np.dstack([np.zeros_like(mask),mask,np.zeros_like(mask),mask*.15]))
                case['NMS_display_audit'] = OLD._nms(ax,task['new_chain_observation'],bundle)
            elif stage==2:
                if original_domains:
                    PREVIOUS._new_canvas(ax,image,task['spec']['domains'],bounds)
                else:
                    ax.text(.02,.97,'이 사례는 O 상태 없음',transform=ax.transAxes,va='top',fontsize=9)
                PREVIOUS._context(ax,task)
            elif stage==3:
                PREVIOUS._old_o(ax,original_domains)
                initial=np.asarray(geometry['initial_projection_xy'])
                DRAW._points(ax,{k:initial[k] for k in eligible_S},DRAW.PURPLE,'^',prefix='초기S모델')
                if not eligible_S:
                    ax.text(.02,.97,'초기 −2° confident S prior 대상 없음',transform=ax.transAxes,va='top',fontsize=9)
            else:
                key = METHODS[stage-4]
                _pose_panel(ax,projected[key],arms[key],key,metrics[key],energies[key],bounds,task['states'])
            path = assets/f'case{index}_stage{stage+1}.png';DRAW._save(fig,path,CAPTIONS[stage])
            case['stages'].append(dict(title=title,caption=CAPTIONS[stage],asset=str(path.relative_to(OUTPUT))))
        cases.append(case);pictures.append((image,bounds,projected,metrics,energies))
    fig,axes = plt.subplots(len(METHODS),len(CASES),figsize=(17,14),squeeze=False)
    for col,(image,bounds,projected,metrics,energies) in enumerate(pictures):
        for row,key in enumerate(METHODS):
            ax=axes[row,col];ax.imshow(image)
            if projected[key] is not None:
                DRAW._wire(ax,projected[key],COLORS[key],width=1.4)
            ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:]);ax.set_aspect('equal');ax.set_axis_off()
            ax.set_title((CASES[col]+'\n' if row==0 else '')+NAMES[key]+'\n'+OLD._metric_text(metrics[key])+'\n'+_energy_text(energies[key]),fontsize=8)
    fig.suptitle('실제 사진 · 고정 네 사례 / e → 19 native → 22 bounded O → 22 bounded O+S\nwire는 저장 모델투영 · O/S는 별도 coarse regularizer · GT wire 없음',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUTPUT/'overview_actual.png',dpi=160,bbox_inches='tight');plt.close(fig)
    C.write(OUTPUT/'PROCESS_DATA.json',dict(schema='BOUNDED_O_S_ACTUAL_PROCESS_V4_22',predeclared_cases=list(CASES),
        uniform_native_methods=list(METHODS),cases=cases,source_bindings=bindings,
        no_reference_pose_or_corner_reads=True,no_GT_wire=True,new_inference_or_fit=0,
        S_raw_not_read=True,U_range_only=True,coarse_priors_not_independent_image_evidence=True,default_e_unchanged=True))
    sections=[]
    for case in cases:
        rows=''.join('<tr><td>'+NAMES[key]+'</td><td>'+OLD._metric_text(case['methods'][key]['metric'])+'</td><td>'+html.escape(str(case['methods'][key]['WD_family']))+
            '</td><td>'+('joint 수용' if case['methods'][key]['accepted'] else 'fallback')+'</td><td>'+_energy_text(case['methods'][key]['separate_model_prior_energy'])+'</td></tr>' for key in METHODS)
        stages=''.join('<h3>'+s['title']+'</h3><img src="'+s['asset']+'" alt="실제 사진 저장 처리 단계"><p>'+s['caption']+'</p>' for s in case['stages'])
        sections.append('<details open><summary>'+html.escape(case['id'])+'</summary><p>고정 상태: '+' '.join(case['states'])+'</p><div class="tablewrap"><table><tr><th>방법</th><th>명목 참조 오차</th><th>W/D</th><th>상태</th><th>별도 에너지</th></tr>'+rows+'</table></div><p><a href="'+_url(C.ROOT/case['image'])+'">실제 원본 RGB</a></p>'+stages+'</details>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bounded O/S prior · 실제 사진 과정</title><style>body{margin:0;background:#edf2f6;color:#173044;font-family:system-ui,sans-serif}main{max-width:1120px;padding:18px;margin:auto}img{display:block;max-width:100%;height:auto}details{background:white;margin:18px 0;padding:16px;border-radius:10px}summary{font-weight:bold}p{line-height:1.6}.tablewrap{overflow-x:auto}table{border-collapse:collapse;max-width:100%;font-size:13px}td,th{padding:7px;border:1px solid #cbd6df;text-align:left}@media(max-width:500px){main{padding:8px}details{padding:9px}table{font-size:11px}td,th{padding:4px}}</style><main><h1>Bounded O/S coarse prior: 실제 사진 과정</h1><p>네 사례는 새22 점수 전에 기존 진단 사례에서 고정했습니다. 모든 사진에서 동일한 e → 19 native → 22 bounded O native → 22 bounded O+S native를 비교합니다. 사례별 최적 방법을 섞지 않으며 대표성·성공을 보장하지 않습니다.</p><p>O는 정밀 점이 아닙니다. hard domain은 계속 닫힌 캔버스이고, 보관된 경험 O 원 밖 위반만 bounded soft prior로 추가합니다. S는 초기 모델 가시성에서 유도한 frame-bounded prior이며 S 영상 조회·2D 관측은 없습니다. 두 prior는 coarse 예측과 상관된 regularizer로, 독립 영상 정보·참가시성·관측 rank·유일성을 증명하지 않습니다. U는 범위만 유지합니다.</p><p>표시 energy의 영상 항은 기존 native likelihood, O/S 항은 별도 model prior입니다. e/19의 미적용 항을 0이라는 측정값으로 바꾸지 않습니다. wire는 저장 모델투영이며 GT/참조 wire는 없습니다. 추가143 오차는 기존 annotation/PnP에서 유도한 명목 참조 기준으로 독립 계측이 아닙니다. 반복 사용한 자료의 대조실험이며 기존 e 기본값은 유지합니다.</p><div class="tablewrap">'''+_summary(reports,parentreports)+'''</div><img src="overview_actual.png" alt="고정 네 사례 네 방법 실제 사진 비교"><p><a href="PROCESS_DATA.json">모든 표시 값·출처 JSON</a> · <a href="'''+_url(PREVIOUS.OUTPUT/'index.html')+'''">19 실제 과정</a> · <a href="'''+_url(OLD.OUTPUT/'index.html')+'''">18c 실제 과정</a> · <a href="'''+_url(DRAW.OUTPUT/'index.html')+'''">첫49 e 과정</a></p>'''+''.join(sections)+'''</main></html>'''
    (OUTPUT/'index.html').write_text(page)
    for relative,digest in bindings.items():
        path=Path(relative) if Path(relative).is_absolute() else C.ROOT/relative
        if C.sha(path)!=digest:
            raise ValueError('Bound source/evidence changed during rendering')
    C.write(OUTPUT/'RENDER_SEAL.json',dict(source_sha256=C.sha(Path(__file__)),input_sha256=bindings,
        assets={str(p.relative_to(OUTPUT)):C.sha(p) for p in OUTPUT.rglob('*') if p.is_file()},
        cases=4,stages_per_case=8,actual_RGB_only=True,predeclared_case_order=list(CASES),
        uniform_methods=list(METHODS),no_GT_pose_corner_reads_or_overlays=True,new_inference_or_fit=0))
    print(dict(output=str(OUTPUT),cases=list(CASES),methods=list(METHODS)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','render'));args=parser.parse_args()
    if args.command=='prepare':
        prepare_plan()
    else:
        render()
