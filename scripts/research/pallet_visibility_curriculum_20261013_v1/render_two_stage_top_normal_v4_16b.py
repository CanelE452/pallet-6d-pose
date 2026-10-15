"""Postsealed actual-image comparison; stored poses only, no inference or fit.

The first49/default e and both predeclared top-normal controls remain distinct.
Pictures are chosen after scoring for explanation, never for production routing.
"""
from pathlib import Path
import argparse
import html
import os
from urllib.parse import quote

import cv2
import numpy as np
from matplotlib import pyplot as plt

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_two_stage_top_normal_v4_16b as EXP
from . import evaluate_fixed_wd_archived_tangent_v4_15 as ARCHIVE
from . import compare_remainder_posefix_cache_v4_14 as PF
from . import render_remainder_process_v4_14 as PREVIOUS
from . import render_frozen_wd_process_v4_13e_v2 as DRAW
from .scoring import pose_metric

OUTPUT=EXP.OUTPUT.parent/'two_stage_top_normal_actual_report_v4_16b'
EXPERIMENT=EXP.OUTPUT.parent/'two_stage_top_normal_eval_v4_16b_cpu8'
METHODS=('INITIAL_N3','POSEFIX_RAW',*EXP.ARMS)
NAMES={'INITIAL_N3':'저장 초기 N3 자세','POSEFIX_RAW':'저장 PoseFix RAW 자세',
    EXP.ARMS[0]:'기존 e 기본값',EXP.ARMS[1]:'법선 σ2° + ALL_T4',
    EXP.ARMS[2]:'법선 CALmax σ4.13° + ALL_T4'}
COLORS={'INITIAL_N3':'#e68020','POSEFIX_RAW':'#bd34bc',EXP.ARMS[0]:'#10a4db',
    EXP.ARMS[1]:'#09a15c',EXP.ARMS[2]:'#6651db'}


def _link(path,output):
    return quote(os.path.relpath(Path(path).resolve(),output),safe='/')


def _pose_projection(pose,K):
    # The saved camera-facing model owns this basis/extent; no GT permutation.
    return C.project(C.cuboid(*pose['cf_extents']),pose['R_cf'],pose['centroid'],K)


def _inputs(folder):
    folder=Path(folder).resolve()
    protocol=C.read(folder/'PROTOCOL.json');ps=C.read(folder/'PREDICTION_SEAL.json')
    ss=C.read(folder/'SCORE_SEAL.json')
    if (ps['rows']!=238 or not ps['all3_arms_complete']
        or C.sha(folder/'PREDICTIONS.jsonl.gz')!=ps['predictions_sha256']
        or C.sha(folder/'PROTOCOL.json')!=ps['protocol_sha256']):
        raise ValueError('Complete238 controls must be sealed before rendering/reference reads')
    E._verify_binding(ss['prediction_seal'])
    for key in ('source_sha256','input_sha256','protected_user_sha256'):
        E._verify_sources(protocol[key])
    EXP.CLOSED.verify_additional_source_closure()
    scores={}
    for partition,count in (('CAL',46),('TEST',49),('REMAINDER',143)):
        path=folder/f'{partition}_SCORE.json'
        if C.sha(path)!=ss['score_sha256'][partition]:raise ValueError('Postseal score changed')
        report=C.read(path)
        if report['count']!=count or not report['all_rows_kept']:raise ValueError('Full population required')
        scores[partition]=report
    tasks=list(C.rows(ARCHIVE.OUTPUT/'INPUTS.jsonl.gz'))
    rows=list(C.rows(folder/'PREDICTIONS.jsonl.gz'))
    if [t['id']for t in tasks]!=protocol['ids']or [r['id']for r in rows]!=protocol['ids']:
        raise ValueError('All ordered238 archived inputs and results required')
    pfseal=C.read(PF.OUTPUT/'BASELINE_SEAL.json')
    if C.sha(PF.OUTPUT/'PREDICTIONS.jsonl.gz')!=pfseal['predictions_sha256']:
        raise ValueError('Saved143 PoseFix cache changed')
    posefix={r['id']:r for r in C.rows(PF.OUTPUT/'PREDICTIONS.jsonl.gz')}
    for r in C.rows(PF.OLD/'PREDICTIONS.jsonl.gz'):
        if r['method']=='POSEFIX_RAW':posefix[r['id']]=r
    if set(posefix)!={t['id']for t in tasks}:raise ValueError('Same saved238 PoseFix rows required')
    return protocol,scores,{t['id']:t for t in tasks},{r['id']:r for r in rows},posefix


def _reference_rows(tasks):
    # Caller reaches here only after its complete prediction+score seals.
    references={};metadata={};bindings={}
    for partition in ('CAL','TEST'):
        path=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json'if partition=='CAL'
            else'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(path)['rows'];bindings[str(path.relative_to(C.ROOT))]=C.sha(path)
        for t in tasks.values():
            if t['partition']==partition:
                references[t['id']]=labels[t['id']]['target']['truth']
                metadata[t['id']]=labels[t['id']].get('metadata',{})
    plan=C.read(ARCHIVE.REF.OUTPUT/'PLAN.json')
    for record in plan['annotations']:
        path=C.ROOT/record['annotation']
        if C.sha(path)!=record['annotation_sha256']:raise ValueError('Declared nominal reference changed')
        references[record['id']]=ARCHIVE.REF.reference_from_document(C.read(path))
        bindings[record['annotation']]=record['annotation_sha256']
        metadata[record['id']]={'severity':'CLEAN/MODERATE cohort; nominal derived reference'}
    return references,metadata,bindings


def _initial_pose(task):
    f=task['stage1_family_seed_bank']['ORIGINAL_WD']['family'];p=f['coarse_pose']
    return dict(R_cf=p['R'],centroid=p['t'],R_physical=(np.asarray(p['R'])@f['physical_registration_Q']).tolist(),
        cf_extents=f['cf_extents'],physical_registration_Q=f['physical_registration_Q'],available=True)


def render(output=OUTPUT,experiment_folder=EXPERIMENT):
    folder=Path(experiment_folder).resolve()
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every earlier actual-image report')
    protocol,scores,tasks,rows,posefix=_inputs(folder)
    refs,metadata,bindings=_reference_rows(tasks)
    for p in (Path(__file__),folder/'PROTOCOL.json',folder/'PREDICTION_SEAL.json',
              folder/'PREDICTIONS.jsonl.gz',folder/'SCORE_SEAL.json',
              ARCHIVE.OUTPUT/'INPUT_SEAL.json',PF.OUTPUT/'BASELINE_SEAL.json',
              PF.OUTPUT/'PREDICTIONS.jsonl.gz',PF.OLD/'PREDICTIONS.jsonl.gz'):
        bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    scored={r['id']:r for report in scores.values()for r in report['rows']}
    # One maximum per arm/partition shows regressions as well as improvements.
    ids=[]
    for partition in ('REMAINDER','TEST'):
        for arm in EXP.ARMS:
            fid=max(scores[partition]['rows'],key=lambda r:r['metrics'][arm]['translation_cm'])['id']
            if fid not in ids:ids.append(fid)
    font=DRAW._font();output.mkdir(parents=True);assets=output/'assets';assets.mkdir()
    if font.is_file():bindings[str(font)]=C.sha(font)
    cases=[];table=[]
    for fid,t in tasks.items():
        image=C.ROOT/t['image']
        if not image.is_file()or C.sha(image)!=t['image_sha256']:raise ValueError('Bound actual RGB required')
        pf=posefix[fid]
        if pf['image']!=t['image']or pf['image_sha256']!=t['image_sha256']:
            raise ValueError('PoseFix and current actual RGB differ')
        poses={'INITIAL_N3':_initial_pose(t),'POSEFIX_RAW':pf['actual_pose'],
            **{a:rows[fid]['arms'][a]['actual_pose']for a in EXP.ARMS}}
        metric=ARCHIVE.REF.metric_for_reference if t['partition']=='REMAINDER'else pose_metric
        values={a:metric(poses[a],refs[fid])for a in METHODS}
        for a in EXP.ARMS:
            if values[a]!=scored[fid]['metrics'][a]:raise ValueError('Displayed control metric differs from sealed score')
        table.append(dict(id=fid,partition=t['partition'],session=t['session'],states=t['spec']['states'],
            metrics=values,accepted={a:rows[fid]['arms'][a]['accepted']for a in EXP.ARMS},
            original_RGB_relative_url=_link(image,output),image_sha256=t['image_sha256']))
        if fid not in ids:continue
        image_bgr=cv2.imread(str(image))
        if image_bgr is None:raise ValueError('Actual image decode failed')
        rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB);K=t['spec']['K'];ref=refs[fid]
        qref=None if ref['R']is None else C.project(C.cuboid(*ref['xyz']),ref['R'],ref['t'],K)
        projected={a:_pose_projection(poses[a],K)for a in METHODS}
        mask=np.zeros(rgb.shape[:2],bool)
        # The actual saved outline is shown as its measured samples, not a new mask.
        actual=np.asarray(t['outline'].get('observed_boundary_samples_xy',[]),float).reshape((-1,2))
        case=dict(id=fid,partition=t['partition'],states=t['spec']['states'],metrics=values,
            stored_poses=poses,model_projections={a:q.tolist()for a,q in projected.items()},
            postseal_reference=ref,source_RGB_relative_url=_link(image,output),
            normal_controls={a:rows[fid]['normal_two_stage_audit'][a]for a in EXP.ARMS[1:]},stages=[])
        # Tight actual-photo crop improves readability for tiny distant pallets.
        points=np.vstack([np.asarray(t['finish_input']['_original_source']['initial_geometry']['initial_projection_xy']),actual])
        points=points[np.isfinite(points).all(1)]
        if len(points):
            lo=points.min(0);hi=points.max(0);padding=np.maximum(15,(hi-lo)*.3)
            bounds=(max(0,lo[0]-padding[0]),min(rgb.shape[1],hi[0]+padding[0]),
                    min(rgb.shape[0],hi[1]+padding[1]),max(0,lo[1]-padding[1]))
        else:bounds=(0,rgb.shape[1],rgb.shape[0],0)
        fig,axes=plt.subplots(2,len(METHODS),figsize=(20,8),squeeze=False)
        for col,a in enumerate(METHODS):
            for row_index in range(2):
                ax=axes[row_index,col];ax.imshow(rgb);DRAW._wire(ax,projected[a],COLORS[a],width=1.7)
                if qref is not None:DRAW._wire(ax,qref,'white',ls=':',width=2.0)
                if row_index:ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[2:])
                else:ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0)
                ax.set_aspect('equal');ax.set_axis_off()
                rtext='미확정'if values[a]['rotation_deg']is None else f"{values[a]['rotation_deg']:.2f}°"
                ax.set_title(f"{NAMES[a]}\n{values[a]['translation_cm']:.2f}cm / {rtext}",fontsize=10)
        fig.suptitle(fid+'\n색 wire=각 저장 자세의 모델투영; 흰 점선=봉인 후 참조, 영상 측정점 아님',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.91));name=f'case{len(cases)}_methods_actual.png'
        fig.savefig(assets/name,dpi=150,bbox_inches='tight');plt.close(fig)
        case['comparison_asset']='assets/'+name
        initial=t['finish_input']['_original_source']['initial_geometry'];hidden=initial['initial_self_hidden']
        source=t['finish_input']['_original_source'];raw_source=source.get('original_raw_points')
        # Access raw by individual non-S index only; never coerce a whole array.
        raw={k:raw_source[k]for k in range(8)if not hidden[k]and t['spec']['states'][k]!='S'}if raw_source is not None else {}
        descriptions=(('① 최초 저장 예측과 조건부 기하', '예측 ×와 coarse 모델투영 △는 구분. S 원좌표는 표시하지 않음.'),
            ('② 실제 관측 선과 BI 상태','녹색은 저장 실제 RGB 지지선/표본, B/I 역할은 초기 기하 조건부.'),
            ('③ O/U·접선 범위','O는 경험영역, U와 접선은 빌린 CAL 범위. 확률·실제 corner 보증 없음.'),
            ('④ 서로 다른 저장 자세 비교','아래 비교 사진에서 각 방법을 독립적으로 표시. 사후 참조는 이 비교에서만 사용.'))
        for index,(title,caption)in enumerate(descriptions[:3]):
            fig,ax=DRAW._base(rgb,title,expanded=index==2)
            if index==0:
                DRAW._points(ax,raw,'#61758a','x',prefix='예측q')
                DRAW._wire(ax,np.asarray(initial['initial_projection_xy']),DRAW.ORANGE,ls='--')
            elif index==1:
                if len(actual):ax.scatter(*actual.T,s=2,color=DRAW.GREEN,alpha=.55)
                DRAW._partial(ax,t['boundary_all']);PREVIOUS._visible_points(ax,t['spec']['visible'],{b['native_corner']for b in t['boundary_all']})
            else:
                DRAW._domains(ax,t['spec']['domains'],t['u_regions']);DRAW._partial(ax,t['boundary_all'],range_only=True)
            asset=assets/f'case{len(cases)}_stage{index+1}.png';DRAW._save(fig,asset,caption)
            case['stages'].append(dict(title=title,caption=caption,asset=str(asset.relative_to(output))))
        cases.append(case);bindings[t['image']]=t['image_sha256']
    summaries={p:scores[p]['summary']for p in scores}
    C.write(output/'REPORT_DATA.json',dict(schema='ACTUAL_NORMAL_CONTROL_COMPARISON_V16B',cases=cases,
        all238_table=table,summaries=summaries,bindings=bindings,actual_RGB=True,new_inference_or_fit=0,
        picture_selection_after_score_not_production_selection=True,default_e_not_replaced=True,
        normal_regularizer_correlated_not_new_image_measurement=True,remaining_nominal_reference_not_independent_metrology=True))
    summary_rows=[]
    for partition in ('TEST','REMAINDER','CAL'):
        for a in EXP.ARMS:
            s=summaries[partition][a]
            summary_rows.append(f'<tr><td>{partition}</td><td>{NAMES[a]}</td><td>{s["translation_cm"]["mean"]:.3f}</td><td>{s["translation_cm"]["max"]:.3f}</td><td>{s["translation_le5"]}</td><td>{s["successes_5cm_5deg"]}</td><td>{s["fallback_count"]}</td><td>{html.escape(s["max_translation_id"])}</td></tr>')
    sections=[]
    for case in cases:
        stage_html=''.join(f'<h3>{s["title"]}</h3><img src="{s["asset"]}"><p>{s["caption"]}</p>'for s in case['stages'])
        sections.append(f'<details><summary>{html.escape(case["id"])} · {case["partition"]}</summary>{stage_html}<h3>④ N3 / PoseFix / e / 두 법선 control</h3><img src="{case["comparison_asset"]}"><a href="{case["source_RGB_relative_url"]}">실제 원본 RGB</a></details>')
    metric_rows=[]
    for r in sorted(table,key=lambda v:(v['partition'],-v['metrics'][EXP.ARMS[0]]['translation_cm'])):
        values=''.join(f'<td>{r["metrics"][a]["translation_cm"]:.3f}</td>'for a in METHODS)
        metric_rows.append(f'<tr><td>{r["partition"]}</td><td>{html.escape(r["id"])}</td><td>{" ".join(r["states"])}</td>{values}<td><a href="{r["original_RGB_relative_url"]}">RGB</a></td></tr>')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>상단면 법선 사전 실제사진 비교 V16b</title><style>body{{margin:0;background:#edf2f6;color:#18344b;font-family:system-ui,sans-serif}}main{{max-width:1400px;margin:auto;padding:20px}}img{{display:block;max-width:100%;height:auto}}details,.note{{background:white;padding:16px;margin:16px 0;border-radius:10px}}summary{{cursor:pointer;font-weight:bold}}.scroll{{overflow:auto;max-width:100%}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #d6e0e8;text-align:left}}th{{white-space:nowrap}}td:nth-child(2){{overflow-wrap:anywhere;min-width:140px}}p{{line-height:1.6}}@media(max-width:500px){{main{{padding:9px}}details{{padding:9px}}}}</style><main><h1>상단면 법선만 제한한 두 고정 control — 실제사진 비교</h1><p>원본 e 기본값과 σ2° / CALmax σ4.13°를 분리했습니다. W/D 가족을 먼저 선택하고 그 가족 안에서 부분 관측+불확실 접선을 연속 최적화했습니다. 각 control은 238개 전부 봉인한 뒤 같은 참조로 계산했습니다.</p><p>이 법선은 초기 backbone 자세에서 온 상관된 사전입니다. 새 독립 영상 관측 또는 유일성 증거가 아닙니다. 초기 위치·깊이·native yaw를 직접 벌점 주지 않습니다. remaining143의 참조는 저장 nominal 주석 PnP이며 부호축 미확정·카메라 품질 UNKNOWN입니다. 독립 실측이나 새 무접촉 검증이 아닙니다.</p><div class="note scroll"><table><thead><tr><th>집합</th><th>방법</th><th>평균 cm</th><th>최대 cm</th><th>≤5cm</th><th>≤5cm·5°</th><th>fallback</th><th>최대 프레임</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div><p>CAL은 전체 CAL 모델 재사용 진단이며 독립 LOGO 성능으로 해석하지 않습니다. 아래 사진은 각 방법의 최대오차 사례를 봉인 후 선택했으며, 모델 선택·런타임 파라미터 결정에 쓰지 않았습니다. 색 선은 모델투영, 영상 KP 실측점이 아닙니다. 흰 점선은 사후 참조입니다.</p>{''.join(sections)}<div class="note"><a href="{_link(DRAW.OUTPUT/'index.html',output)}">원래49 성공 실제사진 보고서</a> · <a href="REPORT_DATA.json">수치·출처 JSON</a></div><details><summary>전체238 위치오차 표 — 실패·fallback 포함</summary><div class="scroll"><table><thead><tr><th>집합</th><th>프레임</th><th>상태0–7</th>{''.join('<th>'+NAMES[a]+' cm</th>'for a in METHODS)}<th>사진</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div></details></main></html>'''
    (output/'index.html').write_text(page)
    # A concrete actual-photo overview for embedding; no generated bitmap.
    if cases:
        import shutil
        shutil.copyfile(output/cases[0]['comparison_asset'],output/'overview_actual.png')
    E._verify_sources(protocol['protected_user_sha256'])
    for relative,digest in bindings.items():
        path=Path(relative)if Path(relative).is_absolute()else C.ROOT/relative
        if C.sha(path)!=digest:raise ValueError('Render evidence changed')
    C.write(output/'RENDER_SEAL.json',dict(source_sha256=C.sha(Path(__file__)),cases=len(cases),all238_table_rows=238,
        complete_prediction_seal=E._binding(folder/'PREDICTION_SEAL.json'),complete_score_seal=E._binding(folder/'SCORE_SEAL.json'),
        assets={str(p.relative_to(output)):C.sha(p)for p in output.rglob('*')if p.is_file()},
        actual_RGB_only=True,no_new_inference_or_pose_fit=True,no_default_method_replacement=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--experiment-folder',type=Path,default=EXPERIMENT);a=p.parse_args()
    render(a.output,a.experiment_folder)
