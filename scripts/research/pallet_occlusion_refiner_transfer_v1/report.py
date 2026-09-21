"""E1 galleries, population provenance, final gate report. No training."""
import html
import json
import os
import subprocess
from collections import Counter

import numpy as np

from . import run as R


def panel(fid, arm, pred, metric, truth, image):
    base=R.O.top(pred['R0'][fid]);p=R.O.top(pred[arm][fid]);gt=np.array(truth['gt']);valid=truth['valid']
    content=f'<image href="{html.escape(image)}" width="640" height="480"/>'
    if p:
        points=np.array(p['keypoints_xy']);initial=np.array(base['keypoints_xy'])
        for a,b in R.S.G.EDGES:
            if np.isfinite(points[[a,b]]).all():
                content+=f'<line x1="{points[a,0]}" y1="{points[a,1]}" x2="{points[b,0]}" y2="{points[b,1]}" stroke="#38baff" stroke-width="1.5"/>'
        for j in range(8):
            x,y=points[j];ix,iy=initial[j]
            if not np.isfinite([x,y,ix,iy]).all():continue
            content+=f'<line x1="{ix}" y1="{iy}" x2="{x}" y2="{y}" stroke="#ffa64d"/><circle cx="{x}" cy="{y}" r="3" fill="#38baff"/><text x="{x+4}" y="{y-4}" fill="white" font-size="13">P{j}</text>'
    for j in range(8):
        if valid[j]:
            x,y=gt[j];content+=f'<path d="M{x-4},{y}h8 M{x},{y-4}v8" stroke="#69ff63" stroke-width="2"/>'
    errors=metric.get('canonical_errors',[])
    text=' / '.join(f'G{j}: {e:.1f}' for j,e in enumerate(errors) if e is not None)
    move=np.linalg.norm(np.array(p['keypoints_xy'])[:8]-np.array(base['keypoints_xy'])[:8],axis=1).round(2).tolist() if p and base else []
    return f'<article><h4>{arm}</h4><p>mean {metric.get("frame_mean_px",0):.2f}px · matched {metric["matched"]} · branch {metric.get("branch")}</p><svg viewBox="0 0 640 480">{content}</svg><small>canonical GT error(px): {text}<br>native P0..P7 movement(px): {move}</small></article>'


def gallery():
    frozen=R.read(R.RAW/'FROZEN_PREDICTIONS.json');pred=frozen['predictions']
    frame=R.read(R.RAW/'E1_FRAME_METRICS.json');metrics=frame['metrics'];truth=frame['truth_for_display_only']
    heads=R.read(R.DOC/'E1_CANDIDATE_HEADROOM.json')['populations']
    roles=R.read(R.DOC/'DATA_ROLE_MANIFEST.json')['populations']
    records={r['id']:r for r in frozen['records']};_,conditions=R.condition_map();manifest={};links=[]
    for name,pop in roles.items():
        ids=pop['ids'];delta={fid:metrics['REPLAY'][fid]['frame_mean_px']-metrics['R0'][fid]['frame_mean_px'] for fid in ids}
        groups={'improvement':sorted(ids,key=lambda fid:(delta[fid],fid))[:10],
                'damage_or_least_improvement':sorted(ids,key=lambda fid:(-delta[fid],fid))[:10],
                'random_seed1':np.random.default_rng(1).choice(sorted(ids),min(10,len(ids)),replace=False).tolist()}
        manifest[name]=dict(groups=groups,ranking='Replay minus R0 frame mean error; lower is better; no claim all top10 improve/damage',
            overlap_between_groups_allowed=True,requested10_but_population_may_be8=True)
        cards=[]
        for group,selected in groups.items():
            cards.append('<h2>'+group+'</h2>')
            for fid in selected:
                r=records[fid];image=os.path.relpath(R.ROOT/r['image']['path'],R.OUT)
                meta=conditions.get(r['image']['path'],{})
                base=R.O.top(pred['R0'][fid]);confidence=base['keypoints_conf'][:8] if base else []
                arms=heads[name]['available_candidates'];panels=[]
                for arm in arms:panels.append(panel(fid,arm,pred,metrics[arm][fid],truth[fid],image))
                oracle=frame['oracle'][name][fid]
                cards.append(f'<section><h3>{html.escape(fid)}</h3><p>[확인] Replay−R0 {delta[fid]:+.2f}px · diagnostic oracle={oracle} · occlusion={meta.get("occlusion","unknown")} / truncation={meta.get("truncation","unknown")}</p><p>R0 kp confidence (not corrected confidence): {confidence}</p><div class="panels">{"".join(panels)}</div></section>')
        page='<!doctype html><meta charset="utf-8"><title>E1 '+name+'</title><style>body{font:16px system-ui;background:#101c22;color:white;margin:20px}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}svg{width:100%}small{overflow-wrap:anywhere}section{border-top:2px solid #789;margin:25px 0}a{color:#8bd8ff}</style>'
        page+=f'<h1>E1 {name}</h1><p>[확인] 보정 전후 frozen 후보, 학습 없음. 초록+ = GT/reference, 파랑 = 출력, 주황 = R0부터 이동. GREEN150만 manual-only GT. 나머지는 수동/unknown/PnP reference 혼합. GT branch는 채점에만 사용.</p><p>[확인] 그룹은 Replay−R0 평균오차 기준이며 같은 프레임이 여러 그룹에 나올 수 있음. E2/A11은 미실행이므로 표시하지 않음.</p>'+''.join(cards)
        R.freeze(R.OUT/f'E1_{name}.html',page);links.append(f'<li><a href="E1_{name}.html">{name}</a></li>')
    R.freeze(R.OUT/'E1_GALLERY_SELECTION.json',manifest)
    R.freeze(R.OUT/'index.html','<!doctype html><meta charset="utf-8"><h1>[확인] E1 frozen candidate 진단</h1><ul>'+''.join(links)+'</ul><p>[확인] E2는 clean pool gate에서 미실행. A11 결과 없음.</p><a href="geometry.html">E1 geometry overlays</a>')
    # CAD hidden-only before/after overlays with failure reasons.
    geo=R.read(R.DOC/'E1_GEOMETRY_RECOVERABILITY.json');cards=[]
    for row in geo['rows']:
        fid=row['id'];arm={'R0':'R0','N3_DIM_SYM_seed1':'N3','REPLAY_RAW':'REPLAY'}[row['arm']]
        image=os.path.relpath(R.ROOT/records[fid]['image']['path'],R.OUT)
        before=R.O.top(pred[arm][fid])['keypoints_xy'];after=R.O.top(pred[arm+'_PNP'][fid])['keypoints_xy']
        overlay=R.S.svg(image,before,after,truth[fid]['gt'],truth[fid]['valid'],row['hidden'])
        maxshift=max(v['max_output_shift_px'] for v in row['sensitivity'])
        cards.append(f'<section><h3>{fid} / {arm}</h3><p>[확인] {row["reason"]}; used={row["used"]}; hidden={row["hidden"]}; visible hull={row["visible_hull_area_px2"]:.1f}px²; axis ±5% maximum output change={maxshift:.2f}px</p>{overlay}</section>')
    R.freeze(R.OUT/'geometry.html','<!doctype html><meta charset="utf-8"><title>E1 geometry</title><style>body{background:#14222c;color:white;font:16px system-ui}svg{max-width:800px;width:100%}section{margin:20px}</style><h1>[확인] CAD18 자기 가림 sanity</h1><p>[확인] external occlusion physical pair 없음. 파랑=대체후,주황=대체전 hidden점,초록=기존reference. GT를 PnP 입력으로 사용하지 않음. 기존 GT 일부는 PnP 유래 가능.</p>'+''.join(cards))


def report():
    head=R.read(R.DOC/'E1_CANDIDATE_HEADROOM.json');p=head['populations'];pool=R.read(R.DOC/'E1_PSEUDOLABEL_POOL.json')
    frame=R.read(R.RAW/'E1_FRAME_METRICS.json')['metrics'];_,conditions=R.condition_map();records=R.inputs()
    subgroup={};provenance={}
    for name,role in R.read(R.DOC/'DATA_ROLE_MANIFEST.json')['populations'].items():
        rr=[r for r in records if r['id'] in role['ids']];counts=Counter()
        for r in rr:
            ann=R.read(R.ROOT/r['annotation']['path'])['objects'][0]['keypoint_annotations']
            valid=frame['R0'][r['id']]['canonical_valid']
            for i,v in enumerate(valid):
                if v:counts[ann[i].get('source','MISSING')]+=1
        provenance[name]=dict(valid_scored_corners=sum(counts.values()),source_counts=dict(counts),frames=len(role['ids']),
                              conditions=dict(Counter(str((conditions.get(r['image']['path'],{}).get('occlusion','unknown'),conditions.get(r['image']['path'],{}).get('truncation','unknown'))) for r in rr)))
    for field in ('occlusion','truncation'):
        values=sorted({conditions[r['image']['path']][field] for r in records if r['kind']=='PLASTIC'})
        for value in values:
            ids=[r['id'] for r in records if r['kind']=='PLASTIC' and conditions[r['image']['path']][field]==value]
            subgroup[field+'='+value]={a:R.summarize([frame[a][fid] for fid in ids],[frame['R0'][fid] for fid in ids]) for a in ('R0','N2','N3','POSEFIX_SYNTH','REPLAY')}
    R.freeze(R.DOC/'E1_CONDITION_SUBGROUPS.json',dict(status='[확인]',subgroups=subgroup,population_provenance=provenance,
        CLEAN_NONCAD69_caveat='Historical name: means no external occlusion, not necessarily no truncation; see exact condition counts',
        GREEN_geometry_caveat='Historical source-camera mismatch audit exists; GREEN geometry was not used here. Do not certify geometry contract as resolved.'))
    lines=['# Occlusion robust refiner — 실행 결과','',
        '[확인] **E1 무학습 진단 수행. E2는 clean pool 무결성 gate에서 STOP, 학습 0 update.** E2 성능 FAIL이 아니라 입력 조건 미충족이다. E3~E6·commit·push는 실행하지 않았다.','',
        '## 문제와 이번 차이','',
        '[확인] 기존 clean8 학생 fine-tuning은 occlusion 평가에서 일관된 개선이 없었다. 이번 계획은 frozen R0를 유지하고 보정기만 학습하며, 가린 RGB에서 R0를 실제 재추론한 입력과 clean pseudo target을 분리하는 것이다. 기존 pose-only 실험을 이름만 바꿔 재실행하지 않았다.','',
        '[추정] 이 대조가 실행되면 artificial occlusion 입력과 source replay의 복원/보존 효과를 구분할 수 있다. 이번에는 gate에서 멈췄으므로 그 효과를 결론낼 수 없다.','',
        '## E1-A: 현재 후보로 얼마나 복구할 여지가 있는가','',
        '| [확인] 집합 | R0 PCK10 | N2 | N3 | 합성 PoseFix | Replay | 진단 oracle | 최고 실제 대비 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('PRIMARY_OCC96','CLEAN_NONCAD69','GREEN150_MANUAL','DEV72_REFERENCE_UNKNOWN','CAD18','SELECTED_CLEAN8'):
        s=p[name]['summary'];vals=[f'{100*s[a]["PCK"]["10"]:.2f}%' if a in s else '미제공' for a in ('R0','N2','N3','POSEFIX_SYNTH','REPLAY','B_ORACLE')]
        lines.append('| '+name+' | '+' | '.join(vals)+f' | +{p[name]["oracle_gain_vs_best_actual_pp"]:.2f}pp |')
    lines += ['', '[확인] CAD18/clean8 oracle에는 R0/N3/Replay의 기존 자기 가림 PnP 후보도 포함한다. 다른 집합에서는 이 후보가 없고 GREEN N3 저장본도 없어 포함하지 않았다. 모든 집합은 별도 분모이며 합산하지 않았다. 전체 지표와 후보 가용성은 E1_CANDIDATE_HEADROOM에 있다.', '',
        '[확인] occlusion96은 737코너, R0/N2/Replay 모두 동일 검출에서 매칭88/96이다. N2와 Replay PCK10은 같은47.49%지만 Replay는 R0 >20→≤10 복구4개 / R0 <5→>10 손상2개(좋은116개 중1.72%), N2는 복구0/손상0이다. P90은 R0 64.14px, N2 63.95px, Replay65.64px다.', '',
        '[확인] GREEN 수동681코너에서는 Replay가 PCK10 77.39%로 R0 82.23%보다 낮다. 좋은344코너 중14개(4.07%)를 손상시켰다. 진단 oracle도 최고 실제 대비+1.47pp이며 누락된 N3 후보에 대해서는 판단하지 않는다.', '',
        '[추정] 현재 확인한 후보 집합의 occlusion 선택 여지는 중간(+3.80pp)이며, 선택기만으로 큰 오류 전부를 해결할 수 있다는 근거는 없다. oracle도 >20px 꼬리가 남는다. 후보 생성/표현 개선 가능성과 target-domain 적응 가능성은 별도 질문이다.', '',
        '## E1-B: geometry','',
        '[확인] camera/pallet movement parity가 검증된 physical clean/occlusion pair는 감사한 자료에서 발견하지 못했다. CAD18×3후보의 자기 가림 출력54개를 재현했다. 40개 hidden-only 적용,14개 reliable visible point<6으로 원본 유지. 사용점 분포·재투영 residual·hidden set·각 치수축±5% 민감도를 기록했다.', '',
        '[확인] GT-visible external occlusion oracle은 미실행이다. 숨은 코너 GT 일부는 PnP 유래일 수 있다. 기하 sanity를 독립 물리 복구 검증으로 해석하지 않는다. GREEN의 과거 카메라 메타데이터 불일치를 이번 작업에서 해결했다고 주장하지 않는다.', '',
        '## E1-C 및 E2 gate','',
        f'[확인] 기존 pool/scout/expanded 후보 {pool["source_rows"]}행, 이미지 SHA 중복 제거 {pool["unique_candidate_images"]}장. 기존 메타데이터와 세션 분리까지 확인한 eligible clean 후보 {pool["eligible_before_inference"]}장. RGB_NOT_YET_REVIEWED나 condition 미기록을 clean으로 간주하지 않았다. 수도레이블 생성·E2 학습은 미실행이다.', '',
        '[확인] 기존 선택CAD8은 평가 세션 제외 규칙에 따라 이번 학습으로 자동 전환하지 않았다. 이전 실험의 역할 변경은 그 실험에 한정된 것이며 이번 지시를 덮어쓰지 않는다.', '',
        '## 최종 질문에 대한 답','',
        '1. [확인] clean pseudo-label로 보정기가 적응했는가? **미검증** — E2 미실행.',
        '2. [확인] occluded-R0 input이 CLEAN보다 좋은가? **미검증** — 기존 학생 결과로 대체 답변하지 않는다.',
        '3. [확인] source replay가 필요한가? **이번 대조에서는 미검증**. 이전 Replay 결과는 과거 별도 설정의 증거다.',
        '4. [확인] 새 보정기가 N2/N3/Replay보다 나은가? **새 모델 없음**. frozen 후보의 차이만 측정했다.',
        '5. [확인] 큰 오류 복구와 손상은? occlusion96 Replay 복구4/손상2, N2 복구0/손상0; GREEN Replay 복구1/손상14. 세부 분모와 모든 후보는 E1 표 참조.',
        '6. [확인] 실행을 막은 병목은 **조건 확인된 disjoint clean pool 부재**다. [추정] 기존 후보들의 제한된 oracle 여지는 후보 생성/표현 병목도 시사하지만 원인 확정은 아니다.',
        '7. [확인] 지금 E3/E5/representation 신규 학습으로 넘어가지 않는다. 먼저 평가와 분리된 clean 후보≥8장의 조건을 확인하거나 데이터 역할 변경에 대한 별도 지시가 필요하다. 새 코너 좌표 annotation이 필수라는 뜻은 아니다.', '',
        '## 산출물·한계','',
        '- [확인] [후보별 전체 수치](E1_CANDIDATE_HEADROOM.md), [기하 검사](E1_GEOMETRY_RECOVERABILITY.md), [pool 감사](E1_PSEUDOLABEL_POOL.md), [갤러리](../../../outputs/pallet_occlusion_refiner_transfer_v1/index.html).',
        '- [확인] 갤러리는 모집단별 개선10/악화 또는 최소개선10/seed1 랜덤10이다. clean8은 각8장만 제공한다. GT는 그림과 사후 점수에만 쓰며 좋은 사례만 보고하지 않는다.',
        '- [확인] CLEAN_NONCAD69는 역사적 이름이며 external occlusion 없음 집합이다. truncation까지 없다는 뜻은 아니며 E1_CONDITION_SUBGROUPS에 분리 집계했다.',
        '- [확인] 재사용 DEV·점별 unknown/PnP GT·후보 누락·독립 paired physical occlusion GT 부재가 한계다. 통계적 유의성 주장은 하지 않는다.',
        '- [확인] GPU 추론은 RTX3080에서 수행, 관측 최대58°C. 기존 모델·annotation·split·논문표 변경 없음. 기존 CuDNN workaround 경고는 있었으나 실행 완료했으며 패키지/시스템 변경 없음.', '']
    R.freeze(R.DOC/'RESULTS_KO.md','\n'.join(lines))


def audit():
    binding=R.read(R.DOC/'INPUT_BINDINGS.json');checked=0
    for b in list(binding['checkpoints'].values())+binding['inputs']+[binding['code']]:R.verify(b);checked+=1
    lock=R.read(R.DOC/'E1_PREDICTIONS_LOCK.json')
    for b in [lock['predictions']]+lock['sources']:R.verify(b);checked+=1
    cached=R.read(R.RAW/'CACHED_PREDICTIONS.json')
    for b in cached['sources']:R.verify(b);checked+=1
    from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved
    predictions=R.read(R.RAW/'FROZEN_PREDICTIONS.json')['predictions'];n=0
    for arm,rows in predictions.items():
        for fid,p in rows.items():assert_preserved(predictions['R0'][fid],p);n+=1
    # Previous frozen R0 metrics reproduce on all 344 frames; pooled denominators remain unchanged.
    old=R.read(R.ROOT/'data/pallet/results/pallet_type_selftrain_v1/EVAL_METRICS.json')['R0']
    baseline=R.read(R.RAW/'E1_FRAME_METRICS.json')['metrics']['R0'];parity=0
    for r in old:
        if r['id'] in baseline:
            q=baseline[r['id']];assert q['matched']==r['matched'];np.testing.assert_allclose(q['errors'],r['errors'],atol=1e-8);parity+=1
    completed=subprocess.run([sys.executable,'-m','pytest','-q','scripts/research/pallet_occlusion_refiner_transfer_v1/test_contracts.py','scripts/research/pallet_posefix_large_error_v1/test_core.py'],capture_output=True,text=True)
    assert completed.returncode==0,completed.stdout+completed.stderr
    R.freeze(R.DOC/'AUDIT.json',dict(status='[확인]',input_hashes_rechecked=checked,prediction_contract_checks=n,
        R0_historical_metric_parity_frames=parity,tests_output=completed.stdout,
        E2_tests='NOT_RUN: clean/occluded target parity, actual occluded R0 rerun, four-arm exposure parity; E2 gate STOP',
        training_updates=0,evaluation_filtering=0,GT_in_new_inference=False,old_artifact_mutation=False,
        gate=R.read(R.DOC/'E1_TO_E2_GATE.json'),
        code=[R.bind(R.Path(__file__)),R.bind(R.Path(__file__).parent/'test_contracts.py')],
        git_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_diff=subprocess.check_output(['git','diff','--stat'],text=True),auto_commit=False,auto_push=False))
    print('AUDIT_COMPLETE',checked,n,parity,flush=True)


if __name__=='__main__':
    import sys
    gallery();report();audit()
