"""Freeze four-arm predictions before GT scoring; report all contrasts and gates."""
import html
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from . import run as C
from . import pilot as P


def score():
    protocol=C.read(C.DOC/'E2_PROTOCOL.json');baseline=C.read(C.V.RAW/'FROZEN_PREDICTIONS.json')
    files={a:C.RAW/f'PREDICTIONS_{a}.json' for a in C.ARMS}
    fits={a:C.read(C.DOC/f'FIT_{a}.json') for a in C.ARMS}
    for fit in fits.values():C.verify(fit['checkpoint'])
    C.freeze(C.DOC/'E2_PREDICTIONS_LOCK.json',dict(status='[확인]',predictions={a:C.bind(f) for a,f in files.items()},
        checkpoints={a:f['checkpoint'] for a,f in fits.items()},baseline=C.bind(C.V.RAW/'FROZEN_PREDICTIONS.json'),GT_scoring_after_lock=True))
    # Only now access previously fixed evaluation targets.
    oldmetrics=C.read(C.V.RAW/'E1_FRAME_METRICS.json');truth=oldmetrics['truth_for_display_only']
    metadata={r['id']:r for r in protocol['eval_records']};ids=set(metadata)
    metrics={a:{fid:r for fid,r in rows.items() if fid in ids} for a,rows in oldmetrics['metrics'].items()}
    pred={a:{fid:r for fid,r in rows.items() if fid in ids} for a,rows in baseline['predictions'].items()}
    groups={r['object_type']:r['permutations'] for r in C.read(C.N.C.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    for arm,file in files.items():
        payload=C.read(file);pp=payload['predictions'];assert set(pp)==ids;pred[arm]=pp;metrics[arm]={}
        for fid,p in pp.items():
            base=metrics['R0'][fid];r=metadata[fid];t=truth[fid];candidate=C.P.top(p)
            C.assert_preserved(pred['R0'][fid],p)
            points=np.full((9,2),np.nan) if candidate is None else candidate['keypoints_xy']
            result=C.V.M.measure(points,t['gt'],t['valid'],groups[r['object_type']],[480,640],base['matched'],base['detected'])
            metrics[arm][fid]=dict(id=fid,session=r['session'],**result)
    summary={};contrasts={}
    for name,fids in protocol['populations'].items():
        available=[a for a in ['R0','N2','N3','POSEFIX_SYNTH','REPLAY',*C.ARMS] if set(fids)<=set(metrics[a])]
        reference=[metrics['R0'][fid] for fid in fids]
        summary[name]={a:C.V.summarize([metrics[a][fid] for fid in fids],reference) for a in available}
        contrasts[name]={}
        for label,a,b in [('C1_OCC_no_source','A01','A00'),('C2_OCC_with_source','A11','A10'),
                          ('C3_source_clean','A10','A00'),('C4_source_occ','A11','A01')]:
            s,t=summary[name][a],summary[name][b]
            contrasts[name][label]=dict(after=a,before=b,PCK10_delta_pp=100*(s['PCK']['10']-t['PCK']['10']),
                PCK20_delta_pp=100*(s['PCK']['20']-t['PCK']['20']),median_delta_px=s['matched_pooled_corner8_median_px']-t['matched_pooled_corner8_median_px'],
                P90_delta_px=s['matched_pooled_corner8_P90_px']-t['matched_pooled_corner8_P90_px'],
                damage_delta=s['recovery_damage']['damaged']-t['recovery_damage']['damaged'])
        contrasts[name]['C5_A11_vs_existing']={a:dict(PCK10_delta_pp=100*(summary[name]['A11']['PCK']['10']-summary[name][a]['PCK']['10']),
            P90_ratio=summary[name]['A11']['matched_pooled_corner8_P90_px']/summary[name][a]['matched_pooled_corner8_P90_px']) for a in ('N2','N3','REPLAY') if a in summary[name]}
    C.freeze(C.RAW/'E2_FRAME_METRICS.json',metrics)
    C.freeze(C.DOC/'E2_REAL_RESULTS.json',dict(status='[확인]',summary=summary,contrasts=contrasts,populations=protocol['populations'],
        note='Historical population names retained for traceability; actual frames reduced by recording exclusions. Not same denominator as prior papers.',
        no_eval_filtering=True,GT_only_for_scoring=True))
    source={'INITIAL_SYNTH':C.read(C.DOC/'SOURCE_BEFORE.json'),**{a:C.read(C.DOC/f'SOURCE_{a}.json') for a in C.ARMS}}
    C.freeze(C.DOC/'E2_SOURCE_RESULTS.json',dict(status='[확인]',models=source))
    C.freeze(C.DOC/'E2_FITS.json',fits)
    return protocol,summary,contrasts,source,pred,metrics,truth


def decision(summary,source):
    occ=summary['PRIMARY_OCC96'];clean=summary['CLEAN_NONCAD69']
    comparator=max(['N2','N3','REPLAY'],key=lambda a:occ[a]['PCK']['10'])
    a=occ['A11'];checks={}
    def check(name,actual,op,threshold):
        passed=False if actual is None else actual>=threshold if op=='>=' else actual<=threshold
        checks[name]=dict(actual=actual,op=op,threshold=threshold,pass_check=bool(passed))
    check('occlusion_PCK10_delta_vs_R0_pp',100*(a['PCK']['10']-occ['R0']['PCK']['10']),'>=',3.)
    check('occlusion_PCK10_delta_vs_best_existing_pp',100*(a['PCK']['10']-occ[comparator]['PCK']['10']),'>=',0.)
    check('occlusion_P90_ratio_vs_best_existing',a['matched_pooled_corner8_P90_px']/occ[comparator]['matched_pooled_corner8_P90_px'],'<=',1.05)
    check('clean_PCK10_delta_vs_N2_pp',100*(clean['A11']['PCK']['10']-clean['N2']['PCK']['10']),'>=',-1.)
    check('source_clean_PCK10_delta_pp',100*(source['A11']['clean']['PCK10']-source['INITIAL_SYNTH']['clean']['PCK10']),'>=',-1.)
    check('source_clean_P90_ratio',source['A11']['clean']['P90_px']/source['INITIAL_SYNTH']['clean']['P90_px'],'<=',1.10)
    for name,rows in summary.items():
        d=rows['A11']['recovery_damage'];check(name+'_normal_damage_rate',d['damage_rate'],'<=',.01)
    failures=[k for k,v in checks.items() if not v['pass_check']]
    recovery=checks['occlusion_PCK10_delta_vs_R0_pp']['pass_check'] and checks['occlusion_PCK10_delta_vs_best_existing_pp']['pass_check']
    preservation=all(v['pass_check'] for k,v in checks.items() if 'normal_damage' in k or 'clean_PCK10' in k or 'source_' in k)
    status='PASS_E2' if not failures else 'PARTIAL' if recovery or preservation else 'FAIL_E2'
    labels=[]
    if recovery and not preservation:labels.append('RECOVERY_OK_PRESERVATION_FAIL')
    if preservation and not recovery:labels.append('PRESERVATION_OK_RECOVERY_FAIL')
    if not recovery and not preservation:labels.append('RECOVERY_AND_PRESERVATION_FAIL')
    if not checks['occlusion_P90_ratio_vs_best_existing']['pass_check']:labels.append('TAIL_FAIL')
    if any(not v['pass_check'] for k,v in checks.items() if k.startswith('source_')):labels.append('SOURCE_FORGETTING')
    result=dict(status='[확인]',decision=status,checks=checks,failed=failures,failure_axes=labels,
        primary_comparator=comparator,thresholds_are_pilot_not_statistical=True,
        global_final_model_changed=False,followup_training_authorized=False,
        process_deviation='A00 began before test collection issue resolved; paused, all10 checks passed, resumed same state. See EXECUTION_NOTE.md')
    C.freeze(C.DOC/'E2_DECISION.json',result);return result


def gallery(protocol,pred,metrics,truth):
    from scripts.research.pallet_occlusion_refiner_transfer_v1.report import panel
    _,conditions=C.V.condition_map();meta={r['id']:r for r in protocol['eval_records']};selection={};links=[]
    for name in ('PRIMARY_OCC96','CLEAN_NONCAD69','GREEN150_MANUAL'):
        ids=protocol['populations'][name]
        delta={fid:metrics['A11'][fid]['frame_mean_px']-metrics['R0'][fid]['frame_mean_px'] for fid in ids}
        groups=dict(improvement=sorted(ids,key=lambda fid:(delta[fid],fid))[:10],
                    damage_or_least_improvement=sorted(ids,key=lambda fid:(-delta[fid],fid))[:10],
                    random_seed1=np.random.default_rng(1).choice(sorted(ids),min(10,len(ids)),replace=False).tolist())
        selection[name]=groups;cards=[]
        for group,fids in groups.items():
            cards.append('<h2>'+group+'</h2>')
            for fid in fids:
                record=meta[fid];image=os.path.relpath(C.ROOT/record['image']['path'],C.OUT);condition=conditions.get(record['image']['path'],{})
                panels=[panel(fid,a,pred,metrics[a][fid],truth[fid],image) for a in ('R0','N2','REPLAY','A11')]
                raw=C.P.top(pred['R0'][fid]);confidence=raw['keypoints_conf'][:8] if raw else []
                cards.append(f'<section><h3>{html.escape(fid)}; A11−R0 {delta[fid]:+.2f}px</h3><p>[확인] occlusion={condition.get("occlusion","unknown")}; truncation={condition.get("truncation","unknown")}; R0 kp confidence={confidence}</p><div class="panels">{"".join(panels)}</div></section>')
        C.freeze(C.OUT/f'{name}.html','<!doctype html><meta charset="utf-8"><title>E2 '+name+'</title><style>body{font:16px system-ui;background:#10212b;color:white;margin:20px}.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}svg{width:100%}small{overflow-wrap:anywhere}section{border-top:2px solid #789;margin:30px 0}</style><h1>'+name+f' / {len(ids)} images</h1><p>[확인] R0 / N2 / 기존Replay / A11. 초록+는GT/reference, 파랑은예측, 주황은R0부터 이동. 그룹은A11−R0 평균오차; 그룹 중복가능. GREEN만manual-only, 나머지legacy reference. 후보번호는native, 오차는whole-object branch의canonical GT번호.</p>'+''.join(cards))
        links.append(f'<li><a href="{name}.html">{name}: {len(ids)}장 평가의 개선/악화/랜덤 사례</a></li>')
    previews=''.join(f'<figure><figcaption>{p.stem}: clean / artificial occlusion</figcaption><img style="max-width:100%" src="augmentation/{p.name}"></figure>' for p in sorted((C.OUT/'augmentation').glob('*.png')))
    C.freeze(C.OUT/'index.html','<!doctype html><meta charset="utf-8"><h1>[확인] DAY264 보정기 2×2 실험</h1><ul>'+''.join(links)+'</ul><h2>[확인] 실제 학습 입력 예시</h2>'+previews)
    C.freeze(C.OUT/'GALLERY_SELECTION.json',dict(status='[확인]',selection=selection,ranking='A11−R0 frame mean, deterministic seed1, no best-only selection'))


def audit(protocol):
    lock=C.read(C.DOC/'E2_INPUT_LOCK.json');all_traces={a:C.read(C.RAW/f'TRACE_{a}.json') for a in C.ARMS}
    for a,trace in all_traces.items():
        assert len(trace)==300
        assert [r['real_ids'] for r in trace]==[r['real_ids'] for r in all_traces['A00']]
        assert [r['real_supervised'] for r in trace]==[r['real_supervised'] for r in all_traces['A00']]
        fit=C.read(C.DOC/f'FIT_{a}.json');assert fit['BN_buffers_exact'] and fit['updates']==300 and fit['real_exposures']==2400
        C.verify(fit['checkpoint'])
    for b in C.read(C.DOC/'DATA_ROLE_MANIFEST.json')['checkpoint_bindings'].values():C.verify(b)
    for b in lock['paired_files']:C.verify(b)
    test=subprocess.run([sys.executable,'-m','pytest','-q','scripts/research/pallet_occlusion_refiner_transfer_v2/test_contracts.py'],capture_output=True,text=True)
    assert test.returncode==0,test.stdout+test.stderr
    sources=C.read(C.N.DOC/'INPUT_LOCK.json')
    for b in sources['cache_bindings']+[sources['orders']]:C.verify(b)
    C.freeze(C.DOC/'AUDIT.json',dict(status='[확인]',four_arm_actual_real_order_parity=True,four_arm_target_mask_parity=True,
        steps_per_arm=300,real_exposures_per_arm=2400,unique_real_exposed=len({fid for r in all_traces['A00'] for fid in r['real_ids']}),
        frozen_detector_checkpoint_hash_unchanged=True,BN_buffers_exact=True,failed_eval_frames_removed=0,
        GT_training_or_inference_input=False,teacher_training_same_recording_disclosed=True,
        tests_output=test.stdout,original_split_and_labels_untouched=True,auto_commit=False,auto_push=False,
        execution_order_deviation=C.bind(C.DOC/'EXECUTION_NOTE.md'),code=C.bind(Path(__file__)),
        source_metadata=C.bind(C.N.DOC/'INPUT_LOCK.json'),protocol=C.bind(C.DOC/'E2_PROTOCOL.json')))


def markdown(protocol,summary,contrasts,source,d):
    pool=C.read(C.DOC/'E1_PSEUDOLABEL_POOL.json');lock=C.read(C.DOC/'E2_INPUT_LOCK.json')
    lines=['# DAY264 clean pseudo → refiner adaptation 2×2','',f'[확인] **판정: {d["decision"]}**. 실패 축: `{d["failure_axes"]}`. 기존 최종 모델은 변경하지 않았다.','',
        '## 설정과 데이터','',
        '[확인] 사용자가 지정한 낮 플라스틱 표시1~264 (`005480.png`~`005743.png`)를 고정했다. 264장 중253장이 기존필터 통과,11장은raw confidence에서 제외됐다. 통과를정답으로 해석하지 않는다. 연속된 한촬영본이며253개의 독립장면이 아니다.', '',
        '[확인] 기존Replay와 자기 가림PnP로 만든 clean 타깃을 네arm에서 동일하게 사용했다. paired crop 공통support mask도 고정했다. 원영상좌표로 같은타깃이며 crop별좌표는 roundtrip1e-4px로 검증했다. 253장 중185장에 기존recipe의인공가림을 적용했고, OCC입력은 모든253장에서 실제 R0를 재추론했다.', '',
        '[확인] R0는동결, synthetic-only PoseFix에서동일초기화·seed1·300update·real8·TFAdam1e-4·BN통계고정. sourcearm만 합성8을추가했다. 네arm 실사2400노출을동일하게유지했다. compute-matched아님. 마지막체크포인트만평가했다.', '',
        '[확인] REC_001 촬영 전체를 이번 평가에서 제외해 기존 플라스틱 44장이 빠졌다. Replay 수동 학습 세션도 평가에서 제외해 총 278장(비초록 128 + 초록 150)을 평가했다. 기존 가림 96장/비가림 69장 모집단은 이번 평가에서 각각 93장/17장이다. 기존 split/논문표는 변경하지 않았다.', '',
        '[확인] Replay teacher가 학습했던목재6장은같은DAY recording에서왔다. 이점은teacher-independent adaptation이아니다. 선택264장과동일teacher TRAIN이미지SHA중복은0이고,이번평가와학습 recording/image중복은0이다.', '',
        '[확인] A00=CLEAN/no source, A01=OCC/no source, A10=CLEAN/source, A11=OCC/source.','']
    for name,rows in summary.items():
        lines += ['## '+name+f' (actual {len(protocol["populations"][name])} frames)','',
            '| [확인] arm | PCK10 % | PCK20 % | median px | P90 px | >20px count (%) | recovery/hard | damage/good | match |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm,s in rows.items():
            rd=s['recovery_damage'];lines.append(f'| {arm} | {100*s["PCK"]["10"]:.2f} | {100*s["PCK"]["20"]:.2f} | {s["matched_pooled_corner8_median_px"]:.2f} | {s["matched_pooled_corner8_P90_px"]:.2f} | {s["gross20_count"]} ({100*s["gross20"]:.2f}) | {rd["recovered"]}/{rd["hard"]} | {rd["damaged"]}/{rd["good"]} | {s["matched"]}/{s["total_frames"]} |')
        lines += ['', '| [확인] contrast | PCK10 delta pp | P90 delta px |','|---|---:|---:|']
        for key,row in contrasts[name].items():
            if key.startswith('C5'):continue
            lines.append(f'| {key} | {row["PCK10_delta_pp"]:+.2f} | {row["P90_delta_px"]:+.2f} |')
        lines += ['']
    lines += ['## 합성 source heldout 256장','', '| [확인] arm | input | PCK10 % | PCK20 % | median px | P90 px | recovery/hard | damage/good |','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,modes in source.items():
        for mode,s in modes.items():lines.append(f'| {arm} | {mode} | {100*s["PCK10"]:.2f} | {100*s["PCK20"]:.2f} | {s["median_px"]:.2f} | {s["P90_px"]:.2f} | {s["recovered"]}/{s["hard"]} | {s["damaged"]}/{s["good"]} |')
    lines+=['','## 사전 pilot gate','',f'[확인] occlusion 기존 comparator: {d["primary_comparator"]}; clean comparator:N2; source comparator:초기syntheticPoseFix. 통계적비열등성검정아님.','',
        '| [확인] check | actual | threshold | pass |','|---|---:|---|---|']
    for name,c in d['checks'].items():lines.append(f'| {name} | {c["actual"]} | {c["op"]} {c["threshold"]} | {c["pass_check"]} |')
    occ=contrasts['PRIMARY_OCC96'];lines+=['','## 질문별 해석','',
        '[확인] 실제로253개의clean pseudo-label로보정기4개를각300update학습했다. 학습완료와실사가림일반화는별개이며위93장결과로판단한다.',
        f'[확인] OCC입력효과: source없이A01−A00 PCK10 {occ["C1_OCC_no_source"]["PCK10_delta_pp"]:+.2f}pp; source있이A11−A10 {occ["C2_OCC_with_source"]["PCK10_delta_pp"]:+.2f}pp.',
        f'[확인] source효과: CLEAN에서A10−A00 {occ["C3_source_clean"]["PCK10_delta_pp"]:+.2f}pp; OCC에서A11−A01 {occ["C4_source_occ"]["PCK10_delta_pp"]:+.2f}pp. 정상점·합성보존효과는해당표를별도로본다.',
        '[확인] N2/N3/Replay대비A11비교,복구/손상,전체지표를모두보고했다. GREEN N3저장본은없어N2/Replay와만비교한다. 좋은arm하나만골라기여라고주장하지않는다.',
        '[확인] 이전 E1의 가림 96장 whole-frame candidate oracle은 PCK10 51.29%, 기존 최고 47.49%로 headroom +3.80pp였다. 이는 이번 93장과 분모가 다르다. [E1 보고서](../pallet_occlusion_refiner_transfer_v1/RESULTS_KO.md).',
        '[확인] 독립적인 물리적 clean/가림 정답 pair는 확보하지 못했다. E1 geometry 결과는 CAD18의 기존 자기 가림 PnP sanity이며 외부 가림 복원 검증이 아니다. 비초록 legacy reference와 GREEN manual GT는 별도 표로 유지했다. 이번 DEV는 반복 사용한 개발 평가이며 독립 최종 test 주장이 아니다.',
        '[추정] 연속한좁은촬영구간,teacher촬영노출,사용자clean근사판정,pseudo-target오차,인공가림과실제가림차이가남는다. 이실험만으로원인을한가지로확정하거나보정기전체의불가능을주장하지않는다.',
        '[확인] E3~E6은실행하지않는다. PASS가아니면동일DEV에서seed/lr/threshold구제탐색을하지않는다.', '',
        '## 감사와 시각화','',
        '[확인] 테스트수집import문제로모든자동검사완료전에A00학습이시작됐다. 프로세스를잠시정지하고테스트진입부만수정,10개검사통과후동일optimizer/order로재개했다. 설정변경이나checkpoint선택은없었다. 엄밀한실행순서일탈은 EXECUTION_NOTE.md에남겼다.',
        '[확인] 최종 감사에서 네 조건의 update/exposure/실사 순서/mask/BN/원본 checkpoint를 재검사했다. [AUDIT](AUDIT.json).',
        '[확인] [개선·악화·랜덤 갤러리 및 실제 가림입력](../../../outputs/pallet_occlusion_refiner_transfer_v2/index.html). GT는평가/표시전용,R0 confidence는보정후confidence아님.', '']
    C.freeze(C.DOC/'RESULTS_KO.md','\n'.join(lines))
    C.freeze(C.DOC/'NEXT_STAGE_PLAN.md',f'# 다음 단계\n\n[확인] 현재 {d["decision"]}. E3~E6 신규학습은실행하지않았다.\n\n[추정] 우선현재2×2의회복/손상/source꼬리를검토해야한다. 원인분리없이연속프레임수만늘리는것은다양성검증이아니다. 추가clean세션·물리적가림pair·표현변경중어느것이필요한지는현결과와함께다음지시에서결정한다.\n')


if __name__=='__main__':
    protocol,summary,contrasts,source,pred,metrics,truth=score()
    d=decision(summary,source);gallery(protocol,pred,metrics,truth);audit(protocol);markdown(protocol,summary,contrasts,source,d)
    print('REPORT_COMPLETE',d['decision'],d['failed'],flush=True)
