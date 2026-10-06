"""Saved-row LOCAL_CAP/JOINT8 comparison; no model/F or old namespace writes."""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from .common import ROOT, DOC, QUICK_DOC, LOSS_SETUP, OUTPUT, read, sha
from scripts.research.pallet_quick_pose_loss_20261006_v1 import reporting as Q

R = Q.R
NEW = ('LOCAL_CAP', 'JOINT8')
ORDER = ('RAW', 'N3', 'PoseFix', 'SOFT2D_GEO', 'HARD6D_GEO', 'SOFT6D', 'EXPECT6D', *NEW, 'GEO_ORACLE')
PAIRS = (('JOINT8', 'LOCAL_CAP'), ('LOCAL_CAP', 'N3'), ('JOINT8', 'N3'),
         ('LOCAL_CAP', 'PoseFix'), ('JOINT8', 'PoseFix'))
TOL = dict(R.TOLERANCES)


def bind(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=sha(path), bytes=path.stat().st_size)


def save(path, value, text=False):
    path = Path(path)
    assert path.resolve().is_relative_to(DOC.resolve()), 'new reporting destination changed'
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(value if text else json.dumps(R.clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def load_split(split, fits):
    ids, methods, inputs = Q.load_split(split, read(QUICK_DOC/'TRAIN_RECEIPTS.json'))
    old_receipt = read(QUICK_DOC/f'results/{split}_SOFT6D_seed1_EXECUTION.json')
    for method in NEW:
        path = DOC/f'results/{split}_{method}_seed1.jsonl'
        receipt_path = DOC/f'results/{split}_{method}_seed1_EXECUTION.json'
        receipt = read(receipt_path)
        assert receipt['status']=='DONE' and receipt['complete'] and receipt['full_denominator']==len(ids)
        assert receipt['GT_inference_access'] is False and receipt['readout']=='unchanged J'
        assert receipt['checkpoint_sha256']==fits[method]['checkpoint_sha256']
        assert receipt['rowfile_sha256']==sha(path)
        assert receipt['code_sha256']==sha(Path(__file__).with_name('evaluation.py'))
        assert receipt['protocol_sha256']==sha(DOC/'PROTOCOL.json')
        assert receipt['setup_sha256']==sha(LOSS_SETUP)
        for key in ('bank_binding','target_sha256'):
            assert receipt[key]==old_receipt[key], 'fixed evaluation input differs: '+key
        assert receipt['bank_artifact']['sha256']==old_receipt['bank_artifact']['sha256']
        rows = R.ordered([json.loads(line) for line in path.read_text().splitlines() if line.strip()], ids)
        for row, raw, oracle in zip(rows, methods['RAW'], methods['GEO_ORACLE']):
            assert row['session']==raw['session']
            assert type(row['selected_index']) is int and 0<=row['selected_index']<oracle['action_count']
            assert row['action_count']==oracle['action_count'] and row['oracle_selected_index']==oracle['selected_index']
            assert row['F_attempt'] and row['F_complete'] and row['F_available']==row['pose']['available']
            assert isinstance(row['RAW_native_equal'],bool) and isinstance(row['RAW_pose_metric_equal'],bool)
            assert row['corner']['evaluable']==raw['corner']['evaluable']
            if row['corner']['evaluable']:
                assert row['corner']['canonical_valid']==raw['corner']['canonical_valid']
        methods[method] = rows
        inputs.extend([bind(path),bind(receipt_path)])
    return ids, {name:methods[name] for name in ORDER}, inputs


def joint_frame_improvement(new, base):
    dt = R.delta_vector(new, base, 'pose', 'translation_cm')
    dr = R.delta_vector(new, base, 'pose', 'rotation_deg')
    signed = collections.Counter()
    for t, r in zip(dt, dr):
        if not np.isfinite(t) or not np.isfinite(r):
            signed['UNAVAILABLE']+=1
            continue
        st = -1 if t < -TOL['translation_cm'] else 1 if t > TOL['translation_cm'] else 0
        sr = -1 if r < -TOL['rotation_deg'] else 1 if r > TOL['rotation_deg'] else 0
        signed[f'T{st}_R{sr}']+=1
    result = dict(full_frames=len(new), common_pose_success=int(np.isfinite(dt).sum()),
        both_improve=signed['T-1_R-1'], mixed=signed['T-1_R1']+signed['T1_R-1'],
        both_worsen=signed['T1_R1'], with_one_or_both_numeric_ties=sum(v for k,v in signed.items() if k!='UNAVAILABLE' and ('T0_' in k or '_R0' in k)),
        both_tie=signed['T0_R0'], unavailable=signed['UNAVAILABLE'], signed_counts=dict(signed),
        scope='same original frame: NEW-minus-base T/R signs with inherited 1e-9 tolerances; distinct from marginal difference-of-medians')
    assert result['both_improve']+result['mixed']+result['both_worsen']+result['with_one_or_both_numeric_ties']+result['unavailable']==len(new)
    return result


def comparison(new, base, new_summary, base_summary, new_damage, base_damage, bootstraps, primary):
    paired = R.compare(new, base, bootstraps)
    axes = Q.axes(paired, new_summary, base_summary, primary)
    corner = Q.preservation_flags(new_summary, base_summary, new_damage, base_damage)
    pose_flags = {name:axes['B'][metric][statistic]['nonworse'] for name,metric,statistic in (
        ('ADD_median_nonworse','ADDsym_m','median'), ('ADD_P90_nonworse','ADDsym_m','P90'),
        ('T_P90_nonworse','translation_cm','P90'), ('R_P90_nonworse','rotation_deg','P90'))}
    pose_flags['failures_nonincrease'] = corner['flags']['pose_failures_nonincrease']
    pose = dict(flags=pose_flags, all_preserved=all(v is True for v in pose_flags.values()),
                violated=[k for k,v in pose_flags.items() if v is False],
                difference_of_medians={k:axes['B'][k]['median']['delta'] for k in R.METRICS},
                difference_of_P90={k:axes['B'][k]['P90']['delta'] for k in R.METRICS})
    lower = [f'{key}.{statistic}' for key in R.METRICS for statistic in ('mean','median')
             if axes['B'][key][statistic]['strictly_lower']]
    higher = [f'{key}.{statistic}' for key in R.METRICS for statistic in ('mean','median','P90')
              if axes['B'][key][statistic]['delta'] is not None and axes['B'][key][statistic]['delta']>TOL[key]]
    return dict(paired=paired, axes=axes, center_both_better=axes['B']['both_T_R_medians_strictly_lower'],
        joint_frame_improvement=joint_frame_improvement(new,base), pose_preservation=pose,
        corner_preservation=corner, tradeoff=bool(lower and (higher or pose['violated'] or corner['violated'])),
        improved_pose_mean_or_median=lower, worsening_pose_statistics=higher,
        overall_direction_and_preservation=axes['B']['both_T_R_medians_strictly_lower'] and pose['all_preserved'] and corner['all_preserved'])


def raw_equality(rows, raw):
    keys=('available',*R.METRICS,'ADDsym_normalized','IoU3D','yaw_deg')
    matches=[all(R.clean(a['pose'].get(k))==b['pose'].get(k) for k in keys) and
             a.get('final_hypothesis')==b.get('final_hypothesis') for a,b in zip(rows,raw)]
    for row,equal in zip(rows,matches):
        if 'RAW_pose_metric_equal' in row:
            assert row['RAW_pose_metric_equal']==equal
    count=sum(matches)
    has_native = all('RAW_native_equal' in r for r in rows)
    has_contract_pose = all('RAW_pose_equal' in r for r in rows)
    return dict(full_frames=len(rows), native_equal_count=sum(r['RAW_native_equal'] for r in rows) if has_native else None,
        native_equal_denominator=len(rows) if has_native else 0,
        pose_metric_and_final_hypothesis_equal_count=count, pose_metric_equal_denominator=len(rows),
        pose_equal_by_same_native_same_F_contract_count=sum(r['RAW_pose_equal'] is True for r in rows) if has_contract_pose else None,
        pose_same_input_contract_eligible=sum(r['RAW_pose_equal'] is not None for r in rows) if has_contract_pose else 0,
        direct_saved_R_t_equality='NOT_AVAILABLE: old saved RAW rows lack R/t and native coordinates',
        scope='new native equality is measured against original frame.q; stored T/R/ADD metrics and final hypothesis checked independently; same-native/same-F pose equality is a deterministic contract, not an independent saved R/t comparison; unavailable old native equality stays NA')


def analyze_split(split, fits):
    ids, methods, inputs = load_split(split,fits)
    group = 'scenario' if split=='SYNTH_HELDOUT' else 'session'
    primary = 'frame' if split=='SYNTH_HELDOUT' else 'session'
    boots = dict(frame=R.SharedBootstrap(ids))
    boots[group] = R.SharedBootstrap(ids,[r['session'] for r in methods['RAW']])
    summaries = {name:R.summarize(rows) for name,rows in methods.items()}
    damages = {name:R.M.damage([r['corner'] for r in methods['RAW']],[r['corner'] for r in rows]) for name,rows in methods.items()}
    for summary in summaries.values():
        assert summary['corner']['corners']==summaries['RAW']['corner']['corners']
    comparisons = {f'{new}_minus_{base}':comparison(methods[new],methods[base],summaries[new],summaries[base],damages[new],damages[base],boots,primary) for new,base in PAIRS}
    mirrored = comparison(methods['LOCAL_CAP'],methods['JOINT8'],summaries['LOCAL_CAP'],summaries['JOINT8'],damages['LOCAL_CAP'],damages['JOINT8'],boots,primary)
    return dict(full_frames=len(ids), primary_bootstrap=primary, bootstrap={k:b.meta() for k,b in boots.items()},
        summaries=summaries, RAW_canonical_damage=damages, comparisons=comparisons,
        LOCAL_CAP_minus_JOINT8=mirrored,
        RAW_equality={name:raw_equality(rows,methods['RAW']) for name,rows in methods.items()},
        oracle_recovery={name:Q.oracle_analysis(methods['RAW'],methods[name],methods['GEO_ORACLE']) for name in NEW},
        W_D_hypothesis_recovery={name:Q.wd_analysis(methods['RAW'],methods[name],methods['GEO_ORACLE']) for name in NEW},
        source_bindings=inputs)


def decision(method, splits):
    real = splits['REAL_DEV']['comparisons'][f'{method}_minus_N3']
    n3_gain = real['overall_direction_and_preservation']
    control = splits['REAL_DEV']['comparisons']['JOINT8_minus_LOCAL_CAP']
    control_gain = method=='JOINT8' and control['overall_direction_and_preservation'] and not n3_gain
    no_op = all(s['summaries'][method]['NoOp']==s['full_frames'] for s in splits.values())
    cis = {k:real['paired']['statistics'][k]['session']['CI95'] for k in R.METRICS}
    spans = {k:ci is None or ci[0]<=0<=ci[1] for k,ci in cis.items()}
    if no_op: label='NOOP_COLLAPSE'
    elif n3_gain: label='N3_GAIN_SIGNAL'
    elif control_gain: label='CONTROL_ONLY_GAIN'
    elif real['tradeoff']: label='TRADEOFF'
    else: label='NO_N3_GAIN'
    diagnostics = [name for name,flag in (('NOOP_COLLAPSE',no_op),('N3_GAIN_SIGNAL',n3_gain),
        ('CONTROL_ONLY_GAIN',control_gain),('TRADEOFF',real['tradeoff']),('NO_N3_GAIN',not n3_gain)) if flag]
    return dict(label=label, diagnostic_flags=diagnostics, N3_REAL_direction_and_preservation=n3_gain,
        JOINT8_REAL_control_only_gain=control_gain, both_REAL_T_R_medians_lower=real['center_both_better'],
        direction_only_unconfirmed=any(spans.values()), REAL_N3_paired_mean_CI_includes_zero_or_missing=spans,
        all_REAL_N3_pose_paired_mean_CI_upper_negative=all(ci is not None and ci[1]<0 for ci in cis.values()),
        control_direction_only_unconfirmed=any(control['paired']['statistics'][k]['session']['CI95'] is None or
            control['paired']['statistics'][k]['session']['CI95'][0]<=0<=control['paired']['statistics'][k]['session']['CI95'][1] for k in R.METRICS) if method=='JOINT8' else None,
        N3_rule_population='REAL_DEV only; no added synthetic gate',
        scope='one seed/reused DEV direction and preservation flags, not stable or independent confirmation')


def ci_text(stat):
    ci=stat['CI95']; return Q.fmt(stat['mean_paired_difference'],6)+(' [NA]' if ci is None else f' [{Q.fmt(ci[0],6)}, {Q.fmt(ci[1],6)}]')


def render(result):
    real=result['splits']['REAL_DEV']; gain=[m for m,d in result['decisions'].items() if d['N3_REAL_direction_and_preservation']]
    first='[확인] N3보다 REAL T/R 중앙값이 함께 낮고 모든 고정 보존 조건을 만족한 방법: '+(', '.join(gain) if gain else '없음')+'. 한 seed의 반복 사용 DEV 진단이다.'
    lines=['# 같은 SOFT6D 감독의 LOCAL_CAP / JOINT8 비교','',first,'',
        '[확인] LOCAL_CAP: **'+result['decisions']['LOCAL_CAP']['label']+'**; JOINT8: **'+result['decisions']['JOINT8']['label']+'**. CI가 0을 포함하면 방향 신호·미확증으로 표시하며 안정적 개선으로 부르지 않는다. 개별 개선·손상 flag는 최종 라벨과 함께 읽는다.','',
        '[확인] REAL의 동급 용량 비교에서 LOCAL_CAP가 JOINT8보다 T/R 중앙값·전체 보존 조건에 모두 우세한가: '+str(real['LOCAL_CAP_minus_JOINT8']['overall_direction_and_preservation'])+'. JOINT8가 LOCAL_CAP보다 같은 조건에 우세한가: '+str(real['comparisons']['JOINT8_minus_LOCAL_CAP']['overall_direction_and_preservation'])+'.','']
    for split,data in result['splits'].items():
        lines += [f'## {split}','',f"[확인] 전체 {data['full_frames']}장; 주 bootstrap {data['primary_bootstrap']}. 동일 원 ID/원 세션·시나리오의 draw를 공유한 10,000회, seed20260917이다. 합성은 원 scenario, REAL은 frame 보조 구간도 RESULTS.json에 남긴다.",'',
            '| seed1 방법 | T mean / med / P90 cm | R mean / med / P90 degree | ADD mean / med / P90 m | F 성공/전체 | NoOp |', '|---|---:|---:|---:|---:|---:|']
        for method in ORDER:
            summary=data['summaries'][method];p=summary['pose']
            vals=[' / '.join(Q.fmt(p[k][x],6 if k=='ADDsym_m' else 4) for x in ('mean','median','P90')) for k in R.METRICS]
            lines.append(f'| {method} | '+' | '.join(vals)+f" | {p['available']}/{p['total_frames']} | {summary['NoOp'] if summary['NoOp'] is not None else 'NA'} |")
        lines += ['', '| 방법 | 2D med / P90 px | PCK10 / gross20 | GT/관측 코너 | RAW good5→bad10 / bad20→good10 | RAW native / pose metric 같음 |','|---|---:|---:|---:|---:|---:|']
        for method in ORDER:
            c=data['summaries'][method]['corner'];d=data['RAW_canonical_damage'][method];eq=data['RAW_equality'][method]
            lines.append(f"| {method} | {Q.fmt(c['matched_pooled_corner8_median_px'])} / {Q.fmt(c['matched_pooled_corner8_P90_px'])} | {Q.fmt(c['PCK']['10'],6)} / {Q.fmt(c['gross20'],6)} | {c['corners']}/{c['observed_corners']} | {d['good5_to_bad10']}/{d['bad20_to_good10']} | {eq['native_equal_count'] if eq['native_equal_count'] is not None else 'NA'} / {eq['pose_metric_and_final_hypothesis_equal_count']} |")
        c=data['summaries']['RAW']['corner'];lines += ['',f"[확인] pose 분모는 {data['full_frames']}장, 2D evaluable {c['evaluable_frames']}·matched {c['matched']}장이다. 2D median/P90은 매칭 관측 코너 조건부, PCK/gross20은 전체 GT 코너+결측 페널티. 손상·복구는 프레임 수가 아닌 같은 canonical GT 코너 수다. old RAW 저장행에 R/t/native 좌표가 없어 독립 R/t 재대조는 NA이며, pose 같음 표는 저장 T/R/ADD·최종 가설의 정확 일치다. 새 native 같음은 frame.q와 직접 비교했고 동일 native·동일 F의 pose 동일성은 별도 계약 근거다.",'',
            '| NEW−대조 | T paired mean [95% CI] cm | R paired mean [95% CI] degree | ADD paired mean [95% CI] m | T/R/ADD difference-of-medians | T/R 동시개선 / 혼합 / 동시악화 / 동률포함 |', '|---|---:|---:|---:|---:|---:|']
        for name,comp in data['comparisons'].items():
            values=[ci_text(comp['paired']['statistics'][k][data['primary_bootstrap']]) for k in R.METRICS]
            med=' / '.join(Q.fmt(comp['pose_preservation']['difference_of_medians'][k],6) for k in R.METRICS)
            j=comp['joint_frame_improvement'];counts=f"{j['both_improve']} / {j['mixed']} / {j['both_worsen']} / {j['with_one_or_both_numeric_ties']}"
            lines.append(f'| {name} | '+' | '.join(values)+f' | {med} | {counts} |')
        lines += ['', '[확인] 음의 paired 차이는 낮은 오차 방향이다. marginal 중앙값 감소와 같은 프레임 T/R 동시 개선은 별개다. mean의 CI가 0을 포함한 경우 확정 개선이 아니며 pair별 pose·corner 보존 flag 및 악화 지표는 아래와 RESULTS.json에 유지했다.','',
            '| 비교 | T/R 중앙값 둘 다 낮음 | pose 보존 위반 | corner 보존 위반 | TRADEOFF |','|---|---|---|---|---|']
        for name,comp in data['comparisons'].items():
            lines.append(f"| {name} | {comp['center_both_better']} | {', '.join(comp['pose_preservation']['violated']) or '없음'} | {', '.join(comp['corner_preservation']['violated']) or '없음'} | {comp['tradeoff']} |")
        for method in NEW:
            rec=data['oracle_recovery'][method];wd=data['W_D_hypothesis_recovery'][method]['groups']['SWITCH'];eq=data['RAW_equality'][method]
            lines += ['',f"[확인] {method}: RAW native 동일 {eq['native_equal_count']}/{eq['native_equal_denominator']}, RAW 저장 pose 지표·가설 동일 {eq['pose_metric_and_final_hypothesis_equal_count']}/{eq['pose_metric_equal_denominator']}. oracle action {rec['exact_oracle_action']['count']}/{rec['exact_oracle_action']['denominator']}, headroom>1e-7m {rec['ratio_eligible_frames']}장에서 rho 중앙값 {Q.fmt(rec['ratio']['distribution']['median'])}·음수 {rec['ratio']['negative']}·>1 {rec['ratio']['over_one']}·오차허용 초과 oracle 위반 {rec['ratio']['oracle_bound_violations_beyond_tolerance']}. RAW→oracle 실제 최종 W/D 전환 {wd['frames']}장 중 {wd['final_hypothesis_recovered']}/{wd['common_F_available']} 회수. 생성 가설 라벨로 회수를 세지 않았다."]
        lines.append('')
    lines += ['## 실행과 해석 범위','',
        '[확인] 저장된 동일 FP32 SOFT6D target을 재사용해 각각 원래 seed1 초기값부터 6000 update를 수행했다. 두 추가 head는 실제 각 4680 parameter이며 공통 projection과 기존 frontend가 같다. 파라미터 수가 같아도 연산량·메모리·최적화 조건까지 같다고 주장하지 않는다. NoOp descriptor는 RAW 위치에서 새 patch를 뽑은 것이 아니라 기존 이동후보 pooled/coverage 집계와 null_pool을 사용했다.','']
    for method in NEW:
        fit=result['run_receipts']['methods'][method];probe=result['train_probes'][method]
        lines.append(f"[확인] {method}: {fit['updates']} update/{fit['exposures']} exposure, 실제 학습 루프 {fit['seconds']:.3f}s, 마지막 checkpoint {fit['checkpoint_sha256']}. 동일 고정 256 TRAIN ID probe 요약: {probe['summary']}. 기존 비용만 조회하고 F 추가0이며 전체 TRAIN 정확도·모델 선택 근거로 확대하지 않는다.")
    for method,dec in result['decisions'].items():
        if 'NOOP_COLLAPSE' in dec['diagnostic_flags']:
            lines += ['',f'[확인] {method}는 전체 합성+REAL 2304장 모두 index0을 골라 RAW로 돌아갔다. 새로운 보정 이득은 0이다.']
    lines += ['', '[확인] 한 seed와 반복 사용 DEV이며 다중 비교 보정이 없다. REAL은 2D 주석·치수에 의존한 재구성 참조다. 독립 물리 실측은 0쌍/BLOCKED_DATA이므로 실제 물리 T/R 개선의 확증이 아니다. 이번 두 고정 구조의 결과로 8코너 관계의 필요성이나 6D 학습의 불가능을 입증하지 않는다. 추가 seed·구조·손실·온도 검색, 비용 재계산, backbone 재추출, bank 재생성, 원고/PDF/bib 변경·빌드는 0이다.','']
    return '\n'.join(lines)


def report():
    began=time.monotonic(); protocol=read(DOC/'PROTOCOL.json'); run=read(DOC/'RUN_RECEIPTS.json');fits=run['methods']
    assert set(fits)==set(NEW)
    oldfit=read(QUICK_DOC/'TRAIN_RECEIPTS.json')['methods']['SOFT6D']
    for method in NEW:
        fit=fits[method]
        assert fit['status']=='DONE' and fit['complete'] and fit['updates']==6000 and fit['exposures']==96000
        assert fit['params']==26169 and fit['head_params']==4680 and fit['phi_params']==1230
        assert fit['order_sha256']==oldfit['order_sha256'] and fit['initial_state_sha256']==oldfit['initial_state_sha256']
        assert Path(fit['checkpoint_path']).resolve().is_relative_to(OUTPUT.resolve())
        assert sha(Path(fit['checkpoint_path']))==fit['checkpoint_sha256']
    assert fits['LOCAL_CAP']['phi_initial_state_sha256']==fits['JOINT8']['phi_initial_state_sha256']
    splits={name:analyze_split(name,fits) for name in ('SYNTH_HELDOUT','REAL_DEV')}
    probes={}
    setup=read(LOSS_SETUP)
    for method in NEW:
        probe=read(DOC/f'TRAIN_PROBE_{method}_seed1.json')
        assert probe['status']=='DONE' and probe['complete'] and probe['execution']['new_F_calls']==0
        assert probe['checkpoint_sha256']==fits[method]['checkpoint_sha256']
        assert probe['code_sha256']==sha(Path(__file__).with_name('evaluation.py')) and probe['protocol_sha256']==sha(DOC/'PROTOCOL.json')
        assert probe['setup_sha256']==sha(LOSS_SETUP) and probe['GT_inference_access'] is False
        assert [r['id'] for r in probe['rows']]==setup['calibration']['ids'][:256]
        assert [r['source_cache_row'] for r in probe['rows']]==setup['calibration']['rows'][:256]
        probes[method]=probe
    result=dict(schema='quick_joint_scorer_saved_rows_v1',status='DONE',seeds=[1],splits=splits,
        decisions={method:decision(method,splits) for method in NEW},run_receipts=run,train_probes=probes,
        source_bindings=[bind(DOC/'PROTOCOL.json'),bind(DOC/'RUN_RECEIPTS.json'),bind(LOSS_SETUP),
                         *[bind(DOC/f'TRAIN_PROBE_{method}_seed1.json') for method in NEW]],
        code_binding=bind(Path(__file__)),report_execution=dict(NN_forwards=0,final_F=0,optimizer_updates=0,
            bootstrap_seed=20260917,resamples=10000,seconds=time.monotonic()-began))
    save(DOC/'RESULTS.json',result);save(DOC/'RESULT_KO.md',render(result),text=True)
    interpreter='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
    readme='\n'.join(['# LOCAL_CAP / JOINT8 고정 비교','',
        f"[확인] LOCAL_CAP **{result['decisions']['LOCAL_CAP']['label']}**, JOINT8 **{result['decisions']['JOINT8']['label']}**. [RESULT_KO.md](RESULT_KO.md), [RESULTS.json](RESULTS.json), [PROTOCOL.json](PROTOCOL.json), [RUN_RECEIPTS.json](RUN_RECEIPTS.json)이 실제 원행·짝 비교·설정·실행량 근거다.",'',
        '```bash','cd /home/minjae/Documents/github/pallet-pose',
        f'export PALLET_BASELINE_ROOT={Q.OLD_DOC.parents[2]}',
        f'{interpreter} -B -m scripts.research.pallet_quick_joint_scorer_20261006_v1.run --help',
        f'{interpreter} -B -m scripts.research.pallet_quick_joint_scorer_20261006_v1.reporting', '```','',
        f"[확인] source `{protocol['source_root']}`, bank `{protocol['candidate_bank_cache']}`, cost `{protocol['pose_cost_cache']}`와 기존 LOSS_SETUP 저장 soft target을 읽기 전용으로 재사용했다. private 가중치는 `{OUTPUT}/fits/{{LOCAL_CAP,JOINT8}}_seed1/last.pt`에 있으며 SHA·bytes는 RUN_RECEIPTS에 기록한다. 불변 a22 작업트리와 private 데이터/feature/cache가 필요하므로 공개 checkout만의 완전 재현을 주장하지 않는다.",'',
        '[확인] 완료 영수증만 검증 후 재사용한다. 중단된 학습·평가를 quiet retry하지 않는다. reporting은 완료된 저장 원행만 읽고 새 경로에 결과를 쓴다. NN/F/optimizer 추가0, 원고/PDF/bib 작업0.',''])
    save(DOC/'README_KO.md',readme,text=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args(argv)
    result=report();print(json.dumps({m:d['label'] for m,d in result['decisions'].items()},ensure_ascii=False))


if __name__=='__main__':main()
