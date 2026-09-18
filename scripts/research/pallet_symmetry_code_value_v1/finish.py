"""Apply preregistered decision; audit old roots, budget, checkpoints and outputs."""
import subprocess
import numpy as np
import torch
import cv_env as E
from fit import bindings,tests

def benefit(c):return c.get('CI95') is not None and c['CI95'][1]<0 and c['improved_seeds']>=2
def advantage(m):
    if not m.get('applicable'):return False
    c=m['contrast'];return c['CI95'] is not None and c['CI95'][0]>0 and sum(x>0 for x in c['per_seed_delta'])>=2 and m['movement']['mean_coordinate_change_px']>=.01
def decision(a,b,c):
    headline=benefit(a['SYNTH']['contrast']) or sum(benefit(x) for x in [b['SYNTH']['groups']['C1'],b['SYNTH']['groups']['C2'],b['SQUARE']['groups']['C4']])>=2
    safety=a['DEV']['safety_pass'] and b['DEV']['safety_pass'];square=b['SQUARE']['contrast']['CI95'][0]<=0
    perturb=any(all(advantage(c[arm]['SYNTH']['modes'][m]) for m in ['NEUTRAL','WRONG']) for arm in c)
    anybenefit=any(benefit(x) for tr in [a,b] for p in tr.values() for x in [p['contrast'],*p['groups'].values()])
    anyvalue=any(advantage(p['modes'][m]) for arm in c.values() for p in arm.values() for m in ['NEUTRAL','WRONG'])
    result='KEEP_CODE_DEV_EVIDENCE' if headline and safety and square and perturb else 'DROP_CODE' if not anybenefit and not anyvalue else 'UNRESOLVED'
    return dict(result=result,headline_benefit=headline,A_and_B_real_safety_pass=safety,B_square_not_clearly_worse=square,
      same_multigroup_population_neutral_and_wrong_advantage=perturb,any_matched_capacity_benefit=anybenefit,any_correct_code_value=anyvalue,
      thresholds_source=E.bound(E.DOC/'PROTOCOL_LOCK.json'),additional_sweep=False)
def fmt(x):return 'N/A' if x is None else f'{x:.7f}' if isinstance(x,(float,np.floating)) else str(x)
def ci(c):return 'N/A' if c.get('CI95') is None else f"[{c['CI95'][0]:.7f}, {c['CI95'][1]:.7f}]"
def main():
    tests()
    a=E.read(E.DOC/'A_C1C2_RESULTS.json')['populations'];b=E.read(E.DOC/'B_C1C2C4_RESULTS.json')['populations'];c=E.read(E.DOC/'CODE_PERTURBATION_RESULTS.json')['models']
    for name in ['REGRESSION_TESTS','SMOKE_COMPLETE','A_TRAINING_COMPLETE','B_TRAINING_COMPLETE','POSE_SECONDARY_RESULTS','RUNTIME_AND_PARAMS']:
        r=E.read(E.DOC/(name+'.json'));assert r.get('complete',r.get('PASS'))
    lock=E.protocol();bound=bindings();fits=[]
    for track in ['A','B']:
        for seed in [1,2,3]:
            pairs=[]
            for arm in E.ARMS[track]:
                record=E.read(E.DOC/f'fits/{arm}_seed{seed}.json');E.verify([record['checkpoint']]);ck=torch.load(E.ROOT/record['checkpoint']['path'],map_location='cpu',weights_only=False)
                assert ck['complete'] and ck['step']==6000 and ck['bindings']==bound and not ck['smoke']
                assert ck['optimizer']['state'] and all(int(v['step'])==6000 for v in ck['optimizer']['state'].values())
                assert all(torch.isfinite(v).all() for v in ck['state'].values());assert record['params']==20307
                assert ck['initial_state_sha256']==lock['initial'][str(seed)]['full_state_sha256']
                assert ck['order_sha256']==lock['orders'][f'{track}_seed{seed}']['sequence_sha256'];pairs.append(record);fits.append(record)
            assert pairs[0]['initial_state_sha256']==pairs[1]['initial_state_sha256'] and pairs[0]['order_sha256']==pairs[1]['order_sha256']
            first=[E.read(E.RAW/f'runs/{arm}_seed{seed}/FIRST_STEP.json') for arm in E.ARMS[track]]
            assert first[0]['context_first5']==first[1]['context_first5']
            assert np.array_equal(np.asarray(first[0]['context_last3'],np.float32),np.full((16,3),1/3,np.float32))
            assert np.isin(first[1]['context_last3'],[0.,1.]).all() and np.all(np.asarray(first[1]['context_last3']).sum(-1)==1)
    assert sum(x['steps'] for x in fits)==72000
    smoke=E.read(E.DOC/'SMOKE_COMPLETE.json');assert smoke['updates']==200 and len(smoke['fits'])==4
    for record in smoke['fits']:
        E.verify([record['checkpoint']]);ck=torch.load(E.ROOT/record['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['smoke'] and ck['step']==50 and all(int(v['step'])==50 for v in ck['optimizer']['state'].values())
    E.write(E.DOC/'TRAINING_COMPLETE.json',dict(complete=True,main_fits=12,main_updates=72000,smoke_updates=200,actual_optimizer_state_steps_verified=True,smoke_discarded=True,tests_optimizer_updates=0,C_optimizer_updates=0,R0_updates=0,fits=fits))
    source=E.read(E.DOC/'SOURCE_BINDING.json');E.verify(source['files']);E.verify(source['original_source_files'])
    for st in source['paper_cache_stat'].values():
        p=E.ROOT/st['path'];assert p.stat().st_size==st['bytes'] and p.stat().st_mtime_ns==st['mtime_ns']
    E.verify(E.read(E.DOC/'EVALUATION_CODE_LOCK.json')['files']);checked=0
    for p in (E.RAW/'predictions').rglob('*.json'):
        r=E.read(p);assert r['complete'] and not r['GT_input'] and not r['GT_arrays_read'] and r['detector_contract_exact'] and r['T']==1.
        E.verify([r['checkpoint'],r['logits']]);checked+=len(r['records'])
    assert checked==83355,checked
    assert E.sha(E.D.R0)==E.D.R0_SHA
    damage={tr:{pop:dict(per_seed=r['damage'],gross20_delta=r['aware']['gross20']-r['blind']['gross20'],net_good5_bad10=sum(d['good5_to_bad10']-d['reverse_good5_to_bad10'] for d in r['damage']),safety_pass=r['safety_pass']) for pop,r in pops.items()} for tr,pops in [('A',a),('B',b)]}
    E.write(E.DOC/'DAMAGE_DECOMPOSITION.json',dict(complete=True,tracks=damage,canonical_GT_identity_aligned=True,branch_change_separate=True,safety_rule=lock['safety']))
    verdict=decision(a,b,c);E.write(E.DOC/'DECISION.json',verdict)
    lines=['# Explicit C1/C2/C4 code value isolation v1 — 완료 보고','',f"판정: **{verdict['result']}**",'',
      '두 모델은 dimensions, symmetry target, architecture, parameters, training budget가 같고 explicit group-code information만 다르다.','',
      '동일 8-D 모델(20,307 parameters), seed별 모든 초기 tensor·batch order 일치, T=1/lambda=1/cap=0.01 고정. Blind는 항상 [1/3,1/3,1/3], aware는 approved one-hot. R0는 frozen이며 모든 과거 DCP 파일을 보존했다.',
      '', '## A exact-capacity paper','', '| split | arm | E_sym | median px | P90 px | PCK10 |','|---|---|---:|---:|---:|---:|']
    caution=['## 판정의 적용 범위','',
      '사전 등록 규칙의 형식 판정과 모든 환경에서 코드가 필수라는 주장은 다르다. A synthetic-only matched-capacity 비교는 전체 synthetic/REAL_DEV 모두 CI가 0을 포함했다. B mixed 비교에서는 C1/C2/C4 모두 개선됐지만 domain/group confounding이 있는 작은 개발 효과이다.',
      '중요한 반례: A1과 B1 모두 REAL_DEV에서는 NEUTRAL이 CORRECT보다 낮은 E_sym을 보였다(각 CI가 0 아래). A1의 WRONG도 CORRECT보다 유의하게 좋았고, B1 WRONG의 점추정도 더 좋지만 CI는 0을 포함한다. 이를 숨기거나 mapping을 다시 고르지 않았다. 올바른 코드가 synthetic/square에서 유용하다는 결과를 실사 C2 배포의 보편적 필요성으로 확대하지 않는다.',
      '따라서 KEEP_CODE_DEV_EVIDENCE가 나오더라도 그것은 명시된 mixed/개발 조건에서의 유지 근거이다. Synthetic-only paper A의 추가 이득이나 독립 실사 일반화 증명이 아니며, 이번 결과로 DEV에 맞춰 코드 모드를 새로 선택하지 않았다.','']
    lines[8:8]=caution
    for pop,r in a.items():
        for arm in ['blind','aware']:
            s=r[arm];lines.append(f"| {pop} | {arm} | {s['E_sym']:.7f} | {s['matched_pooled_corner8_median_px']:.4f} | {s['matched_pooled_corner8_P90_px']:.4f} | {s['PCK10']:.5f} |")
    lines+=['','모든 표는 3-seed 평균. median/P90은 매칭 관측 corner, E_sym/PCK는 missing-detection penalty 포함 전체 분모. E_sym은 이미지 대각선으로 정규화되어 px가 아니다.','',
      '| split | aware − blind | CI95 | seed별 delta | 개선 seed |','|---|---:|---|---|---:|']
    for pop,r in a.items():
        x=r['contrast'];lines.append(f"| {pop} | {x['delta']:.7f} | {ci(x)} | {x['per_seed_delta']} | {x['improved_seeds']}/3 |")
    lines+=['','## B mixed routing — DIAGNOSTIC ONLY','','| eval population | group coverage | blind | aware | delta | CI95 |','|---|---|---:|---:|---:|---|']
    for pop,r in b.items():lines.append(f"| {pop} | {', '.join(r['groups'])} | {r['blind']['E_sym']:.7f} | {r['aware']['E_sym']:.7f} | {r['contrast']['delta']:.7f} | {ci(r['contrast'])} |")
    lines+=['','## Group contrasts','','| track/pop | group | n | delta | CI95 | improved seeds |','|---|---|---:|---:|---|---:|']
    for tr,pops in [('A',a),('B',b)]:
        for pop,r in pops.items():
            for g,x in r['groups'].items():lines.append(f"| {tr}/{pop} | {g} | {x['n']} | {x['delta']:.7f} | {ci(x)} | {x['improved_seeds']}/3 |")
    lines+=['','## C inference code perturbation','','E_sym. Correct checkpoint 및 dimensions/evaluator를 고정하고 network code만 변경했다. 새 optimizer update는 0.','',
      '| model/pop | correct | neutral | zero | wrong | shuffled |','|---|---:|---:|---:|---:|---:|']
    for arm,pops in c.items():
        for pop,r in pops.items():
            vals=[fmt(r['modes'][m]['summary']['E_sym']) if r['modes'][m]['applicable'] else 'N/A(single group)' for m in ['NEUTRAL','ZERO','WRONG','SHUFFLED']]
            lines.append(f"| {arm}/{pop} | {r['correct']['E_sym']:.7f} | "+' | '.join(vals)+' |')
    lines+=['','| model/pop/code | delta vs correct | CI95 | mean move px | max move px | JS | top1 change |','|---|---:|---|---:|---:|---:|---:|']
    for arm,pops in c.items():
        for pop,r in pops.items():
            for mode,x in r['modes'].items():
                if not x['applicable']:continue
                m=x['movement'];lines.append(f"| {arm}/{pop}/{mode} | {x['contrast']['delta']:.7f} | {ci(x['contrast'])} | {m['mean_coordinate_change_px']:.4f} | {m['max_coordinate_change_px']:.4f} | {m['JS']:.7f} | {m['top1_change_rate']:.5f} |")
    lines+=['','## Damage / runtime / 6D','','| track/pop | net good<5 → bad>10 | gross20 delta | safety guard |','|---|---:|---:|---|']
    for tr,pops in damage.items():
        for pop,r in pops.items():lines.append(f"| {tr}/{pop} | {r['net_good5_bad10']} | {r['gross20_delta']:.7f} | {r['safety_pass']} |")
    runtime=E.read(E.DOC/'RUNTIME_AND_PARAMS.json')
    lines+=['','| arm | params | P-only median ms | RGB-to-PnP median ms |','|---|---:|---:|---:|']
    for arm,r in runtime['summary'].items():lines.append(f"| {arm} | 20307 | {r['P_only_ms']['median']:.3f} | {r['full_ms']['median']:.3f} |")
    lines+=['','동일 PnP adapter의 6D secondary는 `POSE_SECONDARY_RESULTS.json`에 보존했다. 실사/square pose reference는 재구성 geometry이며 독립 물리 GT가 아니다. 6D는 primary 판정 변경에 사용하지 않았다.',
      '', '## Decision / interpretation','',f"`{verdict['result']}`. 사전 고정 판정 게이트: `{verdict}`.",'',
      ('Code 유지 개발 근거를 확보했다. Deployment에서 dimensions와 approved group code를 외부에서 알아야 한다.' if verdict['result']=='KEEP_CODE_DEV_EVIDENCE' else
       'Code 없는 단순한 버전을 main method로 권고한다. Dimensions와 symmetry-aware supervision/evaluation은 그대로 유지하며 code-aware는 ablation으로 보존한다. 기존 배포 모델을 자동 교체하지 않았다.'),
      '', '## 제한 및 재현','',
      '- A에는 C4가 없다. 실제 source TRAIN C1=19,268/C2=36,712. Heldout C1=511/C2=1,474.',
      '- 여기서 검증한 code-free 대조군은 동일한 8-D 모델의 마지막 3채널을 neutral로 고정한 버전이다. 기존 5-D N3로 교체했을 때의 수치를 이 실험에서 새로 주장하지 않는다.',
      '- REAL_DEV319는 모두 C2: 그룹 간 routing이 아니라 constant-token 영향. Wood는 physical symmetry UNREVIEWED와 기존 C2 benchmark convention을 구분했다.',
      '- B는 C1/C2 synthetic, C4 real-supervised의 domain/group confounding이 있다. Pooled 성공 지표나 symmetry causal concept 학습 주장 금지.',
      '- Bootstrap 10,000, seed20260918. Synthetic frame primary/scenario secondary; real session-cluster 및 LOSO. Seed는 3개의 paired mean이며 seed population의 불확실성을 별도 resample하지 않았다.',
      '- Session subgroup에 cluster가 하나뿐이면 CI N/A. Empty C1/C2/C4 group에 수치를 만들지 않았다. Multiplicity-adjusted confirmatory evidence가 아니며 DEV는 재사용되었다.',
      '- Shuffled donor는 score 보기 전 고정. DEV/SQUARE는 단일 group이므로 N/A. 같은 code/frame donor 수는 CODE_PERTURBATION_RESULTS.json에 기록.',
      '- Neutral/zero는 aware 학습에서 보지 않은 code 값이고 A의 cyclic C2→C4도 미학습 group 입력이다. 따라서 교란 민감도만으로 대칭 개념이나 인과적 routing 학습을 주장하지 않는다.',
      '- 온도 calibration, seed/step/lr sweep, 추가 architecture/loss, 신규 데이터, FINAL/sealed 접근, R0 학습은 없다.',
      '- 사전 검사 첫 시도는 결측 GT의 NaN==NaN 비교 때문에 실패했고 equal_nan=True, rtol=atol=0 검사로 바로잡았다. 데이터/모델 수정이나 optimizer update는 없었다.',
      '- `python -B scripts/research/pallet_symmetry_code_value_v1/runner.py`가 고정 순서 실행/완료 checkpoint 재사용을 제공한다. Python 환경: pallet-yolo26. 소스/모델/metadata SHA는 SOURCE_BINDING과 각 예측 파일에 보존.',
      '- 코드·통계·보고서는 Git으로 공유한다. 기존 저장소 정책상 `data/**`, checkpoint, logits는 Git 제외 로컬 산출물이다. RAW_ARTIFACT_INDEX.json에 경로·크기·SHA를 남겼으며 다른 PC 재현에는 명시된 기존 데이터/cache가 필요하다.',
      '',f"주학습 72,000 + 폐기 smoke 200 update. 회귀검사 {E.read(E.DOC/'REGRESSION_TESTS.json')['tests']}개 통과. 원본 경로는 read-only 감사 통과.",'']
    (E.DOC/'FINAL_REPORT_KO.md').write_text('\n'.join(lines))
    E.write(E.DOC/'RAW_ARTIFACT_INDEX.json',dict(complete=True,git_tracked=False,root=str(E.RAW.relative_to(E.ROOT)),files=[E.bound(p) for p in sorted(E.RAW.rglob('*')) if p.is_file() and 'logs' not in p.parts and not p.name.endswith('.pending')]))
    audit=dict(complete=True,time=E.now(),decision=verdict,main_updates=72000,smoke_updates=200,C_optimizer_updates=0,R0_frozen=True,R0_sha256=E.sha(E.D.R0),
      same_8D_architecture=True,same_params=20307,every_initial_tensor_exact=True,same_target=True,same_order=True,only_context_last3_differs=True,
      tests=E.read(E.DOC/'REGRESSION_TESTS.json')['tests'],original_DCP_files_unchanged=True,old_source_bindings_unchanged=True,paper_cache_stat_unchanged=True,
      predictions_checked=checked,GT_free_inference=True,T1_all=True,FINAL_access=False,new_data=False,extra_sweep=False,main_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
      start_SHA=lock['start_SHA'],runtime_parity_exact=runtime['parity_exact'],raw_root=str(E.RAW.relative_to(E.ROOT)),scripts=[E.bound(p) for p in sorted(E.HERE.glob('*.py'))])
    assert audit['main_branch']=='main';E.write(E.DOC/'FINAL_AUDIT.json',audit);print('FINAL_DECISION',verdict,flush=True)
if __name__=='__main__':main()
