"""Render evidence-only Korean A report and supplementary LaTeX snippet.

Reads final locked receipts; no model, reference, inference or optimizer access.
"""
from pathlib import Path
import json
from .a_common import DOC


def number(value, digits=6):
    return 'NA' if value is None else f'{value:.{digits}f}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |', *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def main():
    s=json.loads((DOC/'results/A_summary.json').read_text())
    m=json.loads((DOC/'A_manifest.json').read_text())
    assert s['formal_updates']==36000 and len(s['fits'])==6
    real_controls=s['comparisons']['REAL_DEV']
    meanings={'WORSENED':'악화','SUPPORTED':'고정 손상/보존 조건에서 탐색적 이익 지지','UNRESOLVED':'이익 미확립'}
    outcome='[확인] 실제 개발 자료의 GEO--J minus N3/PoseFix 전체 방법 비교 판정은 각각 '+meanings[real_controls['FIT_GEO_J_minus_N3']['exploratory_verdict']]+'/'+meanings[real_controls['FIT_GEO_J_minus_PoseFix']['exploratory_verdict']]+'야. GEO--J minus RAW/PERM의 평균 정규화 오차 차이 구간이 음수여도 고정 손상/보존 규칙을 함께 적용해 각각 '+meanings[real_controls['FIT_GEO_J_minus_RAW']['exploratory_verdict']]+'/'+meanings[real_controls['FIT_GEO_J_minus_FIT_PERM_J']['exploratory_verdict']]+'라는 판정으로 기록했어. synthetic 개발 진단의 GEO--J minus PERM 판정은 '+meanings[s['comparisons']['SYNTH_HELDOUT']['FIT_GEO_J_minus_FIT_PERM_J']['exploratory_verdict']]+'야.'
    parts=['# A 실행 결과',
        '[확인] source oracle, 동결 N3 세 seed의 GEO/PERM×I/J, GEO/PERM 세 seed의 6,000회 정식 학습과 마지막 checkpoint 평가를 완료했어. 큰 특징 cache와 가중치는 저장소 밖에 있고 기존 N3/PoseFix 결과를 동일 입력 해시로 재사용했어. 실제 DEV는 독립 확인 자료가 아니야.',
        outcome,
        '[확인] 아래 중앙값/90백분위는 매칭되고 관측된 코너를 모은 조건부 통계야. 실제 DEV는 검출 319/319, 8유효 예측 코너 319/319, GT instance 매칭 311/319를 구분해. 전체 penalty 포함 평균 정규화 거리와 PCK는 참조 코너 전체 분모를 사용하고, F 자세 coverage는 전체 frame을 분모로 사용해. 실제 참조 자세는 같은 2D 레이블·치수에서 재구성했어.',
        '[확인] 실제 DEV의 유효 참조 코너는 2,499개이고 매칭된 관측 거리의 분모는 2,445개야. synthetic 개발 진단의 유효 참조/관측 코너는 각각15,658개야. 전체 참조와 조건부 관측 분모를 서로 바꾸지 않아.',
        '[확인] PCK(Percentage of Correct Keypoints, 허용 거리 안의 코너 비율), ADD(Average Distance of Model Points, 모델 점 평균 거리)의 대칭 적용 진단 ADDsym, PnP(Perspective-n-Point, 2D/3D 대응 자세 계산)를 사용해. ADDsym은 승인된 proper rotation group에서 정준 8코너 대응 평균 거리 m이며 표면 ADD-S가 아니야. I는 코너별 hard argmax, J는 한 action의 hard argmax야.',
        '## 동일 분모의 세 seed 원결과']
    for split,label in [('REAL_DEV','실제 개발 319장/13세션'),('SYNTH_HELDOUT','기존에 소진된 synthetic 개발 진단 1,985장')]:
        rows=[]
        for name,v in s['summaries'][split].items():
            c,p=v['corner'],v['pose']
            rows.append([name,number(c['matched_pooled_corner8_median_px'],4),number(c['matched_pooled_corner8_P90_px'],4),number(c['PCK']['10'],5),number(c['gross20'],5),number(p['translation_cm']['median'],4),number(p['rotation_deg']['median'],4),f"{p['available']}/{p['total_frames']}",v['NoOp'] if name.startswith(('FIT','FROZEN')) else 'N/A'])
        parts.extend([f'### {label}',table(['방법/seed','코너 med px','P90 px','PCK10','gross20','T med cm','R med deg','F coverage','NoOp'],rows)])
    parts.append('## practical 기준선 및 결합/decoder 대조')
    for split,label in [('REAL_DEV','실제 개발'),('SYNTH_HELDOUT','synthetic 개발 진단')]:
        comparisons=s['comparisons'][split];rows=[]
        for name,v in comparisons.items():
            e=v['E_sym'];rows.append([name,number(e['delta'],8),'['+','.join(number(x,8) for x in e['CI95'])+']','/'.join(number(x,8) for x in e['per_seed_delta']),v['exploratory_verdict'],v['contract_safety_pass']])
        parts.extend([f'### {label}: seed 평균 paired E_sym 차이',table(['새 방법 minus 기준','평균 차이','95% CI','seed1/2/3 차이','판정','손상/coverage 보존'],rows)])
        damage=[];pose=[]
        focused=[k for k in comparisons if k.startswith('FIT_') or k in ['FROZEN_GEO_J_minus_FROZEN_GEO_I','FROZEN_PERM_J_minus_FROZEN_PERM_I']]
        for name in focused:
            v=comparisons[name]
            for r in v['per_seed']:
                d,p=r['corner_damage'],r['pose_paired'];damage.append([name,r['seed'],d['good5_to_bad10'],d['bad20_to_good10'],f"{d['improved_frames']}/{d['harmed_frames']}/{d['unchanged_frames']}",f"{p['common_success']}/{p['new_only_success']}/{p['base_only_success']}/{p['both_failed']}"])
            for key,value in v['pose_seed_mean_paired'].items():
                pose.append([name,key,value['common_success_all3seed_frames'],number(value['delta'],8),'['+','.join(number(x,8) for x in value['CI95'])+']','/'.join(number(x,8) for x in value['per_seed_delta'])])
        parts.extend([table(['대조','seed','good<5 → bad>10 코너','bad>20 → good<10 코너','개선/손상/동일 frame','양성공/새것만/기준만/양실패'],damage),table(['대조','자세 지표','3seed 공동성공 frame','평균 paired 차이','95% CI','seed1/2/3 차이'],pose)])
    parts.extend(['## 고정 bank의 실제 F oracle와 선택 gap',
        '[확인] source 1,031장은 학습 진입 판단용 개발 oracle이며 이전 heldout 1,985장/실제 319장 진단과 분리해. NoOp 반복 F의 ADDsym 차이는 0m였고 첫 metric 전 고정한 numerical tie 허용오차 1e-7m보다 큰 여지만 비영으로 세었어. 진입은 GEO 여지만으로 판단했어. oracle 선택에는 참조를 쓰지만 후보 생성·특징 점수·배포 J에는 쓰지 않아.'])
    oracle_rows=[]
    for split,file in [('SOURCE_SELECTION','A_SOURCE_ORACLE.json'),('SYNTH_HELDOUT','A_SYNTH_HELDOUT_ORACLE.json'),('REAL_DEV','A_REAL_DEV_ORACLE.json')]:
        x=json.loads((DOC/'results'/file).read_text())
        for arm in ['GEO','PERM']:
            rr=[r for r in x['rows'] if r['arm']==arm];finite=[r for r in rr if r['headroom'] is not None]
            hs=sorted(r['headroom'] for r in finite);n=len(hs);median=(hs[n//2] if n%2 else (hs[n//2-1]+hs[n//2])/2) if n else None
            oracle_rows.append([split,arm,len(rr),sum(r['raw']['available'] for r in rr),sum(r['oracle']['available'] for r in rr),sum(r['headroom']>1e-7 for r in finite),number(median,8),sum(r['oracle']['translation_cm']<=r['raw']['translation_cm']+1e-9 and r['oracle']['rotation_deg']<=r['raw']['rotation_deg']+1e-9 for r in finite),sum(r['actions'] for r in rr),sum(r['failures'] for r in rr)])
    parts.append(table(['자료','bank','전체','RAW 성공','oracle 성공','비영 여지','여지 med m','같은 후보 T/R 동시 비악화','후보 F 호출','후보 F 실패'],oracle_rows))
    oracle_corners=[]
    for split in ['SOURCE','SYNTH_HELDOUT','REAL_DEV']:
        x=json.loads((DOC/f'results/A_{split}_ORACLE_CORNERS.json').read_text())
        for arm,v in x['summary'].items():
            a,b,d=v['raw'],v['oracle'],v['damage']
            oracle_corners.append([split,arm,number(a['matched_pooled_corner8_median_px'],4)+' / '+number(b['matched_pooled_corner8_median_px'],4),number(a['matched_pooled_corner8_P90_px'],4)+' / '+number(b['matched_pooled_corner8_P90_px'],4),number(a['PCK']['10'],5)+' / '+number(b['PCK']['10'],5),d['good5_to_bad10'],d['bad20_to_good10']])
    parts.append(table(['자료','bank','바로 그 선택 RAW/oracle med px','RAW/oracle P90 px','RAW/oracle PCK10','good5 → bad10','bad20 → good10'],oracle_corners))
    for split in ['REAL_DEV','SYNTH_HELDOUT']:
        g=json.loads((DOC/f'results/A_{split}_ORACLE_GAPS.json').read_text());rows=[]
        for name,v in g['summary'].items():
            rows.append([name,v['common_success'],number(v['median_gap_m'],8),number(v['P90_gap_m'],8),v['chosen_F_failure'],v['oracle_F_failure'],v['negative_beyond_numeric_tolerance']])
        parts.extend([f'### {split}: 선택 minus 같은 bank oracle ADDsym',table(['방법/seed','공동성공','gap med m','gap P90 m','선택 F 실패','oracle F 실패','음수 > 허용오차'],rows)])
    parts.extend(['[확인] 선택된 단일 oracle의 바로 그 2D 결과/손상과 이동·회전은 `A_SOURCE_ORACLE_CORNERS.json`, `A_SYNTH_HELDOUT_ORACLE_CORNERS.json`, `A_REAL_DEV_ORACLE_CORNERS.json`에 남아. index/bank 재사용 보완의 추가 F 호출은 0회야. I는 bank 밖의 코너 조합이라 oracle보다 좋을 수 있지만 J의 음수 gap은 허용오차 밖에서 0건이어야 해.',
        '## 실행량과 검증','```json\n'+json.dumps(m['execution_accounting'],ensure_ascii=False,indent=2)+'\n```',
        '[확인] 6 fit×6,000 update=36,000 update, batch16의 576,000 노출이야. smoke는 각 arm 2회씩 총4회이며 가중치를 폐기하고 같은 seed의 원래 무작위 초기화로 정식 fit을 시작했어. seed별 두 arm의 초기 state SHA/순서 SHA/20,259 parameter가 같음을 `A_manifest.json`에서 검증해. 실제 TRAIN 제외 target 노출 수와 최종 checkpoint SHA는 각 `A_fits/*.json`에 있어. GPU는 RTX3080, FP32, matmul TF32 off/cudnn TF32 on의 원래 수치 계약이야.',
        '[확인] 첫 optimizer 이전의 NPZ 반복 materialization/인위적 NaN backward/프로세스 import 정합성 수정과 재시작에는 정식·smoke update가 각각 0회였어. 유효 oracle/bank를 재사용했어. NoOp·cap·중심 회전·양의 깊이·원본/네트워크 affine·PERM 주변 multiset·기존 radial API·결측 backward 검증은 `A_NUMERICAL_PARITY.json`, `A_INDEPENDENT_PRELIMINARY_QA.json`, 독립 검토 영수증에 연결돼. trained dynamic radial adapter의 최대 logits 차이 4.482269287109375e-5는 FP32 coordinate 재구성 범위이며 기존 radial API exact parity와 구분해.',
        '[확인] 마지막 fresh 분석은 `A_ANALYSIS_REAGGREGATION.json`의 wall126.04초로 측정됐어. 완료fit6개/예측파일18개/oracle3개를 재사용해 추가 optimizer·CNN 점수 추론·최종 F 채점은 모두0회야. 기존 평가기 구조 때문에 실제319개 bank를 3번(957개 bank) 다시 구성했고 내부의 초기 prediction-only selector/PnP는 수행됐어. 이 초기 PnP의 개별 호출수는 미계측이므로 0회라고 쓰지 않아. 분석 보완을 새 학습이나 최종 F 성능 실험으로 세지 않아.',
        '[확인] `results/A_SAMPLING_MASK_RECEIPT.json`은 synthetic1,985+real319의 2,304개 frame에서 원시좌표/affine/box/input bounds/기존32점 stencil만으로 sampling mask·coverage·공통support를 CPU FP32로 재구성했어. 원래 실행한 GPU tensor는 보존하지 않았으므로 원tensor 저장본이라고 쓰지 않아. GEO/PERM mask 주변 multiset과9개checkpoint stencil은같아. source저장bank는초기PnP0회로재사용했고realbank1회재생성은실측solvePnP1,276회/RefineLM1,276회, 전체보완wall20.449초야. 추가CNN/최종F/update는0회이며bit-packed mask와realbank는외부cache에해시와함께저장했어. 보조mask/subgroup receipt는parentmanifest가따로바인딩해.',
        '[확인] 실제는 session, synthetic 주 분석은 frame으로 10,000회/seed20260917의 같은 bootstrap 추출을 공유했어. synthetic scenario secondary는 미실행이야. 여러 대조의 multiplicity 보정 없는 탐색적 구간이며 독립 확증이 아니야.',
        '## 연구 판단과 남은 의존성',
        '[추정] 비영 oracle 여지는 이 고정 bank 안에 더 나은 최종 F 출력이 존재한다는 진단이야. practical N3/PoseFix와의 방법 전체 비교, GEO/PERM 유한 결합 대조, 같은 bank I/J decoder 대조의 CI·손상·실패를 각각 읽어야 해. 좋은 seed·중앙값·인위적 대조 승리만으로 센서 기반 6D 개선을 선언하지 않아. 동결 scorer의 실패는 특징 정보 부재의 증명이 아니며 이번 6 fit 결과는 고정 후보/특징/참조/예산 범위만 제한해.',
        '[확인] 물리 가림의 독립 상태와 clean/실물 가림 동일 상대 자세 pair·독립 센서 참조는 BLOCKED_DATA야. human grade/가시성 상태/기존 거리 tag의 보조 재집계는 `A_DESCRIPTIVE_SUBGROUPS_KO.md`와 연결하지만 물리 가림 정도나 독립 거리 계측으로 승격하지 않아. 새로운 모델·cap·seed·epoch 탐색은 추가하지 않았어.',
        '[확인] 배포 비용은 root의 동일 26frame/20warmup/5repeat 전체 경계 A runtime 영수증에서 별도로 통합해. 위 전체 후보 oracle F 탐색은 오프라인 비용이며 배포 비용과 합치지 않아.',
        '## 결과와 재개',
        '`A_protocol.json`, `results/A_ID_MANIFEST.json`, `results/A_summary.json`, `results/A_*_COMPARISONS.json`, `results/A_*_METRIC_ROWS.tsv`, `A_manifest.json`, `A_RESULT_SCHEMA_KO.md`를 함께 읽어. 원본 private 입력이 없으면 모델/특징 재실행 의존성이 막히며 작은 집계 파일만으로 새 성능을 생성하지 않아.',
        '주 재개 명령은 `python -m scripts.research.pallet_joint_action_handoff_20261006_v1.run --stage resume --source-root ORIGINAL_READ_ONLY_CHECKOUT --cache-dir EXTERNAL_CACHE`야. 실제 경로는 상위 README에 있어. 이 진입점은 추가663개 입력의 전량 SHA 및 source17배열 receipt를 먼저 검증한 뒤 완료된 6 fit와 원결과를 재사용해.'])
    (DOC/'A_RESULT_KO.md').write_text('\n\n'.join(parts)+'\n')
    real=s['summaries']['REAL_DEV'];snippet=[]
    snippet.append('source selection 1,031장의 GEO bank에서 실제 전체 자세 함수 $F$의 대칭 8코너 거리(ADDsym) oracle 여지가 수치오차 $10^{-7}$m보다 큰 프레임은 954장이었고, 여지 중앙값은 0.011915m였다. 원시와 oracle 모두 1,030/1,031장에서 자세를 산출했다. 이 상한은 참조로 고른 개발 진단이며 배포 결과가 아니다. 이에 따라 GEO/PERM 각각 세 seed를 원래 무작위 초기화부터 6,000 update씩, 총 36,000 update와 576,000 노출로 학습했다. smoke 4 update는 별도로 폐기했다.')
    snippet.append(r'\begin{table}[t]\centering\small\caption{실제 DEV의 추가 탐색 실험. 세 seed의 원결과; 코너 중앙값/90백분위는 매칭 관측 코너의 조건부 통계이며 PCK10은 전체 참조 코너를 분모로 한다. 자세는 같은 2D 레이블에서 재구성한 참조를 사용한다.}\label{sup:joint_action_results}\begin{tabular}{llrrrrr}\toprule 방법&seed&med(px)&P90(px)&PCK10&T(cm)&R(deg)\\\midrule')
    for name in ['GEO','PERM']:
        for seed in [1,2,3]:
            x=real[f'FIT_{name}_J_seed{seed}'];c,p=x['corner'],x['pose']
            snippet.append(f"{name}--J & {seed} & {number(c['matched_pooled_corner8_median_px'],3)} & {number(c['matched_pooled_corner8_P90_px'],3)} & {number(c['PCK']['10'],3)} & {number(p['translation_cm']['median'],3)} & {number(p['rotation_deg']['median'],3)}"+r'\\')
    snippet.append(r'\bottomrule\end{tabular}\end{table}')
    for ref in ['N3','PoseFix','RAW','FIT_PERM_J']:
        v=s['comparisons']['REAL_DEV'][f'FIT_GEO_J_minus_{ref}'];e=v['E_sym']
        snippet.append(f"GEO--J minus {ref.replace('_','--')}의 frame별 seed 평균 정규화 코너 거리 차이는 {number(e['delta'],8)}, 95\\% session bootstrap 구간은 [{number(e['CI95'][0],8)}, {number(e['CI95'][1],8)}]이고 고정 손상/보존 판정은 {v['exploratory_verdict']}였다.")
    snippet.append('실제 319장/13세션은 반복 사용한 개발 자료다. 모든 319장에 검출과 유효한 8코너가 존재하지만 GT instance 매칭은 311장이다. 실패와 매칭을 분리했고, 원결과/seed별 손상/같은 bank oracle gap 및 synthetic 개발 진단은 실행 보고서에 남겼다. GEO/PERM의 결합 대조와 동결 scorer의 I/J 대조는 기전의 제한된 범위만 다루며 N3/PoseFix 대비 방법 전체의 효과와 구분한다. 새로운 독립 참조의 clean/실물 가림 pair가 없으므로 독립적인 센서 6D 정확도 확증은 완료하지 못했다. 배포 runtime은 동일 입력/장치/전체 경계의 별도 영수증과 통합한다. 기존 N3 headline은 변경하지 않았다.')
    (DOC/'A_PAPER_SNIPPET.tex').write_text('\n\n'.join(snippet)+'\n')
    print('A_REPORT_AND_SNIPPET_DONE',DOC/'A_RESULT_KO.md',DOC/'A_PAPER_SNIPPET.tex')


if __name__=='__main__':main()
