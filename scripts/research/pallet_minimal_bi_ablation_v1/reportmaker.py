"""Render compact Korean results from sealed scores; never open reference data."""
from __future__ import annotations
from pathlib import Path
import json

from .scoring import sha
from .statistics import ARMS, CONTRASTS

LABELS = ('SIMPLER_CANDIDATE_SUPPORTED','UPSTREAM_STILL_USEFUL',
          'TRADEOFF_OR_UNRESOLVED','NO_CLEAR_ADDED_VALUE','BLOCKED_OR_INCOMPLETE')


def fmt(value, places=3):
    return 'NA' if value is None else f'{value:.{places}f}'


def mean_sd(x):
    return f"{fmt(x['mean'])} ± {fmt(x['sample_std'])} (n={x['n']})"


def make_report(results, paired, *, judgement, rationale, phase10_status='미완료/별도 진단 대기',
                timing_status='별도 실제 계측 보고서 참조'):
    if judgement not in LABELS or not rationale.strip():
        raise ValueError('Explicit evidence-based recommendation and explanation required')
    lines = ['# 최소 상태별 B/I 보정 — 전체 비교 결과','',
        f'판정: **{judgement}**. {rationale}','',
        '238개 원영상에서 네 경로를 비교했다. PRIMARY192(TEST49+추가143)가 주평가이며 CAL46은 별도다. '
        '이 자료는 반복 개발에 노출됐고 TEST라는 이름이 독립 최종 시험을 뜻하지 않는다. 운영 기본 e는 변경하지 않았다.','',
        'R0=RAW+기존 F, R1=RAW+최소 B/I+공통 B/I-only 자세 계산, '
        'N0=N3+SubPix+기존 F, N1=N3+SubPix+같은 최소 B/I 후단이다. '
        'N3와 SubPix의 개별 기여를 이 비교만으로 분리하지 않는다.','',
        '주 성공은 위치 ≤5cm와 회전 ≤5°를 동시에 만족하는 경우다. '
        '과거 strict <5/<5 결과와 정확한 경계 건수도 별도로 보존했다. '
        '±SD는 관측 오차의 표본 산포(ddof=1)이며 평균의 신뢰구간이 아니다. '
        'no-pose는 전체 성공 분모에 실패로 남기고 연속 오차는 가용 n을 표기한다.','']
    for population in ('PRIMARY','TEST','REMAINDER','CAL'):
        group = results['populations'][population]
        lines += [f'## {population} ({group["rows"]}장)','',
            '| 경로 | 성공/전체 (≤5/≤5) | 과거 strict | 위치 cm 평균±SD | 최대 cm | 회전 ° 평균±SD | 최대 ° | no-pose | fallback | 제약수락 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            s = group['arms'][arm]
            acc = 'NA' if not s['accepted_available_n'] else f"{s['accepted_n']}/{s['accepted_available_n']}"
            fall = 'NA' if not s['fallback_available_n'] else str(s['fallback_n'])
            lines.append(f"| {arm} | {s['successes_le5']}/{s['denominator']} | {s['successes_strict_lt5']} | "
                f"{mean_sd(s['errors']['translation_cm'])} | {fmt(s['errors']['translation_cm']['max'])} | "
                f"{mean_sd(s['errors']['rotation_deg'])} | {fmt(s['errors']['rotation_deg']['max'])} | "
                f"{s['no_pose_n']} | {fall} | {acc} |")
        lines += ['', '연속 오차는 공통 가용 영상의 방법별 통계와 각 대비의 공통 쌍 n도 RESULTS.json/PAIRED_RESULTS.json에 함께 보존했다. '
                  '미산출 때문에 어려운 영상이 평균에서 빠진 경우 이를 개선으로 해석하지 않는다.','']
    primary = results['populations']['PRIMARY']
    lines += ['## 사전 고정 네 대비','',
        '| 대비 | 성공률 차이 %p | 95% 군집 CI %p | 실패→성공 | 성공→실패 | 둘 다 성공 | 둘 다 실패 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,b in CONTRASTS:
        x = paired['PRIMARY']['comparisons'][a+'-'+b]
        d = x['success_rate']; c = x['transition']['counts']; ci = d['CI95']
        interval = 'NA' if ci is None else f"[{fmt(ci[0]*100)}, {fmt(ci[1]*100)}]"
        lines.append(f"| {a}−{b} | {fmt(None if d['mean_difference'] is None else d['mean_difference']*100)} | "
            f"{interval} | {c['failure_to_success']} | {c['success_to_failure']} | {c['both_success']} | {c['both_failure']} |")
    boot = primary['bootstrap']
    lines += ['', f"원녹화 군집 {boot['units']}개를 통째로 10,000회 재표집했다(seed 20260917). "
        '모든 대비·지표에서 동일 draw를 공유하며 재표집 내부의 원영상 수로 가중해 원표의 영상 성공률 차이와 같은 estimand를 유지한다. '
        '이 seed는 기존 통계 재표집 seed이며 새 학습 seed가 아니다. 군집별 장수와 draw SHA는 JSON에 있다. '
        '소수 군집·반복 개발 자료의 불확실성 요약이며 과거 방법 선택 전체를 보정한 독립 검증 구간이 아니다.','',
        '기존 성공률 검정 정의를 검증하지 못해 원 p값과 4개 대비 Holm 보정 p값은 NA다. '
        'CI에서 p값을 역산하지 않았고 비유의·CI의 0 포함을 동등성/비열등성 증거로 쓰지 않는다. '
        '허용 손실폭을 새로 만들지 않았다.','',
        '## 단계 진단과 남은 확인','',
        f'좌표 교체 진단: {phase10_status}. 상태·공분산·지지에 조건화된 진단이며 자연스러운 무보정 기준선이나 인과 증명은 아니다.','',
        f'시간: {timing_status}. 캐시 읽기나 서로 다른 저장 시간을 합친 값을 전체 RGB 추론 시간으로 부르지 않는다.','',
        '실제 관측 원·좌표와 최종 모델 wire·투영을 구분한다. S는 초기 모델 자세를 조건으로 한 상태이며 '
        '후단 국소 질의 제외가 초기 F의 S 영향까지 없앴다는 뜻은 아니다. 고정 상태 모델은 CAL 실사 감독을 사용한 이력이 있고 '
        'RAW 출처의 적용 범위 제한이 남는다.','',
        '모든 성공 전환 ID는 [CHANGED_CASES.csv](CHANGED_CASES.csv), 전체952행은 '
        '[ROW_METRICS.csv](ROW_METRICS.csv), 통계 원문은 [RESULTS.json](RESULTS.json)과 '
        '[PAIRED_RESULTS.json](PAIRED_RESULTS.json)에 있다.','']
    return '\n'.join(lines)


def write_report(folder, *, judgement, rationale, **kwargs):
    folder = Path(folder)
    seal = json.loads((folder/'SCORE_SEAL.json').read_text())
    for name, digest in seal['assets'].items():
        if sha(folder/name) != digest: raise ValueError('Sealed score asset changed')
    target = folder/'RESULTS_KO.md'
    if target.exists(): raise FileExistsError('Preserve the previous report')
    results = json.loads((folder/'RESULTS.json').read_text())
    paired = json.loads((folder/'PAIRED_RESULTS.json').read_text())
    target.write_text(make_report(results, paired, judgement=judgement, rationale=rationale, **kwargs))
    return sha(target)
