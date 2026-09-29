"""Build the public Korean synthesis from frozen numeric evidence, never train/score.

No RGB, annotations, model tensors, private coordinates or new selection are read.
The root decision must exist; pending branches remain explicitly NOT_FINAL.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import common as C

ARMS = ('R0', 'OLD_REF', 'CLEAN_RAW_CLEAR', 'CLEAN_REF_CLEAR', 'CLEAN_RAW_OCC', 'CLEAN_REF_OCC')
LABELS = dict(zip(ARMS, ('R0', 'OLD REF', 'RAW CLEAR', 'REF CLEAR', 'RAW OCC', 'REF OCC')))
PAIRS = (
    ('CLEAN_REF_CLEAR', 'CLEAN_REF_OCC', '보정 타깃에서 입력 가림'),
    ('CLEAN_RAW_CLEAR', 'CLEAN_RAW_OCC', 'RAW 타깃에서 입력 가림'),
    ('CLEAN_RAW_CLEAR', 'CLEAN_REF_CLEAR', 'CLEAR에서 좌표 보정'),
    ('CLEAN_RAW_OCC', 'CLEAN_REF_OCC', 'OCC에서 좌표 보정'),
    ('OLD_REF', 'CLEAN_REF_OCC', '기존 main REF 대비'),
    ('R0', 'CLEAN_REF_OCC', 'self-training 전 대비'),
)
GROUPS = ('CLEAN29', 'MOD21', 'SEV78', 'NATURAL99', 'FULL128')
REQUIRED = ('FINAL_DECISION.json', 'EVAL_RESULTS_S42.json', 'PRIMARY_ANALYSIS_S42.json',
    'CANDIDATE_ORACLE_S42.json', 'TRAIN_TARGET_FOLLOWING_S42.json', 'TRAIN_CURVE_AUDIT.json',
    'PAIR_INTEGRITY_S42.json', 'CLEAN_LOCK.json', 'PRIMARY_PROTOCOL.json', 'PREFLIGHT.json',
    'CONTRACT_DIFFERENCE.json', 'OLD_CLEAN19_POSE_REFERENCE.json', 'RESOURCE_LEDGER.json',
    'SELECTOR_SUPERVISION_PROVENANCE.json',
    'CONTRACT_DIFFERENCE_AUDIT.md', 'CLEAN_POOL_AUDIT.md', 'TRAIN_CURVE_AUDIT.md')


def fmt(value, signed=False):
    return 'NA' if value is None else format(value, '+.4f' if signed else '.4f')


def cell(value):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def metric(row, key, stat='median'):
    return row['conditional'][key][stat]


def tr(row, stat='median'):
    return '/'.join(fmt(metric(row, key, stat)) for key in ('translation_cm', 'rotation_deg'))


def difference(before, after, stat='median'):
    return '/'.join(fmt(metric(after, key, stat)-metric(before, key, stat), True)
                    for key in ('translation_cm', 'rotation_deg'))


def decision_status(decision):
    """A document can be final without the scientific goal being achieved."""
    pending = []
    for name, branch in decision.get('branches', {}).items():
        status = branch.get('status', 'PENDING') if isinstance(branch, dict) else str(branch)
        if any(word in status.upper() for word in ('PENDING', 'RUNNING', 'NOT_READY')):
            pending.append(name)
    declared = str(decision.get('status', 'NOT_FINAL')).upper()
    explicit_final = decision.get('finalized') is True or declared in ('FINAL', 'COMPLETE', 'CLOSED')
    return ('FINAL' if explicit_final and not pending else 'NOT_FINAL'), pending


def branch_text(decision, name):
    text = decision.get('report_sections', {}).get(name)
    if text:
        return str(text)
    branch = decision.get('branches', {}).get(name)
    if branch is None:
        return 'PENDING: 이 분기의 최종 결정이 아직 제공되지 않았다.'
    if isinstance(branch, str):
        return branch
    return '`'+str(branch.get('status', 'PENDING'))+'`: '+str(branch.get('reason', '사유 미제공'))


def public_input(path):
    path = Path(path)
    if not path.is_absolute():
        path = C.DOC/path if len(path.parts) == 1 else C.ROOT/path
    path = path.resolve()
    if not path.is_relative_to(C.DOC) or path.suffix not in ('.json', '.md'):
        raise ValueError('Report may only read public namespace JSON/Markdown: '+str(path))
    return path


def verify_bindings(value):
    """Verify dependency hashes without parsing private contents."""
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            C.verify(value)
        for child in value.values():
            verify_bindings(child)
    elif isinstance(value, list):
        for child in value:
            verify_bindings(child)


def load_inputs():
    decision_path = C.DOC/'FINAL_DECISION.json'
    if not decision_path.exists():
        return None, dict(status='NOT_FINAL', reason='WAITING_FINAL_DECISION', passed=False)
    decision = C.read(decision_path)
    paths = [C.DOC/name for name in REQUIRED]
    for name in ('ZERO_FIT_SELECTOR_RESULTS.json', 'SELECTOR_PAIR_RESULTS_S42.json',
                 'SELECTOR_PAIR_RESULTS_S43.json', 'SELECTOR_SUPERVISION_PROVENANCE.json',
                 'SELECTOR_SUPERVISION_PROVENANCE.md', 'REPLICATION_DECISION.json',
                 'TRAIN_AUGMENTED_FOLLOWING.json', 'TRAIN_AUGMENTED_CPU_PREFLIGHT.json'):
        if (C.DOC/name).exists():
            paths.append(C.DOC/name)
    paths += [public_input(row['path'] if isinstance(row, dict) else row)
              for row in decision.get('result_paths', [])]
    note = C.DOC/'followups/REPEAT_PRIMARY_S43/PREFLIGHT_ENVIRONMENT_AND_LEDGER_NOTE.md'
    if note.exists():
        paths.append(note)
    paths = list(dict.fromkeys(paths))
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        return None, dict(status='NOT_FINAL', reason='MISSING_REQUIRED_EVIDENCE', missing=missing, passed=False)
    values = {path.name: C.read(path) if path.suffix == '.json' else path.read_text() for path in paths}
    for value in values.values():
        if isinstance(value, dict):
            verify_bindings(value)
    result = values['EVAL_RESULTS_S42.json']
    assert values['CLEAN_LOCK.json']['clean_locked_count'] == 78
    assert result['classification']['corrected_occlusion_value']['eligible_joint'] is False
    assert result['classification']['versus_R0']['eligible_joint'] is False
    ancestry = values['SELECTOR_SUPERVISION_PROVENANCE.json']['combined_ancestry']
    assert ancestry['unique_manual_image_ID_count'] == 19 and ancestry['unique_manual_point_ID_count'] == 86
    for arm in ARMS:
        assert result['groups']['FULL128'][arm]['frames'] == 128
        assert result['groups']['FULL128'][arm]['twoD']['corners'] == 985
        assert result['groups']['NATURAL99'][arm]['valid_pose'] == 99
    return values, dict(inputs=[C.bind(path) for path in paths])


def final_figure(result, destination):
    """Numeric-only fixed population medians; separate axes avoid label overlap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#777777', '#aa5c57', '#88acc8', '#266a9a', '#c8a759', '#298a54']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row_index, (group, title) in enumerate((('NATURAL99', 'Natural occlusion: 99 reused DEV frames'),
                                                ('CLEAN29', 'Clean preservation: 29 reused DEV frames'))):
        for column, (key, title_metric, units) in enumerate((('translation_cm', 'Translation', 'cm'),
                                                            ('rotation_deg', 'Full rotation', 'deg'))):
            axis = axes[row_index, column]
            values = [metric(result['groups'][group][arm], key) for arm in ARMS]
            bars = axis.barh([LABELS[arm] for arm in ARMS], values, color=colors)
            axis.invert_yaxis()
            axis.bar_label(bars, labels=[fmt(value) for value in values], padding=4, fontsize=9)
            axis.set_xlim(0, max(values)*1.24)
            axis.set_title(title+'\n'+title_metric, fontsize=10)
            axis.set_xlabel('Median error ('+units+'); lower is better')
            axis.grid(axis='x', alpha=.2)
            axis.set_axisbelow(True)
    fig.suptitle('Primary 2 x 2 only | fixed D9 | seed 42 | no uncertainty intervals', fontsize=12)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main_table(result, evidence=None):
    lines = ['| Method | Clean T/R | Moderate T/R | Severe T/R | Mod+Sev99 T/R | vs R0 ΔT/ΔR | vs current REF ΔT/ΔR |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        values = [tr(result['groups'][group][arm]) for group in GROUPS[:4]]
        primary = result['groups']['NATURAL99']
        values += [difference(primary[baseline], primary[arm]) for baseline in ('R0', 'OLD_REF')]
        label = LABELS[arm]+(' (S42, D9)' if arm not in ('R0','OLD_REF') else ' (D9)')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    for seed in (42,43):
        followup = (evidence or {}).get(f'SELECTOR_PAIR_RESULTS_S{seed}.json')
        if not followup:
            continue
        for arm in ('RAW_GEO','REF_GEO'):
            values = [tr(followup['groups'][group][arm]) for group in GROUPS[:4]]
            primary = followup['groups']['NATURAL99']
            values += [difference(primary[baseline],primary[arm]) for baseline in ('R0','OLD_REF')]
            lines.append(f'| {arm} OCC (S{seed}, old GEO) | '+' | '.join(values)+' |')
    return lines


def selector_table(selector):
    lines = ['| 집합 | 선택기 (동일 REF OCC S42) | T median/P90 cm | R median/P90 ° | valid/전체 | axis mismatch |',
             '|---|---|---:|---:|---:|---:|']
    for group in GROUPS:
        for name in ('D9', 'OLD_GEO'):
            row = selector['groups'][group][name]
            t = '/'.join(fmt(metric(row, 'translation_cm', stat)) for stat in ('median', 'P90'))
            r = '/'.join(fmt(metric(row, 'rotation_deg', stat)) for stat in ('median', 'P90'))
            lines.append(f'| {group} | {name} | {t} | {r} | {row["valid_pose"]}/{row["frames"]} | {row["axis_mismatch_count"]} |')
    return lines


def matched_selector_tables(values):
    lines = ['### 동일 선택기에서 RAW/REF를 공정하게 비교','',
        'GEO의 REF−R0는 학생과 선택기 둘 다 바뀐 총효과다. 좌표 보정 비교는 각 seed에서 동일 frozen GEO를 RAW/REF 두 학생에 적용한다. '
        'S43은 OCC 두 팔만 반복했으므로 CLEAR/OCC augmentation 효과의 독립 반복이 아니다.','',
        '| Seed | 동일 OCC 학생/선택기 | 자연99 T median/P90 cm | R median/P90 ° | Clean29 T/R median | valid/99 |',
        '|---|---|---:|---:|---:|---:|']
    observed = []
    for seed in (42, 43):
        result = values.get(f'SELECTOR_PAIR_RESULTS_S{seed}.json')
        if result is None:
            lines.append(f'| {seed} | PENDING / NOT_RUN | NA | NA | NA | NA |')
            continue
        observed.append((seed, result))
        for arm in ('RAW_D9', 'REF_D9', 'RAW_GEO', 'REF_GEO'):
            row = result['groups']['NATURAL99'][arm]
            t = '/'.join(fmt(metric(row, 'translation_cm', stat)) for stat in ('median', 'P90'))
            r = '/'.join(fmt(metric(row, 'rotation_deg', stat)) for stat in ('median', 'P90'))
            lines.append(f'| {seed} | {arm} | {t} | {r} | {tr(result["groups"]["CLEAN29"][arm])} | {row["valid_pose"]}/99 |')
    lines += ['', '| Seed | 공통 GEO 비교 | ΔT median/P90 cm | ΔR median/P90 ° | 둘 다 개선/악화/동률 frames |',
        '|---|---|---:|---:|---:|']
    for seed, result in observed:
        for before in ('RAW_GEO', 'R0', 'OLD_REF', 'REF_D9'):
            group = result['groups']['NATURAL99']
            after = 'REF_GEO'
            vals = ['/'.join(fmt(metric(group[after], key, stat)-metric(group[before], key, stat), True)
                            for stat in ('median', 'P90')) for key in ('translation_cm', 'rotation_deg')]
            counts = result['paired']['NATURAL99'][after+'-minus-'+before]['paired_direction_counts']
            count = '/'.join(str(counts[name]) for name in ('T_IMPROVE__R_IMPROVE', 'T_WORSEN__R_WORSEN', 'T_TIE__R_TIE'))
            lines.append(f'| {seed} | REF_GEO−{before} | {vals[0]} | {vals[1]} | {count} (99) |')
    lines += ['', '양성 median만 고르지 않고 두 seed·두 선택기를 모두 제시했다. T P90과 Clean 손익, recording 의존성은 별도 한계다. '
        '동일 DEV의 두 seed는 완전히 독립된 데이터 검증이 아니다.','']
    return lines


def augmented_train_table(result):
    if result is None:
        return ['고정 augmented-input의 covered/unmasked 잔차는 PENDING/NOT_RUN이다. 실제 노출 수와 native 잔차만으로 그 차이를 추정하지 않는다.','']
    lines = ['### 실제 증강 prefix의 planned-covered / unmasked 추종','',
        f'사전 잠근 K{result["K_batches"]}의 REAL {result["real_occurrences"]} occurrence/{result["real_unique"]} unique만 사용했다. '
        '**640×640 증강 canvas pixel** 단위이며 위 native pixel과 섞지 않는다. 각 frozen 학생을 자신의 CLEAR/OCC 입력으로 평가했다. '
        '따라서 CLEAR↔OCC 차이는 모델과 입력 난도를 함께 포함하며, 동일 입력에서의 모델 단독 비교가 아니다. '
        'canonical REF plan의 corner ID를 모든 팔에 공통 적용했다. CLEAR의 covered는 “가릴 예정이었던 점”이지 실제 가려진 점이 아니고, '
        'RAW 좌표는 동일 ID여도 사각형 밖에 있을 수 있다.','',
        '| 학생 | REF 타깃 corner 그룹 | observed/occurrence | 평균 L1 좌표 px | 평균/P90 L2 px | missing penalty 포함 L2 mean px |',
        '|---|---|---:|---:|---:|---:|']
    for arm in ARMS[2:]:
        for split, label in (('ALL_SUPERVISED','전체'),('CANONICAL_REF_PLANNED_COVERED','REF planned-covered'),('CANONICAL_REF_UNMASKED','REF unmasked')):
            row = result['groups']['ALL'][arm]['ref'][split]
            lines.append(f'| {LABELS[arm]} | {label} | {row["observed"]}/{row["point_occurrences"]} | {fmt(row["l1_mean_xy_px"]["mean"])} | '
                f'{fmt(row["l2_px"]["mean"])}/{fmt(row["l2_px"]["P90"])} | {fmt(row["L2_missing_diagonal_penalty_mean_px"])} |')
    lines += ['', '최대 confidence detection만 사용하고 target-nearest box를 고르지 않았다. true-ignore·center·source는 제외했다. '
        'covered 21개는 코너 occurrence이며 독립 이미지 21장이 아니다. 작은 prefit prefix의 TRAIN pseudo-target 추종이며 물리 정확도나 자연 가림 성능·전체320-step 학습 잔차를 뜻하지 않는다. '
        'RAW 타깃 잔차와 recording별 값도 [증강 TRAIN 진단](TRAIN_AUGMENTED_FOLLOWING.json)에 보존했다.','']
    return lines


def markdown(values, status, pending, figure):
    decision = values['FINAL_DECISION.json']
    result = values['EVAL_RESULTS_S42.json']
    analysis = values['PRIMARY_ANALYSIS_S42.json']
    oracle = values['CANDIDATE_ORACLE_S42.json']
    train = values['TRAIN_TARGET_FOLLOWING_S42.json']
    clean = values['CLEAN_LOCK.json']
    parity = values['PAIR_INTEGRITY_S42.json']
    ledger = values['RESOURCE_LEDGER.json']['totals']
    selector = values.get('ZERO_FIT_SELECTOR_RESULTS.json')
    old = values['OLD_CLEAN19_POSE_REFERENCE.json']
    verdict = decision.get('final_verdict', decision.get('verdict', 'UNSPECIFIED'))
    lines = ['# Clean 실사 → 자연 가림 자세 전이: 최종 근거 보고','',
        f'보고서 상태: `{status}`. 최종 판정: `{cell(verdict)}`. 원래 고정 D9의 primary 2×2는 **joint pose gain 없음**이다.', '',
        '가림 추가 자체의 효과, 좌표 보정 효과, 후보 선택기 교체 효과를 구분한다. 기존 GEO를 재사용하면 별도의 회복 신호가 관찰되지만, '
        '같은 GEO를 쓴 RAW/REF 비교 및 유효 seed 반복은 아래에서 별도로 판정한다. 새 manual 좌표는 0이며, '
        'GEO의 간접 학습 이력까지 합친 추적 가능한 감독은 **19이미지/86코너**로 current teacher의 9/38과 다르다. '
        '최종 판정은 [FINAL_DECISION.json](FINAL_DECISION.json)의 제한을 포함하며, 문서 완료는 물리 정답 확인이나 독립 TEST 성공을 뜻하지 않는다.','']
    if status != 'FINAL':
        lines += ['미종결 보고: '+(', '.join(pending) if pending else '최종 결정의 finalized 상태 미확정')+'. 목표 달성/종결을 주장하지 않는다.','']
    pairs = [values.get(f'SELECTOR_PAIR_RESULTS_S{s}.json') for s in (42,43)]
    if all(pairs):
        primary = [p['groups']['NATURAL99'] for p in pairs]
        beats_r0 = all(all(metric(g['REF_GEO'],k) < metric(g['R0'],k) for k in ('translation_cm','rotation_deg')) for g in primary)
        beats_raw = all(all(metric(g['REF_GEO'],k) < metric(g['RAW_GEO'],k) for k in ('translation_cm','rotation_deg')) for g in primary)
        lines += ['핵심 수치: **REF OCC + 기존 GEO**의 자연99 T/R median은 '
            f'**S42 {tr(primary[0]["REF_GEO"])}**, **S43 {tr(primary[1]["REF_GEO"])}** (cm/°)다. '
            f'R0는 {tr(primary[0]["R0"])}다. '
        f'두 stream 모두 R0보다 두 median이 낮음: {"예" if beats_r0 else "아니오"}; 동일 GEO의 RAW보다 두 median이 낮음: {"예" if beats_raw else "아니오"}. '
            '**이 판정은 median에 한정된다.** REF의 T P90은 두 seed 모두 R0/RAW보다 나쁘고, 같은 학생의 D9에서 GEO로 바꾸면 Clean 손상도 있다. '
            '이를 가림 augmentation 자체의 성공이나 모든 꼬리오차 개선으로 읽으면 안 된다.','']
    lines += ['## 1. 과거 Clean19 양성과 현재 main 음성은 같은 계약이 아니다','',
        '| 계약 | 과거 Clean19 계열 | 현재 main / 이번 clean78 |','|---|---|---|',
        '| Teacher·manual budget | Plastic10/48 + Wood9/39, 총19/87 provenance; 별도 type Replay teacher | 현재 Replay9/38: Plastic3/15 + Wood6/23; teacher 변경 없음 |',
        '| 실사·감독 | Plastic10, 48 corner support, center ignore | main217의 기존 pseudo support; 이번78로 제한하되 새 manual 좌표0 |',
        '| 학생 학습 범위 | 498 parameter tensors / 3,043,704 scalars | pose/flow132 / 539,514; 보호747 tensors·buffers |',
        '| LR·업데이트 | 1e−4, 320 updates | 1e−5, 320 updates; 같은 AdamW/5 epochs/last |',
        '| 실사 반복·합성 | 2560 real / 2560 source, Plastic10 평균256회 | 동일총노출, 217 평균11.80회 →78 평균32.82회; source512 유지 |',
        '| 가림 | native canvas, random/structured S2 placement 동시 gate | 전체640 canvas의 random-only REF plan, S2 gate 없음 |',
        '| 좌표·support | 기존 teacher와48점 계약 | 동일 RNG RAW/REF post-affine 공통 support 교집합; 기존 ignore 보존 |',
        '| pose·평가 | 기존 D9 두 후보/선택, population 별 수치 구분 | primary D9 유지; 같은128/99 재사용 DEV; selector는 별도 단계 |','',
        '과거 S1−S0의 자연99 T/R median은 '+tr(old['groups']['MODERATE_PLUS_SEVERE99']['S0'])+' → '+tr(old['groups']['MODERATE_PLUS_SEVERE99']['S1'])+' (cm/°)였다. '
        'Severe78은 '+tr(old['groups']['SEVERE78']['S0'])+' → '+tr(old['groups']['SEVERE78']['S1'])+'이나, Clean29은 '+tr(old['groups']['CLEAN29']['S0'])+' → '+tr(old['groups']['CLEAN29']['S1'])+'로 악화했다. '
        '과거 Severe81 수치와 섞지 않는다. 이 차이는 teacher/LR/scope/recording/반복량 중 하나가 원인이라는 인과 분리가 아니다.','',
        '[계약 전수 감사](CONTRACT_DIFFERENCE_AUDIT.md) · [과거 동일128 재집계](OLD_CLEAN19_POSE_REFERENCE.md)','',
        '## 2. current217 clean 감사','',
        f'`USED217_CLEAN={clean["USED217_CLEAN"]}`, `ACCEPTED249_CLEAN={clean["ACCEPTED249_CLEAN"]}`. 후보1000→accepted249→실제TRAIN217이었다. '
        '기존 사람 difficulty123장의 SHA overlap은0이어서 그 태그를 현재217의 근거로 대체하지 않았다. 모델/GT/error overlay 없는 accepted249 원본 RGB를 전수 검토했다. '
        '이는 assistant의 시각 분류이며 새 사람 manual label이나 좌표 감독이 아니다. 사용자 요청대로 마커판 부착 영상도 제외했다.','',
        'accepted249는 clean89 / 마커판 등 제외158 / 사람 겹침 불확실2, used217는 clean78 / 제외137 / 불확실2다. 외부 가림 없음은 모든 코너 visible·쉬운 시점·정확한 pseudo target을 뜻하지 않는다. '
        '개별 RGB·contact sheet·좌표는 공개하지 않는다. [clean 감사](CLEAN_POOL_AUDIT.md)','',
        '## 3. 잠근 clean 학습 모집단','',
        f'기존 used217 안의 **{clean["clean_locked_count"]}장**만 사용했다. accepted-but-unused clean11장은 추가하지 않았다. 야간62/주간16, recording4개이며 저앙각·self-occlusion·truncation은 자동 제외하지 않았다. '
        '선정은 새 fit/평가 전에 잠겼고 DEV T/R/PCK·confidence·보정량을 보지 않았다. 원본 RGB·pseudo label·manual 예산은 유지했다.','',
        '| Recording | 고유 이미지 |','|---|---:|']
    lines += [f'| {recording} | {count} |' for recording, count in clean['clean_locked_recordings'].items()]
    lines += ['','## 4. Primary 2×2와 실제 가림 노출','',
        'RAW/REF는 좌표만, CLEAR/OCC는 추가 RGB 가림만 달랐다. 같은 R0 초기879 tensors, source512/real512 epoch slots, bbox/order/base HSV·affine, optimizer320, pose/flow-only 범위와 last checkpoint를 사용했다. '
        'post-affine RAW/REF 교집합을 공통 support로 삼고 기존 true-ignore1은 승격하지 않았다. 실제320-batch 전수 parity와 frozen747 검사를 보존했다. '
        '초기879 exact는 실행 중 torch.equal assertion과 R0 hash로 확인했으며 별도 초기 tensor snapshot을 보존했다는 뜻은 아니다.','',
        '사각형은 bbox 면적0.1/0.2/0.3, 종횡비0.5/1/2,8×8 noise fill, 확률0.5,32회 시도, canonical REF에서 ≥1 covered·≥2 remaining으로 고정했다. '
        '가린 타깃도 amodal 감독으로 유지하며 가림 때문에 좌표/v/support를 바꾸지 않았다. 동일 mask에서 RAW/REF covered 수는 좌표 차이 때문에 자연히 다를 수 있다.','',
        '| 계약 | scheduled/2560 real | applied/2560 | masked supervised corners (RAW/REF) | failed placement |',
        '|---|---:|---:|---:|---:|',
        '| 과거 S1 Plastic | 1241 | 730 (28.52%) | 998 (old support) | 508 paired +3 low-support |',
        '| current v2 A | 1302 | 542 (21.17%) | 781/824 | 760 |',
        '| current v2 C exposure | 2560 | 1054 (41.17%) | 1470/1559 | 1506 |']
    raw, ref = [parity['arms'][arm] for arm in ('CLEAN_RAW_OCC', 'CLEAN_REF_OCC')]
    plan = ref['plans']; real_count = ref['roles']['REAL']['images']
    lines += [f'| 이번 clean78 S42 OCC | {plan["scheduled"]} | {plan["actually_applied"]} ({100*plan["actually_applied"]/real_count:.2f}%) | {raw["plans"]["masked_supervised_corners"]}/{plan["masked_supervised_corners"]} | {ref["scheduled_failed_placements"]} |','',
        f'이번 REF의 전체 remaining corner occurrence는{plan["remaining_supervised_corners"]}, RAW는{raw["plans"]["remaining_supervised_corners"]}다. '
        f'실사/source supervised all9 occurrence는 각각{ref["roles"]["REAL"]["supervised"]}/{ref["roles"]["SOURCE"]["supervised"]}이며 이미지50:50을 loss50:50으로 해석하지 않는다. '
        '표의 old/current는 support·canvas·S2 gate가 달라 단일 가림량 인과 비교가 아니다. [전수 parity·recording 노출](PAIR_INTEGRITY_S42.json)','',
        '## 5. 자연 Moderate/Severe 자세: 사전6개 비교','',
        '아래 T/R는 translation cm / C2-symmetric full rotation °의 median이다. vs열은 같은 자연99에서 after−baseline이며 음수가 개선이다. '
        '원래 primary 6팔 모두 pose128/128·주99/99 valid다. 아래 GEO 보완 행은 별도 단계다. 자연99는 Moderate21+Severe78 원오차를 합쳐 계산했으며 두 난도 median의 평균이 아니다.','']
    lines += main_table(result, values)
    lines += ['', f'![분리 축의 primary T/R 및 Clean preservation]({figure})','',
        '그림은 고정 D9 primary 결과만 표시한다. 막대는 median이며 오차막대·신뢰구간·독립 반복 증거가 아니다.','',
        '| 사전 비교 (후−전) | 의미 | ΔT median/P90 cm | ΔR median/P90 ° | 둘 다 개선/악화 frames |',
        '|---|---|---:|---:|---:|']
    for before, after, meaning in PAIRS:
        key = after+'-minus-'+before
        delta = analysis['comparisons']['NATURAL99'][key]
        count = result['contrasts']['NATURAL99'][key]['paired_direction_counts']
        t, r = ['/'.join(fmt(delta[metric_key][stat], True) for stat in ('median', 'P90'))
                for metric_key in ('translation_cm', 'rotation_deg')]
        lines.append(f'| {LABELS[after]}−{LABELS[before]} | {meaning} | {t} | {r} | {count["T_IMPROVE__R_IMPROVE"]}/{count["T_WORSEN__R_WORSEN"]} (99) |')
    lines += ['','REF OCC−REF CLEAR는 두 median 모두 악화했다. RAW OCC−RAW CLEAR는 T만 아주 작게 감소하고 R은 악화했다. '
        'REF OCC는 OLD REF 대비 R은 개선하지만 T는 악화하고, R0 대비 T/R 둘 다 악화했다. PCK/AUC 이득을 pose joint gain으로 바꾸어 말하지 않는다. '
        '위 median 차이와 개별 paired차이의 median은 다르며, 전수9분류/recording/LORO는 원 JSON에 있다.','',
        '| 모델 (자연99) | T median/P90 cm | R median/P90 ° | valid/99 |','|---|---:|---:|---:|']
    for arm in ARMS:
        row = result['groups']['NATURAL99'][arm]
        vals = ['/'.join(fmt(metric(row, key, stat)) for stat in ('median','P90')) for key in ('translation_cm','rotation_deg')]
        lines.append(f'| {LABELS[arm]} | {vals[0]} | {vals[1]} | {row["valid_pose"]}/99 |')
    lines += ['','[전체 난도·yaw·camera x/z·IoU3D·axis·paired·recording·LORO](EVAL_RESULTS_S42.json) · [상세 주분석](PRIMARY_ANALYSIS_S42.md)','',
        '## 6. Clean preservation','',
        'Clean29는 주99와 섞지 않는다. REF OCC−REF CLEAR의 Clean T/R median 변화는 '+difference(result['groups']['CLEAN29']['CLEAN_REF_CLEAR'], result['groups']['CLEAN29']['CLEAN_REF_OCC'])+' (cm/°)로 T 이득/R 손상이다. '
        'OLD REF 대비는 '+difference(result['groups']['CLEAN29']['OLD_REF'],result['groups']['CLEAN29']['CLEAN_REF_OCC'])+'다. 다음 P90까지 공개하며 작은 median 개선으로 tail 손상을 숨기지 않는다.','',
        '| 모델 (Clean29) | T median/P90 cm | R median/P90 ° |','|---|---:|---:|']
    for arm in ARMS:
        row = result['groups']['CLEAN29'][arm]
        vals = ['/'.join(fmt(metric(row, key, stat)) for stat in ('median','P90')) for key in ('translation_cm','rotation_deg')]
        lines.append(f'| {LABELS[arm]} | {vals[0]} | {vals[1]} |')
    lines += ['','## 7. Localization / candidate / selector 3층 분해','',
        '### L1. 좌표와 TRAIN 타깃 전달','',
        '| 모델 (자연99) | PCK5/10/20 % | >20px count/756 | full-penalty median/P90 px | axis mismatch/99 |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        row = result['groups']['NATURAL99'][arm]; xy = row['twoD']
        pck = '/'.join(f'{100*xy["PCK"][q]:.3f}' for q in ('5','10','20'))
        lines.append(f'| {LABELS[arm]} | {pck} | {xy["tail_gt20_count"]}/{xy["corners"]} | {fmt(xy["full_penalty_median_px"])}/{fmt(xy["full_penalty_P90_px"])} | {row["axis_mismatch_count"]}/99 |')
    lines += ['','2D full128 분모985 corners, matched120/128; 주99는756 corners, matched91/99다. box mismatch8을 삭제하지 않고 실패 penalty를 포함했다. '
        '고정ID 값도 원 JSON에 분리 보존했다. covered/unmasked의 실제 TRAIN 노출 및 추가 진단은 아래에서 구분한다. '
        '자연 DEV의 point를 인공 covered로 간주하지 않는다.','',
        '| TRAIN78 native 학생 | RAW 타깃 평균 px | REF 타깃 평균 px | REF occurrence가중 평균 px |',
        '|---|---:|---:|---:|']
    for arm, targets in train['groups']['ALL78'].items():
        lines.append(f'| {LABELS.get(arm,arm)} | {fmt(targets["raw_target"]["unique_corner_pooled"]["mean_px"])} | {fmt(targets["ref_target"]["unique_corner_pooled"]["mean_px"])} | {fmt(targets["ref_target"]["occurrence_weighted_corner_pooled"]["mean_px"])} |')
    lines += ['','REF 타깃 전달은 부분적으로 있으나 입력 가림을 더한 팔의 native 잔차는 더 낮지 않았다. 이 값은 100px reflection pad를 정확히 한 번 적용한 고정 native TRAIN의 pseudo-target 잔차이며, 물리정답 오차·실제가림 배치 loss·완전수렴의 증거가 아니다. '
        'center/ignore를 분리했다. 네 팔의 혼합 pose loss는 epoch1 대비 epoch5에 감소했지만 모두 마지막 epoch에서 직전보다 상승했고, real/source 개별 loss·gradient log는 없다. '
        '기존217에서는 LR1e−4의 더 낮은 TRAIN loss가 Severe T/R 동시 악화와 함께 나타났다. 낮은 pseudo loss만으로 LR 증가나 긴 학습을 처방하지 않는다. '
        '[TRAIN 추종](TRAIN_TARGET_FOLLOWING_S42.md) · [학습 곡선 감사](TRAIN_CURVE_AUDIT.md)','']
    lines += augmented_train_table(values.get('TRAIN_AUGMENTED_FOLLOWING.json'))
    lines += ['### L2. 같은 후보 집합의 사후 상한','',
        '| 모델 (자연99) | 실제 T/R median | T-optimal 후보의 T/R | R-optimal 후보의 T/R | 동일한 한 후보 joint headroom |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        row = oracle['groups']['NATURAL99'][arm]
        lines.append(f'| {LABELS[arm]} | {tr(row["current"])} | {tr(row["translation_optimal"])} | {tr(row["rotation_optimal"])} | {row["frames_with_same_candidate_joint_gain"]}/99 |')
    lines += ['','T-optimal과 R-optimal은 각각 실제 한 후보의 완전한 pose 벡터다. 서로 다른 두 최소값을 합친 가상 pose는 존재/배포한다고 주장하지 않는다. '
        'GT oracle는 학생 fit·배포 선택에 들어가지 않은 사후 진단뿐이다.','',
        '### L3. 후보 선택','',
        'REF CLEAR→REF OCC에서 PCK10은 감소했지만 T-optimal T 및 R-optimal R median은 작게 감소했다. '
        '따라서 자동 분석의 엄격한 localization+두oracle 동시 gate는 `NO_COMPLETE_SELECTOR_TRIGGER`였다. '
        '사전 허용된 후속 결정은 후보 측 제한적 근거와 REF OCC의 동일한 한 후보 joint headroom28/99를 이유로 기존 GEO의 무학습 compatibility만 먼저 실행했다. '
        '이는 selector가 전체 실패의 유일 원인이라는 판단이 아니다. [분기 이유](DECISION_AFTER_PRIMARY.md)','',
        '## 8. Selector recovery: 원래 primary와 분리','', branch_text(decision, 'selector'),'']
    if selector:
        lines += selector_table(selector)
        lines += ['',f'기존 frozen GEO는 REF OCC S42의 자연99에서 {selector["changed_by_group"]["NATURAL99"]}/99, 전체에서는 {selector["changed_by_group"]["FULL128"]}/128개 선택을 바꿨다. '
            '2D 좌표와 두 후보 pose는 그대로이며 선택을 잠근 뒤 metric을 읽었다. real GT를 선택 입력이나 fit에 넣지 않았고 새로운 selector fit은0이다. '
            'median 개선과 달리 T P90은 그대로라 R0보다 여전히 크며, Clean T/R는 같은 학생의 D9보다 악화한다. '
            '94feature의 synthetic→real 분포 차이는 기술 통계이지 새 feature 선택 근거나 최적성 증명이 아니다. '
            '[무학습 선택기 결과](ZERO_FIT_SELECTOR_RESULTS.json)','']
    lines += matched_selector_tables(values)
    lines += ['### 선택기 재사용의 감독 provenance','',
        '새 manual 좌표는 0이고 이번 학생의 teacher는 current Replay9/38 그대로다. 재사용 old GEO의 synthetic prediction feature에는 과거 Plastic10/48 학생·teacher의 간접 감독 이력이 있다. '
        '현재9/38과 과거 Plastic10/48은 ID·RGB SHA·point 중복이 0이므로 추적 가능한 합집합은 **19이미지/86코너**다. '
        '과거 Wood9/39는 이 selector feature의 의존 경로가 아니어서 더하지 않는다. '
        '따라서 selector를 포함한 최종 pipeline 전체를 “manual9/38만 사용”이라고 주장하지 않는다. '
        '합성 정답으로 selector를 학습했다는 사실과 feature를 만든 학생의 과거 real/manual 감독은 별도로 계산한다.','']
    if 'SELECTOR_SUPERVISION_PROVENANCE.md' in values:
        lines += ['[선택기 감독 이력 상세](SELECTOR_SUPERVISION_PROVENANCE.md)','']
    elif 'SELECTOR_SUPERVISION_PROVENANCE.json' in values:
        lines += ['[선택기 감독 이력 상세](SELECTOR_SUPERVISION_PROVENANCE.json)','']
    else:
        lines += ['PENDING: 선택기 간접 감독 budget 감사가 아직 제공되지 않았다.','']
    lines += ['## 9. Bridge 실험','',branch_text(decision, 'bridge'),'','기존 SmoothL1 보조항을 반복하거나 detector focal을 pose 실패의 직접 해결책으로 시험하지 않았다. '
        '남은 예산은 실행 의무가 아니며 이번 감사만으로 LR/scope/teacher/가림량 중 하나를 원인으로 확정하지 않는다.','',
        '## 10. Seed validity와 재현 한계','',
        '과거 v2의 nominal seed43은 고정 loader seed 때문에 실제 stream과879 tensors가42와 동일하여 유효 반복이 아니었다. '
        '이번 새 namespace에서만 loader construction seed를 수정했다. 실제 데이터 workers2 K8 preflight에서42↔43 order/base RGB/plan 모두8/8 달라지고43 반복8/8 exact였다. '
        '같은 seed RAW/REF 입력/box/support/plan 및 CLEAR/OCC 가림 전 입력/타깃 일치는 실제 trace로 확인했다. CPU stream 차이만으로 모델 반복 성공을 주장하지 않는다.','',
        branch_text(decision,'seed'),'']
    if 'PREFLIGHT_ENVIRONMENT_AND_LEDGER_NOTE.md' in values:
        lines += ['S43의 첫 CPU 사전검사는 sandbox의 tensor IPC 권한 때문에 중단했고, 동일 코드·workers2·seed를 host IPC 권한으로 재실행해 통과했다. '
            '이는 fit/optimizer/GPU 0인 환경 오류 재시도이며 성능이 나빠 다시 학습한 알고리즘 실패가 아니다. '
            '학습 전 자원 원장은 동일 바이트 snapshot으로 보존하고, 누적 비용은 하나의 최상위 원장에 계속 합산했다. '
            '[환경 오류 및 역사적 원장 binding 설명](followups/REPEAT_PRIMARY_S43/PREFLIGHT_ENVIRONMENT_AND_LEDGER_NOTE.md)','']
    lines += ['## 11. 설명한 것 / 배제하지 못한 것','',
        '| Cause | Evidence | Experiment | Result | Ruled out? | Still possible? |','|---|---|---|---|---|---|',
        '| current217이 모두 clean이라는 전제 | RGB249 전수, 마커판/불확실 분리 | used217 중78 lock | MIXED 확인 | 모두 clean 전제 반박 | 저앙각·self-occlusion·pseudo 오류 |',
        '| clean만 쓰면 현재 D9에서 가림 이득 복구 | source/scope/LR 고정 clean78 2×2 | REF OCC−CLEAR 주99 | T/R 모두 소폭 악화 | 이 recipe의 충분조건은 반박 | 다른 teacher/scope/data 계약 |',
        '| 보정이 전혀 학생에 전달되지 않음 | REF native TRAIN 잔차 감소 | same78 RAW/REF | 부분 전달 있음 | 완전 무전달은 반박 | 잔여 추종오차·target 자체 오류 |',
        '| 현재 pose selector가 후보 이득을 숨김 | 동일 후보 joint headroom28/99 | old GEO zero-fit | 별도 결과·seed 단계 참조 | 일부 선택 병목 지지 | tail/다른 recording/선택기 domain shift |',
        '| LR 증가만 필요 | old217 LR4 loss는 낮아도 Severe T/R 악화 | 이번 추가 LR fit 없음 | 단순 lower loss⇒better pose 반례 | 일반적인 LR 원인은 미배제 | clean78 범위의 optimization 차이 |',
        '| 가림량 또는 diversity 부족 | 실제 applied418/2560, 야간62/4 recordings | 고정 random-only primary | 현재 recipe joint gain 없음 | 원인 미확정 | 낮은 applied율·자연가림 gap·data diversity |',
        '| seed 이름만 다른 반복 | v2 stream/tensors identical | 새 loader+실제 K8/후속 trace | 아래 결정의 effective 반복 결과 | 기존 반복 신뢰성 반박 | 재사용 DEV 한계는 지속 |','',
        decision.get('report_sections',{}).get('explained','동일 모집단으로 과거 이득이 존재함을 재확인했고, 이번 clean선정만으로 현재 D9의 clean→occlusion gain이 복구되지 않음을 보였다. 과거 S1의 이득을 하나의 원인으로 설명한 것은 아니다.'),'',
        decision.get('report_sections',{}).get('unresolved','물리적 target 정답, teacher provenance의 인과효과, broad trainable scope, recording 다양성, 가림량/형태, 더 긴 최적화의 효능은 분리 확정하지 못했다. 평가6D는 기존2D와 known dimensions의 기하 재구성이며 독립 물리 측정이 아니다. 반복 DEV 결과로 독립 일반화를 주장하지 않는다.'),'',
        '## 12. 다음 한 단계와 비용','',
        decision.get('report_sections',{}).get('next_step',str(decision.get('next_step','PENDING: 최종 결정 파일에 다음 한 단계가 지정되지 않았다.'))),'',
        f'기록 시점 총 student fits={ledger["student_fits"]}, optimizer updates={ledger["optimizer_updates"]}, selector fits={ledger["selector_fits"]}, GPU training={ledger["GPU_training_seconds"]:.3f}s. '
        '상한10 students/1 selector/21600s이며 남는 예산으로 sweep하지 않았다. 이 보고 생성은 CPU 숫자 집계·도표 작성만 하며 GPU/fit/optimizer/새채점0이다. '
        'camera x/z는 카메라 축이지 차량 좌표가 아니다. 원래 annotation·checkpoints·결과는 덮어쓰지 않는다.','',
        '[재현 방법](REPRODUCE.md) · [code/data 잠금](CODE_LOCK.json) · [preflight](PREFLIGHT.json) · [비용 원장](RESOURCE_LEDGER.json) · [보고 입력/출력 SHA](REPORT_BUILD.json)','']
    additional = [public_input(row['path'] if isinstance(row,dict) else row).name for row in decision.get('result_paths', [])]
    if additional:
        lines += ['### 후속 단계 고정 근거','']+[f'- [{name}]({name})' for name in additional]+['']
    return '\n'.join(lines)


def main():
    values, meta = load_inputs()
    if values is None:
        print(json.dumps(meta, ensure_ascii=False), flush=True)
        return meta
    decision = values['FINAL_DECISION.json']
    status, pending = decision_status(decision)
    figure = C.DOC/'figures/final_primary_medians_S42.png'
    final_figure(values['EVAL_RESULTS_S42.json'], figure)
    output = C.DOC/'REPORT_KO.md'
    C.save(output, markdown(values, status, pending, str(figure.relative_to(C.DOC))))
    build = dict(passed=True, status=status, pending_branches=pending, generated_at=C.now(),
        final_verdict=decision.get('final_verdict', decision.get('verdict', 'UNSPECIFIED')),
        scientific_goal_achievement_not_inferred_from_build=True,
        inputs=meta['inputs'], artifacts=[C.bind(output), C.bind(figure)],
        implementation=[C.bind(Path(__file__))],
        publication=dict(numeric_only=True, RGB_images=0, annotation_coordinates=0,
            figure_scope='Primary S42 fixed D9 medians only; optional selector and replication reported separately'),
        new_fits=0, optimizer_updates=0, GPU_seconds=0, new_scoring=0)
    C.save(C.DOC/'REPORT_BUILD.json', build)
    print(json.dumps(dict(status=status, report=C.bind(output), figure=C.bind(figure)), ensure_ascii=False), flush=True)
    return build


if __name__ == '__main__':
    main()
