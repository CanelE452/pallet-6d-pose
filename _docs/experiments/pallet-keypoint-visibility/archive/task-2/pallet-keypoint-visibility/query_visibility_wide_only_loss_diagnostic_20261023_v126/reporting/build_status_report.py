"""Saved-results reporting only. Does not load models, feature matrices, or RGB."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

V = Path(__file__).resolve().parents[1]
P = V.parent
OUT = V / 'visual_followup'
OUT.mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text())

def bind(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

control = read(V / 'saved_controlled_review/REVIEW.json')
capacity = read(V / 'capacity_probe/FULL_BATCH_LBFGS_MAX400/RESULTS.json')
capacity_review = read(V / 'capacity_probe_review/REVIEW.json')
assert control['PASS'] and capacity_review['PASS'] and capacity['completed']
primary = control['primary_fixed1000_comparison']
current = read(P / 'LATEST_STATUS_KO.json')['latest_current_evaluation']
assert current['version'] == 'V124_C16'
cms = current['paired_D_E_rows_D_E_cols_visible_hidden_U']
d_ok = sum(cm[0][0] for cm in cms.values())
d_n = sum(sum(cm[0]) for cm in cms.values())
e_ok = sum(cm[1][1] for cm in cms.values())
e_n = sum(sum(cm[1]) for cm in cms.values())
assert (d_ok, d_n, e_ok, e_n) == (1137, 1147, 35, 90)
for name, pair in [('C17_FINAL1000', (223, 121)), ('C18_FINAL1000', (229, 120))]:
    m = primary[name]['VALIDATION_COMBINED']
    assert (m['E_qOUT']['correct'], m['SHAM_D_qIN']['correct']) == pair
assert capacity['groups']['TRAIN']['all']['correct_available'] == 795

results = {
    'completed': True,
    'created_UTC': datetime.now(timezone.utc).isoformat(),
    'goal_status': 'active', 'fully_solved': False,
    'phase': 'V126_C18_MATCHED_OLDLOSS_ZERO_CONTROL_AND_TRAIN_CAPACITY_DIAGNOSIS',
    'outcome_KO': '원인 진단은 진전됐지만 새 장면의 외부 가림 판단은 해결되지 않았다.',
    'primary_fixed1000_control': primary,
    'controlled_intervention_KO': 'C17/C18은 동일 C16 warmstart·배치·특징·head·1,000회에서 old192 손실 계수만 1→0. selected900 결과와 혼용하지 않음.',
    'capacity_probe_separate_from_control': capacity,
    'latest_current_evaluation_unchanged': current,
    'new_current_or_V112_or_PnP_calls': 0,
    'findings_KO': [
        '기존 손실만 제거한 고정1,000회 실험의 효과는 작고 조건별로 상충한다. oldloss 제거만으로 해결되지 않았다.',
        '동일 1,928차원 특징과 221,825개 파라미터 head는 별도 최적화 검사에서 TRAIN795/795를 맞혔다. 이 유한 자료를 맞힐 능력은 있다.',
        '그 head의 별도 공개 검증 정답은2664/3582(74.4%), 어려운 가림325/483(67.3%), 가림체가 query를 덮어도 목표 구조는 보이는 경우197/394(50.0%)였다.',
        'TRAIN은6개 cue/3개 원본에서 만든 상관된 변형이며, 검증도 공개 개발 자료다. 서로 독립인 수천 개 자연 장면으로 해석하면 안 된다.',
        '과적합 검사에서 검증 BCE5.03으로 확신한 오판도 남았다. TRAIN100%는 전이 성공·신뢰도 보정·배포 성능을 뜻하지 않는다.',
        '현재 실사 최신 평가 C16은 보임1137/1147, 외부가림35/90이며55개 외부가림을 놓친다. C17/C18/용량검사 head의 현재 실사 평가는 수행하지 않았다.',
    ],
    'goal_assessment_KO': '계속 검증할 근거는 있으나 성공을 예측할 근거는 아직 부족하다. 목표가 불가능하다고 판정할 결과도 없다. 학습 자료 적합과 새 장면 일반화를 분리하는 단계다.',
    'next_action_KO': [
        '공개 원본의 구조·가림체·조명 다양성을 늘리고 촬영 원본 단위로 분리해 일반화를 검증한다.',
        '현재 pooled 특징과 경계의 위치 관계를 유지한 특징을 동일 공개 조건에서 비교한다. 특징 부족은 아직 입증되지 않은 가설이다.',
        '현재223은 평가에만 사용한다. 자기 가림은 초기 기하 단계에서 처리하고, 외부 가림 판단에 집중한다. 좌표 보정/PnP는 하지 않는다.',
    ],
    'limits_KO': [
        '공개 합성가림은 기록된 특정 구조 cue를 기준으로 한 조건부 AUX 라벨이다. native8 물리 키포인트의 인간 정답과 동일하다고 주장하지 않는다.',
        '현재223은 이전 논의에서 이미 본 개발 평가셋으로 새 blind holdout이 아니다. 이번 실험의 학습/문턱/체크포인트 선택에는 사용하지 않았다.',
        '기존 손실의 영향이 모든 설정에서 없다고 증명한 실험이 아니다. 특징이나 head의 일반적인 충분성도 증명하지 않았다.',
    ],
    'evidence': {name: bind(V / rel) for name, rel in {
        'prospective_control_design': 'ROOT_C18_DESIGN_BEFORE_SOURCE_FIT.json',
        'actual_C18_fit_entry': 'ROOT_ACTUAL_C18_SOURCE_FIT_ENTRY.json',
        'controlled_saved_review': 'saved_controlled_review/REVIEW.json',
        'capacity_policy': 'ROOT_CAPACITY_PROBE_POLICY_BEFORE_FIT.json',
        'actual_capacity_entry': 'ROOT_ACTUAL_CAPACITY_PROBE_ENTRY.json',
        'capacity_results': 'capacity_probe/FULL_BATCH_LBFGS_MAX400/RESULTS.json',
        'capacity_saved_review': 'capacity_probe_review/REVIEW.json',
        'representation_contract': 'representation_contract_review/REVIEW.json',
    }.items()},
}

font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
plt.rcParams.update({'font.family': 'NanumGothic', 'axes.unicode_minus': False, 'font.size': 12})
fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
fig.patch.set_facecolor('#f7f8fa')
fig.suptitle('진단의 진전과 남은 성능 문제', fontsize=22, y=.98)
labels = ['가려졌고 q는 가림체 밖', '구조는 보이고 q는 가림체 안']
colors = ['#697b8f', '#d58034']
for idx, name in enumerate(['C17_FINAL1000', 'C18_FINAL1000']):
    m = primary[name]['VALIDATION_COMBINED']
    vals = [m['E_qOUT'], m['SHAM_D_qIN']]
    x = [i + (idx - .5) * .34 for i in range(2)]
    bars = axes[0].bar(x, [v['correct'] / v['available_N'] * 100 for v in vals], .32,
                       label=['기존 손실 포함', '기존 손실 제거'][idx], color=colors[idx])
    for bar, value in zip(bars, vals):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                     f"{value['correct']}/{value['available_N']}", ha='center', fontsize=10)
axes[0].set_xticks([0, 1], ['가림 · q 밖', '보임 · q 안'])
axes[0].set_title('① 기존 손실 제거만으로는 해결 안 됨\n공개 검증 · 동일 1,000회 비교', fontsize=13, pad=15)
axes[0].legend(loc='upper right', fontsize=10)

groups = capacity['groups']
vs = [groups['TRAIN']['all'], groups['VALIDATION_COMBINED']['all']]
bars = axes[1].bar([0, 1], [v['available_accuracy']*100 for v in vs], color=['#27836c', '#be7841'], width=.55)
for bar, value in zip(bars, vs):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 f"{value['correct_available']}/{value['known_available']}", ha='center', fontsize=11)
axes[1].set_xticks([0, 1], ['학습 자료', '별도 공개 검증'])
axes[1].set_title('② 같은 특징·head로 학습 자료는 맞힘\n별도 최적화 검사 · 실사 성능 아님', fontsize=13, pad=15)

bars = axes[2].bar([0, 1], [d_ok/d_n*100, e_ok/e_n*100], color=['#27836c', '#c65050'], width=.55)
for bar, k, n in zip(bars, [d_ok, e_ok], [d_n, e_n]):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f'{k}/{n}', ha='center', fontsize=11)
axes[2].set_xticks([0, 1], ['보임 정답률', '외부 가림 정답률'])
axes[2].set_title('③ 현재 실사의 최신 평가 C16\n이번 진단 모델의 실사 평가는 아직 없음', fontsize=13, pad=15)
for ax in axes:
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.17)
    ax.set_axisbelow(True)
fig.text(.5, .035, '공개 자료는 특정 구조 cue의 보임/가림 과제 · 현재 실사는 별도 과제/집합 · 세 패널을 동일 성능 척도로 합산하지 않음', ha='center', fontsize=11)
fig.tight_layout(rect=[.01, .09, .99, .91], w_pad=2)
figure = OUT / 'STATUS_AND_REMAINING_GAP_KO.png'
fig.savefig(figure, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
results['figure'] = bind(figure)
write(V / 'ROOT_RESULTS_KO.json', results)

def esc(text):
    return html.escape(str(text), quote=True)

def link(path, label):
    return f'<a href="{esc(path)}">{esc(label)}</a>'

def rate(k, n):
    return f'{k:,}/{n:,} ({k/n*100:.1f}%)'

public_rows = ''
for label, group in [('학습 자료', 'TRAIN'), ('별도 LOCO 검증', 'LOCO_VAL'), ('TOMIE 개발 검증', 'TOMIE_DEV'), ('공개 검증 합계', 'VALIDATION_COMBINED')]:
    a = groups[group]['all']
    e, d = (groups[group]['strata'][key] for key in ['E_qOUT', 'SHAM_D_qIN'])
    public_rows += '<tr>'+''.join(f'<td>{esc(x)}</td>' for x in [label, rate(a['correct_available'], a['known_available']), rate(e['correct_available'], e['known_available']), rate(d['correct_available'], d['known_available']), f"{a['mean_available_BCE']:.5g}"] )+'</tr>'
current_rows = ''
for kind, cm in cms.items():
    current_rows += '<tr>'+''.join(f'<td>{esc(x)}</td>' for x in [kind, rate(cm[0][0],sum(cm[0])), rate(cm[1][1],sum(cm[1])), str(cm)])+'</tr>'
evidence_links = '<li>'+('</li><li>'.join(link(b['path'], name) for name,b in results['evidence'].items()))+'</li>'
body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>V126 현재 진전과 성공 가능성</title>
<style>body{{font-family:system-ui,"Noto Sans CJK KR",sans-serif;background:#f6f7fa;color:#202b35;margin:0}}main{{max-width:1150px;margin:auto;padding:30px}}section{{background:white;padding:22px 26px;margin:20px 0;border:1px solid #dfe4e9;border-radius:12px}}h1{{font-size:30px}}h2{{font-size:21px}}p,li{{line-height:1.8}}.lead{{font-size:19px}}.note{{color:#4a5968}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce2e8}}a{{color:#1467a4}}.scroll{{overflow:auto}}</style>
<main><h1>원인 진단은 진전됐고, 외부 가림 성능은 아직 부족합니다</h1>
<p class="lead">성공 가능성을 더 검증할 근거는 있습니다. 다만 현재 실사의 외부 가림은 <b>35/90개</b>만 맞혀 신뢰할 수 있는 해결책을 확보했다고 말할 단계는 아닙니다.</p>
<section><img src="visual_followup/STATUS_AND_REMAINING_GAP_KO.png" alt="기존 손실 제거 통제, 학습 적합 검사, 현재 실사 성능을 구분한 그래프"><p class="note">공개 검증과 현재 실사는 과제와 집합이 다릅니다. 위 수치를 한 성능 추세로 연결하거나 합산하지 않습니다.</p></section>
<section><h2>이번에 확인한 것</h2><p>기존 목표들이 새 가림 판단을 방해하는지 보려고, 동일한 데이터·초기 모델·배치·1,000회 학습에서 기존 손실만 제거했습니다. 공개 검증의 어려운 가림 정답은 <b>223→229/483</b>, 어려운 보임 정답은 <b>121→120/394</b>였습니다. 이 변경만으로 해결되지 않았습니다. 비교는 양쪽 고정1,000회이며 부차적인 selected900 결과를 섞지 않았습니다.</p>
<p>그다음 같은 특징과 head가 학습 자료 자체를 맞힐 수 있는지 별도 최적화 검사로 확인했습니다. <b>795/795개를 맞혔습니다.</b> 이 유한 학습 자료를 맞히지 못하는 것이 근본적인 한계라는 설명은 지지되지 않습니다. 그러나 이 검사는 정규화 없는 적합 능력 진단이며, 기존 실험과의 단일 요인 성능 비교가 아닙니다.</p>
<p>별도 공개 검증에서는 여전히 틀리고, 틀린 답에도 큰 확신을 보였습니다. 학습 자료는 6개 구조 cue와 3개 원본에서 만든 변형입니다. 원본 장면과 구조의 다양성, 특정 목표 구조와 가림체의 위치 관계를 새 장면에서도 구별하는 능력이 남은 검증 대상입니다.</p></section>
<section><h2>별도 적합 능력 검사의 결과</h2><p>‘가림·q 밖’은 목표 구조가 가려졌지만 초기 좌표 q 자체는 가림체 밖인 경우입니다. ‘보임·q 안’은 가림체가 q를 덮어도 목표 구조는 보이는 경우입니다. 단순한 q 픽셀의 소유 여부와 구별하려는 조건입니다.</p>
<div class="scroll"><table><tr><th>집합</th><th>전체 정답</th><th>가림·q 밖</th><th>보임·q 안</th><th>평균 BCE</th></tr>{public_rows}</table></div>
<p class="note">공개 자료의 합성 가림 및 조건부 구조 cue 라벨입니다. 수천 개의 독립 자연 실사 정답이나 native8 키포인트 정답을 확보한 결과가 아닙니다. 학습 예정846개 중 입력을 사용할 수 없는51개는 U로 남기고795개만 최적화했습니다. 검증3,582개는 모두 입력이 유효했습니다. 검증 라벨은 최적화나 체크포인트 선택에 쓰지 않았습니다.</p></section>
<section><h2>현재 실사의 최신 평가: C16 유지</h2><div class="scroll"><table><tr><th>종류</th><th>보임 recall</th><th>외부 가림 recall</th><th>혼동행렬 D/E 행, D/E/U 열</th></tr>{current_rows}</table></div>
<p>합계 보임 <b>{rate(d_ok,d_n)}</b>, 외부 가림 <b>{rate(e_ok,e_n)}</b>입니다. 외부 가림55개를 놓칩니다. 이번 C17/C18/적합 검사 head로 현재 실사 평가를 새로 돌리지 않았습니다. 자기 가림은 초기 기하에서 처리한다는 방침을 유지합니다.</p>
<p>{link(current['report'], 'C16 실제 crop 성공·실패 및 전체 외부가림90개 갤러리')}</p></section>
<section><h2>성공 가능성과 다음 판단 기준</h2><p>현재 결과로 성공을 장담하거나 확률을 정할 수는 없습니다. 목표가 불가능하다고 판정할 근거도 없습니다. 초기 좌표 주변에서 어떤 목표 구조가 보여야 하는지와 실제 가림을 구별하는 과제 자체는 유지할 만합니다. 다만 지금의 공개 자료 수와 표현으로 일반화가 된다는 증거는 부족합니다.</p><p>다음은 공개 원본의 구조·재질·가림체·조명을 다양하게 만들고 원본 단위로 분리한 검증에서, 보이는 구조를 지키면서 외부 가림도 함께 맞히는지 확인하는 것입니다. 경계의 공간 배치를 유지한 특징도 같은 공개 조건에서 비교할 수 있습니다. 특징 부족은 아직 가설이며 이번 TRAIN100%만으로 충분성을 결론내리지 않습니다.</p><p class="note">현재223은 학습·문턱·체크포인트 선택에 사용하지 않습니다. 이미 논의에 노출된 개발 평가셋이므로 새 blind holdout으로 부르지 않습니다. PnP와 초기 좌표 보정은 수행하지 않았습니다.</p></section>
<section><h2>검증 자료</h2><p>두 저장 결과 검토 모두 PASS입니다. 아래 기록에서 배치 일치, 기존 손실 gradient0, 분리된 고정1,000회 결과 및 TRAIN/검증 계산을 확인할 수 있습니다.</p><ul>{evidence_links}</ul><p>{link(V/'ROOT_RESULTS_KO.json','전체 결과 JSON')}</p><p class="note">이 HTML은 정적 링크·수치를 검사했습니다. 그래프 PNG를 직접 시각 검수하며, 브라우저 렌더링 확인을 주장하지 않습니다.</p></section></main></html>'''
(V / 'ROOT_READOUT_KO.html').write_text(body)
print(json.dumps({'report': str(V/'ROOT_READOUT_KO.html'), 'figure':str(figure), 'results':str(V/'ROOT_RESULTS_KO.json'), 'current_D':[d_ok,d_n], 'current_E':[e_ok,e_n]}, ensure_ascii=False))
