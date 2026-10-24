"""Scientific saved-result readout. No model, original RGB or evaluation-set reads."""
from pathlib import Path
import hashlib, html, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

V = Path(__file__).resolve().parent


def bind(p):
    p = Path(p).resolve()
    assert p.is_relative_to(V)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):
    return json.loads(Path(p).read_text())


def pct(x):
    return '—' if x is None else f'{100*x:.1f}%'


def run():
    ev = read(V / 'saved_common_query_evaluation_v1/COMPLETE.json')
    assert ev['PASS']
    metrics = read(ev['metrics']['path'])
    sensitivity = read(V / 'saved_crop_geometry_sensitivity_v1/COMPLETE.json')
    cues = read(V / 'saved_raw_cue_audit_v1/COMPLETE.json')
    pilot = read(V / 'public_pallox_rgb_mask_pilot_v1/COMPLETE.json')
    visual = read(V / 'crop_geometry_visual_v1/COMPLETE.json')
    assert sensitivity['PASS'] and cues['PASS'] and pilot['PASS'] and visual['PASS']
    arms = list(metrics['arms'])
    names = ['RGB crop', 'RGB + frozen SAM']
    out = V / 'scientific_readout_v1'
    assert not out.exists()
    out.mkdir()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.suptitle('SOURCE TRAIN diagnostic — not heldout or real-world accuracy', fontsize=15, y=.98)
    for j, arm in enumerate(arms):
        pair = metrics['arms'][arm]['paired']['BEFORE_to_COVER']['joint_visible_only_to_foreign_only']
        rate = pair['both_supports_correct_at_both_images'] / pair['both_available']
        ax[0].bar(j, rate * 100, width=.58, color=['#276aa4', '#b16b20'][j])
        ax[0].text(j, rate * 100 + 2, f"{pair['both_supports_correct_at_both_images']} / {pair['both_available']}", ha='center', fontsize=12)
    ax[0].set(title='Same q/L: clear → covered\nBoth support labels correct in both images', ylabel='Matched pairs (%)', ylim=(0, 108), xticks=[0, 1], xticklabels=names)
    xx = np.arange(2)
    original = [sensitivity['arms'][a]['original_correct'] / sensitivity['arms'][a]['same_known_truth_and_both_available'] * 100 for a in arms]
    common = [sensitivity['arms'][a]['common_correct'] / sensitivity['arms'][a]['same_known_truth_and_both_available'] * 100 for a in arms]
    ax[1].bar(xx - .18, original, .34, color='#57786c', label='Original BEFORE crop')
    ax[1].bar(xx + .18, common, .34, color='#a54d38', label='COVER q/L on same BEFORE image')
    for j, arm in enumerate(arms):
        s = sensitivity['arms'][arm]
        ax[1].text(j - .18, original[j] + 1.6, f"{s['original_correct']}/401", ha='center', fontsize=10)
        ax[1].text(j + .18, common[j] - 6, f"{s['common_correct']}/401", ha='center', fontsize=10, color='white')
    ax[1].set(title='Same image; known truth unchanged\nOnly crop geometry changes for RGB', ylim=(0, 108), xticks=[0, 1], xticklabels=names, ylabel='Both support labels correct (%)')
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
        a.set_yticks([0, 25, 50, 75, 100])
        a.grid(axis='y', alpha=.2)
        a.set_axisbelow(True)
    ax[1].legend(loc='upper center', bbox_to_anchor=(.5, -.13), frameon=False, fontsize=10)
    fig.text(.5, .03, 'COVER images are training inputs. C51 also changes the SAM prompt when q8 changes. No model or threshold selected.', ha='center', fontsize=9)
    fig.subplots_adjust(top=.79, bottom=.23, wspace=.3)
    png = out / 'SAME_QUERY_RESPONSE_AND_CROP_SENSITIVITY.png'
    fig.savefig(png, dpi=160)
    fig.savefig(out / 'SAME_QUERY_RESPONSE_AND_CROP_SENSITIVITY.pdf')
    plt.close(fig)
    rows = []
    for arm, label in zip(arms, names):
        for support, title in [('visible_end', '끝부위 보임'), ('foreign_front', '끝부위 외부 가림')]:
            x = metrics['arms'][arm]['strata']['ALL'][support]
            rows.append(f'<tr><td>{label}</td><td>{title}</td><td>{x["known"]}</td><td>{x["unknown"]}</td><td>{pct(x["positive_precision"])}</td><td>{pct(x["positive_recall_including_U"])}</td><td><code>{html.escape(str(x["CM_NO_YES_by_NO_YES_U"]))}</code></td></tr>')
    strata = []
    for group in [g for g in metrics['arms'][arms[0]]['strata'] if g.startswith(('material:', 'aspect:'))]:
        for arm, label in zip(arms, names):
            x = metrics['arms'][arm]['strata'][group]['foreign_front']
            strata.append(f'<tr><td>{html.escape(group)}</td><td>{label}</td><td>{x["known"]}</td><td>{x["unknown"]}</td><td>{pct(x["positive_precision"])}</td><td>{pct(x["positive_recall_including_U"])}</td></tr>')
    body = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>고정 초기 query 가림 진단 V182</title>
<style>body{max-width:1200px;margin:36px auto;padding:0 24px;font:17px/1.65 sans-serif;color:#192332;background:#f7f8fa}h1{font-size:30px}h2{margin-top:38px}a{color:#165b91}img{max-width:100%;height:auto}table{border-collapse:collapse;background:white;width:100%;font-size:15px}th,td{padding:9px;border:1px solid #cbd3dd;text-align:left}code{font-size:12px}.note{border-left:5px solid #c38a2d;background:#fff4df;padding:16px}.scroll{overflow-x:auto}</style>
<h1>고정 초기 query에서 가림 변화와 crop 민감성 확인</h1>
<p class="note"><strong>소스 TRAIN 진단입니다. 실사 성능 개선·최종 모델 채택은 미확인입니다.</strong> 새 학습 0회, 현재 평가 223장·공개 heldout·합성 VAL 신규 접근 0회, PnP·기존 원본 q/L 변경 0회입니다. 새 진단 입력은 기존 COVER 초기 추론 q/L을 4개 저장 이미지에 고정한 별도 입력이며 원래 production replay가 아닙니다.</p>
<h2>가림 변화에 반응하지만 작은 crop 변화에도 오답이 생김</h2>
<p>전체 128개 TRAIN base × 4 variant × native8 = 4,096행을 보존했습니다. 기준 COVER crop 1,024개는 원래 저장된 RGB·valid와 byte 일치했고, SAM 입력과 기존 COVER 모델 점수도 일치했습니다. COVER의 원 U24개가 4variant에 반복되어 진단 입력 U96개(2.34%)입니다. U와 정답 UNKNOWN은 다릅니다.</p>
<p>정답이 [보임, 외부 가림] = [1,0] → [0,1]인 95쌍 중, 두 모델 각각 84쌍에서 전후 두 항목을 모두 맞췄습니다. 두 모델이 함께 맞춘 공통 사례는 81쌍이며, 오류 사례가 완전히 같지는 않습니다. 각 모델의 나머지 11쌍은 가리기 전부터 적어도 한 항목이 틀렸습니다. 이 값은 학습 이미지의 반응 진단이며 새 장면 일반화 성능이 아닙니다.</p>
<img src="scientific_readout_v1/SAME_QUERY_RESPONSE_AND_CROP_SENSITIVITY.png" alt="고정 query 전후 반응과 crop 변화 민감성 그래프">
<p>같은 BEFORE 원본 이미지에서 원 q/L 대신 COVER q/L로 자른 경우, 정답이 그대로인 알려진 401행 중 두 모델 각각 11행에서 맞던 판정이 틀렸습니다. 각 4행은 q 이동 1px 이하, 각 9행은 3px 이하입니다. q와 L이 함께 달라졌으므로 위치 이동만의 효과로 분리한 결과는 아닙니다. RGB 모델은 SAM 입력이 없어 crop 기하 변화만의 민감성이고, SAM 모델은 prompt box 변화도 함께 포함합니다.</p>
<h2>저장 crop 오류 사례</h2>
<p>아래는 C50 정답→오답 사례 중 양의 q 이동이 가장 작은 TRAIN 사례입니다. 결과를 보고 선택한 오류 설명용 예시이며 대표 성공률을 뜻하지 않습니다. 같은 원본 이미지에서 q 이동 약 0.3805px, L 변화 약 +0.0545px만으로 RGB 모델은 [1,0]에서 [0,1]로 바뀌었습니다.</p>
<a href="crop_geometry_visual_v1/CARD.png"><img src="crop_geometry_visual_v1/CARD.png" alt="같은 원본 이미지의 두 저장 crop과 실제 가림 crop 비교"></a>
<p><a href="fixed_common_query_gallery_v1/index.html">결과와 무관하게 정한 8개 factor 조합·256 crop 보기</a></p>
<h2>전체 고정-query TRAIN의 알려진 정답 지표</h2><p>두 support는 독립이며 동시에 YES일 수 있습니다. CM은 정답 NO/YES 행, 예측 NO/YES/U 열입니다. Self·등록 불확실·해당 끝부위 증거 부족은 UNKNOWN으로 남겼고 외부 가림 음성으로 바꾸지 않았습니다.</p>
<div class="scroll"><table><tr><th>모델</th><th>항목</th><th>정답 known</th><th>정답 UNKNOWN</th><th>YES precision</th><th>YES recall</th><th>혼동행렬</th></tr>'''+''.join(rows)+'''</table></div>
<h2>재질·가로세로 형태별 외부 가림</h2><p>합성의 재질과 형태는 서로 다른 요인입니다. 이 표는 사용자 실사 목재/정사각형 평가표를 대체하지 않습니다.</p><table><tr><th>소스 그룹</th><th>모델</th><th>known</th><th>UNKNOWN</th><th>YES precision</th><th>YES recall</th></tr>'''+''.join(strata)+'''</table>
<h2>입력 단서와 공개 실사</h2><p>알려진 가림 전이 95쌍 모두 RGB가 달랐습니다. 바뀐 crop RGB 픽셀 수는 최소 203개·중앙값 657개이며, 해당 끝부위만의 대비 측정은 아닙니다. 전체 입력에서 byte가 같은데 알려진 support 정답이 반대인 충돌은 발견되지 않았습니다. 이것만으로 사람이거나 모델이 안정적으로 구별할 수 있다고 증명되지는 않습니다.</p>
<p>Pallox는 전체 322장 중 마스크가 있는 290장, 없는 32장을 구분했고, 미리 고정한 8쌍의 원본 16파일을 공식 크기·MD5·SHA로 검증했습니다. 실제 마스크는 앞의 cardboard를 제외하는 경계가 있지만 일부 포크 구멍을 채우고 내부 내용물의 의미도 모호합니다. 전체 bin의 보조 마스크로도 정의를 먼저 정해야 하며 native/end/외부/자기 가림 정답으로 바로 사용하지 않습니다. 이번 학습 편입 0회입니다.</p>
<p><a href="public_pallox_rgb_mask_pilot_v1/index.html">공개 실사 8쌍 원본·마스크 보기</a></p>
<h2>남은 작업</h2><p>학습 소스에서 확인한 crop 민감성을 줄이도록 같은 q의 가림 전후·비가림 대조와 작은 q/L 변화에 대한 감독을 구성하는 것이 다음 단계입니다. crop을 바꾸면 support 정답도 다시 계산하고, 모호한 공개 마스크 영역은 강한 정답으로 넣지 않습니다. 기존 평가셋이나 반복 사용한 VAL의 결과를 데이터·모델·문턱 선택에 사용하지 않습니다. 자연 실사에서 믿을 만한 외부 가림 판별은 아직 입증하지 못했습니다.</p>
<p><a href="saved_common_query_evaluation_v1/FIXED_SAME_QUERY_METRICS_AND_PAIRED_RESPONSES.json">전체 지표·전이 셀 JSON</a> · <a href="ROOT_PHASE_RESULT.json">실행 및 감사 기록</a></p></html>'''
    report = V / 'ROOT_READOUT_KO.html'
    assert not report.exists()
    report.write_text(body)
    result = dict(completed=True, source_phase_complete=True, whole_goal_solved=False, selected_model=None,
        metrics=ev['metrics'], crop_sensitivity=bind(V / 'saved_crop_geometry_sensitivity_v1/COMPLETE.json'),
        raw_cues=bind(V / 'saved_raw_cue_audit_v1/COMPLETE.json'), public_pilot=bind(V / 'public_pallox_rgb_mask_pilot_v1/COMPLETE.json'),
        plot=bind(png), report=bind(report), synthetic_TRAIN_only_not_transfer_evidence=True,
        new_training_updates=0, current_private_publicheld_VAL_new_access=0, original_qL_native_or_PnP_changes=0, goal_status='active_unsolved')
    f = V / 'ROOT_RESULTS_KO.json'
    assert not f.exists()
    f.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(completed=True, results=bind(f), report=bind(report))))


if __name__ == '__main__':
    run()
