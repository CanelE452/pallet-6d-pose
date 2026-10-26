"""Explain the bounded result using saved scores/crops and actual pooling geometry."""
from pathlib import Path
import csv
import hashlib
import html
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

W = Path(__file__).absolute().parent
R = W.parent


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(x in str(p).lower() for x in ('v112', 'cache', '.pyc'))
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink()
    return p


def bind(p):
    p = safe(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size_bytes=p.stat().st_size)


def read(b):
    assert bind(b['path']) == b
    return safe(b['path'])


def save(name, text):
    p = safe(W / name); assert not p.exists(); p.write_text(text)


def main():
    complete = json.loads(safe(W/'evaluation_v1/COMPLETE.json').read_text())
    summary = json.loads(read(complete['summary']).read_text())
    syn = json.loads(safe(W/'syn_training_v1/SYN_DEV_RESULTS.json').read_text())
    pred_freeze = json.loads(safe(W/'CURRENT_PREDICTIONS_FROZEN.json').read_text())
    with np.load(read(pred_freeze['feature_packet']), allow_pickle=False) as z:
        uv = z['uv'].astype(float)
    with np.load(read(complete['visuals']['heatmap_source']), allow_pickle=False) as z:
        probability = z['probability'].astype(float)
    c = json.loads(read(pred_freeze['current_source_complete']).read_text())
    support = np.load(read(c['outputs']['TOKEN_SUPPORT.npy']), allow_pickle=False).astype(bool)
    yy, xx = np.mgrid[:16, :16]
    grid = (np.stack((xx, yy), -1) + .5) / 16
    reconstructed = (probability[..., None] * grid).sum((1, 2))
    maxdiff = float(np.abs(reconstructed - uv).max())
    assert maxdiff <= 2e-6
    case_data = json.loads(read(complete['visuals']['case_data']).read_text())
    cases = {r['global_index']: r for r in case_data['cases']}
    original_join = safe(R/'V263_CURRENT_EVALUATION_A/actual_evaluation_v1/ALL1784_CLASSIFICATION_LOCATION_JOIN.csv')
    assert bind(original_join)['sha256'] == 'df7b619649ede26af4e21644c22b10e84b334c6bc8708f4e5a668ec373db5175'
    rows = list(csv.DictReader(original_join.open()))
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.6), dpi=170)
    for ri, index in enumerate((840, 908)):
        case, row = cases[index], rows[index]
        rgb = np.asarray(Image.open(read(case['clean'])).convert('RGB'))
        assert rgb.shape == (96, 96, 3)
        d2 = ((grid * 16 - uv[index] * 16) ** 2).sum(-1)
        weight = np.exp(-d2 / (2 * 1.5 ** 2)) * support[index]
        weight /= weight.sum()
        centre = uv[index] * 96 - .5
        q = np.array([float(row['query_x']), float(row['query_y'])])
        reference = np.array([float(row['human_reference_x']), float(row['human_reference_y'])])
        reference = (reference - q) * 96 / float(row['JOINT_SELECTED__crop_L']) + 47.5
        state = '보임 D' if ri == 0 else '외부 가림 E'
        score = f"G{index} · 기존 주석 {state} · 전체 평균 {case['GLOBAL_MATCHED']['pred']} → 위치 주변 {case['HEATMAP_LOCAL']['pred']}"
        axes[ri, 0].text(0, 1.17, score, transform=axes[ri, 0].transAxes,
                         fontproperties=font, fontsize=11, clip_on=False)
        for ci, title in enumerate(('실제 crop', '고정 모델의 위치 heatmap', '새 분류기의 국소 token 가중 영역')):
            ax = axes[ri, ci]
            ax.imshow(rgb, interpolation='nearest')
            if ci:
                value = probability[index] if ci == 1 else weight
                value = (value-value.min()) / max(float(value.max()-value.min()), 1e-12)
                ax.imshow(value, cmap='magma', alpha=.58, interpolation='bilinear',
                          extent=(-.5,95.5,95.5,-.5), vmin=0, vmax=1)
            ax.plot(47.5, 47.5, 'o', color='#00d4ff', fillstyle='none', markersize=8)
            ax.plot(*reference, '+', color='#ff42c4', markersize=10, mew=2)
            ax.plot(*centre, '+', color='#f5ff30', markersize=10, mew=2)
            if ci == 2:
                ax.add_patch(plt.Circle(centre, 18, fill=False, color='#a4ff40', linewidth=1.3))
            ax.set_xlim(-.5,95.5); ax.set_ylim(95.5,-.5); ax.axis('off')
            ax.set_title(title, fontproperties=font, fontsize=10)
    fig.suptitle('보이는 모서리를 살리는 사례와 가림을 놓치는 사례', fontproperties=font, fontsize=17)
    fig.text(.5, .025, '파랑○ 초기 query · 분홍+ 기존 주석 · 노랑+ heatmap 위치 · 초록 원 국소 Gaussian 2σ\n색은 위치/가중치의 상대 표시이며 보임 확률이 아닙니다. DINO 공간 특징은 문맥을 포함합니다.',
             fontproperties=font, fontsize=10, ha='center')
    fig.subplots_adjust(left=.035,right=.985,top=.84,bottom=.12,wspace=.05,hspace=.48)
    p = safe(W/'LOCAL_POOL_SUCCESS_FAILURE_KO.png'); assert not p.exists(); fig.savefig(p);plt.close(fig)
    primary = [r for r in summary['classification'] if r['cohort']=='PRIMARY']
    result = {'completed':True, 'new_model_adopted':False, 'new_fits':2, 'steps_each':1000,
              'synthetic_only_new_fit':True, 'real223_role':'evaluation only; previously repeatedly consulted development data',
              'location_head_and_coordinates_changed':False, 'PnP':0,
              'SYN_DEV':syn['results'], 'current_PRIMARY':primary,
              'current_near_manual_D': [r for r in summary['classification'] if r['cohort']=='MANUAL_CLICK_D_LE3_VALID_INPUT'],
              'heatmap_reuse_check':{'max_abs_uv_difference_saved_vs_recomputed':maxdiff,'tolerance':2e-6},
              'conclusion':'Local pooling rescues visible points but misses more external occlusions; not an overall D/E improvement.',
              'visual_hypothesis':'G908 local pool follows cone boundary displaced from labelled requested point. Target identity/direct evidence remains unresolved; this is not a causal proof.',
              'qualifications':['G996 disputed E annotation/native mapping retained, not presented as definite failure.',
                                'G981/G200 small3px threshold crossings not gross visual failures.',
                                'No further fit or threshold search after current results.'],
              'actual_gaussian_visualization':bind(W/'LOCAL_POOL_SUCCESS_FAILURE_KO.png')}
    save('FINAL_RESULT_SUMMARY.json',json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    table=[]
    for r in primary:
        if r['model'] not in ('GLOBAL_MATCHED','HEATMAP_LOCAL','V263_JOINT','V261_CONTROL','CAL_LEGACY'):continue
        values=[r['group'],r['model'],f"{r['D_D']}/{r['truth_D']}",f"{r['E_E']}/{r['truth_E']}"]
        values += [f"{r[k]*100:.1f}%" for k in ('E_precision','E_recall','E_F1','U_rate')]
        table.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in values)+'</tr>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V264 · Heatmap 주변 특징으로 가림 판단</title>
<style>body{font:17px/1.7 sans-serif;max-width:1180px;margin:36px auto;padding:0 22px;color:#172431}img{width:100%}aside{padding:18px;background:#fff2df}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #cdd5df;padding:6px}code{background:#eef3f8}a{color:#145ea5}</style>
<h1>V264 · Heatmap이 찾은 위치 주변 특징으로 보임/외부 가림 판단</h1>
<aside><b>완료 · 새 모델 채택 안 함</b><br>보이는 점을 가림으로 지우는 오류는 줄었지만 외부 가림 누락이 늘었습니다. 합성의 개선이 실사 외부 가림 판별까지 이어지지 않았습니다.</aside>
<h2>무엇을 바꿨나</h2><p>기존 DINO·위치 heatmap·crop·초기 query·native 대응·좌표를 고정했습니다. 동일한 780→32→1 분류기 두 개를 학습했습니다.
전체 평균 대조군은 [전역 특징, 전역 특징, 보조 정보], 위치 주변 모델은 [전역 특징, heatmap 예측 위치의 Gaussian 국소 특징, 같은 보조 정보]를 읽습니다.
국소 σ는 1.5 token 칸으로 사전에 고정했습니다. 보조 정보는 키포인트 번호·예측 이동량·분포 peak·entropy로 동일합니다.</p>
<p>합성 FIT86 base/DEV24 base를 유지했습니다. FIT 라벨 D4,626/E190 중 입력 불가를 제외한 D4,572/E188을 D16/E16 균형 배치로 표집했습니다.
초기값·배치가 같고 각25,025 parameter·1,000 AdamW update·seed17입니다. 최종 checkpoint와 logit0 임계값을 실사 추론 전 동결했습니다. 자기 가림 S는 E 라벨로 만들지 않았습니다.</p>
<h2>합성에서는 개선, 실사에서는 상충</h2><p>합성 DEV는 전체 평균 D→E66·E 정답50/56에서, 위치 주변 D→E9·E 정답54/56으로 개선됐습니다.
실사 PRIMARY1,237점에서는 D→E502→119로 줄었지만 E→D19→67로 늘었습니다. 균형 recall은67.6→57.6%, 외부 가림 F1은21.4→19.8%입니다.</p>
<p>현재223장은 이번 fit·normalizer·설정/threshold 선택에 사용하지 않았습니다. 과거 반복 확인한 개발 평가 자료이므로 독립 검증이라고 부르지 않습니다.
옛 V263과 차이에는 균형 학습·특징/분류기 차이도 포함되며, 국소 pooling 효과는 새 전체 평균 대조군과 비교해야 합니다. CAL_LEGACY는 과거 실사 CAL 학습 이력이 있습니다.</p>
<img src="evaluation_v1/PRIMARY_COMPARISON_KO.png" alt="동일1237점 모델 비교">
<table><tr><th>유형</th><th>모델</th><th>보임 정답</th><th>외부 가림 정답</th><th>E precision</th><th>E recall</th><th>E F1</th><th>보류율</th></tr>TABLE</table>
<h2>실제 읽는 위치와 성공/실패</h2><img src="LOCAL_POOL_SUCCESS_FAILURE_KO.png" alt="실제crop과 학습위치heatmap 및 분류기의국소가중영역">
<p>G840은 보이는 팔레트 모서리 쪽 국소 특징으로 D를 복구했습니다. G908은 기존 E 주석 위치와 달리 콘의 경계 쪽으로 heatmap/국소 가중 영역이 이동했고 D로 판정했습니다.
이 사례는 지정 키포인트 대신 인접 구조를 읽을 가능성을 보여줍니다. 특정 원인을 확정하는 인과 실험은 아닙니다.</p>
<p>초기 오차≤3px인 수동 보임376점의 E 오판은181→38점입니다. 정사각형111→3, 목재70→35로 효과가 다릅니다.
위치 좌표를 새로 개선한 결과가 아니라 고정 위치 정보로 가림 판정을 바꾼 결과입니다.</p>
<p><a href="evaluation_v1/INDEX_KO.html">고정16셀·유형별 전체 혼동행렬과 실제 crop</a> · <a href="evaluation_v1/WOOD_FIXED8_CELLS_KO.png">목재8셀</a> · <a href="evaluation_v1/SQUARE_FIXED8_CELLS_KO.png">정사각형8셀</a></p>
<h2>재현과 한계</h2><p>실사 결과를 본 뒤 추가 학습·임계값 변경·PnP는 하지 않았습니다. E188은 반복 source family와 적은 물리 장면을 포함하며 native6/7의 합성 DEV 외부 가림은 없습니다.
가린 점에는 원래 위치 heatmap 정답을 주지 않았고 softmax는 항상 봉우리를 만듭니다. 국소 특징만으로 지정 점의 직접 보임을 보장하지 못합니다.</p>
<p>G996은 기존 E 주석과 사용자 해석의 불일치로 재검토 대상입니다. G981/G200의 작은3px 경계 초과를 큰 실패로 단정하지 않았습니다. 원 주석은 유지했습니다.</p>
<p><a href="PROTOCOL_BEFORE_FIT.json">학습 전 고정 조건</a> · <a href="NEW_CONFIG_FROZEN_BEFORE_CURRENT.json">실사 전 동결</a> · <a href="CURRENT_PREDICTIONS_FROZEN.json">주석 평가 전 예측 동결</a> · <a href="FINAL_RESULT_SUMMARY.json">요약 JSON</a> · <a href="RUN_GUIDE_KO.md">재현 안내</a> · <a href="VERIFY_METRICS.json">96개 집계 독립 재현</a></p></html>'''
    save('ROOT_READOUT_KO.html',page.replace('TABLE',''.join(table)))
    print(json.dumps({'completed':True,'UV_reuse_maxdiff':maxdiff,'readout':bind(W/'ROOT_READOUT_KO.html'),
                      'figure':bind(W/'LOCAL_POOL_SUCCESS_FAILURE_KO.png')},ensure_ascii=False))


if __name__ == '__main__':
    main()
