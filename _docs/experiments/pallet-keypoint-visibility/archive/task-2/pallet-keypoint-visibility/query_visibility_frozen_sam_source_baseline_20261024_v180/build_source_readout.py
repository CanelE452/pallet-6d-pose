"""Read completed source-only runs and draw a scientific diagnostic, never fit."""
from pathlib import Path
import collections
import datetime
import hashlib
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / 'saved_sam_allTRAIN4096_evaluation_c_v3'
PRED = ROOT / 'frozen_sam_allTRAIN4096_predictions_v2'
END = ROOT / 'source_end_supervision_train4096_v1'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def bind(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def ratio(a, b):
    return None if not b else a / b


def pct(value):
    return '—' if value is None else f'{value * 100:.1f}%'


evaluation = read(EVAL / 'COMPLETE.json')
prediction = read(PRED / 'COMPLETE.json')
end = read(END / 'COMPLETE.json')
assert all(x['completed'] and x['PASS'] for x in (evaluation, prediction, end))
metrics = read(EVAL / 'FIXED_CONFUSIONS_AND_STRATA.json')
images = lines(EVAL / 'ALL_IMAGES_TARGET_RASTER_OVERLAP.jsonl')
all_block = next(g for g in metrics['groups'] if g['group'] == 'ALL')
point = all_block['point']['bilinear']
cm = np.asarray(point['soft_confusion'], np.float64)
assert cm.shape == (5, 3)
groups = []
for material, aspect in sorted({(r['factors']['material'], r['factors']['aspect']) for r in images}):
    rows = [r for r in images if (r['factors']['material'], r['factors']['aspect']) == (material, aspect)]
    usable = [r['metrics'] for r in rows if r['metrics']['IoU'] is not None]
    counts = {k: sum(r[k] for r in usable) for k in ('TP', 'FP', 'FN', 'TN', 'FP_on_FOREIGN', 'FP_on_BG')}
    groups.append(dict(material=material, aspect=aspect, images=len(rows), usable=len(usable),
                       macro_IoU=float(np.mean([r['IoU'] for r in usable])),
                       micro_IoU=ratio(counts['TP'], counts['TP'] + counts['FP'] + counts['FN']), **counts))
whole_counts = {k: sum(r['metrics'][k] for r in images if r['metrics']['IoU'] is not None)
                for k in ('TP', 'FP', 'FN', 'TN', 'FP_on_FOREIGN', 'FP_on_BG')}
whole = dict(images=512, usable=506, prompt_missing_images=6,
             all_registered_target_pixels=sum(r['metrics']['target_pixels'] for r in images),
             unavailable_target_pixels=sum(r['metrics']['target_pixels'] for r in images if r['metrics']['IoU'] is None),
             macro_IoU=float(np.mean([r['metrics']['IoU'] for r in images if r['metrics']['IoU'] is not None])),
             micro_IoU=ratio(whole_counts['TP'], whole_counts['TP'] + whole_counts['FP'] + whole_counts['FN']), **whole_counts)
end_all = read(END / 'SUMMARY.json')['groups'][0]
summary = dict(
    schema='V180_SOURCE_ONLY_SAM_AND_LOCAL_SUPPORT_READOUT_v1',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    completed=True, whole_goal_solved=False, selected_model=None, adopted=False,
    source_TRAIN_rows=4096, source_TRAIN_images=512, original_input_U=80, missing_original_q=48,
    source_only_no_private_or_public_held_development=True, new_training_updates=0,
    original_q_L_native_and_PnP_unchanged=True,
    model='Official SAM1 ViT-B, original q8 box only, single mask logits>0, frozen FP32',
    whole_image_visible_target_mask=whole, material_aspect=groups,
    point_bilinear_soft_diagnostic=point,
    foreign_nonmask_recall=ratio(cm[1, 1], cm[1].sum()),
    BG_nonmask_recall=ratio(cm[2, 1], cm[2].sum()),
    source_end_support=end_all,
    limits=[
        'Synthetic TRAIN diagnostics are neither real transfer nor independent generalization evidence.',
        'A class-agnostic SAM box mask has no selected-pallet/native identity certificate.',
        'Nonmask is not external occlusion; foreign and background are kept separate.',
        'Literal q owner, whole-image target mask, crop-any-pallet and fixed native-end support are different questions.',
        'New end supports are non-exclusive and connect source geometry to the fixed crop; they do not relabel original source native D/E/S or private human annotations.',
        'Whole image IoU can hide very small foreign covers; the fixed16 foreign union contains58 pixels.',
        'Predicted mask quality/logits are uncalibrated; no threshold was selected.',
        'Source supervision training_admission remains false; no new head was fitted in V180.',
    ],
    evidence=dict(prediction=bind(PRED / 'COMPLETE.json'), evaluation=bind(EVAL / 'COMPLETE.json'),
                  source_end=bind(END / 'COMPLETE.json'),
                  SAM_saved_audit=bind(ROOT / 'independent_actual_sam_evaluation_review_b/REVIEW.json'),
                  SAM_actual_visual_QA=bind(EVAL / 'ACTUAL_VISUAL_QA_A.json'),
                  end_audit=bind(ROOT / 'independent_actual_end_support_review_c/REVIEW.json'),
                  raw_contrast=bind(ROOT / 'source_end_spatial_visual128_v1/FIXED16_CONFIRMED_END_FOREIGN_RAW_RGB_CONTRAST.json')),
)
plot_dir = ROOT / 'scientific_readout_v1'
plot_dir.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.7), gridspec_kw={'width_ratios': [1.12, 1.05, 1.15]})
normalized = np.divide(cm[:3], cm[:3].sum(1, keepdims=True), out=np.zeros((3, 3)), where=cm[:3].sum(1, keepdims=True) != 0)
axes[0].imshow(normalized, vmin=0, vmax=1, cmap='Blues')
axes[0].set_xticks(range(3), ['mask', 'nonmask', 'U'])
axes[0].set_yticks(range(3), ['target', 'foreign', 'background'])
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f'{normalized[i,j]*100:.1f}%', ha='center', va='center',
                     color='white' if normalized[i,j] > .55 else 'black', fontsize=12)
axes[0].set_title('At original q: bilinear logits > 0\nRow-normalized source soft owner mass', fontsize=11)
axes[0].set_xlabel('Mask proposal response; not native D/E/S')
names = [('Wood' if g['material'] == 'PROCEDURAL_WOOD' else 'Polymer') + '\n' +
         ('Rect.' if g['aspect'] == 'RECTANGULAR' else 'Square') for g in groups]
values = [g['macro_IoU'] for g in groups]
axes[1].bar(range(4), values, color='#427b9c')
axes[1].set_xticks(range(4), names)
axes[1].set_ylim(0, 1.20)
axes[1].set_yticks(np.linspace(0, 1, 6))
axes[1].set_ylabel('Mean image IoU (available images only)')
for i, (value, group) in enumerate(zip(values, groups)):
    axes[1].text(i, value + .025, f'{value:.3f}\n{group["usable"]}/{group["images"]}', ha='center', fontsize=10)
axes[1].set_title('Whole visible selected-target mask\nAmbiguity excluded; 6 prompt-U retained', fontsize=11)
left = np.zeros(2)
for label, key, color in [('YES', '1', '#289b65'), ('NO', '0', '#6287a2'), ('UNKNOWN', '-1', '#a5a8ad')]:
    count = np.asarray([end_all['visible_counts'][key], end_all['foreign_counts'][key]])
    axes[2].barh([1, 0], count, left=left, color=color, label=label)
    for y, n, offset in zip([1, 0], count, left):
        if n > 180:
            axes[2].text(offset + n/2, y, str(n), ha='center', va='center', color='white', fontsize=10)
    left += count
axes[2].set_yticks([1, 0], ['end visible\nsupport', 'foreign-front\nsupport'])
axes[2].set_xlim(0, 4096)
axes[2].set_xlabel('Original TRAIN queries; supports are non-exclusive')
axes[2].legend(loc='lower center', bbox_to_anchor=(.5, -.35), ncol=3, fontsize=9)
axes[2].set_title('New geometry-derived supervision\nCounts are labels, not model accuracy', fontsize=11)
fig.suptitle('V180 | Fixed initial inputs | SYNTHETIC TRAIN ONLY | full task remains unsolved', fontsize=14, y=.99)
fig.tight_layout(rect=(0, .07, 1, .93))
plot = plot_dir / 'SOURCE_SAM_AND_END_SUPPORT_V2.png'
fig.savefig(plot, dpi=160, bbox_inches='tight')
fig.savefig(plot.with_suffix('.pdf'), bbox_inches='tight')
plt.close(fig)
summary['plot'] = bind(plot)
(ROOT / 'ROOT_RESULTS_KO.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
body = ['<!doctype html><meta charset="utf-8"><title>V180 합성 TRAIN SAM·영역 감독 진단</title>',
        '<style>body{font:17px/1.65 system-ui;max-width:1250px;margin:36px auto;padding:0 18px;color:#172b3b}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #d7e0e6;text-align:left}.note{background:#f0f5f8;padding:18px}a{color:#096b9b}</style>',
        '<h1>V180: 동결 SAM 실행과 crop 주변 보임 감독</h1>',
        '<p class="note"><b>합성 TRAIN에서 실제 실행과 감독 생성까지 완료했습니다. 사용자 데이터 성능 개선·모델 채택·전체 해결은 아직 입증하지 않았습니다.</b> 초기 q/L/native8과 PnP는 그대로이며 이번 단계의 새 학습은 0회입니다.</p>',
        '<p>공식 SAM ViT-B에 초기 8점의 min/max 박스만 넣었습니다. GT 박스·양성 키포인트·부위 정답을 입력하지 않았습니다. 512장 중 초기 8점이 없는 6장은 보류, 506장 추론과 첫 장 반복을 완료했고 원 U80을 보존했습니다. 마스크 threshold는 공식 0으로 고정했습니다.</p>',
        '<img src="scientific_readout_v1/SOURCE_SAM_AND_END_SUPPORT_V2.png">',
        '<h2>대상 마스크와 원 q의 소속은 별도 진단</h2>',
        f'<p>전체 대상 마스크 평균 IoU: <b>{whole["macro_IoU"]:.4f}</b> (가용 506장). 원 q의 팔레트 소속 soft 재현율: <b>{pct(point["TARGET_recall_including_prediction_U"])}</b>, precision: <b>{pct(point["TARGET_precision_known_truth"])}</b>. 외부 물체 소속 질량에 nonmask 응답: <b>{pct(summary["foreign_nonmask_recall"])}</b>, 배경 소속 질량에는 <b>{pct(summary["BG_nonmask_recall"])}</b>.</p>',
        f'<p>원 4,096행 중 q 소속 측정은 {all_block["point_truth_measured_N"]:,}행, q 결측은 {all_block["point_truth_missing_N"]}행입니다. 측정 행의 INVALID/예측 U 질량 {point["INVALID_SUPPORT_mass"]:.0f}은 별도로 보존했으며, 결측 48행을 합친 원 입력 U는 총 80행입니다.</p>',
        '<p>모두 합성 TRAIN의 저장 raster 기준이며 실사 전이 정확도가 아닙니다. nonmask에는 배경과 다른 물체가 함께 들어가므로 이를 외부 가림 판정으로 바꾸지 않았습니다. 전체 면적 IoU가 높아도 작은 가림 영역을 놓칠 수 있습니다.</p>',
        '<table><tr><th>합성 그룹</th><th>가용/전체 영상</th><th>평균 마스크 IoU</th><th>Foreign 위 오포함 pixel</th></tr>']
for g in groups:
    body.append(f'<tr><td>{html.escape(g["material"] + " / " + g["aspect"])}</td><td>{g["usable"]}/{g["images"]}</td><td>{g["macro_IoU"]:.4f}</td><td>{g["FP_on_FOREIGN"]}</td></tr>')
body += ['</table><h2>중심 한 픽셀과 구분한 새 감독</h2>',
         f'<p>원 crop 안의 고정된 작은 팔레트 끝 영역에서 보임 YES/NO/UNKNOWN은 {end_all["visible_counts"]["1"]}/{end_all["visible_counts"]["0"]}/{end_all["visible_counts"]["-1"]}입니다. 해당 영역 앞의 외부 차폐는 {end_all["foreign_counts"]["1"]}/{end_all["foreign_counts"]["0"]}/{end_all["foreign_counts"]["-1"]}입니다. 두 조건은 비배타적이며 부분 가림에서는 함께 YES일 수 있습니다. 자기 가림·대응 불명·원 입력 결측은 따로 남겼고 기존 native D/E/S를 교체하지 않았습니다.</p>',
         '<p>고정 16장에는 확인된 foreign-front 원 픽셀이 합계 58개뿐입니다. 그중 BEFORE와 RGB가 완전히 같은 픽셀은 1개입니다. 같은 재질 가림의 대비가 작다는 관찰은 있으나, 영상이 동일하거나 추론 불가능하다는 증거는 아닙니다. 새 감독의 training_admission은 false이며 새 head fit은 아직 수행하지 않았습니다.</p>',
         '<h2>고정 네 그룹의 COVER 사례</h2><img src="scientific_readout_v1/FIXED4_COVER_INPUT_SAM_TARGET_END.png">',
         '<p>각 그룹의 원 COVER 대상 슬롯을 모두 표시했습니다. G40에서는 외부 물체 소속인 원 q를 SAM이 대상으로 포함합니다. G521/553에서는 배경색과 비슷한 가림물을 마스크에서 제외합니다. G8은 원 q 자체는 대상에 속하지만 주변의 지정 끝 영역은 가려져 있어, 두 정답이 다를 수 있음을 보여줍니다. 이 네 장에서는 형상과 가림물 재질 조건도 함께 달라지므로 형상만의 효과로 해석하지 않습니다.</p>',
         '<p><a href="saved_sam_allTRAIN4096_evaluation_c_v3/index.html">고정 16장·128개 SAM/원 crop/대상 raster 대조</a> · <a href="source_end_spatial_visual128_v1/index.html">동일 128개의 부위 보임·차폐 감독 위치</a></p>',
         '<h2>실행 및 남은 범위</h2><p>첫 SAM 실행은 Torch 2.1의 strict deterministic cumsum guard에서 종료됐습니다. 공식 모델 코드는 그대로 두고 predict 구간의 guard만 완화했으며 CPU/CUDA ones-grid 일치·첫 영상 반복·가중치 불변을 확인했습니다. 저장 평가기의 행 병합/함수명 충돌 두 오류도 실패 기록을 보존한 뒤 수정했습니다. 입력·마스크·문턱·정답 기준을 고친 것은 아닙니다.</p>',
         '<p>사용자223장과 공개 보류45프레임의 원영상·특징·개별 주석을 이번 단계에서 추가로 열람하거나 적합에 사용하지 않았습니다. 기존 v178 실사 결과는 그대로입니다. 다음에는 이 영역별 감독의 실제 학습 효과와 공개 실사에서의 대상 식별·작은 외부 가림 구분을 검증해야 합니다. SAM 마스크 하나를 가림 모델로 채택한 상태는 아닙니다.</p>',
         '<p>PNG는 별도 실제 시각 QA 기록을 확인하십시오. 이 HTML은 파일 링크 및 소스 검사를 하며 브라우저 렌더 완료를 주장하지 않습니다.</p>']
(ROOT / 'ROOT_READOUT_KO.html').write_text('\n'.join(body), encoding='utf-8')
print(json.dumps(dict(results=bind(ROOT / 'ROOT_RESULTS_KO.json'), report=bind(ROOT / 'ROOT_READOUT_KO.html'), plot=bind(plot)), ensure_ascii=False))
