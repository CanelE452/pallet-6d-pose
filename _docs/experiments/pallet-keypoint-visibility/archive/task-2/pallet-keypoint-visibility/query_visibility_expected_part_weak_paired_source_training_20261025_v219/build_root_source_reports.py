"""Render saved SOURCE results and authentic saved crops; never run a model."""
from pathlib import Path
import json, hashlib, html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

V = Path(__file__).absolute().parent
P = V.parent
W = P / 'query_visibility_public_train_intended_part_query_targets_20261025_v218'
C = 'STRONG_ONLY_CONTROL'
T = 'STRONG_PLUS_EXPECTED_PART_WEAK'
def binding(p):
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))
def read(p):
    return json.loads(p.read_text())
def new(p, body):
    assert not p.exists(), str(p)
    p.write_text(body)
def newj(p, body):
    new(p, json.dumps(body, ensure_ascii=False, indent=2) + '\n')

fit = read(V / 'paired_source_fit_v1/COMPLETE.json')
metrics = read(V / 'saved_source_result_readout_v1/SOURCE_METRICS_AND_PAIRED.json')
assert fit['completed'] and fit['PASS'] and fit['steps_each'] == 4000
assert metrics['no_protected_evaluation'] and metrics['threshold_raw_logit'] == 0
rows = read(V / 'saved_source_result_readout_v1/ALL57_PAIRED_SOURCE_ROWS.json')['rows57']
byid = {x['row_id']: x for x in rows}
canonical_path = W / 'joined_weak_intended_part_targets_v1/ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl'
assert binding(canonical_path)['sha256'] == '52ac5bb8d0fb19a1329b9bdb7f7b56964fed954a2a4d50489e3dcb7d598d139d'
canonical = {x['row_id']: x for x in map(json.loads, canonical_path.read_text().splitlines())}
ids = ['F05_C000000_S0_N1', 'F05_C000000_S2_N1',
       'F08_C000003_S0_N1', 'F08_C000003_S0_N4',
       'F08_C000003_S0_N5', 'F05_C000000_S0_N5',
       'F10_C000000_S0_N5', 'F05_C000000_S0_N4']
notes = {
 ids[0]: 'DEV regression: visible part sits left/below query; C65 +0.212 is near zero.',
 ids[1]: 'Same nominal part with +0.25L vertical query shift; recovered, not independent.',
 ids[2]: 'Visible deck against dark background; false E becomes D.',
 ids[3]: 'Visible thin deck edge against bright background; false E becomes D.',
 ids[4]: 'Both miss weak E: crossing wood remains plausible target-looking wood.',
 ids[5]: 'Both detect weak E: blue foreground object is visually distinct.',
 ids[6]: 'Both detect weak E; crop alone gives little target identity information.',
 ids[7]: 'Both preserve weak D; small wood patch against dark background.'}
fig = plt.figure(figsize=(15, 11))
axes = [fig.add_axes([.02 + .245 * col, .53 if row == 0 else .055, .235, .32])
        for row in range(2) for col in range(4)]
for ax, rid in zip(axes, ids):
    x = canonical[rid]
    r = byid[rid]
    rec = x['generated_input']['RGB96']
    path = Path(rec['path'])
    assert W / 'public_source_gt_query_jitter_v1' == path.parent
    assert binding(path) == rec
    with Image.open(path) as im:
        assert im.size == (96, 96)
        ax.imshow(im.convert('RGB'), interpolation='nearest')
    ax.plot([47.5], [47.5], '+', color='#00e5ff', markersize=11, markeredgewidth=1.5)
    for who, color in [('A', '#ffff00'), ('C', '#64ff64')]:
        a, b, c, d = x['support_checks']['evidence_boxes'][who]['transformed_roi96']
        ax.add_patch(Rectangle((a-.5, b-.5), c-a, d-b, fill=False, edgecolor=color, linewidth=1))
    y = 'DE'[r['target']]
    pc, pt = ('DE'[r['predictions'][arm]] for arm in [C, T])
    lc, lt = (r['raw_logits'][arm] for arm in [C, T])
    status = 'RECOVERED' if pc != y and pt == y else 'REGRESSED' if pc == y and pt != y else 'BOTH CORRECT' if pt == y else 'BOTH WRONG'
    ax.set_title(f'{rid}\nweak {y} | C64 {pc} ({lc:+.2f}) -> C65 {pt} ({lt:+.2f})\n{status}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Actual SOURCE_DEV RGB96 | cyan: query | yellow/green: weak evidence ROIs\nAll 7 nominal DEV cases + one correlated shifted case; overlays for diagnosis only', fontsize=13)
fig.savefig(V / 'SOURCE_DEV_ACTUAL_CROPS.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for i, key in enumerate(['SOURCE_DEV/ALL_STATES', 'SOURCE_DEV/NOMINAL_S0']):
    group = metrics['weak'][key]
    for j, arm in enumerate([C, T]):
        ax = axes[i, j]
        cm = [z[:2] for z in group['models'][arm]['metrics']['CM_D_E_by_D_E_U']]
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=9)
        for yy in range(2):
            for xx in range(2): ax.text(xx, yy, str(cm[yy][xx]), ha='center', va='center', fontsize=22, color='white' if cm[yy][xx] > 4 else 'black')
        ax.set_xticks([0,1], ['pred D', 'pred E'])
        ax.set_yticks([0,1], ['true D', 'true E'])
        ax.set_title(f'{"C64 baseline" if j == 0 else "C65 + weak real"}\n{key}: n={group["rows"]}, components={group["unique_components"]}')
fig.suptitle('Fixed logit threshold 0; forced binary U=0\nSOURCE weak labels, previously inspected, small correlated sample')
fig.tight_layout(rect=(0,0,1,.92)); fig.savefig(V/'SOURCE_DEV_CONFUSIONS.png', dpi=150); plt.close(fig)

visual = dict(completed=True, PASS=True, scope='saved SOURCE_DEV only',
 root_actual_view_image_tool=True, images_viewed=8,
 figure= binding(V/'SOURCE_DEV_ACTUAL_CROPS.png'),
 source_images=[dict(row_id=rid, RGB96=canonical[rid]['generated_input']['RGB96'], saved_prediction=byid[rid], observation=notes[rid]) for rid in ids],
 selection='all seven nominal SOURCE_DEV examples plus shifted variant of the one regression',
 label_authority='assistant expected finite-part weak; not human physical native-point visibility gold',
 interpretation='Appearance/part-identity confusion is a hypothesis from these crops, not a causal ablation. Crop-only identity remains unresolved for some weak E examples.',
 original_initializer_coordinates_changed=False, new_image_model_inference=False,
 protected_evaluation_access=0, browser_render_QA=False)
newj(V/'ROOT_ACTUAL_SOURCE_CROP_VISUAL_REVIEW.json', visual)

def pct(v): return '—' if v is None else f'{100*v:.2f}%'
lines = []
for key in ['SOURCE_FIT/ALL_STATES', 'SOURCE_DEV/ALL_STATES', 'SOURCE_DEV/NOMINAL_S0']:
    for arm in [C,T]:
        g=metrics['weak'][key]; m=g['models'][arm]['metrics']; d,e=m['classes']
        lines.append(f'| {key} ({g["rows"]}) | {"C64" if arm==C else "C65"} | {m["CM_D_E_by_D_E_U"]} | {pct(d["precision"])} / {pct(d["recall"])} | {pct(e["precision"])} / {pct(e["recall"])} | 0% |')
table='\n'.join(lines)
report='''# V219: 공개 실사 약한 부재 감독을 추가한 실제 학습 결과

공개 실사의 작은 내부 검증에서 보이는 부재를 잘못 가림으로 분류하는 오류가 줄었습니다. 외부 가림을 놓치는 오류는 그대로입니다. 현재 평가 223장이나 v61·기존 모델 대비 개선은 이번 단계에서 측정하지 않았습니다. 모델 채택과 문제 해결을 선언하지 않습니다.

## 실제로 바꾼 것

C64는 기존 SOURCE 보조 손실과 합성의 지점 D/E 손실로 학습했습니다. C65에는 공개 실사에서 두 관찰자가 동의한 ‘기대하는 유한한 부재가 보이는가/다른 물체에 가려지는가’ 약한 손실을 0.25배 추가했습니다. 둘 다 새 초기화, 같은 118,359개 전체 파라미터, 같은 초기 가중치와 기존·합성 지점 추출 순서, 고정 4,000회 학습입니다. 32회씩의 실행 시간 시험은 폐기했고 본 학습을 초기 상태에서 시작했습니다. 학습 횟수는 실행 시간으로만 결정했고 출력이나 검증 성능으로 조정하지 않았습니다. 학습 완료까지 약 328초였습니다.

입력은 RGB96·유효 마스크·로컬 좌표와 native 순서 및 나머지 query의 상대 위치 32값입니다. SAM은 사용하지 않았습니다. 합성 및 원래 보조 데이터의 q/L은 저장된 초기 추론값을 유지했습니다. 추가된 공개 실사 약한 표본은 GT에서 만든 별도 학습용 q8과 고정 이동을 사용합니다. 이 GT 유래 학습용 query와 실제 초기 추론 query 사이의 전이 문제는 남아 있습니다. 추가 GT pose/reference/ROI/mask를 모델 입력 채널로 넣지는 않았습니다.

## 데이터와 라벨

- SOURCE 학습용 캐시: 합성 7,040행, 기존 공개 실사 보조 1,208행, 공개 변형 보조 3,816행. 서로 다른 보조 과제의 행이므로 합계가 독립된 D/E 라벨 수는 아닙니다.
- 합성 7,040행 중 지점 손실에 들어간 것은 6,096행(D 5,853/E 243)입니다. 나머지 944행은 해당 지점 손실에서 제외했고 E로 바꾸지 않았습니다.
- 공개 실사 추가 후보 360행 중 부재 약한 D/E가 확정된 57행만 사용 가능했습니다. FIT 45행(D 39/E 6), DEV 12행(D 9/E 3)입니다. 57행은 이동 변형이 상관된 27개 명목 지점, 11개 frame/component에서 나왔습니다.
- DEV 12행은 3개 component입니다. 이동을 제외한 DEV 원형은 7행(D 4/E 3)뿐입니다. 원래 DEV는 4개 component/32개 native slot이며, 그중 3개 component에서만 약한 D/E 지원 표본이 남았습니다.
- 새 DEV component에서 나온 기존 공개 보조 32행은 모든 학습·정규화 풀에서 제외했습니다. 이전 HOLD 합성 9,216행·공개 240행과 새 DEV의 수치는 추출하거나 모델에 입력하지 않았습니다. 혼합 압축 파일의 전체 해시·헤더 검사·제외 구간의 바이트 통과는 있었으므로 ‘제외 파일의 바이트도 전혀 읽지 않았다’고 표현하지 않습니다.
- 나머지 303행은 화면 밖 150, 다른 부재/기하 대응 미해결 72, UNKNOWN 81입니다. 이들은 정답이 미해결된 행이며 모델 보류(U)와 다릅니다.

D는 기대 부재가 보임, E는 그 부재가 외부 물체에 가려짐을 뜻하는 약한 라벨입니다. 두 AI 관찰자의 합의이며 사람이 확정한 정확한 물리 키포인트 gold가 아닙니다. 부재가 crop 안에 보인다는 것과 투영된 키포인트 한 점이 직접 보인다는 것은 다를 수 있습니다. 임의의 목재가 있다고 D로 인정하지 않았지만 정확한 native anatomy/instance 대응이 모두 해결된 것은 아닙니다. 자기 가림은 상위 기하 필터가 담당한다는 원래 범위를 유지했습니다. 이전 9개 COVER/SHAM 부재 변형은 이번 학습에 넣지 않았습니다.

## 고정 문턱에서의 저장 출력

문턱은 학습 전 고정한 raw logit 0입니다. 혼동행렬은 행이 실제 D/E, 열이 예측 D/E/U입니다. 모든 출력은 유한한 이진 예측이므로 U=0입니다. 이는 불확실성이나 입력 U 문제가 해결됐다는 뜻이 아닙니다.

| 표본 | 모델 | 혼동행렬 D/E → D/E/U | D precision / recall | E precision / recall | 보류율 |
|---|---|---|---|---|---|
'''+table+'''

DEV에서 D recall은 5/9(55.56%) → 8/9(88.89%)로, E precision은 2/6(33.33%) → 2/3(66.67%)로 좋아졌습니다. E recall은 2/3(66.67%)로 그대로입니다. 같은 12행에서 4행을 회복하고 1행을 새로 틀려 순증 3행입니다. 이동을 뺀 원형 7행에서는 2행 회복·1행 악화로 순증 1행입니다. 상관된 이동 변형을 독립 성공으로 세어 일반화하지 않습니다.

FIT 45행에서 C65는 전부 맞혔지만 이는 학습 자료 적합도입니다. 합성 학습 지점 6,096행도 두 모델 모두 전부 맞혔고, 독립 검증 성능이 아닙니다. DEV는 이전에 관찰한 SOURCE이며 새 blind 평가가 아닙니다. 이 작은 표본만으로 실제 운용 성능이나 우월성을 확정할 수 없습니다.

![SOURCE DEV confusion](SOURCE_DEV_CONFUSIONS.png)

## 실제 crop에서 남은 문제

아래 그림은 DEV 원형 7개를 모두 포함하고, 새로 틀린 동일 부재의 이동 변형 1개를 추가했습니다. 청록 십자는 query, 노랑/초록 상자는 기존 두 관찰자의 부재 근거 영역입니다. 이 표시는 확인용으로만 그렸고 학습 RGB에는 없습니다.

![실제 SOURCE DEV crop과 저장 예측](SOURCE_DEV_ACTUAL_CROPS.png)

- F08 N1/N4: 어두운 바닥 또는 밝은 배경 앞에 드러난 목재 경계를 C64는 가림으로 분류했고 C65는 보임으로 회복했습니다.
- F08 N5: 다른 목재가 앞을 가로질러도 두 모델 모두 보임으로 분류했습니다. ‘목재가 보임’과 ‘기대한 대상 부재가 보임’을 구별하지 못하는 후보 사례입니다. crop만으로 대상 부재의 정확한 물리 대응을 증명할 수는 없습니다.
- F05 N1: 기대 부재 근거가 query 왼쪽 아래에 있고 중앙에는 다른 구조가 있습니다. C65는 원형을 새로 가림으로 오판했고(logit +0.212), 같은 부재의 수직 이동 표본은 회복했습니다. crop 위치와 문맥에 따른 불안정성이 남았습니다. 이 두 행은 독립된 사건이 아닙니다.
- F05 N5의 파란 전경 물체는 두 모델 모두 가림으로 분류했습니다. F10의 흐린 회색 crop도 둘 다 E이지만, 이 crop 단독으로 대상 정체를 확인하기는 어렵습니다.

위 관찰은 외관·부재 식별 혼동이라는 가설을 지지하는 예시이지 원인을 확정한 ablation은 아닙니다. 물리 키포인트 한 점의 D/E와 유한 부재 약한 목표의 차이, GT 유래 학습 q와 초기 추론 q의 차이, 적은 외부 가림 표본 및 native 4/5에 몰린 E 지원은 여전히 제약입니다. native 2/6/7에 약한 D/E 지원은 없고 BOTTOM_RUNNER 표본은 소수 D뿐입니다.

## 검증과 다음 판단

독립 검사는 두 모델 8,000행 로그의 공통 추출·해시·마스크와 point/weak 난수 추출, 118,359개 파라미터·AdamW 4,000 step을 확인했습니다. 저장된 weak57 출력을 CPU batch32로 재생했고 최대 logit 차이는 C64 3.624e-5/C65 3.052e-5, 모든 D/E 부호가 같았습니다. old192의 클래스 조건부 추출 확률 자체는 기존 target NPZ를 다시 열지 않아 독립 재구성하지 않았고, 합성 RGB forward 전체를 재생하지는 않았습니다.

현재 평가 223장·기존 보호 평가·V112에 새로 접근하지 않았고 PnP도 실행하지 않았습니다. 기준 모델과 추가 감독 모델을 둘 다 보존하며 채택 모델은 없습니다. 현재 실사 비교의 최신 완료 기록은 V213으로 유지합니다. 다음에는 고정된 두 모델을 원래 query/동일 평가 코호트에서 v61·기존 가림 모델과 함께 비교해야 실제 전이를 판단할 수 있습니다. 그 결과로 이 단계의 데이터·문턱·체크포인트를 되돌려 고르지 않습니다.

학습 코드, 데이터 lineage, 저장 출력, 로그, 두 SOURCE 모델, 표·이미지 및 검증 기록을 이번 main 게시에 포함합니다. 원본/학습 입력 NPZ는 게시하지 않습니다.

- [57행 전체 실제 crop·예측 표](saved_source_result_readout_v1/PORTABLE_SOURCE_READOUT_KO.html)
- [전체 precision/recall·혼동행렬 CSV](saved_source_result_readout_v1/SOURCE_CONFUSIONS_PRECISION_RECALL.csv)
- [기계 판독 가능한 전체 지표](saved_source_result_readout_v1/SOURCE_METRICS_AND_PAIRED.json)
- [실제 학습 완료 기록](paired_source_fit_v1/COMPLETE.json)
- [독립 체크포인트·출력 재생 QA](independent_actual_pair_QA_a/actual_review_v1/REVIEW.json)
- [이전 공개 SOURCE crop·라벨 정의](../query_visibility_public_train_intended_part_query_targets_20261025_v218/PORTABLE_REPORT_KO.md)
'''
new(V/'ROOT_REPORT_KO.md', report)
new(V/'PORTABLE_REPORT_KO.md', report)
cards=[]
for rid in ids:
    row=byid[rid]; target='DE'[row['target']]
    img='../'+W.name+'/public_source_gt_query_jitter_v1/'+rid+'_RGB96.png'
    cards.append(f'<article><h3>{rid}</h3><img src="{img}" alt="actual {rid}"><p>weak {target} · C64 {"DE"[row["predictions"][C]]} → C65 {"DE"[row["predictions"][T]]}</p><p>{html.escape(notes[rid])}</p></article>')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V219 실제 SOURCE 학습 결과</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#17212e}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd2d9;padding:8px}img.figure{width:100%}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}article{border:1px solid #ccc;padding:12px}article img{width:192px;image-rendering:pixelated}h3{font-size:15px}</style><h1>V219: 보이는 부재 오삭제는 감소, 가림 누락은 남음</h1><p>공개 실사의 내부 DEV 12행(3 component)에서 D recall 55.56% → 88.89%, E precision 33.33% → 66.67%, E recall 66.67%로 동일합니다. 4행 회복·1행 악화. 원형 7행에서는 순증 1행입니다.</p><p><b>현재 평가 223장·v61·기존 모델 대비 개선은 이번 단계에서 측정하지 않았습니다.</b> 약한 부재 라벨, GT 유래 학습 query, 이전에 관찰한 작은 SOURCE 검증 표본입니다. 새 blind 평가·정확한 물리 키포인트 gold가 아닙니다.</p><p><a href="PORTABLE_REPORT_KO.md">데이터·목표·한계 전체 보고서</a> · <a href="saved_source_result_readout_v1/PORTABLE_SOURCE_READOUT_KO.html">57행 실제 crop 전체 표</a> · <a href="saved_source_result_readout_v1/SOURCE_CONFUSIONS_PRECISION_RECALL.csv">precision/recall CSV</a></p><img class="figure" src="SOURCE_DEV_CONFUSIONS.png" alt="DEV confusion matrices"><h2>잘되는 사례와 남은 실패</h2><p>아래 원본 crop은 진단 표시 없이 저장된 RGB96 그대로입니다. 전체 DEV 원형 7개와 상관된 이동 변형 1개를 표시합니다. 도식의 청록 query·노랑/초록 근거 상자는 학습 입력에 없습니다.</p><div class="cards">'''+''.join(cards)+'''</div><img class="figure" src="SOURCE_DEV_ACTUAL_CROPS.png" alt="Actual crops with diagnostic query and evidence ROIs"><p>겹친 목재를 기대 부재로 오인하는 사례와 crop 이동에 따라 판단이 달라지는 사례가 남았습니다. 부재의 가시성과 물리 키포인트 한 점의 직접 가시성 차이는 미해결입니다.</p><p>같은 초기 상태·같은 기존/합성 지점 추출·각 4,000회, 전체 118,359개 파라미터를 학습했습니다. 두 최종 모델을 모두 동결한 뒤에 DEV 출력을 만들었습니다. 두 모델을 보존했고 채택 모델은 없습니다. PnP 없음, 보호 평가 접근 없음.</p></html>'''
new(V/'ROOT_READOUT_KO.html', page)
result=dict(completed=True,PASS=True,phase='V219',role='V219_ROOT_SCIENTIFIC_REPORT',source_only=True,
 source_training_updates_each=4000,source_training_updates_total=8000,source_probe_disposable_updates=64,source_model_parameters_each=118359,
 weak_known57=57,weak_FIT45=45,weak_DEV12=12,weak_DEV_components=3,weak_DEV_nominal7=7,
 paired_fit_complete=binding(V/'paired_source_fit_v1/COMPLETE.json'),
 saved_source_reduction=binding(V/'saved_source_result_readout_v1/COMPLETE.json'),
 source_metrics=binding(V/'saved_source_result_readout_v1/SOURCE_METRICS_AND_PAIRED.json'),
 report_bindings=dict(root_markdown=binding(V/'ROOT_REPORT_KO.md'),portable_markdown=binding(V/'PORTABLE_REPORT_KO.md'),root_html=binding(V/'ROOT_READOUT_KO.html')),
 root_visual_review=binding(V/'ROOT_ACTUAL_SOURCE_CROP_VISUAL_REVIEW.json'),
 independent_actual_pair_QA=binding(V/'independent_actual_pair_QA_a/actual_review_v1/REVIEW.json'),
 weak_DEV=metrics['weak']['SOURCE_DEV/ALL_STATES'],weak_DEV_nominal=metrics['weak']['SOURCE_DEV/NOMINAL_S0'],
 labels='ASSISTANT_EXPECTED_FINITE_PART_WEAK_NOT_PHYSICAL_NATIVE_POINT_GOLD',
 GT_derived_training_query=True,protected_evaluation_access=0,new_private_evaluation=False,
 latest_completed_real_benchmark='V213',selected_arm=None,adopted=False,fully_solved=False,
 scientific_claim='SOURCE_DEV visible-part false E reduced; E recall unchanged. Transfer to original initializer and current223 remains untested this phase.',
 browser_render_QA=False)
newj(V/'ROOT_RESULT.json', result)
print(json.dumps(dict(completed=True, reports=4, figures=2, result=binding(V/'ROOT_RESULT.json')),ensure_ascii=False))
