from pathlib import Path
import base64, datetime, hashlib, html, json

V = Path(__file__).resolve().parent
P = V.parent
E = V / 'source_evaluation/OWNERSHIP_FROZEN_COMPARISON'
R = json.loads((E / 'SOURCE_RESULTS.json').read_text())
A = json.loads((V / 'SOURCE_ADMISSION_DECISION.json').read_text())
F = json.loads((V / 'runs/OWNERSHIP_LOO_PAIR_V1_RUNTIME_V2/MODEL_FREEZE.json').read_text())
G = V / 'source_audit/FIXED_SOURCE_GALLERIES'
PALL = V / 'source_audit/PALLOX_E_DISAGREEMENT_REVIEW'
S = P / 'query_visibility_reference_semantic_20261023_v69'

def bind(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def dump(path, value):
    assert not path.exists(), f'Preserve existing artifact: {path}'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(str(x))+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'

def pct(x):
    return '—' if x is None else f'{100*x:.1f}%'

def image(path, caption):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

def link(path, title):
    import os
    return f'<a href="{html.escape(os.path.relpath(path, V))}">{html.escape(title)}</a>'

sources = ['SYNTH', 'ORIGINAL_YCB', 'CONTROLLED_YCB', 'PALLOX']
model_names = {
    'V61_FROZEN': 'v61 합성 pixel head',
    'V65_BASELINE': 'v65',
    'V66_CONTROLLED_PLUS_PALLOX': 'v66',
    'V67_VISIBLE_BCE1': 'v67',
    'MOBILE_SAM_RAW': 'MobileSAM',
    'SAM21_BASE_PLUS_FROZEN': 'SAM2.1',
    'V70_OWNERSHIP_LOO_PAIR_V1': 'v70 ownership head',
}
metrics = []
for src in sources:
    for model, name in model_names.items():
        d = R['models'][model]['per_source'][src]
        m = d['visible_binary_ALL_known_VEO']
        ee = d['known_E_to_binary_nonvisible']
        metrics.append([src, name, m['N'], f"{m['TP']}/{m['FP']}/{m['FN']}/{m['TN']}", pct(m['precision']), pct(m['recall']), '—' if not ee['N'] else f"{ee['binary_nonvisible']}/{ee['N']} ({pct(ee['recall'])})", pct(m['U']/m['N'])])

matrices = []
for src in sources:
    for model in ['V61_FROZEN', 'V70_OWNERSHIP_LOO_PAIR_V1']:
        c = R['models'][model]['per_source'][src]['class_known_confusion']
        matrices.append('<h4>'+html.escape(src+' · '+model_names[model])+'</h4>'+table(['정답 \\ 판정','보임','안 보임','보류 U'], [[lab]+row for lab,row in zip(c['GT_rows'],c['counts'])]))

cond_rows=[]
for src in sources:
    m=R['models'][R['primary']]['per_source'][src]['conditional_known_VE']
    cond_rows.append([src,m['N'],f"{m['TP']}/{m['FP']}/{m['FN']}/{m['TN']}",pct(m['precision']),pct(m['recall']),pct(m['nonvisible_recall']),f"{m['NLL']['mean']:.4f}"])

root_visual = {
    'schema': 'V70_ROOT_FIXED_SOURCE_RGB_REVIEW',
    'completed': True,
    'scope': 'Read actual preselected public RGB gallery PNGs and saved scores only; no relabel, refit, new forward, current data or PnP.',
    'observations_ko': [
        '고정 V 네 사례는 모두 source native5x5=V. v70은 pitcher label과 세제 병 label의 두 점을 안 보임으로 판정하고 drill/다른 pitcher 점은 보임으로 판정한다. 사례 선택은 V70 점수 이전에 고정됐다.',
        '고정 E 네 사례는 불투명 다른 물체가 q의 native5x5 전체를 덮는 개입. v70은 세 점을 안 보임으로 판정하지만 캔 뚜껑 위 cardboard 상자 사례는 pV .968로 보임이다. 이 네 사례는 SAM2.1 false-visible 집합으로 미리 고른 진단 사례이며 대표 성능 추정용이 아니다.',
        'Pallox 두 쌍을 실제 확인하면 selected A의 q는 별도 cardboard 위에 있고 paired B의 동일 q에는 목재 bin 판재가 있다. 고정 crop에서 경계보다 충분히 안쪽이다. v70은 A도 .685/.839로 보임이다. 이 두 사례는 teacher 경계만의 문제보다 실제 foreground ownership 실패와 더 일치한다.',
        'Pallox teacher는 제공 pose/depth/SAM 정합 조건부이며 RGB-depth native 정합 인증이나 사람 semantic gold가 아니다. 같은 촬영 family의 두 진단 사례를 독립 모집단 성능으로 해석하지 않는다.',
    ],
    'images': [bind(G/'FIXED_VISIBLE_SAVED_COMPARISON.png'),bind(G/'FIXED_EXTERNAL_SAVED_COMPARISON.png'),bind(PALL/'FIXED_TWO_PUBLIC_PALLOX_E_DISAGREEMENTS.png')],
    'mechanism_causal_ablation_performed': False,
    'current223_new_access': 0,
    'new_forward': 0,
    'new_PnP': 0,
}
dump(V/'source_audit/ROOT_FIXED_SOURCE_RGB_REVIEW.json',root_visual)

report = {
    'schema': 'V70_TARGET_OWNERSHIP_RESULTS_KO_V1',
    'completed': True,
    'experiment_completed': True,
    'full_user_goal_solved': False,
    'goal_status': 'active',
    'verdict': A['verdict'],
    'main_finding_ko': '조건부 V/E 구분은 개선했으나 O/UNKNOWN 범위를 포함한 사용 목표의 완성 모델이 아니다. 전체 source 보임 precision과 공개 목재 E recall이 사전 조건에 미달하므로 현재223장 새 평가는 실행하지 않았다.',
    'user_current_capture_provenance': bind(P/'query_visibility_controlled_real_20261022_v66/provenance/EVALUATION_FAMILY_PROVENANCE.json'),
    'current223_fit_threshold_checkpoint_selection': False,
    'current223_new_inference': 0,
    'new_PnP': 0,
    'training': {'parameters':F['parameters'],'optimizer_updates':F['optimizer_updates'],'best_update':F['best_update'],'threshold':.5,'checkpoint_sha256':F['checkpoint_sha256'],'selection':F['training_plan']['checkpoint_selection'],'source_macro_knownVE_NLL':F['best_source_macro_knownVE_NLL']},
    'source_population':{'queries':R['queries'],'targets':R['targets'],'native_valid':R['native_valid'],'native_invalid':R['native_invalid'],'exact_class_known':7247,'native_valid_class_UNKNOWN':1337},
    'source_results': bind(E/'SOURCE_RESULTS.json'),
    'source_admission':bind(V/'SOURCE_ADMISSION_DECISION.json'),
    'source_selection':bind(V/'SOURCE_SELECTION_SEAL.json'),
    'root_visual_review':bind(V/'source_audit/ROOT_FIXED_SOURCE_RGB_REVIEW.json'),
    'fixed_gallery':bind(G/'FIXED_SOURCE_GALLERIES.json'),
    'pallox_disagreement':bind(PALL/'PALLOX_E_DISAGREEMENT_REVIEW.json'),
    'cue_ownership_diagnosis':bind(S/'source_cue_reference_proposals/FROZEN_CUE_OWNERSHIP_SANITY.json'),
    'cue_interpretation_addendum':bind(S/'source_cue_reference_proposals/FROZEN_CUE_OWNERSHIP_INTERPRETATION_ADDENDUM.json'),
    'no_new_cue_training_gold': True,
    'next_direction_ko': 'source 실제 cue/reference와 query jitter를 분리하고 crop miss는 U/위치 문제로 유지하는 작은 semantic component를 구축한다. exact ownership은 보조 단서로만 유지하고, source AI cue는 검토된 조건부 teacher로 표시하며 native8 semantic 정답으로 승격하지 않는다. 사후 문턱/정책 완화나 현재셋 반복 선택으로 다음 실험을 고르지 않는다.',
    'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
dump(V/'RESULTS_KO.json',report)

css='''body{font:16px/1.65 system-ui,sans-serif;color:#172536;background:#eef2f7;margin:0}main{max-width:1180px;margin:auto;background:white;padding:42px 50px}h1{font-size:32px;line-height:1.3}h2{margin-top:42px;border-top:2px solid #dce5ef;padding-top:20px}h3{margin-top:28px}h4{margin-bottom:6px}.lead{font-size:19px}.box{padding:18px 22px;background:#fff3db;border-left:5px solid #bc7000}.good{background:#eef7f4;border-color:#14816a}.small,figcaption{color:#536577;font-size:14px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}th,td{border:1px solid #dce3ed;padding:8px 11px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf3f9}.scroll{overflow-x:auto}figure{margin:24px 0}img{max-width:100%;height:auto}a{color:#1765a5}code{overflow-wrap:anywhere;font-size:13px}.matrixgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px 24px}@media(max-width:850px){main{padding:20px}.matrixgrid{display:block}}'''
body = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v70 — 대상 소유권 head와 실제 가림 진단</title><style>{css}</style><main>
<p class="small">2026-10-23 · source-only 실험 완료 · 현재 223장 새 평가 0 · PnP 0 · 전체 목표 진행 중</p>
<h1>v70: 보임을 더 살렸지만<br>외부 물체와 배경을 구분하는 문제가 남았다</h1>
<p class="lead">같은 source 1,091개 target·8,590개 query에서 동결 v61 및 기존 모델과 비교했다. 새 head는 YCB 가림 개입의 known V/E 집합에서 보임 recall <b>85.3%</b>, known E를 안 보임으로 판단하는 recall <b>77.1%</b>를 보였다. 그러나 배경/대상 밖 O를 포함하면 보임 precision은 <b>50.5%</b>다. 실제 공개 목재의 known E는 <b>15/31 (48.4%)</b>만 안 보임으로 판정했다.</p>
<div class="box"><b>HOLD — 현재 223장에는 아직 적용하지 않았다.</b><br>사전 고정한 비교 자격 8항목 중 3항목 실패. strict25를 포함한 기존 진단 10항목은 5통과·5실패다. 이번 component 실험 완료가 사용자의 가림 판별 목표 해결을 뜻하지 않는다.</div>
<h2>1. 무엇을 학습하고 비교했는가</h2>
<p>40,137개 parameter의 작은 head가 query 주변의 RGB·P3·동결 MobileSAM과, 같은 대상의 다른 초기 ROI에서 만든 외형 prototype을 읽는다. 현재 ROI 자체는 prototype pooling에서 제외한다. 다른 ROI의 SAM foreground가 실제 대상이라는 보장은 없고, 겹치는 ROI도 있을 수 있다. GT mask·reference 좌표·native semantic ID는 입력으로 쓰지 않았다.</p>
<p>합성과 공개 YCB 원본·YCB 불투명 다른 물체 개입·Pallox 목재 쌍에서 <b>정답이 알려진 V/E만</b> 학습했다. O/UNKNOWN을 E로 바꾸지 않았다. 2,000 update를 실행했고, 사전에 정한 네 source 균등 VAL NLL의 첫 최솟값인 <b>100 update</b> checkpoint를 선택했다. 고정 문턱은 pV≥0.5. 이후 성능표·crop을 보고 checkpoint나 문턱을 바꾸지 않았다.</p>
<p>이 출력은 <b>V/E 조건부 보임 점수</b>다. ‘안 보임’ 출력만으로 원인이 외부 가림·자기 가림·위치 오류인지 추론할 수 없다. O를 학습하지 않은 한계는 전체 V/E/O 성능에 그대로 드러낸다. 자기 가림의 semantic 정답 학습은 이번 component에 포함하지 않았다.</p>
<p class="small">실제 학습 전 metadata 읽기 오류와 첫 backward의 cuBLAS 결정성 환경 오류를 각각 보존했다. 둘 다 optimizer update 0에서 중단했고, 환경만 고쳐 새 run에서 같은 protocol을 실행했다. 최종 code·data·checkpoint hash가 고정돼 있다. 네 source의 architecture/objective/sampler가 함께 바뀐 pilot이므로 특정 branch의 인과 기여는 입증하지 않았다.</p>
<h2>2. 같은 source에서의 저장 출력 비교</h2>
<p><b>주 지표 분모:</b> native-valid이며 exact class가 알려진 V/E/O 7,247점. V=그 query 픽셀이 보이는 대상, E=알려진 외부 가림, O=그 픽셀의 대상 밖/배경. semantic reference나 crop 전체의 대상 유무와 동일하지 않다. 별도 class UNKNOWN 1,337점과 invalid 6점은 V/E/O 정답으로 편입하지 않았다.</p>
<p>TP/FP/FN/TN은 V 대 (E+O)다. ‘E→안 보임’은 known E의 recall이며 E라는 원인을 예측한 precision이 아니다. native-valid 출력은 이 고정 binary 문턱에서 보류 0%; invalid 6/8,590만 U(0.070%)로 유지했다. GT UNKNOWN은 모델 보류와 다르다.</p>
{table(['source','모델','known N','TP/FP/FN/TN','V precision','V recall','known E→안 보임','known 보류율'],metrics)}
<p class="small">v61도 같은 8,590개 고정 feature/query에서 원 checkpoint를 새 forward한 뒤 점수를 저장해 비교했다. 이전 겹치는 source 출력과 판정 차이 0. YCB는 팔레트가 아닌 물체이므로 v61의 YCB recall만으로 현재 팔레트의 전이 실패를 증명하지 않는다. Pallox는 한 목재 bin의 촬영 쌍이며 낮은 팔레트 전체를 대표하지 않는다. 이 source 표를 현재 목재/정사각형 성능으로 재명명하지 않는다.</p>
<h3>조건부 V/E만 보면 좋은 수치와, 전체 분모의 차이</h3>
{table(['source','V/E N','TP/FP/FN/TN','조건부 V precision','V recall','E→안 보임 recall','V/E NLL'],cond_rows)}
<p>YCB 가림 개입에서 조건부 precision 90.4%는 O 1,096점을 뺀 수치다. O를 포함한 precision 50.5%와 함께 봐야 한다. O 1,096점 중 839점을 보임으로 판정했다. 원본 YCB의 조건부 precision 100%는 이 분모에 E가 0개라서 생기는 값이며 negative 구분 성능의 증거가 아니다.</p>
<details><summary>v61 / v70 혼동행렬 보기 · 행=GT V/E/O, 열=보임/안 보임/U</summary><div class="matrixgrid">{''.join(matrices)}</div></details>
<h3>같은 원본→가림 개입 쌍에서 둘 다 맞았는가</h3>
<p>원래 V이며 개입 뒤에도 V인 1,126점: 원본 V 978, 개입 뒤 V 960, <b>둘 다 V 937/1,126</b>. 개입 후 41점을 잃고 23점을 회복했다. 원래 V에서 E가 된 445점: 원본 V 392, 개입 뒤 안 보임 343, <b>둘 다 올바른 전환 298/445</b>. 나머지에는 원본부터 보임을 놓친 점도 있다.</p>
<p>5×5 전부 V인 기존 고정 core에서도 원본 637/714, 개입 265/296으로 이전 reference 수치 679/714, 278/296에 미달했다. 좁은 pixel core 진단만으로 semantic 평가를 봉쇄하지 않도록 별도 broad 정책을 학습 전에 만들었지만, 이번에는 그 8개 조건도 통과하지 못했다.</p>
<h2>3. 실제 crop: 좋아진 사례와 남은 실패</h2>
<p>다음 4 V·4 E는 V70 점수를 보기 전에 기존 버전에서 고정한 동일 ID다. 모델별 표시는 저장 점수이며 crop·query·대상 box를 바꾸지 않았다. 보이는 점 4개 중 V70은 2개를 계속 놓친다. SAM2.1은 이 4개 V를 모두 살렸지만, 아래 불투명 다른 물체 개입 4개는 모두 보임으로 잘못 판단했다.</p>
{image(G/'FIXED_VISIBLE_SAVED_COMPARISON.png','고정 V 4개: V70은 2개 보임, 2개 안 보임. 이 소집합은 전체 recall 추정용이 아니다.')}
{image(G/'FIXED_EXTERNAL_SAVED_COMPARISON.png','고정 E 4개: V70은 3개 안 보임, 1개 보임. 캔 뚜껑 위 cardboard는 여전히 pV 0.968이다.')}
<h3>공개 목재에서도 실제 가림을 보임으로 보는가</h3>
<p>known E이며 V70은 보임·raw MobileSAM은 안 보임인 11점에서, ID SHA 순으로 서로 다른 두 target을 고정한 뒤 RGB를 열었다. 두 selected A의 q는 cardboard 위에 있고, paired B의 같은 q에는 목재 판재가 보인다. V70은 가려진 A도 <b>0.685 / 0.839</b>로 보임이다. MobileSAM·v65·v66은 두 A 모두 안 보임이다.</p>
<p>고정 native5×5는 A=E, B=V가 25/25점이고 paired depth 차이는 최소 149mm / 812mm다. 실제 crop과 함께 보면 이 두 실패는 경계 1px 오차보다 <b>실제 foreground 소유권 구분 실패</b>와 더 일치한다. 다만 source teacher는 SAM·제공 pose/depth 정합 조건부이며 native RGB-depth 정합 인증이나 human semantic gold가 아니다. 두 disagreement 사례를 31개 E 또는 다른 촬영 family의 대표 표본으로 해석하지 않는다.</p>
{image(PALL/'FIXED_TWO_PUBLIC_PALLOX_E_DISAGREEMENTS.png','동일 query의 실제 공개 목재 RGB 쌍. 왼쪽은 앞 cardboard, 오른쪽은 대상 목재. 원본 crop과 표시 crop을 함께 보존했다.')}
<h2>4. 라벨 목표 차이는 여전히 별도로 해결해야 한다</h2>
<p><b>query 한 픽셀의 target 소유권, 주변에 target 단서가 있는지, 특정 reference 코너가 보이는지는 서로 다르다.</b> 합성 TRAIN 8장·64개 기존 reference/query를 실제 RGB로 점검하니 virtual cuboid reference가 바닥/빈 공간을 가리키는 경우가 있었다. 기존 proxy D 41점 중 AI 관찰로 7점은 물리적 cue가 없는 virtual/floor, 4점은 모호했다. 이것을 semantic D 정답으로 자동 승격하지 않는다.</p>
<p>별도로 영상만 보고 고정한 새 2D cue 제안 8개 중 6개에 좌표를 제안하고, 좌표를 움직이지 않은 채 mask 소유권을 점검했다. 1개는 고정 불확실성 영역 전체가 target 밖이라 보류, 1개는 target 내부이나 corner/rim 연관이 추가 확인 필요, 1개는 실제 cue가 보여도 기존 q/crop 밖(MISS)이다. 명확한 경계 cue이면서 기존 crop에 들어오는 것은 이 소집합에서 3개뿐이다. 내부의 실제 3D edge도 유효한 cue일 수 있으므로 silhouette에만 목표를 한정하지 않는다.</p>
<p>한 제안은 대표 cue 픽셀의 M4가 0이어도 고정 불확실성 영역 안에 보이는 target edge가 있었다. 따라서 semantic D에 query-pixel M4=0만 붙이면 같은 오답 원인이 반복된다. 이 제안들은 <b>AI 검토 후보이며 human gold 또는 기존 native8 semantic label이 아니고, 새 학습에는 아직 투입하지 않았다.</b> crop MISS를 E나 물리적 부재로 바꾸지도 않았다.</p>
<p>{link(S/'source_anchor_review/FIXED8_TRAIN_REFERENCE_QUERY_CONTACT_SHEET.png','합성 8장·64 reference/query 실제 영상')} · {link(S/'source_cue_reference_proposals/FROZEN_CUE_OWNERSHIP_SANITY_OVERLAY.png','고정 cue와 mask 소유권 대조')} · {link(S/'source_cue_reference_proposals/FROZEN_CUE_OWNERSHIP_SANITY.json','정확한 판정·분모 JSON')}</p>
<h2>5. 현재 평가셋의 상태와 다음 작업</h2>
<p>현재 223장은 사용자가 직접 촬영했고 KU Leuven 자료와 무관함을 확인했다. 출처 확인과 hash denylist를 보존했다. 현재 데이터의 학습·문턱·checkpoint 선택은 없고, 이번 V70 현재 추론도 0이다. 아래는 <b>기존 저장 결과</b>의 좌표 오차≤3px인 human D recall이며 이번 모델의 성능이 아니다.</p>
{table(['기존 current 모델','목재 153점','정사각형 223점'],[['v61','42/153 (27.5%)','43/223 (19.3%)'],['v63','80/153 (52.3%)','154/223 (69.1%)'],['v65','62/153 (40.5%)','132/223 (59.2%)'],['v66','47/153 (30.7%)','105/223 (47.1%)']])}
<p>다음 실험은 source의 <b>실제 cue/reference 상태를 query jitter와 분리해 고정</b>하고, crop 관측 가능성과 외부 가림을 각각 감독하는 component로 이어간다. exact ownership은 보조 단서로 사용할 수 있지만 현재 head의 점수를 곧바로 semantic 가림 결과로 쓰지 않는다. source teacher의 종류·불확실성, foreground 소유권, crop MISS, 실제 자기 가림을 구분하고 현재 평가셋으로 정답 정의나 문턱을 맞추지 않는다. R2B 공개 낮은 목재 영상도 확보했으나 native8/semantic gold는 아직 없으며, 유사 촬영 family의 두 clip을 독립 train/VAL로 나누지 않았다.</p>
<div class="box good">현재까지 확인된 원인은 두 가지다. <b>기존 합성 라벨과 원하는 semantic 목표의 차이</b>가 있고, 목표가 분명한 공개 실사의 실제 가림에서도 <b>분류 실패</b>가 있다. 라벨 이름만 바꾸거나 SAM 모델 크기만 키워 해결됐다고 할 수 없다. 새 source 실험은 두 문제를 각각 검증할 수 있게 이어간다.</div>
<h2>6. 재현·검산 파일</h2>
<p>{link(V/'RESULTS_KO.json','이 보고서 JSON')} · {link(E/'SOURCE_RESULTS.json','7개 모델 전체 집계')} · {link(E/'SOURCE_RESULTS.csv','전체 지표 CSV')} · {link(E/'PER_QUERY.csv','동일 query별 저장 판정')} · {link(V/'SOURCE_ADMISSION_DECISION.json','사전 정책에 따른 HOLD')} · {link(V/'FIT_PROTOCOL.json','학습 전 고정 protocol')} · {link(V/'runs/OWNERSHIP_LOO_PAIR_V1_RUNTIME_V2/MODEL_FREEZE.json','최종 model freeze')} · {link(P/'query_visibility_controlled_real_20261022_v66/RESULTS_KO.html','기존 current 평가 보고서')}</p>
<p class="small">새 학습은 공개 source와 합성만 사용. 현재 human annotation 평가 전용, PnP 변경 없음. 원 START_HERE/HANDOFF와 이전 부정 결과 보존. 모델 checkpoint SHA256 <code>{F['checkpoint_sha256']}</code>.<br>결과 작성: {html.escape(report['created_at_UTC'])}. 전체 목표 진행 중.</p>
</main></html>'''
assert not (V/'RESULTS_KO.html').exists()
(V/'RESULTS_KO.html').write_text(body)
print(json.dumps({'report':bind(V/'RESULTS_KO.html'),'JSON':bind(V/'RESULTS_KO.json'),'root_review':bind(V/'source_audit/ROOT_FIXED_SOURCE_RGB_REVIEW.json')},ensure_ascii=False))
