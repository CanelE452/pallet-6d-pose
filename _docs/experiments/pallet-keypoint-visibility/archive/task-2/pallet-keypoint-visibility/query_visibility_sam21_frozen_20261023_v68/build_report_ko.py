"""Publish the frozen, source-only SAM2.1 comparison from saved results."""
import base64
import datetime
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT.parent

def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def pct(value):
    return '—' if value is None else f'{100*value:.1f}%'

def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'

def inline_image(path, caption):
    src = 'data:image/png;base64,' + base64.b64encode(Path(path).read_bytes()).decode()
    return '<figure><img src="'+src+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'

def main():
    result_path = ROOT/'source_evaluation/SAM21_FIXED_COMPARISON/SOURCE_RESULTS.json'
    d = json.loads(result_path.read_text())
    assert d['completed'] and d['source_only'] and d['source_diagnostic_status']=='SOURCE_HOLD'
    assert not d['current_inference_allowed'] and not d['solved']
    assert not (ROOT/'RESULTS_KO.html').exists()
    primary = d['models'][d['primary_model']]
    names = {'MOBILE_SAM_RAW':'기존 MobileSAM 원출력', 'SAM21_BASE_PLUS_FROZEN':'SAM2.1 고정 모델',
             'V65_BASELINE':'V65 head', 'V66_CONTROLLED_ONLY':'V66 공개 cutout head',
             'V66_CONTROLLED_PLUS_PALLOX':'V66 + Pallox head', 'V67_VISIBLE_BCE1':'V67 가시성 손실 추가 head'}
    source_names={'SYNTH':'합성', 'ORIGINAL_YCB':'공개 YCB 원본', 'CONTROLLED_YCB':'같은 YCB + 다른 물체 가림', 'PALLOX':'공개 Pallox 조건부 teacher'}
    rows=[]
    for source in source_names:
        for model in names:
            m=d['models'][model]['per_source'][source]['exact']; b=m['visible_binary_on_class_known']
            rows.append([source_names[source],names[model],m['known_class_GT'],m['UNKNOWN_class_GT'],
                f"{b['TP']}/{b['positive']}",pct(b['precision']),pct(b['recall']),
                pct(m['known_E_to_binary_nonvisible']['recall']),pct(m['known_O_to_binary_nonvisible']['recall'])])
    diag=[]
    for c in d['source_diagnostic_checks']:
        diag.append([source_names.get(c['source'],c['source']),c['condition']+' / '+c['metric'],
            str(c['value']) if c['metric']=='TP' else pct(c['value']),str(c['minimum']) if c['metric']=='TP' else pct(c['minimum']), '통과' if c['passed'] else '미달'])
    strict=[]
    for model in names:
        for source in ['ORIGINAL_YCB','CONTROLLED_YCB']:
            s=d['strict25_visible'][model][source]
            strict.append([names[model],source_names[source],f"{s['TP']}/{s['N']}",pct(s['recall'])])
    confusion=''
    for source in source_names:
        pairs=[]
        for model in ['MOBILE_SAM_RAW','SAM21_BASE_PLUS_FROZEN']:
            m=d['models'][model]['per_source'][source]['exact']
            pairs.append('<div><h4>'+names[model]+'</h4>'+table(['GT ↓ / 출력 →','보임','안 보임','보류'],[[g]+n for g,n in zip(['V: 보이는 표면','E: 알려진 외부 가림','O: 대상 영역 밖'],m['confusion']['counts'])])+'</div>')
        confusion+='<h3>'+source_names[source]+'</h3><div class="pair">'+''.join(pairs)+'</div>'
    radius=[]
    for source in source_names:
        for model in ['MOBILE_SAM_RAW','SAM21_BASE_PLUS_FROZEN']:
            m=d['models'][model]['per_source'][source]['radius2_diagnostic'];b=m['visible_binary_on_class_known']
            radius.append([source_names[source],names[model],m['known_class_GT'],pct(b['precision']),pct(b['recall']),pct(m['known_E_to_binary_nonvisible']['recall'])])
    fixed=ROOT/'source_audit/FIXED_PUBLIC_VISIBLE_CROPS/FIXED_PUBLIC_VISIBLE_CROPS.png'
    assert fixed.exists()
    extra=ROOT/'source_audit/SAM21_EXTERNAL_FALSE_VISIBLE_CROPS'
    extra_jsons=list(extra.glob('*.json')) if extra.exists() else []
    extra_pngs=[extra/'CONTACT_SHEET.png'] if (extra/'CONTACT_SHEET.png').exists() else []
    assert extra_jsons and extra_pngs, 'Wait for actual controlled-occlusion RGB diagnosis'
    v69=P/'query_visibility_reference_semantic_20261023_v69'
    source_audit=P/'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_VISIBILITY_INTAKE.json'
    summary={
        'schema':'V68_SAM21_SOURCE_COMPARISON_REPORT_KO',
        'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'verdict':'SOURCE_HOLD','fully_solved':False,'source_diagnostic_pass':4,'source_diagnostic_fail':6,
        'completed_targets':1091,'completed_queries':8590,'native_valid':8584,'native_invalid_U':6,
        'training_updates':0,'new_PnP':0,'current223_new_inference':0,'current223_selection_or_fit':False,
        'fixed_recipe':'Official SAM2.1 hiera base-plus; same RGB+same box; no query clicks; max predicted-quality branch; exact logit>0; source labels read only after output seal.',
        'result':ref(result_path),'extraction':ref(ROOT/'source_outputs/SAM21_BASE_PLUS/EXTRACTION_SEAL.json'),
        'fixed_four_examples':ref(fixed),'external_false_visible_gallery':[ref(x) for x in extra_jsons+extra_pngs],
        'source_reference_annotation_audit':ref(source_audit),
        'source_initializer_residual_diagnosis':ref(v69/'SOURCE_INITIALIZER_RESIDUAL_DIAGNOSIS.json'),
        'semantic_contract_draft':ref(v69/'DRAFT_PROTOCOL.json'),
        'full_goal_limit':'Exact query-pixel target support and original reference-keypoint semantic visibility are different labels. SAM mask absence is not an E/S causal classifier.',
        'next_work':'Preserve existing pixel metrics. Complete versioned fixed-reference/query-jitter semantic contract, including reference identity identifiability, before another semantic-label training pilot.',
        'references':{'official_SAM2':'https://github.com/facebookresearch/sam2','current_evaluation_unchanged':str(P/'query_visibility_controlled_real_20261022_v66/RESULTS_KO.html'),'semantic_contract_draft_directory':str(v69)}}
    (ROOT/'RESULTS_KO.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    body='''<h1>SAM2.1로 교체하면 가림 판별이 해결되는가?</h1>
<p class="lead">공개·합성 데이터의 같은 8,590개 query를 비교한 결과, <b>교체만으로 해결되지 않았습니다.</b> 고정한 10개 진단 조건 중 4개 통과·6개 미달입니다.</p>
<div class="callout">새 학습 0 · 새 PnP 0 · 현재 평가 223장 새 추론 0<br>1,091개 대상 / 유효 query 8,584개 / 화면 밖 query 6개는 보류(U)</div>
<p>공개 YCB 원본의 보임 recall은 MobileSAM <b>65.8% → SAM2.1 69.0%</b>로 올랐습니다. 그러나 같은 영상에 실제 다른 물체를 붙여 가린 445개 점 중, SAM2.1이 안 보임으로 판정한 것은 <b>178개(40.0%)</b>였습니다. 실제 가림 물체를 대상 마스크에 포함하는 사례도 아래 crop에서 확인할 수 있습니다.</p>
<h2>비교 조건과 라벨의 범위</h2>
<p>공식 SAM2.1 hiera base-plus를 고정하고 기존과 같은 RGB·box만 입력했습니다. query 클릭은 사용하지 않았습니다. 마스크 3개 중 모델의 예측 품질이 최대인 것을 선택하고, native 해상도의 저장 float32 logit &gt; 0을 보임으로 판정했습니다. 정답은 출력 저장 이후 평가에서만 읽었습니다. 기존 모델은 저장 점수를 재사용했습니다.</p>
<p>V/E/O는 각각 해당 query의 보이는 대상 표면 / 알려진 외부 가림 / 대상 영역 밖이라는 소스 조건부 라벨입니다. 합성·BOP 렌더 및 depth mask·실제 물체 cutout·Pallox 조건부 teacher의 근거 수준은 다릅니다. <b>팔레트 참조 코너의 사람 D/E/S 정답과 동일하지 않습니다.</b> SAM 출력은 보임·안 보임의 이진 판정이므로 E/O 행의 안 보임 비율만 집계하며, E 원인 분류 precision은 없습니다.</p>
<h2>Exact query 기준 전체 비교</h2>
<p>Precision/recall은 유효하고 V/E/O 클래스가 알려진 동일 행에서 계산했습니다. UNKNOWN은 성능 분모에서 빼되 수를 남겼습니다. 학습 head의 pV ≥ 0.5, 기존 MobileSAM의 pV &gt; 0.5, SAM2.1 logit &gt; 0을 각각 유지했습니다. 이 표의 head 이진 판정은 joint confidence 보류를 적용한 3상태 원인 판정과 다른 집계입니다.</p>'''
    body+=table(['소스','모델','GT 알려짐','GT UNKNOWN','보임 TP / GT V','보임 P','보임 R','E → 안 보임 R','O → 안 보임 R'],rows)
    body+='<h2>혼동행렬과 보류</h2><p>유효 query에서는 mask 이진 판정의 보류율이 0%입니다. 이는 자신감이 검증됐다는 의미가 아닙니다. 무효 Pallox 6개는 U로 유지했습니다. 유효 UNKNOWN 1,337개(533+533+271)는 별도로 남겼으며 정답 음성으로 채우지 않았습니다.</p>'+confusion
    body+='<h2>표면 내부가 확실한 점도 놓치는가?</h2><p>기존에 고정한, native 5×5가 모두 보이는 표면인 동일 점만 비교했습니다. SAM2.1은 학습 head V66/V67보다 이 집합의 recall을 회복했지만, V65 기준 679/714·278/296에는 미치지 못했습니다.</p>'+table(['모델','소스','보임 TP / GT V','보임 recall'],strict)
    body+='<h2>미리 고른 실제 공개 실사 4개 crop</h2><p>아래 네 점은 SAM2.1 실행 전에 고른 V66의 회귀 사례입니다. SAM2.1은 네 점 모두 보임으로 판정했습니다. 기존 MobileSAM도 이 점들은 이미 보임으로 맞혔으므로, 네 사례의 회복을 SAM2.1 전체 우수성으로 해석할 수 없습니다.</p>'
    body+=inline_image(fixed,'동일 RGB/query: 원본 → 저장된 visible-mask GT → SAM2.1 선택 마스크. 빨간 십자는 query. 확대는 nearest pixel입니다.')
    body+='<h2>실제로 다른 물체가 가렸는데 보인다고 하는 사례</h2><p>모델 점수 순위를 쓰지 않고 sample ID 해시 순서로 고른 공개 cutout 오류입니다. 같은 원본/가림 후 RGB와 고정 SAM2.1 마스크를 대조했습니다. 이것은 네 개 오류의 실제 원인을 점검하는 자료이며 전체 오차 비율은 위 혼동행렬에서 판단해야 합니다.</p>'
    for png in extra_pngs: body+=inline_image(png,'공개 소스의 고정 외부 가림 오류 crop. 상세 선택·라벨·픽셀 검산은 같은 폴더의 JSON에 보존했습니다.')
    body+='<h2>미리 고정한 10개 소스 진단 조건</h2>'+table(['소스','조건','관측','최소 기준','결과'],diag)
    body+='<details><summary>주변 5×5(radius2) 진단 보기 — 별도 목표와 분모</summary><p>주변의 다른 대상 픽셀을 찾는 조건입니다. Exact 결과를 대체하거나 사후 채택하지 않았습니다. GT도 동일한 주변 조건으로 바뀌므로 exact 표와 recall만 직접 비교하면 안 됩니다.</p>'+table(['소스','모델','GT 알려짐','보임 P','보임 R','E → 안 보임 R'],radius)+'</details>'
    body+='''<h2>다음 목표를 바르게 고정하기</h2>
<p>합성 1,003개 원본 라벨을 다시 추적했습니다. renderer 참조 좌표는 있으나 기존 9-ray 차단 비율·convex-cuboid 자기 가림은 조건부 proxy이고, 추가 kp12의 visibility 값은 전부 null입니다. 기존 query는 새 무작위 jitter가 아니라 저장된 초기 추론 좌표입니다.</p>
<p>합성 TRAIN 6,144개 참조점에서 초기 q와 renderer r의 거리 중앙값은 2.67px, 90분위는 8.59px, 95분위는 19.63px였습니다. 현재 실사 좌표를 문턱 선택에 사용하지 않고도 source initializer의 오차 범위를 추적할 수 있습니다. 같은 프레임에서 조건부 가림 proxy가 다른 두 참조점이 0.60px까지 가까워지는 사례도 있었지만, 이를 실제 semantic 정답 충돌의 증거로 승격하지 않았습니다.</p>
<p>다음 계약에서는 참조점 r의 상태와 query q의 오차를 분리해야 합니다. q만 이동해도 참조점의 D/E/S는 바뀌지 않고, 참조점이 crop 밖으로 빠진 상태는 별도로 기록해야 합니다. 또한 동일 RGB·q·box에서 어떤 참조점인지 구분할 수 없는 경우를 검사해야 합니다. 좌표 오차에 대한 불변성만으로 특정 코너의 가림 상태를 학습할 수 있다고 주장할 수 없습니다.</p>
<p>공개 실사 R2B의 낮은 목재 팔레트 RGB는 확보했지만 AI가 제안한 코너는 검증된 사람 정답이 아닙니다. 공개 데이터의 촬영 묶음 분리와 명시적 teacher 근거를 유지해 이 계약에 맞춘 다음 실험을 준비합니다. 현재 평가 223장의 사람 주석은 계속 평가 전용입니다.</p>
<p><a href="source_evaluation/SAM21_FIXED_COMPARISON/SOURCE_RESULTS.csv">전체 수치 CSV</a> · <a href="source_evaluation/SAM21_FIXED_COMPARISON/SOURCE_RESULTS.json">분모·UNKNOWN·binary-known 추가 집계 JSON</a> · <a href="RESULTS_KO.json">보고서 근거 JSON</a></p>'''
    doc='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V68 SAM2.1 공개·합성 동일 대상 비교</title><style>body{font:16px/1.65 Arial,"Noto Sans CJK KR",sans-serif;color:#1e293b;max-width:1260px;margin:40px auto;padding:0 24px;background:#f8fafc}h1{font-size:30px;line-height:1.3}h2{margin-top:42px}h3{margin-bottom:12px}.lead{font-size:20px}.callout{background:#fff7ed;border-left:5px solid #ea580c;padding:16px}table{border-collapse:collapse;width:100%;background:white;font-size:14px;margin:16px 0 28px}th,td{border:1px solid #cbd5e1;padding:8px 10px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e2e8f0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}figure{margin:22px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#475569}details{background:#eef2ff;padding:16px}a{color:#0369a1}summary{cursor:pointer}@media(max-width:850px){.pair{display:block}table{font-size:12px}}</style><body>'+body+'</body></html>'
    (ROOT/'RESULTS_KO.html').write_text(doc)
    print(json.dumps({'completed':True,'HTML':ref(ROOT/'RESULTS_KO.html'),'JSON':ref(ROOT/'RESULTS_KO.json')}))

if __name__ == '__main__':
    main()
