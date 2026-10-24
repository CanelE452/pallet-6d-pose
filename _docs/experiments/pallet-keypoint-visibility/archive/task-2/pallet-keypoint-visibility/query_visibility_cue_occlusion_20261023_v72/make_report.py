"""Assemble a standalone saved-result Korean report; no models or selection."""
from pathlib import Path
import base64,csv,hashlib,html,json
R=Path(__file__).resolve().parent
P=R.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def rows(p):return list(csv.DictReader(Path(p).open()))
def pct(x):return '—' if x is None or x=='' else f'{float(x)*100:.1f}%'
def table(headers,data):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'
def img(path,caption):
    p=R/path
    return '<figure><img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
def main():
    src=read(R/'source_evaluation/SOURCE_RESULTS.json');freeze=read(R/'runs/CUE_ATTENTION_V3/MODEL_FREEZE.json')
    current=rows(R/'contract/V72_CURRENT_CLASSIFICATION_METRICS.csv');manual=rows(R/'contract/V72_CURRENT_MANUAL_NEAR_D_LE3.csv')
    sourceaudit=read(R/'source_audit/INDEPENDENT_SAVED_SOURCE_METRIC_RECONCILIATION.json')
    group={'WOOD_PRIMARY105':'목재','SQUARE118':'정사각형'};names={'v61':'v61','legacy':'기존 가림 모델','V72_CUE_ATTENTION':'V72 새 head','V72_CUE_ATTENTION+GEOMETRY':'V72 + 저장 기하 proxy'}
    primary=[x for x in current if x['cohort']=='paired_same_query_RGB' and x['task']=='D_vs_E' and x['model'] in ['v61','legacy','V72_CUE_ATTENTION']]
    selfrows=[x for x in current if x['cohort']=='same_slot_pipeline' and x['task']=='D_vs_S' and x['group']=='WOOD_PRIMARY105']
    evidence=[bind(R/x) for x in ['FIT_PROTOCOL.json','source_cache/FEATURE_RECEIPT.json','runs/CUE_ATTENTION_V3/MODEL_FREEZE.json','runs/CUE_ATTENTION_V3/FIT_COMPLETE.json','private_evaluation/CUE_ATTENTION_V3/INFERENCE_SEAL.json','source_evaluation/SOURCE_RESULTS.json','contract/V72_CURRENT_CLASSIFICATION_METRICS.csv','contract/V72_CURRENT_MANUAL_NEAR_D_LE3.csv','source_audit/INDEPENDENT_SAVED_SOURCE_METRIC_RECONCILIATION.json','source_audit/INDEPENDENT_SOURCE_CACHE_AUDIT.json','source_audit/INDEPENDENT_YCB_REFERENCE_COVER_AUDIT.json','source_audit/INDEPENDENT_SEALED_CURRENT_METRIC_RECONCILIATION.json','diagnostics/FIXED_DIAGNOSTIC_SEAL.json']]
    report=dict(schema='V72_SEMANTIC_CUE_HEAD_ACTUAL_FROZEN_RESULTS',goal_status='active',fully_solved=False,
      user_provenance_confirmation='Current223 directly photographed by user, unrelated to KU Leuven Pallox; carried from V66 provenance',
      outcome='Current v61 false-absence improved at same fixed query; public VAL unseen cue failure and instance/cue association remain unresolved',
      source=dict(N=3905,D=117,E=48,U=3740,TRAIN_known=125,VAL_known=40,VAL_cues=3,selection_step=freeze['best_update'],source_only_selection=True),
      current_primary=primary,manual_coordinate_near_D=manual,self_diagnostic=selfrows,
      invalid_current_slots=45,current_slots=1784,current_valid=1739,current_training_updates=0,current_model_or_threshold_selection=False,
      new_training_updates=1000,new_PnP_calls=0,new_backbone_source_RGB_forwards=62,new_R2B_initializer_RGB_forwards=48,
      threshold_probability_D=.5,learned_U=False,learned_S=False,source_native8_gold=False,
      interpretation=['V61 originally predicts query-center target pixel; V72 conditional source cue visibility uses same fixed q/L crops and whole-image frozen P3.',
        'Current manual near3px subset wood153/square223; broader stored-coordinate wood238 also includes85 nonmanual reference coordinates.',
        'Visible recall improvement does not establish calibrated causal external/self classification or reliable new-scene performance.',
        'Three public VAL cue objects fail; training geometry/source teacher reuse and tiny correlated groups are explicit.',
        'V70 source HOLD unchanged. V72 once current diagnostic was specified before fit; no source/current reselection.',
        'Current223 historically repeated development cohort; not a fresh final independent holdout.'],
      evidence=evidence)
    (R/'RESULTS_KO.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    body='''<h1>V72 · 고정 cue 가림 학습과 실제 평가</h1>
    <p class="lead">동일한 초기 query에서 v61의 보이는 점 누락은 줄었습니다. 공개 검증 cue의 일반화와 가림 대상의 동일성은 아직 해결되지 않았습니다.</p>
    <div class="notice">현재223장은 학습·체크포인트·문턱 선택에 사용하지 않았습니다. 공개·합성 자료로 단일 1,000 update 학습 → 공개 VAL로 step50 선택 → checkpoint·0.5 문턱 고정 → 현재223장 진단을 실행했습니다. 새 PnP는 0회입니다. 직접 촬영한 평가자료와 KU Leuven 자료가 무관하다는 사용자 답변은 기존 출처 기록에 반영했습니다.</div>
    <h2>좌표가 이미 가까운 보이는 점</h2>
    <p>기존 수동 클릭 좌표에 대한 초기 오차 ≤3px 집단입니다. 목재153개·정사각형223개를 그대로 비교합니다. 85개의 비수동 reference 좌표를 포함하는 목재238개 집단은 별도 ERROR_STRATA에 있으며 이 표와 섞지 않았습니다.</p>'''
    body+=table(['자료','모델','보인다고 판단 / D','보이는 점 recall','보류'],[[group[x['group']],names[x['model']],x['pred_visible']+'/'+x['N'],pct(x['semantic_D_recall']),x['pred_U']] for x in manual if x['model']!='V72_CUE_ATTENTION+GEOMETRY'])
    body+='''<h2>외부 가림 · 같은 RGB / 같은 query</h2>
    <p>D=사람이 보인다고 표시한 점, E=사람이 외부 물체에 가렸다고 표시한 점입니다. TP/FN은 진짜 D의 보임/없음, FP/TN은 진짜 E의 보임/없음입니다. 기존 모델의 U는 그대로 보류로 남겼습니다. 새 head의 음성은 조건부 ‘없음/비가시’ 판단이며 외부 원인의 확정은 아닙니다.</p>'''
    body+=table(['자료','모델','D / E','TP / FN / FP / TN','D precision','D recall','가림 precision','가림 recall','보류율'],[[group[x['group']],names[x['model']],x['visible_support']+' / '+x['occluded_support'],' / '.join(x[k] for k in ['TP','FN_decided','FP','TN']),pct(x['visible_precision']),pct(x['visible_recall_all_GT']),pct(x['occluded_precision']),pct(x['occluded_recall_all_GT']),pct(x['abstention_rate'])] for x in primary])
    body+='''<p>V72는 목재239개·정사각형217개의 보이는 점을 여전히 누락합니다. v61보다 가림 recall은 낮아졌습니다. 기존 가림 모델은 과거 학습·CAL 이력과 보류 규칙을 가진 비교 기준이며, 새 모델과 학습 조건이 같지는 않습니다. 정사각형의 기존 mapping 미확정 한 장을 제외한 paired cohort는117장입니다. 전체223장·1784슬롯 출력은 모두 보존했습니다. 유효하지 않은45슬롯(전체의2.52%)은 UNKNOWN으로 남깁니다. 위 paired 유효 집단의 새 head 보류율0%는 U 판별 기능이 없기 때문이며, 높은 신뢰도의 보증이 아닙니다.</p>
    <h2>자기 가림 · 목재만 별도</h2>
    <p>같은 슬롯 파이프라인의 D626 / S136입니다. 정사각형은 S 정답이 없어 정량 자기 가림 평가를 만들지 않았습니다. 새 head는 S를 학습하지 않았으므로, 아래 S recall은 사람이 S인 점을 단순 비가시로 억제한 비율입니다.</p>'''
    body+=table(['모델','TP / FN / FP / TN','D precision','D recall','S 비가시 억제율','보류율'],[[names[x['model']],' / '.join(x[k] for k in ['TP','FN_decided','FP','TN']),pct(x['visible_precision']),pct(x['visible_recall_all_GT']),pct(x['occluded_recall_all_GT']),pct(x['abstention_rate'])] for x in selfrows])
    body+='''<p>저장된 cuboid 기하 proxy를 따로 적용하면 S 억제가67/136→124/136이 되고, 보이던 D 2개를 추가로 억제합니다. 이는 새 학습이나 PnP 결과가 아니며, 학습된 자기 가림 판별로 합산하지 않습니다.</p>
    <h2>라벨 목표를 어떻게 맞췄나</h2>
    <p>v61의 학습 정답은 초기 query 중심 픽셀에 대상이 있는가였습니다. V72는 먼저 고정한 실제 cue와 불확실 영역 전체가 보이면 D, 다른 opaque 물체가 그 영역 전체를 덮으면 E를 줍니다. 일부만 덮거나 cue가 crop 밖에 있으면 U로 남깁니다. q·L·box는 reference에 맞춰 옮기거나 확대하지 않았습니다. 같은 q/L의 D→E와 cue를 보존한 중립 개입 쌍으로 학습합니다.</p>
    <p>공개 YCB 실제 cutout과 합성 팔레트로 3,905개 crop을 구성했습니다. 감독 가능한 D117/E48, U3740입니다. 학습에는 합성3 cue와 공개실사4 cue의125개 D/E 행만 사용했고, 공개실사3 cue의40개 행으로 선택했습니다. 같은 cue의 여러 query와 개입은 독립 물체 수를 늘리지 않습니다. 모두 조건부 AI cue teacher이며 기존 팔레트 native8의 사람 정답은 아닙니다. 누락·부분 가림·관계 미확정 U를 음성으로 바꾸지 않았습니다.</p>
    <h2>실제로 전이됐는가</h2>'''
    sr=[]
    for name in ['v61','V72']:
      for key in ['SYNTH|TRAIN','YCB|TRAIN','YCB|VALIDATION']:
        v=src['models'][name]['cohorts'][key]
        sr.append([name,key,str(v['D'])+' / '+str(v['E']),f"{v['TP_D']}/{v['D']}",f"{v['TN_D']}/{v['E']}"])
    body+=table(['모델','출처/분할','D / E','D 맞춤','E 맞춤'],sr)
    body+='''<p>새 head의 학습 D/E는 모두 맞지만 공개 VAL D는0/30입니다. 가림 전후 두 장을 모두 맞춘 VAL 쌍은0/10, 중립 개입 후 D 보존도0/10입니다. D cue attention peak도 학습87/87에서 불확실 영역 안에 있었지만 VAL은0/30입니다. 목표를 맞춘 것이 보편적인 cue 인식·대상 대응을 보장하지 않았습니다. 현재 평가에서의 개선과 함께 이 전이 실패를 보존합니다.</p>'''
    body+=img('source_evaluation/FIXED_FIRST_FULL_CUE_PAIRS.png','각 cue의 고정 첫 FULL query. 실제 모델 RGB96 crop. GT 표시는 렌더링 때만 추가했습니다. 빨간 attention peak는 예측 위치이며 가린 E 위치에는 보조 정답을 주지 않았습니다.')
    body+='''<h2>기존 고정 실사 crop 사례</h2>
    <p>새 점수로 사례를 다시 고르지 않고 기존 네 D·두 E를 그대로 사용합니다. 네 D에서 v61은0/4, V72는2/4를 보인다고 했습니다. 두 E에서는 V72가1/2만 비가시로 판단했고, 나머지는 약98%의 D 출력이었습니다. 이 값은 신뢰도 보정이 된 확률로 해석할 수 없습니다.</p>'''
    for path,title in [('diagnostics/FIXED_D_MODEL_INPUT_AND_ATTENTION.png','고정 D 네 사례'),('diagnostics/FIXED_E_MODEL_INPUT_AND_ATTENTION.png','고정 E 두 사례')]:
      if (R/path).exists():body+=img(path,title)
    body+='''<h2>공개 팔레트 영상에서의 입력 단계 문제</h2>
    <p>공개 R2B의 고정48 RGB에서 frozen initializer를 실행했습니다. 반환 후보220개를 모두 저장했고, 최고 confidence 선택을 유지한 채8개 미검출을 남겼습니다. 기존 목재 cue7개 중 한 개는 미검출, 검출된6개 중5개는 가장 가까운 raw9점도76.6~566.3px 떨어졌습니다. 나머지 한 개는2.13px였지만 native8 의미 대응 정답은 여전히 미확정입니다. 이 수치는 팔레트 가림 정확도가 아니라 입력 선택·좌표 대응 진단입니다. 벽 콘센트에 box가 붙은 사례도 있습니다.</p>'''
    body+=img('public_pallet_initializer/VISUAL_REVIEW/FIXED_CUE_ASSOCIATION_RAW_MARKED.png','왼쪽 실제 RGB cue, 오른쪽 같은 confidence 선택의 raw9 출력. GT에 가까운 다른 후보를 고르지 않았습니다.')
    body+='''<h2>판정과 다음 작업</h2>
    <p><strong>문제 전체는 미해결이며 배포 모델로 채택하지 않습니다.</strong> v61의 픽셀 존재 목표와 semantic cue 가시성 사이의 차이는 줄였고, 현재 동일 query에서 D 재현이 개선됐습니다. 공개 신규 cue의 일반화 실패, 다른 물체의 cue를 읽을 가능성, 입력 대상 오선택, 자기 가림·U 부재는 남았습니다.</p>
    <p>다음 데이터 확장은 현재223장이 아니라 독립 공개 팔레트 RGB와 기존 합성 native8을 기준으로 합니다. 대상 instance·native keypoint ID 대응을 먼저 고정하고, 같은 query에서 보임/완전 외부 가림/부분 가림/중립 개입과 다른 물체의 코너를 분리해야 합니다. 공개 일반 물체의 모서리는 보조 시각 감독으로 쓸 수 있지만 그 자체를 팔레트 native8 정답으로 바꿀 수 없습니다. 소수 cue의 query 수만 늘리기보다 독립 팔레트·장면·촬영 계열을 늘리는 것이 우선입니다. 문턱과 다음 체크포인트 선택도 source 분할만 사용합니다.</p>
    <p>현재223장은 과거부터 반복 관찰된 개발 평가 집단이므로 새로운 최종 일반화 보증으로 부르지 않습니다. 이번 모델은 학습 전 명시한 단일 진단으로만 평가했고, 결과를 보고 재학습·재선택·문턱 변경을 하지 않았습니다. V70의 source HOLD도 그대로입니다.</p>
    <h2>다음 공개 실사 후보</h2><p>공식 TUM LOCO는 팔레트를 포함한 물류 RGB와 COCO annotation을 제공하고 CC0로 공개되어 있습니다. <a href="https://github.com/tum-fml/loco">공식 저장소</a> 기준 5개 물류 환경의37,988장 중5,593장이 수동 annotation 대상입니다. 현재 native8 가림 정답으로 쓸 수 있는지와 촬영 계열 분리·원본 hash는 별도 확인이 필요합니다. 이 자료를 이번 모델 학습에 추가하지는 않았습니다.</p><h2>재현 자료</h2><ul>'''
    for b in evidence:body+='<li><a href="'+html.escape(b['path'])+'">'+html.escape(Path(b['path']).name)+'</a> · <code>'+b['sha256']+'</code></li>'
    body+='</ul><p>원본 인수인계 파일·기존 실험 산출물은 보존했습니다. 오류가 난 source 정규화와 빈 집단 metrics의 최초 코드·로그도 provenance에 남겼고, 모델 재학습은 한 번뿐입니다.</p>'
    css='''body{font:17px/1.65 "Noto Sans CJK KR","NanumGothic",sans-serif;max-width:1450px;margin:36px auto;padding:0 28px;color:#172436;background:#f8fafc}h1{font-size:36px}h2{margin-top:40px;font-size:25px}table{border-collapse:collapse;width:100%;font-size:15px;background:white}th,td{padding:10px 12px;border:1px solid #ced6df;text-align:left}th{background:#eaf0f6}figure{margin:24px 0}img{max-width:100%;height:auto;border:1px solid #ddd}figcaption{color:#536272;font-size:14px}code{font-size:11px;word-break:break-all}.lead{font-size:22px;font-weight:bold}.notice{padding:18px;border-left:5px solid #3169a8;background:#edf4fb}a{color:#175ca0}'''
    (R/'RESULTS_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>V72 고정 cue 가림 비교</title><style>'+css+'</style><body>'+body+'</body></html>')
    print(json.dumps(dict(report=bind(R/'RESULTS_KO.html'),json=bind(R/'RESULTS_KO.json'))))
if __name__=='__main__':main()
