from pathlib import Path
import json, csv, hashlib, html
from datetime import datetime, timezone

V=Path(__file__).resolve().parent
P=V.parent
B=V/'frozen_C14_source_baseline'
S=json.loads((B/'saved_weak_evaluation/SOURCE_RESULTS.json').read_text())
C=json.loads((P/'query_visibility_consistent_query_context_20261023_v119/ROOT_RESULTS_KO.json').read_text())
def bind(p):
    p=Path(p)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,x):
    with p.open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2); f.write('\n')

paired=list(csv.DictReader((B/'saved_weak_evaluation/PAIRED_QUERY_RANK2304_DIAGNOSTICS.csv').open()))
ranks=[]
for rank in range(8):
    a=[r for r in paired if int(r['donor_rank'])==rank and r['teacher_triplet_known_D_E_D']=='True']
    centre=[r for r in a if int(r['jitter_index'])==4]
    changed=[]
    for cue in range(32):
        q=[r for r in a if int(r['cue_index'])==cue]
        if len({r['cover_prediction'] for r in q})>1: changed.append(cue)
    ranks.append({'rank':rank,'centre_N':len(centre),'centre_cover_E':sum(r['cover_prediction']=='1' for r in centre),'centre_sham_D':sum(r['sham_prediction']=='0' for r in centre),'all9_N':len(a),'all9_cover_E':sum(r['cover_prediction']=='1' for r in a),'cover_decision_changes_across9_offsets_conditions':len(changed)})

out={
 'completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),
 'goal_status':'active','fully_solved':False,
 'phase':'V122_PUBLIC_FLAT_PALLET_FROZEN_C14_BASELINE_AND_VISUAL_DIAGNOSIS',
 'outcome_KO':'공개32cue의 제어 합성에서 C14 중앙 BEFORE32/32·SHAM229/230·COVER184/230. 콘101/102와 목재 블록3/32의 큰 차이. 현재 실사 외부가림15/90은 미해결.',
 'actual_work':{'source_cues':32,'source_RGB_images':9,'source_capture_groups':6,'planned_occurrences':4896,'known_D':2358,'known_E':2070,'teacher_U':468,'unique_model_inputs':4428,'encoder_primary_calls':554,'encoder_repeat_calls':1,'frozen_head_calls':2,'new_fit_updates':0,'current_model_calls':0,'PnP':0},
 'source_metrics':S['metrics'],'source_rank_summary':ranks,
 'current_comparison_saved_only':C['current_primary_comparison'],
 'observation_KO':[
  '현재 실사: 노출된 모서리는 보임으로 유지되지만, 콘과 팔레트가 섞인 경계 및 콘이 중심을 차지한 경우 모두 가림 누락이 있다.',
  '공개 합성: 같은 cue/query에서 가리는 물체를 바꾸면 판정이 달라진다. 중앙 콘 cover101/102, 목재 블록 cover3/32. 가림 물체 외형에 따른 차이가 재현된다.',
  '공개 합성의 원래 보이는 cue32/32와 옆에 둔 물체229/230은 D로 유지된다. 외부 물체 존재만으로 E가 되는 모델은 아니다.',
  '공개 합성 콘이 잘 되지만 자연 촬영 콘은 실패한다. 이 차이만으로 재질·크기·문맥·query 대응 중 하나를 단독 원인으로 확정할 수 없다.'
 ],
 'interpretation_KO':'현재 입력/학습 조건에서 전이와 가리는 물체 외형 일반화가 부족하다는 증거다. 단순 D/E 목표 자체가 과도하다는 증거는 아니다. 모든 작은 crop에서 100% 판정을 보장하지 않으며 좌표의 정확성을 검증한 결과도 아니다.',
 'next_action_KO':'공개 source에서 확인한 목재 블록 등의 COVER 누락과 같은 물체의 SHAM 보존을 함께 학습하는 최소 변경을 검증한다. 이미 잘 되는 합성 콘만 반복해서 맞추는 것을 개선으로 간주하지 않는다. 공개 촬영 조건·재질·크기·가림 경계 다양성을 늘리고 source에서 고정한 뒤 current를 평가한다.',
 'limits_KO':['32cue/9RGB에서 만든4896개는 독립 실사4896장이 아니다.','AI 약한 구조 단서+알려진 합성 alpha에 기반한 조건부 보조 라벨이며 native 키포인트/사람 gold는 아니다.','TOMIE4장은 같은 촬영 episode이며 전부 development validation. LOCO 원래 fold 유지.','9offset은 각축±3px이며 대각선은4.24px. 새로운 public query 모사이며 실제 초기 q를 교정하지 않는다.','Teacher U468과 모델 보류는 별개다. 모델은4896개 모두 판정했으며 known D/E에서 보류0.','current223은 과거 개발에 노출되어 fresh blind holdout으로 부르지 않는다. 학습·정규화·문턱·checkpoint 선택에는 사용하지 않았다.','Self occlusion은 초기 기하 경로에서 분리한다. 학습 head는 D/E이며 PnP/xy 회귀를 하지 않는다.'],
 'evidence':{'corpus':bind(V/'actual_corpus/CORPUS_SEAL.json'),'prediction':bind(B/'PREDICTION_SEAL.json'),'source_evaluation':bind(B/'saved_weak_evaluation/EVALUATION_SEAL.json'),'current_saved_results':bind(P/'query_visibility_consistent_query_context_20261023_v119/ROOT_RESULTS_KO.json'),'public_actual_visual_review':bind(V/'ROOT_ACTUAL_VISUAL_REVIEW_AND_AUX_SCOPE.json')}
}
write(V/'ROOT_RESULTS_KO.json',out)

page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>보임·외부 가림: 성공과 실패를 실제 crop으로 비교</title>
<style>body{margin:0;background:#eef2f5;color:#172c39;font:17px/1.7 system-ui,"Noto Sans CJK KR",sans-serif}main{max-width:1250px;margin:auto;padding:32px 22px 80px}h1{font-size:32px;line-height:1.35}h2{font-size:25px;margin:0 0 14px}p{margin:10px 0}section{background:white;border:1px solid #dbe3e8;border-radius:14px;padding:24px;margin:22px 0}img{width:100%;height:auto;display:block;margin:16px 0}table{width:100%;border-collapse:collapse;font-size:16px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d9e0e5}th{background:#eef3f5}.lead{font-size:21px}.note{font-size:15px;color:#526876}.status{display:inline-block;padding:5px 12px;background:#fff0d3;border-radius:8px;color:#785215}a{color:#076b9a}ul{padding-left:24px}.scroll{overflow-x:auto}.evidence{font-size:14px}details{margin-top:16px}summary{cursor:pointer;font-weight:700}</style>
<main><span class="status">C14 현재 모델 · 해결 진행 중 · 새 학습 없음</span>
<h1>어디서 맞고, 어디서 놓치는가</h1>
<p class="lead">보이는 모서리는 잘 살리지만, 실제 외부 가림은 많이 놓칩니다. 목표를 낮춰야 한다는 결론보다, 학습한 조건과 실제 입력 사이의 차이를 확인해야 한다는 결과입니다.</p>
<p>판정 대상은 <b>고정된 초기 query 부근의 기대한 팔레트 단서가 보이는지, 외부 물체에 가려졌는지</b>입니다. 자기 가림은 초기 기하로 분리하며, 이 head는 좌표를 찾거나 고치지 않습니다.</p>

<section><h2>1. 현재 실사에서 맞고 틀린 사례</h2>
<p>같은 고정 사례에 최신 C14 저장 판정을 붙였습니다. 무표시 crop, 전체 영상, 표시용 query 복사본을 함께 봅니다. 위는 보임·가림 성공, 아래는 외부 가림을 보임으로 놓친 경우입니다.</p>
<img src="current_visual_followup/C14_FIXED4_SUCCESS_FAILURE.png" alt="C14 실제 촬영 성공과 실패 네 사례">
<div class="scroll"><table><tr><th>현재 평가의 동일 query</th><th>보임 정답</th><th>외부 가림 정답</th><th>D/E 내 보류</th></tr><tr><td>목재 C14</td><td>599/616 · 97.2%</td><td>9/29 · 31.0%</td><td>0</td></tr><tr><td>정사각형 C14</td><td>531/531 · 100%</td><td>6/61 · 9.8%</td><td>0</td></tr></table></div>
<p><b>공통 관찰:</b> 남은 팔레트와 콘이 함께 보이는 경계에서도 실패하고, crop 중심이 콘으로 채워진 경우에도 실패합니다. 따라서 “주변 목재가 조금 보여서 그렇다” 하나로 전부 설명할 수 없습니다.</p>
<p class="note">선택한 그림은 진단 사례이며 실패 빈도는 전체 평가표로 판단합니다. 가려진 점의 reference는 추정값일 수 있어 작은 q−r만으로 좌표가 정확하다고 보지 않습니다. 기존 실사 모델은 가림83/90을 잡지만 보임655/1147·보류177/1237의 다른 절충이 있으므로 가림 recall만으로 전체 우열을 말하지 않습니다.</p></section>

<section><h2>2. 공개 영상에서 같은 지점만 덮어 보았습니다</h2>
<p>목재·회색·파란색·밝은색 팔레트의 구조 단서32개를 고정했습니다. 원본, 그 단서를 덮은 합성, 같은 물체를 옆으로 옮겨 단서는 보이게 둔 합성을 비교합니다. 전체 영상과 crop 모두 같은 합성을 사용합니다.</p>
<img src="source_prediction_visual_review/SAME_CUE_CONE_VS_WOODBLOCK.png" alt="같은 공개 팔레트 단서에 콘과 목재 블록을 덮었을 때의 차이">
<p><b>같은 query에서도 가리는 물체에 따라 달라집니다.</b> 중앙 기준 콘 cover는101/102, 목재 블록 cover는3/32만 가림으로 판정했습니다. 목재 블록은 목재 팔레트 자체와 다른 외부 물체이며, 알려진 불투명 mask가 원래 단서의 불확실성 영역 전체를 덮습니다.</p>
<p class="note">2D 붙여 넣기이므로 원근·조명·흐림·그림자에 자연 촬영과 차이가 있습니다. 공개 cue는 약한 구조 라벨이며 native 키포인트/사람 정답은 아닙니다. 콘에서의 높은 성능이 자연 촬영 콘 가림 해결을 뜻하지 않습니다.</p>
<img src="source_prediction_visual_review/SOURCE_SUCCESS_FAILURE_TRIPLETS.png" alt="공개 합성의 성공과 실패 triplet">
</section>

<section><h2>3. 공개 source의 숫자와 남은 차이</h2>
<img src="source_prediction_visual_review/SOURCE_METRICS.png" alt="공개 source에서 물체별 C14 성능">
<p>중앙에서 원래 보이는 단서32/32와 옆에 놓은 물체229/230은 보임으로 유지했습니다. 덮은 경우는184/230으로,46개를 놓쳤습니다. 모든 known D/E에서 모델 보류는0입니다.</p>
<p>32cue ×9query offset ×17조건으로4,896행을 만들었습니다. 그중 D2,358·E2,070·라벨 미정468입니다. <b>9장의 원본에서 파생된 상관된 사례</b>이므로4,896장의 독립 실사 성능으로 해석하면 안 됩니다. 작은 콘이 단서 전체를 덮지 못한26조건은 U로 유지했습니다.</p>
<p>실사에서 외부 가림15/90만 잡는 반면, 위 합성 콘은 거의 다 잡습니다. 현재 증거는 <b>일반적인 외부 가림 개념을 학습한 것과 실제 query·가림 외형·촬영 조건까지 일반화한 것이 다르다</b>는 점을 보여줍니다. 단독 원인은 아직 확정하지 않았습니다.</p>
<details><summary>source 구분·offset·실행 근거</summary><p>LOCO10cue의 원래 fold를 유지했고 TOMIE22cue는 같은 episode의 development validation에만 두었습니다. 새 fit은0이며 기존 C14 가중치와 logit0 문턱을 고정했습니다. 4,428개의 동일 입력 단위에 DINO를 적용하고 전체4,896행에 대응했습니다. query는 공개 reference에서 각축−3/0/+3px로 모사했으며 대각선 거리는4.24px입니다. 실제 초기 q를 교정한 결과가 아닙니다. Teacher U468에도 모델은 판정했으며 라벨 U를 모델 보류로 바꾸지 않았습니다.</p><p><a href="frozen_C14_source_baseline/saved_weak_evaluation/SOURCE_RESULTS.json">원본 source 평가 JSON</a> · <a href="frozen_C14_source_baseline/saved_weak_evaluation/PAIRED_QUERY_RANK2304_DIAGNOSTICS.csv">query·물체별 대응표</a> · <a href="source_prediction_visual_review/READOUT_KO.html">상세 source 그림·32cue 링크</a> · <a href="fixed_corpus_visual_review/GALLERY_KO.html">합성 전체32cue·8조건</a></p></details></section>

<section><h2>4. 목표가 너무 높은가, 다음에는 무엇을 바꿀 것인가</h2>
<p><b>자기 가림을 분리한 뒤 보임/외부 가림을 판단한다는 목표 자체가 과도하다는 증거는 없습니다.</b> 실제 성공도 있고, 공개 데이터에서 동일 query의 cover/sham을 구별하는 능력도 확인했습니다. 다만 모든 작은 crop에서100%를 보장하거나, 가림 분류만으로 초기 키포인트 좌표의 정답까지 확인하는 것은 정보가 부족할 수 있습니다.</p>
<ul><li><b>라벨:</b> “어떤 팔레트 재질이 보임”과 “기대한 특정 단서가 보임”을 구분합니다. 가림 물체가 옆에 있는 경우도 반드시 보임 사례로 둡니다.</li><li><b>데이터:</b> 이미 잘 되는 합성 콘만 늘리는 대신, 공개 source에서 확인한 목재 블록 등 cover 실패와 같은 물체의 sham 보존을 함께 학습·검증합니다. 촬영 거리·흐림·재질·경계 가림의 다양성이 필요합니다.</li><li><b>모델과 검증:</b> 기존 모델에도 문맥 입력은 있습니다. 단순히 더 큰 crop을 넣는 것만으로 해결됐다고 보지 않고, 기대한 팔레트 단서와 query를 덮는 물체의 관계를 학습하는 최소 변경을 source에서 검증합니다. source에서 설정을 고정한 뒤 현재 실사는 평가만 합니다.</li></ul>
<p class="note">현재223은 개발에 노출된 자료이므로 독립 최종 blind 성능을 뜻하지 않습니다. 이번 단계에서는 새 학습·현재 평가 추론·문턱 변경·좌표 보정·PnP를 하지 않았습니다. 공개 source의 frozen 모델 추론과 저장 실사 출력 시각화만 추가했습니다.</p></section>
<p class="evidence"><a href="ROOT_RESULTS_KO.json">이 보고서 수치·근거 JSON</a> · <a href="../query_visibility_consistent_query_context_20261023_v119/ROOT_READOUT_KO.html">C14 전체 비교표</a> · <a href="../query_visibility_natural_cue_query_baseline_20261023_v121/ROOT_READOUT_KO.html">이전 자연 가림37쌍 진단</a> · <a href="public_occluders/ATTRIBUTION.txt">공개 이미지 출처·라이선스</a></p>
</main></html>'''
with (V/'ROOT_READOUT_KO.html').open('x') as f:f.write(page)
print(json.dumps({'completed':True,'results':bind(V/'ROOT_RESULTS_KO.json'),'report':bind(V/'ROOT_READOUT_KO.html')},ensure_ascii=False))
