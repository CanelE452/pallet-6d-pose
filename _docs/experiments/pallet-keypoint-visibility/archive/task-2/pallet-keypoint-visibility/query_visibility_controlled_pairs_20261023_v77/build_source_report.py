"""Render source-only saved results; never loads models or current223 data."""
from pathlib import Path
import json, hashlib, html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

R=Path(__file__).resolve().parent
def read(p): return json.loads((R/p).read_text())
def bind(p):
 p=Path(p); return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def pct(x): return f'{100*x:.1f}%'
old=read('frozen_V76_source_baseline/SOURCE_RESULTS.json')['results']
new=read('runs/PAIRED_SUPPORT_V1/SOURCE_RESULTS.json')['results']
audit=read('independent_saved_audit/INDEPENDENT_SELECTED_SAVED_AUDIT.json')
oa=read('independent_saved_audit/INDEPENDENT_FROZEN_V76_BASELINE_SAVED_AUDIT.json')
assert audit['PASS'] and oa['PASS']
assert read('runs/PAIRED_SUPPORT_V1/FIT_COMPLETE.json')['optimizer_updates']==2000
assert audit['selection']['best_update']==600
rows=[]
for domain in ('NATURAL_ORIGINAL','CONTROLLED_AFTER'):
 for source in ('SYNTHETIC','YCB_PUBLIC_REAL'):
  key=f'{domain}|{source}|val'
  for model,result in [('V76',old),('V77',new)]:
   v=result['natural_and_after'][key]
   rows.append(dict(domain=domain,source=source,model=model,**v))
comparison=dict(schema='V77_SAVED_SOURCE_COMPARISON',rows=rows,
 paired_V76=old['paired_anchors'],paired_V77=new['paired_anchors'],
 labels='Reference-local support weak teacher C=0,V=1,U=-1. Not human D/E/S.',
 checkpoint='NaturalVAL4strata macroBCE only, earliest strict minimum update600/2000; fixed threshold0.5.',
 current_reads=0,PnP=0,fully_solved=False,
 bindings=dict(old=bind(R/'frozen_V76_source_baseline/BASELINE_SEAL.json'),
 new=bind(R/'runs/PAIRED_SUPPORT_V1/FIT_COMPLETE.json'),
 independent=bind(R/'independent_saved_audit/AUDIT_SEAL.json')))
(R/'SAVED_SOURCE_COMPARISON.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')

fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
history=read('runs/PAIRED_SUPPORT_V1/HISTORY.json')
axes[0].plot([v['update'] for v in history],[v['naturalVAL4strata_macroBCE'] for v in history],color='#24568a')
axes[0].scatter([600],[audit['selection']['natural4strata_macroBCE']],color='#d35a29',zorder=4)
axes[0].axhline(.40797048886343945,color='#777777',linestyle='--',label='V76 selected')
axes[0].set(title='Natural VAL selector (4 strata)',xlabel='Optimizer update',ylabel='Macro BCE; lower is better')
axes[0].legend(fontsize=8)
metrics=['Visible recall','Covered recall','Covered precision']
for i,field in enumerate(['visible_recall','covered_recall','covered_precision']):
 vals=[q['natural_and_after']['NATURAL_ORIGINAL|YCB_PUBLIC_REAL|val']['metrics'][field] for q in [old,new]]
 axes[1].bar([i-.17,i+.17],vals,width=.33,color=['#8a98a9','#24568a'])
 for x,y in zip([i-.17,i+.17],vals): axes[1].text(x,y+.02,f'{100*y:.1f}',ha='center',fontsize=8)
axes[1].set(title='Natural public YCB VAL (%)',xticks=range(3),xticklabels=metrics,ylim=(0,1.05))
axes[1].tick_params(axis='x',labelsize=8)
axes[1].text(.02,.98,'Grey: V76   Blue: V77',transform=axes[1].transAxes,va='top',fontsize=8)
for i,source in enumerate(['SYNTHETIC','YCB_PUBLIC_REAL']):
 vals=[q['paired_anchors'][source+'|val']['both_correct_rate'] for q in [old,new]]
 axes[2].bar([i-.17,i+.17],vals,width=.33,color=['#8a98a9','#24568a'])
 for x,y in zip([i-.17,i+.17],vals): axes[2].text(x,y+.02,f'{100*y:.1f}',ha='center',fontsize=9)
axes[2].set(title='Controlled pair: both correct (%)',xticks=range(2),xticklabels=['Synthetic (230)','YCB public (731)'],ylim=(0,1.05))
for ax in axes[1:]: ax.yaxis.set_major_formatter(PercentFormatter(1))
for ax in axes:
 ax.spines[['top','right']].set_visible(False)
 ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.savefig(R/'SOURCE_RESULTS.png',dpi=150);plt.close(fig)

def table(items):
 s='<table><tr><th>자료·모델</th><th>N</th><th>CM: 행/열 C,V</th><th>V precision</th><th>V recall</th><th>C precision</th><th>C recall</th><th>BCE</th></tr>'
 for v in items:
  m=v['metrics'];s+='<tr>'+''.join(f'<td>{x}</td>' for x in [html.escape(v['source']+' / '+v['model']),m['N'],m['confusion_rows_GT_cols_pred_COVERED_VISIBLE'],pct(m['visible_precision']),pct(m['visible_recall']),pct(m['covered_precision']),pct(m['covered_recall']),f"{m['mean_BCE_FP32']:.4f}"])+'</tr>'
 return s+'</table>'
pr='<table><tr><th>검증 자료</th><th>모델</th><th>가리기 전 V 정답</th><th>가린 후 C 정답</th><th>둘 다 정답</th><th>전 확률 &gt; 후 확률</th></tr>'
for src in ['SYNTHETIC','YCB_PUBLIC_REAL']:
 for name,q in [('V76',old),('V77',new)]:
  v=q['paired_anchors'][src+'|val'];n=v['N'];pr+='<tr>'+''.join(f'<td>{x}</td>' for x in [src,name,f"{v['before_VISIBLE_correct']}/{n}",f"{v['after_COVERED_correct']}/{n}",f"{v['both_correct']}/{n} ({pct(v['both_correct_rate'])})",f"{v['before_logit_gt_after']}/{n}"])+'</tr>'
pr+='</table>'
diags=audit['paired_and_same_after_control_diagnostics']
pr2='<table><tr><th>검증 자료</th><th>동일 after의 C anchor·V nonanchor 둘 다 정답</th><th>같은 after V recall</th><th>V→V: 맞음→틀림 / 틀림→맞음</th></tr>'
for d in diags:
 if d['split']=='val':
  n=d['same_after_C_anchor_V_nonanchor_query_pairs'];pr2+='<tr>'+''.join(f'<td>{x}</td>' for x in [d['source'],f"{d['same_after_both_correct']}/{n} ({pct(d['same_after_both_correct']/n)})",pct(d['after_nonanchor_metrics']['V_recall']),f"{d['nonanchor_V_before_correct_after_wrong']} / {d['nonanchor_V_before_wrong_after_correct']} (N={d['nonanchor_V_to_V_stability_N']})"])+'</tr>'
pr2+='</table>'
gallery=R/'fixed_saved_score_gallery/CONTACT_SHEET.png'
extra=''
if gallery.exists(): extra=f'<h2>점수와 무관하게 미리 고른 실제 RGB 4쌍</h2><p>동일 query 중심·크기의 before/after crop입니다. 공개 실사 위에 다른 공개 객체를 합성한 영상도 포함합니다. 성공 사례만 골라낸 그림이 아닙니다. 표시한 C/V는 source 약한 교사이며 사람 주석이 아닙니다.</p><img src="{gallery}" alt="미리 고정한 실제 source crop 전후와 V76·V77 저장 확률">'
links={
 '고정 실험 조건':'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json',
 '전체 source 결과·혼동행렬':'SAVED_SOURCE_COMPARISON.json',
 '선택 체크포인트·문턱 동결':'runs/PAIRED_SUPPORT_V1/MODEL_FREEZE.json',
 '실제 4,799쌍 데이터 감사':'independent_data_audit/DATA_AUDIT.json',
 '학습·선택·저장 결과 독립 검산':'independent_saved_audit/INDEPENDENT_SELECTED_SAVED_AUDIT.json',
 '기존 V76 같은 입력 저장 결과 검산':'independent_saved_audit/INDEPENDENT_FROZEN_V76_BASELINE_SAVED_AUDIT.json',
 '가림 전후 실제 P3·RGB 입력 검산':'after_input_pack/INPUT_PACK_QA.json',
 '마지막 현재 팔레트 평가: V76 (이전 단계)':'../query_visibility_reference_local_support_20261023_v76/RESULTS_KO.html'}
linkhtml='<ul>'+''.join(f'<li><a href="{(R/p).resolve()}">{k}</a></li>' for k,p in links.items())+'</ul>'
doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V77 source 가림 전후 비교</title>
<style>body{font:16px/1.7 "NanumGothic",sans-serif;color:#213047;background:#f6f8fc;max-width:1400px;margin:35px auto;padding:0 25px}h1{font-size:32px}h2{margin-top:42px}p.lead{font-size:21px;font-weight:bold;padding:22px;background:#fff0d9;border-left:6px solid #c86b23}table{border-collapse:collapse;width:100%;background:#fff;font-size:14px;margin:20px 0}th,td{border:1px solid #d5dfeb;padding:10px;text-align:left}th{background:#e6eef9}img{max-width:100%;height:auto}a{color:#165eac}</style>
<h1>V77 — 동일 query에서 가림 전후를 구분하는가</h1>
<p class="lead">합성한 가림 변화에는 더 잘 반응합니다. 원래 공개 실사의 보이는 점 누락은 늘어, 아직 가림 판별을 해결한 결과로 채택하지 않습니다.</p>
<p>합성 1,003개와 YCB 공개 객체 4,200개의 기존 입력을 보존하고, 그중 고정 규칙으로 고른 4,799개에 가림 전후 쌍을 만들었습니다. query·참조점·crop 크기·8점 배치·기하 조건은 같고 RGB와 그 RGB에서 새로 추출한 동결 P3만 바뀝니다. 기존 41,237개 파라미터 구조와 V75 초기화를 유지해 학습 1회·2,000 update를 실행했습니다.</p>
<p>현재 평가 223장 접근·학습·문턱 선택·추가 추론은 이번 단계 모두 0회, PnP 0회입니다. 현재 자료는 직접 촬영했고 KU Leuven 자료와 무관하다는 사용자 확인을 기록했습니다. 아래 수치는 현재 팔레트 평가가 아닙니다.</p>
<h2>라벨과 선택 규칙</h2><p>V는 참조점 주변의 예상 대상 support가 남아 있는 상태, C는 예상 support가 가려진 상태, U는 판단 불명인 약한 교사입니다. 중심 픽셀 존재와 사람의 외부·자기 가림 D/E/S를 동일시하지 않습니다. target-only M0에 예상 support가 없으면 C로 바꾸지 않고 U로 남깁니다. 가상 cuboid 참조점을 물리 mesh 점으로 이동하지 않았습니다.</p>
<p>2,000 update 중 자연 VAL의 source×class 4개 집단 평균 BCE가 가장 낮은 600 update를 선택했습니다(0.39034; V76은 0.40797). controlled VAL과 현재 평가는 선택에 사용하지 않았고 문턱은 0.5입니다. 균등 가중 4집단 BCE 개선과 실제 비율로 계산한 공개 VAL 전체 BCE 악화는 함께 보고합니다.</p>
<h2>원래 영상의 source VAL</h2>'''+table([v for v in rows if v['domain']=='NATURAL_ORIGINAL'])+'''
<p>공개 자연 VAL에서 C 정답은 173→202/287로 늘지만 V 정답은 1,157→1,080/1,731로 줄었습니다. C precision은 23.2%→23.7%에 머뭅니다. 전체 known query BCE는 0.61795→0.67994로 악화합니다. 4집단을 균등 가중한 checkpoint 지표가 좋아졌다는 이유로 실사 분류가 전반적으로 좋아졌다고 결론내릴 수 없습니다.</p>
<p>자연 VAL 전체 슬롯은 합성 1,880개(V/C 허용 1,625, U187), 공개 6,400개(V/C 허용 2,018, U4,299)입니다. 나머지 known 중 입력 무효·참조점 crop 이탈도 평가 loss에서 제외합니다. U 비율은 교사·입력의 평가 가능 범위이며 학습된 confidence 보류율이 아닙니다. binary head는 유효 입력에서 confidence 보류를 학습하지 않았습니다.</p>
<h2>가림 전후 anchor를 둘 다 맞혔는가</h2>'''+pr+'''
<p>전체 4,799쌍의 지정 anchor는 같은 교사에서 before V→after C가 되었고, 독립 마스크 감사와 고정 RGB 재현 검사를 통과했습니다. TRAIN/VAL donor partition을 유지하고 같은 RGB 및 공개 target와 같은 객체 ID donor를 제외했습니다. 새로운 촬영 환경·새 객체 종류를 보장하는 분할은 아닙니다.</p>
<h2>같은 가린 이미지에서도 다른 보이는 점을 유지하는가</h2>'''+pr2+'''
<p>공개 VAL의 같은 after 이미지에서 C anchor와 V nonanchor를 함께 맞힌 비율은 V76 258/948(27.2%)→V77 421/948(44.4%)입니다. 합성 물체가 보인다는 이유로 모든 query를 C로 만드는 극단적인 설명만으로는 결과를 설명할 수 없지만, 여전히 절반 이상을 함께 맞히지 못합니다. 합성 경계·donor 외형 단서와 query별 대상 식별의 영향을 분리한 증거도 아닙니다.</p>
<h2>가린 후 전체 known query 성능</h2>'''+table([v for v in rows if v['domain']=='CONTROLLED_AFTER'])+'''
<p>가려진 anchor 외의 7개 점도 같은 교사로 다시 계산했습니다. 다른 점이 함께 가려졌으면 C, 남아 있으면 V, 모호하면 U로 남깁니다. after의 높은 C recall만으로 원래 실사나 사람 semantic 외부 가림으로의 전이를 입증할 수 없습니다.</p>
<img src="SOURCE_RESULTS.png" alt="자연 VAL 학습 곡선, 공개 자연 recall·precision, controlled 전후 정답률">'''+extra+'''
<h2>해석과 남은 한계</h2><p>동일한 공개 자연 VAL의 q-r≤3px·STRONG3×3 교사·중앙 셀(L∞≤1/18) 집단에서도 V76은 V459개 중 212개, V77은 217개를 놓칩니다. 148개는 두 모델에서 모두 누락됩니다. C76개 중 정답은 46→49개입니다. 큰 좌표 이탈·참조점 crop 이탈·footprint fallback만으로 남은 공개 실사 오류를 설명하기 어렵습니다. <a href="source_failure_audit/PUBLIC_NATURAL_FIXED_GEOMETRY_SAVED_COMPARISON.json">고정 집단 저장 결과 검산</a></p><p>이번 결과는 동일 위치·기하 조건에서 RGB의 가림 변화에 반응하도록 학습할 수 있음을 보여줍니다. 학습 구성도 자연32·가린 후 비anchor32·전후 pair32로 바뀌었으므로 손실 구성과 데이터 증가를 분리한 순수 인과 실험은 아닙니다. 공개 자연 실사 분리와 보이는 점 보존이 약해 현재 평가로 새 후보를 선택하지 않습니다.</p>
<p>공개 query는 합성 TRAIN의 초기 좌표 오차를 적용한 source augmentation이며 실제 팔레트 초기 추론기의 출력이 아닙니다. 모든 source VAL은 반복 개발 자료입니다. 합성 VAL 235개 중 84개는 V75 선택에도 사용됐습니다. 사람 semantic D/E/S 성능, 보류 확률의 신뢰성, 새 촬영에 대한 일반화는 이번 결과로 보증하지 않습니다.</p>
<p>다음 후보는 공개 source에서만 검토합니다. 기존 특징의 표현 한계와 대상 소유권 문제를 구분하고, source에서 개선을 확인하기 전 현재 평가를 반복해서 열지 않습니다. 학습·가중치·선택 조건·부정 결과는 보존합니다.</p><h2>검산과 파일</h2>'''+linkhtml+'</html>'
(R/'RESULTS_KO.html').write_text(doc)
print(json.dumps(dict(report=str(R/'RESULTS_KO.html'),comparison=bind(R/'SAVED_SOURCE_COMPARISON.json'),actual_fixed_gallery=gallery.exists()),ensure_ascii=False))
