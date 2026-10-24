"""Report from sealed source/current saved results; no model or pose execution."""
from pathlib import Path
import csv,datetime,hashlib,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;P=R.parent;RUN=R/'mixed_fit/runs/MIXED_SUPPORT_V1'
def load(p):return json.loads(p.read_text())
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def pct(v):return '—'if v is None or v=='' else f'{100*float(v):.1f}%'
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table>'
def href(p,label):return f'<a href="{p}">{html.escape(label)}</a>'
def main():
 assert not (R/'RESULTS_KO.html').exists()
 source=load(RUN/'SOURCE_RESULTS.json');freeze=load(RUN/'MODEL_FREEZE.json');cur=load(R/'contract/V76_CURRENT_PRIMARY_RESULTS.json');audit=load(R/'mixed_fit_independent_saved_audit/INDEPENDENT_SAVED_RESULT_AUDIT.json');inf=load(R/'private_evaluation/MIXED_SUPPORT_V1/INFERENCE_SEAL.json')
 assert audit['PASS']and freeze['checkpoint']==audit['checkpoint']and inf['source_model_freeze']==b(RUN/'MODEL_FREEZE.json')
 metrics=cur['tables']['CLASSIFICATION_METRICS'];models=['v61','legacy','V72_CUE_ATTENTION','V76_SUPPORT'];names={'v61':'v61','legacy':'기존 가림 모델','V72_CUE_ATTENTION':'V72 이전 head','V76_SUPPORT':'V76 새 head','V76_SUPPORT+GEOMETRY':'V76 + 저장 자기 가림 기하','V72_CUE_ATTENTION+GEOMETRY':'V72 + 저장 자기 가림 기하'};groups={'WOOD_PRIMARY105':'목재','SQUARE118':'정사각형'}
 mainrows=[x for x in metrics if x['cohort']=='paired_same_query_RGB'and x['task']=='D_vs_E'and x['model']in models]
 history=load(RUN/'HISTORY.json');fig,axes=plt.subplots(1,3,figsize=(15,4));steps=[h['update']for h in history]
 for key,label in [('heldout4strata_macroBCE','4-strata mean')]:axes[0].plot(steps,[h[key]for h in history],label=label)
 axes[0].plot(steps,[h['stratumBCE']['YCB_PUBLIC_REAL|0']for h in history],label='YCB covered');axes[0].axvline(200,color='gray',linestyle='--');axes[0].set(xlabel='Source optimizer updates',ylabel='Validation BCE',title='Fixed source selection: update200');axes[0].legend()
 for ax,g in zip(axes[1:],groups):
  rr=[next(x for x in mainrows if x['group']==g and x['model']==m)for m in models];x=np.arange(4);ax.bar(x-.18,[float(r['visible_recall_all_GT'])for r in rr],.36,label='Human visible recall');ax.bar(x+.18,[float(r['occluded_recall_all_GT'])for r in rr],.36,label='Human external recall');ax.set_xticks(x,['v61','legacy','V72','V76']);ax.set(ylim=(0,1.08),title='Wood'if g=='WOOD_PRIMARY105'else'Square');ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(R/'SOURCE_AND_CURRENT_RESULTS.png',dpi=150);plt.close(fig)
 body='<h1>V76 공개 실사·합성 혼합 head 비교</h1><p class="lead">보이는 점의 누락은 크게 줄었지만, 외부 가림을 보인다고 판단하는 오류가 증가했습니다. 현재 단계는 가림 판별 문제의 해결로 채택할 수 없습니다.</p>'
 body+='<p>현재 평가 223장은 직접 촬영했고 KU Leuven 자료와 무관하다는 사용자 확인을 기록했습니다. 이 자료의 학습·문턱 선택 사용은 0회이며, PnP는 변경하지 않았습니다. 공개·합성 자료에서 고른 200-step 체크포인트와 문턱 0.5를 고정한 뒤 평가 1회를 실행했습니다. 비교 결과를 보고 체크포인트나 문턱을 바꾸지 않았습니다.</p>'
 body+='<h2>동일 query·RGB의 사람 D/E 평가</h2><p>D=직접 보임, E=외부 가림. 각 배열은 [보임, 없음/비가시, 보류]입니다. 새 head의 음성은 외부 가림 원인을 판별했다는 뜻이 아닙니다.</p>'
 for g,label in groups.items():
  rr=[next(x for x in mainrows if x['group']==g and x['model']==m)for m in models]
  body+='<h3>'+label+'</h3>'+table(['모델','D → [V,A,U]','E → [V,A,U]','보임 precision','보임 recall','외부 가림 recall','보류율'],[[names[x['model']],f"[{x['TP']}, {x['FN_decided']}, {x['U_visible']}]",f"[{x['FP']}, {x['TN']}, {x['U_occluded']}]",pct(x['visible_precision']),pct(x['visible_recall_all_GT']),pct(x['occluded_recall_all_GT']),pct(x['abstention_rate'])]for x in rr])
 body+='<p>목재: D616/E29, 105장. 정사각형: D531/E61, paired 117장(기존 대응 미확정 1장 제외). 전체 출력은 223장·1,784점 모두 보존했습니다. 새 head의 입력 유효 1,739점, 입력 불명/무효 45점(2.52%)은 UNKNOWN입니다. 위 유효 paired 집단에서 보류율 0%인 이유는 confidence U 판별기를 학습하지 않았기 때문이며, 확신도 보증이 아닙니다. 기존 가림 모델은 과거 학습·CAL·보류 이력이 달라 학습 조건이 같지는 않습니다.</p>'
 body+='<h2>수동 참조 좌표와 오차 ≤3px인 보이는 점</h2>'+table(['자료','모델','보임 / D','보임 recall','보류'],[[groups[x['group']],names[x['model']],str(x['pred_visible'])+'/'+str(x['N']),pct(x['semantic_D_recall']),x['pred_U']]for x in cur['manual_near_D_LE3']if x['model']in models])
 body+='<p>목재153점·정사각형223점은 manual_click 참조만 사용합니다. 목재의 더 큰 stored-coordinate ≤3px 집단에는 비수동 좌표가 섞이므로 이 표와 합치지 않았습니다. 좌표 오차는 기존 참조와의 차이이며, 실제 pixel GT나 가림 원인의 증명은 아닙니다.</p>'
 body+='<h2>자기 가림은 별도 평가</h2>'
 ss=[x for x in metrics if x['cohort']=='same_slot_pipeline'and x['task']=='D_vs_S']
 body+=table(['자료','모델','D → [V,A,U]','S → [V,A,U]','D recall','S recall'],[[groups[x['group']],names.get(x['model'],x['model']),f"[{x['TP']}, {x['FN_decided']}, {x['U_visible']}]",f"[{x['FP']}, {x['TN']}, {x['U_occluded']}]",pct(x['visible_recall_all_GT']),pct(x['occluded_recall_all_GT'])]for x in ss])
 body+='<p>목재 same-slot D626/S136. 정사각형에는 대응되는 확정 S 집단이 없습니다. 새 binary head 자체는 자기 가림을 학습한 모델이 아닙니다. 저장된 초기 cuboid 기하만 별도로 적용하면 목재 S136개 중 116개를 비가시로 분류하지만 D10개를 추가로 잃습니다. 새 자세 추정이나 PnP는 실행하지 않았습니다.</p>'
 body+='<h2>합성·공개 실사에서 실제 학습한 것</h2><p>초기 query 중심 3×3 특징과 전체 local crop, target context, 8개 query 배치를 읽는 작은 head(41,237 파라미터)를 학습했습니다. V75 encoder/fusion을 재사용하고 새 binary 출력을 초기화했습니다. 이전 attention 위치 추정과 위치 auxiliary는 제거했습니다. reference-local support는 중심 한 픽셀의 존재나 사람 D/E/S와 다른 약한 교사 라벨입니다.</p>'
 body+='<p>합성 1,003개와 YCB 공개 객체 4,200개(고유 RGB 1,000개), 총 41,624점을 모두 보존했습니다. teacher V17,897/C2,149/U21,578. 실제 loss 허용 TRAIN15,465점·VAL3,643점입니다. U, 입력 무효, 참조점의 crop 이탈은 음성으로 바꾸지 않고 loss0으로 유지했습니다. YCB는 공식 AABB 참조에 합성 TRAIN768개의 초기 좌표 오차 벡터를 적용한 source augmentation입니다. 공개 검증 수치는 실제 팔레트 초기 추론기의 성능이 아닙니다.</p>'
 srcrows=[]
 for key,x in source['source_metrics'].items():
  if key.endswith('|val'):
   m=x['metrics'];srcrows.append([key,m['N'],str(m['confusion_rows_GT_cols_pred_COVERED_VISIBLE']),pct(m['visible_precision']),pct(m['visible_recall']),pct(m['covered_precision']),pct(m['covered_recall']),f"{x['admitted']}/{x['all_slots']}"])
 body+=table(['source VAL','N','CM: 행/열 C,V','V precision','V recall','C precision','C recall','입력 전체 중 평가 허용'],srcrows)
 body+='<p>공개 실사 COVERED precision23.2%·recall60.3%에서 이미 전이 한계가 드러납니다. 200-step 이후 공개 COVERED 검증 손실이 증가하며 과적합합니다. 합성235개 VAL은 기존 반복 검증 자료이며, 84개는 V75 선택에도 사용됐습니다. 새 독립 일반화 시험으로 주장하지 않습니다. 원래 TRAIN/VAL 분할을 유지했습니다.</p>'
 body+='<img src="SOURCE_AND_CURRENT_RESULTS.png" alt="source curve and current recall"/>'
 body+='<h2>실제 crop과 라벨 정의</h2><p>query 픽셀은 target0인데 주변 참조 support는 VISIBLE인 공개 실사와, query 픽셀은 target1이지만 참조점은 COVERED인 사례가 모두 있습니다. 따라서 작은 좌표 오차에서도 중심 픽셀 라벨과 참조 주변 라벨은 달라질 수 있습니다. 이 목표 차이는 실제 존재하지만, 목표를 바꾼 것만으로 실사 가림 판별이 해결되지는 않았습니다.</p>'
 sg=R/'source_crop_diagnosis/BUILD_V2/CONTACT_SHEET.png';body+='<p>'+href(sg,'고정 source 5사례: 원본 crop·M0/M4·q/r 비교')+'</p><img src="'+str(sg)+'" alt="Actual source crops and masks"/>'
 cg=R/'current_crop_diagnosis/CONTACT_SHEET.png'
 if cg.exists():body+='<p>'+href(cg,'동일한 기존 평가 6사례: v61·V72·V76 저장 출력')+'</p><img src="'+str(cg)+'" alt="Fixed actual current input crops"/>'
 body+='<h2>남은 판단과 자료</h2><p>현재 결과는 보이는 점 보존과 외부 가림 검출 사이의 큰 trade-off를 보여줍니다. 새 head를 가림 문제의 해결로 채택하지 않습니다. 다음 진단은 공개 source 검증에서 이미 확인된 COVERED 오분류를 기준으로 진행합니다. query와 참조 위치의 차이, local 전체의 다른 target support, 라벨 strong/fallback 분기를 나눠 확인하며 현재 평가셋으로 새 문턱이나 모델을 선택하지 않습니다.</p>'
 links=[('contract/V76_CURRENT_PRIMARY_RESULTS.json','모든 현재 평가표·오차 구간·기하 trade-off'),('mixed_fit/MIXED_FIT_PROTOCOL.json','학습 전 고정한 프로토콜'),('mixed_fit/runs/MIXED_SUPPORT_V1/MODEL_FREEZE.json','source 선택 체크포인트'),('mixed_fit_independent_saved_audit/INDEPENDENT_SAVED_RESULT_AUDIT.json','source 저장 결과 독립 감사'),('current_independent_saved_audit/AUDIT.json','현재 저장 결과 독립 감사'),('public_query_emulation/SEAL.json','공개 query emulation 정의와 봉인'),('public_input_pack/FINAL_INPUT_PACK_SEAL.json','공개 입력 팩'),('gt_corpus_build/GT_ONLY_CORPUS_SEAL.json','교사 라벨 corpus')]
 body+='<ul>'+''.join('<li>'+href(R/p,label)+'</li>'for p,label in links if(R/p).exists())+'</ul>'
 page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V76 혼합 head 평가</title><style>body{font:17px/1.65 "Noto Sans CJK KR",sans-serif;color:#18283d;max-width:1450px;margin:35px auto;padding:0 25px;background:#f7f9fc}h1{font-size:34px}h2{margin-top:48px}p.lead{font-size:23px;font-weight:bold;background:#fff0d9;padding:22px;border-left:6px solid #d87514}table{border-collapse:collapse;background:white;width:100%;font-size:15px;margin:20px 0}th,td{padding:11px;border:1px solid #d5deea;text-align:left}th{background:#e5eefb}img{max-width:100%;height:auto;margin:15px 0}a{color:#145fb0}</style>'+body+'</html>'
 (R/'RESULTS_KO.html').write_text(page)
 summary=dict(schema='V76_MIXED_SUPPORT_SOURCE_AND_CURRENT_RESULTS',fully_solved=False,goal_status='active',selected_update=200,source_fit_updates=2000,current_evaluation_runs=1,current_train_or_selection=False,PnP=0,checkpoint=freeze['checkpoint'],source_val={k:x for k,x in source['source_metrics'].items()if k.endswith('|val')},current_primary=mainrows,manual_near_D_LE3=cur['manual_near_D_LE3'],self_same_slot=ss,unknown_input_slots=45,report=b(R/'RESULTS_KO.html'),chart=b(R/'SOURCE_AND_CURRENT_RESULTS.png'),source_audit=b(R/'mixed_fit_independent_saved_audit/INDEPENDENT_SAVED_RESULT_AUDIT.json'))
 (R/'REPORT_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');print(json.dumps(b(R/'RESULTS_KO.html')))
if __name__=='__main__':main()
