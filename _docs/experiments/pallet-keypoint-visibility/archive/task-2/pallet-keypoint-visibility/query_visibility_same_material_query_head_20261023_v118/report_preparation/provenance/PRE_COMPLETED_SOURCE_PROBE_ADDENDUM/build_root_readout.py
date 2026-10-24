"""Read root's frozen aggregate JSON only; no source/current arrays or model calls."""
from pathlib import Path
import hashlib,html,json
H=Path(__file__).resolve().parent;V=H.parent
R=V/'ROOT_RESULTS_KO.json';EXPECTED='3c5c074606ba7ad30fda7be4cb7cdeace8e72cae34085c3f90c47cf6b990b573'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
assert bind(R)['sha256']==EXPECTED
r=json.loads(R.read_text());assert r['completed']and r['fully_solved']is False
e=html.escape
def pct(x):return '—'if x is None else f'{x*100:.1f}%'
def link(path,label):
 p=Path(path);relative=Path(__import__('os').path.relpath(p,V));return f'<a href="{e(str(relative))}">{e(label)}</a>'
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'
def matrix(m):return '['+', '.join('['+', '.join(str(x)for x in row)+']'for row in m)+']'
labels={'WOOD_PRIMARY105':'목재105','SQUARE118':'정사118'}
comparison=r['current_primary_comparison'];assert len(comparison)==10
current=[]
for x in comparison:
 m=x['confusion_rows_D_E_cols_D_hidden_U'];assert sum(m[0])==x['human_D_N']and sum(m[1])==x['human_E_N']
 assert abs(x['D_recall']-m[0][0]/sum(m[0]))<1e-12 and abs(x['E_recall']-m[1][1]/sum(m[1]))<1e-12
 current.append([labels[x['group']],e(x['display_model']),f'{m[0][0]}/{sum(m[0])}',pct(x['D_precision']),pct(x['D_recall']),f'{m[1][1]}/{sum(m[1])}',pct(x['E_precision']),pct(x['E_recall']),pct(x['U_rate'])])
source=r['public_source_comparison'];source_rows=[]
for name,key in [('목재 foreground VAL','VALIDATION_WOOD16'),('나머지 foreground VAL','VALIDATION_OTHER'),('전체 통제 VAL','VALIDATION')]:
 g=source['old194'][key]
 for model in('C11','C12','C13'):
  x=g['models'][model];source_rows.append([name,model,f'{x["cover_E"]}/{g["N"]}',f'{x["sham_D"]}/{g["N"]}',f'{x["all3_correct"]}/{g["N"]}',f'{x["mean_cover_E_BCE"]:.4f}',f'{x["margin_ge2"]}/{g["N"]}'])
evaluation_summary=Path(r['evidence']['current_evaluation']['path']).parent/'CURRENT_PRIMARY_RESULTS.json'
agg=json.loads(evaluation_summary.read_text())['tables'];cm_rows=agg['D_vs_external_E_2x3']
c13rows=[x for x in cm_rows if x['model']=='V118_C13_RAW'and x['cohort']=='paired_same_query_RGB'];assert len(c13rows)==2
cms=''.join('<div class="metric"><h3>'+labels[x['group']]+'</h3><code>'+matrix([[x['D_pred_D'],x['D_pred_hidden'],x['D_pred_U']],[x['E_pred_D'],x['E_pred_hidden'],x['E_pred_U']]])+'</code><p>D '+pct(x['D_recall'])+' · E '+pct(x['external_E_response_recall'])+'</p></div>'for x in c13rows)
same=[]
for x in cm_rows:
 if x['cohort']=='same_slot_pipeline'and x['model'].startswith('V118_C13'):
  same.append([labels[x['group']],'raw'if x['model'].endswith('_RAW')else'초기 기하 + raw',matrix([[x['D_pred_D'],x['D_pred_hidden'],x['D_pred_U']],[x['E_pred_D'],x['E_pred_hidden'],x['E_pred_U']]]),pct(x['D_recall']),pct(x['external_E_response_recall'])])
near=[x for x in r['manual_near_D_rows']if x['model']=='V118_C13_RAW'];assert[(int(x['pred_visible']),int(x['N']))for x in near]==[(153,153),(223,223)]
visual=V/'source_visual_followup/RESULT.json';vr=json.loads(visual.read_text())
visual_cards=''.join('<a href="source_visual_followup/CASE_%02d.png">사례%d 원 크기</a>'%(i,i+1)for i in range(3))
evidence='<ul>'+''.join('<li>'+link(x['path'],k)+'</li>'for k,x in r['evidence'].items())+'</ul>'
scope='<ul>'+''.join('<li>'+e(x)+'</li>'for x in r['scope_KO'])+'</ul>'
body=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C13: source 개선과 실사 E 전이의 차이</title><style>
body{{margin:0;background:#edf2f6;color:#183246;font:17px/1.75 system-ui,'NanumGothic',sans-serif}}main{{max-width:1180px;margin:auto;padding:28px 22px}}header,section{{background:#fff;border:1px solid #d6e0e9;border-radius:13px;margin:20px 0;padding:27px}}h1{{font-size:31px;line-height:1.4}}h2{{font-size:24px;line-height:1.45}}h3{{font-size:19px}}p{{margin:13px 0}}.tag{{font-size:14px;color:#60788a}}.warn{{background:#fff0e6;border-left:5px solid #ae5929;padding:15px 18px}}.note{{background:#eaf3f9;border-left:5px solid #3b7194;padding:15px 18px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.metric{{padding:16px;background:#edf5f8;border-radius:9px}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #d7e1e9;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead{{background:#edf3f7}}code{{font-size:15px}}a{{color:#155f90;overflow-wrap:anywhere}}img{{width:100%;height:auto;display:block}}.links a{{margin-right:20px}}figcaption{{font-size:14px;color:#5c7181}}li{{margin:6px 0}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}h1{{font-size:27px}}section,header{{padding:18px}}}}</style></head><body><main>
<header><div class="tag">V118 · C13 · source 선택 step1000 / logit0 · 목표 미완료</div><h1>공개 목재 가림은 개선됐지만,<br>실사 외부가림 판단은 좋아지지 않았습니다.</h1><p>같은 목재를 외부 물체로 붙인 공개 source 62쌍을 추가했습니다. 기존 공개 목재 VAL cover 정답은 <b>C12 27/35 → C13 34/35</b>로 늘었습니다. 그러나 현재 실사 평가의 외부가림 E는 <b>목재 8/29, 정사 6/61</b>로 C12와 같습니다.</p><p class="warn">source gate 15/15 PASS는 이 source 계약을 통과했다는 뜻입니다. 기존 실사 모델을 보편적으로 대체하거나 전체 목표를 해결했다는 결과가 아닙니다. C13은 시간순 최신 결과이며 현재 평가로 최선 모델을 고른 것이 아닙니다.</p><p>{link(R,'상세 결과 JSON')} · {link(V/'source_visual_followup/RESULT.json','공개 source 실제 관찰')} · {link(V/'saved_current_review/REVIEW.json','저장 현재 평가 독립 감사')}</p></header>
<section id="source"><h2>공개 source에서는 무엇이 달라졌나</h2><p>기존 C12의 96개 샘플 순서를 유지하고 같은 목재 cover/sham 16쌍을 추가해 배치128(D64/E64)로 학습했습니다. 동일한 동결 DINO1928·작은 head221,825개·초기화 seed211·2,000 updates·선택 기준 10개·logit0을 유지했습니다. 기존 BCE 질량은 0.75, 기존 hinge 질량은 1/3이 돼 재질 데이터 하나만의 인과 효과로 해석하지 않습니다.</p><p>새 448번 시도 중 62쌍만 성공했고 실패386쌍은 U로 남겼습니다. 새924행과 기존 자료를 합친 <b>46,108행을 모두 보존</b>했습니다. 성공62쌍은 14질의·8이미지·11donor에 모여 있고 모두 PALLOX입니다. LOCO 성공0, 새 자연 가림이나 독립 환경 gold가 아닙니다.</p>{table(['공개 통제 집합','모델','cover E','sham D','3상태 모두','cover BCE','margin≥2'],source_rows)}<p class="note">새 TRAIN 62쌍은 cover·sham·원 BEFORE를 모두 62/62 맞혔습니다. 이는 학습한 새 통제 관계를 맞힌 결과이며 새 VAL이 아닙니다. 기존 VAL cover는 111/119 → 118/119, 3상태 동시 정답은 106/119 → 116/119입니다. 상대 margin 통과만으로 절대 E 정답을 대신하지 않았습니다.</p></section>
<section id="visual"><h2>실제로 남은 source 오류와 개선</h2><p>이미지 읽기 전에 각 결과 조건의 원래 첫 행을 고정했습니다. 세 카드는 동일한 원 q/L의 저장 RGB96이며 새 crop·재중심·모델 호출이 없습니다. 원 무표시 패널과 q 표시 사본을 분리했습니다.</p><figure><a href="source_visual_followup/CONTACT_FIXED3_SOURCE_TRIPLES.png"><img src="source_visual_followup/CONTACT_FIXED3_SOURCE_TRIPLES.png" alt="고정된 공개 source의 남은 cover 실패, 개선, sham 오탐 triplet3개"></a><figcaption>공개 조건부 cue AUX · native 대응 UNKNOWN · 인간/자연 가림 gold 아님. 클릭하면 원 크기.</figcaption></figure><p class="links">{visual_cards}</p><ul><li><b>남은 cover 실패:</b> q 중심4가 모두 바뀌어도 C13 z=−2.5868로 D입니다. cover−sham=8.4266으로 상대 margin은 통과합니다.</li><li><b>고쳐진 cover:</b> C12 z=−1.4540에서 C13 +3.4126으로 절대 E 기준을 넘고 BEFORE/SHAM은 D를 유지했습니다.</li><li><b>sham 오탐:</b> 중심4는 그대로지만 주변 central32의 236/1024픽셀이 바뀌며 C13은 E입니다. C11/C12도 이미 이 sham을 E로 판단했습니다.</li></ul><p>단순한 목재 유무·중심 RGB 변화·상대 순서 하나로 q에 대응하는 기대 단서의 가시성을 대체하기 어렵다는 관찰입니다. 이 그림은 local96만 보여 주므로 target-context나 DINO 묘사의 영향을 분해하거나 단일 원인을 증명하지 않습니다.</p></section>
<section id="current"><h2>현재 실사: 외부가림 E 개선은 확인되지 않음</h2><p>아래는 원래 동일한 <b>paired D/E 평가 코호트</b>입니다. WOOD D616/E29, SQUARE D531/E61의 분모를 유지했습니다. H는 binary hidden 응답이며 학습된 S나 가림 원인 확정이 아닙니다.</p><div class="grid">{cms}</div><p class="tag">C13 CM: 인간 [D,E] 행 × 예측 [D,H,U] 열.</p>{table(['그룹','모델','D 정답/수','D 정밀도','D 재현율','E 정답/수','E 정밀도','E 재현율','U율'],current)}<p>C12→C13에서 목재 E29행은 21개 D·8개 E로 개별 응답도 같습니다. 정사 E61행은 <b>1개 D→E 개선과 1개 E→D 퇴행</b>이 서로 상쇄해 정답6개가 유지됐습니다. 목재 D는 새 오류3개·개선1개로 601/616 → 599/616입니다.</p><p class="note">전체223프레임·1,784슬롯은 유지됐고 유효1,739 / factual U45 / 추가 U0입니다. 이 전체 가용성 분모와 위 paired D/E 부분집합 분모는 다릅니다. v61·legacy는 학습 자료와 입력 경로가 다르고, C11/C12/C13만 같은 저장 DINO feature/operator를 공유합니다. 표는 동일 평가 행에서의 기술적 비교이지 head 한 요소의 순수 우열 실험이 아닙니다.</p></section>
<section id="geometry"><h2>기하 self 경로와 좌표 정확성은 별도</h2><p>동일 native 슬롯 파이프라인은 paired 코호트와 분모가 다릅니다. 아래는 C13 raw와 기존 초기 기하 self 경로를 더한 응답을 분리한 표입니다.</p>{table(['그룹','경로','CM [D,E] × [D,H,U]','D 재현율','E 재현율'],same)}<p>목재 same-slot 인간 S136개에서 raw hidden은 8개, 초기 기하 경로를 더한 hidden은115개입니다. 이는 별도 기하 필터의 효과이며 head가 S를 학습했다는 뜻이 아닙니다. 정사에는 human S gold가 없습니다.</p><p>기존 manual-click D·q-reference≤3px 부분집합에서는 목재153/153, 정사223/223을 D로 판단했습니다. <b>좌표가 맞는지 모델이 검증한 성능이 아닙니다.</b> 일반 q-reference 오차 구간과 E의 추정/출처 미지정 reference도 그대로 별도 보존합니다. q/L·native순서·주석·reference 변경 및 PnP0입니다.</p></section>
<section id="scope"><h2>결론의 범위와 다음 공개 source 진단</h2>{scope}<p class="warn">보임 D를 대부분 맞히는 것과 외부가림 E를 신뢰할 수 있게 분리하는 것은 다른 조건입니다. C13은 source 내 통제 관계를 개선했지만 현재 실사의 E 약점을 해결하지 못했습니다. 현재 데이터로 새 문턱·checkpoint·모델·학습 자료를 선택하지 않습니다.</p><p>다음 공개 source T/Q/F/C/P 묘사 기여 probe는 별도 진행 중인 진단입니다. 이 보고서에는 아직 결과를 가정하거나 성공으로 계산하지 않았습니다. 혼합 feature의 모델 응답은 물리 영상의 원인 증명이나 새 학습 label이 아닙니다.</p><details><summary>실행·감사·동결 근거 링크</summary>{evidence}</details></section></main></body></html>'''
out=V/'ROOT_READOUT_KO.html'
with out.open('x')as f:f.write(body)
receipt=dict(completed=True,PASS=True,source_results=bind(R),code=bind(Path(__file__)),output=bind(out),actual_new_model_pixel_feature_or_current_array_reads=0,aggregate_JSON_only=True,root_results_unchanged=True,evaluation_aggregate=bind(evaluation_summary),source_probe_reported_pending=True)
with(H/'BUILD_RECEIPT.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
