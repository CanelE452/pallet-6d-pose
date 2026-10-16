"""Measured direct-vector results and interactive report; fixed protocol, no tuning."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v18-mpl')
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context18 import *
LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'잘림10%','crop30':'잘림30%'}
NAMES={'stock':'기존 YOLO','base':'V12 점 기준','amodal':'V12 전체 복원','balanced_old':'V15 균형 후보 선택','strip':'V16 선 후보 선택','high_none':'V17 RGB 후보 선택','point_vector':'V18 점 → 보정량','line_vector':'V18 점+선 → 보정량','point_risk':'V18 점 → 보정량+악화 벌점','line_risk':'V18 점+선 → 보정량+악화 벌점','line_vector_off':'선 모델 · 선 분기 끄기','line_vector_donor':'선 모델 · 다른 영상 선 특징','line_risk_off':'벌점 선 모델 · 선 분기 끄기','line_risk_donor':'벌점 선 모델 · 다른 영상 선 특징'}
FOCUS=(('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside'))

def avg(s,a,c,t=None):
    rows=s['results']['real/'+c];rs=[rows['stock']] if a=='stock' else [rows[f'{a}_seed{i}'] for i in SEEDS]
    return float(np.mean([r['strata'][t]['mean_px'] if t else r['frame_mean_px'] for r in rs]))

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});s=read(RUN/'SUMMARY.json');d=read(RUN/'DIAGNOSIS.json');audit=read(RUN/'AUDIT.json');integ=read(RUN/'INTEGRATION_CHECK.json');coverage=read(RUN/'COVERAGE.json');gallery=read(RUN/'GALLERY.json')['rows'];assert all(x['status']=='PASS' for x in (d,audit,integ))
    assert not s['stable_arms'] and not any(v['pass_gate'] for by in s['line_gates'].values() for v in by.values())
    verdict='직접 보정은 효과 있음 · 선의 안정적 추가 이득은 확인 못함'
    lines=['# 선 특징으로 보정 방향·크기 직접 학습: V18','',verdict,'',
      '4구조 × 3시드, 24,000업데이트를 완료했다. 기존 후보 선택 대신 연속적인 2차원 보정 벡터를 출력했다. 어려운 조건에서 기존 후보 선택보다 좋아졌지만 점 특징만 쓰는 같은 구조도 거의 같은 이득을 냈다. 선을 추가했을 때의 개선과 직접 보정 구조 자체의 개선을 구분해야 한다. 모든 조건 안정성 기준을 통과한 구조와 선의 추가 개선 기준을 통과한 조건은 없었다.','',
      '## 실제 평가 결과','',
      '3시드 평균px, 작을수록 좋음. 원본은 검출 프레임별 평균의 평균, 가림/잘림은 해당 대상점 평균이다. 원래 ID와 검출 실패를 유지했다. 전체 PCK·P90·검출 실패 포함 정규화 오차는 HTML 선택표에 있다.','',
      '| 방법 | 원본 | 코너 가림 | 잘림20% 밖 | 잘림40% 밖 | 잘림10% 밖 | 잘림30% 밖 |','|---|---:|---:|---:|---:|---:|---:|']
    for a in list(NAMES)[:10]:lines.append('| '+NAMES[a]+' | '+' | '.join(f'{avg(s,a,c,t):.3f}' for c,t in FOCUS)+' |')
    lines+=['','V16 선 후보 선택과 비교하면 직접 선 보정은 코너 가림16.413→12.138px, 잘림40%26.186→20.323px로 개선됐다. 그러나 대응하는 점 보정만으로도12.077/20.292px였다. 실제 개선을 선의 고유 효과로 해석할 근거는 부족하다.','',
      '## 선이 주는 추가 효과','',
      '같은 부모 특징·초기값·배치·학습 횟수·손실에서 새 선 토큰만 유무를 바꾼 비교다. 음수는 선 추가로 개선. 실제9세션 paired bootstrap95%구간, 다중 비교 보정 없음. 사전 기준은 모든3시드에서1% 이상 개선과 구간 상한0 미만을 동시에 요구한다.','',
      '| 비교 | 조건 | 차이px | 95%구간 | 시드별 개선% |','|---|---|---:|---|---|']
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,(c,t) in zip(axes,FOCUS[1:4]):
        for j,(pair,label) in enumerate([('line_vector_vs_point_vector','일반 점 손실'),('line_risk_vs_point_risk','악화 벌점 추가')]):
            v=s['comparisons']['real/'+c][pair]['subgroups'][t];lo,hi=v['ci95'];mu=v['mean_px'];lines.append(f'| {label} | {LABELS[c]} | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | '+', '.join(f'{x:+.2f}' for x in v['per_seed_gain_percent'])+' |');ax.plot([lo,hi],[j,j],color='#2563eb');ax.scatter(mu,j,color='#2563eb')
        ax.axvline(0,color='gray',ls='--');ax.set_yticks([0,1],['일반 점 손실','악화 벌점'] if ax is axes[0] else ['','']);ax.set_ylim(1.5,-.5);ax.set_title(LABELS[c]);ax.set_xlabel('선 추가에 따른 오차 차이px');ax.grid(alpha=.2)
    fig.suptitle('새 선 분기의 추가 효과 · 3시드 평균과 실제9세션95%구간');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(RUN/'effects.png',dpi=125);plt.close(fig)
    lines+=['','일반 손실의 잘림20%와 벌점 손실의 코너 가림/잘림40%에는 작은 긍정 신호가 있다. 그러나 시드별 효과가 작거나 일관되지 않아 사전 추가 개선 기준을 통과하지 못했다. 효과가 수학적으로0이라고 입증한 결과는 아니다.','',
      '## 왜 악화되는가: 방향과 이동량 분리','',
      '보정 d, 정답까지 남은 벡터 t에 대해 제곱오차 변화는 ||d||²−2(d·t)이다. 악화된 점에서 d·t≤0이면 방향 오류, d·t>0이면 해당 방향으로 너무 멀리 이동한 경우로 분류했다. 후자는 정답을 직선상으로 지나쳤다는 뜻만이 아니라 방향 오차를 포함한 그 방향의 적정 이동량을 넘었다는 뜻이다. 아래는 선 보정 모델의3시드 평균이며, 개수의 소수는 시드 평균이다. 이동0.1px 초과는 진단용 집계 기준이며 추론 필터가 아니다.','',
      '| 조건 | 대상점 | 평균 이동px | 방향 cos | 개선점 | 악화점 | 방향 오류 | 과도 이동 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c,t in FOCUS[:4]:
        rs=[s['diagnostics']['real/'+c][f'line_vector_seed{i}'][t or 'all'] for i in SEEDS];keys=['points','mean_movement_px','mean_direction_cosine','helpful_points','harmful_points','wrong_direction_harms','overshoot_harms'];lines.append('| '+LABELS[c]+' | '+' | '.join(f'{np.mean([r[k] for r in rs]):.3f}' for k in keys)+' |')
    lines+=['','잘림20%에서는 평균557개 이동점 중357개가 악화됐고, 그중326.7개(91.5%)는 반대 방향이었다. 평균 방향cos도−0.154다. 이동량을 줄이는 것만으로 방향 판단을 해결하지 못한다. 반면 코너 가림과 잘림40%의 방향cos는0.644/0.560이고, 평균 시드 결과로9개 세션 모두 기준점 대비 개선됐다. 잘림20%는9세션 중8개가 악화되어 단일 프레임의 문제가 아니다. 수동 클릭 오차는 작은 차이에 영향을 줄 수 있으나 이번에 재주석으로 측정하지 않았다.','',
      '악화 벌점을 추가하면 잘림20% 오차16.554→14.384px, 평균 이동7.570→4.059px로 줄어든다. 그래도 기준12.372px보다 나쁘며 잘림40%는20.323→21.267px로 이득도 줄어든다. 전역적인 보수성만으로 어려운 경우와 이미 맞는 경우를 충분히 구분하지 못했다.','',
      '## 선 연결은 작동하는가','',
      '6개 선 모델 모두 최종 점 손실에서 line_project까지 유한하고0이 아닌 기울기를 확인했다. 선 토큰을 끄면 학습 샘플 좌표가 바뀌며, 선 표현의 부호를 뒤집어도 예측은 같다. 따라서 gradient가 끊겨 선이 전혀 학습되지 않는 문제는 아니다. 별도의 선 분류 손실이나 gradient surgery는 사용하지 않았다.','',
      '| 같은 일반 선 모델의 입력 개입 | 조건 | 실제 입력−교체 입력px | 95%구간 |','|---|---|---:|---|']
    for c,t in FOCUS[1:4]:
        for mode in ['off','donor']:
            v=s['comparisons']['real/'+c]['line_vector_vs_line_vector_'+mode]['subgroups'][t];lo,hi=v['ci95'];lines.append(f'| {mode} | {LABELS[c]} | {v["mean_px"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines+=['','off는 선 토큰만0으로 만들고, donor는 선을 샘플링하는 영상 특징만 다른 프레임으로 교체한다. 점 주변 특징과 점/선 위치·연결 확률은 유지한다. 일반 선 모델의 세 조건 모두 구간이0을 포함한다. 선 영상 특징의 유용한 보정 신호는 강하게 확인되지 않았다. 입력 교체는 분포 밖 개입이므로 독립적인 일반화 증명은 아니다.','',
      '## 잘린 점 주변에는 실제 영상 근거가 남는가','',
      '추가 진단으로 해당 점의63개 선 샘플이 실제 이미지와 검출 ROI 안에 놓이는지 계산했다. 이는 선이 물체 경계인지 또는 가림 뒤에 있는지의 정답이 아니라, 영상 특징을 읽을 수 있는 위치인지의 검사다.','',
      '| 조건 | 대상점 | 유효 선 샘플0개인 점 | 전체 평균 유효 샘플% |','|---|---:|---:|---:|']
    for c,t in FOCUS[:4]:
        rs=[coverage['rows'][f'{c}/seed{i}'] for i in SEEDS];lines.append(f'| {LABELS[c]} | '+ ' | '.join(f'{np.mean(v):.2f}' for v in [[r['all']['points'] for r in rs],[r['no_valid_line_samples']['points'] for r in rs],[100*r['all']['mean_valid_sample_fraction'] for r in rs]])+' |')
    lines+=['','샘플이 전부 화면 밖이면 이 분기는 그 점 주변의 실제 영상 근거를 읽지 못한다. 선 좌표·연결 확률과 부모 문맥은 여전히 입력되므로 선 정보 전체가 없다는 뜻은 아니다. 이 집계와 실패 사례는 관측 가능한 영상 근거 부족이라는 가설을 지지하지만, 이를 단독 원인으로 입증하거나 특정 방식의 해결을 검증한 것은 아니다.','',
      '## 정답과 재현 검증','',
      f'원본 수동 클릭 source/ID/유효점과8조건 좌표 이동을 원본 JSON에서 재검산했다. {audit["prediction_frame_rows"]:,}개 예측 프레임 행의 지표를 독립 재계산했고 최대 차이는{audit["metric_replay_max"]:.3g}이다. {audit["vector_frame_rows"]:,}개 검출 프레임 행의 벡터를 float64로 재구성한 최대 차이는{audit["vector_replay_max_px"]:.8f}px다. 통합12모델×6조건×4프레임 추론과 캐시의 최대 차이는{integ["max_error_px"]:.8f}px였다. 부모 체크포인트와 모든 학습 데이터/정답 해시는 유지됐다.','',
      '정답 파일과 계산이 일치한다는 확인이며 수동 클릭이 물리적으로 오차0이라는 인증은 아니다. 자동 추정점은 평가 유효점에 포함하지 않았다. 이번 지표는 이전41.49 수치와 평가 조건을 맞추지 않은 채 직접 비교할 수 없다.','',
      '## 설계와 해석 범위','',
      'YOLO 관측 → 고정 점/복원/DHT 부모 → 점 주변3×3 특징 + 해당 점에 연결된 상위3개 선의7×3 특징 → 융합 MLP → 연속2차원 보정 → 기준점에 더하기. 최종 보정 반경은 검출박스 대각선15%로 학습 전에 고정했다. 점 대조군은 새 선 토큰만0이며 부모 문맥에는 이전 선 학습의 영향이 남아 있어 완전히 선 없는 YOLO와는 다르다.','',
      '이는 후처리 필터나 후보 선택이 아니라 학습 가능한 잔차 보정 모듈을 추가한 추론 아키텍처 변경이다. YOLO와 DHT 자체는 고정했으며 전체를 end-to-end로 재학습한 실험은 아니다. 직접 벡터 모델과 이전 후보 선택 모델은 출력 공간과 구조가 함께 달라졌으므로, 후보 제한만이 과거 실패의 유일 원인이라고 단정하지 않는다.','',
      '기존8후보의 GT 최적선택과15% 연속 반경의 이론적 가능성을 별도 계산했다. 예를 들어 잘림20%의 연속 반경 GT 최적 오차 하한은 약0.130px인데 실제16.554px이므로, 이 조건의 큰 오차를 보정 반경 부족만으로 설명할 수 없다. 이 하한은 정답을 알고 매점 최적 방향을 선택한 특권적 진단이며 모델이 달성한 성능이 아니다.','',
      '합성253장/1462검출 관측/6조건에서 원래 유효점으로 학습했다. 시드별 동일 초기값·배치, AdamW3e-4, float32,2000업데이트 고정 최종 체크포인트다. 평가 실제297장·9세션·1251수동점과 합성505장은 이전부터 재사용한 DEV이며8개 인공 가림/잘림 조건이다. 독립 최종 holdout과 자연 가림 주석이 없고 실제 GT로 학습·후보·시드를 선택하지 않았다. 원본 검출 누락은 모든 방법에서 동일하다.','',
      '후속 가설은 잘린 점의 현재 위치만 중심으로 선을 읽는 대신, 남아 있는 물체 부분에서 근거를 모아 점별 방향을 추론하는 구조다. 합성/실사에서 같은 시각적 근거가 서로 다른 보정 방향을 요구하는지도 별도 확인할 필요가 있다. 현재 결과는 정보 중복이나 원리적 융합 불가능을 증명하지 않으며, 이 후속 구조의 성공을 보장하지도 않는다.','',
      '전체 안정성 기준: 모든3시드에서 원본·잘림10/20/30% 악화1% 이내, 코너 가림·잘림40%는 기존 전체 복원 이득80% 이상 유지. 네 구조 모두 실패했다. 사례는 고정 SHA 예시와 사후 최대 개선/악화 예시를 함께 제시하며 평균 성능을 대표한다고 주장하지 않는다. 청록은 실제 선 샘플 위치이지 attention 기여도 지도는 아니다. 실험 체크포인트12개는 export에 보존했으며 기존 운영 모델은 교체하지 않았다.']
    report='\n'.join(lines)+'\n';(RUN/'REPORT_KO.md').write_text(report);(ROOT/'_docs/notes/pallet_vector_v18_results.md').write_text(report)
    design='\n'.join(['# V18 보정 벡터 실험 설계','',lines[lines.index('## 설계와 해석 범위')+2],'',json.dumps(read(RUN/'PROTOCOL.json'),ensure_ascii=False,indent=2)])+'\n';(RUN/'DESIGN_KO.md').write_text(design);(ROOT/'_docs/notes/pallet_vector_v18_design.md').write_text(design)
    # Small local Markdown-to-HTML renderer for our own controlled report text.
    body=[];in_table=False
    for line in lines[1:]:
        if line.startswith('|'):
            if line.startswith('|---'):continue
            if not in_table:body.append('<div class="scroll"><table>');in_table=True
            body.append('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
        if in_table:body.append('</table></div>');in_table=False
        if line:body.append('<h2>'+html.escape(line[3:])+'</h2>' if line.startswith('## ') else '<p>'+html.escape(line)+'</p>')
    if in_table:body.append('</table></div>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V18 선 특징 보정 벡터 검증</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}select{padding:8px;margin:6px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.note{color:#596579}img{width:100%;height:auto}a{color:#0369a1}</style><h1>선 정보는 어디로, 얼마나 움직이라고 알려주는가?</h1><section><b>'''+verdict+'''</b><p>4구조 × 3시드 · 24,000업데이트 · 기존 후보 선택 대조군 · 수동 GT/벡터/통합 추론 검산 완료</p><p>코너 가림과 큰 잘림에서 직접 보정의 이득은 컸습니다. 그러나 점 특징만으로도 거의 같은 개선이 나왔고, 잘림20%에서는 반대 방향 보정으로 악화됐습니다.</p><img src="effects.png"></section><section><h2>전체 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 점PCK5</option><option value="pck10">전체 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts" class="note"></p></section><section><h2>실제 보정 화살표와 선 샘플 위치</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p class="note">seed1 · 고정 SHA 및 사후 최대 개선/악화 예시. 화살표를 증폭하지 않았습니다. 회색은 입력 밖, 청록은 선 샘플 위치이며 기여도 해석이 아닙니다.</p></section><section>'''+''.join(body)+'''<p><a href="REPORT_KO.md">보고서</a> · <a href="DESIGN_KO.md">설계</a> · <a href="SUMMARY.json">전체 수치</a> · <a href="DIAGNOSIS.json">학습 경로 진단</a> · <a href="COVERAGE.json">선 샘플 범위</a> · <a href="AUDIT.json">GT·좌표 검산</a> · <a href="INTEGRATION_CHECK.json">통합 검사</a> · <a href="PROTOCOL.json">사전 기준</a></p></section><script>'''
    # Reuse the already verified generic selectable table renderer only.
    previous=(KIT.parent/'pallet_detail_v17_fixed/report17.py').read_text();js=previous[previous.index('const params=new URLSearchParams'):previous.index("for(const id of ['split','seed','metric'])document.getElementById(id).onchange")]
    page+='const data='+json.dumps(s['results'],ensure_ascii=False)+',labels='+json.dumps(LABELS,ensure_ascii=False)+',names='+json.dumps(NAMES,ensure_ascii=False)+',gallery='+json.dumps(gallery,ensure_ascii=False)+';\n'+js
    page+="""for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · '+r.harm_type;}sel.onchange=show;show();</script></html>"""
    (RUN/'index.html').write_text(page);write(RUN/'CONCLUSION.json',dict(status='FULL_STABILITY_FAIL',stable_arms=s['stable_arms'],line_gates=s['line_gates'],direct_vector_hard_condition_benefit=True,line_incremental_stability_confirmed=False,adoption=False,no_real_tuning=True));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY.json'),source_sha256=sha(Path(__file__)),figures=len(gallery)))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)

if __name__=='__main__':main()
