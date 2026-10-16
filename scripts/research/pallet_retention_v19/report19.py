"""Report initial retention experiment and explicitly exploratory BCE controls."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v19-mpl')
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context19 import *
from eval_bce19 import ARMS as ALL_ARMS
LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'잘림10%','crop30':'잘림30%'}
NAMES={'stock':'기존 YOLO','base':'V12 점 기준','amodal':'V12 전체 복원','strip':'V16 선 후보 선택','point_vector':'V18 점 직접 보정','line_vector':'V18 점+선 직접 보정','line_risk':'V18 악화 벌점 보정','fixed_half':'고정50% 적용','local_gate':'국소 적용률 · L1','area_gate':'남은 영역 적용률 · L1','line_gate':'남은 선 적용률 · L1','local_bce':'국소 적용률 · BCE(후속)','area_bce':'남은 영역 적용률 · BCE(후속)','line_bce':'남은 선 적용률 · BCE(후속)','area_gate_off':'영역 L1 · 추가 근거 끄기','area_gate_donor':'영역 L1 · 다른 영상 특징','line_gate_off':'선 L1 · 추가 근거 끄기','line_gate_donor':'선 L1 · 다른 영상 특징','area_bce_off':'영역 BCE · 추가 근거 끄기','area_bce_donor':'영역 BCE · 다른 영상 특징','line_bce_off':'선 BCE · 추가 근거 끄기','line_bce_donor':'선 BCE · 다른 영상 특징'}
FOCUS=(('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside'))

def avg(s,a,c,t=None):
    rows=s['results']['real/'+c];rs=[rows['stock']] if a=='stock' else [rows[f'{a}_seed{i}'] for i in SEEDS]
    return float(np.mean([r['strata'][t]['mean_px'] if t else r['frame_mean_px'] for r in rs]))

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});s=read(RUN/'SUMMARY_ALL.json');ds=[read(RUN/'DIAGNOSIS.json'),read(RUN/'DIAGNOSIS_BCE.json')];aud=read(RUN/'AUDIT_ALL.json');ins=[read(RUN/'INTEGRATION_CHECK.json'),read(RUN/'INTEGRATION_CHECK_BCE.json')];gallery=read(RUN/'GALLERY.json')['rows'];assert all(x['status']=='PASS' for x in ds+[aud]+ins)
    assert not s['stable_arms'] and not any(v['pass_gate'] for by in s['evidence_gates'].values() for v in by.values())
    verdict='부분 개선 · 큰 이득 유지와 약한 잘림 악화 제거의 동시 달성은 실패'
    lines=['# V19: 좋은 보정은 유지하고 잘못된 보정은 줄일 수 있는가','',verdict,'',
      '고정 V18 선 보정기에 점별 연속 적용률 모듈을 추가했다. 국소 특징/남은 영상 영역/남은 선 구간의3구조 × 3시드를 먼저 비교했다. 학습에서도 적용률이1 근처에 붙는 문제가 보여, 같은 초기값·배치·구조의 BCE 손실3구조 × 3시드를 추가했다. 합계18모델/36,000업데이트이며, BCE는 초기 결과를 본 뒤 시작한 탐색적 후속 실험이다.','',
      '## 실제 평가 결과','',
      '3시드 평균px, 작을수록 좋음. 원본은 검출 프레임 평균, 가림/잘림은 해당 대상점 평균이다. 원래 ID·GT·검출 누락을 유지했다. 전체 프레임/P90/PCK/검출 실패 포함 정규화 오차는 아래 선택표에서 비교한다.','',
      '| 방법 | 원본 | 코너 가림 | 잘림20% 밖 | 잘림40% 밖 | 잘림10% 밖 | 잘림30% 밖 |','|---|---:|---:|---:|---:|---:|---:|']
    for a in list(NAMES)[:14]:lines.append('| '+NAMES[a]+' | '+' | '.join(f'{avg(s,a,c,t):.3f}' for c,t in FOCUS)+' |')
    lines+=['','남은 선 구간을 쓰는 초기 L1 적용률 모델은 잘림20% 오차를16.554→15.343px로 줄였고 코너 가림12.138→12.156px, 잘림40%20.323→20.735px로 큰 개선을 대부분 유지했다. 이 모델은 어려운 두 조건의 기존 이득90% 보존을 모든 시드에서 통과했다. 하지만 잘림20% 기준점12.372px보다 여전히 나쁘고 원본/잘림10/30%도 일부 시드에서 제한을 넘어서 전체 안정성은 실패했다.','',
      '후속 BCE 선 모델은 잘림20%를14.324px까지 줄였으나 코너 가림은12.963px로 악화됐다. 큰 잘림은20.751px다. 원래 V18의 큰 이득 대부분은 남지만, 가림 이득90% 유지와 약한 잘림 악화1% 이내를 동시에 만족시키지 못했다. 고정50% 적용도 잘림20%14.199px까지 낮추지만 코너 가림14.078px로 좋은 보정까지 줄인다.','',
      '## 선의 추가 효과와 손실 변경 효과를 분리','',
      '음수는 개선이다. 실제9세션 paired bootstrap95%구간이며 다중 비교 보정이 없다. 새 근거의 추가 개선 기준은 모든3시드에서1% 이상 개선과 구간 상한0 미만을 동시에 요구한다. 이를 통과한 조건은 없었다. 초기 실험과 후속 실험을 모두 보고하고 실사로 시드나 체크포인트를 선택하지 않았다.','',
      '| 비교 | 조건 | 오차 차이px | 95%구간 | 시드별 개선% |','|---|---|---:|---|---|']
    pairs=[('line_gate_vs_area_gate','선−영역 · 초기 L1'),('line_bce_vs_area_bce','선−영역 · 후속 BCE'),('line_bce_vs_line_gate','BCE−L1 · 선 모델')];fig,axes=plt.subplots(1,3,figsize=(14,5))
    for ax,(c,t) in zip(axes,FOCUS[1:4]):
        for j,(pair,label) in enumerate(pairs):
            v=s['comparisons']['real/'+c][pair]['subgroups'][t];lo,hi=v['ci95'];mu=v['mean_px'];lines.append(f'| {label} | {LABELS[c]} | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | '+', '.join(f'{x:+.2f}' for x in v['per_seed_gain_percent'])+' |');ax.plot([lo,hi],[j,j],color='#2563eb');ax.scatter(mu,j,color='#2563eb')
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(3),[p[1].replace('−','-') for p in pairs] if ax is axes[0] else ['']*3);ax.set_ylim(2.5,-.5);ax.set_title(LABELS[c]);ax.set_xlabel('오차 차이px · 음수가 개선');ax.grid(alpha=.2)
    fig.suptitle('선 특징과 손실 변경의 효과 · 3시드 평균 / 실제9세션95%구간');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(RUN/'effects.png',dpi=125);plt.close(fig)
    lines+=['','BCE 전환은 잘림20%에서 L1보다1.020px 개선했고 구간은[−1.583,−0.529]px였다. 반면 코너 가림에서는0.806px 악화했으며 구간은[+0.486,+1.142]px였다. 선과 영역을 동일 손실에서 비교한 효과는 작거나 조건/시드에 따라 바뀌었다. 따라서 손실 변경의 개선을 선의 고유한 이득으로 보지 않는다.','',
      '## 모델은 좋은 보정과 나쁜 보정을 구분했는가','',
      '보정 적용 전 고정 V18 벡터가 기준점보다 오차를0.1px 넘게 낮추면 좋은 보정,0.1px 넘게 높이면 나쁜 보정으로 사후 집계했다. 이 기준과 GT는 평가용이며 추론의 필터나 입력으로 사용하지 않는다. 아래 적용률은3시드 평균이다. 적용률은 정답일 확률이 아니다.','',
      '| 조건 | 모델 | 나쁜 보정 적용률 | 좋은 보정 적용률 | 전체 평균 | GT 최적 평균(진단) |','|---|---|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(c,t) in zip(axes,FOCUS[1:4]):
        for j,arm in enumerate(('line_gate','line_bce')):
            rs=[s['diagnostics']['real/'+c][f'{arm}_seed{i}'][t] for i in SEEDS];vals=[np.mean([r[k] for r in rs]) for k in ('alpha_on_original_harmful','alpha_on_original_helpful','mean_alpha','oracle_alpha')];lines.append('| '+LABELS[c]+' | '+NAMES[arm]+' | '+' | '.join(f'{v:.1%}' for v in vals)+' |');ax.bar(np.array([0,1])+j*.33,vals[:2],width=.3,label='초기 L1' if j==0 else '후속 BCE')
        ax.set_xticks([.165,1.165],['나쁜 보정','좋은 보정']);ax.set_ylim(0,1.07);ax.set_title(LABELS[c]);ax.set_ylabel('평균 적용률');ax.legend(fontsize=8)
    fig.suptitle('포화는 줄었지만 좋은/나쁜 보정의 적용률 차이는 작음');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(RUN/'alpha.png',dpi=125);plt.close(fig)
    lines+=['','잘림20%에서 초기 L1은 나쁜 보정에도91.9%, 좋은 보정에는99.2%를 적용했다. BCE는 각각61.2%/65.9%였다. BCE가 나쁜 보정을 더 줄이지만 좋은 보정도 비슷하게 줄인다. 잘림40%의 BCE는 오히려 나쁜 보정73.1%, 좋은 보정71.2%다. 큰 이득과 오류 억제를 동시에 달성하려면 단순히 전체 적용률을 낮추는 것보다 이 구분이 좋아져야 한다.','',
      '## 포화 가설에서 확인한 것과 남은 것','',
      '초기 손실은 ||d||·|alpha−alpha_star|이며 alpha=sigmoid(logit)이다. logit에 대한 기울기에 alpha(1−alpha)가 곱해져 잘못된100% 근처에서도 약해질 수 있다. 후속은 같은 이동량 가중치를 둔 BCE-with-logits이며 기울기는 가중치×(alpha−alpha_star)다. soft target은 원래 합성 GT로 계산한 최적 적용률이다.','',
      '후속에서 포화는 줄고 잘림20% 악화도 완화됐다. 그러나 BCE는 조건부 최적값의 학습 목표도 바꾸므로 이것만으로 포화가 유일 원인이었다고 증명할 수 없다. 학습 데이터에서도 초기 모델은 나쁜 보정에 높은 적용률을 주었다. 실사/합성 차이만으로 실패를 설명하기도 어렵다. 모델 표현·조건부 정보·학습량/목표의 한계는 여전히 남아 있다.','',
      '| 학습 데이터의 화면 밖 점 | 나쁜 보정 적용률 | 좋은 보정 적용률 | 최종 오차px |','|---|---:|---:|---:|']
    for arm,diag in [('line_gate',ds[0]),('line_bce',ds[1])]:
        rs=[diag['training'][f'{arm}_seed{i}']['outside'] for i in SEEDS];lines.append('| '+NAMES[arm]+' | '+ ' | '.join(f'{np.mean([r[k] for r in rs]):.4f}' for k in ('mean_alpha_bad','mean_alpha_good','after_px'))+' |')
    lines+=['','## 실제 남은 선을 읽는 경로는 동작하는가','',
      '추가 근거가 있는12개 모델에서 최종 점 관련 손실이 영상 투영층까지 유한하고0이 아닌 기울기를 전달했다. 추가 토큰을 끄면 예측이 변하고, 같은 선의 부호를 바꾸어도 예측은 같다. 수평·수직·대각선 교차와 ROI를 만나지 않는 선을 독립 검사했다. 실제 평가의 선 샘플도 원래 선 위에 놓이는지 별도 float64로 확인했다.','',
      'area는 검출박스와 원래 이미지가 겹치는 사각형에서6×6개를 읽는다. line은 해당 점에 연결된 상위3개 DHT 선과 같은 사각형의 교차 구간에서12개씩 읽는다. 선이 겹치는 영역을 지나지 않으면 그 선은 마스킹된다. 두 구조 모두36샘플과 같은 투영층/attention/적용률 헤드를 사용한다. 물리적으로 올바른 파렛트 선이나 가림 뒤의 실제 표면이라는 뜻은 아니다.','',
      '| 조건 | 선 샘플 평균 유효 비율 | 선 샘플0개인 대상점(시드 평균) |','|---|---:|---:|']
    for c,t in FOCUS[:4]:
        rs=[r for r in ds[0]['clipped_line_geometry'] if r['condition']==c];lines.append(f'| {LABELS[c]} | {np.mean([r["mean_coverage"] for r in rs]):.1%} | {np.mean([r["no_valid_line_samples"] for r in rs]):.1f} |')
    lines+=['','같은 모델에서 추가 근거를0으로 만들거나 그 영상 특징만 다른 프레임 것으로 바꾼 결과도 전체 수치에 보존했다. 점 패치·좌표·고정 보정 벡터·선 위치와 연결 확률은 유지했다. 이 입력 교체는 분포 밖 개입이며, 유용한 실제 선 의미를 입증하는 충분조건은 아니다.','',
      '## 정답·수치·실험 공정성 검증','',
      f'실제 원본 수동 클릭의 source/ID/유효점과8조건 좌표 이동을 재검산했다. {aud["prediction_frame_rows"]:,}개 예측 프레임 행의 지표를 독립 재계산했고 최대 차이는{aud["metric_replay_max"]:.3g}이다. 적용률을 쓰는{aud["vector_frame_rows"]:,}개 검출 프레임 행에서 p0+alpha*d를 float64로 재구성한 최대 차이는{aud["retention_replay_max_px"]:.8f}px다. 모든 alpha는[0,1]이고 기준점/보정 벡터는 기존 V18과 동일했다. 통합18모델×6조건×4프레임의 최대 좌표 차이는{max(x["max_error_px"] for x in ins):.8f}px였다.','',
      '초기 내보내기에서 공통 평가기가 요구하는 기존 YOLO 점 메타데이터 original_points가 빠져 채점이 중단됐다. 동일 불변 캐시에서 이 필드만 보완했고 예측·적용률·ID·검출 배열이 모두 그대로임을 확인했다. 초기 소스 스냅샷과 변경 기록을 남겼으며 학습 코드/가중치나 GT를 수정한 것은 아니다. SUMMARY는 보완 후 처음 생성됐다.','',
      '수동 GT와 계산이 일치한다는 검증이며 클릭의 물리적 오차0을 보장하지 않는다. 실제297장·9세션·1251수동점과 합성505장은 기존에 재사용한 DEV다. 자연 가림 주석이나 새 최종 holdout은 확보하지 않았다. 손실 후속은 탐색적 결과로 해석해야 한다. 자동 생성점은 실제 유효 GT에 포함하지 않았고 검출 누락은 모든 방법에서 동일하다.','',
      '## 아키텍처와 판정','',
      '고정 YOLO 관측 → 고정 점/DHT/전체 복원 부모 → 고정 V18 보정 d. 별도 국소/남은 영역/남은 선 특징 → 학습 적용률 alpha. 최종점=p0+alpha*d. 기존 점을 잇는 방식, GT 필터, self-training은 사용하지 않았다. 학습 가능한 모듈을 추가한 추론 아키텍처 변경이며 YOLO/DHT 전체를 end-to-end로 재학습한 실험은 아니다.','',
      '국소 대조군도 이전 부모의 선 학습 영향을 공유한다. 따라서 새 선 분기 유무를 검사하는 실험이며 완전히 선 없는 YOLO와의 비교는 아니다. 원래 합성253장/1462검출 관측의6조건을 사용했고 매스텝3원본의 모든 가용 변형을 묶었다. 유효점을 버리지 않았다. 동일 초기값/배치/2000업데이트, AdamW3e-4, float32, 마지막 체크포인트를 사용했다. 동결 V18이 이미 본 합성 데이터의 잔차를 사용하므로 별도 평가와 구분해야 한다.','',
      '기준은 모든3시드에서 원본·잘림10/20/30% 악화1% 이내와 코너 가림·잘림40%의 V18 이득90% 이상 유지다. 초기3구조와 후속3구조 모두 전체 기준에 실패했다. GT가 매점 최적 적용률을 알려주는 oracle의 실사 오차는 코너 가림11.56/잘림20%10.97/잘림40%15.72px이지만 이는 정답을 쓰는 기하학적 가능성일 뿐 학습 모델의 결과가 아니다.','',
      '이번에 얻은 것은 기존 큰 개선을 대부분 보존하는 적용률 구조와 일부 악화 완화다. 아직 해결하지 못한 것은 좋은 보정과 반대 방향 보정의 구분이다. 입력 범위 확대나 sigmoid 포화만 고치면 된다는 가설은 충분하지 않았다. 이 결과를 모든 점·선 융합의 불가능으로 확대하지 않는다. 추가 구조를 배포하지 않았고 실험용 통합 체크포인트18개를 export에 보존했다.']
    report='\n'.join(lines)+'\n';(RUN/'REPORT_KO.md').write_text(report);(ROOT/'_docs/notes/pallet_retention_v19_results.md').write_text(report)
    design='# V19 적용률 아키텍처 설계\n\n초기 계획과 실패 진단 후 추가한 BCE 대조군을 구분한다.\n\n```json\n'+json.dumps(read(RUN/'PROTOCOL.json'),ensure_ascii=False,indent=2)+'\n```\n\n## 탐색적 손실 대조군\n\n```json\n'+json.dumps(read(RUN/'BCE_PROTOCOL.json'),ensure_ascii=False,indent=2)+'\n```\n';(RUN/'DESIGN_KO.md').write_text(design);(ROOT/'_docs/notes/pallet_retention_v19_design.md').write_text(design)
    body=[];inside=False
    for line in lines[1:]:
        if line.startswith('|'):
            if line.startswith('|---'):continue
            if not inside:body.append('<div class="scroll"><table>');inside=True
            body.append('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
        if inside:body.append('</table></div>');inside=False
        if line:body.append('<h2>'+html.escape(line[3:])+'</h2>' if line.startswith('## ') else '<p>'+html.escape(line)+'</p>')
    if inside:body.append('</table></div>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V19 좋은 보정 유지와 오류 억제 검증</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}select{padding:8px;margin:6px;font-size:15px;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.note{color:#596579}img{width:100%;height:auto}a{color:#0369a1}</style><h1>큰 보정 이득을 유지하면서 오류를 막을 수 있는가?</h1><section><b>'''+verdict+'''</b><p>기존 보정기 고정 + 점별 적용률 · 국소/남은 영역/남은 선 비교 · 초기9모델 + 탐색적 손실 대조9모델</p><p>선 적용률 L1은 큰 이득을 대부분 유지하며 잘림20%를16.55→15.34px로 줄였습니다. BCE는14.32px까지 줄였지만 코너 가림 이득 일부를 잃었습니다. 기준점12.37px보다 여전히 나쁩니다.</p><p><b>최종점 = 기준점 + 학습 적용률 × 고정 V18 보정 벡터</b></p><img src="effects.png"></section><section><h2>좋은 보정과 나쁜 보정에 주는 적용률</h2><img src="alpha.png"><p class="note">정답으로 사후 구분한 평가 진단입니다. 적용률은 정답일 확률이 아닙니다. BCE는 포화를 줄이지만 두 집단을 충분히 구분하지 못했습니다.</p></section><section><h2>전체 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 점PCK5</option><option value="pck10">전체 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts" class="note"></p></section><section><h2>원본 GT · 실제 선 샘플 · 보정 적용률과 화살표</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p class="note">고정seed1 · SHA 예시와 사후 최대 개선/악화 사례를 구분합니다. 평균 성능을 대표하는 예시가 아니며 화살표를 증폭하지 않았습니다.</p></section><section>'''+''.join(body)+'''<p><a href="REPORT_KO.md">보고서</a> · <a href="DESIGN_KO.md">설계</a> · <a href="SUMMARY_ALL.json">전체 수치</a> · <a href="SUMMARY.json">초기 결과</a> · <a href="DIAGNOSIS.json">초기 학습 진단</a> · <a href="DIAGNOSIS_BCE.json">후속 학습 진단</a> · <a href="AUDIT_ALL.json">GT·좌표 검산</a> · <a href="PROTOCOL.json">초기 사전 기준</a> · <a href="BCE_PROTOCOL.json">탐색적 후속 계획</a> · <a href="EXPORT_METADATA_REPAIR.json">평가 메타데이터 보완 기록</a></p></section><script>'''
    previous=(KIT.parent/'pallet_detail_v17_fixed/report17.py').read_text();js=previous[previous.index('const params=new URLSearchParams'):previous.index("for(const id of ['split','seed','metric'])document.getElementById(id).onchange")];page+='const data='+json.dumps(s['results'],ensure_ascii=False)+',labels='+json.dumps(LABELS,ensure_ascii=False)+',names='+json.dumps(NAMES,ensure_ascii=False)+',gallery='+json.dumps(gallery,ensure_ascii=False)+';\n'+js
    page+="""for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · L1 적용 '+(100*r.alpha_l1).toFixed(1)+'% / BCE 적용 '+(100*r.alpha_bce).toFixed(1)+'%';}sel.onchange=show;show();</script></html>"""
    (RUN/'index.html').write_text(page);write(RUN/'CONCLUSION.json',dict(status='PARTIAL_IMPROVEMENT_FULL_STABILITY_FAIL',stable_arms=s['stable_arms'],evidence_gates=s['evidence_gates'],initial_line_gate_retains_hard_gains_all_seeds=True,bce_is_exploratory=True,adoption=False));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY_ALL.json'),source_sha256=sha(Path(__file__)),figures=len(gallery)));print('REPORT COMPLETE',RUN/'index.html',flush=True)

if __name__=='__main__':main()
