"""Report matched data/optimizer factors without selecting a real-data winner."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v21-mpl')
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context21 import *
from eval21 import METHODS,FACTORS
LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'잘림10%','crop30':'잘림30%'}
NAMES=dict(zip(METHODS,('기존 YOLO','V12 점 기준','V18 점+선 전체 보정','V20 맞는 점 증강','기존 잔차 · 일반학습','기존 잔차 · PCGrad','새 잔차 · 일반학습','새 잔차 · PCGrad','새 잔차 · 일반학습 · 국소')))
FOCUS=(('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside'))

def avg(s,split,a,c,t=None):
    r=s['results'][split+'/'+c];rs=[r['stock']] if a=='stock' else [r[f'{a}_seed{i}'] for i in SEEDS]
    return float(np.mean([x['strata'][t]['mean_px'] if t else x['frame_mean_px'] for x in rs]))

def main():
    s=read(RUN/'SUMMARY.json');d=read(RUN/'DIAGNOSIS.json');a=read(RUN/'AUDIT.json');train=read(RUN/'TRAINING_COMPLETE.json')
    assert all(read(RUN/f)['status']=='PASS' for f in ('AUDIT.json','DIAGNOSIS.json','INTEGRATION_CHECK.json','SPLIT_AUDIT.json','TARGET_AUDIT.json','REPRODUCTION.json'))
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});verdict='안정성 통과: '+', '.join(s['stable_arms']) if s['stable_arms'] else '전체 안정성 미달: 데이터 변경과 PCGrad만으로 동시 해결하지 못함'
    lines=['# V21 새 잔차 데이터와 gradient surgery 검증','',verdict,'',
      '동일 적용률 아키텍처에서 기존/새 합성 잔차 × 일반학습/PCGrad의2×2와 새 데이터 국소 대조군을 비교했다. 총5설정×3시드=15모델,30,000업데이트다. 데이터와 optimizer 이외의 모델 구조·초기값·손실식·업데이트 수를 유지했다.','']
    lines+=(RUN/'INTERPRETATION_KO.md').read_text().splitlines()
    lines+=['','## 평가 결과','',
      '3시드 평균px, 낮을수록 좋다. 원본은 검출 프레임 평균, 나머지는 해당 가림/잘림 대상점 평균이다. 검출 누락은 모든 방법에서 동일하며 누락을 포함한 정규화 오차/P90/PCK는 전체 선택표에 제공한다.']
    for split,title in [('real','기존 실제 DEV297장'),('synth_val','기존 합성 DEV505장'),('new_check','새 보정기 확인용 합성253장')]:
        focus=FOCUS if split!='new_check' else FOCUS[:4];lines+=['','### '+title,'','| 방법 | '+' | '.join(LABELS[c]+(' 대상점' if t else '') for c,t in focus)+' |','|---|'+'---:|'*len(focus)]
        for arm in METHODS:lines.append('| '+NAMES[arm]+' | '+' | '.join(f'{avg(s,split,arm,c,t):.3f}' for c,t in focus)+' |')
    lines+=['','## 데이터 변경과 PCGrad의 효과 분리','',
      '음수는 개선. 실제9세션 paired bootstrap5,000회95%구간이며 다중 비교 보정은 없다. 추가 개선 판정은 각 시드1% 이상 개선 및 구간 상한0 미만을 함께 요구한다.','',
      '| 비교 | 조건 | 오차 차이px | 95%구간 | 시드별 개선% | 판정 |','|---|---|---:|---|---|---|']
    pairlabels=['기존 잔차: PCGrad - 일반','일반학습: 새 잔차 - 기존','새 잔차: PCGrad - 일반','PCGrad: 새 잔차 - 기존','새 잔차: 선 - 국소']
    fig,axes=plt.subplots(1,3,figsize=(14,5.5))
    for ax,(c,t) in zip(axes,FOCUS[1:4]):
        for j,((x,y),label) in enumerate(zip(FACTORS,pairlabels)):
            pair=x+'_vs_'+y;v=s['comparisons']['real/'+c][pair]['subgroups'][t];lo,hi=v['ci95'];mu=v['mean_px'];passed=s['evidence_gates'][pair][c]['pass_gate'];lines.append('| '+label+' | '+LABELS[c]+f' | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | '+', '.join(f'{z:+.2f}' for z in v['per_seed_gain_percent'])+' | '+('통과' if passed else '미통과')+' |');ax.plot([lo,hi],[j,j],color='#2563eb');ax.scatter(mu,j,color='#2563eb')
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(5),pairlabels if ax is axes[0] else ['']*5);ax.set_ylim(4.5,-.5);ax.set_title(LABELS[c]);ax.set_xlabel('오차 차이px · 음수가 개선');ax.grid(alpha=.2)
    fig.suptitle('새 데이터와 PCGrad의 효과 · 실제9세션95%구간');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(RUN/'effects.png',dpi=125);plt.close(fig)
    lines+=['','## Gradient surgery는 무엇에 적용했는가','',
      '현재 부모 YOLO/DHT/점 보정기는 동결돼 있다. 학습 가능한 적용률 모델의 원래 목적은 이동량 가중 BCE 하나다. 그대로는 점 손실과 선 손실 사이에 PCGrad를 적용할 두 목적이 없다. 따라서 합성 GT 최적 적용률alpha*≤0.05인 멈춤 그룹과 나머지 이동 그룹으로 원래 손실을 분해했다. 두 항 모두 전체 유효점 수로 나누므로 더하면 원래 손실과 같다. 그룹별 평균으로 바꿔 클래스 가중치를 변경하지 않았다.','',
      '각 그룹 기울기를 g0,g1이라 할 때 내적이 음수면 g0에서 g1 방향 성분을, g1에서 g0 방향 성분을 제거한 뒤 합한다. 각 투영은 수정 전 상대 기울기를 사용한다. 비충돌/빈 그룹/영 기울기에서는 일반 합을 사용한다. 원래 gradient sum과의 동등성, 알려진 벡터의 투영값, 실제 모델 기울기의 별도 float64 투영 계산을 검산했다. 참고: https://arxiv.org/abs/2001.06782 및 저자 구현 https://github.com/tianheyu927/PCGrad .','',
      '이는 멈춤/이동 그룹에 적용한 PCGrad다. 점/선 보조 손실을 함께 학습하는 전체 네트워크의 PCGrad 실험으로 확대해 해석하지 않는다. 서로 다른 정답을 가진 그룹의 충돌은 흔할 수 있으며, 높은 충돌률만으로 성능 악화의 원인이 증명되지는 않는다. 아래 값은 PCGrad 투영 전 원래 그룹 기울기다.','',
      '| 학습 설정 | 충돌률 | 평균 cosine |','|---|---:|---:|']
    for arm in ARMS:
        rs=[x for x in train['runs'] if x['arm']==arm];lines.append('| '+NAMES[arm]+' | '+f'{np.mean([x["sampled_conflict_fraction"] for x in rs]):.1%} | {np.mean([x["sampled_mean_cosine"] for x in rs]):.3f} |')
    lines+=['','각 실행2000업데이트 중 동일 간격100배치를 집계했다. PCGrad는 매 업데이트에 적용했고 일반학습은 진단만 수행했다. 진단을 추가한 기존 일반학습은 V20 방향 정규화·가중 BCE 결과를 모든 학습 관측에서0.001px 이내로 재현했다.','',
      '## 새 잔차는 실제 필요한 보정량에 가까워졌는가','',
      '실제 화면 밖 유효점의8개 ID 비중으로 사후 구성비를 맞췄다. 이 가중치나 실제 GT는 학습에 쓰지 않았다. 자세/크기/질감/문맥까지 맞춘 것은 아니다.','',
      '| 조건 | 데이터 | 기준점 오차px | 거의 멈춰야 하는 비율 | GT 최적 적용률 |','|---|---|---:|---:|---:|']
    for c in ('crop20','crop40'):
        rs=[x for x in d['population'] if x['condition']==c]
        for pop,label in [('old_train','기존 학습'),('new_train','새 학습'),('new_check','새 확인'),('real','실제 DEV')]:
            val={k:np.mean([r['point_id_matched_means'][pop][k] for r in rs]) for k in ('base_error_px','stop_fraction','alpha')};lines.append(f'| {LABELS[c]} | {label} | {val["base_error_px"]:.3f} | {val["stop_fraction"]:.1%} | {val["alpha"]:.1%} |')
    lines+=['','## 실제에서 좋은 보정과 나쁜 보정을 구분했는가','',
      '원래 벡터를100% 적용하면 기준점 오차가0.1px 넘게 줄어드는 경우를 좋은 보정,0.1px 넘게 늘어나는 경우를 나쁜 보정으로 사후 집계했다. 적용률은 정답 확률이 아니다.','',
      '| 조건 | 학습 설정 | 나쁜 보정 적용률 | 좋은 보정 적용률 |','|---|---|---:|---:|']
    for c,t in FOCUS[1:4]:
        for arm in ARMS:
            rs=[s['diagnostics']['real/'+c][f'{arm}_seed{i}'][t] for i in SEEDS];lines.append('| '+LABELS[c]+' | '+NAMES[arm]+' | '+ ' | '.join(f'{np.mean([r[k] for r in rs]):.1%}' for k in ('alpha_on_original_harmful','alpha_on_original_helpful'))+' |')
    lines+=['','## 데이터 분리와 실험 범위','',
      '원래 YOLO train 및 모든 기존 보정기 train/calibration/synth_val의 원본 pair_group_id를 제외한 후보에서 GT/예측 성능을 보지 않고 SHA 순서로506그룹을 골랐다. 253장 학습/253장 확인으로 분리했다. 선택된506개 이미지 해시는 고유하며 원래 학습·기존 평가56,492개 이미지의 파일 해시와도 겹치지 않는다. 자산·연속 장면까지 독립이라는 뜻은 아니다.','',
      '새 이미지는 원래 YOLO 검증 풀에서 가져왔다. 가중치 학습의 원본 그룹과는 분리됐지만 과거 YOLO 체크포인트 선택에 영향을 줬을 수 있다. 새 확인셋은 이번 적용률 모델에 대한 사전 분리 평가이고, 전체 시스템의 완전히 손대지 않은 최종 holdout이라고 부르지 않는다. 기존 실제/합성 평가셋도 재사용DEV다.','',
      '원본 이미지에서 반사100px 전처리와 최고 confidence≥0.25 검출 정책으로 YOLO를 다시 실행했다. clean/중앙 가림20/40%/잘림20/40%/코너 가림의6조건에서 모든 동결 부모의 점·선·전체 복원·문맥·보정 벡터를 함께 새로 계산했다. GT로 검출을 고르거나 기준점을 옮기지 않았다. 기존1462관측/새1447관측이며 둘 다253원본이다. 같은 원본3장씩의 변형을 묶어2000회 업데이트했지만 검출 수 차이 때문에 총 관측 forward 수는 약간 다르다.','',
      '새 데이터 변경은 미학습 여부뿐 아니라 장면·구도·자산 구성 변화도 포함한다. 따라서 모든 차이를 미학습 잔차 효과 하나로 인과적으로 귀속하지 않는다. 원래 데이터를 버리고 새 풀에서 학습하는 대조이며, 두 풀을 합치는 실험은 아니다. 새 국소 대조군도 고정 부모의 선 정보를 공유하므로 추가 선 토큰의 효과를 검사한다.','',
      '## GT와 수치 검증','',
      f'새506장의 원래 YOLO 점 ID·가시성·패딩 좌표를 유지하고 렌더러3D 투영의 점 집합과0.01px 이내 일치함을 검산했다. 이 대응은 GT 검사에만 쓰며 추론 점 번호를 바꾸지 않는다. 실제 수동 클릭 source/유효점/원본 좌표와 가림·잘림 좌표 이동도 유지했다. {a["prediction_frame_rows"]:,}예측 프레임 행의 지표 재계산 최대 차이는{a["metric_replay_max"]:.3g}, 보정 재구성 최대 오차는{a["retention_replay_max_px"]:.8f}px다.','',
      '원래 데이터/체크포인트는 변경하지 않았다. 새 통합15모델을 실제6조건×4프레임씩 캐시와 비교해0.001px 이내 일치를 확인했다. 학습30,000업데이트의 손실·기울기는 유한했다. 수동 GT를 코드가 정확히 사용하는지 검사했으며 클릭의 물리적 오차0을 보증하지 않는다. 자연 가림 주석이나 새 실제 최종 holdout은 확보하지 않았다. 운영 모델은 교체하지 않았다.','',
      '## 아키텍처와 남은 한계','',
      '최종점=p0+sigmoid(g(점 주변 특징, 부모 문맥, 선 특징, 보정 방향/크기))×고정 보정 벡터d. 이번 변경은 이 모듈의 학습 데이터와 optimizer다. 새로운 선 연결 방식이나 YOLO/DHT 전체 end-to-end 재학습은 아니다. 적용률0~1은 잘못된 벡터를 멈출 수 있으나 새 방향으로 회전시키지는 못한다.','',
      '선 분기는 상위3개 예측 선과 화면/검출박스 교차 구간의36개 샘플을 읽는다. 화면 안에 있어도 가림 영역이나 배경이 포함될 수 있다. 기하학적으로 유효한 샘플이 실제 파렛트 경계라는 뜻은 아니다. 이번 데이터/PCGrad 대조는 이 선 가시성 문제를 직접 수정한 실험이 아니다. 결과를 모든 점·선 융합이나 모든 gradient surgery 방식의 불가능으로 일반화하지 않는다.']
    report='\n'.join(lines)+'\n';(RUN/'REPORT_KO.md').write_text(report);(ROOT/'_docs/notes/pallet_residual_v21_results.md').write_text(report)
    design='# V21 사전 설계\n\n```json\n'+json.dumps(read(RUN/'PROTOCOL.json'),ensure_ascii=False,indent=2)+'\n```\n';(RUN/'DESIGN_KO.md').write_text(design);(ROOT/'_docs/notes/pallet_residual_v21_design.md').write_text(design)
    body=[];inside=False
    for line in lines[1:]:
        if line.startswith('|'):
            if line.startswith('|---'):continue
            if not inside:body.append('<div class="scroll"><table>');inside=True
            body.append('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
        if inside:body.append('</table></div>');inside=False
        if line:body.append('<h2>'+html.escape(line.lstrip('# '))+'</h2>' if line.startswith('#') else '<p>'+html.escape(line)+'</p>')
    if inside:body.append('</table></div>')
    gallery=read(RUN/'GALLERY.json')['rows']
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V21 새 잔차와 PCGrad 검증</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}select{padding:8px;margin:6px;font-size:15px;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.note{color:#596579}img{width:100%;height:auto}a{color:#0369a1}</style><h1>새 잔차 데이터와 gradient surgery로 해결되는가?</h1><section><b>'''+verdict+'''</b><p>15모델 · 3시드 ·30,000업데이트 · 데이터 변경과 PCGrad의2×2 대조 + 국소 대조군</p><p>실제 발생한 잔차를 사용했습니다. 기존/새 데이터 모두 동일 모델을 학습하며, PCGrad는 멈춤/이동 그룹의 충돌에 적용했습니다.</p><img src="effects.png"></section><section><h2>전체 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">기존 합성505장</option><option value="new_check">새 확인용 합성253장</option></select><select id="seed"><option value="mean">3시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 점PCK5</option><option value="pck10">전체 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts"></p></section><section><h2>GT와 선 샘플 · 실제 보정 화살표</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p>고정seed1 · SHA 예시와 사후 최대 개선/악화 사례를 구분합니다. 화살표를 증폭하지 않았고 샘플 위치는 중요도 지도가 아닙니다.</p></section><section>'''+''.join(body)
    for f in ['REPORT_KO.md','DESIGN_KO.md','SUMMARY.json','DIAGNOSIS.json','AUDIT.json','SPLIT_AUDIT.json','TARGET_AUDIT.json','TRAINING_COMPLETE.json']:page+='<a href="'+f+'">'+f+'</a> · '
    page+='<p>PCGrad 출처: <a href="https://arxiv.org/abs/2001.06782">원 논문</a> · <a href="https://github.com/tianheyu927/PCGrad">저자 구현</a></p></section><script>const data='+json.dumps(s['results'],ensure_ascii=False)+',labels='+json.dumps(LABELS,ensure_ascii=False)+',names='+json.dumps(NAMES,ensure_ascii=False)+',gallery='+json.dumps(gallery,ensure_ascii=False)+';'
    previous=(KIT.parent/'pallet_detail_v17_fixed/report17.py').read_text();js=previous[previous.index('const params=new URLSearchParams'):previous.index("for(const id of ['split','seed','metric'])document.getElementById(id).onchange")]
    js=js.replace("let out=", "const keys=Object.keys(labels).filter(c=>data[split+'/'+c]);let out=").replace('Object.values(labels).map(x=>','keys.map(c=>labels[c]).map(x=>').replace('for(const c of Object.keys(labels))','for(const c of keys)').replace('Object.keys(labels).map(c=>','keys.map(c=>');page+=js
    page+="""for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · 기준 '+r.base_error_px.toFixed(2)+'px / 새 데이터+PCGrad '+r.corrected_error_px.toFixed(2)+'px';}sel.onchange=show;show();</script></html>"""
    (RUN/'index.html').write_text(page);write(RUN/'CONCLUSION.json',dict(status='STABILITY_PASS' if s['stable_arms'] else 'FULL_STABILITY_FAIL',stable_arms=s['stable_arms'],evidence_gates=s['evidence_gates'],adoption=False,models=15,updates=30000,PCGrad_scope='stop/move groups, not separate point/line losses',new_final_holdout=False));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY.json'),source_sha256=sha(Path(__file__)),figures=len(gallery)));print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
