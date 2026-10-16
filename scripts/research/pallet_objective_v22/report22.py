"""V22 objective/context factorial report; all outcomes preserved."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v22-mpl')
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context22 import *
from eval_warm22 import METHODS,FACTORS
ARMS=ARMS+('unit_warm_bce','unit_warm_pixel')
LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'잘림10%','crop30':'잘림30%'}
NAMES=dict(zip(METHODS,('기존 YOLO','V12 점 기준','V18 전체 보정','V20 맞는 점 증강','V21 기존 잔차 PCGrad','기존 BCE 재현','점 거리 손실','문맥 제거 · BCE','문맥 제거 · 점 거리','점 거리 · 국소','문맥 제거 · 점 거리 · 국소','BCE 추가학습','BCE 후 점 거리')))
FOCUS=(('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside'))

def avg(s,split,a,c,t=None):
    r=s['results'][split+'/'+c];rs=[r['stock']] if a=='stock' else [r[f'{a}_seed{i}'] for i in SEEDS]
    return float(np.mean([x['strata'][t]['mean_px'] if t else x['frame_mean_px'] for x in rs]))

def main():
    s=read(RUN/'SUMMARY_ALL.json');d=read(RUN/'DIAGNOSIS.json');a=read(RUN/'AUDIT_ALL.json');probe=read(RUN/'PROBE.json')
    warmdiag=read(RUN/'DIAGNOSIS_WARM.json');d['training'].update(warmdiag['training']);d['paths']+=warmdiag['paths']
    assert all(read(RUN/f)['status']=='PASS' for f in ('AUDIT_ALL.json','DIAGNOSIS.json','DIAGNOSIS_WARM.json','INTEGRATION_CHECK.json','INTEGRATION_WARM_CHECK.json','REPRODUCTION.json','PROBE.json'))
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});verdict='안정성 통과: '+', '.join(s['stable_arms']) if s['stable_arms'] else '전체 안정성 미달 · 이번 수정도 큰 이득 유지와 악화 억제를 동시에 해결하지 못함'
    lines=['# V22 점 거리 손실과 부모 문맥 검증','',verdict,'','기존 실패를 손실 목표와 부모 문맥 의존이라는 두 가설로 나눴다. 동일 합성 데이터·초기값·배치에서 BCE/점 거리 손실 × 부모64D 문맥 유지/제거의2×2와 선 토큰을 제거한 두 국소 대조를 실행했다. 초기18모델·36,000업데이트에, 결과 확인 후 BCE 사전학습 상태에서 손실을 전환하는 후속6모델·12,000업데이트를 추가했다. 총24개 최종 모델이며 실제 실행한 업데이트는48,000회다. 후속 모델은 초기 BCE의2000업데이트를 공유한다.','']+(RUN/'INTERPRETATION_KO.md').read_text().splitlines()
    lines+=['','## 실제와 합성 평가','', '3시드 평균px, 낮을수록 좋다. 원본은 검출 프레임 평균이고 가림/잘림은 해당 대상점 평균이다. 서로 다른 집계 단위이므로 이 표를 단순 평균하지 않는다. 전체 지표·P90·PCK5/10·검출 실패 포함 정규화 오차는 선택표와 SUMMARY_ALL.json에 보존했다.']
    for split,title in [('real','실제297장 · 재사용DEV'),('synth_val','합성505장 · 재사용DEV'),('new_check','V21 확인용 합성253장 · 이번에는 재사용')]:
        focus=FOCUS if split!='new_check' else FOCUS[:4];lines+=['','### '+title,'','| 방법 | '+' | '.join(LABELS[c]+(' 대상점' if t else '') for c,t in focus)+' |','|---|'+'---:|'*len(focus)]
        for arm in METHODS:lines.append('| '+NAMES[arm]+' | '+' | '.join(f'{avg(s,split,arm,c,t):.3f}' for c,t in focus)+' |')
    lines+=['','## 원인 가설별 대조 결과','', '음수는 개선. 실제9세션 paired bootstrap5,000회95%구간이며 다중 비교 보정은 하지 않았다. 추가 개선 판정은 모든 시드 각각1% 이상 개선과 구간 상한0 미만을 함께 요구한다.','', '| 비교 | 대상 | 오차 차이px | 95%구간 | 시드별 개선% | 판정 |','|---|---|---:|---|---|---|']
    pairlabels=['문맥 유지: 거리 - BCE','BCE: 문맥 제거 - 유지','거리: 문맥 제거 - 유지','문맥 제거: 거리 - BCE','문맥 유지 거리: 선 - 국소','문맥 제거 거리: 선 - 국소','후속: 거리 전환 - BCE 연장','후속: 사전학습 - 처음부터 거리','후속: BCE 연장 - 원래 BCE'];fig,axes=plt.subplots(1,3,figsize=(14,8))
    for ax,(c,t) in zip(axes,FOCUS[1:4]):
        for j,((x,y),label) in enumerate(zip(FACTORS,pairlabels)):
            pair=x+'_vs_'+y;v=s['comparisons']['real/'+c][pair]['subgroups'][t];lo,hi=v['ci95'];mu=v['mean_px'];passed=s['evidence_gates'][pair][c]['pass_gate'];lines.append('| '+label+' | '+LABELS[c]+f' | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | '+', '.join(f'{z:+.2f}' for z in v['per_seed_gain_percent'])+' | '+('통과' if passed else '미통과')+' |');ax.plot([lo,hi],[j,j],color='#2563eb');ax.scatter(mu,j,color='#2563eb')
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(9),pairlabels if ax is axes[0] else ['']*9);ax.set_ylim(8.5,-.5);ax.set_title(LABELS[c]);ax.set_xlabel('오차 차이px · 음수가 개선');ax.grid(alpha=.2)
    fig.suptitle('점 거리 목표와 부모 문맥 제거의 효과 · 실제9세션95%구간');fig.tight_layout(rect=(0,0,1,.9));fig.savefig(RUN/'effects.png',dpi=125);plt.close(fig)
    lines+=['','## 학습 프로브는 무엇을 보여줬는가','', '현재 가중 BCE는 GT로 구한 최적 적용률을 맞춘다. 점별로는 실제 거리와 최적점이 같지만, 서로 다른 점의 실제 오차 증가량을 똑같이 반영하지는 않는다. 예를 들어 길이10px인 수평 보정에서 GT가 기준점에 있으면 전체 이동의 악화는10px이고, GT가 수직20px 떨어져 있으면 악화는2.361px이다. 두 사례의 최적 적용률0과 BCE 가중치10은 같다.','', '이 차이가 실패의 주원인인지는 별개다. 실제 합성 학습 프로브에서 두 손실 기울기의 부호가 반대인 유효점은0개였고, 기존 BCE 기울기 상위20%가 실제 점 거리 기울기의 큰 비중도 차지했다. 따라서 프로브만으로 심한 목표 충돌이나 거리 손실의 우월성을 주장하지 않았다.','', '| 학습 프로브 ·3시드 평균 | 문맥 제거 시 점 이동px | 새 선 토큰 제거 시 점 이동px |','|---|---:|---:|']
    for typ in ('all','hidden','outside'):
        rs=[x['statistics'][typ]['intervention_mean_displacement'] for x in probe['rows']];lines.append('| '+typ+' | '+' | '.join(f'{np.mean([x[k] for x in rs]):.3f}' for k in ('no_context','no_extra_lines'))+' |')
    lines+=['','문맥 제거의 영향이 더 크다는 것은 기존 모델의 민감도다. 학습 후 문맥을0으로 바꾸는 것은 분포 밖 개입이며 문맥이 나쁘다는 증명이 아니다. 처음부터 해당 문맥 없이 재학습한 대조가 필요해 이번2×2에 포함했다.','', '## 학습 데이터에서 적용률이 포화되는가','', '| 모델 | 화면 밖 나쁜 보정 적용률 | 좋은 보정 적용률 | 적용률>99% 비율 | 학습 대상점 오차px |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        rs=[d['training'][f'{arm}_seed{i}']['outside'] for i in SEEDS];lines.append('| '+NAMES[arm]+' | '+' | '.join(f'{np.mean([x[k] for x in rs]):.4f}' for k in ('mean_alpha_bad','mean_alpha_good','alpha_above99','after_px'))+' |')
    lines+=['','좋은/나쁜 보정은 원래 전체 보정이 기준점 오차를0.1px 넘게 줄이거나 키우는지 GT로 사후 구분했다. 추론 필터나 입력으로 사용하지 않는다. 적용률은 정답일 확률이 아니다.','', '## 현재 벡터에서 더 개선할 기하학적 여지는 있는가','', '| 실제 대상 | 반대 방향 점 수 | BCE 오차px | GT 최적 적용률 오차px | 적용률 판단의 차이px |','|---|---:|---:|---:|---:|']
    for c in ('corner20','crop20','crop40'):
        rs=[x for x in d['geometry'] if x['condition']==c];lines.append('| '+LABELS[c]+' | '+' | '.join(f'{np.mean([x[k] for x in rs]):.3f}' for k in ('wrong_direction_points','actual_px','oracle_px','learned_scale_regret_px'))+' |')
    lines+=['','GT 최적 적용률은 학습 모델이 아니며 추론할 수 없다. 이 결과는 같은 벡터에서도 현재 적용률 판단이 놓치는 개선 여지가 있다는 뜻이다. 벡터에 수직인 오차와0~1 범위 밖의 필요 이동은 이 구조가 해결할 수 없다. 이를 점·선 융합 전체의 한계로 확대하지 않는다.','', '## 정확히 바꾼 것','', '점 거리 손실은 최종점과 원래 GT 사이의 raw-pixel Euclidean 거리 평균이다. 원래 유효점을 모두 사용하며 새 그룹 가중치·악화 벌점·GT 필터·self-training은 추가하지 않았다. 원래 BCE와 동일2000업데이트, AdamW3e-4, weight decay1e-4, float32, gradient clip5, 마지막 체크포인트로 비교했다. 학습 데이터는 기존 합성253장/1462검출 관측/6조건이다.','', '후속은 초기 unit_bce의 최종 상태를 양쪽에 똑같이 불러온 뒤 optimizer를 동일하게 초기화하고 같은2000배치를 추가한다. BCE를 유지하는 대조와 점 거리로 전환하는 모델을 비교한다. 초기 결과를 본 뒤 추가한 탐색적 실험이다. 원래 체크포인트를 덮어쓰지 않았다.','', '문맥 제거는 rep.context64D만0으로 만든다. 부모의 가시성 확률·전체 복원 차이·점 ID·YOLO 영상 특징·좌표·보정 방향과 길이는 유지한다. 완전히 부모 정보를 없앤 실험은 아니다. 추가 선 토큰을 제거한 국소 대조군도 기존 부모의 선 학습 영향을 공유한다.','', '최종점=p0+sigmoid(logit)×고정 V18 보정 벡터d. 손실과 입력 문맥을 바꾼 학습 가능한 적용률 모듈이며 YOLO/DHT 전체 end-to-end 재학습이나 새로운 방향의 보정은 아니다. 이번 단계에 PCGrad를 함께 추가하지 않아 손실/문맥 변경 효과와 구분했다.','', '선 샘플의 유효성은 화면과 검출박스의 교차로 정한다. 화면 안의 가림 영역이나 배경도 포함할 수 있으며 실제 파렛트 경계를 검증한 것은 아니다. 이 가시성 문제를 수정한 실험으로 해석하지 않는다.','', '## GT·기울기·통합 검증과 한계','', f'실제 수동 클릭 source/ID/유효점과 조건별 좌표 이동을 재검산했다. 총{a["prediction_frame_rows"]:,}예측 프레임 행의 지표 최대 재계산 차이는{a["metric_replay_max"]:.3g}, 보정 재구성 최대 차이는{a["retention_replay_max_px"]:.8f}px다. 원래 기준점·벡터·검출 누락을 유지했다. 거리 손실 기울기는 독립 float64 유한차분 검사에서 최대 차이{d["distance_gradient_finite_difference_max"]:.3g}였다.','', '문맥 제거 모델은 문맥을 바꿔도 출력이 정확히 같았다. 선 부호 불변성과 추가 선 투영층의 기울기 경로를 확인했고, 통합24모델×6조건×4프레임의 좌표가 캐시와0.001px 이내로 일치했다. 48,000업데이트의 손실·기울기가 유한했고 BCE 대조군은 이전 모델을0.001px 이내로 재현했다.','', '실제297장·9세션·1251수동점, 기존 합성505장과 V21 확인용253장을 재사용한 탐색적 평가다. 마지막253장도 이번에는 이미 확인한 데이터이며 원래 YOLO 검증 풀 출신이다. 새 최종 holdout이나 자연 가림 GT를 확보한 것은 아니다. 코드가 클릭을 정확히 사용했음을 확인했지만 수동 클릭의 물리적 오차0을 인증하지는 않는다. 운영 모델을 교체하지 않았고 실사 성능으로 시드나 체크포인트를 선택하지 않았다.']
    report='\n'.join(lines)+'\n';(RUN/'REPORT_KO.md').write_text(report);(ROOT/'_docs/notes/pallet_objective_v22_results.md').write_text(report)
    design='# V22 사전 대조 설계\n\n```json\n'+json.dumps(read(RUN/'PROTOCOL.json'),ensure_ascii=False,indent=2)+'\n```\n\n## 결과 확인 후 탐색적 추가\n\n```json\n'+json.dumps(read(RUN/'WARM_PROTOCOL.json'),ensure_ascii=False,indent=2)+'\n```\n';(RUN/'DESIGN_KO.md').write_text(design);(ROOT/'_docs/notes/pallet_objective_v22_design.md').write_text(design)

    body=[];inside=False
    for line in lines[1:]:
        if line.startswith('|'):
            if line.startswith('|---'):continue
            if not inside:body.append('<div class="scroll"><table>');inside=True
            body.append('<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
        if inside:body.append('</table></div>');inside=False
        if line:body.append('<h2>'+html.escape(line.lstrip('# '))+'</h2>' if line.startswith('#') else '<p>'+html.escape(line)+'</p>')
    if inside:body.append('</table></div>')
    gallery=read(RUN/'GALLERY_WARM.json')['rows']+read(RUN/'GALLERY.json')['rows']
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V22 점 거리와 부모 문맥 검증</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}select{padding:8px;margin:6px;font-size:15px;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.note{color:#596579}img{width:100%;height:auto}a{color:#0369a1}</style><h1>점 거리 손실과 부모 문맥이 실패의 원인인가?</h1><section><b>'''+verdict+'''</b><p>초기18모델 + 후속6모델 ·3시드 ·48,000업데이트 · 손실/문맥 대조와 BCE 사전학습 후 전환</p><p>기존 합성 데이터와 보정 벡터를 유지하고 최종 점 거리 손실 및 부모 문맥 제거의 효과를 분리했습니다.</p><img src="effects.png"></section><section><h2>전체 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">기존 합성505장</option><option value="new_check">V21 합성253장 · 재사용</option></select><select id="seed"><option value="mean">3시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 점PCK5</option><option value="pck10">전체 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts"></p></section><section><h2>GT와 선 샘플 · 실제 보정 화살표</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p>고정seed1 · SHA 예시와 사후 최대 개선/악화 사례를 구분합니다. 화살표를 증폭하지 않았고 샘플 위치는 중요도 지도가 아닙니다.</p></section><section>'''+''.join(body)
    for f in ['REPORT_KO.md','DESIGN_KO.md','SUMMARY_ALL.json','DIAGNOSIS.json','DIAGNOSIS_WARM.json','AUDIT_ALL.json','PROBE.json','CONSTANT_DIAGNOSIS.json','CONSTANT_MATCHED_DIAGNOSIS.json','PROTOCOL.json','TRAINING_COMPLETE.json']:page+='<a href="'+f+'">'+f+'</a> · '
    page+='</section><script>const data='+json.dumps(s['results'],ensure_ascii=False)+',labels='+json.dumps(LABELS,ensure_ascii=False)+',names='+json.dumps(NAMES,ensure_ascii=False)+',gallery='+json.dumps(gallery,ensure_ascii=False)+';'
    previous=(KIT.parent/'pallet_detail_v17_fixed/report17.py').read_text();js=previous[previous.index('const params=new URLSearchParams'):previous.index("for(const id of ['split','seed','metric'])document.getElementById(id).onchange")]
    js=js.replace("let out=", "const keys=Object.keys(labels).filter(c=>data[split+'/'+c]);let out=").replace('Object.values(labels).map(x=>','keys.map(c=>labels[c]).map(x=>').replace('for(const c of Object.keys(labels))','for(const c of keys)').replace('Object.keys(labels).map(c=>','keys.map(c=>');page+=js
    page+="""for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · 기준 '+r.base_error_px.toFixed(2)+'px / 문맥 제거+점 거리 '+r.corrected_error_px.toFixed(2)+'px';}sel.onchange=show;show();</script></html>"""
    (RUN/'index.html').write_text(page);write(RUN/'CONCLUSION.json',dict(status='STABILITY_PASS' if s['stable_arms'] else 'FULL_STABILITY_FAIL',stable_arms=s['stable_arms'],evidence_gates=s['evidence_gates'],adoption=False,models=24,updates=48000,scope='objective/context factorial, no PCGrad',new_final_holdout=False));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY_ALL.json'),source_sha256=sha(Path(__file__)),figures=len(gallery)));print('REPORT COMPLETE',flush=True)

if __name__=='__main__':main()
