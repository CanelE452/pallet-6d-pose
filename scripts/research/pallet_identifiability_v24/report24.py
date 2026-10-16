"""Report of memorization, direction information and actual correction behavior."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v24-mpl')
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context24 import *
from eval24 import METHODS,FACTORS
LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'잘림10%','crop30':'잘림30%'}
NAMES=dict(zip(METHODS,('기존 YOLO','V12 점 기준','V18 전체 보정','V22 기존 BCE','V23 손실계수 균형','방향 판별 · 선 포함','방향 판별 · 국소','방향 판별 · 부모 기하')))
FOCUS=(('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside'))
def avg(s,split,a,c,t=None):
    r=s['results'][split+'/'+c];rs=[r['stock']] if a=='stock' else [r[f'{a}_seed{i}'] for i in SEEDS]
    return float(np.mean([x['strata'][t]['mean_px'] if t else x['frame_mean_px'] for x in rs]))
def main():
    s=read(RUN/'SUMMARY.json');t=read(RUN/'TINY_COMPLETE.json');cl=read(RUN/'CLASSIFICATION.json');audit=read(RUN/'AUDIT.json');assert all(read(RUN/f)['status']=='PASS' for f in ('AUDIT.json','TINY_AUDIT.json','INTEGRATION_CHECK.json'))
    verdict='전체 안정성 통과: '+', '.join(s['stable_arms']) if s['stable_arms'] else '전체 안정성 미달 · 학습 가능성과 일반화 문제를 분리해 확인'
    lines=['# V24 보정 방향 정보와 작은 집합 학습 검증','',verdict,'','고정된 점·Hough 보정 모듈을 두 경로로 진단했다.12개 합성 원본의 모든 검출 변형으로6모델을6000회씩 반복 학습했다. 별도로 기존764원본/4409관측의 보정 방향 부호를3구조×3시드×2000회 학습했다. 총15개 신경망 모델/54,000업데이트이며, 정답을 직접 기억하는3개 lookup 대조의6000업데이트는 별도다.','']+(RUN/'INTERPRETATION_KO.md').read_text().splitlines()
    lines+=['','## 작은 집합: 같은 구조로 보정 중단을 학습할 수 있는가','', '원본12장과 별도48장은 SHA로 학습 전에 지정했고 GT로 골라내지 않았다. 동일 원본의 모든 clean/가림/잘림 변형을 같은 집합에 두었다. 같은 초기값·배치·가중BCE·AdamW3e-4에서 선 포함/국소 대조를 비교했다. 이48장은 tiny 학습에서만 제외되며 기존V23와 이번 전체 방향 판별에는 학습용인 원본이다. 새로운 전체 파이프라인 holdout이 아니다.','', '판정: 모든 시드의 최종 학습 오차가GT최적 적용률 오차보다0.25px 이내이고, 기준 오차5px 이하인 점의 평균이 기준보다0.1px 넘게 나빠지지 않을 것. 이 기준은 실사 일반화 기준이 아니다.','', '| 모델 | 시드 | 학습 최적점 차이2000px | 학습 최적점 차이6000px | 정확한 점 기준px | 정확한 점 결과px | 별도48장 최적점 차이px | 암기 기준 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in t['rows']:
        early=r['curve'][1]['train'];final=r['curve'][-1];lines.append(f'| {r["arm"]} | {r["seed"]} | {early["all"]["regret_px"]:.4f} | {final["train"]["all"]["regret_px"]:.4f} | {final["train"]["accurate"]["base_px"]:.4f} | {final["train"]["accurate"]["after_px"]:.4f} | {final["check"]["all"]["regret_px"]:.4f} | '+('통과' if r['memorization_pass'] else '미통과')+' |')
    lines+=['','정답 lookup은 관측·점마다 독립적인 logit를 두고 합성GT로 직접 학습한 특권 대조다. 영상 특징으로 추론하는 모델이 아니고 실사로 이전할 수 없다. lookup의 최적점 오차 차이: '+', '.join(f'seed{r["seed"]} {r["statistics"]["all"]["regret_px"]:.5f}px' for r in t['lookup'])+'.','', '## 현재 특징에 방향을 구분할 단서가 있는가','', 'GT잔차와 고정 보정 벡터의 내적이 양수면1이다. GT잔차·보정 길이<=0.1px의 퇴화 사례만 분류에서 제외했다. 점 오차 평가에서는 원래 유효점을 모두 유지했다. 합성 학습 집합의 양/음성 계수 합을 같게 한 BCE를 사용했다. 실사 GT로 학습·선택·보정하지 않았다.','', '선 포함, 추가 선 토큰 제거, 부모 기하 중심 대조를 비교했다. 마지막 대조는192채널 현재 영상 특징과64D 문맥을0으로 만들지만 부모 가시성 확률·amodal·선 좌표·보정 벡터·content mask는 유지한다. 따라서 완전히 영상 정보가 없는 대조가 아니다.','', 'AUROC0.5는 순위 구분이 없는 수준이다. 실제 잘림20%에서 모든 시드AUROC>=0.70 및9세션95%구간 하한>0.5를 신호 기준으로 정했다. 추가 선 이득은 각 시드AUROC차이>=0.02 및 쌍대 구간 하한>0이다. 임계값을 결과에 맞춰 변경하지 않았다.','', '| 집합·조건 | 모델 | 시드별 AUROC | 평균 |','|---|---|---|---:|']
    for split in ('training','prospective','synth_val','real'):
        for c,typ in [('crop20','outside'),('crop40','outside'),('corner20','hidden')]:
            for arm in ARMS:
                aa=[cl['results'][split+'/'+c]['models'][f'{arm}_seed{i}'][typ]['auc'] for i in SEEDS];lines.append('| '+split+'/'+c+' | '+NAMES[arm]+' | '+', '.join('—' if x is None else f'{x:.4f}' for x in aa)+' | '+('—' if any(x is None for x in aa) else f'{np.mean(aa):.4f}')+' |')
    lines+=['','## 분류 점수를 실제 보정량으로 쓰면 좋아지는가','', '방향의 부호는 적절한 이동량을 알려주지 않는다. 기준(0,0),GT(1,0),보정(10,0)이면 방향은 맞지만 전체 이동은1px 오차를9px로 키운다. 최적 적용률은0.1이다. 따라서 sigmoid 방향 점수를 적용률로 사용하는 아래 비교는 별도 탐색적 검증이며, 분류 성공을 융합 성공으로 간주하지 않는다. 학습의 클래스 균형으로 점수는 보정된 확률도 아니다.','', '동일 검출·기준점·벡터에서 p=p0+sigmoid(logit)×d. 임계값 필터나GT 경로가 없다.3시드 평균px, 원본은 검출 프레임 평균, 가림/잘림은 해당 대상점 평균이다. 모든 평가 집합을 재사용한DEV다.']
    for split,title in [('real','실제297장'),('synth_val','기존 합성505장'),('new_check','V21 합성253장'),('prospective','V23 합성128장')]:
        focus=FOCUS if split in ('real','synth_val') else FOCUS[:4];lines+=['','### '+title,'','| 방법 | '+' | '.join(LABELS[c] for c,k in focus)+' |','|---|'+'---:|'*len(focus)]
        for a in METHODS:lines.append('| '+NAMES[a]+' | '+' | '.join(f'{avg(s,split,a,c,k):.3f}' for c,k in focus)+' |')
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});fig,axes=plt.subplots(1,3,figsize=(16,5))
    for arm,color in zip(TINY_ARMS,('#2563eb','#e87925')):
        rr=[r for r in t['rows'] if r['arm']==arm];steps=[x['step'] for x in rr[0]['curve']]
        for split,style in [('train','-'),('check','--')]:
            vals=[np.mean([r['curve'][j][split]['all']['regret_px'] for r in rr]) for j in range(len(steps))];axes[0].plot(steps,vals,style,color=color,label=('선 포함' if arm=='unit_tiny' else '국소')+' '+('학습' if split=='train' else '별도48장'))
    axes[0].set_yscale('log');axes[0].set_yticks([.01,.1,1],['0.01','0.1','1']);axes[0].set_title('작은 집합: GT최적 적용률 대비 오차');axes[0].set_xlabel('업데이트');axes[0].set_ylabel('오차 차이px · 로그축');axes[0].legend(fontsize=8)
    for j,a in enumerate(ARMS):
        v=cl['results']['real/crop20']['intervals'][a+'/outside'];lo,hi=v['ci95'];axes[1].plot([lo,hi],[j,j],c='#2563eb');axes[1].scatter(v['mean_auc'],j,c='#2563eb')
    axes[1].axvline(.5,color='gray',ls='--');axes[1].axvline(.7,color='orange',ls=':');axes[1].set_yticks(range(3),['선 포함','국소','부모 기하']);axes[1].set_title('실사 잘림20% 방향 AUROC');axes[1].set_xlabel('9세션95%구간 · 클수록 좋음')
    for j,(c,k) in enumerate(FOCUS[1:4]):
        v=s['comparisons']['real/'+c]['unit_sign_vs_unit_bce']['subgroups'][k];lo,hi=v['ci95'];axes[2].plot([lo,hi],[j,j],c='#2563eb');axes[2].scatter(v['mean_px'],j,c='#2563eb')
    axes[2].set_yticks(range(3),['코너 가림','잘림20%','잘림40%']);axes[2].axvline(0,color='gray',ls='--');axes[2].set_title('방향 점수 적용 - 기존 BCE');axes[2].set_xlabel('오차 차이px · 음수가 개선')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(RUN/'effects.png',dpi=130);plt.close(fig)
    lines+=['','## 검증과 한계','',f'점 오차 평가{audit["prediction_frame_rows"]:,}프레임 행의 독립 지표 재계산 차이는{audit["metric_replay_max"]:.3g}, 보정 재구성 차이는{audit["retention_replay_max_px"]:.7f}px였다. 실제 manual_click GT의 원래ID·유효점·잘림 원점을 재검산했다. GT는 클릭값의 올바른 사용을 확인했으며 물리적 클릭 오차0을 보장하지 않는다.','', '54,000신경망 업데이트의 손실·기울기가 유한했다. tiny 학습 곡선과 최적 적용률은 재실행 및NumPy로 대조했다. 부모 기하 대조는 제거 대상 영상 특징/문맥을 바꿔도 출력이 정확히 같았다.9개 방향 모델의 통합 추론과 캐시는0.001px 이내로 일치했다. 세션 AUROC bootstrap은 동점이 있는 알려진 예제에서독립 가중AUROC와 검산했다.','', '실제9세션 paired bootstrap5000회95%구간이며 다중 비교 보정은 없다. 기존 실사297장/1251수동점과 합성 평가를 재사용했고 새로운 실사 holdout은 없다. 작은 집합의 반복 학습 성공은 암기 능력의 증거이며 충분한 일반화·방향 정보·안정성을 보장하지 않는다. 분류 실패도 정보가 원천적으로 없다는 증명이 아니다.','', '선 샘플의 유효성은 화면·검출박스 교차만 판단해 가림·배경이 포함될 수 있다. 방향을 회전시키거나 보정 범위 밖으로 이동시키는 새 벡터 모듈은 학습하지 않았다. 운영 모델은 교체하지 않았다.']
    report='\n'.join(lines)+'\n';(RUN/'REPORT_KO.md').write_text(report);(ROOT/'_docs/notes/pallet_identifiability_v24_results.md').write_text(report);design='# V24 사전 설계\n\n'+json.dumps(read(RUN/'PROTOCOL.json'),ensure_ascii=False,indent=2);(RUN/'DESIGN_KO.md').write_text(design);(ROOT/'_docs/notes/pallet_identifiability_v24_design.md').write_text(design)
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
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V24 보정 방향 정보와 학습 가능성</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}select{padding:8px;margin:6px;font-size:15px;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}img{width:100%;height:auto}a{color:#0369a1}</style><h1>보정 중단은 학습 가능한가? 방향 단서는 있는가?</h1><section><b>'''+verdict+'''</b><p>작은 집합6모델 + 방향 판별9모델 ·54,000신경망 업데이트</p><img src="effects.png"></section><section><h2>전체 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">기존 합성505장</option><option value="new_check">V21 합성253장</option><option value="prospective">V23 합성128장 · 재사용</option></select><select id="seed"><option value="mean">3시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 점PCK5</option><option value="pck10">전체 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts"></p></section><section><h2>GT와 선 샘플 · 실제 보정 화살표</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p>seed1 고정 SHA 및 사후 극단 사례. 화살표 증폭 없음. 청록은 샘플 위치이며 기여도 지도가 아닙니다.</p></section><section>'''+''.join(body)
    for f in ('REPORT_KO.md','DESIGN_KO.md','SUMMARY.json','CLASSIFICATION.json','TINY_COMPLETE.json','AUDIT.json','TINY_AUDIT.json','INTEGRATION_CHECK.json','PROTOCOL.json'):page+='<a href="'+f+'">'+f+'</a> · '
    page+='</section><script>const data='+json.dumps(s['results'],ensure_ascii=False)+',labels='+json.dumps(LABELS,ensure_ascii=False)+',names='+json.dumps(NAMES,ensure_ascii=False)+',gallery='+json.dumps(gallery,ensure_ascii=False)+';'
    previous=(KIT.parent/'pallet_detail_v17_fixed/report17.py').read_text();js=previous[previous.index('const params=new URLSearchParams'):previous.index("for(const id of ['split','seed','metric'])document.getElementById(id).onchange")];js=js.replace('let out=',"const keys=Object.keys(labels).filter(c=>data[split+'/'+c]);let out=").replace('Object.values(labels).map(x=>','keys.map(c=>labels[c]).map(x=>').replace('for(const c of Object.keys(labels))','for(const c of keys)').replace('Object.keys(labels).map(c=>','keys.map(c=>');page+=js
    page+="""for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · 기준 '+r.base_error_px.toFixed(2)+'px / 방향 판별 '+r.corrected_error_px.toFixed(2)+'px';}sel.onchange=show;show();</script></html>"""
    (RUN/'index.html').write_text(page);write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY.json'),source_sha256=sha(Path(__file__)),figures=len(gallery)));print('REPORT COMPLETE',flush=True)
if __name__=='__main__':main()
