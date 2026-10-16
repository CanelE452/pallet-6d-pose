"""Corrected, factorial results; no invalid-mask results in final tables."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v17-mpl')
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context17 import *
from eval17 import METHODS

LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 가림','crop10':'잘림10%','crop30':'잘림30%'}
NAMES={'stock':'기존 YOLO','base':'V12 점 기준','amodal':'V12 전체 복원','balanced_old':'V15 보수적·균형','strip':'V16 선 특징','role':'V16 경계 역할','low_none':'저해상도 · 보조 없음','high_none':'고해상도 · 보조 없음','low_proxy':'저해상도 · 기존 정답','high_proxy':'고해상도 · 기존 정답','low_mask':'저해상도 · 가시 마스크','high_mask':'고해상도 · 가시 마스크','high_none_low':'고해상도 모델 · 세부 정보 제거','high_mask_low':'마스크 모델 · 세부 정보 제거','high_none_donor':'고해상도 모델 · 다른 RGB','high_mask_donor':'마스크 모델 · 다른 RGB'}
FOCUS=(('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside'),('crop10','outside'),('crop30','outside'))
PAIR_LABELS={'high_none_vs_low_none':'해상도 증가 · 보조 없음','high_proxy_vs_low_proxy':'해상도 증가 · 기존 정답','high_mask_vs_low_mask':'해상도 증가 · 가시 마스크','low_mask_vs_low_proxy':'정답 수정 · 저해상도','high_mask_vs_high_proxy':'정답 수정 · 고해상도'}

def avg(rows,a,t=None):
    rs=[rows['stock']] if a=='stock' else [rows[f'{a}_seed{i}'] for i in SEEDS]
    return float(np.mean([r['strata'][t]['mean_px'] if t else r['frame_mean_px'] for r in rs]))

def effects(s):
    fig,axes=plt.subplots(1,3,figsize=(15,5.5))
    for ax,(c,t) in zip(axes,[('corner20','hidden'),('crop20','outside'),('crop40','outside')]):
        for j,(pair,label) in enumerate(PAIR_LABELS.items()):
            v=s['comparisons']['real/'+c][pair]['subgroups'][t];lo,hi=v['ci95'];mu=v['mean_px'];ax.plot([lo,hi],[j,j],color='#2563eb');ax.scatter([mu],[j],color='#2563eb',s=35)
        ax.axvline(0,color='#64748b',ls='--');ax.set_yticks(range(5),list(PAIR_LABELS.values()) if ax is axes[0] else ['']*5);ax.invert_yaxis();ax.set_xlabel('오차 차이(px) · 음수가 개선');ax.set_title(LABELS[c]);ax.grid(alpha=.2)
    fig.suptitle('해상도와 정답 변경 효과를 따로 비교\n점: 3시드 평균 차이 · 선: 실제9세션 paired95%구간 · 다중 비교 보정 없음',fontsize=12);fig.tight_layout(rect=(0,0,1,.86));fig.savefig(RUN/'effects.png',dpi=125,bbox_inches='tight');plt.close(fig)

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});s=read(RUN/'SUMMARY.json');d=read(RUN/'DIAGNOSIS.json');a=read(RUN/'AUDIT.json');integ=read(RUN/'INTEGRATION_CHECK.json');ma=read(RUN/'MASK_DECODING_AUDIT.json');g=read(RUN/'GALLERY.json')['rows']
    assert all(x['status']=='PASS' for x in (d,a,integ,ma));effects(s)
    stable=s['stable_arms'];verdict='전체 안정성 기준 미달' if not stable else '전체 안정성 통과: '+', '.join(stable)
    passes={p:[c for c,v in by.items() if v['pass_gate']] for p,by in s['line_gates'].items()}
    lines=['# 고해상도 RGB와 가시 마스크 검증: '+verdict,'',
        '세부 영상 정보 부족과 선 보조 정답 불일치를 2×3 비교로 검사했다. 원본 RGB96과 같은 이미지를24로 줄인 뒤96으로 복원한 입력을 비교했고, 각각 보조 정답 없음/기존 cuboid 경계/실제 가시 마스크가 지지하는 경계로 학습했다. 모든 구조에서 기존 YOLO/점/복원/DHT 부모와8후보를 고정했다.', '',
        '이번 결과에서는 해상도 증가와 가시 마스크 정답 변경 모두 사전에 정한 추가 개선 기준을 통과한 조건이 없었다. 보조 없는 모델의 잘림40% 오차는 저해상도26.386→고해상도26.429px로 뚜렷한 개선이 없었다. 마스크는 고해상도 모델의 합성 높이선 F1을0.300→0.322로 올렸지만, 실제 점 성능의 안정성을 확보하지 못했다. 이 결론은 새 RGB 분기와 고정 부모를 쓰는 현재 구조에 한정한다.', '',
        '## 실제 결과','',
        '3시드 평균px, 작을수록 좋음. 원본은 프레임 평균, 나머지는 해당 가림/잘림 대상점 평균이다. 모든 표는 마스크 해독 오류 수정 후의 유효 결과다.', '',
        '| 방법 | 원본 | 코너 가림 | 잘림20% 밖 | 잘림40% 밖 | 잘림10% 밖 | 잘림30% 밖 |','|---|---:|---:|---:|---:|---:|---:|']
    measured=[]
    for arm in ('stock','base','amodal','balanced_old','strip')+ARMS:
        v=[avg(s['results']['real/'+c],arm,t) for c,t in FOCUS];measured.append((NAMES[arm],v));lines.append('| '+NAMES[arm]+' | '+' | '.join(f'{x:.3f}' for x in v)+' |')
    lines+=['','## 해상도와 마스크 효과를 분리하면','',
        '아래는 다른 요인을 유지한 비교다. 양수는 악화, 음수는 개선이다. 보고된95%구간은 실제9세션을 묶어 재표집한 paired bootstrap이며, 모든 시드1% 이상 개선 기준을 함께 확인한다.', '',
        '| 비교 | 잘림20% 차이 | 잘림40% 차이 [95%구간] | 모든 시드·구간 기준을 통과한 조건 |','|---|---:|---:|---|']
    for pair,name in PAIR_LABELS.items():
        m=s['comparisons']['real/crop20'][pair]['subgroups']['outside'];h=s['comparisons']['real/crop40'][pair]['subgroups']['outside'];lo,hi=h['ci95'];lines.append(f'| {name} | {m["mean_px"]:+.3f} | {h["mean_px"]:+.3f} [{lo:+.3f}, {hi:+.3f}] | '+(', '.join(LABELS[c] for c in passes[pair]) or '없음')+' |')
    lines+=['','전체 안정성 판정: **'+verdict+'**. 원본과 잘림10/20/30%의 악화1% 이내 유지, 코너 가림과 잘림40%에서 기존 전체 복원 이득80% 이상 유지라는 사전 기준을 그대로 적용했다. 어느 조건의 소폭 개선만으로 전체 성공을 선언하지 않는다.', '',
        '## 모델이 RGB 세부 정보를 사용했는가','',
        '같은 고해상도 모델에 저해상도 RGB 또는 다른 이미지 RGB를 넣고 후보와 YOLO 특징은 유지했다. 별도 학습한 high-low 비교와 같은 모델에 대한 입력 개입을 구분한다. 아래는 실제 잘림40% 화면 밖 점의 차이다.', '',
        '| 같은 모델 개입 비교 | 실제 입력 - 교체 입력(px) | 세션95%구간 |','|---|---:|---:|']
    for arm in ('high_none','high_mask'):
        for variant in ('low','donor'):
            v=s['comparisons']['real/crop40'][arm+'_vs_'+arm+'_'+variant]['subgroups']['outside'];lo,hi=v['ci95'];lines.append(f'| {NAMES[arm]} / {variant} | {v["mean_px"]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    lines+=['','분포 밖 입력 교체 실험이므로 이 값만으로 유용한 실사 인식이나 일반화를 입증하지 않는다. 특히 고해상도 분기의 존재 자체는 새로운 정보가 충분히 학습됐다는 뜻이 아니다.', '',
        '## 짧은 높이선 역할은 나아졌는가','',
        '모든 구조를 동일한 가시 마스크 정답에 평가했다. 비교 대상은 마스크와8점이 유효한 합성 원본 평가 프레임의 동일 샘플이다. 서로 다른 정답의 F1을 직접 비교하지 않는다. 아래는3시드 평균이다.', '',
        '| 구조 | 긴 경계 F1 | 높이 경계 F1 | 높이 경계 precision |','|---|---:|---:|---:|']
    role_stats={}
    for arm in ARMS:
        v=[d['role_metrics'][f'synth_val/{arm}/seed{i}']['mask'] for i in SEEDS];f=np.mean([x['f1'] for x in v],0);pr=np.mean([x['precision'] for x in v],0);role_stats[arm]=dict(long_f1=float(f[1]),height_f1=float(f[2]),height_precision=float(pr[2]));lines.append(f'| {NAMES[arm]} | {f[1]:.3f} | {f[2]:.3f} | {pr[2]:.3f} |')
    lines+=['','보조 역할 분류와 최종 점 오차는 다른 목표다. 역할 분류가 좋아져도 잘못된 복원 방향/크기를 구분하지 못하면 점 성능이 개선되지 않는다. 보조 없는 모델의 역할 출력은 의미 분류기로 학습한 것이 아니므로 해당 F1은 기전 대조용이다.', '',
        '## 이번에 확인한 정답과 수정 기록','',
        f'처음 마스크를0보다 큰 값으로 읽었으나 시각 검증에서 배경까지 포함한 오류를 발견했다. 파일의 배경은0/1/2, 물체는254/255였다. 기존 로더의 MASK_THRESHOLD=127 규약에 맞춰 수정했고, 대응되는{len(ma["rows"])}마스크 모두 해독한 전경 픽셀 수가 렌더러 mask_area_visible 기록과 정확히 일치했다. 준비된 학습 RGB169장도 렌더러 원본과 픽셀 단위로 일치했다.', '',
        '초기 실행은 INVALID_INPUT_AUDIT.json으로 무효 표시하고 보존했다. 수정으로 영향을 받는 마스크 모델6개를 같은 설정에서 재학습했다. 나머지12개 대조군은 RGB·proxy 정답·초기화·학습 순서의 정확한 일치를 확인하고 재사용했다. 성능을 보고 임계값이나 손실을 조정하지 않았다. 최종 평가에는 수정 결과만 포함했다.', '',
        '가시 마스크는 물체의 실루엣이며 내부 면 경계의 가시성 정답이 아니다. 기존 cuboid 경계와 실루엣이 어긋난 샘플을 배경으로 바꾸면서 학습 목표가 좁아진다. 따라서 전체 파렛트 옆 edge 정답을 완성했다고 해석하지 않는다. 자연 가림과 수동 실사 클릭오차도 새로 주석하거나 측정하지 않았다.', '',
        '## 해석과 한계','',
        '이 실험으로 해상도와 마스크 정답 각각의 영향은 분리할 수 있다. 하지만 고정 DHT/YOLO 특징을 end-to-end로 다시 학습하거나 실제 내부 면 경계의 가시성까지 감독한 실험은 아니다. 최종 안정성 실패를 곧바로 점·선 결합의 원리적 불가능으로 확대하지 않는다.', '',
        '현재 설계는 후보가 정답에 가까워지는지 학습 점수로 선택한다. 선처럼 보이는지 또는 실루엣과 맞는지는 그 선택에 필요한 정보의 일부다. 보조 정답과 세부 정보만으로 안정성이 확보되지 않으면, 복원 후보의 방향/크기와 관측 가능한 영상 근거를 직접 연결하는 학습 목표가 남은 가설이다. 이는 이번 실험으로 검증을 끝낸 해결책은 아니다.', '',
        f'독립 검사: {a["prediction_frame_rows"]:,}예측 프레임 행, 후보 재구성 최대 오차{a["candidate_replay_max_px"]:.8f}px, 지표 재계산 최대 차이{a["metric_replay_max"]:.3g}. 보조 정답{d["independent_aux_samples"]}샘플을 별도 float64 계산으로 검산했다. 통합18모델×6조건×4프레임의 선택은 캐시와 일치했고 최대 좌표 차이는{integ["max_error_px"]:.8f}px였다.', '',
        '유효 최종18모델은36,000업데이트 분량이며 그중24,000업데이트는 오류 영향을 받지 않은 대조군이다. 별도로 폐기한 초기 마스크 학습12,000업데이트가 있다. 실제297장·9세션·수동1251점 및 합성505장의 재사용DEV, 인공 가림/잘림8조건이다. 원래 ID·검출 실패 포함 지표·PCK5/10·P90를 보존했다. 실제 GT는 학습/추론/시드 선택에 사용하지 않았으며 배포 모델을 교체하지 않았다.']
    text='\n'.join(lines)+'\n';(RUN/'REPORT_KO.md').write_text(text);(ROOT/'_docs/notes/pallet_detail_v17_results.md').write_text(text);(RUN/'DESIGN_KO.md').write_text((ROOT/'_docs/notes/pallet_detail_v17_design.md').read_text())
    write(RUN/'CONCLUSION.json',dict(status='FULL_STABILITY_FAIL' if not stable else 'DEV_STABILITY_PASS',stable_arms=stable,factor_passes=passes,role_stats=role_stats,corrected_mask_run=True,adoption=False,no_real_tuning=True))
    table='<table><tr><th>방법</th>'+''.join('<th>'+LABELS[c]+'</th>' for c,t in FOCUS)+'</tr>'+''.join('<tr><td>'+n+'</td>'+''.join(f'<td>{x:.3f}</td>' for x in v)+'</tr>' for n,v in measured)+'</table>'
    role_table='<section><h2>같은 가시 마스크 정답에서 평가한 경계 역할</h2><table><tr><th>모델</th><th>긴 경계 F1</th><th>높이 경계 F1</th><th>높이 경계 precision</th></tr>'+''.join('<tr><td>'+NAMES[k]+'</td>'+''.join(f'<td>{v[x]:.3f}</td>' for x in ('long_f1','height_f1','height_precision'))+'</tr>' for k,v in role_stats.items())+'</table><p class="note">합성 원본의 동일한 유효 마스크 샘플 · 3시드 평균 · 실사 경계 분류 정확도가 아닙니다.</p></section>'
    body=''.join('<h2>'+html.escape(x[3:])+'</h2>' if x.startswith('## ') else '<p>'+html.escape(x)+'</p>' for x in lines[1:] if x and not x.startswith('|'))
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V17 해상도와 가시 마스크 — 수정 후 검증</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1380px;margin:auto;padding:26px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}h2{margin-top:26px}select{padding:8px;margin:6px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}td:first-child,th:first-child{text-align:left}.scroll{overflow:auto}.note{color:#596579}img{width:100%;height:auto}a{color:#0369a1}.verdict{border-left:5px solid #b45309}</style><h1>RGB 세부 정보 × 가시 실루엣 정답</h1><p>6구조 × 3시드 · 같은 후보와 학습 순서 · 마스크 해독 오류 수정 후 결과</p><section class="verdict"><b>'''+verdict+'''</b><p>해상도·마스크 변경 모두 추가 개선 기준 통과 조건은 없었습니다. 높이선 분류는 소폭 개선됐지만 실제 점 오차의 안정적 개선으로 이어지지 않았습니다. 아래는 수정 후 유효한18개 모델과 기존 방법의 결과입니다.</p><div class="scroll">'''+table+'''</div><p class="note">3시드 평균px, 작을수록 좋음. 원본은 프레임 평균, 나머지는 해당 가림/잘림 대상점 평균입니다.</p></section><section><h2>두 요인을 분리한 효과</h2><img src="effects.png"><p>0의 왼쪽이 개선입니다. 세션 구간과 모든 시드1% 이상 개선을 함께 요구합니다. 다중 비교 보정 없는 탐색적 구간입니다.</p></section><section><h2>전체 조건과 지표</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임 평균px</option><option value="outside">화면 밖 점 평균px</option><option value="hidden">가려진 점 평균px</option><option value="frame_p90_px">프레임P90 px</option><option value="pck5">전체 점PCK5</option><option value="pck10">전체 점PCK10</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div class="scroll"><table id="table"></table></div><p id="counts" class="note"></p></section><section><h2>입력 세부 정보와 보조 정답이 달라진 부분</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p class="note">고정 SHA로 고른 합성 학습 예시입니다. 실제 읽는 RGB와 마스크 정답을 보여주며 attention 설명이나 실제 평가 성공 사례가 아닙니다. 회색은 입력 범위 밖입니다.</p></section>'''+role_table+'''<section>'''+body+'''<p><a href="REPORT_KO.md">보고서</a> · <a href="DESIGN_KO.md">설계</a> · <a href="SUMMARY.json">전체 수치</a> · <a href="DIAGNOSIS.json">역할 분류</a> · <a href="MASK_DECODING_AUDIT.json">마스크 해독 검산</a> · <a href="AUDIT.json">좌표·지표 검산</a> · <a href="INTEGRATION_CHECK.json">통합 검사</a> · <a href="PROTOCOL.json">사전 기준</a></p></section><script>
const data='''+json.dumps(s['results'],ensure_ascii=False)+''',labels='''+json.dumps(LABELS,ensure_ascii=False)+''',names='''+json.dumps(NAMES,ensure_ascii=False)+''',gallery='''+json.dumps(g,ensure_ascii=False)+''';
const params=new URLSearchParams(location.search);for(const id of ['split','seed','metric'])if(params.has(id))document.getElementById(id).value=params.get(id);
function render(){const split=document.getElementById('split').value,seed=document.getElementById('seed').value,metric=document.getElementById('metric').value;let out='<tr><th>방법</th>'+Object.values(labels).map(x=>'<th>'+x+'</th>').join('')+'</tr>';for(const a of Object.keys(names)){out+='<tr><td>'+names[a]+'</td>';for(const c of Object.keys(labels)){const r=data[split+'/'+c],rs=a==='stock'?[r.stock]:(seed==='mean'?[1,2,3]:[+seed]).map(s=>r[a+'_seed'+s]),v=rs.map(x=>['hidden','outside'].includes(metric)?x.strata[metric].mean_px:metric.startsWith('pck')?x.strata.all[metric]:x[metric]);out+='<td>'+(v.some(x=>x===null)?'—':(v.reduce((x,y)=>x+y,0)/v.length).toFixed(metric==='full_normalized_mean'?6:3))+'</td>';}out+='</tr>';}document.getElementById('table').innerHTML=out;document.getElementById('counts').textContent='검출: '+Object.keys(labels).map(c=>labels[c]+' '+data[split+'/'+c].stock.detected_frames+'/'+data[split+'/'+c].stock.frames).join(' · ');}
for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.id;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · 원본/실루엣 대응과 저·고해상도 입력';}sel.onchange=show;show();</script></html>'''
    (RUN/'index.html').write_text(page);write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY.json'),source_sha256=sha(Path(__file__)),figures=len(g)))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)

if __name__=='__main__':main()
