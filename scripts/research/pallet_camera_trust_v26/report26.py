"""Source-driven figures and Korean report; oracles explicitly separated."""
from common26 import *
import numpy as np,cv2,markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS'
    s=read(RUN/'TRUST_SUMMARY.json');cam=read(RUN/'CAMERA_SUMMARY.json');per=read(RUN/'CAMERA_PER_IMAGE.json')['rows'];m=read(RUN/'cache/manifest.json')['rows'];figdir=RUN/'figures';figdir.mkdir(exist_ok=True)
    selected=[r for r in s['rows'] if (r['split'],r['condition'],r['subset'])==('real','crop20','outside')]
    def group(pg='all',lg='all'):return [r for r in selected if r['point_quality']==pg and r['line_quality']==lg]
    def avg(key,pg='all',lg='all'):return float(np.mean([r[key] for r in group(pg,lg)]))
    quadrants=[]
    for pg,lg in [('bad','bad'),('bad','good'),('good','bad'),('good','good')]:
        quadrants.append(dict(point=pg,line=lg,counts=[r['points'] for r in group(pg,lg)],**{k:avg(k,pg,lg) for k in ['base_px','projection_px','normal_error_px','tangent_error_px','null_preferred','height_edge']}))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');q=quadrants;x=np.arange(4)
    axes[0].bar(x-.18,[r['base_px'] for r in q],.36,label='Point baseline');axes[0].bar(x+.18,[r['projection_px'] for r in q],.36,label='Forced top non-null line');axes[0].set_ylabel('Mean point error (px)');axes[0].legend(fontsize=8)
    axes[1].bar(x-.18,[r['normal_error_px'] for r in q],.36,label='Normal to reference edge');axes[1].bar(x+.18,[r['tangent_error_px'] for r in q],.36,label='Along reference edge');axes[1].set_ylabel('Mean absolute error component (px)');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(x,['Bad point\nbad line','Bad point\ngood line','Good point\nbad line','Good point\ngood line']);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Reused real DEV / crop20 outside / equal mean of 3 seeds\nQuality strata use GT; forced projection is NOT the learned model');fig.savefig(figdir/'trust.png',dpi=170);plt.close(fig)
    keys=['base_px','projection_px','oracle_predicted_px','oracle_exact_incident_px'];vals=[avg(k) for k in keys];fig,ax=plt.subplots(figsize=(9,4),layout='constrained');bars=ax.barh(np.arange(4),vals,color=['#357bb0','#c55a47','#999999','#999999']);bars[2].set_hatch('//');bars[3].set_hatch('//');ax.set_yticks(np.arange(4),['Point baseline','Forced top non-null line','GT best of baseline + top3 projections','GT best exact incident-line projection']);ax.invert_yaxis();ax.set_xlabel('Error (px), lower is better');ax.set_xlim(0,18)
    for i,v in enumerate(vals):ax.text(v+.15,i,f'{v:.3f}',va='center')
    ax.set_title('557 supported points per seed / last two bars use GT at selection\nOracle capacity is not achieved inference performance',fontsize=11);fig.savefig(figdir/'oracle.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.2),layout='constrained');ax.axhline(0,color='#267c4c',label='Exact line y=0');ax.scatter([0,12,12],[0,2,0],c=['#267c4c','#356ca3','#bb4c45'],s=65);ax.annotate('GT (0,0)',(0,0),xytext=(0,-1.3));ax.annotate('Point (12,2)',(12,2),xytext=(8,3.2));ax.annotate('After projection (12,0)',(12,0),xytext=(7,-1.3));ax.annotate('',(12,0),(12,2),arrowprops=dict(arrowstyle='->',color='#bb4c45'));ax.set(xlim=(-2,16),ylim=(-2,4),xlabel='Tangent direction: 12 px remains',ylabel='Normal');ax.set_aspect('equal');ax.set_title('A perfectly correct line does not determine position along itself');fig.savefig(figdir/'geometry.png',dpi=170);plt.close(fig)
    conditions=['clean','crop_left20','crop_right20','physical_panel'];heights=[.12,.4,.9];yaws=[-30,-10,0,10,30]
    camera_figures=[]
    for arm in ['stock','base','line_vector','gate_paired','vector_local_edge','vector_line_edge']:
        fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
        for ax,c in zip(axes.flat,conditions):
            vals=np.full((3,5),np.nan);counts=np.zeros((3,5),int)
            for i,h in enumerate(heights):
                for j,y in enumerate(yaws):
                    rr=[r for r in cam['rows'] if (r['arm'],r['condition'],r['height'],r['yaw_deg'],r['subset'])==(arm,c,h,y,'front')];counts[i,j]=rr[0]['detected'];vv=[r['mean_px'] for r in rr if r['mean_px'] is not None];vals[i,j]=np.mean(vv) if vv else np.nan
            im=ax.imshow(np.ma.masked_invalid(vals),cmap='YlOrRd',vmin=0,vmax=25,aspect='auto')
            for i in range(3):
                for j in range(5):ax.text(j,i,f'{vals[i,j]:.1f} px\n{counts[i,j]}/6' if counts[i,j] else 'MISS\n0/6',ha='center',va='center',fontsize=9,color='white' if vals[i,j]>17 else 'black')
            ax.set_xticks(range(5),yaws);ax.set_yticks(range(3),heights);ax.set_xlabel('Camera yaw (deg)');ax.set_ylabel('Camera height (m)');ax.set_title(c)
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Detected front-4 point error (px); color capped at 25');fig.suptitle(f'{arm}: frozen models / 2 assets x 3 colors per camera\nConditional errors with detection counts; missing cases excluded ONLY from px mean');name=f'camera_{arm}.png';fig.savefig(figdir/name,dpi=150);plt.close(fig);camera_figures.append(name)
    renders=read(RUN/'RENDER_COMPLETE.json')['rows'];gallery=[];fig,axes=plt.subplots(3,4,figsize=(15,8),layout='constrained')
    for i,h in enumerate(heights):
        for j,(asset,yaw,state) in enumerate([('scene.usd',-30,'clean'),('scene.usd',0,'clean'),('scene.usd',30,'panel'),('scene_1.usd',0,'panel')]):
            r=next(r for r in renders if (r['asset'],r['appearance'],r['yaw_deg'],r['camera_height_m'],r['state'])==(asset,0,yaw,h,state));im=cv2.cvtColor(cv2.imread(str(ROOT/r['image'])),cv2.COLOR_BGR2RGB);axes[i,j].imshow(im);uv=np.array(r['points']);axes[i,j].scatter(uv[:,0],uv[:,1],s=10,c='#ffdd22');axes[i,j].set_xlim(80,560);axes[i,j].set_ylim(370,130);axes[i,j].axis('off');axes[i,j].set_title(f'{asset} yaw={yaw} h={h}\n{state} / visible={r["visible_fraction"]:.2f}',fontsize=9);gallery.append(dict(id=r['id'],image=r['image'],selection='Fixed green appearance, prescribed yaw/height/state; no error-based selection'))
    fig.suptitle('Actual CAD camera renders; yellow = projected cuboid GT (not physical edge labels)');fig.savefig(figdir/'gallery.png',dpi=160);plt.close(fig)
    fulltable=[]
    for arm in ['stock','base','line_vector','gate_paired','vector_local_edge','vector_line_edge']:
        row=dict(arm=arm)
        for c in conditions:
            pop=[r for r in per if (r['arm'],r['condition'],r['subset'])==(arm,c,'front')];byseed=[];norm=[]
            for seed in sorted({r['seed'] for r in pop}):
                rr=[r for r in pop if r['seed']==seed];byseed.append(np.mean([r['mean_px'] for r in rr if r['detected']]));norm.append(np.mean([r['full_normalized'] for r in rr]))
            row[c]=dict(px=float(np.mean(byseed)),full_normalized=float(np.mean(norm)))
        fulltable.append(row)
    detection={c:sum(r['detected'] for r in m if r['condition']==c) for c in conditions}
    lines=['# V26 카메라·가시 면과 점/선 신뢰도 진단','', '**진단 완료 · 신규 학습 0회 · 안정적 융합 개선은 아직 미해결.**','',
    '점과 선의 부정확함은 원인 중 하나다. 하지만 정확한 선도 **그 점에 대응하는 선인지, 오차를 줄이는 방향을 제약하는지**에 따라 가치가 다르다. 따라서 둘 중 하나를 통째로 더 믿는 방식보다, 점의 어떤 방향 성분을 어느 관측이 뒷받침하는지 구별해야 한다.','',
    '## 1. 실사에서 직접 확인한 것','',
    'V25와 같은 실사297장·9세션의 재사용 DEV, 잘림20% 화면 밖 점을 분석했다. 유효한 인접 GT 선분이 필요한 진단이므로 **전체558점 중557점/시드**만 포함한다. 아래는 각 시드 평균을 동등하게 평균했으며 같은 영상을 반복한3시드를 독립 표본으로 세지 않았다.','',
    '좋은 점: GT 오차≤5px. 좋은 선: 해당 점에 연결되는 유효 cuboid GT 선분의 **양 끝점과 예측 무한직선 사이 RMS≤5px**. 이는 물리적 픽셀 경계/가시성의 정답이 아니며, 수동 GT의 실제 오차가5px 이내라고 인증하는 기준도 아니다.','',
    '| GT로 나눈 상태 | 시드1/2/3 점 수 | 점 기준 | 강제 선 투영 | 선에 수직인 오차 | 선을 따라가는 오차 |','|---|---:|---:|---:|---:|---:|']
    names={'bad':'부정확','good':'정확'}
    for q in quadrants:lines.append(f'| 점 {names[q["point"]]}·선 {names[q["line"]]} | '+ '/'.join(map(str,q['counts']))+f' | {q["base_px"]:.3f} | {q["projection_px"]:.3f} | {q["normal_error_px"]:.3f} | {q["tangent_error_px"]:.3f} |')
    lines+=['','단위px. **강제 투영은 학습 모델의 출력이 아니다.** null 선택을 무시하고 가장 높은 non-null 선에 점을 수직 투영한 진단이며 이동은 box 대각선15%로 제한했다. 실제 모델에는 이미 null/보정 게이트가 있다. 수직·접선 성분은 해당 예측 선과 가장 잘 맞는 인접 GT 선분을 기준으로 분해했다.','',
    f'점은 부정확·선은 정확한 집합에서도 평균 오차가 **{avg("base_px","bad","good"):.2f}→{avg("projection_px","bad","good"):.2f}px**였다. 접선 성분 **{avg("tangent_error_px","bad","good"):.2f}px**가 수직 성분 **{avg("normal_error_px","bad","good"):.2f}px**보다 컸다. 정확한 선으로 당기는 것만으로는 남은 위치를 정하기 어렵다는 해석을 지지한다. 다만 시드별 집합은62/109/25점으로 다르고, 개선폭도 일정하지 않다.','',
    '![정확도별 투영 결과와 오차 방향](figures/trust.png)','', '![정확한 선에서도 접선 오차가 남는 예](figures/geometry.png)','',
    '## 2. 신뢰도를 제대로 고르면 가능한가','',
    '| 진단 | 평균 오차px | 추론 때 GT 필요 |','|---|---:|---|',
    f'| 점 기준 | {avg("base_px"):.3f} | 아니오 |',f'| 상위 non-null 선에 강제 투영 | {avg("projection_px"):.3f} | 아니오, 다만 실제 모델과 다른 진단 |',f'| 기준 유지+상위3개 선 투영 중 GT 최적 선택 | {avg("oracle_predicted_px"):.3f} | **예·oracle** |',f'| 정확한 인접 GT 선 투영 중 최적 선택+기준 유지 | {avg("oracle_exact_incident_px"):.3f} | **예·oracle** |','',
    '![후보 선택의 GT 이용 상한](figures/oracle.png)','',
    f'현재 후보들에서 GT로 정답 선택까지 해도 **{avg("base_px"):.2f}→{avg("oracle_predicted_px"):.2f}px**다. 후보 선택에 일부 여지는 있지만 이 숫자는 학습 가능한 성능을 보장하지 않는다. 정확한 GT 선 oracle와의 차이는 선의 위치·역할 대응·후보 구성도 개선할 여지가 있음을 보여준다. 정확한 선을 고르는 이 oracle도 단일 선 투영 후보만 비교하므로 모든 가능한 아키텍처의 한계가 아니다.','',
    '상위 선 association 점수로 강제 투영의 이득을 구별한 AUROC(시드1/2/3)는 '+ '/'.join(f'{r["association_help_auc"]:.3f}' for r in group())+'였다. 현재 점수는 보정 이득의 신뢰도로 검증되지 않았다. 단순 반전으로 해결된다고 결론 내릴 수도 없다. null을 더 선호한 비율도 '+ '/'.join(f'{r["null_preferred"]:.1%}' for r in group())+'로 크게 달라, 강제 투영의 악화를 기존 모델 전체의 악화로 해석하면 안 된다.','',
    'GT 허용오차 민감도(같은557점/시드):','', '| 선 허용 RMS | 해당 점에 맞는 선 수: 시드1/2/3 | 그 집합의 강제 투영 악화율: 시드1/2/3 |','|---|---:|---:|']
    for tol in (2,5,10):
        rr=[r for r in s['GT_tolerance_sensitivity'] if (r['split'],r['condition'],r['subset'],r['tolerance_px'])==('real','crop20','outside',tol)];lines.append(f'| {tol}px | '+ '/'.join(str(r['compatible_points']) for r in rr)+' | '+ '/'.join(f'{r["compatible_projection_harm_fraction"]:.1%}' for r in rr)+' |')
    lines+=['','이는 수동 GT 오차에 대한 보조 민감도 분석이다. 새 독립 재어노테이션을 수행하지 않았으므로 실제 annotation 불확실성 자체를 제거한 결과는 아니다.','',
    '## 3. 실제 카메라를 바꾼 실험','',
    '**실제 USD 메시2개×색3개×방위5개×높이3개**의90카메라, 물리 가림판 유무로180RGB를 새 렌더링했다. clean·좌측20%잘림·우측20%잘림·가림판의360관측에 YOLO와 기존 V18/V25의3시드 고정 모델을 추론했다. 카메라 방위는 −30/−10/0/10/30도, 높이는0.12/0.40/0.90m다. 모델 재학습·실사 GT 학습은 없다.','',
    '![실제 카메라와 물리 가림판 이미지](figures/gallery.png)','',
    '검출: '+', '.join(f'{c} **{detection[c]}/90**' for c in conditions)+'. **총172/360**으로 검출 실패가 많다. 새 렌더의 조명·평면 재질·배경은 기존 Isaac 학습 분포와 다르므로 실패를 카메라 각도 하나의 원인으로 확정하지 않는다.','',
    '![점 기준의 카메라별 오차와 검출 수](figures/camera_base.png)','',
    '정면 clean은 검출된 사례에서 잘 맞는 경우가 있다. 반면 같은 정면에서도 높이·가림·잘림에 따라 오차와 검출률이 변한다. 따라서 **“정면이라 항상 불가능” 또는 “정면 학습 데이터 수만 늘리면 해결”은 확인되지 않았다.** 높이·남은 면·외관을 함께 통제해야 한다.','',
    '아래는 앞면4점의 검출 조건부 평균 오차. 각 시드에서 같은 조건의90관측을 집계한 뒤 시드 평균했다. 결측은 이px 평균에 포함되지 않으므로 바로 아래 전체 분모 지표와 함께 본다.','',
    '| 고정 방법 | clean | 좌측 잘림20 | 우측 잘림20 | 물리 가림판 |','|---|---:|---:|---:|---:|']
    for r in fulltable:lines.append('| '+r['arm']+' | '+' | '.join(f'{r[c]["px"]:.3f}' for c in conditions)+' |')
    lines+=['','검출 실패를1, 검출은 min(점 오차/원본 대각선800px,1)로 둔 전체90관측 평균. 검출률과 위치 오차를 합친 이 보조 지표는 임의로 명시한 실패 비용이며 표준AP나 일반 실사 성능이 아니다.','', '| 고정 방법 | clean | 좌측 잘림20 | 우측 잘림20 | 물리 가림판 |','|---|---:|---:|---:|---:|']
    for r in fulltable:lines.append('| '+r['arm']+' | '+' | '.join(f'{r[c]["full_normalized"]:.4f}' for c in conditions)+' |')
    lines+=['','같은 메시/색/높이에서 정면−좌우30도 평균을 비교한 점 기준(seed1) 짝비교. 음수이면 정면의 오차가 더 작다. 검출된3구도가 모두 있는 짝만 쓰는px 결과와 전체18짝의 실패 포함 차이를 분리했다.','', '| 조건 | 공통 검출 짝/전체 | 정면−사선px | 전체 정규화 차이 |','|---|---:|---:|---:|']
    for r in cam['paired_camera_effects']:
        if r['arm']=='base' and r['seed']==1:lines.append(f'| {r["condition"]} | {r["matched_detected_triplets"]}/{r["requested_triplets"]} | {r["frontal_minus_oblique_px"]:.3f} | {r["frontal_minus_oblique_normalized"]:.4f} |')
    lines+=['','2메시×3색은6개의 독립 형상이 아니다. 표본이 작고 재질이 제한되므로 유의성/실사 일반화 주장은 하지 않는다. 깊이로 계산한 가시 비율은 같은 카메라의 고립 물체 픽셀 중 실제 관측된 비율이다. 빈 공간에 놓인 cuboid 모서리의 물리적 가시성을 뜻하지 않는다.','',
    '## 4. 검증과 다음 설계 판단','',
    '새 GT는 world cuboid→기존 camera-facing ID→화면 투영 순서로 만들고, 별도 OpenCV projectPoints로 재검사했다. 180개 마스크·cuboid 투영 범위, 360관측의 잘림 좌표,16개 예측 파일의 전체 모집단/검출 실패, 개별 영상 지표를 검사했다. 이전 V25 완료 파일도 변경하지 않았다. [감사 결과](AUDIT.json).','',
    '**이번에 해결한 범위:** 카메라 실제 개입을 실행했고, 점/선 정확도와 보정에 유용한 방향을 구분하는 진단을 만들었다. **미해결:** 이를 예측하는 새 신뢰도 모델의 학습/일반화, 물리적 가시 선·끝점 GT 감독, 봉인된 새 실사 세션의 안정적 추가 개선.','',
    '다음 학습은 다음 조건을 갖춘 작은 대조 실험이어야 한다.','',
    '1. **관측과 역할을 먼저 정의:** 합성에서 실제 보이는 긴 경계·짧은 높이선·끝점과 cuboid 점의 관계를 따로 표시한다. 보이지 않는 cuboid 선을 픽셀 경계로 감독하지 않는다. 긴 선만 남은 점은 접선 위치가 부족하다는 상태를 입력/감독에 포함한다.',
    '2. **방향별 효용을 학습:** 점 기준의 출력을 유지 후보로 두고, 대응 선의 법선 보정과 비평행 두 번째 선/끝점의 보정을 분리한다. 합성 GT에서 각 후보가 줄이는 점 오차와 이동량을 감독하고, 실사 GT는 평가에만 쓴다. 학습 때 정답만 보고 고른 선을 추론 입력으로 제공하지 않는다.',
    '3. **동일 예산의 대조와 중단 기준:** 점만·선 위치/역할 개선만·방향별 효용 추가를 비교한다. 별도 합성 원본과 재사용DEV에서 약한 잘림 보존 및 가림/큰 잘림의 기존 이득을 함께 확인한다. 후보 oracle 여지가 있는데도 효용 판별이 이전되지 않으면 신뢰도 구조의 일반화 실패로 분리하고, 새 실사 봉인평가 전 반복 튜닝을 끝낸다.','',
    '**중복 제안 주의:** V10에는 이미 선 법선 행렬과 null 선택이 있고 이후 V11~V25에서 게이트/방향 보정을 시험했다. 단순히 confidence gate나 법선 행렬을 다시 추가하는 것은 새 해결책이 아니다. 위 후속의 차이는 실제 가시 역할·두 번째 방향 단서·후보별 보정 효용 감독이다. 이것도 아직 성공이 보장된 설계는 아니다. 현재 결과만으로 gradient surgery가 이 관측 부족/접선 모호성을 해결한다고 볼 근거도 없다.','',
    '## 파일과 재현','', '[등록 계획](PROTOCOL.json) · [점/선 전체 표·정의](TRUST_SUMMARY.json) · [카메라 전체 표](CAMERA_SUMMARY.json) · [그림 입력 수치](FIGURE_DATA.json)','']
    for f in camera_figures:lines+= [f'![고정 방법 구도별 결과 {f}](figures/{f})','']
    report='\n'.join(lines);(RUN/'REPORT_KO.md').write_text(report)
    body=markdown.markdown(report,extensions=['tables','fenced_code']);html='<!doctype html><html lang="ko"><meta charset="utf-8"><title>V26 카메라와 점·선 신뢰도</title><style>body{font:17px/1.75 sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#183048}h1,h2{line-height:1.4}h2{margin-top:2.5em}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;border-bottom:1px solid #ccd5de;text-align:left}th{background:#eaf0f6}a{color:#096ba9}strong{color:#122e53}</style>'+body+'</html>';(RUN/'index.html').write_text(html)
    write(RUN/'FIGURE_DATA.json',dict(quadrants=quadrants,oracle_values={k:avg(k) for k in keys},camera_comparison=fulltable,detection=detection,gallery=gallery,sources={n:sha(RUN/n) for n in ['TRUST_SUMMARY.json','CAMERA_SUMMARY.json','CAMERA_PER_IMAGE.json','AUDIT.json']}))
    note=NOTE.read_text().split('## 결과')[0].replace('상태: 실행 중.','상태: 진단 완료 · 신규 학습 없음 · 안정적 융합은 미해결.');prefix='../../../../data/pallet/results/pallet_camera_trust_v26/'
    import re
    content=re.sub(r'\]\(([^)]+)\)',lambda m:']('+prefix+m.group(1)+')',report)
    NOTE.write_text(note+'## 결과\n\n[브라우저 보고서]('+prefix+'index.html) · [재현 스크립트](../../../../scripts/research/pallet_camera_trust_v26/README.md)\n\n'+content)
    write(RUN/'REPORT_COMPLETE.json',dict(complete=True,report_sha256=sha(RUN/'REPORT_KO.md'),figures=len(list(figdir.glob('*.png'))),source_sha256=sha(Path(__file__))));record('V26 실제 렌더 갤러리·방법별 카메라 히트맵·점/선4분면·oracle 상한·접선 모호성 그림과 한국어 MD/HTML 작성.');print('REPORT26',{k:avg(k) for k in keys},detection,flush=True)

if __name__=='__main__':main()
