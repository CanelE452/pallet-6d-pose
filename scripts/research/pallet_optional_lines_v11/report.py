"""Source-backed Korean report and inspectable point/line overlays."""
import html
import importlib.util
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

LABELS={'clean':'원본','occlude20':'중앙 가림 20%','occlude40':'중앙 가림 40%',
        'crop20':'잘림 20%','crop40':'잘림 40%','corner20':'코너 주변 가림 (추가 진단)'}
NAMES={'stock':'기존 YOLO','v10_local':'기존 점 보정','local_extra':'선 없는 추가 보정',
       'lines_frozen':'고정 Hough 추가','lines_joint':'Hough 공동학습'}
COLORS={'stock':'#f59e0b','v10_local':'#3b82f6','local_extra':'#06b6d4','lines_frozen':'#8b5cf6','lines_joint':'#ef4444'}


def average(rows,arm,field='frame_mean_px'):
    return rows['stock'][field] if arm=='stock' else float(np.mean([rows[f'{arm}_seed{s}'][field] for s in SEEDS]))


def corner_diagnostics(results):
    diagnostics={}
    for key,rows in results.items():
        split,condition=key.split('/');manifest=read(RUN/'cache'/split/condition/'manifest.json')
        targets=read(RUN/'targets'/f'{split}_{condition}.json')['rows'];gt=np.array([r['points'] for r in targets]);valid=np.array([r['valid'] for r in targets])
        pred={a:[np.load(RUN/'predictions'/split/condition/(('stock' if a=='stock' else f'{a}_seed{s}')+'.npz')) for s in SEEDS] for a in ('stock','v10_local','local_extra','lines_joint')}
        det=pred['stock'][0]['detected'];sessions=np.array([r['session'] for r in targets]);subgroups={}
        for stratum in ('hidden','outside','visible'):
            hidden=np.array([r['hidden'] for r in targets]);outside=np.array([r['outside'] for r in targets])
            mask=valid&({'hidden':hidden,'outside':outside,'visible':~hidden&~outside}[stratum])
            if not mask.any():continue
            eligible=mask&det[:,None];groupcomp={}
            for a,b in [('v10_local','stock'),('lines_joint','local_extra')]:
                ea=np.stack([np.linalg.norm(p['points']-gt,axis=-1) for p in pred[a]])
                eb=np.stack([np.linalg.norm(p['points']-gt,axis=-1) for p in pred[b]])
                diff=np.mean(ea-eb,axis=0);sums=np.where(eligible,diff,0.).sum(-1);counts=eligible.sum(-1)
                if split=='real':
                    groups=np.unique(sessions);sums=np.array([sums[sessions==g].sum() for g in groups]);counts=np.array([counts[sessions==g].sum() for g in groups])
                keep=counts>0;sums=sums[keep];counts=counts[keep]
                if not len(counts):continue
                rng=np.random.default_rng(112233);ix=rng.integers(0,len(counts),(5000,len(counts)))
                boots=sums[ix].sum(-1)/counts[ix].sum(-1)
                groupcomp[f'{a}_vs_{b}']=dict(mean_point_difference_px=float(sums.sum()/counts.sum()),ci95=np.quantile(boots,[.025,.975]).tolist(),
                    per_seed_difference_px=[float((ea[s]-eb[s])[eligible].mean()) for s in range(3)],clusters=len(counts))
            subgroups[stratum]=dict(total_points=int(mask.sum()),detected_points=int(eligible.sum()),comparisons=groupcomp)
        diagnostics[key]=subgroups
    write(RUN/'CORNER_DIAGNOSTICS.json',dict(posthoc=True,results=diagnostics,
        scope='Shared original GT, manual clicks only on real; paired point-mean error with session bootstrap for real, frame bootstrap for synthetic. Descriptive subgroup intervals; no model selection.'))
    return diagnostics


def draw_line(ax,l,w,h,**kwargs):
    a,b,c=l;pts=[]
    if abs(b)>1e-9:
        for x in (0,w):
            y=-(a*x+c)/b
            if 0<=y<=h:pts.append((x,y))
    if abs(a)>1e-9:
        for y in (0,h):
            x=-(b*y+c)/a
            if 0<=x<=w:pts.append((x,y))
    if len(pts)>=2:ax.plot([pts[0][0],pts[1][0]],[pts[0][1],pts[1][1]],**kwargs)


def gallery(results):
    figdir=RUN/'figures';figdir.mkdir(exist_ok=True);records=[]
    for condition in (*CONDITIONS,'corner20'):
        split='real';manifest=read(RUN/'cache'/split/condition/'manifest.json');targets=read(RUN/'targets'/f'{split}_{condition}.json')['rows']
        paths={a:RUN/'predictions'/split/condition/(('stock' if a=='stock' else f'{a}_seed1')+'.npz') for a in ('stock','v10_local','local_extra','lines_joint')}
        preds={a:np.load(p) for a,p in paths.items()};gt=np.array([r['points'] for r in targets]);valid=np.array([r['valid'] for r in targets])
        eligible=np.flatnonzero(preds['stock']['detected'])
        diff=np.linalg.norm(preds['lines_joint']['points']-gt,axis=-1)-np.linalg.norm(preds['local_extra']['points']-gt,axis=-1)
        frame=np.where(valid,diff,0.).sum(-1)/valid.sum(-1)
        selected=[(int(sorted(eligible,key=lambda i:digest('v11-gallery:'+manifest['rows'][i]['id']))[0]),'SHA 고정 예시'),
                  (int(eligible[np.argmin(frame[eligible])]),'선 추가 최대 개선'),(int(eligible[np.argmax(frame[eligible])]),'선 추가 최대 악화')]
        if condition=='corner20':
            hidden=np.array([r['hidden'] for r in targets])&valid;available=eligible[hidden[eligible].any(-1)]
            selected[0]=(int(sorted(available,key=lambda i:digest('v11-hidden:'+manifest['rows'][i]['id']))[0]),'가려진 수동 GT가 있는 고정 예시')
        for index,selection in selected:
            rec=manifest['rows'][index];im=cv2.imread(str(ROOT/rec['image']))
            if rec['padded']:im=im[100:-100,100:-100]
            canvas=im.copy();origin=np.array(rec['origin'])
            if rec['mask'] is not None:
                x1,y1,x2,y2=rec['mask'];canvas[y1:y2,x1:x2]=rec['fill_bgr']
            if condition.startswith('crop'):
                canvas=canvas[:,rec['cut']:] if rec['crop_side']=='left' else canvas[:,:rec['cut']]
            fig,axes=plt.subplots(1,3,figsize=(18,5.6));h,w=canvas.shape[:2]
            axes[0].imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));axes[0].set_title('원본 · 직접 클릭한 GT')
            og=gt[index]+origin;v=valid[index];axes[0].scatter(og[v,0],og[v,1],s=65,c='#22c55e',marker='+',linewidths=2)
            for j in np.flatnonzero(v):axes[0].annotate(str(j),og[j],color='#22c55e',fontsize=10)
            for ax in axes[1:]:ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));ax.set_facecolor('#303946')
            axes[1].set_title('변형 입력 · 실제 크기의 점 보정');axes[2].set_title('선 후보에 준 가중치 · 공동학습 seed 1')
            g=gt[index];axes[1].scatter(g[v,0],g[v,1],s=70,c='#22c55e',marker='+',linewidths=2,label='직접 클릭 GT')
            for a in ('stock','v10_local','local_extra','lines_joint'):
                pp=preds[a]['points'][index];axes[1].scatter(pp[v,0],pp[v,1],s=32,facecolors='none',edgecolors=COLORS[a],label=NAMES[a])
            axes[1].legend(fontsize=8,loc='best')
            hidden=np.array(targets[index]['hidden'])&v
            choices=np.flatnonzero(hidden if hidden.any() else v);corner=int(choices[np.argmax(np.abs(diff[index,choices]))])
            p=preds['lines_joint'];weights=p['weights'][index,corner,:-1];modes=np.argsort(weights)[-3:][::-1]
            palette=['#facc15','#f472b6','#38bdf8']
            for rank,(mode,color) in enumerate(zip(modes,palette)):
                draw_line(axes[2],p['lines'][index,mode],w,h,color=color,alpha=.9,linewidth=1.6,label=f'후보 {mode}: {weights[mode]:.3f}')
            axes[2].scatter(g[corner,0],g[corner,1],marker='+',s=100,color='#22c55e',label=f'GT 점 {corner}')
            for a in ('local_extra','lines_joint'):
                pp=preds[a]['points'][index,corner];axes[2].scatter(*pp,s=50,facecolors='none',edgecolors=COLORS[a],label=NAMES[a])
            axes[2].legend(fontsize=8,loc='best')
            gate=float(p['gate'][index,corner,0]);null=float(p['weights'][index,corner,-1])
            axes[2].set_xlabel(f'추가 gate={gate:.3f}, null={null:.3f} · 정확도 확률 아님')
            # Include out-of-frame GT coordinates in the crop overlays, grey outside actual image.
            allxy=np.concatenate([g[v]]+[preds[a]['points'][index,v] for a in ('stock','lines_joint')])
            lo=np.minimum(allxy.min(0)-15,[0,0]);hi=np.maximum(allxy.max(0)+15,[w,h])
            for ax in axes[1:]:ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
            for ax in axes:ax.tick_params(labelsize=8)
            fig.suptitle(f'{LABELS[condition]} · {selection}\n{rec["id"]} · 선 추가 - 선 없는 보정: {frame[index]:+.3f}px',fontsize=13)
            fig.tight_layout();name=f'{condition}_{len(records):02d}.png';fig.savefig(figdir/name,dpi=125);plt.close(fig)
            records.append(dict(condition=condition,id=rec['id'],selection=selection,path='figures/'+name,
                seed=1,corner=corner,frame_difference_px=float(frame[index]),gate=gate,null=null,
                selected_by_gt_for_explanation=True,scope='Visualization only. No GT at inference. Original-scale displacement, no magnification.'))
    write(RUN/'GALLERY.json',dict(rows=records));return records


def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
    main=read(RUN/'SUMMARY.json');extra=read(RUN/'TARGETED_SUMMARY.json');audit=read(RUN/'AUDIT.json');assert audit['status']=='PASS'
    results=dict(main['results'],**extra['results']);comparisons=dict(main['comparisons'],**extra['comparisons']);diag=corner_diagnostics(results)
    gallery_rows=gallery(results)
    real=results['real/clean'];base_gain=[100*(1-real[f'v10_local_seed{s}']['frame_mean_px']/real['stock']['frame_mean_px']) for s in SEEDS]
    verdict=dict(status='REAL_LOCAL_GAIN_REPRODUCED_NO_PRACTICAL_HOUGH_GAIN_UNDER_STRESS',
        real_local_gain_percent=base_gain,primary_gates=main['gates'],
        natural_occlusion_established=False,targeted_corner_diagnostic_posthoc=True,
        conclusions=['실제 원본에서 기존 점 보정은 세 시드 모두 YOLO보다 개선.',
            '공동학습 선 추가는 실제 원본/중앙가림 평균에서 작은 이득; 세 시드 모두 1% 미만.',
            '실제로 가려진 코너의 개선은 확인되지 않으며 기존 점 보정도 표적 코너 가림에서 악화.',
            '잘림40%에서 선 추가는 같은 예산의 선 없는 추가 보정보다 세 시드 모두 악화.',
            '잘림20%에서는 기준 YOLO를 일관되게 넘지 못함. 자연 가림 또는 전체6D 개선 주장 불가.'])
    write(RUN/'CONCLUSION.json',verdict)
    text=['# 실제 이미지와 가림·잘림: 점 보정과 선택적 Hough 추가',
        '', '실제 원본에서는 점 보정의 개선을 재현했다. 선 추가의 실용적인 이득과 가려진 코너 복구는 확인하지 못했다.', '',
        'YOLO와 V10 점 보정을 고정하고, 선택적 추가 보정만 학습했다. 선 없는 추가 보정 / 고정 DHT / DHT 공동학습을 각각 3개 시드, 2,000업데이트로 비교했다. 총 9모델·18,000업데이트. 실제 이미지는 학습에 사용하지 않았다.',
        '', '## 실제 결과 (원본 297장, 9세션, 직접 클릭 1,251점)',
        '', '3개 시드 평균의 프레임별 점 오차(px). 검출된 동일 이미지에서 비교하며 작을수록 좋다. 아래 검출 수에서 빠진 프레임도 별도의 전체 모집단 정규화 지표에는 실패1로 포함했다.', '',
        '| 조건 | 검출 | 기존 YOLO | 기존 점 보정 | 선 없는 추가 보정 | 고정 Hough | Hough 공동학습 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for condition in (*CONDITIONS,'corner20'):
        r=results[f'real/{condition}'];vals=[average(r,a) for a in NAMES]
        text.append('| '+LABELS[condition]+' | '+str(r['stock']['detected_frames'])+'/297 | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    text+=['','원본에서 기존 점 보정의 개선은 시드별 '+', '.join(f'{v:.2f}%' for v in base_gain)+'였다. Hough 공동학습의 추가 이득은 원본과 중앙가림 조건에서 각 시드 1% 미만이었다. 사전 등록한 실용성 기준(모든 시드 ≥1% 개선 등)을 통과한 조건은 없다.',
        '', '## 가림 이미지 평균과 실제로 가려진 점은 다르다', '',
        '| 대상 점 | 검출된 점 수 | 기존 YOLO | 기존 점 보정 | 선 없는 추가 보정 | Hough 공동학습 |',
        '|---|---:|---:|---:|---:|---:|']
    for condition,stratum in [('occlude40','hidden'),('corner20','hidden'),('crop40','outside')]:
        r=results[f'real/{condition}'];values=[]
        for arm in ('stock','v10_local','local_extra','lines_joint'):
            values.append(r['stock']['strata'][stratum]['mean_px'] if arm=='stock' else np.mean([r[f'{arm}_seed{s}']['strata'][stratum]['mean_px'] for s in SEEDS]))
        text.append('| '+LABELS[condition]+' / '+('가려진 점' if stratum=='hidden' else '화면 밖 점')+' | '+str(r['stock']['strata'][stratum]['detected_points'])+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    text+=['', '중앙가림20/40%는 실제 클릭 코너를 10/63개만 가렸다. 이를 확인한 뒤 코너 주변 가림을 추가했다. 이 추가 진단은 이미 학습한 모든 모델을 그대로 사용했고, GT 대신 원본 YOLO의 예측 코너에 패치를 놓았다. 284개 수동 GT 점이 가려졌고, 그중 검출된 프레임의 280점을 비교했다. 이 조건에서 점 보정은 가려진 점의 오차를 오히려 키웠으며, 선 추가도 이를 안정적으로 회복하지 못했다.',
        '', '## 작은 평균 이득을 선의 유효성으로 단정하지 않는 이유', '',
        '같은 공동학습 모델에 다른 이미지의 선을 넣는 비교에서 원본/중앙가림의 효과 부호가 시드마다 섞였다. 선 정보를 전혀 안 쓴다는 뜻은 아니지만, 작은 평균 이득이 올바른 선 덕분이라는 인과 증거는 약하다. 공개된 gate/attention 값은 정답 확률이 아니다.',
        '', '잘림40%에서 화면 안 점은 선으로 조금 좋아져도 화면 밖 점은 더 나빠져 전체 이득을 상쇄했다. 가려진 점과 화면 밖 점에는 국소 영상 근거가 부족하며, 이번의 작은 방향별 추가 보정이 그 위치를 복원하지 못했다. 이는 관측 결과에 맞는 설명이지 모든 실패 원인을 증명한 것은 아니다.',
        '', '## 검증과 범위', '',
        f'- 독립 검산: {audit["prediction_frame_rows"]:,}개 프레임·모델·조건 행. 좌표 재구성 최대 차이 {audit["maxima"]["coordinate_replay_px"]:.8f}px; 점 오차 집계는 부동소수점 오차 범위에서 일치.',
        '- 실제 GT는 keypoint_annotations의 manual_click만 사용했다. PnP 자동 투영점은 주 지표에서 제외했고, 원본 주석을 수정하거나 GT로 점 대응을 바꾸지 않았다. 가림·잘림 뒤에도 원래 클릭 위치를 유지했다.',
        '- 모든 모델에 같은 현재 입력의 YOLO 예측과 검출 실패를 적용했다. 검출 선택은 최고 confidence≥0.25이며 GT 매칭은 없다. 합성 검증은 이전 검출 조건으로 골랐던 505장의 재사용이다.',
        '- 실제 세트는 기존 DEV 297장/9세션 재사용이다. 자연 가림 라벨은 297장 모두 unknown, 자연 잘림 표시는 1장뿐이다. 인공 가림·잘림 결과를 자연 현장 일반화로 해석하지 않는다.',
        '- 잘림은 실제 이미지를 잘라 재추론했다. 정규화 분모는 원본 이미지 대각선이며, 기존 reflect100 입력 패딩은 잘린 경계에도 유지했다. 잘린 내용의 반사 복제 영향은 이 실험에서 분리하지 않았다.',
        '- Bootstrap은 실제는 세션 단위, 합성은 원본 이미지 단위다. 시드는 프레임 내에서 평균했다. 소수 세션, 재사용 평가와 사후 하위집단 분석이므로 확정적 일반화 검증은 아니다.',
        '- 추가 보정 모듈은 58,852개 파라미터. 공동학습은 DHT 143,900개를 추가 학습하므로 선 없는 대조군과 총 학습 가능 용량이 같지는 않다. 세 시드의 체크포인트가 서로 다른 것도 확인했다.',
        '- 현재 실제 2D 수치와 예전 41.49px 또는 전체 6D 결과를 직접 비교하지 않는다. 배포 모델은 변경하지 않았다.',
        '', '## 다음 판단', '',
        '원본 실제 영상의 점 보정은 유효한 후보지만, 가려진 코너에서 악화가 있어 그대로 전 조건에 적용할 근거는 부족하다. 다음 설계에서는 추가 선 분기만 선택하게 할 것이 아니라 기존 국소 점 보정 자체도 관측 가능성에 따라 줄일 수 있어야 한다. 화면 밖 코너 복구는 별도의 amodal 위치·구조 추론 문제로 다뤄야 한다. 이번 결과는 Hough의 원리적 불가능성을 뜻하지 않는다.',
        '', '원시 결과: SUMMARY.json, TARGETED_SUMMARY.json, CORNER_DIAGNOSTICS.json, AUDIT.json. 실행 코드: scripts/research/pallet_optional_lines_v11/.']
    (RUN/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,split in zip(axes,('real','synth_val')):
        conditions=(*CONDITIONS,'corner20');x=np.arange(len(conditions))
        for offset,arm in zip((-.22,0,.22),('v10_local','local_extra','lines_joint')):
            gains=[100*(1-average(results[f'{split}/{c}'],arm)/average(results[f'{split}/{c}'],'stock')) for c in conditions]
            ax.bar(x+offset,gains,.22,label=NAMES[arm],color=COLORS[arm])
        ax.axhline(0,color='#475569',lw=1);ax.set_xticks(x,[LABELS[c].replace(' (추가 진단)','') for c in conditions],rotation=25,ha='right')
        ax.set_title(('실제' if split=='real' else '합성')+' · 검출된 프레임의 점 오차 개선율');ax.set_ylabel('YOLO 대비 개선율 % (음수는 악화)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(RUN/'figures'/'overview.png',dpi=140);plt.close(fig)
    body=[]
    for line in text:
        if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('|'):continue
        elif line:body.append('<p>'+html.escape(line)+'</p>')
    # Tables use embedded data so file:// works without fetch permissions.
    dataset=json.dumps(results,ensure_ascii=False,allow_nan=False);gallery_json=json.dumps(gallery_rows,ensure_ascii=False)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>실제·가림·잘림 — 점과 선 실험 결과</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1250px;margin:auto;padding:28px}h1{font-size:30px}h2{margin-top:36px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}select{padding:9px;margin:6px;font-size:16px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}a{color:#0369a1}.note{color:#596579}#galleryCaption{font-size:14px}button{padding:10px;margin:8px}</style>
<h1>점 보정은 실사에서도 개선 · 가려진 코너 복구는 미확인</h1><p>고정 YOLO + 보존된 점 보정 + 선택적 추가 보정. 9모델, 18,000업데이트, 실제 학습 0장.</p>
<section><h2>조건별 비교</h2><label>영상<select id="split"><option value="real">실제 297장</option><option value="synth_val">합성 505장</option></select></label><label>집계<select id="seed"><option value="mean">3개 시드 평균</option><option value="1">seed 1</option><option value="2">seed 2</option><option value="3">seed 3</option></select></label><label>지표<select id="metric"><option value="frame_mean_px">검출된 프레임 점 오차 px</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option><option value="hidden">실제로 가려진 점 오차 px</option><option value="outside">화면 밖 점 오차 px</option></select></label><div style="overflow-x:auto"><table id="table"></table></div><p class="note">작을수록 좋음. 실제는 직접 클릭한 점만 평가. 가려진 점/화면 밖 점은 점 단위 평균이며 전체 프레임 지표와 다릅니다. 검출 실패 포함 지표는 실패=1, 원본 대각선 정규화입니다. 코너 주변 가림은 모델을 바꾸지 않은 사후 추가 진단입니다.</p></section>
<section><img src="figures/overview.png" alt="조건별 개선율"></section>
<section><h2>실제 이미지와 점별 선 후보</h2><select id="gallerySelect"></select><p id="galleryCaption"></p><a id="galleryLink" target="_blank"><img id="galleryImage" alt="실제 점 보정과 선 후보 시각화"></a><p class="note">GT는 녹색 +. 최대 개선·악화 예시는 설명을 위해 GT로 선택했습니다. 추론에는 GT를 사용하지 않았습니다. 잘림 화면 바깥의 회색 영역은 모델이 볼 수 없는 곳입니다.</p></section>
<section>'''+''.join(body)+'''<p><a href="REPORT_KO.md">전체 보고서</a> · <a href="SUMMARY.json">주 실험 수치</a> · <a href="TARGETED_SUMMARY.json">코너 가림 추가 진단</a> · <a href="CORNER_DIAGNOSTICS.json">가려진 점·화면 밖 점 검산</a> · <a href="AUDIT.json">독립 검산</a></p></section><script>
const data='''+dataset+''',gallery='''+gallery_json+''',labels='''+json.dumps(LABELS,ensure_ascii=False)+''',names='''+json.dumps(NAMES,ensure_ascii=False)+''';
function render(){const split=document.querySelector('#split').value,seed=document.querySelector('#seed').value,metric=document.querySelector('#metric').value;let h='<tr><th>조건</th><th>검출</th>'+Object.values(names).map(n=>'<th>'+n+'</th>').join('')+'</tr>';for(const c of Object.keys(labels)){const r=data[split+'/'+c];h+='<tr><td>'+labels[c]+'</td><td>'+r.stock.detected_frames+'/'+r.stock.frames+'</td>';for(const a of Object.keys(names)){const cells=a==='stock'?[r.stock]:((seed==='mean'?[1,2,3]:[+seed]).map(s=>r[a+'_seed'+s]));const vals=cells.map(v=>['hidden','outside'].includes(metric)?v.strata[metric].mean_px:v[metric]);const v=vals.some(x=>x===null)?null:vals.reduce((x,y)=>x+y,0)/vals.length;h+='<td>'+(v===null?'—':v.toFixed(metric==='full_normalized_mean'?6:3))+'</td>';}h+='</tr>';}document.querySelector('#table').innerHTML=h;}
for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();
const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=labels[r.condition]+' · '+r.selection;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('galleryCaption').textContent=r.id+' · seed 1 · 선 추가 − 선 없는 보정 '+r.frame_difference_px.toFixed(3)+'px (음수가 개선)';}sel.onchange=show;show();</script></html>'''
    (RUN/'index.html').write_text(page)
    write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),gallery_images=len(gallery_rows),
        primary_summary_sha256=sha(RUN/'SUMMARY.json'),targeted_summary_sha256=sha(RUN/'TARGETED_SUMMARY.json'),source_sha256=sha(Path(__file__))))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)


if __name__=='__main__':main()
