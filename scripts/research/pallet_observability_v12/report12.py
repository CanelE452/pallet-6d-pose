"""Numerical diagnosis, Korean report and original-scale attention overlays."""
import html
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context12 import *

LABELS={'clean':'원본','occlude20':'중앙 가림20%','occlude40':'중앙 가림40%','crop20':'잘림20%','crop40':'잘림40%','corner20':'코너 주변 가림'}
NAMES={'stock':'기존 YOLO','local':'기존 점 보정','scalar_gate':'일괄 축소','learned_gate':'학습한 보정 억제','gated_amodal':'억제 + 별도 복원','always_amodal':'복원 항상 적용'}
COLORS={'stock':'#f59e0b','local':'#3b82f6','learned_gate':'#a855f7','gated_amodal':'#ef4444'}

def mean(rows,arm,metric='frame_mean_px',stratum=None):
    cells=[rows['stock']] if arm=='stock' else [rows[f'{arm}_seed{s}'] for s in SEEDS]
    v=[c['strata'][stratum]['mean_px'] if stratum else c[metric] for c in cells]
    return None if any(x is None for x in v) else float(np.mean(v))

def diagnosis(summary):
    result={}
    for condition in ('corner20','crop20','crop40'):
        t=read(V11/'targets'/f'real_{condition}.json')['rows'];gt=np.array([r['points'] for r in t]);v=np.array([r['valid'] for r in t]);kind=target_type(t).numpy()
        metadata=read(V11/'cache'/'real'/condition/'manifest.json')['rows']
        mask=v&(kind==(1 if condition=='corner20' else 2));rows=[]
        for seed in SEEDS:
            p=np.load(RUN/'predictions'/'real'/condition/f'gated_amodal_seed{seed}.npz');eligible=mask&p['detected'][:,None]
            old=p['original_points'].astype(float);before=old+p['keep']*(p['base_points']-old);delta=p['recovery'].astype(float)
            residual=gt-before;dot=(delta*residual).sum(-1);d2=(delta*delta).sum(-1)
            e0=np.linalg.norm(before-gt,axis=-1);e1=np.linalg.norm(p['points']-gt,axis=-1)
            # Exact squared-error decomposition, no GT-based inference action.
            replay=e0**2+d2-2*dot;maxerr=float(np.max(np.abs(replay[eligible]-e1[eligible]**2)))
            bad=(e1>e0)&eligible;wrong=(dot<=0)&bad;overshoot=(dot>0)&bad
            row=dict(seed=seed,points=int(eligible.sum()),mean_without_recovery_px=float(e0[eligible].mean()),
                mean_with_recovery_px=float(e1[eligible].mean()),mean_recovery_length_px=float(np.linalg.norm(delta,axis=-1)[eligible].mean()),
                worsened_points=int(bad.sum()),wrong_direction_points=int(wrong.sum()),overshoot_or_sideways_points=int(overshoot.sum()),
                new_catastrophes_5_to20=int(((e0<=5)&(e1>20)&eligible).sum()),squared_error_replay_max=maxerr)
            if condition.startswith('crop'):
                sign=np.array([-1. if r['crop_side']=='left' else 1. for r in metadata])[:,None]
                outward=delta[...,0]*sign;needed=residual[...,0]*sign
                row.update(moves_outward_fraction=float((outward[eligible]>0).mean()),
                    needs_inward_fraction=float((needed[eligible]<0).mean()),
                    outward_despite_inward_needed_points=int(((outward>0)&(needed<0)&eligible).sum()))
            rows.append(row)
        result[condition]=rows
    write(RUN/'FAILURE_DIAGNOSIS.json',dict(posthoc=True,rows=result,
        scope='Same-model recovery-off comparison from saved tensors. GT only for posthoc error decomposition, never used for model choice or inference.'))
    return result

def visualize(summary):
    directory=RUN/'figures';directory.mkdir(exist_ok=True);gallery=[]
    for condition in ('clean','corner20','crop20','crop40'):
        m=read(V11/'cache'/'real'/condition/'manifest.json');t=read(V11/'targets'/f'real_{condition}.json')['rows']
        preds={a:np.load(RUN/'predictions'/'real'/condition/(('stock' if a=='stock' else f'{a}_seed1')+'.npz')) for a in COLORS}
        gt=np.array([r['points'] for r in t]);v=np.array([r['valid'] for r in t]);kind=target_type(t).numpy()
        errors={a:np.linalg.norm(p['points']-gt,axis=-1) for a,p in preds.items()};det=preds['stock']['detected']
        subgroup=v&(kind==(1 if condition=='corner20' else 2)) if condition!='clean' else v
        candidates=np.flatnonzero(det&subgroup.any(-1));diff=np.where(v,errors['gated_amodal']-errors['local'],0).sum(-1)/v.sum(-1)
        examples=[(int(sorted(candidates,key=lambda i:digest('v12-gallery:'+m['rows'][i]['id']))[0]),'SHA 고정'),
            (int(candidates[np.argmin(diff[candidates])]),'최대 개선'),(int(candidates[np.argmax(diff[candidates])]),'최대 악화')]
        for i,why in examples:
            r=m['rows'][i];im=cv2.imread(str(ROOT/r['image']));canvas=im.copy()
            if r['mask'] is not None:
                a,u,b,w=r['mask'];canvas[u:w,a:b]=r['fill_bgr']
            if condition.startswith('crop'):canvas=canvas[:,r['cut']:] if r['crop_side']=='left' else canvas[:,:r['cut']]
            h,w=canvas.shape[:2];fig,axes=plt.subplots(1,3,figsize=(18,5.8));origin=np.array(r['origin'])
            axes[0].imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));axes[0].set_title('원본 · 사람이 찍은 정답')
            original=gt[i]+origin;axes[0].scatter(original[v[i],0],original[v[i],1],marker='+',c='#22c55e',s=70)
            for j in np.flatnonzero(v[i]):axes[0].annotate(str(j),original[j],color='#22c55e')
            for ax in axes[1:]:ax.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));ax.set_facecolor('#303946')
            axes[1].set_title('변형 입력과 최종 점 위치')
            axes[1].scatter(gt[i,v[i],0],gt[i,v[i],1],marker='+',c='#22c55e',s=80,label='수동 GT')
            for arm,p in preds.items():axes[1].scatter(p['points'][i,v[i],0],p['points'][i,v[i],1],s=40,facecolors='none',edgecolors=COLORS[arm],label=NAMES[arm])
            axes[1].legend(fontsize=8)
            choices=np.flatnonzero(subgroup[i]);j=int(choices[np.argmax(np.abs(errors['gated_amodal'][i,choices]-errors['local'][i,choices]))])
            p=preds['gated_amodal'];box=r['box'];attention=p['attention'][i,j].reshape(8,8)
            axes[2].imshow(attention,extent=[box[0],box[2],box[3],box[1]],cmap='magma',alpha=.45,interpolation='nearest')
            axes[2].scatter(*gt[i,j],s=100,marker='+',c='#22c55e',label='GT')
            for arm in ('local','learned_gate','gated_amodal'):axes[2].scatter(*preds[arm]['points'][i,j],s=55,facecolors='none',edgecolors=COLORS[arm],label=NAMES[arm])
            q=p['probabilities'][i,j];axes[2].set_title(f'점 {j}: 전체 영상 특징을 읽는 가중치')
            axes[2].set_xlabel(f'원본 보정 유지 {p["keep"][i,j,0]:.2f} · 가림 {q[1]:.2f} · 화면 밖 {q[2]:.2f}\n학습한 가중치이며 정답 확률이 아님');axes[2].legend(fontsize=8)
            pts=np.concatenate([gt[i,v[i]]]+[p['points'][i,v[i]] for p in preds.values()]);lo=np.minimum(pts.min(0)-15,[0,0]);hi=np.maximum(pts.max(0)+15,[w,h])
            for ax in axes[1:]:ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1])
            for ax in axes:ax.tick_params(labelsize=8)
            fig.suptitle(f'{LABELS[condition]} · {why} · {r["id"]}\n억제+복원 - 기존 보정: {diff[i]:+.3f}px · seed 1',fontsize=13)
            fig.tight_layout(rect=(0,.04,1,.88));name=f'{condition}_{len(gallery):02d}.png';fig.savefig(directory/name,dpi=125,bbox_inches='tight',pad_inches=.15);plt.close(fig)
            gallery.append(dict(condition=condition,id=r['id'],selection=why,path='figures/'+name,corner=j,seed=1,frame_difference_px=float(diff[i]),
                scope='GT selects illustrative examples only, not inference. Cross-attention averaged over4 heads, not a causal saliency proof.'))
    write(RUN/'GALLERY.json',dict(rows=gallery));return gallery

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
    s=read(RUN/'SUMMARY.json');a=read(RUN/'AUDIT.json');assert a['status']=='PASS';d=diagnosis(s);g=visualize(s)
    verdict=dict(status='SUPPRESSION_PASS_RECOVERY_PARTIAL_CROP20_REGRESSION',gates=s['gates'],
        inference_gt_used=False,real_training=False,natural_occlusion_established=False,production_modified=False,
        interpretation='Original-local suppression passes fixed acceptance. Recovery helps actual hidden corners and severe crop outside corners across all seeds, but mild crop regresses and one seed narrowly fails clean <=1% noninferiority. No across-condition success claim.')
    write(RUN/'CONCLUSION.json',verdict)
    text=['# 가려진 점 보정 억제와 화면 밖 점 복원 실험', '',
        '보정 억제는 사전 기준을 통과했다. 별도 복원은 코너 가림과 심한 잘림에서 개선했지만, 약한 잘림에서는 악화했다. 모든 조건에서 안정적인 구조로 완성된 것은 아니다.', '',
        '## 무엇을 바꿨나', '',
        '기존 YOLO와 V10 점 보정은 고정했다. 예측 점마다 주변 영상, 전체 ROI 영상 특징, 다른 점의 정보를 읽는 작은 attention 모듈을 추가했다. 출력은 원래 YOLO 점 + 유지 가중치×기존 점 보정 + 가림 가중치×가림 복원 + 화면 밖 가중치×화면 밖 복원이다. 이번 실험은 관측 가능성과 복원 경로를 분리해서 검사하므로 Hough 분기를 추가하지 않았다.',
        '', '일괄 축소 / 학습한 보정 억제 / 억제+별도 복원 / 복원 항상 적용의 4구조를 각3개 시드, 2,000업데이트로 학습했다. 총12모델·24,000업데이트. 합성 관측3609개를 사용했고 실제 학습은0장이다. 같은 초기 모듈과 batch 순서, 학습량을 맞췄다. 일괄 축소는 스칼라1개만 활성화되어 용량이 같지는 않다.',
        '', '## 실제로 가려진 점과 화면 밖 점', '',
        '3개 시드 평균, 직접 클릭한 점의 평균 오차(px). 작을수록 좋다. 각 행의 대상 점은 모든 모델에 동일하다.', '',
        '| 조건 / 대상 점 | 점 수 | 원래 YOLO | 기존 점 보정 | 학습한 억제 | 억제+복원 |',
        '|---|---:|---:|---:|---:|---:|']
    for c,typ in [('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        r=s['results']['real/'+c];values=[mean(r,arm,stratum=typ) for arm in ('stock','local','learned_gate','gated_amodal')]
        text.append('| '+LABELS[c]+' / '+('가려진 점' if typ=='hidden' else '화면 밖 점')+' | '+str(r['stock']['strata'][typ]['detected_points'])+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    text+=['', '보정 억제는 가려진280점에서 기존 보정보다 평균0.500px 개선했고, 세션 단위95% 구간은 -0.741~-0.250px였다. 단순 일괄 축소보다도 세 시드 모두 좋아져, 입력에 따라 보정을 줄이는 효과가 확인됐다. 다만 억제만으로 원래 YOLO를 넘지는 못했고, 주로 기존 보정의 악화를 제거했다.',
        '', '별도 복원은 가려진 점에서 억제만 할 때보다 시드별8.4~26.5%, 원래 YOLO보다7.9~26.1% 개선했다. 심한 잘림의 화면 밖571점은 원래 YOLO보다26.4~30.9% 개선했다. 반면 약한 잘림의 화면 밖558점은 원래 YOLO보다29.3~37.0% 악화했다. 약한 잘림의 세션 구간은0을 포함하므로 새 현장에서도 악화량이 같다고 단정하지 않는다.',
        '', '## 전체 프레임 평균', '',
        '실제297장 중 해당 조건에서 검출된 동일 프레임의 점 평균 오차. 검출 실패는 별도 정규화 지표에 실패1로 포함했다.', '',
        '| 조건 | 검출 | 기존 YOLO | 기존 보정 | 학습한 억제 | 억제+복원 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        r=s['results']['real/'+c];text.append('| '+LABELS[c]+' | '+str(r['stock']['detected_frames'])+'/297 | '+' | '.join(f'{mean(r,arm):.3f}' for arm in ('stock','local','learned_gate','gated_amodal'))+' |')
    text+=['','억제 구조는 원본 실사 성능을 유지하는 기준도 통과했다. 억제+복원은 원본에서 한 시드가 기존 보정보다 약1.06% 나빠져, 사전 상한1%를 근소하게 넘었다. 심한 잘림에서 크게 개선했더라도 전체 복원 성공 기준은 통과하지 못한 것으로 기록했다.',
        '', '## 약한 잘림이 나빠진 이유를 확인한 방법', '',
        '같은 모델의 저장된 출력을 이용해 복원항만 뺀 결과와 비교하고, 실제 노출 상태로 분기만 바꾼 특권 진단을 추가했다. GT는 이 사후 진단에만 쓰며 정상 추론·모델 선택에는 쓰지 않았다.',
        '', '약한 잘림에서 실제로 화면 밖인 점을 알려줘도 복원 오차가 줄지 않았다. 화면 밖 점 오차는 정상 분기15.95~16.90px에서 정답 노출 상태 분기17.38~19.64px가 됐다. 따라서 가림/잘림 분류 오류만 고치면 해결되는 문제는 아니다. 복원 분기가 예측하는 이동량 자체에도 문제가 있다.',
        '', 'FAILURE_DIAGNOSIS.json에서 같은 모델의 복원항만 분리하니 약한 잘림의558점 중317~391점이 악화했고, 그중299~369점은 정답으로 향하는 방향과 반대로 이동했다. 단순히 이동량이 너무 크다는 설명만으로는 부족하며 복원 방향 자체도 틀린다. 평균 이동량은5.57~6.63px였다. 큰 잘림에서는 평균14.16~17.94px 이동해 전체적으로 도움이 됐지만, 일부 점의 악화도 남았다. 구도/세션 차이도 섞여 있으므로 잘림 비율 하나만이 원인이라고 단정하지 않는다.',
        '', '같은 모델에서 복원항이 없을 때5px 이내였던 점이 복원 후20px를 넘는 경우도 있었다. 약한 잘림에서 시드별0/2/0점, 큰 잘림에서4/3/2점이다. 평균 개선만으로 개별 점의 안정성이 확보됐다고 보지 않는다.',
        '', '좌우 방향을 따로 계산하면 차이가 더 명확하다. 약한 잘림에서는 화면 밖 점의78.9~79.0%가 안쪽으로 움직여야 했는데, 복원 모델은56.5~72.6%를 바깥쪽으로 움직였다. 큰 잘림에서는 안쪽 이동이 필요한 점이27.7~28.0%였고 바깥 방향 예측이74.6~84.1%였다. 같은 바깥쪽 복원 경향이 큰 잘림에서는 맞고 약한 잘림에서는 틀리는 현상이 확인된다. 이는 GT를 이용한 사후 방향 분석이며, 실제 입력에서 이 차이를 안정적으로 구별하는 방법은 아직 검증하지 않았다.',
        '', '## 검증 및 해석 범위', '',
        f'- {a["prediction_frame_rows"]:,}개 예측 프레임 행을 독립 검산했다. 혼합 좌표 최대 오차 {a["maxima"]["mixture_px"]:.7f}px, 오차 집계는 부동소수점 범위에서 일치했다.',
        '- 합성 학습 시 인위적으로 가린/잘린 위치로 분류 손실을 만들었다. 모델 입력에는 GT, 가림 마스크, 조건 이름, 주석 가시성을 전달하지 않는다. 실사에는 직접 클릭한1251점만 사용하고 PnP 자동 투영점은 제외했다.',
        '- 실제297장/9세션과 합성505장은 재사용 평가다. 실제 데이터는 학습하거나 모델 선택에 쓰지 않았다. 자연 가림 라벨은unknown이므로 인공 가림의 성공이 자연 가림 일반화를 입증하지는 않는다.',
        '- 기존 검출 결과와 reflect100 padding을 모든 모델이 공유한다. 검출 실패를 고친 실험은 아니다. 밖으로 나간 점의 원본 GT는 유지하며 정규화 분모는 원본 대각선이다.',
        '- 분기 확률은 인공 노출 상태에 대한 학습 가중치이며 실제 가시성/정확도의 보정된 확률이 아니다. Attention도 인과적 설명을 증명하는 값은 아니다.',
        '- 이전 실험의 손실 가중치나 기존 파일을 수정하지 않았고 실제 배포 모델도 교체하지 않았다. 이2D 점 지표는 이전41.49px 또는 최종6D 평가와 직접 비교하지 않는다.',
        '', '## 판단', '',
        '관측 가능성에 따라 기존 보정을 억제하는 방향은 유효하다. 별도 복원도 어려운 조건에서 가능성이 아니라 실제 개선을 보였지만, 약한 잘림의 회귀가 있어 전체 구조를 그대로 채택할 단계는 아니다. 다음에는 노출 상태와 별도로 복원 자체의 기대 오차·이득을 학습해 이미 맞는 점을 과도하게 이동시키는 문제를 다뤄야 한다. 이는 다음 설계 제안이며 이번에 검증한 성과와 구분한다.',
        '', '상세 수치: SUMMARY.json. 고정 기준: PROTOCOL.json. 독립 검산: AUDIT.json. 분기 사후 진단: ORACLE_ROUTING_DIAGNOSTIC.json.']
    (RUN/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
    result_json=json.dumps(s['results'],ensure_ascii=False);labels=json.dumps(LABELS,ensure_ascii=False);names=json.dumps(NAMES,ensure_ascii=False);gallery_json=json.dumps(g,ensure_ascii=False)
    body=[]
    for line in text:
        if line.startswith('# '):continue
        if line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line and not line.startswith('|'):body.append('<p>'+html.escape(line)+'</p>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>보정 억제와 화면 밖 복원 — 실험 결과</title>
<style>body{font:16px/1.7 system-ui,sans-serif;background:#f3f6fa;color:#172334;max-width:1250px;margin:auto;padding:28px}h1{font-size:29px}h2{margin-top:30px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}select{padding:8px;margin:6px;font-size:15px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:right;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}a{color:#0369a1}.note{color:#596579}</style>
<h1>보정 억제는 개선 · 별도 복원은 조건에 따라 성공과 악화</h1><p>고정 YOLO + 기존 보정의 학습한 억제 + 가림/화면 밖 복원. 12모델·24,000업데이트·실제 학습0장.</p>
<section><h2>조건별 비교</h2><select id="split"><option value="real">실제297장</option><option value="synth_val">합성505장</option></select><select id="seed"><option value="mean">3개 시드 평균</option><option value="1">seed1</option><option value="2">seed2</option><option value="3">seed3</option></select><select id="metric"><option value="frame_mean_px">프레임별 평균 점 오차px</option><option value="hidden">실제로 가려진 점 오차px</option><option value="outside">화면 밖 점 오차px</option><option value="full_normalized_mean">검출 실패 포함 정규화 오차</option></select><div style="overflow-x:auto"><table id="table"></table></div><p class="note">작을수록 좋음. 실제는 사람이 찍은 점만 평가. 숨겨진 점/화면 밖 점은 점 단위 평균으로 프레임 평균과 다릅니다. 검출 실패 포함 지표는 실패1, 원본 대각선으로 정규화합니다.</p></section>
<section><h2>가림·잘림 예시와 영상 특징을 읽는 가중치</h2><select id="gallerySelect"></select><p id="caption"></p><a id="galleryLink" target="_blank"><img id="galleryImage"></a><p class="note">회색은 잘려 모델이 보지 못한 영역입니다. 최대 개선/악화 예시는 설명을 위해 GT로 골랐습니다. 정상 추론에는 GT를 사용하지 않았습니다. 점 이동 화살표를 과장하지 않은 실제 좌표입니다.</p></section><section>'''+''.join(body)+'''<p><a href="REPORT_KO.md">보고서</a> · <a href="SUMMARY.json">수치와 사전 기준 판정</a> · <a href="AUDIT.json">독립 검산</a> · <a href="FAILURE_DIAGNOSIS.json">복원 실패 분석</a></p></section><script>
const data='''+result_json+''',labels='''+labels+''',names='''+names+''',gallery='''+gallery_json+''';
function render(){const split=document.getElementById('split').value,seed=document.getElementById('seed').value,metric=document.getElementById('metric').value;let out='<tr><th>조건</th><th>검출</th>'+Object.values(names).map(n=>'<th>'+n+'</th>').join('')+'</tr>';for(const c of Object.keys(labels)){const rows=data[split+'/'+c];out+='<tr><td>'+labels[c]+'</td><td>'+rows.stock.detected_frames+'/'+rows.stock.frames+'</td>';for(const a of Object.keys(names)){const r=a==='stock'?[rows.stock]:(seed==='mean'?[1,2,3]:[+seed]).map(s=>rows[a+'_seed'+s]);const v=r.map(x=>['hidden','outside'].includes(metric)?x.strata[metric].mean_px:x[metric]);out+='<td>'+(v.some(x=>x===null)?'—':(v.reduce((x,y)=>x+y,0)/v.length).toFixed(metric==='full_normalized_mean'?6:3))+'</td>';}out+='</tr>';}document.getElementById('table').innerHTML=out;}
for(const id of ['split','seed','metric'])document.getElementById(id).onchange=render;render();const sel=document.getElementById('gallerySelect');gallery.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=labels[r.condition]+' · '+r.selection;sel.appendChild(o);});function show(){const r=gallery[+sel.value];document.getElementById('galleryImage').src=r.path;document.getElementById('galleryLink').href=r.path;document.getElementById('caption').textContent=r.id+' · seed1 · 기존 보정 대비 '+r.frame_difference_px.toFixed(3)+'px (음수 개선)';}sel.onchange=show;show();</script></html>'''
    (RUN/'index.html').write_text(page)
    write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),summary_sha256=sha(RUN/'SUMMARY.json'),source_sha256=sha(Path(__file__)),gallery_images=len(g)))
    print('REPORT COMPLETE',RUN/'index.html',flush=True)

if __name__=='__main__':main()
