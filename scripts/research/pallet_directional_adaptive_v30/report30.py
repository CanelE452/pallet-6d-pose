"""Report state-free adaptive direction experiments, with all seeds and actual paths."""
from context30 import *
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v30-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
import markdown,cv2
from data30 import population
from features25 import actual_image
sys.path.insert(0,str(KIT.parent/'pallet_direction_utility_v27_cases'))
from show_cases import clipline
from model30 import ANCHORS
from model29 import INCIDENT

NAMES={'stock':'기존 YOLO','base':'V12 기준점','strong':'고정한 강한 점 복원기(V25)','line_vector':'V18 기존 융합','line_role_utility':'V27 기존 융합','point_gain':'점 방향만·이득 학습','whole_line':'선 코너 전체 방향·이득 학습','normal_no_evidence':'선 법선·선 구간 영상 없음','normal_gain':'선 법선·방향/이득 적응형','normal_no_lines':'동일 모델·선 경로 끄기','normal_evidence_off':'동일 모델·선 구간 특징0','normal_evidence_donor':'동일 모델·다른 영상 특징','oracle_anchors_GT':'GT 부모 선택(진단)','oracle_point_amount_GT':'GT 점 방향/이동량(진단)','oracle_normal_choice_GT':'GT 후보 선택·예측 이동량(진단)','oracle_normal_amount_GT':'GT 후보/이동량(진단)'}
ORDER=('stock','base','strong','line_vector','line_role_utility')+ARMS+('oracle_anchors_GT','oracle_point_amount_GT','oracle_normal_choice_GT','oracle_normal_amount_GT')
COLS=(('clean',None),('corner20','hidden'),('crop10','outside'),('crop20','outside'),('crop30','outside'),('crop40','outside'))
LABELS=['원본','코너 가림','잘림10','잘림20','잘림30','잘림40']

def value(s,a,c,t=None,split='real'):
    rr=s['results'][split+'/'+c]
    return float(np.mean([rr[f'{a}_seed{i}']['frame_mean_px'] if t is None else rr[f'{a}_seed{i}']['strata'][t]['mean_px'] for i in SEEDS]))

def diagnostic(s,c,t,key,arm='normal_gain'):
    return float(np.mean([r[arm][t][key] for r in s['diagnostics']['real/'+c].values()]))

def table(s,order=ORDER,split='real'):
    cols=[x for x in COLS if split+'/'+x[0] in s['results']];labels=[LABELS[COLS.index(x)] for x in cols]
    return ['| 방법 | '+' | '.join(labels)+' |','|---|'+'---:|'*len(cols)]+['| '+NAMES[a]+' | '+' | '.join(f'{value(s,a,c,t,split):.3f}' for c,t in cols)+' |' for a in order]

def plots(s):
    fd=RUN/'figures';fd.mkdir(exist_ok=True);order=('base','strong','point_gain','whole_line','normal_no_evidence','normal_gain','oracle_point_amount_GT','oracle_normal_amount_GT');matrix=np.array([[value(s,a,c,t) for c,t in COLS] for a in order]);delta=matrix-matrix[0]
    fig,ax=plt.subplots(figsize=(12,6.2),layout='constrained');im=ax.imshow(delta,cmap='RdYlGn_r',vmin=-10,vmax=10,aspect='auto')
    for i in range(len(order)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}\n({delta[i,j]:+.2f})',ha='center',va='center',fontsize=10)
    ax.set_xticks(range(6),['Clean','Corner20 hidden','Crop10 outside','Crop20 outside','Crop30 outside','Crop40 outside']);ax.set_yticks(range(len(order)),order);ax.set_title('V30: per-point signed direction and gain, without state routing\nReal297 reused DEV / 3-seed mean error px (change vs V12 base)');fig.colorbar(im,ax=ax,label='Error change vs base (px)');fig.savefig(fd/'performance.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    for ax,(c,t) in zip(axes,[('corner20','hidden'),('crop20','outside'),('crop40','outside')]):
        keys=['keep_base','keep_strong','partial_point','line_selected'];counts=[diagnostic(s,c,t,k) for k in keys];ax.bar(range(4),counts,color=['#497aa7','#667b9f','#b28643','#189d83']);ax.set_xticks(range(4),['Keep base','Keep strong','Partial point','Line normal'],rotation=18);ax.set_ylabel('Mean selected points per seed');ax.set_title(c+f' / line improve>1px {diagnostic(s,c,t,"line_improve_anchor_over1px"):.1f}, harm>1px {diagnostic(s,c,t,"line_harm_anchor_over1px"):.1f}')
    fig.suptitle('Actual learned choices; line gains measured relative to the chosen point anchor');fig.savefig(fd/'choices.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,4.6),layout='constrained');ax.axis('off');ax.set(xlim=(0,13),ylim=(0,4))
    boxes=[(.1,2.1,2.8,1.2,'Frozen point anchors\nV12 original / strong V25\nBoth exact keep paths'),(.1,.4,2.8,1.2,'Frozen V29 DHT sources\nguided + independent\nActual line coefficients'),(3.6,.9,3.1,2.1,'Partial direction paths\npoint delta / x / y\nline normal from either anchor\nObserved image along each line'),(7.4,.9,2.3,2.1,'New shared head\nsigned amount [-1,1]\npredicted error reduction\nNo visibility/state class'),(10.4,.9,2.4,2.1,'Choose highest gain\nKeep base gain =0\nNormal-only movement\npreserves anchor tangent')]
    for x,y,w,h,text0 in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc='#edf3fa',ec='#497594'));ax.text(x+w/2,y+h/2,text0,ha='center',va='center',fontsize=9)
    for x,y,u,v in [(2.9,2.7,3.6,2.3),(2.9,1,3.6,1.5),(6.7,1.9,7.4,1.9),(9.7,1.9,10.4,1.9)]:ax.annotate('',(u,v),(x,y),arrowprops=dict(arrowstyle='->'))
    ax.set_title('V30 trained architecture / shared41,630-parameter head / all upstream parents fixed');fig.savefig(fd/'architecture.png',dpi=170);plt.close(fig)

def cases():
    gallery=[]
    for key,c,mode in [('case3','crop20','fixed previous case3'),('gain','crop20','largest improvement vs matched point-only control'),('harm','crop20','largest worsening vs matched point-only control'),('hidden','corner20','largest improvement vs matched point-only control')]:
        m,ts=population('real',c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);p=np.load(RUN/'predictions/real'/c/'normal_gain_seed1.npz');q=np.load(RUN/'predictions/real'/c/'point_gain_seed1.npz');error=np.linalg.norm(p['points']-gt,axis=-1);before=np.linalg.norm(q['points']-gt,axis=-1);mask=valid&p['detected'][:,None]&np.array([t['hidden' if c=='corner20' else 'outside'] for t in ts])
        if key=='case3':i=p['ids'].tolist().index('forklift_v4_20260904_144733_manual_gt/000546');j=0
        else:
            change=error-before if key=='harm' else before-error;ij=np.where(mask,change,-np.inf).argmax();i,j=np.unravel_index(ij,mask.shape)
        picked=int(p['chosen'][i,j]);anchor=p['anchors'][i,j,picked];end=p['points'][i,j];base=p['base'][i,j];strong=p['strong'][i,j];raw=actual_image(m['rows'][i]);h,w=raw.shape[:2];co=np.array([gt[i,j],base,strong,anchor,end]);rect=(min(0,co[:,0].min())-15,min(0,co[:,1].min())-15,max(w,co[:,0].max())+15,max(h,co[:,1].max())+15)
        line=None;source=None
        if picked>=8:
            k=(picked-8)%18;source=('guided','independent')[k//9];role=INCIDENT[j][(k%9)//3];midx=k%3
            b=np.load(V29.RUN/'predictions/real'/c/'bundle_seed1.npz');line=b[source+'_lines'][i,role,midx]
        fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
        for ax,zoom in zip(axes,(False,True)):
            view=(co[:,0].min()-25,co[:,1].min()-25,co[:,0].max()+25,co[:,1].max()+25) if zoom else rect
            ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));ax.set_facecolor('#dfe3e8');ax.add_patch(Rectangle((0,0),w,h,fill=False,ls='--',ec='black',lw=.7))
            if line is not None:
                for region,ls in [(view,'--'),((0,0,w,h),'-')]:
                    seg=clipline(line,region)
                    if seg is not None:ax.plot(*np.array(seg).T,c='#00a1a1',ls=ls,lw=1.8,label='Selected DHT line' if ls=='--' else None)
            for label,pos,color,marker in [('GT',gt[i,j],'#00bd46','x'),('Base',base,'#147fff','o'),('Strong point',strong,'#8249b8','s'),('Adaptive',end,'#ed8a00','D')]:ax.scatter(*pos,c=color,marker=marker,s=65,label=label,zorder=5,edgecolors='white' if marker!='x' else None)
            ax.annotate('',end,anchor,arrowprops=dict(arrowstyle='->',color='#d84038',lw=2));ax.set_xlim(view[0],view[2]);ax.set_ylim(view[3],view[1]);ax.set_aspect('equal');ax.legend(fontsize=8);ax.set_title('Detail / arrow starts at selected anchor' if zoom else 'Actual model input / grey is outside observed image',fontsize=10)
        fig.suptitle(f'{key}: frame {m["rows"][i]["id"].split("/")[-1]} / point{j} / seed1\npath{picked}, alpha={p["alpha"][i,j,picked]:+.3f}, actual error={error[i,j]:.2f}px; point control={before[i,j]:.2f}px',fontsize=11);fig.savefig(RUN/'figures'/f'{key}.png',dpi=170);plt.close(fig)
        per=[]
        for seed in SEEDS:
            row=dict(seed=seed)
            for arm in ('base','strong','point_gain','whole_line','normal_gain'):
                pp=np.load(RUN/'predictions/real'/c/f'{arm}_seed{seed}.npz');assert pp['ids'][i]==p['ids'][i];row[arm]=float(np.linalg.norm(pp['points'][i,j]-gt[i,j]))
            per.append(row)
        gallery.append(dict(key=key,condition=c,id=m['rows'][i]['id'],point=int(j),selection=mode,seed=1,chosen=picked,alpha=float(p['alpha'][i,j,picked]),line_source=source,line=None if line is None else line.tolist(),per_seed=per,row=m['rows'][i]))
    return gallery

def main():
    assert read(RUN/'AUDIT.json')['status']=='PASS';s=read(RUN/'SUMMARY.json');gd=read(RUN/'GAIN_DIAGNOSIS.json');plots(s);gallery=cases();status='통과' if s['accepted'] else '미통과';n=read(RUN/'PLAN_seed1.json')['originals']
    body=['# V30: 상태 분류 없이 점별 방향·이득을 학습하는 적응형 점/선 융합','',f'**4대조·12모델·24,000업데이트·3시드 학습/검증 완료. 사전 등록한 안정성+선 추가 이득 동시 기준: {status}.**','',
    '사용자의 핵심은 clean/occlusion/truncation에 정해진 전문가를 배정하는 것이 아니라, 각 점에서 도움이 되는 정보와 방향만 적응적으로 사용하는 것이다. 이번 모델은 조건 이름·가시성 상태 분류를 사용하지 않으며, 조건별 표는 평가를 나눠 보기 위한 것이다.','',
    '## 실제 구조 변경','', '![학습 구조](figures/architecture.png)','',
    'V12 기준점과 기존의 강한 V25 점 복원기를 고정하고 두 출력을 그대로 유지할 수 있게 했다. 새 학습은41,630파라미터의 후보별 방향 이동량·오차 감소량 헤드다. V29의 점 가이드/점 독립 DHT도 고정했다. **부모 출력을 유지 가능한 후보로 보존한 것이지, 학습된 선택 결과가 모든 기존 이득을 자동 보장한다는 뜻은 아니다.**','',
    '선 방향 후보는 선택한 점 부모에서 선에 수직인 방향으로만 움직인다. 접선 방향 좌표는 그대로다. 이동 크기는 box 대각선15% 이내이며 부호를 포함한 적용량[-1,1]을 학습하므로 필요하면 선에서 멀어지는 방향도 허용한다. 따라서 선의 위치를 그대로 정답으로 믿고 붙이는 구조가 아니다. 다른 두 점을 연결해 선을 만드는 것도 아니다.','',
    '| 대조 | 점 유지·부분 보정 | 추가 선 경로 |','|---|---|---|','| point_gain | 원래/강한 점 유지와 full/x/y 부분 보정 | 없음 |','| whole_line | 동일 | 선 디코더의 전체 코너 방향으로 부분 이동 |','| normal_no_evidence | 동일 | 선 법선 방향, 선 구간 영상 특징 없음 |','| normal_gain | 동일 | 선 법선 방향과 실제 영상/미분 구간 특징 |','',
    '각 후보의 부호·이동량은 합성 GT에 대한 투영으로 감독하고, 현재 예측 이동량을 적용했을 때 기준점보다 오차가 얼마나 줄어드는지를 별도로 감독한다. 최종 추론은 예상 이득이 가장 큰 경로를 선택하며, 원래 점 유지의 이득은0으로 고정한다. GT와 가림/잘림 상태는 추론에 들어가지 않는다. 이전 V27의 효용 학습과 완전히 무관한 새 원리는 아니며, 이번에 달라진 것은 강한 점 부모 보존·V29 선 소스·부호 있는 부분 방향·선 전체 관측 구간·상태 없는 직접 후보 감독이다.','',
    f'학습은 V29에서 DHT 미세학습에 사용하지 않은 합성{n}원본과 그 변형만 사용했다. **V25 강한 점 부모는 이 원본들을 이미 학습했으므로 그 잔차는 in-sample이라는 한계가 남는다.** 모든 부모까지 학습 원본이 완전히 분리된 실험으로 주장하지 않는다. 물리 가시 경계/끝점 GT나 새 실사 GT는 추가하지 않았다.','',
    '## 실사 전체 비교','',
    '실사297장·9세션 재사용DEV·3시드 평균px. 원본은 검출 영상 평균, 가림은 가린 유효점, 잘림은 화면 밖 유효점이다. 정본140/논문계보나 새 봉인평가와 혼합하지 않는다. 검출 실패 포함 정규화 지표, 중앙 가림20/40을 포함한8조건 전체는 SUMMARY.json에 있다.','',*table(s),'','![실제 비교](figures/performance.png)','',
    f'주 분석 잘림20: 점 방향 대조 {value(s,"point_gain","crop20","outside"):.3f}px → 선 법선 적응형 {value(s,"normal_gain","crop20","outside"):.3f}px. 전체 코너 방향 대조는 {value(s,"whole_line","crop20","outside"):.3f}px다. 기존 기준점은 {value(s,"base","crop20","outside"):.3f}px, 강한 점 부모는 {value(s,"strong","crop20","outside"):.3f}px다. 일부 대조보다 좋아지는 것과 기존 점을 안정적으로 보존하는 것은 별도로 판정했다.','',
    '## 기준 판정','',
    '각 시드 clean/crop10/20/30의 V12 대비 악화1% 이내, corner20/crop40의 V18 복원 이득90% 보존을 요구했다. 주 대조 normal_gain은 point_gain 대비 crop20 각 시드1% 이상 개선 및 세션 bootstrap95% 상한<0까지 함께 필요하다. 평가 뒤 손실·임계값·최종 체크포인트를 바꾸지 않았다.','',
    '| 모델 | 전체 안정성 | 실패 항목 |','|---|---|---|']
    for arm in ARMS:
        gate=s['gates'][arm];bad=['seed'+seed+': '+','.join(k for k,v in vals.items() if not v) for seed,vals in gate['per_seed'].items() if not all(vals.values())];body.append('| '+NAMES[arm]+' | '+('통과' if gate['pass_gate'] else '미통과')+' | '+'; '.join(bad)+' |')
    body+=['', 'normal_gain의 point_gain 대비 잘림20 개선율: '+', '.join(f'{v:.2f}%' for v in s['line_evidence']['per_seed_gain_percent'])+'; 세션95% 차이 구간 '+str([round(x,4) for x in s['line_evidence']['ci95']])+'px. 다중 비교 보정 없는 재사용DEV 분석이다.','',
    '## 실제로 무엇을 선택했는가','', '![선택 분해](figures/choices.png)','',
    '| 대상 | 선 경로 선택/전체 점 | 선이 선택한 점 부모보다1px 이상 개선 | 1px 이상 악화 |','|---|---:|---:|---:|']
    for c,t in [('corner20','hidden'),('crop20','outside'),('crop40','outside')]:body.append(f'| {c} | {diagnostic(s,c,t,"line_selected"):.1f}/{diagnostic(s,c,t,"points"):.0f} | {diagnostic(s,c,t,"line_improve_anchor_over1px"):.1f} | {diagnostic(s,c,t,"line_harm_anchor_over1px"):.1f} |')
    body+=['','분기 선택량만으로 선의 이득을 주장하지 않는다. 강한 점 부모의 성능이 선 경로를 통해 그대로 전달될 수도 있어, 선 이동 직전의 선택된 점 부모 대비 개선/악화도 계산했다.','',
    '## 선의 추가 정보인가, 후보 선택·이동량 문제인가','',
    '| 추론 개입 | 원본 | 코너 가림 | 잘림20 | 잘림40 |','|---|---:|---:|---:|---:|']
    for arm in ('normal_gain','normal_no_lines','normal_evidence_off','normal_evidence_donor'):body.append('| '+NAMES[arm]+' | '+' | '.join(f'{value(s,arm,c,t):.3f}' for c,t in [('clean',None),('corner20','hidden'),('crop20','outside'),('crop40','outside')])+' |')
    body+=['',f'잘림20에서 GT로 현재 예측 이동량의 후보만 고르면 {value(s,"oracle_normal_choice_GT","crop20","outside"):.3f}px, 이동량까지 GT로 정하면 {value(s,"oracle_normal_amount_GT","crop20","outside"):.3f}px다. 점 방향만 GT로 최적 선택/이동하면 {value(s,"oracle_point_amount_GT","crop20","outside"):.3f}px다. 이 값은 달성한 추론 성능이 아니다. 많은 방향 후보와 GT 이동량이 만드는 여지를 실제 Hough 의미 정확도의 증명으로도 해석하지 않는다.','',
    '선 영상0/donor는 이미 학습된 모델의 추론 개입이므로 분포 변화 효과가 섞인다. normal_gain 대 normal_no_evidence의 동일 학습 대조와 함께 본다. 실제 가시 경계 감독이나 동일 길이의 임의 방향으로 학습한 대조는 이번에 포함하지 않아, 모든 차이를 Hough 고유 의미 정보의 효과로 확정하지 않는다.','',
    '## 학습집합과 별도 원본의 이득 판별','',
    '| 모집단/대상 | 후보 이득 AUROC (3시드 평균) | 예측 이동량 절대오차 | 실제 점 오차px | 현재 이동량의 GT 최선px |','|---|---:|---:|---:|---:|']
    for pop,c,t in [('train','all','all'),('prospective','crop20','outside'),('synth_val','crop20','outside'),('real','crop20','outside')]:
        rr=[r['stats'][t] for r in gd['rows'] if r['arm']=='normal_gain' and r['population']==pop and r['condition']==c];avg=lambda k:float(np.mean([x[k] for x in rr]));body.append(f'| {pop}/{t} | {avg("candidate_help_auc"):.3f} | {avg("mean_amount_abs_error"):.3f} | {avg("mean_error"):.3f} | {avg("oracle_current_amount_error"):.3f} |')
    body+=['','AUROC는 같은 점의 상관된 여러 후보에 대한 진단이며 신뢰도 확률의 보정이나 독립 표본 검정이 아니다. 학습집합부터 선택·이동량이 약한지와 별도 합성/실사로 옮기며 달라지는지를 구분한다.','',
    '## 고정3번 사례와 개선/실패 이미지','',
    '3번 사례는 이전과 같은 frame000546·pointID0을 고정했다. 나머지는 seed1에서 대응 점 대조 대비 변화가 가장 큰 사례를 골라 설명하며 대표 평균으로 주장하지 않는다. 화살표는 선택한 점 부모에서 최종 점으로 이동한 방향이고, 청록선은 실제 선택한 DHT 선이다. 선 경로를 선택하지 않았다면 선을 억지로 표시하지 않는다. 모든 사례의3시드 수치를 함께 남겼다.','']
    for case in gallery:
        body+=['### '+case['key']+' / '+case['id']+' / point'+str(case['point']),'','![점과 선택 선·부분 보정](figures/'+case['key']+'.png)','', '| 시드 | 기준점 | 강한 점 | 점 방향 대조 | 선 전체 방향 | 선 법선 적응형 |','|---|---:|---:|---:|---:|---:|']
        for r in case['per_seed']:body.append('| '+str(r['seed'])+' | '+' | '.join(f'{r[a]:.3f}' for a in ('base','strong','point_gain','whole_line','normal_gain'))+' |')
        body+=['']
    body+=['## 별도 합성 원본 비교','',*table(s,('base','strong')+ARMS,'prospective'),'','## 완료 범위와 남은 범위','',
    f'이번 **상태 없는 방향별 적응형 모델의 구현·12모델 학습·전체 조건 비교·기하/수치/부모 재현 검사·실제 이미지 시각화는 완료**했다. 안정성 판정은 {status}다. 부모를 고정한 것과 좋은 부모를 항상 고르는 것은 다르므로 보존 가능한 경로와 실제 보존 결과를 구분한다.','',
    '이번에는 DHT 자체를 다시 학습하거나 Hough 표현 범위를 늘리지 않았다. 물리적 가시 경계/끝점 감독, 기존 점 부모 학습 원본과 완전히 분리된 새 잔차, 자연 가림·새 실사 봉인평가도 미실행이다. 수동 GT는 수정하지 않았고 클릭 오차를 새로 재측정하지 않았다. 결과를 근거로 다음 원인을 좁히되, 이 한 구조의 실패를 adaptive 융합 자체의 불가능성으로 해석하지 않는다.','',
    '[사전 계약](PROTOCOL.json) · [학습 완료](TRAINING_COMPLETE.json) · [전체 수치](SUMMARY.json) · [이득 진단](GAIN_DIAGNOSIS.json) · [감사](AUDIT.json) · [사례/출처](CASES.json)']
    text='\n'.join(body)+'\n';(RUN/'REPORT_KO.md').write_text(text);note=text.replace('(figures/','(../../../../data/pallet/results/pallet_directional_adaptive_v30/figures/')
    for name in ('PROTOCOL.json','TRAINING_COMPLETE.json','SUMMARY.json','GAIN_DIAGNOSIS.json','AUDIT.json','CASES.json'):note=note.replace('('+name+')','(../../../../data/pallet/results/pallet_directional_adaptive_v30/'+name+')')
    NOTE.write_text(note);html=markdown.markdown(text,extensions=['tables','fenced_code']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>V30 점별 방향 적응형 실험 결과</title><style>body{font-family:Arial,sans-serif;max-width:1250px;margin:32px auto;padding:0 20px;color:#203040;line-height:1.7}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cad4df;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf3fa}h2{margin-top:42px;border-bottom:2px solid #d8e5ef}a{color:#146eae}</style><body>'+html+'</body></html>')
    write(RUN/'CASES.json',dict(cases=gallery));write(RUN/'FIGURE_DATA.json',dict(table=[dict(arm=a,values=[value(s,a,c,t) for c,t in COLS]) for a in ORDER]));write(RUN/'REPORT_COMPLETE.json',dict(complete=True,html_sha256=sha(RUN/'index.html'),note_sha256=sha(NOTE)));print('REPORT30 COMPLETE',flush=True)

if __name__=='__main__':main()
