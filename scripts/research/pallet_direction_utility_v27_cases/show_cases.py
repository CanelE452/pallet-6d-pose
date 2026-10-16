"""Measured V27 point/Hough overlays. Read-only parent results, no new training."""
from pathlib import Path
import sys,os
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pallet_direction_utility_v27'))
from context27 import ROOT,PREP,read,write,sha,repcache
from eval27 import population
from features25 import actual_image
import numpy as np,torch,cv2,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v27-cases-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import markdown
KIT=Path(__file__).resolve().parent
PARENT=ROOT/'data/pallet/results/pallet_direction_utility_v27'
RUN=ROOT/'data/pallet/results/pallet_direction_utility_v27_cases'
NOTE=ROOT/'_docs/notes/point_line_fusion/04_failure_analysis/pallet_direction_utility_v27_cases.md'
COLORS={'GT':'#00b33c','Base':'#148dff','V27':'#f13635','Oracle':'#d400d9'}
LINECOLORS=['#ff9c00','#8152c4','#00adbe']
CANDIDATE=['keep','point vector','point x','point y','L1 normal','L2 normal','L3 normal','L1 + L2 intersection','L1 + L3 intersection','L2 + L3 intersection']

def projection(p,gt):
    b=p['base'].astype(float);d=p['candidates'].astype(float);res=gt-b
    with np.errstate(invalid='ignore'):
        amount=np.clip((res[:,:,None]*d).sum(-1)/np.maximum((d*d).sum(-1),1e-12),0,1);cand=b[:,:,None]+amount[...,None]*d;err=np.linalg.norm(cand-gt[:,:,None],axis=-1)
    err=np.where(p['candidate_mask']==1,err,np.inf);choice=err.argmin(-1);oracle=np.take_along_axis(cand,choice[...,None,None],axis=2)[:,:,0];oe=err.min(-1)
    return dict(base=b,amount=amount,candidates=cand,errors=err,choice=choice,oracle=oracle,oracle_error=oe,base_error=np.linalg.norm(b-gt,axis=-1),actual_error=np.linalg.norm(p['points']-gt,axis=-1))

def clipline(line,rect):
    a,b,c=line;x0,y0,x1,y1=rect;pts=[]
    if abs(b)>1e-10:
        for x in (x0,x1):
            y=-(a*x+c)/b
            if y0-1e-6<=y<=y1+1e-6:pts.append((x,y))
    if abs(a)>1e-10:
        for y in (y0,y1):
            x=-(b*y+c)/a
            if x0-1e-6<=x<=x1+1e-6:pts.append((x,y))
    if len(pts)<2:return None
    return max(((p,q) for p in pts for q in pts),key=lambda z:np.linalg.norm(np.array(z[0])-z[1]))

def draw_lines(ax,lines,rect,observed,chosen):
    used=[] if chosen<4 else [chosen-4] if chosen<7 else [(0,1),(0,2),(1,2)][chosen-7]
    for k,line in enumerate(lines):
        for region,ls,alpha in [(rect,'--',.65),(observed,'-',.95)]:
            seg=clipline(line,region)
            if seg is not None:ax.plot(*np.array(seg).T,color=LINECOLORS[k],ls=ls,lw=2.1 if k in used else 1.15,alpha=alpha,label=f'L{k+1}'+(' (oracle)' if k in used else '') if ls=='--' else None)

def point_panel(ax,im,gt,base,actual,oracle,lines,choice,point,rect,zoom=False):
    h,w=im.shape[:2];ax.set_facecolor('#e0e3e7');ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),extent=(0,w,h,0));draw_lines(ax,lines,rect,(0,0,w,h),choice);ax.add_patch(Rectangle((0,0),w,h,fill=False,ec='#313b45',lw=1.3,ls='--'))
    for label,pos,marker in [('GT',gt,'x'),('Base',base,'o'),('V27',actual,'s'),('Oracle',oracle,'*')]:
        if label in ('Base','V27'):ax.scatter(*pos,facecolors='none',edgecolors=COLORS[label],s=165 if label=='Base' else 100,marker=marker,linewidths=1.8,label=label,zorder=6)
        else:ax.scatter(*pos,c=COLORS[label],s=90 if marker=='x' else 65,marker=marker,linewidths=2 if marker=='x' else .7,edgecolors='white' if marker!='x' else None,label=label,zorder=7)
    ax.plot([gt[0],oracle[0]],[gt[1],oracle[1]],c=COLORS['Oracle'],lw=1.8,ls=':')
    ax.set_xlim(rect[0],rect[2]);ax.set_ylim(rect[3],rect[1]);ax.set_aspect('equal');ax.set_xlabel('Original image pixels (cropped origin)')
    if zoom:
        x,y=rect[0]+.08*(rect[2]-rect[0]),rect[3]-.1*(rect[3]-rect[1]);ax.plot([x,x+10],[y,y],c='black',lw=3);ax.text(x+5,y-2,'10 px',ha='center',fontsize=9,bbox=dict(fc='white',alpha=.85,ec='none'))
        mid=(gt+oracle)/2;ax.annotate(f'{np.linalg.norm(gt-oracle):.2f} px',mid,xytext=(12,15),textcoords='offset points',color='#970097',fontsize=10,bbox=dict(fc='white',alpha=.85,ec='none'),arrowprops=dict(arrowstyle='-',color='#970097'))
    ax.set_title(f'Point ID {point}: '+('detail / 10 px scale' if zoom else 'model input + top3 DHT lines'),fontsize=11)

def main():
    torch.set_num_threads(2);RUN.mkdir(parents=True,exist_ok=True);(RUN/'figures').mkdir(exist_ok=True);m,ts=population('real','crop20');gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);outside=np.array([t['outside'] for t in ts]);summary=read(PARENT/'SUMMARY.json');means=[];sources={};saved=[]
    for seed in (1,2,3):
        path=PARENT/'predictions/real/crop20'/f'line_role_utility_seed{seed}.npz';p=np.load(path);z=projection(p,gt);mask=valid&outside&p['detected'][:,None];val=float(z['oracle_error'][mask].mean());expected=summary['diagnostics']['real/crop20'][f'line_role_utility_seed{seed}']['outside']['oracle_candidates_px'];assert abs(val-expected)<1e-10;means.append(val);sources[str(path.relative_to(ROOT))]=sha(path)
        if seed==1:saved=[p,z,mask]
    p,z,mask=saved;mean=float(np.mean(means));r=repcache('real','crop20',1);be=z['base_error'];ae=z['actual_error'];oe=z['oracle_error'];choice=z['choice'];eligible=np.argwhere(mask)
    good=np.argwhere(mask&(choice>=4)&(oe<3)&(ae<be));good=max(good,key=lambda x:(be-ae)[tuple(x)])
    near=min(eligible,key=lambda x:abs(oe[tuple(x)]-mean));miss=max(np.argwhere(mask&(oe<5)),key=lambda x:(ae-oe)[tuple(x)]);bad=max(eligible,key=lambda x:oe[tuple(x)])
    spec=[('good',good,'실제 모델도 개선한 사례','Max actual improvement among outside points with line-selected oracle<3px'),('near_mean',near,'8.78px과 비슷한 오차','Outside point whose oracle error is nearest to the3-seed overall mean'),('missed',miss,'좋은 선 후보를 모델이 놓친 사례','Largest actual minus oracle gap among outside points with oracle<5px'),('oracle_bad',bad,'최적 후보로도 오차가 크게 남는 사례','Largest oracle residual among seed1 outside points')]
    cases=[];mixture_errors=[]
    for key,(i,j),title,selection in spec:
        i=int(i);j=int(j);row=m['rows'][i];obs=row['observation_index'];assert np.allclose(r['base'][obs].numpy(),p['base'][i],atol=1e-6);idx=np.argsort(-r['line_weights'][obs,j,:-1].numpy(),kind='stable')[:3];lines=r['lines'][obs,idx].numpy().astype(float);lines/=np.maximum(np.linalg.norm(lines[:,:2],axis=-1,keepdims=True),1e-12)
        im=actual_image(row);raw=PREP.raw_image(row);h,w=im.shape[:2];origin=np.array(row['origin']);g=gt[i,j];b=z['base'][i,j];a=p['points'][i,j];o=z['oracle'][i,j];k=int(choice[i,j]);points=np.stack((g,b,a,o));diag=np.linalg.norm(np.maximum(np.array(row['box'])[2:]-np.array(row['box'])[:2],1));bound=.15*diag
        # Independent geometric reconstruction checks exact displayed normal/pair candidates.
        candidate=p['candidates'][i,j,k].astype(float)
        if 4<=k<7:
            ll=lines[k-4];v=-(ll[:2]@b+ll[2])*ll[:2];v*=min(1,bound/max(np.linalg.norm(v),1e-12));assert np.max(abs(v-candidate))<1e-3
        elif k>=7:
            ids=[(0,1),(0,2),(1,2)][k-7];L=lines[list(ids)];v=np.linalg.solve(L[:,:2],-(L[:,:2]@b+L[:,2]));v*=min(1,bound/max(np.linalg.norm(v),1e-12));assert np.max(abs(v-candidate))<1e-3
        # Actual learned mixture includes all10 candidates, not just the displayed3lines.
        actual=p['base'][i,j]+np.sum(p['weights'][i,j,:,None]*p['alpha'][i,j,:,None]*p['candidates'][i,j],axis=0);mixture_errors.append(float(np.max(abs(actual-a))));assert mixture_errors[-1]<1e-4
        rect=(min(0,points[:,0].min()-20),max(0,min(points[:,1].min()-70,h-160)),max(w,points[:,0].max()+20),min(h,max(points[:,1].max()+70,160)))
        lo=points.min(0)-15;hi=points.max(0)+15;center=(lo+hi)/2;span=max(float((hi-lo).max()),60);zoom=(center[0]-span/2,center[1]-span/2,center[0]+span/2,center[1]+span/2)
        fig,axes=plt.subplots(1,3,figsize=(16,5.8),layout='constrained');axes[0].imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));rh,rw=raw.shape[:2];x,y=origin;axes[0].add_patch(Rectangle((x,y),w,h,fill=False,ec='#ff8c00',lw=2));
        if x>0:axes[0].axvspan(0,x,color='black',alpha=.4)
        if x+w<rw:axes[0].axvspan(x+w,rw,color='black',alpha=.4)
        axes[0].scatter(*(g+origin),c=COLORS['GT'],marker='x',s=90,linewidths=2);axes[0].text(*(g+origin+np.array([5,-6])),f'GT {j}',c='white',bbox=dict(fc='black',alpha=.7));axes[0].set_title('Uncropped raw: context only\nOrange box = input retained by model',fontsize=11);axes[0].axis('off')
        point_panel(axes[1],im,g,b,a,o,lines,k,j,rect);point_panel(axes[2],im,g,b,a,o,lines,k,j,zoom,True);handles,labels=axes[2].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=7,fontsize=9,bbox_to_anchor=(.5,-.07));fig.suptitle(f'{key} | seed1 / frame {row["id"].split("/")[-1]} / point {j}\nBase {be[i,j]:.2f}px   V27 {ae[i,j]:.2f}px   GT oracle {oe[i,j]:.2f}px | {CANDIDATE[k]}',fontsize=13);path=RUN/'figures'/f'{key}.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
        # Compact two-panel artifact intended for inline viewing.
        fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained');point_panel(axes[0],im,g,b,a,o,lines,k,j,rect);point_panel(axes[1],im,g,b,a,o,lines,k,j,zoom,True);handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=7,fontsize=8,bbox_to_anchor=(.5,-.09));fig.suptitle(f'{key}: Base {be[i,j]:.2f} / V27 {ae[i,j]:.2f} / Oracle {oe[i,j]:.2f}px\nSolid lines: inside input; dashed: extrapolation. Grey: cropped-out area.',fontsize=11);fig.savefig(RUN/'figures'/f'{key}_detail.png',dpi=160,bbox_inches='tight');plt.close(fig)
        case=dict(key=key,title=title,id=row['id'],seed=1,point=j,selection=selection,base_error_px=float(be[i,j]),actual_error_px=float(ae[i,j]),oracle_error_px=float(oe[i,j]),frame_outside_oracle_mean_px=float(oe[i,mask[i]].mean()),candidate=k,candidate_name=CANDIDATE[k],oracle_amount=float(z['amount'][i,j,k]),predicted_amount=float(p['alpha'][i,j,k]),predicted_weight_for_oracle_candidate=float(p['weights'][i,j,k]),candidate_endpoint=(b+candidate).tolist(),gt=g.tolist(),base=b.tolist(),actual=a.tolist(),oracle=o.tolist(),top3_line_indices=idx.tolist(),top3_lines=lines.tolist(),parent_top3_association=r['line_weights'][obs,j,idx].tolist(),crop_origin=row['origin'],input_hw=row['image_hw'],source_row=row,source_image_sha256=__import__('hashlib').sha256(raw.tobytes()).hexdigest());cases.append(case)
    text=['# V27: 8.78px은 이미지에서 어느 정도인가','',f'**8.78px은 실제 모델의 성능이 아니라, GT로 후보와 이동량을 최적 선택한 전체 평균({mean:.6f}px)이다.** 실사 잘림20%의 검출된 화면 밖 유효점558개/시드에서 구하고3시드를 평균했다. 아래는 시드1의 특정 점 사례이며, 개별 점 오차와 전체 평균을 구분한다.','',
    '8.78px은 원본 이미지 좌표에서 GT와 예측점 사이 직선거리다. 확대 화면에서는 더 크게 보이므로 각 그림의 **10px 눈금**을 기준으로 본다. 실제 길이mm로 바꾸려면 거리와 카메라/평면 정보가 필요하다.','',
    '**표시:** 녹색×=GT, 파랑○=기존 V12점, 빨강□=실제 V27 출력, 자홍★=GT oracle. 주황/보라/청록 선=L1/L2/L3(Hough 분기가 만든 선 중 해당 점과 association이 높은3개). 이는 새 Canny/HoughLines 실행이나 attention heatmap이 아니다. 선의 순위가 GT 정확도나 보정 효용 순위를 뜻하지 않는다.','',
    '**실선**은 입력 이미지 안의 예측 선, **점선**은 이미지 밖으로 연장한 선이다. 회색 영역은 실제 입력에서 잘려 나간 부분이다. 점선은 그 위치의 edge를 실제로 관측했다는 뜻이 아니다. 원본 패널은 상황 설명용이고 모델은 잘린 입력만 봤다. GT oracle는 점 후보도 선택할 수 있으며, 그 결과가 반드시 Hough 선 위에 놓이는 것은 아니다.','',
    '| 사례·대상점 | 기존 점 | 실제 V27 | GT oracle | oracle 후보 |','|---|---:|---:|---:|---|']
    for c in cases:text.append(f'| {c["title"]} · ID{c["point"]} | {c["base_error_px"]:.2f} | {c["actual_error_px"]:.2f} | {c["oracle_error_px"]:.2f} | {c["candidate_name"]} |')
    descriptions={
    'good':'실제 모델도 점 오차를 줄인 예다. GT oracle는 두 선 교점 방향의 후보를 선택하면 더 줄일 수 있음을 보여준다. 실제 모델의 개선 전부가 선 덕분이라는 뜻은 아니다.',
    'near_mean':'이 점의 oracle 오차8.79px이 전체 평균8.78px과 거의 같다. 자홍 별과 녹색× 사이의 점선 거리를10px 눈금과 비교하면 크기를 알 수 있다. 이 사례가 전체 데이터의 대표 이미지라는 뜻은 아니다.',
    'missed':'선 후보와 적절한 이동량이 있어도 실제 모델이 그것을 선택하지 못한 경우다. GT oracle는 맞추지만 실제 빨간점은 크게 벗어난다. 회색 영역의 점선은 화면 밖으로 연장한 예측 선이다.',
    'oracle_bad':'GT로 최적 후보와 이동량을 선택해도 큰 오차가 남는 경우다. 현재 후보의 위치/방향/이동 범위가 이 점을 충분히 복원하지 못한다. 수동 GT를 새로 검증한 사례는 아니므로 GT 자체의 완전한 정확성을 인증하지 않는다.'}
    for c in cases:text+=['',f'## {c["title"]}','',descriptions[c['key']],'',f'원본 `{c["id"]}`, 점 ID{c["point"]}. 선택 후보: **{c["candidate_name"]}**. GT 최적 이동 비율 {c["oracle_amount"]:.3f}, 실제 모델의 해당 후보 이동 비율 {c["predicted_amount"]:.3f}, 혼합 가중치 {c["predicted_weight_for_oracle_candidate"]:.3%}. 이동 비율은 제한된 후보 벡터에 곱하는 값이며 가중치는 검증된 확률이 아니다.','',f'![{c["title"]}](figures/{c["key"]}.png)','']
    text+=['## 선택 기준과 검증','', '좋은 사례는 선 후보 oracle<3px이고 실제 모델도 개선한 점 중 최대 개선, 평균 근처는8.78px에 가장 가까운 점, 선택 실패는 oracle<5px에서 실제와 oracle 차이가 가장 큰 점, oracle 실패는 oracle 잔차가 가장 큰 점을 골랐다. 모두 **결과를 기준으로 고른 설명용 사례**이며 분포를 대표하는 무작위 표본이나 새 평가가 아니다.','',
    '저장된 V27 후보·GT에서 oracle 평균을 다시 계산해 이전 결과와 일치함을 검사했다. 표시한 Hough 선의 실제 캐시와 점 ID를 맞췄고, 선택된 법선/교점 후보의 기하를 독립 재계산했다. 실제 모델의 점은 저장된10개 후보·이동량·혼합 가중치에서 재현했다. 원본 이미지와 잘림 좌표를 함께 보존했다. 새 학습이나 평가 수치 변경은 없다.','', '[전체 사례 좌표·선 계수·선택 근거](CASES.json) · [검사 결과](AUDIT.json) · [V27 전체 보고서](../pallet_direction_utility_v27/index.html)','']
    report='\n'.join(text);(RUN/'REPORT_KO.md').write_text(report);body=markdown.markdown(report,extensions=['tables']);(RUN/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>V27 점·Hough 실제 사례</title><style>body{font:17px/1.75 sans-serif;max-width:1450px;margin:35px auto;padding:0 24px;color:#183048}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd5de}th{background:#eaf0f6}h2{margin-top:2.5em}a{color:#096ba9}</style>'+body+'</html>')
    write(RUN/'CASES.json',dict(oracle_mean_px=mean,per_seed_means=means,points_per_seed=558,cases=cases,sources=sources,source_code_sha256=sha(Path(__file__))))
    unchanged=0
    for path,h in read(PARENT/'COMPLETION.json')['files'].items():assert sha(ROOT/path)==h,path;unchanged+=1
    write(RUN/'AUDIT.json',dict(status='PASS',oracle_mean_replayed_px=mean,selected_normal_intersection_candidate_replay_atol_px=.001,actual_mixture_replay_atol_px=1e-4,actual_mixture_replay_max_px=max(mixture_errors),float32_reduction_order_tolerance=True,prior_V27_files_unchanged=unchanged,GT_not_newly_reannotated=True,not_new_model_inference=True))
    import re
    prefix='../../../../data/pallet/results/pallet_direction_utility_v27_cases/';NOTE.write_text(re.sub(r'\]\(([^)]+)\)',lambda m:']('+prefix+m.group(1)+')',report));print('CASES COMPLETE',mean,[(c['key'],c['id'],c['point'],c['oracle_error_px']) for c in cases],flush=True)

if __name__=='__main__':main()
