"""Raw pixels, actual correction vectors and line sampling locations; no heatmap claim."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v18-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from context18 import *
from model18 import geometry
from eval18 import prediction_path

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});folder=RUN/'figures';folder.mkdir(exist_ok=True);gallery=[]
    for condition,typ in [('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        m,O=load('real',condition);ts=targets('real',condition);rep=repcache('real',condition,1);geo=geometry(O,rep,rep['base'][:,:,None]);centers=geo['center'][:,:,0].numpy();samples=geo['sample'][:,:,0].numpy()
        pred={a:np.load(prediction_path('real',condition,a+'_seed1'))['points'] for a in ('base','point_vector','line_vector','line_risk')};gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts])&np.array([t[typ] for t in ts])&np.array([r['detected'] for r in m['rows']])[:,None]
        pairs=np.argwhere(valid);gain=np.linalg.norm(pred['base']-gt,axis=-1)-np.linalg.norm(pred['line_vector']-gt,axis=-1);fixed=min(pairs.tolist(),key=lambda x:digest('v18-case:'+ts[x[0]]['id']+':'+str(x[1])))
        choices=[('SHA 고정',fixed),('개선 최대 · 사후 선택',pairs[np.argmax(gain[valid])]),('악화 최대 · 사후 선택',pairs[np.argmin(gain[valid])])]
        detindex=np.cumsum([r['detected'] for r in m['rows']])-1
        for selection,(i,p) in choices:
            r=m['rows'][i];raw=PREP.raw_image(r);origin=np.array(r['origin']);h,w=r['image_hw'];x,y=origin;im=raw[y:y+h,x:x+w].copy()
            if r.get('mask'):
                a,b,c,d=r['mask'];im[b:d,a:c]=r['fill_bgr']
            base=pred['base'][i,p];truth=gt[i,p];step=pred['line_vector'][i,p]-base;target=truth-base;before=np.linalg.norm(target);after=np.linalg.norm(pred['line_vector'][i,p]-truth);dot=float(step@target);cos=dot/max(np.linalg.norm(step)*before,1e-9)
            fig,axes=plt.subplots(1,3,figsize=(19,6));axes[0].imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));v=np.array(ts[i]['valid']);original=gt[i]+origin;axes[0].scatter(*original[v].T,c='#22c55e',marker='+',s=55);axes[0].scatter(*(truth+origin),facecolors='none',edgecolors='#22c55e',s=180);axes[0].set_title('원본 이미지 · 수동 GT만 표시 · 원은 분석할 점')
            ax=axes[2];ax.set_facecolor('#cbd5e1');ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));j=detindex[i]
            for line in centers[j,p]:
                inside=(line[:,0]>=0)&(line[:,0]<w)&(line[:,1]>=0)&(line[:,1]<h)
                ax.plot(*np.where(inside[:,None],line,np.nan).T,color='#06b6d4',lw=1,alpha=.7)
            sm=samples[j,p].reshape(-1,2);bb=np.array(r['box']);inside=(sm[:,0]>=max(0,bb[0]))&(sm[:,0]<min(w,bb[2]))&(sm[:,1]>=max(0,bb[1]))&(sm[:,1]<min(h,bb[3]));ax.scatter(*sm[inside].T,s=9,color='#06b6d4',alpha=.6,label='해당 점의 선 샘플')
            ax.scatter(*base,c='white',edgecolors='black',s=50,label='보정 시작점',zorder=5)
            for point,color,label in [(truth,'#16a34a','GT까지 필요한 보정'),(pred['point_vector'][i,p],'#8b5cf6','점 특징 보정'),(pred['line_risk'][i,p],'#f59e0b','선+악화 벌점 보정'),(pred['line_vector'][i,p],'#ef4444','선 특징 보정')]:
                ax.annotate('',xy=point,xytext=base,arrowprops=dict(arrowstyle='->',color=color,lw=2));ax.scatter(*point,color=color,marker='+',s=70,label=label,zorder=6)
            coords=np.stack([truth,base]+[pred[a][i,p] for a in ('point_vector','line_vector','line_risk')]);lo=coords.min(0)-30;hi=coords.max(0)+30;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal');ax.set_title(f'점{p} · 실제 좌표 화살표(증폭 없음)\n오차 {before:.2f} → {after:.2f}px · 이동 {np.linalg.norm(step):.2f}px · 방향 cos {cos:.3f}');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
            overview=axes[1];overview.set_facecolor('#cbd5e1');overview.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));overview.scatter(*sm[inside].T,s=10,color='#06b6d4');overview.scatter(*base,c='white',edgecolors='black',s=35);overview.scatter(*truth,c='#16a34a',marker='+',s=60)
            for line in centers[j,p]:
                keep=(line[:,0]>=0)&(line[:,0]<w)&(line[:,1]>=0)&(line[:,1]<h);overview.plot(*np.where(keep[:,None],line,np.nan).T,c='#06b6d4',lw=.7,alpha=.6)
            region=np.vstack((coords,bb.reshape(2,2)));rl=region.min(0)-20;rh=region.max(0)+20;overview.set_xlim(rl[0],rh[0]);overview.set_ylim(rh[1],rl[1]);overview.set_title('실제 가림/잘림 입력과 선 샘플 전체 위치\n청록 점: 이미지·ROI 안에 남은 샘플')
            caption=f'{condition} / {selection} / seed1 / {r["id"]} / 점{p}';fig.suptitle(caption+'\n회색: 잘려서 입력에 없는 영역 · 청록: 샘플 위치이며 attention 기여도 아님',fontsize=11);fig.tight_layout(rect=(0,0,1,.9));name=f'vector_{len(gallery)}.png';fig.savefig(folder/name,dpi=120,bbox_inches='tight');plt.close(fig)
            gallery.append(dict(id=caption,path='figures/'+name,selection=selection,condition=condition,point=int(p),frame_id=r['id'],seed=1,base_error_px=float(before),new_error_px=float(after),movement_px=float(np.linalg.norm(step)),cosine=cos,harm_type='wrong_direction' if after>before and dot<=0 else 'overshoot' if after>before else 'improved'))
    write(RUN/'GALLERY.json',dict(rows=gallery,scope='Fixed seed1; SHA examples plus posthoc extreme GT-selected point examples. No aggregate conclusion from selected images. Arrows unscaled. Line samples are not attribution.'))
    print('VISUAL COMPLETE',len(gallery),flush=True)

if __name__=='__main__':main()
