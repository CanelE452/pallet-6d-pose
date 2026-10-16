"""Original pixels, remaining line support and fixed/gated actual correction arrows."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v21-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from context21 import *
from model20 import observed_samples
from eval21 import prediction_path

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});folder=RUN/'figures';folder.mkdir(exist_ok=True);gallery=[]
    for condition,typ in [('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        m,O=load('real',condition);ts=targets('real',condition);rep=repcache('real',condition,1);sample=observed_samples(O,rep,'line');positions=sample['pos'].numpy();sv=sample['valid'].numpy();pred={a:np.load(prediction_path('real',condition,a+'_seed1')) for a in ('base','unit_old_sum','unit_new_sum','unit_old_pcgrad','unit_new_pcgrad')};gt=np.array([t['points'] for t in ts]);mask=np.array([t['valid'] for t in ts])&np.array([t[typ] for t in ts])&np.array([r['detected'] for r in m['rows']])[:,None];pairs=np.argwhere(mask);gain=np.linalg.norm(pred['unit_old_sum']['points']-gt,axis=-1)-np.linalg.norm(pred['unit_new_pcgrad']['points']-gt,axis=-1);harm=np.linalg.norm(pred['unit_new_pcgrad']['points']-gt,axis=-1)-np.linalg.norm(pred['base']['points']-gt,axis=-1);fixed=min(pairs.tolist(),key=lambda x:digest('v21-case:'+ts[x[0]]['id']+':'+str(x[1])))
        choices=[('SHA 고정',fixed),('기존 일반학습 대비 개선 최대 · 사후 선택',pairs[np.argmax(gain[mask])]),('기준점 대비 악화 최대 · 사후 선택',pairs[np.argmax(harm[mask])])];detindex=np.cumsum([r['detected'] for r in m['rows']])-1
        for selection,(i,p) in choices:
            r=m['rows'][i];raw=PREP.raw_image(r);origin=np.array(r['origin']);h,w=r['image_hw'];x,y=origin;im=raw[y:y+h,x:x+w].copy()
            if r.get('mask'):
                a,b,c,d=r['mask'];im[b:d,a:c]=r['fill_bgr']
            base=pred['base']['points'][i,p];truth=gt[i,p];full=pred['unit_old_sum']['points'][i,p];new=pred['unit_new_pcgrad']['points'][i,p];alpha=float(pred['unit_new_pcgrad']['alpha'][i,p]);a1=float(pred['unit_old_pcgrad']['alpha'][i,p]);before=np.linalg.norm(base-truth);old=np.linalg.norm(full-truth);after=np.linalg.norm(new-truth);j=detindex[i];pos=positions[j,p];valid=sv[j,p]
            fig,axes=plt.subplots(1,3,figsize=(19,6));axes[0].imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));v=np.array(ts[i]['valid']);original=gt[i]+origin;axes[0].scatter(*original[v].T,c='#22c55e',marker='+',s=55);axes[0].scatter(*(truth+origin),facecolors='none',edgecolors='#22c55e',s=180);axes[0].set_title('원본 이미지 · 수동 GT · 원은 분석할 점')
            coords=np.stack([truth,base,full,pred['unit_old_pcgrad']['points'][i,p],new,pred['unit_new_sum']['points'][i,p]]);bb=np.array(r['box']);region=np.vstack((coords,bb.reshape(2,2)))
            for ax in axes[1:]:
                ax.set_facecolor('#cbd5e1');ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.scatter(*pos[valid].T,s=10,color='#06b6d4')
                for line,keep in zip(pos.reshape(3,12,2),valid.reshape(3,12)):ax.plot(*np.where(keep[:,None],line,np.nan).T,c='#06b6d4',lw=.7,alpha=.6)
                ax.scatter(*base,c='white',edgecolors='black',s=35,zorder=5)
            lo=region.min(0)-20;hi=region.max(0)+20;axes[1].set_xlim(lo[0],hi[0]);axes[1].set_ylim(hi[1],lo[1]);axes[1].scatter(*truth,c='#16a34a',marker='+',s=60);axes[1].set_title('실제 입력 · 남은 선 구간 샘플\n청록은 샘플 위치이며 기여도 지도가 아님')
            ax=axes[2]
            for point,color,label in [(truth,'#16a34a','GT까지 필요한 보정'),(full,'#8b5cf6','기존 데이터 · 일반학습'),(pred['unit_new_sum']['points'][i,p],'#64748b','새 데이터 · 일반학습'),(pred['unit_old_pcgrad']['points'][i,p],'#f59e0b',f'기존 데이터 · PCGrad · {a1:.1%}'),(new,'#ef4444',f'새 데이터 · PCGrad · {alpha:.1%}')]:
                ax.annotate('',xy=point,xytext=base,arrowprops=dict(arrowstyle='->',color=color,lw=2));ax.scatter(*point,color=color,marker='+',s=70,label=label,zorder=6)
            lo=coords.min(0)-20;hi=coords.max(0)+20;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal');ax.set_title(f'점{p} · 화살표 증폭 없음\n오차: 기준 {before:.2f} / 기존 {old:.2f} / 수정 {after:.2f}px');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1));caption=f'{condition} / {selection} / seed1 / {r["id"]} / 점{p}';fig.suptitle(caption+'\n회색: 입력 밖 · 실제 잔차 데이터 변경 × stop/move PCGrad',fontsize=11);fig.tight_layout(rect=(0,0,1,.9));name=f'case_{len(gallery)}.png';fig.savefig(folder/name,dpi=120,bbox_inches='tight');plt.close(fig)
            gallery.append(dict(id=caption,path='figures/'+name,condition=condition,frame_id=r['id'],point=int(p),seed=1,selection=selection,base_error_px=float(before),previous_error_px=float(old),corrected_error_px=float(after),alpha_equal_only=a1,alpha_corrected=alpha))
    write(RUN/'GALLERY.json',dict(rows=gallery,scope='Fixed seed1 and SHA examples plus clearly labelled posthoc extreme cases. No representative performance claim from images. GT never used for inference.'));print('VISUAL COMPLETE',len(gallery),flush=True)

if __name__=='__main__':main()
