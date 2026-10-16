"""Actual RGB inputs and paired silhouette supervision, fixed training examples."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-v17-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from context17 import *
from model17 import geometry,EDGES
import torch.nn.functional as F

def main():
    setup();plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False});d=RUN/'figures';d.mkdir(exist_ok=True)
    m,O=load('heldout','clean');rep=repcache('heldout','clean',1);ts=C.target_rows('heldout','clean');detrows=[(r,t) for r,t in zip(m['rows'],ts) if r['detected']]
    provenance={r['id']:r for r in read(RUN/'MASK_PROVENANCE.json')['rows'] if r['split']=='heldout'};cache=torch.load(RUN/'training_roles_seed1.pt',weights_only=True);centers=geometry(O,rep,candidates(O,rep))['center'].numpy()
    selected=sorted([i for i,(r,t) in enumerate(detrows) if r['id'] in provenance and all(t['valid'])],key=lambda i:digest('v17-visual:'+detrows[i][0]['id']))[:4];gallery=[]
    for i in selected:
        r,t=detrows[i];raw=PREP.raw_image(r);mask=cv2.imread(str(ROOT/provenance[r['id']]['path']),0)>0;gt=np.array(t['points']);box=np.array(r['box']);rgb=(O['rgb'][i]+.5).numpy().transpose(1,2,0);low=F.interpolate(F.interpolate(O['rgb'][i:i+1],size=24,mode='area'),size=96,mode='bilinear',align_corners=True)[0].numpy().transpose(1,2,0)+.5
        fig,axes=plt.subplots(2,2,figsize=(13,9));ax=axes[0,0];ax.imshow(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB));ax.contour(mask,levels=[.5],colors=['cyan'],linewidths=.8)
        for u,v in EDGES:ax.plot(*gt[[u,v]].T,color='#f97316',lw=.6)
        ax.scatter(*gt.T,color='#16a34a',marker='+',s=30);ax.set_title('렌더러 원본 · 주황 cuboid / 청록 실제 가시 실루엣')
        uv=(gt-box[:2])/np.maximum(box[2:]-box[:2],1)*95
        for ax,im,title in [(axes[1,0],low,'저해상도 입력: 원본96 → 24 → 96'),(axes[1,1],rgb,'고해상도 입력: 원본에서 직접 읽은96')]:
            ax.imshow(np.clip(im,0,1),interpolation='nearest');ax.set_title(title);ax.set_xlim(-.5,95.5);ax.set_ylim(95.5,-.5)
        ax=axes[0,1];ax.imshow(np.clip(rgb,0,1),interpolation='nearest');p=1;pos=(centers[i,p,0]-box[:2])/np.maximum(box[2:]-box[:2],1)*95;old=cache['proxy'][i,p,0].numpy();new=cache['masked'][i,p,0].numpy()
        for l in range(3):ax.plot(*pos[l].T,color='#64748b',lw=.6)
        keep=(old>0)&(new>0);changed=(old>0)&(new==0)
        if keep.any():ax.scatter(*pos[keep].T,facecolors='none',edgecolors='#16a34a',s=60,label='실루엣 지지 경계')
        if changed.any():ax.scatter(*pos[changed].T,color='#ef4444',marker='x',s=70,label='기존 양성 → 실루엣 지지 없음')
        ax.scatter(*uv[p],color='#16a34a',marker='+',s=100,label='GT 점1');ax.set_xlim(-.5,95.5);ax.set_ylim(95.5,-.5);ax.set_title('같은 선 샘플의 보조 정답 변경 · 유지 후보 주변');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.08),ncol=2)
        fig.suptitle(r['id']+' · 학습 원본 SHA 고정 예시\n마스크는 물체 실루엣만 제공하며 내부 면 경계의 가시성 정답이 아님',fontsize=12);fig.tight_layout(rect=(0,0,1,.93),h_pad=3)
        name=f'paired_inputs_{len(gallery)}.png';fig.savefig(d/name,dpi=115,bbox_inches='tight');plt.close(fig);gallery.append(dict(id=r['id'],path='figures/'+name,selection='SHA-fixed synthetic training example; no evaluation-performance selection',point=1))
    write(RUN/'GALLERY.json',dict(rows=gallery));print('VISUAL COMPLETE',len(gallery),flush=True)

if __name__=='__main__':main()
