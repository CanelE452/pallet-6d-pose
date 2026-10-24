"""Scientific source-data visualization; original RGB and masks are read only."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
V=Path(__file__).resolve().parent;O=V/'factual_sample_v1/original';S=V/'factual_sample_v1/TINY_SOURCE_DOWNLOAD_COMPLETE.json'
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
if __name__=='__main__':
    assert bind(S)['sha256']=='5ba791ec8563224af01af1f177a1082431a225f544e86a3e670482dc08f2ad61'
    seal=json.loads(S.read_text());assert seal['completed']and seal['PASS']
    paths=[O/'val/000000/rgb/000005.jpg',O/'val/000000/depth/000005.png',O/'val/000000/mask_visib/000005_000000.png',O/'val/000000/mask_visib/000005_000001.png']
    declared={x['local_path']:x for x in seal['files']}
    for p in paths:assert bind(p)['sha256']==declared[str(p)]['sha256']
    before=[bind(p)for p in paths]
    rgb=np.array(Image.open(paths[0]).convert('RGB'));depth=np.array(Image.open(paths[1]));a,b=[np.array(Image.open(p))>0 for p in paths[2:]]
    overlap=a&b;zero=overlap&(depth==0);assert int(overlap.sum())==19371 and int(zero.sum())==473
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf');plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax[0,0].imshow(rgb);ax[0,0].set_title('원 RGB · val/000000/frame5 · 팔레트 두 개')
    ax[0,1].imshow(a,cmap='gray',vmin=0,vmax=1);ax[0,1].set_title(f'제공 visible mask · instance0 · {a.sum():,}px')
    ax[1,0].imshow(b,cmap='gray',vmin=0,vmax=1);ax[1,0].set_title(f'제공 visible mask · instance1 · {b.sum():,}px')
    ax[1,1].imshow(overlap,cmap='gray',vmin=0,vmax=1);ax[1,1].set_title('두 visible mask 교집합 19,371px\n이 중 depth 유효18,898px / depth0 473px')
    for t in ax.ravel():t.axis('off')
    fig.suptitle('V151 · 제공 visible mask는 서로 배타적인 pixel-owner 정답이 아님',fontsize=16)
    fig.text(.5,-.015,'MR6D · Gouda et al., 2025 · CC-BY-4.0 · 원본은 변경하지 않았으며 D/E 라벨을 생성하지 않았습니다.',ha='center',fontsize=10)
    p=V/'PROVIDED_MASK_OVERLAP_KO.png';fig.savefig(p,dpi=130,bbox_inches='tight');plt.close(fig)
    assert before==[bind(p)for p in paths]
    r=dict(completed=True,source_seal=bind(S),sources=before,plot=bind(p),overlap19371=int(overlap.sum()),overlap_depth_positive18898=int((overlap&(depth>0)).sum()),overlap_depth_zero473=int(zero.sum()),decode_for_visual_only='PIL JPEG RGB, not an initializer/encoder transport',native_labels_created=0,model_or_fit_calls=0)
    with (V/'PROVIDED_MASK_OVERLAP_PLOT_RECEIPT.json').open('x')as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(plot=bind(p),overlap=int(overlap.sum())),ensure_ascii=False))
