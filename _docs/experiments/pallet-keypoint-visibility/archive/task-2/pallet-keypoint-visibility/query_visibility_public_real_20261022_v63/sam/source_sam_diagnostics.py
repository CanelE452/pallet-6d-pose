"""Fixed qualitative source cases, selected after scoring; never model selection."""
import csv,json,gzip
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent;TASK=ROOT.parent.parent

def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def csvrows(p):
    with Path(p).open(newline='')as f:return list(csv.DictReader(f))
rows=[r for r in csvrows(HERE/'SOURCE_QUERY_SCORED.csv')if r['radius']=='0']
inputs={r['id']:r for r in lines(HERE/'SOURCE_INPUTS.jsonl')};truth={r['id']:r for r in lines(HERE/'SCORING_ONLY_MASKS.jsonl')};pred={r['id']:r for r in lines(HERE/'TARGET_PREDICTIONS.jsonl')}
ycb={r['sample_id']:r for r in csvrows(ROOT/'source_cache/v2/YCB_VALIDATION_QUERIES.csv')}
meta={r['id']:r for r in json.loads((TASK/'external_ray_proxy_head_20261022_v60/SYNTH_FEATURE_METADATA.json').read_text())}
spec=[('YCB external false presence','YCB','EXTERNAL_OCCLUDED_SUPPORT','True'),('YCB visible false absence','YCB','VISIBLE_TARGET_SUPPORT','False'),('SYNTH visible false absence','SYNTH','VISIBLE_TARGET_SUPPORT','False'),('SYNTH external correct absence','SYNTH','EXTERNAL_OCCLUDED_SUPPORT','False')]
chosen=[]
for title,source,state,p in spec:
    candidates=[r for r in rows if r['source']==source and r['source_support_state']==state and r['pred_visible_support']==p]
    r=min(candidates,key=lambda r:(r['id'],int(r['native_corner'])))
    chosen.append(dict(title=title,**r))
fig,axs=plt.subplots(4,4,figsize=(15,13),layout='constrained')
for i,r in enumerate(chosen):
    inp=inputs[r['id']];gt=truth[r['id']];im=np.array(Image.open(inp['image']).convert('RGB'))
    threshold=127 if r['source']=='SYNTH' else 0;vis=np.array(Image.open(gt['mask_vis']))>threshold;full=np.array(Image.open(gt['mask_full']))>threshold
    with np.load(pred[r['id']]['mask_path'])as z:sam=z['mask'].astype(bool)
    q=np.array([float(r['query_x']),float(r['query_y'])]);L=float(ycb[r['query_id']]['roi_side'])if r['source']=='YCB'else float(meta[r['id']]['side'])
    lo=q-L/2;hi=q+L/2
    for j,ax in enumerate(axs[i]):
        ax.set_axis_off()
        if j<2:ax.imshow(im,interpolation='nearest')
        elif j==2:
            overlay=np.zeros((*vis.shape,3),np.uint8);overlay[full&~vis]=[230,130,35];overlay[vis]=[40,205,90];ax.imshow(overlay,interpolation='nearest')
        else:ax.imshow(sam,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        if j==0:
            box=inp['prompt_box_xyxy'];ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,edgecolor='yellow',lw=.9));ax.add_patch(Rectangle(lo,L,L,fill=False,edgecolor='cyan',lw=.9))
        else:ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.plot(*q,'+',color='cyan',ms=11,mew=1.5)
        ax.set_title([r['title'],'Unchanged source query crop','GT green=visible, orange=external','Selected SAM mask'][j],fontsize=10)
    axs[i,0].text(0,-.08,f"{r['id']} k{r['native_corner']}",transform=axs[i,0].transAxes,fontsize=8)
    axs[i,1].text(0,-.08,f"SAM logit={float(r['visible_support_logit']):.3f}; GT={r['source_support_state']}",transform=axs[i,1].transAxes,fontsize=8)
fig.suptitle('Frozen box-only MobileSAM: source ownership/boundary diagnostics',fontsize=14)
fig.supxlabel('Source-only qualitative diagnostics. Cyan + = unchanged stored query. Yellow = source-locked box prompt. No query point is a SAM prompt.\nSynthetic GT threshold >127; YCB GT >0. SAM logit threshold0 and predicted-IoU argmax frozen before source scoring.',fontsize=9)
fig.savefig(HERE/'SOURCE_SAM_DIAGNOSTICS.png',dpi=130);plt.close(fig)
(HERE/'SOURCE_SAM_DIAGNOSTIC_CASES.json').write_text(json.dumps(chosen,indent=2)+'\n')
print(json.dumps({'cases':len(chosen),'out':str(HERE/'SOURCE_SAM_DIAGNOSTICS.png')}))
