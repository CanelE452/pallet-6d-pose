"""Scientific overlays of saved coordinates, not generated/retouched pictures."""
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_large_error_refiner_v1'
RAW = ROOT / 'data/pallet/results/pallet_large_error_refiner_v1'
OUT = ROOT / 'outputs/pallet_large_error_refiner_v1'
NAMES = ('R0','A_N2','B_CAP32','C_SYN_NORMAL','D_SYN_LARGE','E_SYN_REAL_LARGE')
LABELS = ('R0 raw','A: current N2','B: N2 cap 32px','C: wide, normal synthetic','D: wide, large-error synthetic','E: D + manual real')
EXAMPLE = 'eval_pallet07:1778652166837872128'


def read(path): return json.loads(path.read_text())


def main():
    predictions = read(RAW/'PREDICTIONS.json'); metrics = read(RAW/'PER_FRAME_METRICS.json')
    gates = read(RAW/'GATES.json'); split = read(DOC/'SPLIT.json')
    green = read(ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json')
    annotations = {r['id']:r['annotation'] for r in split['evaluation']+green['records']}
    candidates = []
    for dataset in ('DEV72','GREEN150'):
        scorekey = dataset if dataset=='DEV72' else 'GREEN150_MANUAL'
        scores = {k:{r['id']:r for r in v} for k,v in metrics[scorekey]['ungated'].items()}
        gg = {r['id']:r['gates'] for r in gates[dataset]}
        for row in predictions[dataset]:
            ident = row['id']; a,e = scores['A_N2'][ident],scores['E_SYN_REAL_LARGE'][ident]
            if not a['evaluable'] or not a['matched'] or not e['matched']: continue
            candidates.append(dict(dataset=dataset,row=row,scores={k:v[ident] for k,v in scores.items()},gates=gg[ident],
                delta=e['frame_mean_px']-a['frame_mean_px']))
    selected = [('user_example',next(r for r in candidates if r['row']['id']==EXAMPLE)),
                ('largest_E_improvement',min(candidates,key=lambda r:(r['delta'],r['row']['id']))),
                ('largest_E_worsening',max(candidates,key=lambda r:(r['delta'],r['row']['id'])))]
    edges = read(ROOT/'_docs/experiments/pallet_symmetry_three_line_v1/OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['edges']
    OUT.mkdir(parents=True,exist_ok=True); manifest = []
    for tag,entry in selected:
        row = entry['row']; path = ROOT/row['image']['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['image']['sha256']
        annotation = annotations[row['id']]; annpath = ROOT/annotation['path']
        assert hashlib.sha256(annpath.read_bytes()).hexdigest()==annotation['sha256']
        doc = read(annpath); points = doc['objects'][0]['keypoint_annotations']
        gt = np.array([p['xy'] if p.get('xy') is not None else [np.nan,np.nan] for p in points])
        im = np.array(Image.open(path).convert('RGB')); h,w = im.shape[:2]
        pred = {name:np.array(p['candidates'][p['selected_index']]['keypoints_xy']) for name,p in row['ungated'].items()}
        raw = pred['R0']; movement = {name:np.linalg.norm(q[:8]-raw[:8],axis=-1).tolist() for name,q in pred.items()}
        for zoom in [False,True] if tag=='user_example' else [False]:
            fig,axs = plt.subplots(2,3,figsize=(15,9),facecolor='#141b25')
            for ax,name,label in zip(axs.flat,NAMES,LABELS):
                ax.imshow(im); q = pred[name]
                for a,b in edges:
                    ax.plot(gt[[a,b],0],gt[[a,b],1],color='#77ff65',lw=1.4,alpha=.85)
                    if name!='R0': ax.plot(raw[[a,b],0],raw[[a,b],1],color='#00d9ff',lw=.9,ls='--',alpha=.55)
                    ax.plot(q[[a,b],0],q[[a,b],1],color='#00d9ff' if name=='R0' else '#ffac39',lw=1.4)
                ax.scatter(q[:8,0],q[:8,1],s=14,c='#ffe05a',zorder=5)
                if zoom:
                    ax.set_xlim(475,640); ax.set_ylim(425,245)
                    if name!='R0':
                        for a,b in zip(raw[:8],q[:8]):
                            if 475<=b[0]<=640 and 245<=b[1]<=425:
                                ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='white',lw=1.1),zorder=6)
                else: ax.set_xlim(0,w); ax.set_ylim(h,0)
                score = entry['scores'][name]; g = entry['gates'][name]
                move = np.array(movement[name])
                ax.set_title(f"{label}\nmove mean/max {move.mean():.1f}/{move.max():.1f}px | gate {'PASS' if g['accepted'] else 'FALLBACK to A'}\nwhole-object-sym mean error {score['frame_mean_px']:.1f}px",fontsize=10,color='white')
                ax.axis('off')
            title = ('User-specified right edge (same crop for every model)' if zoom else tag.replace('_',' '))
            fig.suptitle(f'{title}\n{row["id"]}',color='white',fontsize=14,y=.98)
            fig.text(.5,.025,'Green = stored annotation (includes generated/unknown-source points); cyan dashed = R0; orange = output\n'
                'UNGATED predictions shown. Gates use no GT; rejected output returns to A. No per-corner GT reassignment.',ha='center',color='white',fontsize=10)
            fig.subplots_adjust(top=.84,bottom=.1,left=.015,right=.985,wspace=.05,hspace=.34)
            dest = OUT/f'{tag}{"_right_zoom" if zoom else ""}.png'
            fig.savefig(dest,dpi=140,facecolor=fig.get_facecolor()); plt.close(fig)
        manifest.append(dict(tag=tag,id=row['id'],dataset=entry['dataset'],delta_E_vs_A_frame_mean_px=entry['delta'],
            movement_px=movement,gates=entry['gates'],scores=entry['scores'],image=row['image'],annotation=annotation,
            selected_by='user fixed example' if tag=='user_example' else 'E minus A symmetry-aware frame mean error, matched cases from fixed DEV72+GREEN150 manual; no training decisions'))
    output = dict(figures=manifest,selection_population=len(candidates),
        predictions_sha256=hashlib.sha256((RAW/'PREDICTIONS.json').read_bytes()).hexdigest(),
        note='No generated imagery, inference rerun, arbitrary GT correspondence, or checkpoint reselection')
    (OUT/'FIGURES.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(OUT)
    for r in manifest: print(r['tag'],r['id'],r['delta_E_vs_A_frame_mean_px'])


if __name__=='__main__': main()
