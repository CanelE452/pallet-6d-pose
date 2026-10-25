"""V259 publication layout only: use saved references/residuals, never compile labels."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


def safe(path, exists=True):
    p=Path(path)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112','cache','.pyc','current223','publicheld','holdout','oldval'))
    for a in [*reversed(p.parents),p]:
        try: mode=os.lstat(a).st_mode
        except FileNotFoundError:
            assert not exists
            continue
        assert not stat.S_ISLNK(mode)
    return p


def read(b):
    fd=os.open(safe(b['path']),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:v=f.read()
    assert len(v)==b['size_bytes'] and hashlib.sha256(v).hexdigest()==b['sha256']
    return v


def binding(p):
    v=safe(p).read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(v).hexdigest(),'size_bytes':len(v)}


def write(p,o):
    with safe(p,False).open('x') as f:json.dump(o,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return binding(p)


def run(cfg_path):
    cfg=json.loads(safe(cfg_path).read_bytes())
    assert cfg['approved_actual_visual_layout_only'] is True
    assert cfg['code']==binding(Path(__file__).absolute())
    root=json.loads(read(cfg['label_entry']))
    complete=json.loads(read(cfg['label_complete']))
    assert complete['completed'] is True and complete['PASS'] is True and complete['entry_sha256']==cfg['label_entry']['sha256']
    assert root['approved_public_aux_label_prototype'] is True and root['training_or_model_inference'] is False
    assert root['driver_sha256']==cfg['original_driver']['sha256']
    read(cfg['original_driver'])  # Audit fixed selection rule as text; no import/call.
    outputs={Path(b['path']).name:b for b in complete['files']}
    contexts=[];selections=[];reads=[cfg['label_entry'],cfg['label_complete'],cfg['original_driver']]
    for case in root['cases']:
        prefix=case['id'];ref_binding=outputs[prefix+'_REFERENCE_ALL_4800.jsonl']
        ref_raw=read(ref_binding)
        refs=[json.loads(line) for line in ref_raw.splitlines() if line]
        assert len(refs)==4800
        rgb=np.asarray(Image.open(io.BytesIO(read(case['RGB']))))
        sam=np.asarray(Image.open(io.BytesIO(read(case['SAM']))))
        residual=np.load(io.BytesIO(read(case['T_raw_residual'])),allow_pickle=False)
        assert rgb.shape==(480,640,3) and rgb.dtype==np.uint8
        assert sam.shape==residual.shape==(480,640) and set(np.unique(sam))<={0,255}
        assert residual.dtype==np.float64
        contexts.append((prefix,rgb,refs,residual,sam))
        reads.extend([ref_binding,case['RGB'],case['SAM'],case['T_raw_residual']])
        # Exact original driver rule; no outcome-based change or new sampling.
        for label in (0,1):
            candidates=[r for r in refs if r['r_derived_label']==label]
            candidates.sort(key=lambda r:hashlib.sha256((prefix+'|'+str(r['r_xy'])).encode()).hexdigest())
            limit=4 if prefix=='0000' and label==1 else 1
            for r in candidates[:limit]:selections.append((prefix,rgb,r,sam))
    assert len(selections)==3
    out=safe(cfg['output_directory'],False);assert not out.exists();out.mkdir()
    try:
        cmap=LinearSegmentedColormap.from_list('fixed_blue_white_red',[(0,'#3b4cc0'),(.5,'#ffffff'),(1,'#b40426')],N=257)
        norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=.1)
        assert float(norm(0))==.5 and np.allclose(cmap(norm(0)),[1,1,1,1],atol=0,rtol=0)
        fig,axes=plt.subplots(len(contexts),2,figsize=(13,10.4),squeeze=False,layout='constrained')
        fig.set_constrained_layout_pads(w_pad=.12,h_pad=.12,wspace=.04,hspace=.08)
        fig.suptitle('Two public pilot frames | saved surface AUX, not classifier output',fontsize=13)
        for row,(prefix,rgb,refs,residual,sam) in enumerate(contexts):
            axes[row,0].imshow(rgb)
            for label,color,name in ((0,'lime','D'),(1,'orange','E')):
                points=np.asarray([r['r_xy'] for r in refs if r['r_derived_label']==label],dtype=float).reshape(-1,2)
                if len(points):axes[row,0].scatter(points[:,0],points[:,1],c=color,s=5,label=name+' AUX')
            axes[row,0].legend(loc='upper left',fontsize=10)
            axes[row,0].set_title(f'Public {prefix} | saved reference labels\nD green / E orange; no model',fontsize=11,pad=7)
            im=axes[row,1].imshow(residual,cmap=cmap,norm=norm)
            cb=fig.colorbar(im,ax=axes[row,1],shrink=.92,ticks=[-1,-.5,-.1,0,.02,.05,.1])
            cb.set_label('Observed - rendered Z (m)\ndisplay range [-1, +0.1]',fontsize=10)
            cb.ax.tick_params(labelsize=9)
            axes[row,1].set_title('Raw residual | zero is white\ninvalid transparent; not model abstention',fontsize=11,pad=7)
            for ax in axes[row]:ax.set_axis_off()
        overview=out/'PUBLIC_AUX_LABELS_OVERVIEW_V2.png'
        fig.savefig(overview,dpi=150);plt.close(fig)
        fig,axes=plt.subplots(len(selections),3,figsize=(13,11.4),squeeze=False,layout='constrained')
        fig.set_constrained_layout_pads(w_pad=.10,h_pad=.10,wspace=.04,hspace=.08)
        fig.suptitle('Fixed 96-source-pixel windows | saved r labels, emulated q\nCyan +=q; green/orange ring=r. AUX, not classifier.',fontsize=13)
        selected=[]
        for row,(prefix,rgb,record,sam) in enumerate(selections):
            rx,ry=record['r_xy'];state='D' if record['r_derived_label']==0 else 'E'
            selected.append({'case':prefix,'row_order':row,'record':record,
                             'record_canonical_sha256':hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                             'original_selection_SHA_rank':hashlib.sha256((prefix+'|'+str(record['r_xy'])).encode()).hexdigest(),
                             'q_offsets_dx':[0,3,6],'q_offset_dy':0,'source_window_pixels':96})
            for col,dx in enumerate((0,3,6)):
                qx,qy=rx+dx,ry;ax=axes[row,col]
                ax.imshow(rgb,interpolation='nearest');ax.set_facecolor('black')
                ax.set_xlim(qx-48,qx+48);ax.set_ylim(qy+48,qy-48)
                ax.plot(qx,qy,'+',color='cyan',markersize=10)
                ax.plot(rx,ry,'o',markerfacecolor='none',markeredgecolor='lime' if state=='D' else 'orange',markersize=8)
                present=bool(sam[qy,qx]==255) if 0<=qx<640 and 0<=qy<480 else None
                ax.set_title(f'{prefix} | r {state} AUX / q dx=+{dx}\nq SAM={present} (emulated)',fontsize=11,pad=7)
                ax.set_axis_off()
        crops=out/'PUBLIC_R_Q_CROP_DIAGNOSTIC_V2.png'
        fig.savefig(crops,dpi=150);plt.close(fig)
        selected_binding=write(out/'SELECTED_RECORDS_UNCHANGED.json',selected)
        # Re-hash every read source leaf; no original label/query/map/PNG writes.
        for b in reads:assert binding(Path(b['path']))==b
        receipt=write(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':cfg['code'],'entry':binding(cfg_path),
                       'original_label_entry':cfg['label_entry'],'original_label_complete':cfg['label_complete'],
                       'inputs_exact':reads,'outputs':[binding(overview),binding(crops),selected_binding],
                       'layout_only':{'panel_title_font':11,'panel_title_lines':2,'zero_centered_norm':{'type':'TwoSlopeNorm','vmin':-1,'vcenter':0,'vmax':.1},
                                      'zero_RGBA':[1,1,1,1],'residual_array_numeric_change':False,'selected3_original_SHA_rank_exact':True,
                                      'same96_source_pixel_windows':True,'same_dx':[0,3,6],'same_colors':{'D':'lime','E':'orange','q':'cyan'}},
                       'all_V1_sources_unchanged':True,'label_compiler_or_label_data_recomputation':False,
                       'new_labels_queries_thresholds_training_model_depth_pose_mesh_current_held_reads':0,
                       'visual_review_pending':True})
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':receipt,'selections':[{'case':s['case'],'r_xy':s['record']['r_xy'],'state':s['record']['r_derived_label']} for s in selected]},sort_keys=True))
    except BaseException as exc:
        write(out/'FAILURE.json',{'completed':False,'PASS':False,'error':repr(exc),'automatic_retry':False})
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True)
    run(ap.parse_args().entry)
