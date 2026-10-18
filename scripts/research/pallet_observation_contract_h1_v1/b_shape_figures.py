"""Plot saved four-condition primitive controls, without rerendering geometry."""
from __future__ import annotations
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.research.pallet_risk_information_v1.common import read,write,sha
from .b_mask_bounds import bounds_with_canvas


def run(control_dir,output,*,GO=False):
    if not GO:raise PermissionError('Root plot GO required')
    control_dir,output=map(Path,(control_dir,output));seal=read(control_dir/'PROCEDURAL_RENDER_SEAL.json')
    for name,digest in seal['artifacts'].items():
        if sha(control_dir/name)!=digest:raise ValueError('Saved procedural control changed')
    data=read(control_dir/'PROCEDURAL_SHAPE_CONTROL.json');output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(4,2,figsize=(13,12),gridspec_kw=dict(width_ratios=[1.25,1]))
    table=[]
    for row,(record,aa)in enumerate(zip(data['cases'],axes)):
        q=record['query'];xy=np.asarray(q['sample_xy']);edge=np.asarray(q['edge_xy']);labels=np.asarray(record['ownership_labels']);offsets=np.asarray(data['config']['query_offsets_raw_px'])
        with Image.open(control_dir/(record['case']+'_RGB.png'))as image:aa[0].imshow(np.asarray(image))
        aa[0].plot(xy[:,0],xy[:,1],color='#ffd400',lw=1.2,label='Fixed normal query')
        aa[0].plot(edge[:,0],edge[:,1],'--',color='#7030a0',lw=2,label='Registered cuboid edge')
        aa[0].scatter(*q['center_xy'],s=28,c='#00aeea',zorder=5)
        aa[0].set_xlim(0,data['config']['image_hw'][1]);aa[0].set_ylim(data['config']['image_hw'][0],0)
        aa[0].set_title(('Pallet-like openings'if row<2 else'Solid bounding cuboid')+(' + occluder'if record['occlusion']else' / no occluder'),fontsize=11)
        aa[0].set_xlabel('Raw canvas x (px)');aa[0].set_ylabel('Raw canvas y (px)')
        if row==0:aa[0].legend(loc='upper right',fontsize=7)
        for value,color,label in [(0,'#555555','Non-target'),(1,'#00864b','Target'),(-1,'#d38100','Ambiguous AA'),(-2,'#9b45a5','Outside')]:
            hit=labels==value
            if hit.any():aa[1].scatter(offsets[hit],labels[hit],s=9,c=color,label=label)
        bounds=bounds_with_canvas(labels);minimum,maximum=(bounds['whole_line'][k]for k in('minimum','maximum'))
        aa[1].set_title(f'Conditional transition count [{minimum}, {maximum}]',fontsize=11)
        aa[1].set_yticks([-1,0,1],['Unknown AA','Non-target','Target']);aa[1].set_ylim(-1.3,1.3)
        aa[1].set_xlim(offsets.min(),offsets.max());aa[1].set_xlabel('Ordered normal-query offset (px)');aa[1].grid(alpha=.2)
        aa[1].legend(loc='upper right',fontsize=7)
        table.append(dict(case=record['case'],shape=record['shape'],occlusion=record['occlusion'],samples=len(labels),
            target_samples=int(sum(labels==1)),non_target_samples=int(sum(labels==0)),ambiguous_samples=int(sum(labels==-1)),outside_samples=int(sum(labels==-2)),
            transition_minimum=minimum,transition_maximum=maximum,conditional_class=bounds['whole_line']['classification'],actual_scene_replay=0,new_pose_solver_calls=0))
    fig.suptitle('New procedural controls: not recovered dataset scenes\nSame camera/query/bounding dimensions; only primitive shape and occluder change',fontsize=13)
    fig.tight_layout(rect=(0,.04,1,.94),h_pad=2,w_pad=2)
    fig.text(.02,.012,'Our own boxes/colors only. Mixed supersamples stay unknown. Target membership is not edge identity.\nNo original RGB, new D1 samples, training, model forward or pose solving.',fontsize=9)
    fig.savefig(output/'procedural_shape_controls.png',dpi=150);plt.close(fig)
    with(output/'PROCEDURAL_TRANSITION_BOUNDS.csv').open('w',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write(output/'FIGURE_SEAL.json',dict(schema='PROCEDURAL_CONTROL_FIGURE_V1',status='COMPLETE',source_sha256=sha(__file__),original_control_seal_sha256=sha(control_dir/'PROCEDURAL_RENDER_SEAL.json'),
        artifacts={p.name:sha(p)for p in output.iterdir()if p.is_file()},rights='Original newly generated primitive boxes/colors; external assets0',
        original_dataset_RGB_published=False,original_scene_replay=0,new_geometry_render=0,actual_saved_software_renders=4,new_model_or_solver_calls=0,
        physical_truth_review=False,human_visual_review=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--render-GO',action='store_true');a=p.parse_args();run(a.control_dir,a.output,GO=a.render_GO)
