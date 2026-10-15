"""Postscore display requested by the user: exactly the saved 191cm case.

This source was saved after the initial display. No new pose, observation,
reference, ROI fit or model is computed. Reproduction refuses overwriting.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import render_clean_fresh_BI_full_result_process_v4_32_r2_v1 as R
from . import render_saved_full_result_roi_v4_32_r2_v1 as V


def render(output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Preserve the displayed image')
    folder=R.OUTPUT/'case1'
    data=json.loads((folder/'PROCESS_DATA.json').read_text())
    photo=np.asarray(Image.open(folder/'source_rgb.png').convert('RGB'))
    x0,x1,y0,y1=V.observed_model_ROI(data,photo.shape[:2])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,title in zip(axes,['PREDICTION — error 191.31 cm / 82.32 deg','FIXED EVALUATION REFERENCE']):
        ax.imshow(photo);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.axis('off');ax.set_title(title,fontsize=14)
    R.DRAW._wire(axes[0],np.asarray(data['final_projection_xy']['CLEAN_FRESH_BI_SOURCE_CONTAINMENT']),
        color='#ff8a32',width=2.5,label='191.31cm prediction')
    R.DRAW._wire(axes[1],np.asarray(data['fixed_reference_projection_xy']),
        color='#65ff65',width=2.5,label='Evaluation reference')
    fig.suptitle('wood_night_01:031426 — actual same image',fontsize=15)
    fig.savefig(output,dpi=150);plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    render(parser.parse_args().output)
