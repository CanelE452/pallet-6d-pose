"""Add readable actual-object crops to the preserved v17 process report."""
from pathlib import Path
import argparse
import numpy as np

from . import build_native_edge_actual_report_v4_17 as BASE

OUTPUT=BASE.OUTPUT.parent/'native_edge_actual_process_report_v4_17b'


def main(output=OUTPUT):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    original=Figure.savefig

    def save_with_actual_crops(fig,*args,**kwargs):
        positions=[]
        for ax in fig.axes[1:5]:
            for line in ax.lines:
                xy=np.asarray(line.get_xydata())
                if xy.ndim==2 and xy.shape[1]==2 and np.isfinite(xy).all():positions.append(xy)
            for patch in ax.patches:
                if hasattr(patch,'center')and hasattr(patch,'width'):
                    c=np.asarray(patch.center);positions.append(np.array([c-[patch.width,patch.height]/np.array([2,2]),c+[patch.width,patch.height]/np.array([2,2])]))
        if positions:
            xy=np.vstack(positions);lo=xy.min(axis=0)-20;hi=xy.max(axis=0)+20
            image=np.asarray(fig.axes[0].images[0].get_array());h,w=image.shape[:2]
            for ax in fig.axes[1:5]:
                ax.set_xlim(max(0,lo[0]),min(w,hi[0]));ax.set_ylim(min(h,hi[1]),max(0,lo[1]))
        return original(fig,*args,**kwargs)

    Figure.savefig=save_with_actual_crops
    try:BASE.main(output)
    finally:Figure.savefig=original
    # The base report records its rendering primitive; bind this crop wrapper
    # separately without changing that preserved report source or predictions.
    BASE.C.write(Path(output)/'CROP_RECEIPT.json',dict(wrapper=BASE.E._binding(Path(__file__)),
        renderer=BASE.E._binding(Path(BASE.__file__)),actual_full_RGB_first_panel=True,
        other_panels_actual_object_crops=True,prediction_or_metric_changes=False))
    BASE.C.write(Path(output)/'CROP_SEAL.json',dict(artifacts=[BASE.E._binding(p)for p in sorted(Path(output).iterdir())if p.is_file()and p.name!='CROP_SEAL.json']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    a=parser.parse_args();main(a.output)
