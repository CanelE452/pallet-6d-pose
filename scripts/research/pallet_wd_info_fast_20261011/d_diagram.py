"""Export the actually implemented matched local residual architecture."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
    target=Path(args.output);assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,7));ax.set(xlim=(0,14),ylim=(0,7));ax.axis("off")
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08",fc=color,ec="#344455",lw=1.4))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=10)
    def arrow(start,end):
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",lw=1.5,color="#344455"))
    box(.3,4.9,2.6,1.1,"Cached frozen R0 features\nP3 64 / P4 128 channels\nPredicted box + base corners","#dcecf6")
    box(3.3,4.7,2.7,1.5,"Frozen image-only P\n18,962 parameters\nOriginal candidate scores\n221 displacements + null","#dcecf6")
    arrow((2.9,5.45),(3.3,5.45))
    box(6.5,4.6,3.1,1.7,"Read-only scorer pre-hooks\nRegular: pooled 48 + context 13\n+ displacement 2 + coverage 1\nNull: mean pooled 48 + context 13\n+ zero padding 3 + null flag","#e8f2e1")
    arrow((6,5.45),(6.5,5.45))
    box(10.3,4.8,3.2,1.3,"Shared feature width 65\n+ metadata width 5 = 70\nSame candidate/null semantics","#e8f2e1")
    arrow((9.6,5.45),(10.3,5.45))
    box(3.3,2.4,3.6,1.2,"D_CONTROL: metadata = zeros\n70 -> 32 (SiLU) -> 1\n2,305 trainable parameters","#fff0d8")
    box(8.0,2.4,3.8,1.2,"D_JOINT_DIM: canonical metadata\n70 -> 32 (SiLU) -> 1\n2,305 trainable parameters","#f4e0eb")
    arrow((11.4,4.8),(10.1,3.6));arrow((11.4,4.8),(5.5,3.6))
    box(4.0,.5,6.4,1.3,"Original logits + zero-initialized residual scores\n222-candidate Softmax / original decode, lambda and cap\nStage 1 -> fixed cornerSubPix -> original Base cap\nUnchanged prediction-only W/D selector + SQPnP / RefineLM","#edf0f3")
    arrow((5.2,2.4),(5.5,1.8));arrow((9.9,2.4),(9,1.8))
    ax.text(7,6.7,"Matched local conditioning pilot: implemented D arms",ha="center",fontsize=15,weight="bold")
    ax.text(7,.12,"Seed 1; matched initialization, batch order and 1,000 updates per arm; frozen P; no real tuning",ha="center",fontsize=10)
    fig.savefig(target,dpi=160,bbox_inches="tight");plt.close(fig)


if __name__=="__main__":main()
