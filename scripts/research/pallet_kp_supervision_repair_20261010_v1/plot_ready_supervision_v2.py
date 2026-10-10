"""Source supervision preparation chart from recorded numeric summaries only."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
P=Path(__file__).parent;OUT=P/'figures'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preparation',type=Path,default=P/'full_source_preparation/FULL_SOURCE_PREPARATION.json');parser.add_argument('--recovery',type=Path,default=P/'depth_recovery_v3/DEPTH_RECOVERY_VALIDATION.json');parser.add_argument('--triangle-check',type=Path,default=P/'FRONT_TRIANGLE_DEPTH_VALIDATION.json');parser.add_argument('--output-dir',type=Path,default=OUT);args=parser.parse_args()
 output_dir=args.output_dir;output_dir.mkdir(exist_ok=True,parents=True);paths=[args.preparation,args.recovery,args.triangle_check,Path(__file__)]
 provenance=output_dir/'READY_SUPERVISION_FIGURE_PROTOCOL.json'
 protocol=dict(schema='ready_supervision_summary_figure_protocol_v1',inputs={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths},created_before_plot=True,
  figure='figures/01_ready_supervision.png',data='Recorded validation JSON only, no source RGB/GT/mesh/weights/features or numeric replay.',
  before='Conservative actual-wire preparation before missing front-depth evidence recovery; not the old original training labels.',
  after='Same prepared actual-wire positions with fixed finite-front NONE recovery. Different split query denominators are explicitly normalized.',
  new_rays=0,new_models=0,new_RGB=0,new_PnP=0,new_training=0)
 if provenance.exists():assert json.loads(provenance.read_text())==protocol,'Recorded figure binding differs; preserve old binding and select a new output directory.'
 else:provenance.write_text(json.dumps(protocol,indent=2)+'\n')
 prep=json.loads(paths[0].read_text());ready=json.loads(paths[1].read_text());triangle=json.loads(paths[2].read_text())
 plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
 fig=plt.figure(figsize=(14,8.5),constrained_layout=False);gs=fig.add_gridspec(2,3,left=.065,right=.965,top=.80,bottom=.16,hspace=.6,wspace=.32,height_ratios=[1.7,1])
 splits=['train','calibration','source_test'];names=['Train','Calibration','Frozen source test'];colors={'POSITIVE':'#289271','NONE':'#3677BD','IGNORE':'#C6CBD2'}
 legend=[]
 for j,(split,name) in enumerate(zip(splits,names)):
  ax=fig.add_subplot(gs[0,j]);n=prep['splits'][split];total=n*84
  for x,data in enumerate([prep['counts'][split],ready['target_counts'][split]]):
   bottom=0
   for state in ['POSITIVE','NONE','IGNORE']:
    count=data.get(state,0);fraction=count/total;bar=ax.bar(x,fraction,bottom=bottom,width=.64,color=colors[state],edgecolor='white',linewidth=1)
    if j==0 and x==1:legend.append(bar[0])
    if count:ax.text(x,bottom+fraction/2,f'{count:,}\n{fraction:.1%}',ha='center',va='center',fontsize=10,color='white' if state!='IGNORE' else '#26313D')
    else:ax.text(x,bottom+.035,'NONE = 0',ha='center',va='bottom',fontsize=9,color='#20548A')
    bottom+=fraction
  ax.set_ylim(0,1.015);ax.set_xticks([0,1],['Before depth\nrecovery','After fixed\nrecovery']);ax.set_title(f'{name}\n{n:,} families / {total:,} queries',fontsize=12,pad=12);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_yticks([0,.25,.5,.75,1]);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
  if j==0:ax.set_ylabel('Share of all existing source queries')
 ax=fig.add_subplot(gs[1,:]);ax.axis('off')
 ax.text(0,1.0,'Fixed recovery execution',fontsize=13,weight='bold',va='top')
 ax.text(0,.73,f"896 existing scenes  |  {ready['new_rays']:,} rays\n{ready['counts']['finite']:,} verified finite-front NONE\n{ready['counts']['infinite']} infinite results remain IGNORE",fontsize=12,va='top',linespacing=1.5)
 ax.text(.44,1.0,'Independent actual triangle check',fontsize=13,weight='bold',va='top')
 ax.text(.44,.73,f"{triangle['finite_front_queries']:,} / {triangle['finite_front_queries']:,} pass the original depth tolerance\nMinimum strict front margin: {triangle['minimum_strict_front_margin_m']*1e6:.2f} µm\nActual hit-face coordinates exported for review",fontsize=12,va='top',linespacing=1.5)
 fig.suptitle('Source supervision prepared from existing RGB and the actual mesh',fontsize=19,weight='bold',x=.515,y=.972)
 fig.text(.515,.926,'Supervision preparation only: no new learner, PnP, or real-pose accuracy result',ha='center',fontsize=12,color='#53606E')
 fig.legend(legend,['Valid physical correspondence','Finite front-surface no match','Ignored or uncertain'],loc='upper center',bbox_to_anchor=(.515,.905),ncol=3,frameon=False,fontsize=10)
 fig.text(.065,.077,'Both stages use the separate actual-wire preparation. Original training labels and features remain preserved.',fontsize=10,color='#53606E')
 fig.text(.065,.047,'Preparation: 74,406 closest queries + 13,757 rays. New RGB / detector / head / PnP / training: all zero.',fontsize=10,color='#53606E')
 output=output_dir/'01_ready_supervision.png';fig.savefig(output,dpi=180,facecolor='white');plt.close(fig)
 (output_dir/'READY_SUPERVISION_FIGURE_MANIFEST.json').write_text(json.dumps(dict(protocol,protocol_sha256=sha(provenance),output=dict(path='figures/'+output.name,sha256=sha(output),bytes=output.stat().st_size)),indent=2)+'\n')
 print(output.name,output.stat().st_size)
if __name__=='__main__':main()
