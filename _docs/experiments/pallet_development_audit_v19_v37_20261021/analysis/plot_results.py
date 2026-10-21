"""Render only author-derived scalar comparisons; no source image assets."""
from pathlib import Path
import argparse,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):
 with p.open(newline='')as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'figures');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'svg.hashsalt':'pallet-development-audit-v19-v37','font.size':9})
 table=read(ROOT/'summary.csv');get=lambda v,m:next(r for r in table if r['version']==str(v)and r['population']=='SYNTHETIC_VAL'and r['method']==m)
 arms=[('v5',get(19,'BASELINE')),('v24',get(24,'after')),('v26',get(26,'INFLUENCE')),('v28',get(28,'FUSION')),('v31',get(31,'ROI_UNION')),('v32',get(32,'ROI_EVIDENCE')),('v33',get(33,'JOINT_EVIDENCE')),('v35',get(35,'SCALE_POLICY'))]
 fig,axes=plt.subplots(2,1,figsize=(7,4.6),sharex=True)
 for ax,metric,unit in zip(axes,['T_mean','R_mean'],['Translation mean (cm)','Rotation mean (deg)']):
  ax.plot(range(len(arms)),[float(r[metric])for _,r in arms],marker='o',color='.15',ls='--');ax.set_ylabel(unit);ax.grid(axis='y',alpha=.25)
 axes[1].set_xticks(range(len(arms)),[n for n,_ in arms]);fig.suptitle('Repeated synthetic VAL 235; all shown outputs available 235\nNo variant passed its complete recorded adoption gate',fontsize=10);fig.tight_layout();fig.savefig(a.output/'synthetic_means.svg',metadata={'Date':None,'Creator':'Author-derived numerical research audit'});plt.close(fig)
 cv=read(ROOT/'results/bank_coverage.csv');fig,axes=plt.subplots(1,2,figsize=(7.4,3.8))
 for ax,split,n in zip(axes,['train','validation'],[768,235]):
  before=lambda v:sum(r['split']==split and r['version']==str(v)and r['before_bank_good']=='True'for r in cv)
  after=lambda v:sum(r['split']==split and r['version']==str(v)and r['after_bank_good']=='True'for r in cv)
  values=[before(24),after(24),after(31),after(37)];bars=ax.bar(range(4),values,color=['.85','.65','.45','.25'],edgecolor='black',hatch=['','//','xx','..']);ax.set_xticks(range(4),['Original','JK1','JK1+ROI','JK1+JK2'],rotation=22);ax.set_ylim(0,n);ax.set_title(f'{split.upper()} n={n}');ax.set_ylabel('Frames with >=1 successful candidate')
  for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+n*.012,str(v),ha='center',fontsize=9)
 fig.suptitle('Eligible-bank coverage diagnostic; NOT achieved method success\nv37: features/coverage only, no head fit or selection evaluation',fontsize=10);fig.tight_layout();fig.savefig(a.output/'bank_coverage.svg',metadata={'Date':None,'Creator':'Author-derived numerical research audit'});plt.close(fig)
 common=next(r for r in read(ROOT/'results/paired_comparisons.csv')if r['version']=='30');assert common['common_available_n']=='232';fig,axes=plt.subplots(1,2,figsize=(6.8,3.5))
 for ax,tag,unit in zip(axes,['T','R'],['cm','deg']):
  vals=[float(common[s+'_common_'+tag+'_mean'])for s in ['before','after']];bars=ax.bar([0,1],vals,color=['.8','.35'],edgecolor='black',hatch=['//','..']);ax.set_xticks([0,1],['N0','ROI']);ax.set_ylabel(f'{tag} mean ({unit})');ax.set_ylim(0,max(vals)*1.2)
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.025,f'{v:.3f}',ha='center')
 fig.suptitle('v30 paired common-output 232; not full-population performance\nFull 235 success N0=103, ROI=99; ROI no-pose=3',fontsize=10);fig.tight_layout();fig.savefig(a.output/'v30_common232.svg',metadata={'Date':None,'Creator':'Author-derived numerical research audit'});plt.close(fig)
 for file in a.output.glob('*.svg'):file.write_text('\n'.join(line.rstrip()for line in file.read_text().splitlines())+'\n')
 print('Rendered3 numerical SVG figures without source RGB.')
if __name__=='__main__':main()
