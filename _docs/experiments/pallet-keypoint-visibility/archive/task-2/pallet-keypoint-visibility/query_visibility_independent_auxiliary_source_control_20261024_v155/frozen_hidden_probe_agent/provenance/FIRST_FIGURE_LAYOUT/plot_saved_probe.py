"""Scientific figure from the completed independent scalar review only."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
review=H/'SAVED_PROBE_REVIEW.json';r=json.loads(review.read_text());assert r['completed']and r['PASS']
stages=('C32_INITIAL','C35_FINAL','C36_FINAL');labels=('C32 initial','C35 visibility only','C36 + auxiliaries')
groups=('ALL28976::ALL_STATES::INPUTVALID','MR1840::SOURCE_VALIDATION::KNOWN_D::INPUTVALID','MR1840::SOURCE_VALIDATION::KNOWN_E::INPUTVALID')
names=('All source valid\nN=28,441','MR held D\nN=93','MR held E\nN=52');colors=('#747b85','#2c75b9','#df8330')
fig,axs=plt.subplots(1,3,figsize=(14.2,4.65),gridspec_kw={'width_ratios':[1.12,1.12,1.]})
x=np.arange(3);width=.24
for ax,key,title,ylim in((axs[0],'entropy_effective_rank','Hidden64 entropy effective rank',(0,2.2)),(axs[1],'top_eigenvalue_fractions','Fraction of variance in first component',(0,1.08))):
 for i,(stage,label,col)in enumerate(zip(stages,labels,colors)):
  vals=[r['results'][stage]['selected_groups'][g][key]for g in groups]
  if key=='top_eigenvalue_fractions':vals=[v['1']for v in vals]
  bars=ax.bar(x+(i-1)*width,vals,width,color=col,label=label)
  for bar,value in zip(bars,vals):ax.text(bar.get_x()+width/2,bar.get_height()+.025 if key=='entropy_effective_rank'else bar.get_height()+.013,f'{value:.2f}'if key=='entropy_effective_rank'else f'{value:.1%}',ha='center',va='bottom',fontsize=8)
 ax.set_xticks(x,names,fontsize=9);ax.set_ylim(*ylim);ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
 ax.spines[['top','right']].set_visible(False)
axs[0].legend(fontsize=8,loc='upper left',frameon=False)
cos=np.array(r['results']['C36_FINAL']['readout_cosine']);im=axs[2].imshow(cos,vmin=-1,vmax=1,cmap='RdBu_r')
ticks=('E-positive\nvisibility','Correct\nentity','Good synthetic\npoint quality')
axs[2].set_xticks(range(3),ticks,fontsize=8);axs[2].set_yticks(range(3),ticks,fontsize=8);axs[2].set_title('C36 readout weight cosine',fontsize=11)
for i in range(3):
 for j in range(3):axs[2].text(j,i,f'{cos[i,j]:+.3f}',ha='center',va='center',color='white'if abs(cos[i,j])>.6 else'black',fontsize=10)
fig.colorbar(im,ax=axs[2],fraction=.046,pad=.04,ticks=[-1,0,1])
fig.suptitle('Frozen source snapshots: concentrated hidden variance and coupled readouts',fontsize=13,y=.98)
fig.text(.015,.025,'Empirical source distributions, not intrinsic capacity or causal proof. Broad-group numerical rank = 59; C32/C35 auxiliary weights = 0 (cosines undefined).',fontsize=8,color='#444444')
fig.subplots_adjust(left=.06,right=.97,top=.84,bottom=.20,wspace=.38)
path=H/'HIDDEN_RANK_AND_READOUT.png';assert not path.exists();fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
receipt=dict(completed=True,PASS=True,code=bind(__file__),independent_saved_review=bind(review),figure=bind(path),new_models_or_features_or_weights_read=0)
with(H/'FIGURE_RECEIPT.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
