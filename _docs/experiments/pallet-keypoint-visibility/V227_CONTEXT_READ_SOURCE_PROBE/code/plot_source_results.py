from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
summary = json.loads((root/'source_pair_fit_v1/SUMMARY.json').read_text())
arms = [('CONTROL_FIXED_Q', 'Fixed query read'), ('TREAT_ATTENTION', 'Learned attention read')]
fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.2), gridspec_kw={'width_ratios':[1,1,1.3]})
for ax,(key,title) in zip(axes[:2],arms):
    m=summary['metrics'][key]['DEV'];cm=np.array(m['CM_D_E_by_D_E_U'])
    ax.imshow(cm, cmap='Blues',vmin=0,vmax=267)
    for (i,j),value in np.ndenumerate(cm):
        ax.text(j,i,str(value),ha='center',va='center',color='white' if value>133 else 'black',fontsize=15)
    ax.set_xticks(range(3),['D','E','U']);ax.set_yticks(range(2),['D (267)','E (36)'])
    ax.set_xlabel('Prediction');ax.set_ylabel('Existing weak role')
    ax.set_title(title+'\nBalanced recall %.1f%%'%(100*m['balanced_recall']))
ax=axes[2];x=np.arange(4);width=.36
for j,(key,title) in enumerate(arms):
    m=summary['metrics'][key]['DEV'];v=[m['D']['precision'],m['D']['recall'],m['E']['precision'],m['E']['recall']]
    bars=ax.bar(x+(j-.5)*width,100*np.array(v),width,label=title)
    for bar,value in zip(bars,v):ax.text(bar.get_x()+bar.get_width()/2,100*value+1.1,'%.1f'%(100*value),ha='center',fontsize=9)
ax.set_xticks(x,['D precision','D recall','E precision','E recall'],rotation=15)
ax.set_ylim(0,112);ax.set_ylabel('%');ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left',fontsize=8)
ax.set_title('Same 303 queries; threshold logit = 0')
fig.suptitle('SOURCE weak-role comparison: attention did not improve both classes',fontsize=14)
fig.text(.5,.015,'FIT D57/E8; reused SOURCE DEV D267/E36. Forced binary U=0; abstention is untested. Not current223 or physical point gold.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.91))
fig.savefig(root/'SOURCE_COMPARISON.png',dpi=180)
plt.close(fig)
