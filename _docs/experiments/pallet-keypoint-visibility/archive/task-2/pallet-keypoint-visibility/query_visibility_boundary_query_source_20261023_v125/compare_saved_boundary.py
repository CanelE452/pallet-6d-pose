"""Saved source result arithmetic and scientific plot, no model/fit/current."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

V=Path(__file__).resolve().parent
R=V/'runs/C17_QUERY_BOUNDARY_AUGMENTED_SPATIAL_QUERY'
OUT=V/'saved_boundary_diagnosis'
def bind(p):
    return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
r=[json.loads(s)for s in(V/'actual_corpus/OCCURRENCES_INPUT_ONLY.jsonl').read_text().splitlines()]
g=[json.loads(s)for s in(V/'actual_corpus/LABELS_GT_ONLY.jsonl').read_text().splitlines()]
y=np.array([z['candidate_aux_hidden_label_GT_only']for z in g])
inside=np.array([z['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is True for z in g])
var=np.array([z['variant']for z in r]);split=np.array([z['split']for z in r]);source=np.array([z['source']for z in r])
files={'C16':V/'frozen_C16_source_baseline/RAW_C16_PREDICTIONS_INPUT_ONLY.npz',
       'C17':R/'SELECTED_ALL55900_SOURCE_SCORES.npz'}
results={}
for name,path in files.items():
    with np.load(path,allow_pickle=False)as f:
        p=f['prediction'].copy();a=f['query_available'].copy()
    if name=='C17':p=p[51004:];a=a[51004:]
    assert p.shape==a.shape==y.shape==(4896,)
    groups={}
    for group,m in [('VALIDATION_COMBINED',split=='VALIDATION'),('LOCO_VAL',(split=='VALIDATION')&(source=='LOCO')),('TOMIE_DEV',source=='TOMIE'),('TRAIN',split=='TRAIN')]:
        blocks={}
        for key,k,truth in [('E_qOUT',(y==1)&~inside,1),('E_qIN',(y==1)&inside,1),('SHAM_D_qIN',(y==0)&(var=='CUE_PRESERVING_SHAM')&inside,0),('SHAM_D_qOUT',(y==0)&(var=='CUE_PRESERVING_SHAM')&~inside,0),('ALL_D',y==0,0),('ALL_E',y==1,1)]:
            mask=m&k;valid=mask&a;correct=int((valid&(p==truth)).sum());den=int(valid.sum())
            blocks[key]=dict(all_planned=int(mask.sum()),available=den,input_U=int((mask&~a).sum()),correct=correct,recall_available=correct/den if den else None)
        groups[group]=blocks
    results[name]=groups
assert results['C16']['VALIDATION_COMBINED']['E_qOUT']['correct']==142
assert results['C17']['VALIDATION_COMBINED']['E_qOUT']['correct']==210
result=dict(completed=True,scope='Source-only correlated expectedweakcue control; combined validation includes LOCO and one TOMIE development episode, not current real/nativegold',models=results,bindings={k:bind(v)for k,v in files.items()},model_current_V112_fit_calls=0)
(OUT/'C16_C17_BOUNDARY_COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(13.5,6.2),dpi=150)
keys=['E_qOUT','SHAM_D_qIN','ALL_D','ALL_E']
labels=['가림 E · q는 가림 물체 밖','보임 SHAM D · q는 물체 안','전체 보임 D','전체 외부 가림 E']
x=np.arange(4);width=.34
for off,(name,color,label)in zip([-.17,.17],[('C16','#879eae','기존 C16'),('C17','#d06a38','경계 추가 C17')]):
    vals=[100*results[name]['VALIDATION_COMBINED'][k]['recall_available']for k in keys]
    bars=ax.bar(x+off,vals,width,color=color,label=label)
    for b,k,val in zip(bars,keys,vals):
        d=results[name]['VALIDATION_COMBINED'][k]
        ax.text(b.get_x()+b.get_width()/2,val+1.8,f'{val:.1f}%\n{d["correct"]}/{d["available"]}',ha='center',va='bottom',fontsize=11)
ax.set_xticks(x,labels,fontproperties=font,fontsize=11)
ax.set_ylim(0,104);ax.set_ylabel('정답률 (%)',fontproperties=font,fontsize=13)
ax.set_title('공개 검증: 경계 조건은 개선했지만, 전체 보임·가림 동시 개선은 실패',fontproperties=font,fontsize=17,pad=18)
ax.legend(prop=font,loc='upper right');ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
fig.text(.5,.015,'LOCO VAL + TOMIE 개발 검증 · TRAIN 제외 · 같은 weak cue/합성물/문턱 유지 · 현재223 실사 성능이 아님',ha='center',fontproperties=font,fontsize=11)
fig.tight_layout(rect=(0,.055,1,1));fig.savefig(OUT/'C16_C17_PUBLIC_VALIDATION_COMPARISON.png');plt.close(fig)
print(json.dumps({'metrics':str(OUT/'C16_C17_BOUNDARY_COMPARISON.json'),'plot':str(OUT/'C16_C17_PUBLIC_VALIDATION_COMPARISON.png')},ensure_ascii=False))
