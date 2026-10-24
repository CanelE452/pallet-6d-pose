"""Summarize fixed saved benchmark outputs; no model/input/cache reads."""
from pathlib import Path
import csv, json, hashlib
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V=Path(__file__).absolute().parent
E=V/'saved_real_head_evaluation_v1'
MODELS=['v61','legacy','C55','C63','C64','C65']
GROUPS=['ALL','WOOD_PRIMARY105','SQUARE118']
def bind(p):
 b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest())
def readcsv(p):
 with p.open()as f:return list(csv.DictReader(f))
def newjson(p,x):
 assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
 assert not p.exists()
 with p.open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
done=json.loads((E/'COMPLETE.json').read_text());assert done['completed']and done['PASS']and done['old_fourteen_primary_metrics_exact']
rows=[json.loads(l)for l in (E/'current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl').read_text().splitlines()]
primary=[r for r in rows if r['paired_same_query_RGB_D_vs_E']=='True'and r['human_state']in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED']]
assert len(primary)==1237
out=[];LE3=[];selfrows=[];changes=[]
for group in GROUPS:
 rr=[r for r in primary if group=='ALL'or r['group']==group]
 for model in MODELS:
  cm=[[sum(r['human_state']==truth and r[model+'_D_E_U']==pred for r in rr)for pred in ['D','E','U']]for truth in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED']]
  d,e=map(sum,cm);dp=cm[0][0]+cm[1][0];ep=cm[0][1]+cm[1][1]
  out.append(dict(group=group,model=model,N=len(rr),D=d,E=e,D_D=cm[0][0],D_E=cm[0][1],D_U=cm[0][2],E_D=cm[1][0],E_E=cm[1][1],E_U=cm[1][2],D_precision=cm[0][0]/dp if dp else None,D_recall=cm[0][0]/d if d else None,E_precision=cm[1][1]/ep if ep else None,E_recall=cm[1][1]/e if e else None,U_rate=(cm[0][2]+cm[1][2])/len(rr),correct=cm[0][0]+cm[1][1]))
  near=[r for r in rows if r['human_state']=='DIRECT_VISIBLE'and(group=='ALL'or r['group']==group)and r['initial_reference_error_px']not in ['',None]and float(r['initial_reference_error_px'])<=3]
  cc=Counter(r[model+'_D_E_U']for r in near)
  LE3.append(dict(group=group,model=model,human_D_reference_LE3=len(near),D=cc['D'],E=cc['E'],U=cc['U'],D_recall=cc['D']/len(near)))
  ss=[r for r in rows if r['human_state']=='SELF_OCCLUDED'and(group=='ALL'or r['group']==group)];cc=Counter(r[model+'_D_E_U']for r in ss)
  selfrows.append(dict(group=group,model=model,all_human_SELF=len(ss),D=cc['D'],binary_E_response=cc['E'],unresolved_or_task_external_geometry_S=cc['U'],not_learned_SELF_class=True))
 for baseline in ['C64','C63','C55','legacy','v61']:
  for truth,state in [('D','DIRECT_VISIBLE'),('E','EXTERNAL_OCCLUDED')]:
   xx=[r for r in rr if r['human_state']==state]
   changes.append(dict(group=group,baseline=baseline,new_model='C65',truth=truth,N=len(xx),both_correct=sum(r[baseline+'_D_E_U']==truth and r['C65_D_E_U']==truth for r in xx),recovered_including_baseline_U=sum(r[baseline+'_D_E_U']!=truth and r['C65_D_E_U']==truth for r in xx),damaged=sum(r[baseline+'_D_E_U']==truth and r['C65_D_E_U']!=truth for r in xx),both_not_correct=sum(r[baseline+'_D_E_U']!=truth and r['C65_D_E_U']!=truth for r in xx),baseline_U=sum(r[baseline+'_D_E_U']=='U'for r in xx)))
csvout(V/'PRIMARY6_SUMMARY.csv',[r for r in out if r['group']=='ALL'])
csvout(V/'GROUP_PRIMARY6_SUMMARY.csv',[r for r in out if r['group']!='ALL'])
csvout(V/'LE3_VISIBLE6_SUMMARY.csv',LE3)
csvout(V/'SELF_RESPONSE6_SUMMARY.csv',selfrows)
csvout(V/'PRIMARY6_PAIRED_CHANGES.csv',changes)
totals={r['model']:r for r in out if r['group']=='ALL'}
assert [totals['C65'][k]for k in ['D_D','D_E','D_U','E_D','E_E','E_U']]==[901,246,0,76,14,0]
fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,group in zip(axes,GROUPS):
 vals=[r for r in out if r['group']==group];ix=list(range(6))
 ax.bar([x-.17 for x in ix],[100*r['D_recall']for r in vals],width=.32,label='Visible D recall',color='#2474b5')
 ax.bar([x+.17 for x in ix],[100*r['E_recall']for r in vals],width=.32,label='External E recall',color='#d55e00')
 ax.set_xticks(ix,MODELS,rotation=30);ax.set_ylim(0,108);ax.set_ylabel('%');ax.set_title(group+'\nD='+str(vals[0]['D'])+' / E='+str(vals[0]['E']));ax.grid(axis='y',alpha=.2)
axes[0].legend(loc='upper right',fontsize=8)
fig.suptitle('Same original query/crop primary cohort: C65 preserves more visible D, misses more external E\nFixed SOURCE models; no threshold/model selection; repeated evaluation, not fresh blind')
fig.tight_layout(rect=(0,0,1,.86));fig.savefig(V/'PRIMARY6_D_E_RECALL.png',dpi=150);plt.close(fig)
fig=plt.figure(figsize=(13,5.6))
axes=[fig.add_axes([.05+i*.32,.15,.27,.64])for i in range(3)]
for ax,model in zip(axes,['C63','C64','C65']):
 r=totals[model];cm=[[r[k]for k in ['D_D','D_E']],[r[k]for k in ['E_D','E_E']]]
 ax.imshow(cm,cmap='Blues',vmin=0,vmax=1000)
 for yy in range(2):
  for xx in range(2):ax.text(xx,yy,str(cm[yy][xx]),ha='center',va='center',fontsize=20,color='white'if cm[yy][xx]>500 else'black')
 ax.set_xticks([0,1],['pred D','pred E']);ax.set_yticks([0,1],['D','E']);ax.set_title(model+' (primary U=0)')
fig.suptitle('Rows: human D1147 / external E90 | same1237 queries\nBinary E response does not identify self vs external cause')
fig.savefig(V/'C63_C64_C65_CONFUSIONS.png',dpi=150);plt.close(fig)
summary=dict(completed=True,PASS=True,phase='V220',source_training_updates_this_phase=0,primary_N=1237,D=1147,E=90,all_registered_slots=1784,original_input_U=45,models=MODELS,metrics=out,LE3_visible=LE3,SELF_response=selfrows,paired_changes=changes,evaluation_complete=bind(E/'COMPLETE.json'),prediction_seal=bind(V/'frozen_C64_C65_real_predictions_v1/PREDICTION_SEAL.json'),selected_model=None,adopted=False,fully_solved=False,conclusion='C65 vs C63: D correct+58, E correct-5. Better visible preservation, external recall still poor; joint task unsolved.',source_DEV_not_current_evaluation='V219 SOURCE weak DEV Drec5/9->8/9 and Erec2/3 unchanged; GT-derived weak training q differs from original initializer q here.')
newjson(V/'ROOT_NUMERIC_SUMMARY.json',summary)
print(json.dumps(dict(completed=True,PASS=True,summary=bind(V/'ROOT_NUMERIC_SUMMARY.json'),totals=totals),ensure_ascii=False))
