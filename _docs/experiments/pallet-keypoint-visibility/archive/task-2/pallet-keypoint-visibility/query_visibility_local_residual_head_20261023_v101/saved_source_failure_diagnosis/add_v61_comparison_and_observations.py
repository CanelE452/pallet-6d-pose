from pathlib import Path
import json,hashlib,textwrap,numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent;R=V/'runs/C5_FROZEN_C1_CENTRAL_RGB_SOURCE_V1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p))
def write(p,r):
 with p.open('x')as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
r=read(H/'NUMERICAL_AND_CROP_RESULTS.json');pol=read(H/'SCOPE_AND_SELECTION_BEFORE_RGB96_READ.json');f=np.load(R/'SELECTED_ALL33984_SOURCE_SCORES.npz',allow_pickle=False);c={k:f[k]for k in f.files};vfile=V/'v61_source_baseline/outputs/SOURCE_SCORES.npz';f=np.load(vfile,allow_pickle=False);b={k:f[k]for k in f.files}
assert np.array_equal(c['query_xy_raw'],b['query_xy'],equal_nan=True)and np.array_equal(c['L_raw'],b['L'],equal_nan=True)and np.array_equal(c['query_available'],b['available'])
tab={}
for sp in('TRAIN','VALIDATION'):
 pp=c['old_accepted_triplets'];pp=pp[c['split'][pp[:,0]]==sp];ii=pp[:,1];vpred=b['decoded_aux_prediction'];cpred=c['prediction']
 tab[sp]=dict(N=len(pp),v61=dict(before_D=int((vpred[pp[:,0]]==0).sum()),cover_hidden=int((vpred[ii]==1).sum()),sham_D=int((vpred[pp[:,2]]==0).sum()),all3=int(((vpred[pp]==[0,1,0]).all(1)).sum())),frozen_C1_base_cover_hidden=int((c['base_logit'][ii]>=0).sum()),selected_C5=dict(before_D=int((cpred[pp[:,0]]==0).sum()),cover_hidden=int((cpred[ii]==1).sum()),sham_D=int((cpred[pp[:,2]]==0).sum()),all3=int(((cpred[pp]==[0,1,0]).all(1)).sum())),v61_cover_hidden_while_C1_base_D=int(((vpred[ii]==1)&(c['base_logit'][ii]<0)).sum()))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13);figs=[]
byi={x['global_occurrence_index']:x for x in r['selected_crop_records']}
for ci,s in enumerate(pol['selections']):
 n=len(s['indices']);fig=Image.new('RGB',(max(800,n*400),735),'#f1f4f7');draw=ImageDraw.Draw(fig)
 title=s['kind']+(' [HISTORICAL SELF ONLY]'if ci==2 else' [EXTERNAL PRIORITY]');draw.text((10,8),title,font=font,fill='#102c43');draw.text((10,35),'Exact RGB96; unchanged q/L. C1/C5: hidden z>=0. v61: visible z>=0 means D.',font=small,fill='#102c43')
 for j,i in enumerate(s['indices']):
  rec=byi[i];rgb=Image.open(rec['exact_RGB96_png']['path']).convert('RGB');fig.paste(rgb.resize((384,384),Image.Resampling.NEAREST),(j*400+8,66))
  pd='U'if c['prediction'][i]<0 else'D'if c['prediction'][i]==0 else'hidden';vp='U'if b['decoded_aux_prediction'][i]<0 else'D'if b['decoded_aux_prediction'][i]==0 else'hidden'
  info=f"index {i} / GT AUX {c['typed_state'][i]} / C5 {pd}\nC1 hidden base {c['base_logit'][i]:+.4f}\nresidual {c['delta_logit'][i]:+.4f} / final {c['query_logit'][i]:+.4f}\nv61 VISIBLE z {b['raw_visible_logit'][i]:+.4f} -> {vp}\nq ({c['query_xy_raw'][i,0]:.4f},{c['query_xy_raw'][i,1]:.4f})\nL {c['L_raw'][i]:.4f} unchanged\n"+'\n'.join(textwrap.wrap(str(c['occurrence_id'][i]),48))
  draw.multiline_text((j*400+10,463),info,font=small,fill='#102c43',spacing=4)
 if ci==2:draw.multiline_text((410,100),'Historical mixed-target C5 only.\nUser steering: geometry handles self-occlusion;\nfuture learned objective is visible vs external.\n\nAUX expected-endpoint association is geometric.\nThis wood/background crop alone does not\ncertify which hidden endpoint is requested.\nNative8 mapping and human gold UNKNOWN.',font=small,fill='#102c43',spacing=8)
 path=H/f'case_{ci:02d}_v61_comparison.png';fig.save(path);figs.append(bind(path))
write(H/'V61_COMPARISON_AND_FIGURES.json',dict(completed=True,code=bind(Path(__file__)),fixed_cases_policy=bind(H/'SCOPE_AND_SELECTION_BEFORE_RGB96_READ.json'),numeric_original=bind(H/'NUMERICAL_AND_CROP_RESULTS.json'),v61_scores=bind(vfile),all33984_q_L_factual_available_exact_between_models=True,old_external_triplets=tab,figures=figs,selection_changed=False,limits=['v61 originally models query-centre target-mask presence; conditional physical-cue labels here differ, so higher cover recall alone is not a solution.','v61 before/sham D and matchedtriples are reported alongside cover to avoid a one-sided comparison.','No learned cause-specific S claim. Historical C5 mixed target unchanged; future self removal is a separate geometry route per user steering.','No pixel/model/fit/threshold/private/PnP changes.']))
print(json.dumps(dict(addendum=bind(H/'V61_COMPARISON_AND_FIGURES.json'),figures=figs,tables=tab)))
