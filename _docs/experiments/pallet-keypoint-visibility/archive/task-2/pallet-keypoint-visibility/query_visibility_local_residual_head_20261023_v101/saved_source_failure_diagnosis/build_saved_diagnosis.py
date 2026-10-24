"""Saved source logit decomposition and seven exact input crops; no model import."""
from pathlib import Path
import json,hashlib,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;R=V/'runs/C5_FROZEN_C1_CENTRAL_RGB_SOURCE_V1'
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def ck(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def stats(x):
 x=np.asarray(x,dtype=float)
 return dict(N=len(x),min=float(x.min())if len(x)else None,p25=float(np.quantile(x,.25))if len(x)else None,median=float(np.median(x))if len(x)else None,p75=float(np.quantile(x,.75))if len(x)else None,max=float(x.max())if len(x)else None,mean=float(x.mean())if len(x)else None)
pol=read(H/'SCOPE_AND_SELECTION_BEFORE_RGB96_READ.json');f=np.load(ck(pol['scores']),allow_pickle=False);a={k:f[k]for k in f.files};z=a['query_logit'];b=a['base_logit'];d=a['delta_logit'];p=a['prediction'];av=a['query_available'];y=a['scalar_target_D0_EorS1_Uminus1']
assert np.array_equal(z,(b+d).astype(np.float32))and len(z)==33984
summary={}
for sp in('TRAIN','VALIDATION'):
 pairs=a['old_accepted_triplets'];pairs=pairs[a['split'][pairs[:,0]]==sp];ii=pairs[:,1];wrong=b[ii]<0;fail=p[ii]!=1;gap=np.maximum(-b[ii],0);margin=z[ii]-z[pairs[:,2]]
 summary[sp]=dict(pairs=len(pairs),base_cover_already_hidden=int((~wrong).sum()),final_cover_hidden=int((p[ii]==1).sum()),cover_factual_U=int((~av[ii]).sum()),wrongbase_required_positive_delta=stats(gap[wrong]),cover_delta=stats(d[ii]),cover_final=stats(z[ii]),positive_delta=int((d[ii]>0).sum()),positive_delta_but_failed=int(((d[ii]>0)&fail).sum()),delta_fraction_required_for_wrongbase=stats(d[ii][wrong]/gap[wrong]),remaining_failed_sign_gap=stats(np.maximum(-z[ii][fail],0)),final_pair_margin=stats(margin),hinge_margin_ge2=int((margin>=2).sum()),hinge_ge2_but_cover_still_D=int(((margin>=2)&(p[ii]==0)).sum()),before_D=int((p[pairs[:,0]]==0).sum()),sham_D=int((p[pairs[:,2]]==0).sum()))
selfdiag={}
for sp in('TRAIN','VALIDATION'):
 m=(a['domain']=='V99_SYNTH')&(a['typed_state']=='S')&(a['split']==sp)
 def sr(mm):
  return dict(N=int(mm.sum()),hidden=int((mm&(p==1)).sum()),false_visible=int((mm&(p==0)).sum()),factual_U=int((mm&~av).sum()),base_hidden=int((mm&(b>=0)).sum()),base_logit=stats(b[mm]),delta=stats(d[mm]),final_logit=stats(z[mm]))
 selfdiag[sp]=dict(all=sr(m))
 for k in('appearance','aspect','azimuth','elevation','variant'):
  selfdiag[sp][k]={str(v):sr(m&(a[k]==v))for v in np.unique(a[k][m])}
 aux=np.r_[np.full(31680,-1),np.arange(2304)%8];selfdiag[sp]['AUX_logical_index']={str(k):sr(m&(aux==k))for k in np.unique(aux[m])}
freeze=read(ck(pol['freeze']));B=freeze['source_input_bindings'];om=read(ck(B['old_manifest']));nm=read(ck(B['new_manifest']));old=lines(ck(om['input_lineage']));new=lines(ck(nm['input_lineage']));base=lines(ck(B['old_base_lineage']));line=old+new
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
(H/'raw_RGB96').mkdir();outrows=[];figures=[]
def getcrop(i):
 r=line[i]
 if i<31680:
  ref=r['input_ref'];ref=base[r['slot_index']]['before_input_refs']['rgb96']if ref['kind']=='BASE_ROW'else dict(file=ref['file'],key='rgb96',index=ref['index'])
  with np.load(ck(ref['file']),allow_pickle=False)as f:rgb=np.array(f[ref['key']][ref['index']],copy=True)
  expected=r['tensor_value_sha256']['rgb96']
 else:
  ch=nm['chunks'][r['chunk_index']]
  with np.load(ck(ch['file']),allow_pickle=False)as f:rgb=np.array(f['rgb96'][r['index_in_chunk']],copy=True)
  expected=r['rgb96_sha256']
 assert rgb.dtype==np.uint8 and rgb.shape==(3,96,96)and hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()==expected
 return rgb.transpose(1,2,0).copy()
for ci,s in enumerate(pol['selections']):
 ids=s['indices'];n=len(ids);fig=Image.new('RGB',(400*n,590),'#f1f4f7');draw=ImageDraw.Draw(fig)
 draw.text((12,8),s['kind'],font=font,fill='#102c43');draw.text((12,34),'Actual unchanged RGB96 inputs; text outside image. Selected step1900.',font=small,fill='#102c43')
 for j,i in enumerate(ids):
  rgb=getcrop(i);raw=H/'raw_RGB96'/f'case_{ci:02d}_item_{j:02d}_occ_{i}.png';Image.fromarray(rgb).save(raw)
  fig.paste(Image.fromarray(rgb).resize((384,384),Image.Resampling.NEAREST),(j*400+8,66))
  r=line[i];label=str(a['typed_state'][i]);pred='U'if p[i]<0 else'D'if p[i]==0 else'hidden'
  info=f"index {i}  {r['variant']}  teacher {label} / pred {pred}\nbase {b[i]:+.4f}  delta {d[i]:+.4f}\nfinal {z[i]:+.4f} (hidden iff >=0)\nq {r['query_xy']}  L {r['L']:.4f}\n"
  info+='\n'.join(textwrap.wrap(str(a['occurrence_id'][i]),55))
  draw.multiline_text((j*400+10,460),info,font=small,fill='#102c43',spacing=4)
  outrows.append(dict(global_occurrence_index=i,occurrence_id=str(a['occurrence_id'][i]),typed_teacher=label,base_logit=float(b[i]),delta_logit=float(d[i]),final_logit=float(z[i]),prediction=pred,query_unchanged=r['query_xy'],L_unchanged=r['L'],exact_RGB96_png=bind(raw),model_pixels_unmarked=True))
 path=H/f'case_{ci:02d}.png';fig.save(path);figures.append(bind(path))
write(H/'NUMERICAL_AND_CROP_RESULTS.json',dict(completed=True,scope_and_selection=bind(H/'SCOPE_AND_SELECTION_BEFORE_RGB96_READ.json'),scores=pol['scores'],all33984_score_rows_retained=True,teacher_U=int((y<0).sum()),factual_U=int((~av).sum()),all_logits_exact_base_plus_delta=True,old_cover=summary,new_SELF_geometric=selfdiag,selected_crop_records=outrows,figures=figures,limits=['Outcome-selected first cases are diagnostic, not prevalence.','Frozen base confidence exceeding learned residual is an observed algebraic result, not a unique causal explanation.','Poor accepted TRAIN-cover correction shows an in-sample fitting limitation; it does not isolate optimization, capacity, regularization, loss balance or cue identifiability.','Large geometricS TRAIN/heldout-azimuth gap is consistent with view/appearance/association generalization weakness, not proof one factor caused it.','Expected geometric endpoint state is conditional AUX; it is not native8/human D/E/S gold.','RGB96 is unchanged; no q recenter, new data, head forward, fit, threshold, private/current or PnP.']))
print(json.dumps(dict(results=bind(H/'NUMERICAL_AND_CROP_RESULTS.json'),figures=figures,old_cover=summary,newSELF={sp:selfdiag[sp]['all']for sp in selfdiag})))
