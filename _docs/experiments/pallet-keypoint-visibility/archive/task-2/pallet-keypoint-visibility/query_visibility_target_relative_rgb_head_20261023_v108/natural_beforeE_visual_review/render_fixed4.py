import ast,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
P=ROOT.parents[1]
MATH=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
SCOPE=ROOT/'FIXED4_SCOPE_AND_REFERENCES_BEFORE_VIEW.json'
scope=json.loads(SCOPE.read_text()); rows=scope['fixed_rows']
node=next(n for n in ast.parse(MATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='crop')
ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(MATH),'exec'),ns);crop=ns['crop']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def a_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def checked(b):
 p=Path(b['path']);assert sha(p)==b['sha256'];return np.array(Image.open(p).convert('RGB'),copy=True)
linefile=P/'query_visibility_same_q_counterfactual_20261023_v95/source_pair_intake/SOURCE960_DRIVER_LINEAGE.jsonl'
lines={i:json.loads(l) for i,l in enumerate(linefile.open()) if i in {r['base_row_index'] for r in rows}}
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
f=ImageFont.truetype(fontpath,17);small=ImageFont.truetype(fontpath,15);big=ImageFont.truetype(fontpath,24)
canvas=Image.new('RGB',(1280,2850),'#f3f3f3');d=ImageDraw.Draw(canvas)
d.text((14,12),'Fixed natural external-cover source evidence: 3 public pairs / 4 unchanged queries',font=big,fill='black')
d.text((14,48),'No model scores. Yellow q; cyan original cue r / uncertainty. Native mapping UNKNOWN; not human gold.',font=f,fill='black')
pairs={};facts=[];beforec=[];afterc=[];validc=[]
for r in rows:
 bdoc=json.loads(Path(r['source_records']['before']['path']).read_text());adoc=json.loads(Path(r['source_records']['after']['path']).read_text())
 bp=bdoc['source_before_RGB'];ap=adoc['source_after_RGB'];key=(bp['path'],ap['path'])
 if key not in pairs:pairs[key]={'before':checked(bp),'after':checked(ap),'refs':[],'source_before':bp,'source_after':ap}
 pairs[key]['refs'].append(r)
for pi,pair in enumerate(pairs.values()):
 y=88+pi*510
 for col,stage in enumerate(['before','after']):
  x=col*640;d.text((x+8,y),f"Pair {pi+1} {stage.upper()}   {Path(pair['source_'+stage]['path']).parent.parent.name}/{Path(pair['source_'+stage]['path']).name}",font=f,fill='black')
  panel=Image.fromarray(pair[stage]);pd=ImageDraw.Draw(panel)
  for r in pair['refs']:
   q=r['original_q'];ref=r['source_reference_xy'];box=r['source_uncertainty_box'];L=r['original_L']
   pd.rectangle(box,outline='cyan',width=1);pd.ellipse([ref[0]-2,ref[1]-2,ref[0]+2,ref[1]+2],outline='cyan',width=1)
   pd.rectangle([q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2],outline='yellow',width=1)
   pd.line([q[0]-4,q[1],q[0]+4,q[1]],fill='yellow',width=1);pd.line([q[0],q[1]-4,q[0],q[1]+4],fill='yellow',width=1)
   pd.text((q[0]+5,q[1]+4),f"slot{r['logical_native_slot_only']}",font=small,fill='yellow')
  canvas.paste(panel,(x,y+28))
for i,r in enumerate(rows):
 bdoc=json.loads(Path(r['source_records']['before']['path']).read_text());adoc=json.loads(Path(r['source_records']['after']['path']).read_text())
 pair=pairs[(bdoc['source_before_RGB']['path'],adoc['source_after_RGB']['path'])]
 q=r['original_q'];L=r['original_L'];ref=r['source_reference_xy'];box=r['source_uncertainty_box']
 bc,bv=crop(pair['before'],q,L,96);ac,av=crop(pair['after'],q,L,96)
 line=lines[r['base_row_index']];assert line['query_xy']==q and line['L']==L
 refs={k:line['before_input_refs'][k] for k in ['rgb96','crop_valid96']}
 values={}
 for k,desc in refs.items():
  assert sha(desc['file']['path'])==desc['file']['sha256']
  with np.load(desc['file']['path'],allow_pickle=False) as z:values[k]=z[desc['key']][desc['index']]
 assert np.array_equal(values['rgb96'],ac);assert np.array_equal(values['crop_valid96'],av)
 y=1638+i*294
 d.text((12,y),f"Row {r['global_occurrence_index']} | {r['canonical_split']} | {r['original_input_id']} | slot{r['logical_native_slot_only']}",font=f,fill='black')
 d.text((12,y+22),f"q=({q[0]:.3f},{q[1]:.3f}), L={L:.3f}; r=({ref[0]},{ref[1]}), unc={box}",font=small,fill='black')
 beforepanel=Image.fromarray(bc.transpose(1,2,0));afterpanel=Image.fromarray(ac.transpose(1,2,0));marked=afterpanel.copy()
 for panel in [beforepanel,marked]:
  pd=ImageDraw.Draw(panel);rq=((ref[0]-q[0])/L*96+47.5,(ref[1]-q[1])/L*96+47.5)
  ub=[(box[0]-q[0])/L*96+47.5,(box[1]-q[1])/L*96+47.5,(box[2]-q[0])/L*96+47.5,(box[3]-q[1])/L*96+47.5]
  pd.rectangle(ub,outline='cyan',width=1);pd.line([rq[0]-2,rq[1],rq[0]+2,rq[1]],fill='cyan');pd.line([rq[0],rq[1]-2,rq[0],rq[1]+2],fill='cyan')
  pd.line([44,47.5,51,47.5],fill='yellow');pd.line([47.5,44,47.5,51],fill='yellow')
 for x,label,panel in [(12,'BEFORE: same q/L + original r/unc',beforepanel),(328,'AFTER: exact unmarked RGB96',afterpanel),(644,'AFTER: separate q/r/unc overlay',marked),(960,'AFTER: factual valid96',Image.fromarray((av[0]*255).astype(np.uint8)).convert('RGB'))]:
  d.text((x,y+48),label,font=small,fill='black');canvas.paste(panel.resize((224,224),Image.Resampling.NEAREST),(x,y+68))
 beforec.append(bc);afterc.append(ac);validc.append(av)
 facts.append({**r,'source_before_RGB':pair['source_before'],'source_after_RGB':pair['source_after'],'before_RGB96_sha256':a_sha(bc),'after_RGB96_sha256':a_sha(ac),'after_valid96_sha256':a_sha(av),'after_cache_RGB96_and_valid_bitexact':True,'cache_refs':refs,'cue_display':'Original fixed r/uncertainty; not relocalized to the after image.'})
canvas.save(ROOT/'CONTACT_FIXED3_PAIRS_ALL4_QUERIES.png')
np.savez_compressed(ROOT/'FIXED4_RGB96_AND_VALID.npz',before_rgb96=np.stack(beforec),after_rgb96=np.stack(afterc),after_valid96=np.stack(validc),query_xy=np.array([r['original_q'] for r in rows]),L=np.array([r['original_L'] for r in rows]))
meta={'completed':True,'scope':bind(SCOPE),'crop_operator':bind(MATH),'crop_AST_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'code':bind(__file__),'source_lineage':bind(linefile),'rows':facts,'pairs_count':len(pairs),'query_count':len(rows),'no_model_score_values_in_render':True,'P3_or_model_or_private_or_fit_or_PnP_reads':False,'outputs':[bind(ROOT/n) for n in ['CONTACT_FIXED3_PAIRS_ALL4_QUERIES.png','FIXED4_RGB96_AND_VALID.npz']]}
(ROOT/'RENDER_PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'figure':bind(ROOT/'CONTACT_FIXED3_PAIRS_ALL4_QUERIES.png'),'metadata':bind(ROOT/'RENDER_PROVENANCE.json'),'all4_after_RGB96_and_valid_cache_exact':True}))
