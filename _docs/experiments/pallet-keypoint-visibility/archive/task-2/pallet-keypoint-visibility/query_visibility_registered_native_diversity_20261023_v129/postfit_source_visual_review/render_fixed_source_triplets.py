"""Saved-only fixed6 source triplets, frozen feature summaries; no model calls."""
from pathlib import Path
import json,hashlib,collections,datetime
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent
RUN=V/'runs/C20_C21_FIXED1000_SOURCE_CONTROL'
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def check(b):assert bind(b['path'])==b
def font(n):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def inside(box,lo,hi):return bool(np.all(np.array(box)[:2]>=lo)and np.all(np.array(box)[2:]<=hi))
def overlap(box,lo,hi):
 b=np.array(box);area=np.prod(b[2:]-b[:2]);return float(np.prod(np.maximum(0,np.minimum(b[2:],hi)-np.maximum(b[:2],lo)))/area)if area>0 else None
def load_score(path):
 with np.load(path,allow_pickle=False)as f:
  mask=f['corpus']=='NATIVE';local=f['source_occurrence_index'][mask];assert np.array_equal(local,np.arange(55296));return {k:np.array(f[k][mask],copy=True)for k in ('prediction','query_logit','query_available','occurrence_id','query_xy_raw','L_raw')}
def main():
 control=read(RUN/'CONTROL_COMPLETE.json');assert control['completed']and control['PASS']and control['eligible_source_arm']is None
 rows=read(V/'source_crops_v1/INPUT_ROWS.json')['rows'];gt=read(V/'source_crops_v1/GT_ROWS.json')['rows'];s21=load_score(RUN/'C21_PUBLIC_PLUS_NATIVE/FINAL_ALL61110_SOURCE_SCORES.npz');s20=load_score(RUN/'C20_PUBLIC_REPLAY_CONTROL/FINAL_ALL61110_SOURCE_SCORES.npz')
 with np.load(V/'frozen_C16_source_baseline_v1/FROZEN_C16_PREDICTIONS_INPUT_ONLY.npz',allow_pickle=False)as f:
  m=f['corpus']=='NATIVE';assert np.array_equal(f['source_occurrence_index'][m],np.arange(55296));s16={k:np.array(f[k][m],copy=True)for k in ('prediction','query_logit','query_available')}
 scores={'C16':s16,'C20':s20,'C21':s21};groups=collections.OrderedDict()
 for i,r in enumerate(rows):
  if r['split']=='VALIDATION':groups.setdefault(r['query_group_id'],{})[r['variant']]=i
 eligible=[]
 for gid,g in groups.items():
  ix=[g[v]for v in ('BEFORE','COVER','NEAR_SHAM')]
  if [gt[i]['raw_state']for i in ix]==['D','E','D']:eligible.append({'group':gid,'indices':ix,'r':rows[ix[0]],'success':s21['prediction'][ix].tolist()==[0,1,0]})
 first=lambda predicate:next(x for x in eligible if predicate(x))
 same_success=first(lambda x:x['r']['factors']['foreign_family']=='SAME_TARGET_MATERIAL'and x['success'])
 selected=[('FIRST_VAL_SAME_MATERIAL_SUCCESS',same_success),('FIRST_VAL_SAME_MATERIAL_COVER_FALSE_D',first(lambda x:x['r']['factors']['foreign_family']=='SAME_TARGET_MATERIAL'and s21['prediction'][x['indices'][1]]!=1)),('IDENTITY_COUNTERPART_OF_SAME_SUCCESS',first(lambda x:x['r']['base_index']==same_success['r']['base_index']and x['r']['logical_slot']==same_success['r']['logical_slot']and x['r']['condition_index']==0)),('FIRST_VAL_SAME_MATERIAL_ANY_FAILURE',first(lambda x:x['r']['factors']['foreign_family']=='SAME_TARGET_MATERIAL'and not x['success'])),('FIRST_VAL_OTHER_MATERIAL_SUCCESS',first(lambda x:x['r']['factors']['foreign_family']!='SAME_TARGET_MATERIAL'and x['success'])),('FIRST_VAL_OTHER_MATERIAL_COVER_FALSE_D',first(lambda x:x['r']['factors']['foreign_family']!='SAME_TARGET_MATERIAL'and s21['prediction'][x['indices'][1]]!=1))]
 assert len({x['group']for _,x in selected})==6
 lock={'completed':True,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'PUBLIC synthetic source-only postfit diagnosis; no current/private/V112. First original metadata order within declared role. Source-score outcome is used for fixed diagnostic roles, never checkpoint/threshold or next model selection.','selection_before_crop_or_feature_value_reads':True,'roles':[{'role':role,'group':x['group'],'occurrences':x['indices'],'base':x['r']['base_index'],'condition':x['r']['condition_id'],'factors':x['r']['factors'],'C21_predictions':s21['prediction'][x['indices']].tolist()}for role,x in selected],'control_complete':bind(RUN/'CONTROL_COMPLETE.json'),'score_bindings':{'C16':bind(V/'frozen_C16_source_baseline_v1/FROZEN_C16_PREDICTIONS_INPUT_ONLY.npz'),'C20':bind(RUN/'C20_PUBLIC_REPLAY_CONTROL/FINAL_ALL61110_SOURCE_SCORES.npz'),'C21':bind(RUN/'C21_PUBLIC_PLUS_NATIVE/FINAL_ALL61110_SOURCE_SCORES.npz')},'input_rows':bind(V/'source_crops_v1/INPUT_ROWS.json'),'GT_rows':bind(V/'source_crops_v1/GT_ROWS.json')}
 write(H/'CASE_SELECTION_BEFORE_CROP_FEATURE_READ.json',lock)
 # Read only feature chunks containing the18 fixed native occurrence indices.
 wanted={i for _,x in selected for i in x['indices']};features={};feature_bindings=[];receipt=read(V/'frozen_C16_source_baseline_v1/FEATURE_COMPLETE.json');assert receipt['completed']and receipt['PASS']
 for b in receipt['feature_chunks']:
  with np.load(b['path'],allow_pickle=False)as f:
   indices=f['global_index'];hit=[(j,int(i))for j,i in enumerate(indices)if int(i)in wanted]
   if hit:
    check(b);feature_bindings.append(b);a=f['features']
    for j,i in hit:features[i]=np.array(a[j],copy=True)
 assert set(features)==wanted
 crops={};valids={};cache={}
 for i in wanted:
  r=rows[i];key=r['rgb96_chunk']['path']
  if key not in cache:
   check(r['rgb96_chunk'])
   with np.load(key,allow_pickle=False)as f:cache[key]=(np.array(f['rgb96']),np.array(f['valid96']),np.array(f['occurrence_index']))
  a,v,idx=cache[key];j=r['rgb96_row'];assert idx[j]==i and ash(a[j])==r['rgb96_sha256']and ash(v[j])==r['valid96_sha256'];crops[i]=a[j].copy();valids[i]=v[j].copy()
 del cache
 cases=[];slices={'T':(0,384),'Q':(384,768),'F':(768,1152),'C':(1152,1536),'P':(1536,1920)}
 for role,x in selected:
  ix=x['indices'];rr=[rows[i]for i in ix];assert all(all(r[k]==rr[0][k]for k in ('query_xy','initial_points','L','context_box_xyxy'))for r in rr)
  details=[]
  for i in ix:
   r,g=rows[i],gt[i];f=features[i];assert f.shape==(1928,)and np.isfinite(f).all()
   for model in ('C20','C21'):assert scores[model]['occurrence_id'][i]==r['id']and np.array_equal(scores[model]['query_xy_raw'][i],r['query_xy'])and scores[model]['L_raw'][i]==r['L']
   b=g['active_witness_support_GT_only']['local96_bbox_xyxy'];rf=(np.array(g['reference_xy'])-r['query_xy'])*96/r['L']+47.5
   footprints={k:{'whole_active_bbox_inside':inside(b,lo,hi),'bbox_area_fraction_inside':overlap(b,lo,hi),'reference_inside':bool(np.all((rf>=lo)&(rf<=hi)))}for k,lo,hi in [('FULL96',-.5,95.5),('CENTRAL32',31.5,63.5),('P_NOMINAL_CENTRAL12',41.5,53.5)]}
   details.append({'occurrence_index':i,'id':r['id'],'variant':r['variant'],'query_xy':r['query_xy'],'L':r['L'],'context_box_xyxy':r['context_box_xyxy'],'reference_xy':g['reference_xy'],'reference_xy_in_local96':rf.tolist(),'error_native_px':g['error_px'],'label':g['raw_state'],'q_pixel':g['q_nearest_pixel_owner_diagnostic'],'active_bbox_local96':b,'nominal_footprints':footprints,'reference_in_context':g['active_witness_support_GT_only']['reference_in_context'],'active_bbox_in_context':g['active_witness_support_GT_only']['active_bbox_in_context'],'input_available':r['query_available'],'crop_chunk':r['rgb96_chunk'],'crop_row':r['rgb96_row'],'crop_sha256':r['rgb96_sha256'],'valid_sha256':r['valid96_sha256'],'feature_vector_sha256':ash(f),'feature_descriptor_norms':{k:float(np.linalg.norm(f[a:b]))for k,(a,b)in slices.items()},'feature_scalars':{'cos_T_C':float(f[1920]),'cos_T_F':float(f[1921]),'cos_Q_C':float(f[1922]),'cos_Q_P':float(f[1923]),'context_valid_fraction':float(f[1924]),'full96_valid_fraction':float(f[1925]),'central32_valid_fraction':float(f[1926]),'Q_inside_context':bool(f[1927])},'scores':{model:{'hidden_logit':float(sc['query_logit'][i]),'prediction_D0_E1_Uminus1':int(sc['prediction'][i]),'query_available':bool(sc['query_available'][i])}for model,sc in scores.items()}})
  transitions={}
  for v,k in [('COVER',1),('NEAR_SHAM',2)]:
   a,b=ix[0],ix[k];different=np.any(crops[a]!=crops[b],axis=2)
   transitions[v]={'RGB96_changed_pixels':int(different.sum()),'central32_changed_pixels':int(different[32:64,32:64].sum()),'nominal_P12_changed_pixels':int(different[42:54,42:54].sum()),'fixed_feature_L2_change':{name:float(np.linalg.norm(features[b][lo:hi]-features[a][lo:hi]))for name,(lo,hi)in slices.items()},'C16_hidden_logit_change':float(s16['query_logit'][b]-s16['query_logit'][a]),'C21_hidden_logit_change':float(s21['query_logit'][b]-s21['query_logit'][a])}
  cases.append({'role':role,'group':x['group'],'base':rr[0]['base_index'],'condition':rr[0]['condition_id'],'factors':rr[0]['factors'],'rows':details,'transitions_from_BEFORE':transitions})
 figs=[];palette={0:'D',1:'E',-1:'U'}
 for sheet in range(2):
  board=Image.new('RGB',(1260,60+3*350),'#edf2f6');d=ImageDraw.Draw(board);d.text((12,8),'Saved source-only known D/E/D |columns BEFORE / COVER / NEAR_SHAM',font=font(19),fill='#132d3e');d.text((12,34),'RAW96 +markedcopy. q magenta;r cyan; activebbox green; central32 yellow; Pnominal12 blue. Marks neverinputs.',font=font(13),fill='#314e60')
  for local,case in enumerate(cases[sheet*3:sheet*3+3]):
   y=60+350*local;first=case['rows'][0];d.text((12,y),f'{case["role"]} |base{case["base"]} {case["condition"]} |L{first["L"]:.3f}px',font=font(14),fill='#173447')
   for col,row in enumerate(case['rows']):
    i=row['occurrence_index'];x=8+420*col;raw=Image.fromarray(crops[i]).resize((192,192),Image.Resampling.NEAREST);marked=raw.copy();md=ImageDraw.Draw(marked)
    coord=lambda z:(z+.5)*2-.5
    for lo,hi,color in [(31.5,63.5,'yellow'),(41.5,53.5,'deepskyblue')]:md.rectangle((coord(lo),coord(lo),coord(hi),coord(hi)),outline=color,width=1)
    bb=row['active_bbox_local96'];md.rectangle(tuple(coord(v)for v in bb),outline='lime',width=2)
    for point,color in [([47.5,47.5],'magenta'),(row['reference_xy_in_local96'],'cyan')]:
     xx,yy=map(coord,point);md.line((xx-5,yy,xx+5,yy),fill=color,width=2);md.line((xx,yy-5,xx,yy+5),fill=color,width=2)
    board.paste(raw,(x,y+39));board.paste(marked,(x+200,y+39));d.text((x,y+22),f'{row["variant"]} row{i} label{row["label"]}',font=font(12),fill='black')
    pr=row['scores'];txt=' | '.join(m+':'+palette[pr[m]['prediction_D0_E1_Uminus1']]+f' {pr[m]["hidden_logit"]:+.3f}'for m in ('C16','C20','C21'));d.text((x,y+235),txt,font=font(12),fill='#173447')
    d.text((x,y+252),f'r_local({row["reference_xy_in_local96"][0]:.2f},{row["reference_xy_in_local96"][1]:.2f}) bboxspan({bb[2]-bb[0]:.2f},{bb[3]-bb[1]:.2f})',font=font(12),fill='#173447')
    fp=row['nominal_footprints'];d.text((x,y+269),f'r in FULL/C32/P12: {fp["FULL96"]["reference_inside"]}/{fp["CENTRAL32"]["reference_inside"]}/{fp["P_NOMINAL_CENTRAL12"]["reference_inside"]};ctx {row["reference_in_context"]}',font=font(11),fill='#173447')
   t=case['transitions_from_BEFORE'];desc=[]
   for v in ('COVER','NEAR_SHAM'):
    w=t[v];desc.append(v+f' pixelchange{w["RGB96_changed_pixels"]} /central{w["central32_changed_pixels"]}; '+','.join(k+f'{z:.3f}'for k,z in w['fixed_feature_L2_change'].items()))
   d.text((12,y+291),' | '.join(desc),font=font(12),fill='#315366');d.text((12,y+310),'Nominal footprint overlap is geometric; DINO tokens have global receptive fields. Feature change is not pixel-causal proof.',font=font(12),fill='#62533e')
  f=H/f'FIXED6_TRIPLETS_{sheet+1:02d}.png';board.save(f);figs.append(bind(f))
 result={'completed':True,'scope':'Saved postfit synthetic-source diagnostic only. Six fixed VAL groups; no private/current/V112, network/model/fit/threshold/candidate changes.','selection_lock':bind(H/'CASE_SELECTION_BEFORE_CROP_FEATURE_READ.json'),'code':bind(__file__),'fixed_feature_receipt':bind(V/'frozen_C16_source_baseline_v1/FEATURE_COMPLETE.json'),'feature_chunks_read':feature_bindings,'cases':cases,'figures':figs,'actual_visual_observations_pending':True,'limits':['These first metadata-order cases all happen to be BOARD_BLOCK_RUNNER/PROCEDURAL_WOOD; they do not represent everygeometry/material or independent captures.','The known source region identity is fixed across qconditions. q/r overlays and region boxes are diagnostic only; no GT is forwarded to models.','FULL96/C32/Pnominal12 are input-coordinate footprint summaries, not hard ViT receptive-field limits or a claim of unobservability.','Feature distances show saved representation changes, not isolated causal attribution.','Both C20/C21 remain source-ineligible; no new current evaluation or real transfer claim.']}
 write(H/'RESULT.json',result);print(json.dumps({'completed':True,'result':bind(H/'RESULT.json'),'figures':figs,'cases':[{'role':x['role'],'rows':[r['occurrence_index']for r in x['rows']],'C16':[r['scores']['C16']['prediction_D0_E1_Uminus1']for r in x['rows']],'C21':[r['scores']['C21']['prediction_D0_E1_Uminus1']for r in x['rows']]}for x in cases]}))
if __name__=='__main__':main()
