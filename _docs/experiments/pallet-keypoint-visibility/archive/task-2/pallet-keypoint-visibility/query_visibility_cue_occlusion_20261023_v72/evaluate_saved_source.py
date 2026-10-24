"""Read saved predictions only; conditional D/E teachers and U remain distinct."""
import hashlib,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
OUT=R/'source_evaluation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def metrics(rows,z,ix):
    known=[i for i in ix if rows[i]['training_state'] in ['D','E']]
    d=np.array([rows[i]['training_state']=='D' for i in known],dtype=bool);p=z[known]>=0
    tp,fn,fp,tn=[int(x.sum()) for x in [d&p,d&~p,~d&p,~d&~p]]
    safe=lambda a,b:float(a/b) if b else None
    loss=np.logaddexp(0,z[known])-d*z[known]
    return dict(N=len(known),D=int(d.sum()),E=int((~d).sum()),confusion_GT_D_E_pred_D_E=[[tp,fn],[fp,tn]],
      TP_D=tp,FN_D=fn,FP_D=fp,TN_D=tn,D_precision=safe(tp,tp+fp),D_recall=safe(tp,tp+fn),E_recall=safe(tn,tn+fp),
      E_precision=safe(tn,tn+fn),known_NLL=float(loss.mean()) if len(loss) else None,
      balanced_NLL=float(.5*loss[d].mean()+.5*loss[~d].mean()) if d.any() and (~d).any() else None,
      trained_U_or_abstention=False)
def main():
    OUT.mkdir(exist_ok=False)
    rows=[json.loads(l) for l in (R/'source_cache/GT_ONLY.jsonl').read_text().splitlines()]
    run=R/'runs/CUE_ATTENTION_V3';zz=np.load(run/'SOURCE_SCORES.npz',allow_pickle=False)
    scores={'V72':zz['logit'].astype(float),'v61':np.load(run/'V61_SOURCE_SCORES.npz',allow_pickle=False)['logit'].astype(float)}
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        for k in ['ALL',r['source']+'|'+r['split'],r['source']+'|'+r['split']+'|'+r['cue_refID']]:groups[k].append(i)
    lookup={(r['source'],r['cue_refID'],tuple(r['query_xy']),r['L'],r['variant']):i for i,r in enumerate(rows)}
    pairs=[]
    for i,r in enumerate(rows):
        if r['training_state']!='D' or r['variant'] not in ['ORIGINAL','BASE_ORIGINAL']:continue
        for variant in ['FULL_COVER','NEUTRAL_D','QUERY_ONLY_CONTROL','PARTIAL_COVER']:
            key=(r['source'],r['cue_refID'],tuple(r['query_xy']),r['L'],variant)
            if key in lookup:
                j=lookup[key];pairs.append(dict(original_index=i,intervention_index=j,source=r['source'],split=r['split'],cue_refID=r['cue_refID'],kind='D_to_'+rows[j]['training_state'],variant=variant))
    results={}
    for name,z in scores.items():
        assert z.shape==(3905,) and np.isfinite(z).all()
        paired=defaultdict(lambda:dict(N=0,both_correct=0,original_D=0,intervention_correct=0))
        for pair in pairs:
            if pair['kind']=='D_to_U':continue
            a,b=pair['original_index'],pair['intervention_index'];k='|'.join([pair['source'],pair['split'],pair['variant'],pair['kind']])
            v=paired[k];v['N']+=1;v['original_D']+=int(z[a]>=0)
            correct=(z[b]>=0) if rows[b]['training_state']=='D' else (z[b]<0)
            v['intervention_correct']+=int(correct);v['both_correct']+=int(z[a]>=0 and correct)
        unknown=defaultdict(lambda:dict(N=0,raw_D=0,raw_E=0))
        for i,r in enumerate(rows):
            if r['training_state']=='U':
                k='|'.join([r['source'],r['split'],r['ureason'],r['crop_availability']]);v=unknown[k];v['N']+=1;v['raw_D']+=int(z[i]>=0);v['raw_E']+=int(z[i]<0)
        offset=defaultdict(list);nullref=0
        for i,r in enumerate(rows):
            if r['reference_xy'] is None:nullref+=1;continue
            e=np.max(np.abs((np.array(r['query_xy'])-r['reference_xy'])/r['L']))
            key='[0,.125]' if e<=.125 else '(.125,.25]' if e<=.25 else '(.25,.5]' if e<=.5 else '>.5'
            offset[r['source']+'|'+r['split']+'|'+key].append(i)
        results[name]=dict(cohorts={k:metrics(rows,z,ix) for k,ix in groups.items()},paired=dict(paired),unknown_output_distribution=dict(unknown),by_reference_offset={k:metrics(rows,z,ix) for k,ix in offset.items()},null_reference_U_rows_excluded_from_offset=nullref)
    # Localization is auxiliary and conditional; peak proximity is no evidence of instance association.
    yy,xx=np.meshgrid(np.arange(9),np.arange(9),indexing='ij');loc=defaultdict(list)
    for i,r in enumerate(rows):
        if r['training_state']!='D':continue
        q=np.array(r['query_xy']);L=r['L'];att=zz['attention'][i];coord=np.stack((xx-4,yy-4),-1)*L/9+q
        pred=(att[...,None]*coord).sum((0,1));peak=coord.reshape(-1,2)[att.argmax()]
        unc=r['uncertainty_bbox_xyxy'];inunc=unc[0]<=peak[0]<unc[2] and unc[1]<=peak[1]<unc[3]
        loc[r['source']+'|'+r['split']].append(dict(row=i,attention_center_distance_px=float(np.linalg.norm(pred-r['reference_xy'])),peak_distance_px=float(np.linalg.norm(peak-r['reference_xy'])),peak_in_uncertainty=bool(inunc)))
    write(OUT/'SOURCE_RESULTS.json',dict(models=results,localization=loc,pairs=pairs,threshold_logit=0.,fixed_semantics='D=original cue visible; E=same cue whole uncertainty opaque-covered; U=unknown/MISS/partial/stress',
      limitation='conditional AI cue teacher, 3 public VAL cues, no native8 gold/U classifier/S supervision',
      source_scores_sha256=sha(run/'SOURCE_SCORES.npz'),v61_scores_sha256=sha(run/'V61_SOURCE_SCORES.npz'),GT_sha256=sha(R/'source_cache/GT_ONLY.jsonl'),full_goal_solved=False))
    # Fixed first FULL original query for each known cue, paired raw input crop.
    data=np.load(R/'source_cache/MODEL_INPUT_ONLY.npz',allow_pickle=False)
    chosen=[];seen=set()
    for p in pairs:
        if p['variant']!='FULL_COVER':continue
        key=(p['source'],p['cue_refID'])
        if key not in seen:chosen.append(p);seen.add(key)
    font=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',16)
    im=Image.new('RGB',(1400,90+len(chosen)*280),'#f2f4f7');dr=ImageDraw.Draw(im)
    dr.text((15,10),'실제 모델 RGB96 crop · 각 cue의 고정 첫 FULL query · 원본 D / 전체가림 E',font=font,fill='black')
    dr.text((15,40),'노란색: 평가용 고정 reference / 빨간색: 예측 attention 최고 셀 · 모델 입력에는 표시 없음',font=small,fill='black')
    for n,p in enumerate(chosen):
        top=80+n*280;a,b=p['original_index'],p['intervention_index'];r=rows[a]
        dr.text((10,top),r['source']+' '+r['split']+' '+r['cue_refID'][:64],font=small,fill='black')
        for col,i in enumerate([a,b]):
            left=10+col*680;rgb=Image.fromarray(data['rgb96'][i].transpose(1,2,0)).resize((210,210),Image.Resampling.NEAREST)
            im.paste(rgb,(left,top+30));q=np.array(rows[i]['query_xy']);L=rows[i]['L'];ref=np.array(rows[i]['reference_xy']);pt=(ref-q)/L*210+105
            dr.ellipse((left+pt[0]-5,top+30+pt[1]-5,left+pt[0]+5,top+30+pt[1]+5),outline='yellow',width=2)
            peak=np.unravel_index(zz['attention'][i].argmax(),(9,9));px=(peak[1]+.5)*210/9;py=(peak[0]+.5)*210/9
            dr.rectangle((left+px-5,top+30+py-5,left+px+5,top+30+py+5),outline='red',width=2)
            labels=[f"GT {rows[i]['training_state']} / row {i}",f"V72 P(D)={1/(1+np.exp(-scores['V72'][i])):.4f}",f"v61 P(pixel)={1/(1+np.exp(-scores['v61'][i])):.4f}",f"q={q.round(1)} L={L:.1f}"]
            for k,t in enumerate(labels):dr.text((left+230,top+38+36*k),t,font=small,fill='black')
    im.save(OUT/'FIXED_FIRST_FULL_CUE_PAIRS.png')
    print(json.dumps({m:{k:v for k,v in obj['cohorts'].items() if k in ['SYNTH|TRAIN','YCB|TRAIN','YCB|VALIDATION']} for m,obj in results.items()},indent=2))
if __name__=='__main__':main()
