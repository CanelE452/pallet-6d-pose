"""Exactly the four pre-fit source previews, sealed V76/V77 scores only.

No model import, newforward/fit/PnP, currentdata, or score-based case selection.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
V76=P/'query_visibility_reference_local_support_20261023_v76'
RUN=ROOT/'runs/PAIRED_SUPPORT_V1'
BASE=ROOT/'frozen_V76_source_baseline'
PAIR=ROOT/'pair_generation'
FIT_SHA='1af4ec060f19bde2dc70282f99d6274d98e3be8f6066c7f5b56f365647d6b823'
BASE_SEAL_SHA='583691e015d91832bcd582c5c5058b9900dd6d3555bca585ab5961452ade45aa'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return {'path':str(p),'sha256':sha(p)}
def check(d):assert sha(d['path'])==d['sha256'],d['path']
def read(p):return json.loads(Path(p).read_text())
def lines(p):return[json.loads(x)for x in p.read_text().splitlines()]
def save(p,d):
    assert not p.exists();p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold'if bold else'')+'.ttf',n)
def mark(im,xy,c):
    x,y=map(float,xy);d=ImageDraw.Draw(im);d.line((x-8,y,x+8,y),fill=c,width=2);d.line((x,y-8,x,y+8),fill=c,width=2)
def probability(z):return float(np.exp(-np.logaddexp(0.,-float(z))))
def state(z):return 'VISIBLE'if float(z)>=0 else'COVERED'
def load_scores(p):
    keys=['support_logit','input_available','primary_loss_mask','id','source','split','domain','after_original_row_index','after_anchor_k','pair_good']
    with np.load(p,allow_pickle=False)as z:return{k:z[k]for k in keys}

def main():
    assert not(OUT/'SAVED_SCORE_GALLERY_METADATA.json').exists(),'Preserve previousoutputs'
    # Existence+exact completed seal first: no predictions are read whilefitlive.
    if not(RUN/'FIT_COMPLETE.json').exists():print('WAIT_FIT_COMPLETE; no predictions read',flush=True);return
    assert sha(RUN/'FIT_COMPLETE.json')==FIT_SHA
    fit=read(RUN/'FIT_COMPLETE.json');assert fit['completed']and fit['optimizer_updates']==2000
    check(fit['model_freeze']);freeze=read(fit['model_freeze']['path']);assert freeze['completed']and freeze['threshold']==.5 and not freeze['current_used_for_fit_or_checkpoint_or_threshold_selection']
    assert sha(BASE/'BASELINE_SEAL.json')==BASE_SEAL_SHA
    baseline=read(BASE/'BASELINE_SEAL.json');assert baseline['completed']and baseline['PASS']and baseline['original_logit_differences']==0 and baseline['weights_unchanged']
    check(baseline['scores']);check(fit['scores'])
    preview=read(PAIR/'fixed_zoom_previews/SEAL.json');check(preview['selection']);check(preview['dataset'])
    selected=read(preview['selection']['path']);indices=selected['after_indices'];cases=preview['cases']
    assert len(cases)==len(indices)==4 and[c['after_row_index']for c in cases]==indices
    allpairs=lines(PAIR/'PAIR_LINEAGE_GT_ONLY.jsonl');pairindex=lines(PAIR/'PAIR_SELECTION_GT_ONLY.jsonl')
    fixed=[allpairs[i]for i in indices]
    expected_groups=[('SYNTHETIC','train'),('SYNTHETIC','val'),('YCB_PUBLIC_REAL','train'),('YCB_PUBLIC_REAL','val')]
    assert[(p['source'],p['split'])for p in fixed]==expected_groups
    save(OUT/'FIXED_CASE_LOCK_BEFORE_SCORE_VALUES.json',{'selection':'Exactly unchanged four metadata-first source/split previewcases, frozen before V77fit; no scoreselection',
        'after_indices':indices,'case_ids':[(x['original_id'],x['anchor_k'])for x in fixed],
        'bindings':[b(p)for p in [PAIR/'fixed_zoom_previews/SEAL.json',Path(preview['selection']['path']),PAIR/'PAIR_LINEAGE_GT_ONLY.jsonl',
            RUN/'FIT_COMPLETE.json',BASE/'BASELINE_SEAL.json',Path(fit['scores']['path']),Path(baseline['scores']['path']),Path(__file__)]],
        'new_current_forward_fit_PnP':0})
    old=load_scores(Path(baseline['scores']['path']));new=load_scores(Path(fit['scores']['path']))
    for key in ['id','source','split','domain','after_original_row_index','after_anchor_k','pair_good','primary_loss_mask','input_available']:
        np.testing.assert_array_equal(old[key],new[key])
    assert old['support_logit'].shape==new['support_logit'].shape==(10002,8)and np.isfinite(old['support_logit']).all()and np.isfinite(new['support_logit']).all()
    panels=[];records=[]
    for j,(case,pair)in enumerate(zip(cases,fixed)):
        ai=case['after_row_index'];oi=pair['original_row_index'];k=case['anchor_k'];fi=5203+ai
        assert pair['anchor_success_COVERED']and k==pair['anchor_k']and old['pair_good'][ai]
        assert old['after_original_row_index'][ai]==oi and old['after_anchor_k'][ai]==k
        assert str(old['id'][oi])==case['original_id']==pair['original_id']and str(old['id'][fi])==pair['after_id']
        assert old['primary_loss_mask'][oi,k]and old['primary_loss_mask'][fi,k]
        q=np.asarray(case['q'],float);r=np.asarray(case['r'],float);L=float(case['L'])
        assert np.array_equal(q,np.asarray(pairindex[ai]['initial_points'][k]))and np.array_equal(r,np.asarray(pair['native_GTref8'][k]))and L==pairindex[ai]['roi_side']
        check(case['before_crop']);check(case['after_crop']);before=Image.open(case['before_crop']['path']).convert('RGB');after=Image.open(case['after_crop']['path']).convert('RGB');assert before.size==after.size==(96,96)
        panel=Image.new('RGB',(1600,600),'#111820');d=ImageDraw.Draw(panel)
        d.text((16,10),f'{j+1}. {pair["source"]}/{pair["split"]} {pair["original_id"]} native{k}',font=font(23,True),fill='white')
        d.text((16,45),'Same pre-fit fixedpair. Weak teacher: originalVISIBLE -> controlledCOVERED. No semantic humanD/E claim.',font=font(18),fill='#d8dce3')
        for x,title,img in [(16,'BEFORE: unchanged entire L inputRGB96',before),(420,'AFTER: sameq/L, opaque foreignobject',after)]:
            im=img.resize((384,384),Image.Resampling.NEAREST);mark(im,[191.5,191.5],(0,240,255));mark(im,((r-q)*96/L+48)*4-.5,(255,0,255))
            panel.paste(im,(x,110));d.text((x,80),title,font=font(18),fill='#d8dce3')
        x=840;d.text((x,90),'Fixed .5 threshold / saved local-support scores',font=font(21,True),fill='white')
        model_values={}
        for n,(name,zs,color)in enumerate([('V76 frozen source-selected',old,'#cbd7eb'),('V77 selected naturalVAL',new,'#ffe091')]):
            bz=float(zs['support_logit'][oi,k]);az=float(zs['support_logit'][fi,k]);bp=probability(bz);ap=probability(az);both=bool(bz>=0 and az<0)
            y=140+n*135
            d.text((x,y),name,font=font(22,True),fill=color)
            d.text((x,y+35),f'Before pV={bp:.6f} -> {state(bz)}',font=font(21),fill=color)
            d.text((x,y+66),f'After  pV={ap:.6f} -> {state(az)}',font=font(21),fill=color)
            d.text((x,y+97),f'both correct={both}; pBefore-pAfter={bp-ap:+.6f}',font=font(18),fill=color)
            model_values[name]={'before_logit':bz,'after_logit':az,'before_pV':bp,'after_pV':ap,'before_prediction':state(bz),'after_prediction':state(az),'both_correct':both,'before_gt_after':bool(bz>az),
                'all8_before_logits':zs['support_logit'][oi].astype(float).tolist(),'all8_after_logits':zs['support_logit'][fi].astype(float).tolist()}
        d.text((x,440),f'V77 checkpoint update={freeze["best_update"]}/2000',font=font(19),fill='#d4d8dd')
        d.text((16,520),f'q cyan={q.round(3).tolist()} | r magenta={r.round(3).tolist()} | q-r={np.linalg.norm(q-r):.3f}px | L={L:.3f}px',font=font(18),fill='#dddddd')
        d.text((16,550),'Public q is GT-derived source emulation. Reference/masks are GT-only. Model also reads P3/target/query context; local view is illustrative.',font=font(17),fill='#dddddd')
        d.text((16,577),'Fixed training and validation examples are descriptive; four cases do not establish generalization or explain a model branch causally.',font=font(17),fill='#dddddd')
        png=OUT/f'PAIR_{ai:06d}_SAVED_V76_V77.png';panel.save(png);panels.append(panel)
        records.append({'after_row_index':ai,'original_row_index':oi,'original_id':case['original_id'],'anchor_k':k,'source':pair['source'],'split':pair['split'],'q':q.tolist(),'r':r.tolist(),'L':L,
            'before_teacher_label':1,'after_teacher_label':0,'teacher_codes':{'1':'VISIBLE','0':'COVERED','-1':'U'},'convex_self_visible_separate':pair['convex_self_visible_proxy8_separate'][k],
            'scores':model_values,'raw_before_RGB96':case['before_crop'],'raw_after_RGB96':case['after_crop'],'panel':b(png),'selection_unchanged':True})
    sheet=Image.new('RGB',(1600,600*4),'black')
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*600))
    sheet.save(OUT/'CONTACT_SHEET.png')
    save(OUT/'SAVED_SCORE_GALLERY_METADATA.json',{'completed':True,'cases':records,'case_lock':b(OUT/'FIXED_CASE_LOCK_BEFORE_SCORE_VALUES.json'),
        'frozen_V76_baseline':b(BASE/'BASELINE_SEAL.json'),'selected_V77_fit':b(RUN/'FIT_COMPLETE.json'),'V77_best_update':freeze['best_update'],
        'scope':'Fixed four pre-exposed source illustrative pairs, not fullsourceevaluation and not semantic humanvisibility benchmark. V76baseline is actual frozenV76 bothbefore/after replay, NOT zeroheadBASELINE0.',
        'threshold':.5,'selection_or_threshold_changes':0,'new_model_forward_fit_current_PnP':0,'contact_sheet':b(OUT/'CONTACT_SHEET.png')})
    save(OUT/'SEAL.json',{'completed':True,'code':b(Path(__file__)),'outputs':[b(p)for p in sorted(OUT.iterdir())if p.is_file()and p.name!='SEAL.json']})
    print(json.dumps({'metadata':b(OUT/'SAVED_SCORE_GALLERY_METADATA.json'),'seal':b(OUT/'SEAL.json')}))
if __name__=='__main__':main()
