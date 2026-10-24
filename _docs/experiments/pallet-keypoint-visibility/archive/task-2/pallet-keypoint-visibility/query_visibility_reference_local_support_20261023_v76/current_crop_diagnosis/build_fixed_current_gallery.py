"""Fixed six current cases, saved output/RGB only after exact approved seal.

No model import/forward/fit/PnP, no new case selection, no pixel GT invented.
RGB96 means entire native span L resampled96, never a native96-pixel window.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
OLD=P/'query_visibility_cue_occlusion_20261023_v72'
RUN=ROOT/'private_evaluation/MIXED_SUPPORT_V1'
OLD_RUN=OLD/'private_evaluation/CUE_ATTENTION_V3'
SELECTION=P/'query_visibility_controlled_real_20261022_v66/diagnostics/FIXED_REAL_CROPS_V66.json'
OLD_META=OLD/'diagnostics/FIXED_CURRENT_CROP_METADATA.json'
CROP_CODE=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
AUTHORIZED_SEAL='ddece1626e16631c960d724a2723841e0b3588c095098e2f0546d7e9ac91d2ac'
KEYS=[('wood_night_01:032494',5),('SQUARE_YOLO:025747',5),('wood_night_01:032376',0),('SQUARE_YOLO:027620',1),('wood_day_01:005249',5),('SQUARE_YOLO:030931',4)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return {'path':str(p),'sha256':sha(p)}
def check(x):assert sha(x['path'])==x['sha256'],x['path']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    assert not p.exists();p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def pure(path,name):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env);return env[name]
def table(p):
    with p.open(newline='')as f:rows=list(csv.DictReader(f))
    result={(r['id'],int(r['native_corner'])):(i,r)for i,r in enumerate(rows)}
    assert len(rows)==len(result)==1784
    return result
def sealed_score(seal,score):
    matches=[x for x in seal['outputs']if Path(x['path']).resolve()==score.resolve()]
    assert len(matches)==1;check(matches[0])
def fbool(x):
    if x is True or x=='True':return True
    if x is False or x=='False':return False
    if x is None or x=='':return None
    raise ValueError('Unexpected saved bool')
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold'if bold else'')+'.ttf',n)
def marker(im,p,color,cross=False):
    x,y=map(float,p);d=ImageDraw.Draw(im)
    if cross:d.line((x-8,y,x+8,y),fill=color,width=3);d.line((x,y-8,x,y+8),fill=color,width=3)
    else:d.ellipse((x-10,y-10,x+10,y+10),outline=color,width=3)
def probability(r,key):
    if r.get(key,'')=='':return None
    p=float(r[key]);assert np.isfinite(p)and 0<=p<=1;return p

def main():
    assert not(OUT/'FIXED_CURRENT_CROP_METADATA.json').exists(),'Preserve existing result'
    # Exact root-provided immutable inference seal gates every current body read.
    newsealpath=RUN/'INFERENCE_SEAL.json'
    assert sha(newsealpath)==AUTHORIZED_SEAL
    newseal=read(newsealpath)
    assert newseal['completed']and newseal['frames']==223 and newseal['slots']==1784 and newseal['valid']==1739
    assert newseal['threshold']==.5 and newseal['threshold_comparison']=='>='and newseal['current_used_for_selection']is False
    assert newseal['model_unchanged']and newseal['repeat_logit_max_difference']==0
    assert newseal['new_backbone']==newseal['optimizer_updates']==newseal['PnP']==0
    check(newseal['preparation']);newprep=read(newseal['preparation']['path']);check(newprep['input'])
    assert Path(newprep['input']['path']).resolve()==(RUN/'CURRENT_SIX_INPUTS.npz').resolve()
    newscore=RUN/'REAL_QUERY_SCORES.csv';sealed_score(newseal,newscore)
    oldseal=read(OLD_RUN/'INFERENCE_SEAL.json');assert oldseal['completed']
    oldscore=OLD_RUN/'REAL_QUERY_SCORES.csv';sealed_score(oldseal,oldscore)
    oldprep=read(OLD_RUN/'PREPARATION_SEAL.json');assert oldprep['completed']
    oldarray=OLD_RUN/'CURRENT_MODEL_INPUT.npz'
    oldinput=[x for x in oldprep['outputs']if Path(x['path']).resolve()==oldarray.resolve()];assert len(oldinput)==1;check(oldinput[0])
    oldcases=read(SELECTION);fixed=oldcases['fixed_semantic_D_cases']+oldcases['fixed_semantic_E_cases']
    metadata=read(OLD_META);prior=metadata['cases']
    assert[(c['id'],c['native_corner'])for c in fixed]==[(c['id'],c['native_corner'])for c in prior]==KEYS
    new=table(newscore);old=table(oldscore)
    save(OUT/'SELECTION_AND_EXECUTION_LOCK_BEFORE_RGB.json',{'completed':True,'authorization':'Root explicitly authorized saved current gallery after exact new inference seal on2026-10-23; no extra inference authorized',
        'exact_selection':KEYS,'fixed_since':'V65/V66; original fourD/twoE order; no new score ranking or selection',
        'bindings':[b(p)for p in [newsealpath,newscore,SELECTION,OLD_META,OLD_RUN/'INFERENCE_SEAL.json',oldscore,OLD_RUN/'PREPARATION_SEAL.json',oldarray,CROP_CODE,Path(__file__)]],
        'actual_RGB_body_reads_before_lock':0,'model_weights_or_features_to_read':False,'threshold':.5})
    # Load only RGB/valid/q/L fields. P3/context/target features are never read.
    with np.load(newprep['input']['path'],allow_pickle=False)as z:newrgb=z['local_rgb96'];newvalid=z['local_valid96']
    with np.load(oldarray,allow_pickle=False)as z:oldrgb=z['rgb96'];oldvalid=z['crop_valid96'];oldq=z['query_xy'];oldL=z['side_L']
    assert newrgb.shape==oldrgb.shape==(1784,3,96,96)and newvalid.shape==oldvalid.shape==(1784,1,96,96)
    crop=pure(CROP_CODE,'crop');rawdir=OUT/'raw_full_L_input96';rawdir.mkdir(exist_ok=False)
    cases=[];panels=[]
    for n,(case,previous)in enumerate(zip(fixed,prior)):
        key=KEYS[n];i,r=new[key];oi,older=old[key]
        assert i==oi==previous['saved_array_index']and key==(case['id'],case['native_corner'])
        q=np.asarray(case['query'],np.float64);ref=np.asarray(case['reference'],np.float64);L=float(case['roi_side'])
        assert np.array_equal(q,np.asarray(previous['query']))and np.array_equal(ref,np.asarray(previous['reference']))and L==previous['roi_side']
        assert np.array_equal(q,[float(r['query_x']),float(r['query_y'])])and np.array_equal(q,[float(older['query_x']),float(older['query_y'])])
        assert L==float(r['roi_side'])==float(older['roi_side'])and np.array_equal(oldq[i],q.astype(np.float32))and oldL[i]==np.float32(L)
        assert r['image_sha256']==older['image_sha256']==case['image_sha256']
        assert sha(case['image'])==case['image_sha256']
        rgb=np.asarray(Image.open(case['image']).convert('RGB'));direct,valid=crop(rgb,q,L,96)
        np.testing.assert_array_equal(direct,newrgb[i]);np.testing.assert_array_equal(direct,oldrgb[i]);np.testing.assert_array_equal(valid,newvalid[i]);np.testing.assert_array_equal(valid,oldvalid[i])
        assert abs(float(np.linalg.norm(ref-q))-case['error_px'])<1e-8
        raw=direct.transpose(1,2,0).copy();rawpath=rawdir/(case['id'].replace(':','_')+f'_k{key[1]}_FULL_L_MODEL_RGB96.png')
        validpath=rawdir/(rawpath.stem+'_VALID.png');Image.fromarray(raw).save(rawpath);Image.fromarray(valid[0].astype(np.uint8)*255).save(validpath)
        p61=float(case['v61_p_visible']);p72=probability(older,'p_visible');p76=probability(r,'p_visible');raw76=probability(r,'raw_p_local_support_visible')
        assert p61==previous['v61_saved_p_visible']and p72==previous['v72_saved_p_D_given_DE']
        valid76=fbool(r['valid']);pred76='U'if not valid76 else('VISIBLE_SUPPORT'if float(r['logit'])>=0 else'COVERED_SUPPORT')
        pred72='U'if not fbool(older['valid'])else('D'if float(older['logit'])>=0 else'E')
        pred61='VISIBLE'if p61>=.5 else'ABSENT'
        canvas=Image.new('RGB',(1680,610),'#111820');d=ImageDraw.Draw(canvas)
        d.text((16,12),f'{n+1}. {key[0]} native{key[1]} | existing human semantic {case["human_state_short"]}',font=font(23,True),fill='white')
        d.text((16,45),'Actual complete input span L; RGB96 is its resampling, not a96-native-pixel window.',font=font(17),fill='#d4d8dd')
        thumb=Image.fromarray(rgb);scale=min(320/rgb.shape[1],330/rgb.shape[0]);thumb=thumb.resize((round(rgb.shape[1]*scale),round(rgb.shape[0]*scale)),Image.Resampling.LANCZOS)
        td=ImageDraw.Draw(thumb);td.rectangle(tuple(np.r_[(q-L/2)*scale,(q+L/2)*scale]),outline=(0,230,255),width=2);marker(thumb,q*scale,(0,230,255),True);marker(thumb,ref*scale,(255,0,220))
        canvas.paste(thumb,(16,105))
        unmarked=Image.fromarray(raw).resize((384,384),Image.Resampling.NEAREST);marked=unmarked.copy()
        ref96=(ref-q)*96/L+47.5;marker(marked,[191.5,191.5],(0,230,255),True);marker(marked,(ref96+.5)*4-.5,(255,0,220))
        canvas.paste(unmarked,(360,105));canvas.paste(marked,(760,105))
        for x,t in [(16,'Original image + unchanged L square'),(360,'Raw full-L model RGB96, enlarged4x'),(760,'Same full-L crop + display-only markers')]:d.text((x,78),t,font=font(17),fill='#ccd3dc')
        d.text((1160,110),'Fixed .5 decisions; saved outputs',font=font(20,True),fill='white')
        d.text((1160,153),f'v61 pV={p61:.6f}',font=font(21),fill='#cbd7eb');d.text((1160,181),'-> '+pred61,font=font(20),fill='#cbd7eb')
        d.text((1160,233),'V72 pD|DE='+('U'if p72 is None else f'{p72:.6f}'),font=font(21),fill='#d2e9d6');d.text((1160,261),'-> '+pred72,font=font(20),fill='#d2e9d6')
        d.text((1160,313),'V76 p(local support visible)',font=font(20,True),fill='#ffe091')
        d.text((1160,345),('U'if p76 is None else f'{p76:.6f}')+' -> '+pred76,font=font(20),fill='#ffe091')
        d.text((1160,390),'Not a calibrated semantic E score.',font=font(18),fill='#ffc4bc')
        d.text((1160,421),'Saved self flag: '+str(fbool(r['geometry_self_hidden'])),font=font(18),fill='#c9d2dd')
        d.text((16,510),f'q cyan={q.round(4).tolist()} | reference magenta={ref.round(4).tolist()} | q-r={case["error_px"]:.4f}px | L={L:.4f}px',font=font(18),fill='#dddddd')
        d.text((16,540),'Coordinate source: '+str(case['annotation_coordinate_source'])+' | reference origin: '+str(case['reference_coordinate_origin']),font=font(17),fill='#dddddd')
        d.text((16,570),'Semantic reference is GT display only; pixel GT unavailable. No q/L/reference movement, attention/localization, model forward, fit or PnP.',font=font(17),fill='#dddddd')
        png=OUT/f'CASE_{n+1}_{case["human_state_short"]}.png';canvas.save(png);panels.append(canvas)
        cases.append(dict(id=key[0],native_corner=key[1],human_semantic_state=case['human_state_short'],fixed_original_case_role=case['fixed_case_role'],
            query=q.tolist(),reference=ref.tolist(),roi_side=L,error_px=case['error_px'],annotation_coordinate_source=case['annotation_coordinate_source'],
            reference_coordinate_origin=case['reference_coordinate_origin'],reference_overlay_exposure=case['reference_overlay_exposure'],reference_source=case['reference_source'],
            v61_p_visible=p61,v61_prediction=pred61,v72_p_D_given_DE=p72,v72_prediction=pred72,
            v76_p_local_support_visible=p76,v76_raw_p_local_support_visible=raw76,v76_prediction=pred76,v76_valid=valid76,
            v76_saved_semantic_compatibility_label=r['predicted_D_or_E'],v76_geometry_self_hidden=fbool(r['geometry_self_hidden']),v76_geometry_self_hidden_if_valid=fbool(r['geometry_self_hidden_if_valid']),
            saved_row_index=i,raw_full_L_RGB96=b(rawpath),raw_valid96=b(validpath),panel=b(png),
            entire_L_preserved=True,all_RGB_and_valid_exact_vs_original_PIL_and_V72=True,pixel_GT='UNAVAILABLE',GT_markers_forwarded=False))
    sheet=Image.new('RGB',(1680,610*len(panels)),'black')
    for i,panel in enumerate(panels):sheet.paste(panel,(0,i*610))
    sheet.save(OUT/'CONTACT_SHEET.png')
    for state,indices in [('D',range(4)),('E',range(4,6))]:
        ids=list(indices);fig=Image.new('RGB',(1680,610*len(ids)),'black')
        for j,i in enumerate(ids):fig.paste(panels[i],(0,j*610))
        fig.save(OUT/f'FIXED_{state}_FULL_L_CROPS.png')
    save(OUT/'FIXED_CURRENT_CROP_METADATA.json',dict(completed=True,selection='Exact existing six V66/V72 cases, no score reselection',
        inference_seal=b(newsealpath),execution_lock=b(OUT/'SELECTION_AND_EXECUTION_LOCK_BEFORE_RGB.json'),cases=cases,
        crop_validation='Every selected full96x96RGBx3 andvalidpixel exact to originalPIL bilinear, V72savedRGB96, andV76reusedRGB96.',
        RGB_body_reads=6,model_backbone_fit_PnP_calls=0,GT_masks_read=0,old_files_modified=False,
        score_scope='V61savedpV, V72semanticreferenceD/Econditional, V76reference-local support weakteacher. Their negatives do not prove external cause.',
        new_native_semantic_or_pixel_groundtruth=False,contact_sheet=b(OUT/'CONTACT_SHEET.png')))
    save(OUT/'SEAL.json',dict(completed=True,code=b(Path(__file__)),inference_seal=b(newsealpath),outputs=[b(p)for p in sorted(OUT.rglob('*'))if p.is_file()and p.name!='SEAL.json']))
    print(json.dumps({'completed':True,'metadata':b(OUT/'FIXED_CURRENT_CROP_METADATA.json'),'seal':b(OUT/'SEAL.json')}))
if __name__=='__main__':main()
