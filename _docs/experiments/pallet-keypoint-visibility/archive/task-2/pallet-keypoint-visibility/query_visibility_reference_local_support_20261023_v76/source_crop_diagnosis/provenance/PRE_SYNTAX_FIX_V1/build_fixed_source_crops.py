"""Fixed source-only teacher/input gallery. No model/scores/current reads.

Select lexicographic(id,nativek) within each source×VAL×teacher class after
factual geometry admission, freeze selection before opening any actual RGB.
"""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
P=ROOT.parent
CORPUS=ROOT/'gt_corpus_build/GT_ONLY_TEACHER_CORPUS.npz'
INDEX=ROOT/'gt_corpus_build/ORDERED_METADATA_INDEX_GT_ONLY.jsonl'
SYNTH=ROOT/'source_inventory/UNCHANGED_MODEL_INPUT_LINKS.jsonl'
PUBLIC=ROOT/'public_query_emulation/INPUTS_RGB_ONLY.jsonl'
GRID=P.parent/'external_ray_proxy_head_20261022_v60/head_and_roi.py'
CROP=P/'query_visibility_loco_native_context_20261023_v74/native_context_cache/compile_input_pack.py'
BOX=P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/current_input_adapter.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return {'path':str(p),'sha256':sha(p)}
def lines(p):return [json.loads(x)for x in p.read_text().splitlines()]
def clean(v):
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,dict):return {str(k):clean(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x)for x in v]
    if isinstance(v,(np.integer,np.bool_)):return v.item()
    if isinstance(v,(float,np.floating)):return float(v)if np.isfinite(v)else None
    return v
def save(p,d):
    assert not p.exists();p.write_text(json.dumps(clean(d),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def pure(p,name):
    n=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
    d={'np':np};exec(compile(ast.Module(body=[n],type_ignores=[]),str(p),'exec'),d);return d[name]
def anycrop(q,L,hw,n):
    if not np.isfinite(q).all()or not np.isfinite(L)or L<=0:return False
    h,w=hw;u=((np.arange(n)+.5)/n-.5)*L
    return bool(np.any((q[0]+u>=-.5)&(q[0]+u<w-.5))and np.any((q[1]+u>=-.5)&(q[1]+u<h-.5)))
def qvalid(q,hw):
    h,w=hw;g=np.isfinite(q).all(1)&~(q==-1).all(1);px=np.floor(np.where(g[:,None],q,-1e8)+.5)
    return g&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)&(px[:,0]>=0)&(px[:,0]<w)&(px[:,1]>=0)&(px[:,1]<h)
def maskcrop(mask,q,L,n=96):
    h,w=mask.shape;u=((np.arange(n)+.5)/n-.5)*L;xx,yy=np.meshgrid(q[0]+u,q[1]+u)
    px=np.floor(xx+.5).astype(int);py=np.floor(yy+.5).astype(int)
    valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)&(px>=0)&(px<w)&(py>=0)&(py<h)
    result=np.zeros((n,n),bool);result[valid]=mask[py[valid],px[valid]];return result
def mark(draw,p,color,radius=5):
    x,y=map(float,p);draw.line((x-radius,y,x+radius,y),fill=color,width=2);draw.line((x,y-radius,x,y+radius),fill=color,width=2)
def scalar(rgb,q,L,n,iy,ix):
    h,w=rgb.shape[:2];x=q[0]+((ix+.5)/n-.5)*L;y=q[1]+((iy+.5)/n-.5)*L
    if not(-.5<=x<w-.5 and -.5<=y<h-.5):return np.zeros(3,np.uint8),False
    x0=int(np.floor(x));y0=int(np.floor(y));dx=x-x0;dy=y-y0;a=np.zeros(3)
    for ox,oy,wt in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        if 0<=x0+ox<w and 0<=y0+oy<h:a+=rgb[y0+oy,x0+ox].astype(float)*wt
    return np.floor(np.clip(a,0,255)+.5).astype(np.uint8),True

def main():
    assert not(OUT/'SELECTION_LOCK_BEFORE_RGB.json').exists()
    index=lines(INDEX);synth=lines(SYNTH);public=lines(PUBLIC)
    assert len(index)==5203 and len(synth)==1003 and len(public)==4200
    arrays=['id','source','split','label8','rule8','native_GTref8','GTref_valid_xy8','rho_px','sigma_px',
        'weighted_expected_mass_A8','weighted_visible_mass_V8','visible_over_expected_ratio8','expected_support_pixels_A8',
        'visible_support_pixels_V8','near_boundary8','convex_self_proxy8_separate']
    with np.load(CORPUS,allow_pickle=False)as z:gt={k:z[k]for k in arrays}
    grid=pure(GRID,'native_grid');boxfun=pure(BOX,'initial_query_box');crop=pure(CROP,'crop')
    candidates=[];source_rows=[]
    for i,row in enumerate(index):
        if i<1003:
            inp=synth[i];assert inp['id']==row['id']
            q=np.asarray(inp['unchanged_initial_query9'],float)[:8];L=float(inp['input_L']);hw=inp['raw_RGB']['raw_hw']
            image=inp['raw_RGB']['path'];image_sha=inp['raw_RGB']['sha256'];producer='Original sealed N3 synthetic initializer'
        else:
            inp=public[i-1003];assert inp['id']==row['id']
            q=np.asarray(inp['initial_points'],float);L=float(inp['roi_side']);hw=inp['raw_hw'];image=inp['image'];image_sha=inp['image_sha256']
            producer='GT-derived source emulation from correlated SYNTH TRAIN residual; NOT actual deployment initializer'
        assert row['id']==str(gt['id'][i])and row['source']==str(gt['source'][i])and row['split']==str(gt['split'][i])
        _,gv,L2=grid(q,hw);assert L==L2
        box=boxfun(q,hw);roi=gv.reshape(8,-1).any(1);qv=qvalid(q,hw);local=np.asarray([anycrop(p,L,hw,96)for p in q])
        tv=False if box is None else anycrop((box[:2]+box[2:])/2,1.25*float(max(box[2:]-box[:2])),hw,128)
        refs=gt['native_GTref8'][i];off=(refs-q)/L
        inside=np.isfinite(refs).all(1)&((off>=-.5)&(off<.5)).all(1)
        admit=np.isin(gt['label8'][i],[0,1])&qv&roi&local&tv&inside
        source_rows.append({'i':i,'q':q,'L':L,'hw':hw,'image':image,'image_sha256':image_sha,'producer':producer,'admit':admit})
        if row['split']=='val':
            for k in np.flatnonzero(admit):candidates.append({'id':row['id'],'nativek':int(k),'source':row['source'],'teacher':int(gt['label8'][i,k]),'corpus_row':i})
    chosen=[];counts={}
    for source in ['SYNTHETIC','YCB_PUBLIC_REAL']:
        for state,code in [('VISIBLE',1),('COVERED',0)]:
            cohort=sorted([c for c in candidates if c['source']==source and c['teacher']==code],key=lambda x:(x['id'],x['nativek']))
            assert cohort
            counts[source+':'+state]=len(cohort)
            chosen.append({**cohort[0],'state':state})
    save(OUT/'SELECTION_LOCK_BEFORE_RGB.json',{'completed':True,'selection':'VAL, knownteacher+factualgeometryadmission only; lexicographic(id,nativek) first per source×VISIBLE/COVERED. No scores/weights/model outputs read.',
        'chosen':chosen,'eligible_VAL_slots':counts,'bindings':[b(p)for p in [CORPUS,INDEX,SYNTH,PUBLIC,GRID,CROP,BOX,Path(__file__)]],
        'RGB_or_masks_opened_before_lock':0,'all_rows_retained_in_upstream_corpus':True})
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',19)
    cases=[];tiles=[]
    for n,c in enumerate(chosen):
        i=c['corpus_row'];k=c['nativek'];row=index[i];inp=source_rows[i];q=inp['q'][k];r=gt['native_GTref8'][i,k];L=inp['L'];h,w=inp['hw']
        assert sha(inp['image'])==inp['image_sha256']
        rgb=np.asarray(Image.open(inp['image']).convert('RGB'));assert list(rgb.shape[:2])==[h,w]
        raw0=np.asarray(Image.open(row['M0_path']));raw4=np.asarray(Image.open(row['M4_path']));threshold=127 if row['source']=='SYNTHETIC'else0
        m0=raw0>threshold;m4=raw4>threshold;assert not np.any(m4&~m0)and m0.shape==m4.shape==(h,w)
        patch,valid=crop(rgb,q,L,96);patch=patch.transpose(1,2,0)
        for iy,ix in [(0,0),(47,47),(48,48),(95,95),(15,71),(72,24)]:
            px,v=scalar(rgb,q,L,96,iy,ix);np.testing.assert_array_equal(px,patch[iy,ix]);assert v==valid[0,iy,ix]
        base=f'{n+1}_{row["source"]}_{c["state"]}_k{k}'
        rawpath=OUT/(base+'_RAW_INPUT_RGB96.png');Image.fromarray(patch).save(rawpath)
        vpath=OUT/(base+'_RAW_INPUT_VALID96.png');Image.fromarray(valid[0].astype(np.uint8)*255).save(vpath)
        mc0=maskcrop(m0,q,L);mc4=maskcrop(m4,q,L)
        Image.fromarray(mc0.astype(np.uint8)*255).save(OUT/(base+'_GT_M0_DISPLAY_NEAREST96.png'))
        Image.fromarray(mc4.astype(np.uint8)*255).save(OUT/(base+'_GT_M4_DISPLAY_NEAREST96.png'))
        overlay=patch.copy().astype(float)
        overlay[mc0]=.55*overlay[mc0]+.45*np.array([0,220,255]);overlay[mc4]=.5*overlay[mc4]+.5*np.array([60,255,60])
        overlay=Image.fromarray(np.floor(np.clip(overlay,0,255)+.5).astype(np.uint8)).resize((288,288),Image.Resampling.NEAREST)
        od=ImageDraw.Draw(overlay);mark(od,[143.5,143.5],(255,255,0),8)
        local_r=((r-q)/L+.5)*96-.5;mark(od,local_r*3+1,(255,0,220),8)
        full=Image.fromarray(rgb);scale=min(480/w,288/h);full=full.resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
        fd=ImageDraw.Draw(full);fd.rectangle([(q-L/2)*scale,(q+L/2)*scale],outline=(255,255,0),width=2);mark(fd,q*scale,(255,255,0),5);mark(fd,r*scale,(255,0,220),5)
        tile=Image.new('RGB',(1500,530),'#111820');d=ImageDraw.Draw(tile)
        d.text((16,10),f'{n+1}. {row["source"]} | {c["state"]} | {row["id"]} native{k}',fill='white',font=bold)
        tile.paste(full,(16,70));tile.paste(Image.fromarray(patch).resize((288,288),Image.Resampling.NEAREST),(510,70));tile.paste(overlay,(814,70))
        tile.paste(Image.fromarray(mc0.astype(np.uint8)*255).convert('RGB').resize((144,144),Image.Resampling.NEAREST),(1120,70))
        tile.paste(Image.fromarray(mc4.astype(np.uint8)*255).convert('RGB').resize((144,144),Image.Resampling.NEAREST),(1290,70))
        for x,title in [(16,'Source RGB / q crop square'),(510,'Exact model RGB96 (enlarged)'),(814,'GT overlay: M0 cyan / M4 green'),(1120,'M0'),(1290,'M4')]:d.text((x,42),title,fill='#cccccc',font=font)
        ratio=float(gt['visible_over_expected_ratio8'][i,k]);rho=float(gt['rho_px'][i]);selfv=int(gt['convex_self_proxy8_separate'][i,k]);rule=str(gt['rule8'][i,k])
        qp=np.floor(q+.5).astype(int);rp=np.floor(r+.5).astype(int)
        centerq=[bool(m0[qp[1],qp[0]]),bool(m4[qp[1],qp[0]])];centerr=[bool(m0[rp[1],rp[0]]),bool(m4[rp[1],rp[0]])]
        text=[f'q yellow={q.round(3).tolist()} | GT r magenta={r.round(3).tolist()} | L={L:.3f} | (r-q)/L={((r-q)/L).round(3).tolist()}',
              f'Teacher {rule}; rho={rho:.3f}px; V/A={ratio:.6f}; weighted A={gt["weighted_expected_mass_A8"][i,k]:.3f}, V={gt["weighted_visible_mass_V8"][i,k]:.3f}',
              f'Convex self-visible flag={selfv} (1=may-visible, 0=self-hidden; kept separate). q-center M0/M4={centerq}; r-center={centerr}',
              inp['producer'], 'Weak reference-local expected support teacher; NOT human semantic D/E/S or a query-pixel target. No model scores used.']
        for j,t in enumerate(text):d.text((16,382+j*27),t,fill='#dddddd',font=font)
        png=OUT/(base+'_DIAGNOSIS.png');tile.save(png);tiles.append(tile)
        cases.append({**c,'RGB':b(Path(inp['image'])),'M0':b(Path(row['M0_path'])),'M4':b(Path(row['M4_path'])),
            'query8_source':inp['producer'],'q':q,'GTref_r':r,'L':L,'offset_over_L':(r-q)/L,'teacher_rule':rule,'rho_px':rho,'ratio':ratio,
            'weighted_expected_mass_A':gt['weighted_expected_mass_A8'][i,k],'weighted_visible_mass_V':gt['weighted_visible_mass_V8'][i,k],
            'expected_pixels_A':gt['expected_support_pixels_A8'][i,k],'visible_pixels_V':gt['visible_support_pixels_V8'][i,k],
            'self_convex_visible_proxy_separate':selfv,'query_center_M0_M4':centerq,'reference_center_M0_M4':centerr,
            'admission_geometry_all':True,'raw_crop':b(rawpath),'raw_valid':b(vpath),'diagnosis':b(png),'GT_masks_used_as_model_input':False})
    sheet=Image.new('RGB',(1500,530*len(tiles)),'black')
    for i,t in enumerate(tiles):sheet.paste(t,(0,i*530))
    sheet.save(OUT/'CONTACT_SHEET.png')
    save(OUT/'FIXED_SOURCE_CROP_DIAGNOSIS.json',{'completed':True,'cases':cases,'selection_lock':b(OUT/'SELECTION_LOCK_BEFORE_RGB.json'),
        'no_model_score_selection':True,'source_RGB_read_count':4,'source_mask_pair_read_count':4,'source_current_or_model_score_reads':0,
        'actual_model_raw_crop_formula':'Same bound V74 bilinear crop96. RGB+valid saved separately without annotations;6scalarpixels/case exact0diff.',
        'display_masks':'Nearest nativepixel at same cropcoordinate grid, GT-only visualization, not classifier input.',
        'human_semantic_success_claim':False,'PnP_fit_forward':0,'contact_sheet':b(OUT/'CONTACT_SHEET.png')})
    save(OUT/'SEAL.json',{'completed':True,'code':b(Path(__file__)),'outputs':[b(p)for p in sorted(OUT.iterdir())if p.is_file()and p.name!='SEAL.json']})
    print(json.dumps({'completed':True,'chosen':chosen,'seal':b(OUT/'SEAL.json')}),flush=True)
if __name__=='__main__':main()
