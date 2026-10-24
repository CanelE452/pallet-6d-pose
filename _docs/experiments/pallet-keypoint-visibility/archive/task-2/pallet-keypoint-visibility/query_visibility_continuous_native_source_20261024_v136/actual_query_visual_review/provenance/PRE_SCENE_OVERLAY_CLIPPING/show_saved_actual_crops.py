"""View evaluator-selected V136 cases using exact already-saved RGB96 arrays.

No crop resampling, GT recrop, model, fitting, private input or selection search.
Full-image overlays are separate scientific views; exported RGB96 files are
lossless copies of saved model-input crops and have no overlay.
"""
from pathlib import Path
import argparse,collections,hashlib,html,json,sys,tempfile,time,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent
EV=V/'saved_frozen_visibility_evaluation_v1';C16=V/'frozen_C16_actual_initializer_v1'
OUT=H/'saved_gallery_v1';VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class Crops:
    def __init__(self):self.cache=collections.OrderedDict()
    def get(self,row,index):
        b=row['rgb96_chunk'];key=(b['path'],b['sha256'])
        if key not in self.cache:
            path=check(b)
            with np.load(path,allow_pickle=False)as f:pack={k:f[k].copy()for k in('rgb96','valid96','crop_constructed','global_index')}
            assert pack['rgb96'].dtype==np.uint8 and pack['valid96'].dtype==pack['crop_constructed'].dtype==bool
            assert pack['rgb96'].shape[1:]==(96,96,3)and pack['valid96'].shape[1:]==(96,96)
            self.cache[key]=pack
            if len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(key);p=self.cache[key];j=row['rgb96_row']
        assert int(p['global_index'][j])==index and bool(p['crop_constructed'][j])==row['crop_constructed']
        rgb,valid=p['rgb96'][j],p['valid96'][j]
        assert array_sha(rgb)==row['rgb96_sha256']and array_sha(valid)==row['valid96_sha256']
        if not row['crop_constructed']:assert not rgb.any()and not valid.any()and not row['feature_constructed']
        return rgb.copy(),valid.copy()

def font(size):
    from PIL import ImageFont
    p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(p),size)if p.exists()else ImageFont.load_default()

def one_sheet(key,cases,scores,facts,crops):
    from PIL import Image,ImageDraw
    base,slot=key;cols=512;top=158;scene_y=top;crop_y=scene_y+512+44;mask_y=crop_y+288+36
    canvas=Image.new('RGB',(cols*4,mask_y+210),(246,247,249));d=ImageDraw.Draw(canvas)
    labels=' | '.join(f"{c['global_index']}:{c['category']}"for c in cases)
    d.text((12,8),f'V136 base {base} / original native channel {slot} | exact saved C16 crop96',font=font(22),fill=(15,20,28))
    d.text((12,38),'MAGENTA = actual initializer q. CYAN = certified CF-primary teacher GT anchor. No coordinate correction.',font=font(17),fill=(15,20,28))
    d.text((12,64),'Full scenes have overlays; C16 RGB96/valid96 are copied exactly. v61 uses its saved P3 ROI, not RGB96. No GT recrop.',font=font(17),fill=(15,20,28))
    d.text((12,91),labels[:225],font=font(13),fill=(25,30,38))
    source=[]
    for vi,variant in enumerate(VARIANTS):
        index=(base*4+vi)*8+slot;r=scores[index];f=facts[index]
        assert r['global_index']==f['occurrence_index']==index and r['variant']==f['variant']==variant
        assert r['query_id']==f['occurrence_id']and r['query_xy']==f['query_xy']and r['roi_side']==f['L']
        assert r['base_index']==f['base_index']==base and r['native_slot']==f['native_slot']==slot
        assert r['raw_RGB_PNG']==f['raw_RGB_PNG']and r['split']==f['split']=='VALIDATION'
        png=check(r['raw_RGB_PNG']);image=Image.open(png).convert('RGB');full=np.asarray(image)
        assert full.shape==(512,512,3)and array_sha(full)==f['decoded_RGB_sha256']
        raw,valid=crops.get(f,index);rp=OUT/f'EXACT_SAVED_RGB96_{index:05d}.png';vp=OUT/f'EXACT_SAVED_VALID96_{index:05d}.png'
        if not rp.exists():Image.fromarray(raw).save(rp)
        if not vp.exists():Image.fromarray(valid.astype(np.uint8)*255).save(vp)
        assert array_sha(np.asarray(Image.open(rp).convert('RGB')))==f['rgb96_sha256']
        assert np.array_equal(np.asarray(Image.open(vp))>0,valid)
        x0=cols*vi;canvas.paste(image,(x0,scene_y));ref=r['reference_xy'];q=r['query_xy'];error=r['error_px'];L=r['roi_side']
        d.text((x0+8,117),f'{variant} | teacher {r["raw_state"]} | error '+('null'if error is None else f'{error:.3f}px')+' | L '+('null'if L is None else f'{L:.3f}'),font=font(14),fill=(15,20,28))
        def score(model):
            v=r[model]['raw_logit'];return r[model]['prediction']+(' (null)'if v is None else f' ({v:.4f})')
        d.text((x0+8,138),f'C16 hidden {score("C16")} | v61 -visible {score("v61")}',font=font(14),fill=(15,20,28))
        for xy,color in((ref,(0,235,255)),(q,(255,0,210))):
            if xy is not None:
                x,y=xy;px=x0+x;py=scene_y+y;d.ellipse((px-7,py-7,px+7,py+7),outline=(0,0,0),width=4);d.ellipse((px-7,py-7,px+7,py+7),outline=color,width=2)
        if q is not None:d.line((x0+q[0],scene_y+q[1],x0+ref[0],scene_y+ref[1]),fill=(255,255,255),width=1)
        d.text((x0+8,crop_y-32),'EXACT saved RGB96 x3; constructed='+str(f['crop_constructed'])+', available='+str(r['C16']['available']),font=font(14),fill=(15,20,28))
        canvas.paste(Image.fromarray(raw).resize((288,288),Image.Resampling.NEAREST),(x0+112,crop_y))
        d.text((x0+8,mask_y-28),f'EXACT saved valid96 x2; valid fraction={valid.mean():.4f}',font=font(14),fill=(15,20,28))
        canvas.paste(Image.fromarray(valid.astype(np.uint8)*255).convert('RGB').resize((192,192),Image.Resampling.NEAREST),(x0+160,mask_y))
        source.append(dict(global_index=index,variant=variant,score_row=r,exact_saved_RGB96=bind(rp),exact_saved_valid96=bind(vp),
            source_crop_chunk=f['rgb96_chunk'],source_crop_row=f['rgb96_row'],RGB96_array_sha256=f['rgb96_sha256'],valid96_array_sha256=f['valid96_sha256'],
            crop_constructed=f['crop_constructed'],feature_constructed=f['feature_constructed'],full_RGB_PNG=r['raw_RGB_PNG'],model_crop_GT_recrop=False))
    path=OUT/f'BASE_{base:04d}_NATIVE_{slot}_ACTUAL_QUERY_AND_SAVED_CROPS.png';canvas.save(path)
    return dict(base_index=base,native_slot=slot,selected_cases=cases,sheet=bind(path),all4variants=source)

def run():
    start=time.monotonic();assert not OUT.exists(),'Preserve existing gallery; no replacement.'
    sealpath=EV/'EVALUATION_SEAL.json';seal=read(sealpath)
    assert seal['completed']and seal['PASS']and seal['RGB_read']is False and seal['source_only']
    results=read(check(seal['results']));selection=read(check(seal['selection']))
    assert results['completed']and results['PASS']and results['original_slots']==20480 and results['original_frames']==2560
    assert results['primary_convention']=='CF_PRIMARY_DECLARED'and results['CF_normal_facing_source_order_certified']
    assert selection['PASS']and selection['RGB_read']is False and selection['labels_or_coordinates_changed']is False
    assert selection['selection_for_diagnosis_only']and selection['obsolete_area_disagreement_is_annotation_only']and selection['CF_primary_all_configs_certified']
    cases=selection['cases'];assert len(cases)<=8 and len({r['global_index']for r in cases})==len(cases)
    scorefile=check(seal['scores']);scores=[json.loads(s)for s in scorefile.read_text().splitlines()if s];assert len(scores)==20480
    for i,r in enumerate(scores):assert r['global_index']==i and r['scene_index']==i//8 and r['native_slot']==i%8
    predseal=read(check(seal['C16_prediction_seal']));assert predseal['completed']and predseal['PASS']and predseal['GT_read']is False
    assert predseal['all2560_scenes20480_slots_preserved']and predseal['actual_original_initializer']and not predseal['source_emulation']
    factsbinding=predseal['input_crop_facts'];fp=read(check(factsbinding));facts=fp['rows'];assert fp['completed']and fp['GT_read']is False and len(facts)==20480
    groups=collections.OrderedDict()
    for case in cases:
        r=scores[case['global_index']];assert r['base_index']==case['base_index']and r['native_slot']==case['native_slot']and r['split']=='VALIDATION'
        groups.setdefault((case['base_index'],case['native_slot']),[]).append(case)
    OUT.mkdir();write(OUT/'SELECTION_AND_INPUTS_BEFORE_RGB_READ.json',dict(completed=True,code=bind(__file__),evaluation_seal=bind(sealpath),selection=seal['selection'],scores=seal['scores'],C16_prediction_seal=seal['C16_prediction_seal'],crop_facts=factsbinding,
        exact_selected_cases=cases,grouping_only='Same base/native slot cases share one4variant sheet; no selected case dropped.',RGB_or_crops_read=0))
    cropcache=Crops();sheets=[one_sheet(key,cs,scores,facts,cropcache)for key,cs in groups.items()]
    assert sum(len(s['selected_cases'])for s in sheets)==len(cases)
    lines=['<!doctype html><meta charset="utf-8"><title>V136 exact saved actual-query crops</title><style>body{font:17px sans-serif;margin:24px;color:#15202b;background:#f5f7fa}img{max-width:100%;height:auto}section{margin:28px 0}code{font-size:14px}</style>',
        '<h1>V136: actual q and exact saved RGB96</h1><p>Magenta=q, cyan=teacher GT. RGB96 uses stored model-input pixels; no GT recrop. All selected CF-primary cases and four same-base variants are preserved. This is synthetic source diagnosis, not real application accuracy.</p>']
    for sheet in sheets:
        lines.append('<section><h2>Base '+str(sheet['base_index'])+', native '+str(sheet['native_slot'])+'</h2><p>'+html.escape(' | '.join(c['category']for c in sheet['selected_cases']))+'</p><a href="'+html.escape(Path(sheet['sheet']['path']).name)+'"><img src="'+html.escape(Path(sheet['sheet']['path']).name)+'"></a></section>')
    index=OUT/'INDEX.html';index.write_text('\n'.join(lines)+'\n')
    result=dict(completed=True,PASS=True,code=bind(__file__),evaluation_seal=bind(sealpath),selection=seal['selection'],C16_prediction_seal=seal['C16_prediction_seal'],source_crop_facts=factsbinding,
        selected_case_count=len(cases),grouped_sheet_count=len(sheets),all_selected_cases_retained=True,index=bind(index),sheets=sheets,
        exact_saved_model_RGB96_and_valid_masks=True,RGB96_GT_recrop=False,new_crop_or_modelcalls=0,raw_RGB_unmodified=True,private_V112_PnP_fit=0,seconds=time.monotonic()-start)
    write(OUT/'GALLERY_COMPLETE.json',result)
    print(json.dumps(dict(completed=True,PASS=True,result=bind(OUT/'GALLERY_COMPLETE.json'),index=bind(index),selected_cases=len(cases),sheets=len(sheets))),flush=True)

def prepare():
    # Invented saved-chunk checks: exact copies, noquery placeholders and
    # corrupt-row rejection. No actual evaluation, RGB, mask or score reads.
    raw=np.zeros((2,96,96,3),np.uint8);raw[0]=np.arange(96*96*3,dtype=np.int32).reshape(96,96,3)%251
    valid=np.zeros((2,96,96),bool);valid[0,3:91,7:84]=True
    with tempfile.TemporaryDirectory()as temp:
        file=Path(temp)/'INVENTED_CROP_CHUNK.npz';np.savez_compressed(file,rgb96=raw,valid96=valid,crop_constructed=np.array([True,False]),global_index=np.array([42,43],np.int32))
        rows=[dict(rgb96_chunk=bind(file),rgb96_row=i,crop_constructed=i==0,feature_constructed=i==0,rgb96_sha256=array_sha(raw[i]),valid96_sha256=array_sha(valid[i]))for i in range(2)]
        cache=Crops();a,b=cache.get(rows[0],42);assert np.array_equal(a,raw[0])and np.array_equal(b,valid[0])
        a[:]=0;b[:]=False;aa,bb=cache.get(rows[0],42);assert np.array_equal(aa,raw[0])and np.array_equal(bb,valid[0])
        a,b=cache.get(rows[1],43);assert not a.any()and not b.any()
        for bad,index in((dict(rows[0],rgb96_sha256='0'*64),42),(rows[0],43)):
            try:cache.get(bad,index)
            except AssertionError:pass
            else:raise AssertionError('Corrupt crop hash/index accepted')
    write(H/'PREPARATION.json',dict(completed=True,prepared_only=True,code=bind(__file__),expected_evaluation_seal=str(EV/'EVALUATION_SEAL.json'),expected_C16_crop_facts_seal=str(C16/'PREDICTION_SEAL.json'),
        selection='Use exact completed evaluator selection; up to8cases, all certifiedCF primary including obsolete-area-disagreeing rows. Group same base/native slot only.',
        view='All4samebase variants: full512 actualq magenta/GTcyan, exact saved C16 RGB96 no overlay, exact saved valid96. v61 consumes P3 ROI. No GT recrop.',
        generated_invented_chunk_check=dict(PASS=True,exact_RGB_and_mask_copy=True,returned_arrays_copy_safe=True,noquery_zero_placeholder_distinct=True,corrupt_hash_and_wrong_global_index_rejected=True),actual_evaluation_scores_GT_RGB_model_reads=0,
        run_after_root_evaluation_complete='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).resolve())+' --run',output=str(OUT)))
    print(json.dumps(dict(prepared_only=True,preparation=bind(H/'PREPARATION.json'))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    try:
        if a.prepare and not a.run:prepare()
        elif a.run and not a.prepare:run()
        else:raise SystemExit('Use --prepare or --run after root reports evaluation complete')
    except BaseException:
        write(H/('FAILED_'+str(time.time_ns())+'.json'),dict(code=bind(__file__),traceback=traceback.format_exc(),source_files_mutated=False,automatic_retry=False));raise
