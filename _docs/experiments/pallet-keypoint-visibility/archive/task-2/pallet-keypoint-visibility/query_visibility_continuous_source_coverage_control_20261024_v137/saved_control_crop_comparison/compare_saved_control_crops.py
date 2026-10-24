"""Fixed root-selected V136 crops vs saved V137 scores; no model or recrop."""
from pathlib import Path
import collections,hashlib,html,json,sys,time
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
RUN=V/'runs/C28_C29_CONTINUOUS_COVERAGE_FIXED8000'
CONTROL=RUN/'CONTROL_COMPLETE.json'
GALLERY=P/'query_visibility_continuous_native_source_20261024_v136/actual_query_visual_review/saved_gallery_v1/GALLERY_COMPLETE.json'
ARMS=dict(C28='C28_OLD_NATIVE_EXTRA_REPLAY_CONTROL',C29='C29_CONTINUOUS_NATIVE_DATA_ADDITION')
FIXED=((519,0),(590,1));VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')

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
def pred(n):return{-1:'U',0:'D',1:'E'}[int(n)]
def font(n):
    from PIL import ImageFont
    p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(p),n)if p.exists()else ImageFont.load_default()

def run():
    start=time.monotonic();assert not(H/'COMPARISON_COMPLETE.json').exists()
    control=read(CONTROL);gallery=read(GALLERY)
    assert control['completed']and control['PASS']and control['selected_arm_source_only']is None
    assert control['final_update8000_both']and control['application_solved']is False and control['private_current_V112_PnP_q_correction_calls']==0
    assert gallery['completed']and gallery['PASS']and gallery['selected_case_count']==8 and gallery['all_selected_cases_retained']
    es=read(check(gallery['evaluation_seal']));assert es['completed']and es['PASS']and es['RGB_read']is False
    cases=read(check(gallery['selection']))['cases'];assert len(cases)==8
    scorefile=check(es['scores']);scores=[json.loads(x)for x in scorefile.read_text().splitlines()if x];assert len(scores)==20480
    oldseal=read(check(es['C16_prediction_seal']));assert oldseal['completed']and oldseal['PASS']and oldseal['GT_read']is False
    with np.load(check(oldseal['predictions']),allow_pickle=False)as f:
        old={k:f[k].copy()for k in('global_index','occurrence_id','query_xy','L','query_available','head_forwarded')}
    packs={};bindings={}
    for label,arm in ARMS.items():
        a=control['arms'][arm];assert a['source_eligible']is False
        fit=read(check(a['fit']));freeze=read(check(a['freeze']));assert fit['completed']and fit['optimizer_updates']==8000 and fit['arm']==arm
        report=read(check(fit['results']));assert report['completed']and report['selected_update']==8000 and report['threshold_logit']==0 and report['source_eligible']is False
        assert freeze['completed']and freeze['update']==8000 and freeze['threshold_logit']==0 and freeze['source_eligible']is False
        path=RUN/arm/'FINAL_NEW_NATIVE20480_SCORES_INPUT_AND_OUTPUT.npz';bindings[label]=bind(path)
        with np.load(path,allow_pickle=False)as f:pack={k:f[k].copy()for k in f.files}
        for k in('global_index','occurrence_id','query_xy','L','query_available','head_forwarded'):
            assert np.array_equal(pack[k],old[k],equal_nan=True)if pack[k].dtype.kind in'fc'else np.array_equal(pack[k],old[k]),(label,k)
        raw=pack['raw_head_logit_before_U'];av=pack['query_available'];fw=pack['head_forwarded']
        assert raw.shape==av.shape==fw.shape==(20480,)and raw.dtype==np.float32 and av.dtype==fw.dtype==bool
        assert np.array_equal(pack['global_index'],np.arange(20480))and np.isfinite(raw[fw]).all()and np.isnan(raw[~fw]).all()
        assert np.array_equal(pack['prediction'],np.where(av,(raw>=0).astype(np.int8),-1))
        assert np.array_equal(pack['query_logit'],np.where(av,raw,0),equal_nan=True)
        assert np.array_equal(pack['raw_teacher_state_CF'],np.array([r['raw_state']for r in scores]))
        assert np.array_equal(pack['split'],np.array([r['split']for r in scores]))
        assert np.array_equal(pack['external_target_CF'],np.array([0 if r['raw_state']=='D'else 1 if r['raw_state']=='E'else -1 for r in scores]))
        assert np.array_equal(pack['convention_boundary'],np.array([r['convention_boundary']for r in scores]))
        for i,r in enumerate(scores):
            assert r['global_index']==i and r['query_id']==str(pack['occurrence_id'][i])
            q=np.asarray(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]);L=np.nan if r['roi_side']is None else r['roi_side']
            assert np.array_equal(q,pack['query_xy'][i],equal_nan=True)and(np.isnan(L)and np.isnan(pack['L'][i])or L==pack['L'][i])
        packs[label]=pack
    sources={(s['base_index'],s['native_slot']):s for s in gallery['sheets']}
    assert all(key in sources for key in FIXED)
    write(H/'INPUT_BINDINGS_BEFORE_CROP_READ.json',dict(completed=True,code=bind(__file__),control_complete=bind(CONTROL),source_gallery=bind(GALLERY),source_evaluation=gallery['evaluation_seal'],
        exact_saved_scores=bindings,fixed_root_selected_base_slot_pairs=[list(k)for k in FIXED],original8case_selection=gallery['selection'],all20480_ID_q_L_availability_teacher_metadata_exact_join=True,
        new_model_error_selection=False,RGB_or_crops_read=0,model_calls=0))
    rows=[]
    for case in cases:
        i=case['global_index'];r=scores[i];z=dict(global_index=i,category=case['category'],base_index=r['base_index'],native_slot=r['native_slot'],variant=r['variant'],teacher=r['raw_state'],q_error_px=r['error_px'],L=r['roi_side'])
        for model in('C16','v61'):z[model]=r[model]
        for model,pack in packs.items():z[model]=dict(prediction=pred(pack['prediction'][i]),hidden_logit=float(pack['raw_head_logit_before_U'][i])if np.isfinite(pack['raw_head_logit_before_U'][i])else None,available=bool(pack['query_available'][i]))
        rows.append(z)
    from PIL import Image,ImageDraw
    strips=[]
    for key in FIXED:
        sheet=sources[key];base,slot=key;canvas=Image.new('RGB',(1360,528),(247,248,250));d=ImageDraw.Draw(canvas)
        d.text((12,8),f'V137 saved control comparison | base {base}, native {slot}',font=font(23),fill=(15,20,28))
        d.text((12,39),'Exact V136 saved C16 RGB96 at 3x. Same original q/L; no GT recrop or new model call.',font=font(17),fill=(15,20,28))
        d.text((12,65),'C28 and C29 both failed source admission; selected arm = none. Fixed examples, not prevalence.',font=font(17),fill=(160,25,25))
        variants=[]
        for vi,source in enumerate(sheet['all4variants']):
            assert source['variant']==VARIANTS[vi];r=source['score_row'];i=source['global_index'];assert i==(base*4+vi)*8+slot
            png=check(source['exact_saved_RGB96']);rgb=np.asarray(Image.open(png).convert('RGB'));assert rgb.shape==(96,96,3)and rgb.dtype==np.uint8 and array_sha(rgb)==source['RGB96_array_sha256']
            x=340*vi;d.text((x+10,94),f'{VARIANTS[vi]} | GT {r["raw_state"]} | err {r["error_px"]:.3f}px',font=font(16),fill=(15,20,28))
            d.text((x+10,117),f'L={r["roi_side"]:.3f} | RGB96 from slot {i}',font=font(14),fill=(35,40,48))
            canvas.paste(Image.fromarray(rgb).resize((288,288),Image.Resampling.NEAREST),(x+26,143))
            vm={}
            for mi,model in enumerate(('C16','v61','C28','C29')):
                if model in('C16','v61'):pr=r[model]['prediction'];z=r[model]['raw_logit']
                else:pr=pred(packs[model]['prediction'][i]);z=float(packs[model]['raw_head_logit_before_U'][i])
                color=(20,115,55)if pr==r['raw_state']else(170,30,30)
                d.text((x+10,438+20*mi),f'{model}: {pr}  h={z:+.4f}',font=font(16),fill=color)
                vm[model]=dict(prediction=pr,hidden_logit=z)
            variants.append(dict(global_index=i,variant=r['variant'],teacher=r['raw_state'],q_error_px=r['error_px'],L=r['roi_side'],actual_query_xy=r['query_xy'],teacher_anchor_xy=r['reference_xy'],saved_RGB96=source['exact_saved_RGB96'],RGB96_array_sha256=source['RGB96_array_sha256'],models=vm))
        path=H/f'BASE_{base:04d}_NATIVE_{slot}_FOUR_VARIANT_RGB96_COMPARISON.png';canvas.save(path)
        strips.append(dict(base_index=base,native_slot=slot,strip=bind(path),variants=variants))
    index=H/'INDEX.html';parts=['<!doctype html><meta charset="utf-8"><title>V137 fixed saved crop controls</title><style>body{font:16px sans-serif;color:#14202a;background:#f6f8fa;margin:24px}img{max-width:100%}td,th{padding:7px;border:1px solid #ccd3dc}table{border-collapse:collapse}</style><h1>V137 fixed saved crop comparison</h1><p>Both new arms are source-ineligible. No arm selected. Same exact saved V136 C16 crop pixels, original q and L. v61 uses P3 ROI. Hidden-logit zero means E for C16/C28/C29, D for v61. No threshold or case selection changed.</p>']
    for s in strips:parts.append('<h2>Base '+str(s['base_index'])+'</h2><img src="'+html.escape(Path(s['strip']['path']).name)+'">')
    parts.append('<h2>All original 8 fixed cases</h2><table><tr><th>slot</th><th>variant</th><th>GT</th><th>q error px</th>'+''.join('<th>'+m+'</th>'for m in('C16','v61','C28','C29'))+'</tr>')
    for r in rows:
        cells=[str(r['global_index']),r['variant'],r['teacher'],f'{r["q_error_px"]:.3f}']
        for model in('C16','v61','C28','C29'):
            z=r[model].get('raw_logit',r[model].get('hidden_logit'));cells.append(r[model]['prediction']+(' null'if z is None else f' {z:+.4f}'))
        parts.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>'for x in cells)+'</tr>')
    parts.append('</table><p>These cases were fixed by the original V136 evaluator and root before reading C28/C29 errors; they do not measure population performance.</p>');index.write_text('\n'.join(parts)+'\n')
    complete=dict(completed=True,PASS=True,code=bind(__file__),control_complete=bind(CONTROL),source_gallery=bind(GALLERY),source_evaluation=gallery['evaluation_seal'],new_saved_predictions=bindings,
        fixed_root_base_slot_pairs=[list(k)for k in FIXED],strips=strips,original8case_outcomes=rows,all8_original_cases_preserved=True,all20480_metadata_join_exact=True,
        source_eligible=dict(C28=False,C29=False),selected_arm_source_only=None,no_model_newcrop_or_selection_calls=True,source_RGB96_unchanged=True,index=bind(index),seconds=time.monotonic()-start)
    write(H/'COMPARISON_COMPLETE.json',complete);print(json.dumps(dict(completed=True,PASS=True,result=bind(H/'COMPARISON_COMPLETE.json'),index=bind(index),rows=rows)),flush=True)

if __name__=='__main__':run()
