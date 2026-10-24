"""Two preselected scientific contact strips from exact existing saved RGB96.

Preparation reads existing metadata only. RGB and V139 saved outputs are read
only after a separate root entry binds completed control and diagnosis seals.
No model, recrop, feature, checkpoint, private data or new case selection.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import sys
import time
import numpy as np
sys.dont_write_bytecode = True
H=Path(__file__).resolve().parent; V=H.parent
POLICY=V/'ROOT_BROAD_SOURCE_GEOMETRY_POLICY_BEFORE_FIT.json'
POLICY_SHA='b1c38b4110f90ada399fe71c15fb29cc931b0adeb852ed920b2b1556dfb00415'
REGISTRY=V/'saved_result_preparation/FIXED_CASES_BEFORE_NEW_SCORES.json'
DIAGNOSIS=V/'saved_geometry_control_diagnosis_v1/DIAGNOSIS_SEAL.json'
CONTROL=V/'runs/C30_C31_BROAD_SOURCE_GEOMETRY_FIXED8000/CONTROL_COMPLETE.json'
FIXED_JSON=V/'saved_geometry_control_diagnosis_v1/FIXED8_CASES_AND_TWO_STRIPS.json'
OUT=V/'saved_geometry_control_visual_report_v1'
PROTOCOL=H/'PROTOCOL_BEFORE_RGB_AND_NEW_SCORES.json'
GENERATED=H/'GENERATED_METADATA_CHECKS.json'
FIXED=((519,0),(590,1)); VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
MODELS=('C16','v61','C29','C30','C31')
FORBIDDEN_FLAGS=('model_forward','fit','new_crop','new_case_selection','current223','V112','PnP','q_L_correction')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def same(a,b):return a['path']==b['path'] and a['sha256']==b['sha256']
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def decode(model,z,available):
    if not available:return 'U'
    assert z is not None and np.isfinite(z)
    return ('E' if z>0 else 'D') if model=='v61' else ('E' if z>=0 else 'D')


def validate_rows(fixed,registry):
    assert fixed['completed'] and fixed['PASS'] and fixed['all8_preserved'] and fixed['no_new_case_selection']
    assert same(fixed['registry'],bind(REGISTRY)) and fixed['RGB_body_reads']==0
    assert registry['fixed_base_slot_pairs']==[list(x) for x in FIXED]
    assert [c['global_index'] for c in fixed['cases']]==registry['fixed8_indices']
    assert len(fixed['strips'])==2
    for (base,slot),strip,old in zip(FIXED,fixed['strips'],registry['strips']):
        assert strip['base_index']==old['base_index']==base and strip['native_slot']==old['native_slot']==slot
        assert len(strip['variants'])==len(old['variants'])==4
        for vi,(row,original) in enumerate(zip(strip['variants'],old['variants'])):
            assert row['global_index']==original['global_index']==(base*4+vi)*8+slot
            assert row['variant']==original['variant']==VARIANTS[vi]
            for key in ('teacher','q_error_px','L','actual_query_xy','teacher_anchor_xy','saved_RGB96','RGB96_array_sha256'):
                assert row[key]==original[key],key
            assert tuple(row['models'])==MODELS
            for model in MODELS:
                score=row['models'][model]
                assert decode(model,score['hidden_logit'],score['available'])==score['prediction']
                if model in ('C16','v61','C29'):
                    assert score['prediction']==original['models'][model]['prediction']
                    assert score['hidden_logit']==original['models'][model]['hidden_logit']
            assert row['models']['C30']['prediction']==row['models']['C29']['prediction']
            assert row['models']['C30']['hidden_logit']==row['models']['C29']['hidden_logit']


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def run(entry):
    e=read(entry);assert e['approved_saved_visual_report'] is True
    assert all(e[k] is False for k in FORBIDDEN_FLAGS)
    assert sha(POLICY)==POLICY_SHA and same(e['code'],bind(__file__)) and same(e['protocol'],bind(PROTOCOL))
    p=read(PROTOCOL)
    for key in ('code','policy','generated','fixed_registry'):
        assert same(e[key],p[key]);check(p[key])
    assert read(GENERATED)['PASS'] and not OUT.exists()
    assert Path(e['control_complete']['path'])==CONTROL and Path(e['diagnosis_seal']['path'])==DIAGNOSIS
    control=read(check(e['control_complete']));diagnosis=read(check(e['diagnosis_seal']))
    assert control['completed'] and control['PASS'] and control['final_update8000_both']
    assert control['private_current_V112_PnP_q_correction_calls']==0
    assert diagnosis['completed'] and diagnosis['PASS'] and diagnosis['all26624_preserved']
    assert diagnosis['RGB_read'] is False and diagnosis['no_new_case_selection']
    assert same(diagnosis['control_complete'],e['control_complete'])
    assert Path(diagnosis['fixed_cases']['path'])==FIXED_JSON
    fixed=read(check(diagnosis['fixed_cases']));registry=read(check(p['fixed_registry']))
    validate_rows(fixed,registry)
    results=read(check(diagnosis['results']));assert results['completed'] and results['PASS'] and results['C30_all_three_saved_score_packets_bitexact_C29']
    OUT.mkdir();start=time.monotonic()
    write(OUT/'INPUT_BINDINGS_BEFORE_RGB_READ.json',dict(completed=True,code=bind(__file__),entry=bind(entry),protocol=bind(PROTOCOL),
        control_complete=e['control_complete'],diagnosis_seal=e['diagnosis_seal'],fixed_cases=diagnosis['fixed_cases'],registry=p['fixed_registry'],
        fixed_base_slots=[list(x)for x in FIXED],all8_companion_metadata_checked=True,RGB_reads=0,no_new_case_selection=True))
    from PIL import Image,ImageDraw
    strips=[];raw_bindings=[]
    for strip in fixed['strips']:
        base,slot=strip['base_index'],strip['native_slot']
        canvas=Image.new('RGB',(1440,620),(247,248,250));d=ImageDraw.Draw(canvas)
        d.text((14,8),f'V139 fixed source crops | base {base}, native {slot}',font=font(24),fill=(15,20,28))
        d.text((14,40),'C30 = exact C29 replay; C31 = same fit with original native G42. Fixed examples, not population rates.',font=font(17),fill=(15,20,28))
        d.text((14,66),'D: visible   E: external cover   U: unavailable | h=hidden logit; h=0 is E except v61 (D).',font=font(17),fill=(15,20,28))
        image_records=[]
        for vi,row in enumerate(strip['variants']):
            png=check(row['saved_RGB96']);rgb=np.asarray(Image.open(png).convert('RGB'))
            assert rgb.dtype==np.uint8 and rgb.shape==(96,96,3) and array_sha(rgb)==row['RGB96_array_sha256']
            x=360*vi
            d.text((x+12,99),f'{row["variant"]} | teacher {row["teacher"]} | qerr {row["q_error_px"]:.3f}px',font=font(16),fill=(15,20,28))
            d.text((x+12,122),f'L={row["L"]:.3f} | original slot {row["global_index"]}',font=font(15),fill=(35,40,48))
            # Nearest-neighbor enlargement is display only; no crop or overlay.
            canvas.paste(Image.fromarray(rgb).resize((288,288),Image.Resampling.NEAREST),(x+36,149))
            for mi,model in enumerate(MODELS):
                score=row['models'][model];z=score['hidden_logit'];pr=score['prediction']
                label=f'{model}: {pr}  h='+('null' if z is None else f'{z:+.6f}')
                color=(20,115,55)if pr==row['teacher']else(175,30,30)
                d.text((x+12,447+24*mi),label,font=font(17),fill=color)
            assert same(row['saved_RGB96'],bind(png))
            image_records.append(dict(global_index=row['global_index'],saved_RGB96=row['saved_RGB96'],RGB96_array_sha256=row['RGB96_array_sha256']))
            raw_bindings.append(row['saved_RGB96'])
        d.text((14,584),'Exact saved C16 RGB96, displayed at 3x. v61 uses P3 ROI. q/L differ across variants; no teacher-centered recrop.',font=font(16),fill=(40,45,55))
        path=OUT/f'BASE_{base:04d}_NATIVE_{slot}_C16_V61_C29_C30_C31_RGB96.png';canvas.save(path)
        strips.append(dict(base_index=base,native_slot=slot,strip=bind(path),variants=strip['variants'],unchanged_saved_RGB96=image_records))
    index=OUT/'INDEX.html'
    text=['<!doctype html><meta charset="utf-8"><title>V139 fixed saved crop comparison</title><style>body{font:16px sans-serif;margin:24px;background:#f7f8fa;color:#15202b}img{max-width:100%}</style>',
          '<h1>V139 fixed source crops</h1><p>Two previously fixed base/slot groups. Original saved C16 RGB96 pixels; no recrop, model execution, or new case selection. v61 logits shown with sign reversed to hidden logit. These examples do not measure prevalence or private-data transfer.</p>']
    for s in strips:text.append('<h2>Base '+str(s['base_index'])+' / native '+str(s['native_slot'])+'</h2><img src="'+html.escape(Path(s['strip']['path']).name)+'">')
    text.append('<p>C30 and C31 source eligibility: '+html.escape(json.dumps({a:x['source_eligible']for a,x in control['arms'].items()}))+'. Selected source arm: '+html.escape(str(control['selected_arm_source_only']))+'.</p>')
    index.write_text('\n'.join(text)+'\n')
    receipt=dict(completed=True,PASS=True,code=bind(__file__),entry=bind(entry),protocol=bind(PROTOCOL),policy=p['policy'],control_complete=e['control_complete'],
        diagnosis_seal=e['diagnosis_seal'],fixed_cases=diagnosis['fixed_cases'],fixed_registry=p['fixed_registry'],fixed_base_slots=[list(x)for x in FIXED],
        strips=strips,source_RGB96=raw_bindings,all8_original_companion_RGB96_hashes_and_decoded_arrays_exact=True,raw_RGB96_unchanged=True,
        nearest_resize_display_only=True,no_overlay_on_RGB96=True,new_crop_model_fit_private_V112_PnP_qcorrection_case_selection_calls=0,index=bind(index),seconds=time.monotonic()-start)
    write(OUT/'VISUAL_REPORT_COMPLETE.json',receipt)
    print(json.dumps(dict(completed=True,PASS=True,manifest=bind(OUT/'VISUAL_REPORT_COMPLETE.json'),images=[s['strip']for s in strips],index=bind(index))),flush=True)


def prepare():
    assert sha(POLICY)==POLICY_SHA
    registry=read(REGISTRY);assert registry['completed'] and registry['prepared_before_new_outputs'] and registry['no_new_score_or_RGB_reads']
    assert registry['fixed_base_slot_pairs']==[list(x)for x in FIXED]
    assert len(registry['cases'])==8 and len(registry['strips'])==2
    assert decode('v61',0,True)=='D' and decode('C31',0,True)=='E' and decode('C31',None,False)=='U'
    for (base,slot),strip in zip(FIXED,registry['strips']):
        assert [r['global_index']for r in strip['variants']]==[(base*4+v)*8+slot for v in range(4)]
        assert [r['variant']for r in strip['variants']]==list(VARIANTS)
    # Prevent the previously corrected prose error from entering this display.
    cover=registry['strips'][1]['variants'][1]
    assert cover['global_index']==18889 and cover['models']['C29']['prediction']=='E' and cover['models']['C29']['hidden_logit']==6.073758602142334
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),metadata_only=True,zero_threshold_v61_tie_decoder_distinct=True,
        fixed_two_groups_all4_companion_identity_order=True,corrected_C29_base590_COVER_metadata_preserved=True,RGB_newscore_model_fit_reads=0))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),generated=bind(GENERATED),fixed_registry=bind(REGISTRY),
        expected_control=str(CONTROL),expected_diagnosis=str(DIAGNOSIS),expected_fixed_json=str(FIXED_JSON),output=str(OUT),
        fixed_pairs=[list(x)for x in FIXED],models=list(MODELS),RGB96='Exact inherited PNG and decoded array SHA; 3x nearest display only; no overlay or recrop.',
        no_RGB_or_partial_new_output_read_before_root_entry=True)
    write(PROTOCOL,p)
    template=dict(approved_saved_visual_report=False,code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],generated=p['generated'],fixed_registry=p['fixed_registry'],
        control_complete=dict(path=str(CONTROL),sha256='FILL_COMPLETED_CONTROL_SHA256'),diagnosis_seal=dict(path=str(DIAGNOSIS),sha256='FILL_COMPLETED_DIAGNOSIS_SHA256'),
        **{k:False for k in FORBIDDEN_FLAGS})
    template_path=H/'ROOT_SAVED_VISUAL_ENTRY_TEMPLATE.json';write(template_path,template)
    print(json.dumps(dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),template=bind(template_path),RGB_newscore_model_fit_reads=0)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use --prepare, or completed-seal root --entry')
