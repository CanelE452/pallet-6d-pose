"""V142 thin saved-only adapter for the two existing V139 contact strips.

Only root --entry opens C32 saved score metadata or the eight original PNGs.
No model, checkpoint, feature pack, GT ledger, recrop or case selection.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import html
import importlib.util
import json
import sys
import time
import numpy as np
sys.dont_write_bytecode = True
H=Path(__file__).resolve().parent; V=H.parent; P=V.parent
POLICY=V/'ROOT_NATIVE_GRADIENT_CONTROL_POLICY_BEFORE_FIT.json'
POLICY_SHA='8364a66eba156bbc01d5789fb790471ce68702dfc02b687060a7dbad9a27520d'
PARENT=P/'query_visibility_broad_source_geometry_control_20261024_v139/visual_report_preparation/show_fixed_saved_crops.py'
PARENT_SHA='80f31d3a87a18f8c4d241659002614d2330d1e4785bec5f67a44506579a2aa8e'
REGISTRY=PARENT.parent.parent/'saved_result_preparation/FIXED_CASES_BEFORE_NEW_SCORES.json'
PRIOR_DIAGNOSIS=PARENT.parent.parent/'saved_geometry_control_diagnosis_v1/DIAGNOSIS_SEAL.json'
PRIOR_FIXED=PARENT.parent.parent/'saved_geometry_control_diagnosis_v1/FIXED8_CASES_AND_TWO_STRIPS.json'
RUN=V/'runs/C32_AUX_TO_NATIVE_GRAD0_FIXED8000'
ARM='C32_AUX_TO_NATIVE_GRAD0_FIXED8000'
CONTROL=RUN/'CONTROL_COMPLETE.json'; FIT=RUN/ARM/'FIT_COMPLETE.json'
PREDICTIONS=RUN/ARM/'FIXED8_CASES_AND_TWO_STRIPS.json'
OUT=V/'saved_native_gradient_fixed_visual_report_v1'
PROTOCOL=H/'PROTOCOL_BEFORE_RGB_AND_C32_SCORES.json'
GENERATED=H/'GENERATED_METADATA_CHECKS.json'
FIXED=((519,0),(590,1)); VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')
MODELS=('C16','v61','C29','C32')
FORBIDDEN_FLAGS=('model_forward','fit','new_crop','new_case_selection','current223','V112','PnP','q_L_correction')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(PARENT)==PARENT_SHA
spec=importlib.util.spec_from_file_location('v142_saved_visual_parent',PARENT)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
bind,read,check,same,write,decode,font,array_sha=h.bind,h.read,h.check,h.same,h.write,h.decode,h.font,h.array_sha


def assemble(c32,prior,registry,registry_binding):
    assert c32['completed'] and same(c32['registry'],registry_binding)
    assert c32['new_case_selection'] is False and c32['RGB_read'] is False and c32['GT_q_L_correction'] is False
    assert c32['prediction_is_actual_original_query'] is True
    assert prior['completed'] and prior['PASS'] and prior['all8_preserved'] and prior['no_new_case_selection']
    assert same(prior['registry'],registry_binding) and prior['RGB_body_reads']==0
    assert registry['fixed_base_slot_pairs']==[list(x)for x in FIXED]
    assert [r['global_index']for r in c32['cases']]==[r['global_index']for r in prior['cases']]==registry['fixed8_indices']
    assert len(c32['cases'])==8 and len(c32['strips'])==len(prior['strips'])==2
    # The fixed8 case list includes four companion groups; validate all eight,
    # while displaying only the previously fixed two strips/eight original PNGs.
    for new,original in zip(c32['cases'],registry['cases']):
        for key in ('global_index','base_index','native_slot','variant','teacher','q_error_px','L'):
            assert new[key]==original[key],key
        assert decode('C32',new['raw_hidden_logit_before_U'],new['available'])==new['prediction']
    by_case={r['global_index']:r for r in c32['cases']}
    strips=[]
    for (base,slot),new,previous,old in zip(FIXED,c32['strips'],prior['strips'],registry['strips']):
        assert new['base_index']==previous['base_index']==old['base_index']==base
        assert new['native_slot']==previous['native_slot']==old['native_slot']==slot
        assert len(new['variants'])==len(previous['variants'])==len(old['variants'])==4
        variants=[]
        for vi,(nr,pr,original)in enumerate(zip(new['variants'],previous['variants'],old['variants'])):
            index=(base*4+vi)*8+slot
            assert nr['global_index']==pr['global_index']==original['global_index']==index
            assert nr['base_index']==base and nr['native_slot']==slot
            assert nr['variant']==pr['variant']==original['variant']==VARIANTS[vi]
            for key in ('teacher','q_error_px','L'): assert nr[key]==pr[key]==original[key],key
            assert nr['query_xy']==pr['actual_query_xy']==original['actual_query_xy']
            if index in by_case: assert nr==by_case[index]
            for key in ('teacher_anchor_xy','saved_RGB96','RGB96_array_sha256'): assert pr[key]==original[key],key
            assert nr['available']==pr['models']['C29']['available']
            models={}
            for model in MODELS[:3]:
                score=pr['models'][model]
                assert score['prediction']==original['models'][model]['prediction']
                assert score['hidden_logit']==original['models'][model]['hidden_logit']
                assert decode(model,score['hidden_logit'],score['available'])==score['prediction']
                models[model]=copy.deepcopy(score)
            score=dict(prediction=nr['prediction'],hidden_logit=nr['raw_hidden_logit_before_U'],available=nr['available'],head_forwarded=nr['head_forwarded'])
            assert decode('C32',score['hidden_logit'],score['available'])==score['prediction']
            models['C32']=score
            row=copy.deepcopy(original);row['models']=models;variants.append(row)
        strips.append(dict(base_index=base,native_slot=slot,variants=variants))
    cover=strips[1]['variants'][1]
    assert cover['global_index']==18889 and cover['models']['C29']['prediction']=='E'
    assert cover['models']['C29']['hidden_logit']==6.073758602142334
    return dict(completed=True,PASS=True,strips=strips,all8_case_ids_preserved=True,RGB_body_reads=0,new_case_selection=False)


def run(entry):
    e=read(entry);assert e['approved_saved_visual_report'] is True
    assert all(e[k]is False for k in FORBIDDEN_FLAGS)
    assert sha(POLICY)==POLICY_SHA and same(e['code'],bind(__file__)) and same(e['protocol'],bind(PROTOCOL))
    p=read(PROTOCOL)
    for key in ('code','policy','generated','parent_code','fixed_registry'):
        assert same(e[key],p[key]);check(p[key])
    assert read(GENERATED)['PASS'] and not OUT.exists()
    assert Path(e['control_complete']['path'])==CONTROL and Path(e['fit_complete']['path'])==FIT
    assert Path(e['fixed_predictions']['path'])==PREDICTIONS and Path(e['prior_diagnosis_seal']['path'])==PRIOR_DIAGNOSIS
    control=read(check(e['control_complete']));fit=read(check(e['fit_complete']))
    assert control['completed'] and control['PASS'] and control['final_update8000_single_arm']
    assert control['AUX_old192_loss_coefficient']==0 and control['new_baseline_fits']==0
    assert control['selected_model'] is None and control['selected_arm_source_only'] is None and control['model_selection_prohibited']
    assert control['private_current_V112_PnP_q_correction_calls']==0
    assert same(control['arms'][ARM]['fit'],e['fit_complete'])
    assert fit['completed'] and fit['optimizer_updates']==8000 and fit['selected_model'] is None and fit['diagnostic_only']
    assert same(fit['fixed_cases'],e['fixed_predictions'])
    # Independent numeric review must explicitly bind this completed control.
    review=read(check(e['independent_review']))
    assert review['completed'] and review['PASS'] and same(review['control_complete'],e['control_complete'])
    prior=read(check(e['prior_diagnosis_seal']))
    assert prior['completed'] and prior['PASS'] and prior['RGB_read']is False and prior['no_new_case_selection']
    assert Path(prior['fixed_cases']['path'])==PRIOR_FIXED
    fixed=assemble(read(check(e['fixed_predictions'])),read(check(prior['fixed_cases'])),read(check(p['fixed_registry'])),p['fixed_registry'])
    OUT.mkdir();start=time.monotonic()
    write(OUT/'INPUT_BINDINGS_BEFORE_RGB_READ.json',dict(completed=True,code=bind(__file__),entry=bind(entry),protocol=bind(PROTOCOL),
        control_complete=e['control_complete'],fit_complete=e['fit_complete'],fixed_predictions=e['fixed_predictions'],independent_review=e['independent_review'],
        prior_diagnosis=e['prior_diagnosis_seal'],registry=p['fixed_registry'],fixed_base_slots=[list(x)for x in FIXED],RGB_reads=0,
        all_fixed8_and_all4x2_geometry_scores_verified_before_pixels=True,new_case_selection=False))
    from PIL import Image,ImageDraw
    strips=[];raw_bindings=[]
    for strip in fixed['strips']:
        base,slot=strip['base_index'],strip['native_slot']
        canvas=Image.new('RGB',(1440,620),(247,248,250));d=ImageDraw.Draw(canvas)
        d.text((14,8),f'V142 fixed source crops | base {base}, native {slot}',font=font(24),fill=(15,20,28))
        d.text((14,40),'C32: C29 matched native losses/draws, AUX gradient removed; same mixed-source C16 initialization.',font=font(17),fill=(15,20,28))
        d.text((14,66),'D: visible   E: external cover   U: unavailable | h=hidden logit; h=0 is E except v61 (D).',font=font(17),fill=(15,20,28))
        for vi,row in enumerate(strip['variants']):
            png=check(row['saved_RGB96']);rgb=np.asarray(Image.open(png).convert('RGB'))
            assert rgb.dtype==np.uint8 and rgb.shape==(96,96,3) and array_sha(rgb)==row['RGB96_array_sha256']
            x=360*vi
            d.text((x+12,99),f'{row["variant"]} | teacher {row["teacher"]} | qerr {row["q_error_px"]:.3f}px',font=font(16),fill=(15,20,28))
            d.text((x+12,122),f'L={row["L"]:.3f} | original slot {row["global_index"]}',font=font(15),fill=(35,40,48))
            canvas.paste(Image.fromarray(rgb).resize((288,288),Image.Resampling.NEAREST),(x+36,149))
            for mi,model in enumerate(MODELS):
                score=row['models'][model];z=score['hidden_logit'];pr=score['prediction']
                label=f'{model}: {pr}  h='+('null'if z is None else f'{z:+.6f}')
                d.text((x+12,447+24*mi),label,font=font(17),fill=(20,115,55)if pr==row['teacher']else(175,30,30))
            raw_bindings.append(row['saved_RGB96'])
        d.text((14,558),'Exact saved RGB96 at 3x; v61 uses P3 ROI. q/L differ across variants; no GT recrop or coordinate correction.',font=font(16),fill=(40,45,55))
        d.text((14,584),'Previously fixed examples; do not measure prevalence or real transfer. Diagnostic only, no model selected.',font=font(16),fill=(40,45,55))
        path=OUT/f'BASE_{base:04d}_NATIVE_{slot}_C16_V61_C29_C32_RGB96.png';canvas.save(path)
        strips.append(dict(base_index=base,native_slot=slot,strip=bind(path),variants=strip['variants']))
    index=OUT/'INDEX.html'
    text=['<!doctype html><meta charset="utf-8"><title>V142 fixed saved crop comparison</title><style>body{font:16px sans-serif;margin:24px;background:#f7f8fa;color:#15202b}img{max-width:100%}</style>',
          '<h1>V142 fixed source crops</h1><p>Original fixed two groups, exact existing RGB96 pixels, C16/v61/C29/C32 saved scores. q/L vary across image variants. C32 retains mixed-source C16 initialization; AUX gradient removal is a net-effect diagnosis. No selected model or private evaluation.</p>']
    for s in strips:text.append('<h2>Base '+str(s['base_index'])+' / native '+str(s['native_slot'])+'</h2><img src="'+html.escape(Path(s['strip']['path']).name)+'">')
    index.write_text('\n'.join(text)+'\n')
    receipt=dict(completed=True,PASS=True,code=bind(__file__),entry=bind(entry),protocol=bind(PROTOCOL),policy=p['policy'],control_complete=e['control_complete'],
        fit_complete=e['fit_complete'],fixed_predictions=e['fixed_predictions'],independent_review=e['independent_review'],prior_diagnosis=e['prior_diagnosis_seal'],
        fixed_registry=p['fixed_registry'],strips=strips,source_RGB96=raw_bindings,all8_PNG_and_decoded_array_hashes_exact=True,
        raw_RGB96_unchanged=True,nearest_resize_display_only=True,no_overlay_on_RGB96=True,selected_model=None,
        new_crop_model_fit_private_PnP_qcorrection_case_selection_calls=0,index=bind(index),seconds=time.monotonic()-start)
    write(OUT/'VISUAL_REPORT_COMPLETE.json',receipt)
    print(json.dumps(dict(completed=True,PASS=True,manifest=bind(OUT/'VISUAL_REPORT_COMPLETE.json'),images=[s['strip']for s in strips],index=bind(index))),flush=True)


def prepare():
    assert sha(POLICY)==POLICY_SHA and sha(PARENT)==PARENT_SHA
    registry=read(REGISTRY)
    assert registry['completed'] and registry['prepared_before_new_outputs'] and registry['no_new_score_or_RGB_reads']
    assert registry['fixed_base_slot_pairs']==[list(x)for x in FIXED] and len(registry['cases'])==8
    assert decode('v61',0,True)=='D' and decode('C32',0,True)=='E' and decode('C32',None,False)=='U'
    for (base,slot),strip in zip(FIXED,registry['strips']):
        assert [r['global_index']for r in strip['variants']]==[(base*4+v)*8+slot for v in range(4)]
    cover=registry['strips'][1]['variants'][1]
    assert cover['global_index']==18889 and cover['models']['C29']['hidden_logit']==6.073758602142334 and cover['models']['C29']['prediction']=='E'
    # Invented C32 scores plus original fixed metadata exercise the adapter before
    # any real C32 score body or RGB is opened. Query/teacher/score tampering fails.
    rb=bind(REGISTRY); lookup={r['global_index']:r for s in registry['strips']for r in s['variants']}
    def invented(r,base,slot):
        return dict(global_index=r['global_index'],base_index=base,native_slot=slot,variant=r['variant'],teacher=r['teacher'],
            q_error_px=r['q_error_px'],L=r['L'],query_xy=r.get('actual_query_xy',[0.,0.]),
            prediction='E',raw_hidden_logit_before_U=0.,available=True,head_forwarded=True)
    c32=dict(completed=True,registry=rb,new_case_selection=False,RGB_read=False,GT_q_L_correction=False,prediction_is_actual_original_query=True,
        cases=[invented(lookup.get(r['global_index'],r),r['base_index'],r['native_slot'])for r in registry['cases']],
        strips=[dict(base_index=s['base_index'],native_slot=s['native_slot'],variants=[invented(r,s['base_index'],s['native_slot'])for r in s['variants']])for s in registry['strips']])
    prior=copy.deepcopy(registry);prior.update(completed=True,PASS=True,all8_preserved=True,no_new_case_selection=True,registry=rb,RGB_body_reads=0)
    for s in prior['strips']:
        for r in s['variants']:
            for score in r['models'].values():score['available']=True
    assemble(c32,prior,registry,rb)
    for key,value in [('query_xy',[999.,999.]),('teacher','U'),('prediction','D')]:
        changed=copy.deepcopy(c32);changed['strips'][0]['variants'][0][key]=value
        try:assemble(changed,prior,registry,rb)
        except AssertionError:pass
        else:raise AssertionError('Changed fixed query/teacher/score accepted')
    write(GENERATED,dict(completed=True,PASS=True,code=bind(__file__),metadata_only=True,parent_decoder_reused=True,
        fixed_two_groups_all4_originalIDs_and_fixed8_preserved=True,C29_base590_COVER_correct_value_preserved=True,
        invented_C32_schema_join_and_query_teacher_prediction_tamper_rejections=True,
        RGB_C32score_feature_GT_weight_body_model_fit_reads=0))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),policy=bind(POLICY),generated=bind(GENERATED),parent_code=bind(PARENT),fixed_registry=bind(REGISTRY),
        expected_control=str(CONTROL),expected_fit=str(FIT),expected_fixed_predictions=str(PREDICTIONS),expected_prior_diagnosis=str(PRIOR_DIAGNOSIS),
        independent_review_contract='completed=true,PASS=true,control_complete exact binding to C32 CONTROL_COMPLETE',
        output=str(OUT),fixed_pairs=[list(x)for x in FIXED],models=list(MODELS),selected_model=None,
        RGB96='Exact inherited8 PNGs/decodedSHA;3xnearestdisplayonly; nooverlay/recrop',no_RGB_or_C32_saved_prediction_body_before_root_entry=True)
    write(PROTOCOL,p)
    template=dict(approved_saved_visual_report=False,code=p['code'],protocol=bind(PROTOCOL),policy=p['policy'],generated=p['generated'],parent_code=p['parent_code'],fixed_registry=p['fixed_registry'],
        control_complete=dict(path=str(CONTROL),sha256='FILL_COMPLETED_CONTROL_SHA256'),fit_complete=dict(path=str(FIT),sha256='FILL_COMPLETED_FIT_SHA256'),
        fixed_predictions=dict(path=str(PREDICTIONS),sha256='INHERIT_COMPLETED_FIT_FIXED_CASES_BINDING'),
        prior_diagnosis_seal=dict(path=str(PRIOR_DIAGNOSIS),sha256='FILL_COMPLETED_PRIOR_DIAGNOSIS_SHA256'),
        independent_review=dict(path='FILL_COMPLETED_INDEPENDENT_REVIEW_PATH',sha256='FILL_COMPLETED_INDEPENDENT_REVIEW_SHA256'),**{k:False for k in FORBIDDEN_FLAGS})
    path=H/'ROOT_SAVED_VISUAL_ENTRY_TEMPLATE.json';write(path,template)
    ready=dict(completed=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(GENERATED),template=bind(path),
        RGB_C32score_feature_GT_weight_body_model_fit_reads=0)
    write(H/'PREPARATION_READY.json',ready);print(json.dumps(ready),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--entry',type=Path);args=parser.parse_args()
    if args.prepare and args.entry is None:prepare()
    elif args.entry and not args.prepare:run(args.entry)
    else:raise SystemExit('Use --prepare or completed-control/review root --entry')
