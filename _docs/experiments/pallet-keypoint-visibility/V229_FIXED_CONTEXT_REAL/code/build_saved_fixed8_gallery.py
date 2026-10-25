"""V229 saved-only fixed-policy gallery. No model initialization or forward."""
import argparse
import ast
import collections
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import stat
import types
import traceback

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
R=P/'publication_root_preparation'
ROOT=R/'V229_SELECTED_FIXED_CONTEXT_REAL'
POLICY=ROOT/'VISUAL_CASE_POLICY_BEFORE_NEW_CURRENT_SCORES.json'
POLICY_SHA='fea028cdd29cdce3a9583c7872b10954757ab2829ff56cfdeaa4d4c79211c76c'
PREDICTOR=R/'V229_FROZEN_INFERENCE_A/infer_fixed_context_v3.py'
PREDICTOR_SHA='187e13310d31a7254ca2d8888044b909b0329e877d07a218c4355c11fce0c7fb'
MANIFEST=R/'V229_REAL_QA_B/current223_input_metadata_v1/INPUT_MANIFEST_CURRENT223_ONLY.json'
MANIFEST_SHA='4a9d8e6c781d2116a22ad824d7e7e5f7bbb45e6458497e8cde3dea452d846dbd'
EVAL_CODE_SHA='e9eb7cb363ba0d03e17be989a416b1f6c9c7b79505dc0f9ed4313ddb70b18b73'
D,E='DIRECT_VISIBLE','EXTERNAL_OCCLUDED'
BUCKETS=('PRIMARY_D_PRED_D','PRIMARY_D_PRED_E_MANUAL_CLICK_LE3_IF_AVAILABLE_OTHERWISE_ANY_D_PRED_E','PRIMARY_E_PRED_E','PRIMARY_E_PRED_D')


def safe(path):
    p=Path(os.path.abspath(os.fspath(path)))
    assert 'v112' not in str(p).casefold(),'V112 before FS'
    for q in reversed((p,*p.parents)):assert not q.is_symlink(),str(q)
    return p


def read(path):
    with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()


def digest(b):return hashlib.sha256(b).hexdigest()
def identity(b):return str(Path(os.path.abspath(b['path']))),b['sha256']
def binding(p):
    raw=read(p);return {'path':str(safe(p)),'sha256':digest(raw),'size_bytes':len(raw)}
def checked(b):
    raw=read(b['path']);assert digest(raw)==b['sha256'],b['path']
    if 'size_bytes' in b:assert len(raw)==b['size_bytes']
    return raw
def write(p,x):
    with safe(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def number(x):
    try:v=float(x);return v if __import__('math').isfinite(v) else None
    except (TypeError,ValueError):return None


def material(row):
    # Existing cohort group supplies the fixed WOOD/SQUARE type; raw material
    # text remains in the case record. No appearance-based material inference.
    return {'WOOD_PRIMARY105':'WOOD','SQUARE118':'SQUARE'}.get(row['group'])


def select_cases(rows):
    assert len({int(r['global_index']) for r in rows})==len(rows)
    used=set();selected=[];buckets=[]
    for mat in ('WOOD','SQUARE'):
        for bi,bucket in enumerate(BUCKETS):
            truth=D if bi<2 else E;pred='D' if bi in (0,3) else 'E'
            pool=[r for r in rows if material(r)==mat and str(r['primarypairedbool'])=='True' and r['human_state']==truth and r['V229_D_E_U']==pred]
            manual_preference=False
            if bi==1:
                near=[r for r in pool if r['annotation_coordinate_source']=='manual_click' and number(r['initial_reference_error_px']) is not None and number(r['initial_reference_error_px'])<=3]
                if near:pool=near;manual_preference=True
            unused=[r for r in pool if int(r['frame_index']) not in used]
            choice=min(unused or pool,key=lambda r:int(r['global_index'])) if pool else None
            record={'material':mat,'bucket':bucket,'candidate_count':len(pool),'manual_click_LE3_preference_used':manual_preference,'missing':choice is None}
            if choice is not None:
                fi=int(choice['frame_index']);record.update({'case_index':len(selected),'global_index':int(choice['global_index']),'frame_index':fi,'row_id':choice['row_id'],'frame_previously_used':fi in used})
                selected.append({'selection':record,'ledger':choice});used.add(fi)
            buckets.append(record)
    assert len(selected)<=8
    return selected,buckets


def operators(np,torch,F):
    raw=read(PREDICTOR);assert digest(raw)==PREDICTOR_SHA
    tree=ast.parse(raw)
    constants={'P','R','EXTRACTOR','EXTRACTOR_SHA','ALIAS_CODE','ALIAS_SHA','SOURCES'}
    functions={'require','safe','read','digest','binding','checked','identity','pure_operators','alias_loader'}
    nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name in functions)]
    assert {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==functions
    ns={'Path':Path,'os':os,'stat':stat,'types':types,'ast':ast,'hashlib':hashlib}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(PREDICTOR)+'::pure_display_only','exec'),ns)
    return ns['pure_operators'](np,torch,F),ns['alias_loader']()


def template():
    return {'approved_saved_evaluation_visualization':False,'code':binding(__file__),'policy':{'path':str(POLICY),'sha256':POLICY_SHA},'predictor_code':{'path':str(PREDICTOR),'sha256':PREDICTOR_SHA},'input_manifest':{'path':str(MANIFEST),'sha256':MANIFEST_SHA},'evaluation_complete':{'path':None,'sha256':None},'prediction_complete':{'path':None,'sha256':None},'prediction_freeze':{'path':None,'sha256':None},'output_directory':str(ROOT/'fixed8_saved_result_gallery_v1')}


def run(entry_path):
    eraw=read(entry_path);entry=json.loads(eraw)
    assert entry['approved_saved_evaluation_visualization'] is True
    assert identity(entry['code'])==identity(binding(__file__))
    for key,path,sha in [('policy',POLICY,POLICY_SHA),('predictor_code',PREDICTOR,PREDICTOR_SHA),('input_manifest',MANIFEST,MANIFEST_SHA)]:assert identity(entry[key])==(str(path),sha)
    policy=json.loads(checked(entry['policy']));assert policy['maximum_cases']==8 and policy['materials']==['WOOD','SQUARE'] and policy['buckets_per_material']==list(BUCKETS)
    ec=json.loads(checked(entry['evaluation_complete']));pc=json.loads(checked(entry['prediction_complete']));freeze=json.loads(checked(entry['prediction_freeze']))
    assert ec['completed'] and ec['PASS'] and ec['code']['sha256']==EVAL_CODE_SHA
    assert pc['completed'] and pc['PASS'] and pc['selected_arm']=='CONTROL_FIXED_Q' and pc['threshold_logit']==0 and pc['annotations_or_baseline_scores_read']==0
    assert freeze['completed'] and freeze['predictions_frozen'] and freeze['annotations_or_baseline_scores_read']==0
    assert identity(ec['prediction_complete'])==identity(entry['prediction_complete']) and identity(ec['prediction_freeze'])==identity(entry['prediction_freeze']) and identity(pc['prediction_freeze'])==identity(entry['prediction_freeze'])
    ledger_binding=ec['outputs']['ALL1784_LEDGER.csv']
    ledger=list(csv.DictReader(io.StringIO(checked(ledger_binding).decode())));assert len(ledger)==1784
    selected,buckets=select_cases(ledger)
    out=safe(entry['output_directory']);assert out.parent==ROOT and not out.exists(),'Fresh direct-child output only'
    out.mkdir()
    # Durable selection precedes any original image SHA/body read or decode.
    minimal=[]
    fields=('global_index','id','row_id','frame_index','native_corner','group','material','human_state','query_x','query_y','roi_side','original_query_valid','initial_reference_error_px','annotation_coordinate_source','v61_D_E_U','legacy_D_E_U','V229_D_E_U','logit','U_reason')
    for case in selected:minimal.append({'selection':case['selection'],'saved':{k:case['ledger'][k] for k in fields}})
    selection_path=out/'FIXED8_CASE_IDS_BEFORE_RGB.json'
    write(selection_path,{'completed':True,'policy':entry['policy'],'evaluation_complete':entry['evaluation_complete'],'ledger':ledger_binding,'buckets':buckets,'selected':minimal,'selected_rows':len(selected),'unique_frames':len({int(x['ledger']['frame_index']) for x in selected}),'original_image_reads_before_selection':0,'new_case_replacement_after_RGB':False})
    try:
        import numpy as np
        import torch
        from torch.nn import functional as F
        from PIL import Image,__version__ as PIL_VERSION
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        op,alias=operators(np,torch,F)
        manifest=json.loads(checked(entry['input_manifest']));frames=manifest['frames223'];inputs=manifest['rows1784'];assert len(frames)==223 and len(inputs)==1784
        selected_frames={int(case['ledger']['frame_index']) for case in selected};images={};imagefacts={}
        for fi in sorted(selected_frames):
            g=frames[fi];assert g['frame_index']==fi and g['domain']=='CURRENT223_1784'
            source=g['source_fullRGB'];p=alias(source['binding'])
            rgb=np.asarray(Image.open(p).convert('RGB'),dtype=np.uint8).copy()
            assert list(rgb.shape)==g['raw_hw']+[3] and digest(rgb.tobytes())==g['decoded_RGB_sha256']
            images[fi]=rgb;imagefacts[fi]={'original_binding':source['binding'],'safe_canonical_path':str(p),'decoded_RGB_sha256':g['decoded_RGB_sha256'],'decoded_RGB_exact':True,'original_decoder':source['decoder'],'actual_Pillow_version':PIL_VERSION}
        outputs=[];details=[]
        for case in selected:
            r=case['ledger'];s=case['selection'];i=int(r['global_index']);fi=int(r['frame_index']);k=int(r['native_corner']);original=inputs[i];g=frames[fi];rgb=images[fi]
            assert i==fi*8+k and original['original_global_index']==i and original['row_id']==r['row_id'] and original['id']==r['id'] and original['native_slot']==k and r['original_query_valid']=='True' and original['original_query_available'] is True
            q=np.asarray(original['query_xy'],np.float64);L=original['L']
            assert np.array_equal(q,np.array([float(r['query_x']),float(r['query_y'])])) and L==float(r['roi_side'])
            local,valid=op['crop'](rgb,q,L,96);box=op['query_extent'](g['original_query_xy8']);assert box is not None
            context,contextvalid=op['sample_target_context'](rgb,box)
            centre=local[:,32:64,32:64];centrevalid=valid[:,32:64,32:64]
            assert local.shape==(3,96,96) and valid.shape==(1,96,96) and centre.shape==(3,32,32)
            qcontext=(q-np.asarray(box[:2]))/(np.asarray(box[2:])-box[:2])*224-.5
            fig,axs=plt.subplots(2,3,figsize=(17,10))
            axs[0,0].imshow(rgb)
            for nk,qk in enumerate(g['original_query_xy8']):
                xy=np.asarray(qk,np.float64)
                if xy.shape==(2,) and np.isfinite(xy).all() and not (xy==[-1,-1]).all():
                    color='cyan' if g['original_query_available8'][nk] else 'gray';axs[0,0].plot(*xy,'+',color=color,ms=6);axs[0,0].text(*xy,str(nk),color=color,fontsize=8,clip_on=True)
            axs[0,0].plot(*q,'+',color='cyan',ms=14,mew=2)
            axs[0,0].add_patch(patches.Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,edgecolor='yellow',linewidth=1))
            axs[0,0].set_xlim(0,rgb.shape[1]);axs[0,0].set_ylim(rgb.shape[0],0);axs[0,0].set_title('Original full RGB / q8 context extent')
            axs[0,1].imshow(local.transpose(1,2,0),interpolation='nearest');axs[0,1].plot(47.5,47.5,'+',color='cyan',ms=16,mew=2);axs[0,1].set_title('Original local96 / original q=47.5')
            axs[0,2].imshow(np.clip(context/255,0,1),interpolation='nearest');axs[0,2].plot(*qcontext,'+',color='cyan',ms=15,mew=2);axs[0,2].set_title('Same q8 context224 / original active q')
            axs[1,0].imshow(centre.transpose(1,2,0),interpolation='nearest');axs[1,0].plot(15.5,15.5,'+',color='cyan',ms=16,mew=2);axs[1,0].set_title('Centre32 from original local96 [32:64]')
            axs[1,1].imshow(valid[0],cmap='gray',vmin=0,vmax=1,interpolation='nearest');axs[1,1].plot(47.5,47.5,'+',color='cyan',ms=15,mew=2);axs[1,1].add_patch(patches.Rectangle((31.5,31.5),32,32,fill=False,edgecolor='yellow'));axs[1,1].set_title('Original local96 pixel support / white=valid')
            axs[1,2].axis('off')
            text=[f"Frozen human state: {r['human_state']}",f"v61 saved:    {r['v61_D_E_U']}",f"legacy saved: {r['legacy_D_E_U']}",f"V229 CONTROL: {r['V229_D_E_U']}",f"V229 fixed logit: {r['logit']}",f"Reference error: {r['initial_reference_error_px']} px",f"Provenance: {r['annotation_coordinate_source']}",f"Original q: {q.tolist()}",f"Original L: {L:.8g} / native {k}",f"Support local96: {int(valid.sum())}/9216",f"Support centre32: {int(centrevalid.sum())}/1024",'No reference point is drawn or used to recenter.']
            axs[1,2].text(0,.98,'\n'.join(text),va='top',fontsize=12,transform=axs[1,2].transAxes)
            for ax in axs.flat:ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f"{s['material']} | {s['bucket']} | g{i} | {r['id']} native{k}",fontsize=13)
            fig.text(.5,.015,'Saved evaluation only. Original q/L unchanged; no new model forward. Fixed selected cases are not representative and do not establish physical native identity or failure cause.',ha='center',fontsize=10)
            fig.tight_layout(rect=(0,.04,1,.95))
            png=out/f"CASE_{s['case_index']:02d}_G{i:04d}.png";assert not png.exists();fig.savefig(png,dpi=140);plt.close(fig)
            outputs.append(binding(png));details.append({'selection':s,'saved':{key:r[key] for key in fields},'original_query_xy':q.tolist(),'original_L':L,'original_q8':g['original_query_xy8'],'context_box':box,'RGB96_CHW_sha256':digest(local.tobytes()),'valid96_bool_sha256':digest(valid.tobytes()),'context224_float32_HWC_sha256':digest(context.tobytes()),'local_valid_pixels':int(valid.sum()),'centre32_valid_pixels':int(centrevalid.sum()),'context_valid_fraction':float(contextvalid.mean()),'cached_RAW_NPZ_pixel_parity_claim':False,'same_original_sampler_math':True,'panel':outputs[-1]})
        write(out/'CASE_ROWS_AND_DISPLAY_FACTS.json',{'completed':True,'cases':details,'buckets':buckets,'selected_rows':len(details),'unique_frames':len(selected_frames),'image_facts':imagefacts,'selection_before_RGB':binding(selection_path),'new_training_or_model_forward_or_PnP_or_recenter':0})
        page='<!doctype html><meta charset="utf-8"><title>V229 fixed saved cases</title><h1>고정 정책의 저장 사례</h1><p>원 q/L/native 유지. GT reference로 recenter하지 않았습니다. 3개 모델의 저장 응답이며 새 forward는 없습니다. 사례는 대표성·원인·물리 native gold를 증명하지 않습니다.</p><p><a href="FIXED8_CASE_IDS_BEFORE_RGB.json">영상 전 고정 case IDs</a> · <a href="CASE_ROWS_AND_DISPLAY_FACTS.json">저장 수치/표시 근거</a></p>'
        for x in details:page+=f'<figure><img src="{Path(x["panel"]["path"]).name}" style="max-width:100%"><figcaption>{html.escape(x["selection"]["material"]+" / "+x["selection"]["bucket"])}; g{x["selection"]["global_index"]}</figcaption></figure>'
        missing=[f'{x["material"]}/{x["bucket"]}' for x in buckets if x['missing']]
        page+='<p>빈 bucket: '+html.escape(', '.join(missing) or '없음')+'</p>'
        with safe(out/'PORTABLE_GALLERY_KO.html').open('x',encoding='utf-8') as f:f.write(page)
        complete={'completed':True,'PASS':True,'code':entry['code'],'policy':entry['policy'],'entry':{'path':str(safe(entry_path)),'sha256':digest(eraw)},'evaluation_complete':entry['evaluation_complete'],'prediction_complete':entry['prediction_complete'],'prediction_freeze':entry['prediction_freeze'],'ledger':ledger_binding,'selection':binding(selection_path),'case_rows':binding(out/'CASE_ROWS_AND_DISPLAY_FACTS.json'),'gallery':binding(out/'PORTABLE_GALLERY_KO.html'),'figures':outputs,'cases':len(details),'unique_frames':len(selected_frames),'missing_buckets':missing,'all_views_are_saved_pixel_displays_not_new_labels':True,'new_model_forward_or_training_or_reference_recenter':0}
        write(out/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
    except BaseException:
        write(out/'FAILED_NO_AUTOMATIC_RETRY.json',{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'automatic_retry':False});raise


def self_test():
    rows=[]
    def row(i,fi,truth,pred,manual=False):return {'global_index':str(i),'frame_index':str(fi),'row_id':f'row{i}','group':'WOOD_PRIMARY105','material':'WOOD','primarypairedbool':'True','human_state':truth,'V229_D_E_U':pred,'annotation_coordinate_source':'manual_click' if manual else 'pnp_projected','initial_reference_error_px':'2' if manual else '9'}
    rows=[row(0,0,D,'D'),row(1,0,D,'D'),row(8,1,D,'D'),row(2,0,D,'E'),row(16,2,D,'E',True),row(3,0,E,'E'),row(24,3,E,'E'),row(4,0,E,'D'),row(32,4,E,'D')]
    selected,b=select_cases(rows)
    assert [x['selection']['global_index'] for x in selected]==[0,16,24,32]
    assert len(b)==8 and sum(x['missing'] for x in b)==4
    assert selected[1]['selection']['manual_click_LE3_preference_used'] is True
    assert number('') is None and number('nan') is None and number('3')==3
    print(json.dumps({'completed':True,'PASS':True,'invented_metadata_only':True,'checks':['fixed_bucket_order','manual_LE3_before_unused_frame_preference','unused_frame_then_min_global','empty_buckets_retained','finite_optional_error'],'actual_GT_score_image_weight_reads':0,'model_instance_or_forward':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--entry');g.add_argument('--entry-template',action='store_true');g.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    elif a.entry_template:print(json.dumps(template(),ensure_ascii=False,indent=2))
    else:run(a.entry)
