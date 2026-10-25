"""One authorized saved SOURCE gallery; IDs sealed before raw96 payload access."""
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import textwrap
import traceback
import numpy as np
from PIL import Image
import PIL

H=Path(__file__).absolute().parent;R=H.parent;OUT=H/'gallery_v1'
os.environ['MPLCONFIGDIR']=str(H/'_temporary_matplotlib_config')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FILES={
 'policy':(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/FIXED12_SOURCE_CASE_POLICY_BEFORE_NEW_PREDICTIONS.json','df482d27f87c4e0e918eea0da5104152256a2e29ebea8c82237ce3fc28c26ea5'),
 'fit_complete':(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/data_only_source_fit_v1/COMPLETE.json','597ddc17d69ee8718c106181e8292e71b230aa768451bfef1f35c546bbf92e48'),
 'control_csv':(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/data_only_source_fit_v1/CONTROL_LOCAL_CROP_ALL_SOURCE_ROWS.csv','e638a38c196f58fe287f998f8636e102f5ea3abe43a7aede9f953c9d527dab8b'),
 'treatment_csv':(R/'V235_ROOT_PUBLIC_COUNTERFACTUAL/data_only_source_fit_v1/TREAT_CF_DATA_ONLY_ALL_SOURCE_ROWS.csv','3a232be1b9ef1a4b6abd93f354d8221725d2331b81a6442eb1c3c1cecaf6a458'),
 'crop_complete':(R/'V233_ROOT_CROP_ADAPTATION/crop_samples_v1/COMPLETE.json','45b294ea7e89ff1a9e7b1b4765a24c88b2b0b402a79aaa3e5e77b65f4ac80535'),
 'diagnostics':(R/'V233_ROOT_CROP_ADAPTATION/crop_samples_v1/SAMPLE_DIAGNOSTICS.jsonl','5862ce437067ae8d0c902fac34b93b476e3b5611d368e6d7451d78be1fb2cc5f'),
 'YCB_teacher_ref':(R/'V234_ROOT_LABEL_DIAG/YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv','8eab4fe1f9c58ba55ab308437605cc64a2c30dc92f07d21416759b455a342b08')}
RAW=(R/'V233_ROOT_CROP_ADAPTATION/crop_samples_v1/raw_rgb96.npy','f9c4dac0a9a021ce248732c335014d7c88d39539ff41a20cc941e6acaff7020a')


def safe(p):
    if 'v112' in str(p).casefold() or '__pycache__' in str(p).casefold() or str(p).casefold().endswith('.pyc'):raise ValueError('Forbidden path before FS')
    for q in reversed((p,*p.parents)):
        if q.is_symlink():raise ValueError('Symlink')
    return p


def read(p):
    fd=os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:return stream.read()


def checked(item):
    p,sha=item;raw=read(p)
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Changed metadata binding: '+str(p))
    return raw


def binding(p):
    raw=read(p);return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


def write_json(p,obj):
    with safe(p).open('x') as stream:json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return binding(p)


def bucket_name(b):
    family='SYN_MAIN' if b['family']==0 else 'YCB_AUX'
    if 'target' in b:return f"{family} target{'D' if b['target']==0 else 'E'} TREAT_{'correct' if b['treatment_correct'] else 'wrong'}"
    return family+' '+b['paired_change']


def decorate(ax,pixels,reference,marked):
    ax.imshow(pixels,interpolation='nearest',origin='upper')
    ax.set_xlim(-.5,95.5);ax.set_ylim(95.5,-.5);ax.set_aspect('equal')
    if marked:
        ax.scatter([47.5],[47.5],marker='+',s=170,c='#00ffff',linewidths=1.7,label='original q')
        if reference is not None:
            x,y=reference
            if -.5<=x<=95.5 and -.5<=y<=95.5:ax.scatter([x],[y],marker='x',s=90,c='#ff00ff',linewidths=1.7,label='teacher ref')
            else:ax.annotate(f'REF OUTSIDE ({x:.1f},{y:.1f})',xy=(np.clip(x,0,95),np.clip(y,0,95)),xytext=(3,8),color='magenta',fontsize=8,arrowprops={'arrowstyle':'->','color':'magenta'})
        ax.set_xticks([0,47.5,95]);ax.set_yticks([0,47.5,95]);ax.tick_params(labelsize=8)
    else:ax.set_xticks([]);ax.set_yticks([])


def build():
    if safe(OUT).exists():raise ValueError('Fresh gallery output required; no automatic reselection')
    raw_text={k:checked(v) for k,v in FILES.items()};policy=json.loads(raw_text['policy']);fit=json.loads(raw_text['fit_complete']);crop=json.loads(raw_text['crop_complete'])
    if fit['completed'] is not True or fit['PASS'] is not True or crop['completed'] is not True:raise ValueError('Completed SOURCE packets required')
    for arm,key in [('CONTROL_LOCAL_CROP','control_csv'),('TREAT_CF_DATA_ONLY','treatment_csv')]:
        if fit['outputs'][arm]['sha256']!=FILES[key][1]:raise ValueError('Exact output closure')
    control=list(csv.DictReader(io.StringIO(raw_text['control_csv'].decode())));treatment=list(csv.DictReader(io.StringIO(raw_text['treatment_csv'].decode())))
    diagnostics=[json.loads(line) for line in raw_text['diagnostics'].splitlines() if line.strip()]
    if not len(control)==len(treatment)==len(diagnostics)==7617:raise ValueError('Original7617 conservation')
    diag={int(row['sample_index']):row for row in diagnostics}
    for i,(a,b) in enumerate(zip(control,treatment)):
        for key in ['sample_index','row_id','family','role','available','target']:
            if a[key]!=b[key]:raise ValueError('Prediction metadata join')
        if int(a['sample_index'])!=i or diag[i]['row_id']!=a['row_id'] or int(a['family'])!=diag[i]['family_code'] or int(a['target'])!=diag[i]['target']:raise ValueError('Original diagnostic identity')
    cases=[]
    for number,bucket in enumerate(policy['8_primary_buckets']+policy['4_change_buckets'],1):
        candidates=[]
        for a,b in zip(control,treatment):
            if b['role']!='DEV' or b['available']!='True' or int(b['family'])!=bucket['family'] or int(b['target']) not in (0,1):continue
            truth='D' if b['target']=='0' else 'E';old=a['prediction']==truth;new=b['prediction']==truth
            if 'treatment_correct' in bucket:
                if int(b['target'])!=bucket['target'] or new!=bucket['treatment_correct']:continue
            elif bucket['paired_change']=='RECOVERED':
                if not(not old and new):continue
            elif bucket['paired_change']=='DAMAGED':
                if not(old and not new):continue
            candidates.append(int(b['sample_index']))
        index=min(candidates) if candidates else None
        cases.append({'case_number':number,'bucket':bucket,'bucket_name':bucket_name(bucket),'eligible_population':len(candidates),'sample_index':index,'missing':index is None})
    indices=sorted({c['sample_index'] for c in cases if c['sample_index'] is not None})
    OUT.mkdir();(OUT/'raw96').mkdir()
    selection={'completed':True,'PASS':True,'policy':binding(FILES['policy'][0]),'source_COMPLETE':binding(FILES['fit_complete'][0]),'output_CSVs':{k:binding(FILES[k][0]) for k in ['control_csv','treatment_csv']},'selection_frozen_before_raw_RGB_hash_header_or_numerical_read':True,'bucket_slots':12,'selected_unique_rows':len(indices),'missing_buckets':sum(c['missing'] for c in cases),'unique_sample_indices':indices,'cases':cases,'minimum_sample_index_no_replacement':True,'source_internal_DEV_reused_not_blind':True,'new_CF_training_images':False}
    selection_binding=write_json(OUT/'CASE_IDS_BEFORE_PIXELS.json',selection)
    # Only now is the declared rawRGB file opened, first opaque SHA then header.
    digest=hashlib.sha256();size=0
    fd=os.open(safe(RAW[0]),os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk);size+=len(chunk)
    if digest.hexdigest()!=RAW[1]:raise ValueError('Original rawRGB opaque hash')
    array=np.load(safe(RAW[0]),mmap_mode='r',allow_pickle=False)
    if array.dtype!=np.uint8 or array.shape!=(7617,3,96,96):raise ValueError('Exact rawCHW96 header')
    # Metadata-only YCB columns; its historical model-output columns are unused.
    ycb={int(row['sample_index']):{key:row[key] for key in ['row_id','sample_index','role','associated_r_state','q_ownership_diagnostic_only','q_x','q_y','r_x_GT_diagnostic_only','r_y_GT_diagnostic_only','q_reference_distance_px','L','reason']} for row in csv.DictReader(io.StringIO(raw_text['YCB_teacher_ref'].decode()))}
    pixels={};raw_bindings={};decode_proofs=[];case_data=[];outputs=[selection_binding]
    for i in indices:
        d=diag[i]
        if d['fit'] is not False or d['available'] is not True:raise ValueError('Only selected original available DEV rows numerically decode')
        chw=np.array(array[i],copy=True);chw_sha=hashlib.sha256(np.ascontiguousarray(chw).tobytes()).hexdigest()
        if chw_sha!=d['pixel_sha256']:raise ValueError('Selected original CHW pixel SHA')
        hwc=chw.transpose(1,2,0).copy();p=OUT/'raw96'/f'SAMPLE_{i:04d}_RAW96.png';Image.fromarray(hwc,'RGB').save(p)
        with Image.open(safe(p)) as image:decoded=np.asarray(image.convert('RGB')).copy()
        restored=np.ascontiguousarray(decoded.transpose(2,0,1));restored_sha=hashlib.sha256(restored.tobytes()).hexdigest()
        if not np.array_equal(restored,chw) or restored_sha!=chw_sha:raise ValueError('PNG roundtrip exact CHW source pixels')
        pixels[i]=hwc;raw_bindings[i]=binding(p);outputs.append(raw_bindings[i]);decode_proofs.append({'sample_index':i,'source_CHW_sha256':chw_sha,'PNG_decoded_HWC_sha256':hashlib.sha256(decoded.tobytes()).hexdigest(),'PNG_reconstructed_CHW_sha256':restored_sha,'exact_original_CHW_restored':True})
    for case in cases:
        item=dict(case);i=case['sample_index']
        if i is None:case_data.append(item);continue
        a,b=control[i],treatment[i];d=diag[i];query=np.asarray(d['query_xy'],np.float64);L=float(d['L'])
        reference=d.get('reference_xy_GT_diagnostic_only');owner='UNKNOWN';ref_authority='Saved SYN witness0 diagnostic reference'
        if d['family_code']==1:
            y=ycb[i]
            if y['row_id']!=d['row_id'] or y['role']!='DEV' or y['associated_r_state']!=d['original_state'] or float(y['L'])!=L or [float(y['q_x']),float(y['q_y'])]!=query.tolist():raise ValueError('YCB teacher-reference original q/L identity')
            reference=[float(y['r_x_GT_diagnostic_only']),float(y['r_y_GT_diagnostic_only'])] if y['r_x_GT_diagnostic_only'] and y['r_y_GT_diagnostic_only'] else None
            owner=y['q_ownership_diagnostic_only'];ref_authority='Saved V234 YCB teacher-only associated-r diagnostic CSV'
        ref=None if reference is None else np.asarray(reference,np.float64)
        local=None if ref is None else ((ref-query)/L*96.+47.5).tolist();distance=None if ref is None else float(np.linalg.norm(ref-query))
        truth='D' if int(b['target'])==0 else 'E';sourceid=d.get('source_teacher_key') if d['family_code']==0 else {'scene':d['scene_id'],'frame':d['source_frame_id'],'instance':d['source_instance_id'],'object':d['object_id'],'aux_slot':d['logical_aux_slot']}
        item.update(row_id=d['row_id'],family=d['family'],native_slot=d.get('native_slot'),native_mapping='SOURCE procedural witness0' if d['family_code']==0 else 'UNKNOWN',source_context=sourceid,role='INTERNAL_SOURCE_DEV',original_outer_role=d['original_component_role'],target=truth,target_referent='SYN intended-native witness0' if d['family_code']==0 else 'YCB associated-r surface',original_q_xy=query.tolist(),original_L=L,raw96_q_xy=[47.5,47.5],teacher_reference_xy=None if ref is None else ref.tolist(),teacher_reference_RAW96_xy=local,q_reference_distance_px=distance,reference_authority=ref_authority,reference_overlay_is_not_input_recenter_or_q_owner_label=True,q_ownership_diagnostic_only=owner,valid_fractions=d['valid_fractions'],central4=d['central4'],control={k:a[k] for k in ['prediction','logit0','logit1','selected_family_logit']},treatment={k:b[k] for k in ['prediction','logit0','logit1','selected_family_logit']},raw96_PNG=raw_bindings[i],original_CHW_pixel_sha256=d['pixel_sha256'])
        folder=OUT/f"CASE_{case['case_number']:02d}_SAMPLE_{i:04d}";folder.mkdir();card=folder/'CARD.png'
        fig=plt.figure(figsize=(14.4,5.2),dpi=140);gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.55],wspace=.24)
        ax=fig.add_subplot(gs[0,0]);decorate(ax,pixels[i],None,False);ax.set_title('Original RAW96 (unchanged)',fontsize=11)
        ax=fig.add_subplot(gs[0,1]);decorate(ax,pixels[i],local,True);ax.set_title('q cyan / teacher ref magenta',fontsize=11)
        ax=fig.add_subplot(gs[0,2]);ax.axis('off')
        lines=[f"sample {i} | native {d.get('native_slot')}" if d['family_code']==0 else f"sample {i} | native UNKNOWN / aux {d['logical_aux_slot']}",*textwrap.wrap(d['row_id'],43),f"Truth @ associated reference: {truth}",f"CONTROL: {a['prediction']}   z={float(a['selected_family_logit']):+.5f}",f"DATA-CF: {b['prediction']}   z={float(b['selected_family_logit']):+.5f}",f"CONTROL (z0,z1)=({float(a['logit0']):+.3f},{float(a['logit1']):+.3f})",f"DATA-CF (z0,z1)=({float(b['logit0']):+.3f},{float(b['logit1']):+.3f})",f"original q=({query[0]:.3f},{query[1]:.3f})  L={L:.3f}",f"ref local={tuple(round(v,3) for v in local)}" if local else 'reference UNKNOWN',f"q-ref distance={distance:.3f}px" if distance is not None else 'distance UNKNOWN',f"q-owner diagnostic: {owner}",f"valid fractions={d['valid_fractions']}",f"Bucket population={case['eligible_population']}; minimum index"]
        ax.text(0,.98,'\n'.join(lines),va='top',family='monospace',fontsize=9.5,linespacing=1.5)
        fig.suptitle(f"B{case['case_number']:02d}  {case['bucket_name']}",fontsize=14,y=.99)
        fig.text(.5,.03,'Original SOURCE internal DEV (reused, not blind) | Not a new CF training image | Reference is teacher-side; not literal-q ownership or new gold',ha='center',fontsize=9)
        fig.subplots_adjust(top=.87,bottom=.12,left=.025,right=.98);fig.savefig(card);plt.close(fig)
        item['card']=binding(card);outputs.append(item['card']);case_data.append(item)
    fig,axes=plt.subplots(4,3,figsize=(13.5,18.5),dpi=150)
    for ax,item in zip(axes.flat,case_data):
        if item['missing']:
            ax.axis('off');ax.text(.5,.5,f"B{item['case_number']:02d} {item['bucket_name']}\nMISSING (population 0)\nNo substitute",ha='center',va='center',fontsize=12);continue
        i=item['sample_index'];decorate(ax,pixels[i],item['teacher_reference_RAW96_xy'],True)
        a=item['control'];b=item['treatment']
        ax.set_title(f"B{item['case_number']:02d} {item['bucket_name']} N={item['eligible_population']}\nsample {i}  Truth:{item['target']}  CTRL:{a['prediction']}  CF:{b['prediction']}\nz {float(a['selected_family_logit']):+.3f} -> {float(b['selected_family_logit']):+.3f} | q-r {item['q_reference_distance_px']:.2f}px",fontsize=9.5,pad=8)
    fig.suptitle(f'V235 original SOURCE DEV: fixed12 buckets / {len(indices)} unique crops\nCyan original q; magenta associated teacher reference. No recrop or new inference.',fontsize=13,y=.995)
    fig.subplots_adjust(top=.945,bottom=.02,left=.04,right=.98,wspace=.25,hspace=.38)
    contact=OUT/'FIXED12_SOURCE_DEV.png';fig.savefig(contact);plt.close(fig);outputs.append(binding(contact))
    data_binding=write_json(OUT/'CASE_DATA.json',{'completed':True,'PASS':True,'selection':selection_binding,'cases':case_data,'unique_original_crops':len(indices),'missing_buckets':sum(c['missing'] for c in case_data),'decode_proofs':decode_proofs,'actual_numerical_RGB_rows':indices,'all_reference_overlays_teacher_only':True,'SOURCE_internal_DEV_reused_not_freshblind':True,'new_CF_training_images':False,'new_visibility_annotations_or_cause_certification':False});outputs.append(data_binding)
    csv_path=OUT/'CASES.csv';columns=['case_number','bucket_name','eligible_population','sample_index','missing','family','native_slot','target','control','treatment','control_logit','treatment_logit','q_x','q_y','L','reference_x','reference_y','reference_raw96_x','reference_raw96_y','q_reference_distance_px','raw96_path','card_path']
    with csv_path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for item in case_data:
            row={k:item[k] for k in columns[:5]}
            if not item['missing']:
                ref=item['teacher_reference_xy'];local=item['teacher_reference_RAW96_xy']
                row.update(family=item['family'],native_slot=item['native_slot'],target=item['target'],control=item['control']['prediction'],treatment=item['treatment']['prediction'],control_logit=item['control']['selected_family_logit'],treatment_logit=item['treatment']['selected_family_logit'],q_x=item['original_q_xy'][0],q_y=item['original_q_xy'][1],L=item['original_L'],reference_x=None if ref is None else ref[0],reference_y=None if ref is None else ref[1],reference_raw96_x=None if local is None else local[0],reference_raw96_y=None if local is None else local[1],q_reference_distance_px=item['q_reference_distance_px'],raw96_path=str(Path(item['raw96_PNG']['path']).relative_to(OUT)),card_path=str(Path(item['card']['path']).relative_to(OUT)))
            writer.writerow(row)
    outputs.append(binding(csv_path))
    body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V235 SOURCE 고정 사례</title><style>body{font-family:sans-serif;max-width:1500px;margin:30px auto;padding:0 18px;line-height:1.6}img{max-width:100%;height:auto}.case{border-top:1px solid #aaa;margin-top:28px}code{overflow-wrap:anywhere}</style><h1>V235: 원 SOURCE DEV 고정12 bucket</h1>',f'<p>최소 sample_index 규칙을 픽셀 읽기 전에 봉인했습니다. 12 bucket 중 {len(indices)}개 고유 crop과 빈 bucket {sum(c["missing"] for c in cases)}개입니다. 원 q/L·RGB96를 유지했고 새 CF 학습이미지가 아닙니다. SOURCE 내부 DEV는 재사용·노출된 자료이며 fresh blind 또는 현재 실사 평가가 아닙니다.</p>','<p>cyan은 원 q(47.5,47.5), magenta는 saved teacher reference를 원 crop에 대수적으로 표시한 것입니다. SYN intended-native witness0와 YCB associated-r의 label을 구분합니다. 화면의 다른 물체·q-owner만으로 D/E를 새로 판정하지 않습니다. Reference recrop·좌표 보정·새 모델 forward·원인 인증은 없습니다.</p>','<p><a href="CASE_IDS_BEFORE_PIXELS.json">사전 사례 ID</a> · <a href="CASE_DATA.json">수치·입력/PNG SHA</a> · <a href="CASES.csv">CSV</a></p>','<img src="FIXED12_SOURCE_DEV.png" alt="fixed12 SOURCE DEV cases">']
    for item in case_data:
        body.append('<section class="case"><h2>'+html.escape(f"B{item['case_number']:02d} {item['bucket_name']} — 모집단 {item['eligible_population']}")+'</h2>')
        if item['missing']:body.append('<p>해당 bucket은 비어 있어 대체 사례를 만들지 않았습니다.</p></section>');continue
        card_relative=str(Path(item['card']['path']).relative_to(OUT));raw_relative=str(Path(item['raw96_PNG']['path']).relative_to(OUT))
        body.append('<p>'+html.escape(f"sample_index={item['sample_index']} / {item['row_id']}")+'</p><img src="'+card_relative+'" alt="saved SOURCE case"><p><a href="'+raw_relative+'">표시를 넣지 않은 원 RAW96 PNG</a></p></section>')
    body.append('</html>');page=OUT/'PORTABLE_GALLERY_KO.html'
    with page.open('x') as stream:stream.write('\n'.join(body))
    outputs.append(binding(page))
    complete={'completed':True,'PASS':True,'code':binding(Path(__file__).absolute()),'inputs':{k:binding(v[0]) for k,v in FILES.items()},'raw_RGB_binding':{'path':str(RAW[0]),'sha256':RAW[1],'size_bytes':size},'selection':selection_binding,'cases':len(cases),'unique_crops':len(indices),'missing_buckets':sum(c['missing'] for c in cases),'numerically_decoded_original_sample_indices':indices,'original_CHW_vs_PNG_roundtrip_all_exact':True,'outputs':outputs,'SOURCE_internal_DEV_not_blind':True,'new_CF_training_images':False,'weight_prefix_original_GT_donor_current_held_reads':0,'new_model_forward_or_training':0,'reference_display_never_recrop_or_new_label':True,'versions':{'numpy':np.__version__,'Pillow':PIL.__version__,'matplotlib':matplotlib.__version__}}
    write_json(OUT/'COMPLETE.json',complete);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(OUT/'COMPLETE.json'),'selected_unique':len(indices),'missing':sum(c['missing'] for c in cases),'contact':binding(contact),'HTML':binding(page)}),flush=True)


if __name__=='__main__':
    try:build()
    except BaseException:
        if OUT.exists() and not (OUT/'FAILED_EXECUTION.json').exists():write_json(OUT/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()})
        raise
    finally:
        temporary=H/'_temporary_matplotlib_config'
        if temporary.exists():shutil.rmtree(temporary)
