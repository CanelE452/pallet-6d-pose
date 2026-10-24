"""Supplemental outcome-selected source error illustration; saved96 only.

No original fullRGB reads, crop sampling, models, fitting or label creation.
Preparation reads JSON completion metadata/code only and invented pixels.
"""
from pathlib import Path
import argparse,hashlib,json,tempfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent
SCOPE=ROOT/'ROOT_SCOPE_BEFORE_NEW_COMMON_QUERY_OR_PUBLIC_PIXELS.json'
SENS=ROOT/'saved_crop_geometry_sensitivity_v1/COMPLETE.json'
HEADS=ROOT/'frozen_common_query_heads_v1/COMPLETE.json'
COMMON=ROOT/'common_query_diagnostic_v1/COMPLETE.json'
EVAL=ROOT/'saved_common_query_evaluation_v1/COMPLETE.json'
PROTOCOL=HERE/'PROTOCOL_BEFORE_SAVED_CROP_GEOMETRY_PIXEL_BODIES.json'
CASE=HERE/'SELECTED_CASE_METADATA_BEFORE_CROP_BODIES.json'
ARMS=('C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX')
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def sha(p):
    p=Path(p).resolve();require(p.is_relative_to(P) and not any(s in str(p).lower() for s in ('_v112','_v177','_v178','_v169','current223','synth_val','source_validation','buffers.npz','canonical_rgb')),'Forbidden non-TRAIN/fullRGB path')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):require(sha(b['path'])==b['sha256'],'SHA '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(b,keys):
    with np.load(checked(b),allow_pickle=False) as z:return {k:z[k].copy() for k in keys}
def same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))

def select_case(rr):
    keep=[r for r in rr if r['arm']==ARMS[0] and r['both_support_truth_known_and_same'] and r['original_available'] and r['common_available'] and r['original_correct'] and not r['common_correct'] and r['q_shift_image_px'] is not None and r['q_shift_image_px']>0]
    require(keep,'Nonempty predeclared positive-shift lost case set')
    return min(keep,key=lambda r:(r['q_shift_image_px'],r['base_index'],r['native_slot'],r['diagnostic_row_index']))

def font(size):return ImageFont.truetype(FONT,size)
def wrapped(draw,xy,text,width,size=20,fill='#e2ebf6',gap=6):
    x,y=xy;f=font(size);line=''
    for word in str(text).split():
        proposed=word if not line else line+' '+word
        if draw.textlength(proposed,font=f)>width and line:
            draw.text((x,y),line,font=f,fill=fill);y+=size+gap;line=word
        else:line=proposed
    if line:draw.text((x,y),line,font=f,fill=fill);y+=size+gap
    return y

def clean_png(p,a):
    require(a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3,'Original saved RGB uint8')
    Image.fromarray(a).save(p);back=np.asarray(Image.open(p).convert('RGB'))
    require(np.array_equal(back,a),'Clean PNG exact decoded source array')
    return dict(**bind(p),decoded_array_sha256=ash(back),shape=list(a.shape))

def card_image(case,columns):
    im=Image.new('RGB',(1980,1280),'#0e1828');d=ImageDraw.Draw(im)
    wrapped(d,(24,18),'Saved crop geometry: outcome-selected source-TRAIN error example',1920,32)
    wrapped(d,(24,66),'BASE59 native1; smallest positive q shift among C50 correct -> wrong with same known support truth. Not a representative success rate.',1920,21)
    wrapped(d,(24,103),'q shift %.6f image px; delta L %+.6f px. q and L changed together; no new image sampling or model call.'%(case['q_shift_image_px'],case['delta_L_image_px']),1920,21)
    for k,c in enumerate(columns):
        left=24+k*652
        wrapped(d,(left,151),c['title'],620,23)
        wrapped(d,(left,188),'q=(%.6f, %.6f); L=%.6f'%(*c['q'],c['L']),620,17)
        x=left+118
        im.paste(Image.fromarray(c['rgb']).resize((384,384),Image.Resampling.NEAREST),(x,238))
        wrapped(d,(left+118,630),'Saved RGB96 x4 nearest (clean)',500,18)
        center=c['rgb'][32:64,32:64]
        im.paste(Image.fromarray(center).resize((256,256),Image.Resampling.NEAREST),(left+182,677))
        wrapped(d,(left+118,942),'Exact central32 [32:64] x8 nearest',500,18)
        wrapped(d,(left+14,981),'Support truth V/F: '+str(c['target'])+'; input '+('AVAILABLE' if c['available'] else 'U'),600,20)
        y=1024
        for arm in ARMS:
            a=c['models'][arm]
            y=wrapped(d,(left+14,y),arm[:3]+' logits(V,F)=[%.6f, %.6f] -> %s; proxy=%d'%(*a['logits'],a['prediction'],a['proxy']),600,18)+5
    wrapped(d,(24,1160),'Two BEFORE columns use the SAME underlying image binding. This illustrates this frozen model response to existing crop geometry, not q-only causality, coordinate-error gold, native visibility or general performance.',1920,19)
    wrapped(d,(24,1220),'C51 also uses SAM: original-BEFORE and common-COVER q8 prompts can differ. GT support is diagnostic only; no new labels, models or threshold choices.',1920,18)
    return im

def prepare():
    sens=read(SENS);heads=read(HEADS);common=read(COMMON);ev=read(EVAL)
    require(sens['completed'] and sens['PASS'] and heads['PASS'] and common['PASS'] and ev['PASS'],'Completed saved source receipts only')
    chosen=select_case(sens['arms'][ARMS[0]]['lost_cases'])
    require((chosen['base_index'],chosen['native_slot'],chosen['diagnostic_row_index'],chosen['anchor_original_global_index'])==(59,1,1889,1897),'Root predeclared BASE59/native1')
    write(CASE,dict(selection_rule='C50 originalcorrect -> wrong, original/common both available, bothsupporttruthknownandsame, qshift>0; minimum(qshift,base,native,diagnosticindex)',outcome_selected=True,not_representative_success_or_causal_rate=True,source_complete=bind(SENS),selected=chosen))
    tests=[]
    def passed(name,ok):require(ok,name);tests.append(dict(name=name,PASS=True))
    toy=dict(chosen);toy.update(q_shift_image_px=.8,base_index=1,diagnostic_row_index=1)
    rejected=dict(chosen);rejected.update(q_shift_image_px=.01,both_support_truth_known_and_same=False)
    wrongarm=dict(chosen);wrongarm.update(q_shift_image_px=.001,arm=ARMS[1])
    passed('selector_positive_shift_same_truth_C50_only_minimum',select_case([toy,rejected,wrongarm,chosen])==chosen)
    source=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3);backup=source.copy()
    cols=[]
    for title in ['BEFORE: original production q/L','BEFORE: common COVER q/L','COVER image: common q/L']:
        cols.append(dict(title=title,q=[1.,2.],L=4.,rgb=source.copy(),target=[1,0],available=True,models={a:dict(logits=[.125,-.25],prediction=[1,0],proxy=0) for a in ARMS}))
    im=card_image(chosen,cols)
    passed('actualPIL_readable3column_geometry_no_source_mutation',im.size==(1980,1280) and np.array_equal(source,backup))
    for i in range(3):
        x=24+i*652+118
        passed('nearest4_panel_exact_invented_column'+str(i),np.array_equal(np.asarray(im)[238:622,x:x+384],np.repeat(np.repeat(source,4,0),4,1)))
    with tempfile.TemporaryDirectory(prefix='V182_invented_clean96_') as temp:
        saved=clean_png(Path(temp)/'clean.png',source) if False else None
        # bind() intentionally permits only project paths, so PNG roundtrip is
        # checked in-memory for invented data instead of widening file access.
        import io
        stream=io.BytesIO();Image.fromarray(source).save(stream,format='PNG');stream.seek(0)
        passed('invented_cleanPNG_decode_exact_and_central32_unmodified',np.array_equal(np.asarray(Image.open(stream).convert('RGB')),source) and np.array_equal(cols[0]['rgb'][32:64,32:64],backup[32:64,32:64]))
    diff=np.abs(source.astype(np.int16)-(source^np.uint8(1)).astype(np.int16)).astype(np.uint8)
    passed('display_diff_amplification_uint16_no_wrap_or_input_change',np.array_equal(np.minimum(diff.astype(np.uint16)*8,255).astype(np.uint8),np.full_like(source,8)) and np.array_equal(source,backup))
    checks=HERE/'GENERATED_INVENTED_CHECKS.json';write(checks,dict(completed=True,PASS=True,tests=tests,new_saved_crop_target_or_score_payload_reads=0))
    protocol=dict(completed=True,prepared_only=True,scope=bind(SCOPE),code=bind(__file__),sensitivity_complete=bind(SENS),common_complete=bind(COMMON),prediction_complete=bind(HEADS),saved_evaluation_complete=bind(EVAL),selected_case=bind(CASE),checks=bind(checks),
        outcome_selected_error_illustration=True,not_success_rate_or_representative=True,
        actual_body_sources='Only sealed original V181 TRAIN savedINPUTS/TARGETS/outputNPZ and V182 commonINPUTS/TARGETS/outputNPZ/metadata/sensitivityledger. Never open sourcefullRGB/GTbuffers/weights/SAM/model.',
        columns=['SAME BEFORE image, originalproductionq/L saved96','SAME BEFORE image, commonCOVERq/L saved96','COVER image, commonCOVERq/L saved96'],
        pixels='Separate exact clean96 PNG and exact[32:64] central32 PNG; enlarged x4/x8 nearest copies. Optional abs RGBdiff x8 clipped only for display; originals unchanged.',
        facts='Old/new fixedend V/F targets and bothfrozenhead supportlogits/pred/proxy copied/decoder-checked from sealed savedarrays; native/positiongold not created.',
        output=str(ROOT/'crop_geometry_visual_v1'),new_crop_sampler_or_model_forward_or_fit=0,private_held_VAL_or_V112_access=0,browser_render_verified=False,actual_PNG_view_QA='PENDING separate root view_image after render',selected_model=None,fully_solved=False)
    write(PROTOCOL,protocol)
    template=HERE/'ROOT_ACTUAL_CROP_GEOMETRY_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_saved_crop_geometry_render=False,source_only=True,code=bind(__file__),protocol=bind(PROTOCOL),output_dir=str(ROOT/'crop_geometry_visual_v1')))
    write(HERE/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(checks),selected_case=bind(CASE),entry_template=bind(template),new_crop_pixel_target_or_score_payload_reads=0))
    print(json.dumps(dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL))))

def run(entry_path):
    e=read(entry_path);require(e.get('approved_saved_crop_geometry_render') is True and e.get('source_only') is True,'Separate root render entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL),'Entry frozen code/protocol')
    p=read(PROTOCOL)
    for key in ('scope','sensitivity_complete','common_complete','prediction_complete','saved_evaluation_complete','selected_case','checks'):checked(p[key])
    sens=read(SENS);common=read(COMMON);heads=read(HEADS);ev=read(EVAL);fixed=read(CASE)
    require(all(d['completed'] and d['PASS'] for d in (sens,common,heads,ev)),'Saved completion receipts')
    chosen=select_case(rows(checked(sens['rows'])));require(chosen==fixed['selected'],'Same outcome-selected frozen case from full2048ledger')
    out=Path(e['output_dir']);require(out==ROOT/'crop_geometry_visual_v1' and not out.exists(),'New fixed phase output only')
    out.mkdir();write(out/'SELECTED_CASE_BEFORE_CROP_PAYLOAD_READ.json',fixed)
    oldx=load(sens['original_source_inputs'],set(META)|{'rgb96','valid96'});oldy=load(sens['original_source_targets'],['support_target'])
    newx=load(common['inputs'],set(META)|{'rgb96','valid96'});newy=load(common['targets'],['support_target']);metadata=rows(checked(common['metadata']))
    require(len(metadata)==4096,'All common metadata')
    i=chosen['diagnostic_row_index'];r=metadata[i];require((r['base_index'],r['native_slot'],r['variant'])==(59,1,'BEFORE'),'Selected common BEFORE identity')
    require(r['source_RGB_binding']==chosen['same_saved_BEFORE_RGB_binding'],'SAME BEFORE image by sealed source binding only; sourcefile never opened')
    cover=[j for j,z in enumerate(metadata) if z['base_index']==59 and z['native_slot']==1 and z['variant']=='COVER'];require(len(cover)==1,'Unique fixed COVER case');c=cover[0]
    oldmap={int(v):j for j,v in enumerate(oldx['original_global_index'])};o=oldmap[chosen['source_image_original_global_index']]
    require(int(newx['original_global_index'][i])==i and int(newx['original_global_index'][c])==c==1897,'New diagnostic IDs')
    require(str(oldx['row_id'][o])==r['source_image_original_query_id'] and str(newx['row_id'][i])==r['row_id'],'Original/new identity')
    require(same(oldx['query_xy'][o],chosen['original_q']) and float(oldx['L'][o])==chosen['original_L'],'Original production qL')
    require(same(newx['query_xy'][i],chosen['common_q']) and float(newx['L'][i])==chosen['common_L'] and same(newx['query_xy'][i],newx['query_xy'][c]) and newx['L'][i]==newx['L'][c],'CommonCOVERqL fixed acrossimages')
    require(oldy['support_target'][o].tolist()==chosen['original_truth'] and newy['support_target'][i].tolist()==chosen['common_truth'],'Original/common targets unchanged')
    fit=read(checked(heads['frozen_V181_fit']));scorekeys=set(META)|{'raw_support_logits','support_prediction','regional_support_proxy_D0_E1_Uminus1','support_order'}
    scores={a:(load(fit['arms'][a]['outputs']['SYNTH_TRAIN4096'],scorekeys),load(heads['arms'][a]['outputs'],scorekeys)) for a in ARMS}
    columns=[];facts=[]
    for label,x,y,idx,scoreid in [('BEFORE: original production q/L',oldx,oldy,o,0),('BEFORE: common COVER q/L',newx,newy,i,1),('COVER image: common q/L',newx,newy,c,1)]:
        require(x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(4096,96,96,3) and x['valid96'].dtype==bool,'Saved crop schema')
        require(y['support_target'].dtype==np.int8 and y['support_target'].shape==(4096,2),'Source support target schema')
        mods={}
        for arm in ARMS:
            z=scores[arm][scoreid]
            for key in META:require(same(z[key][idx],x[key][idx]),'Saved score input identity '+key)
            require(z['support_order'].tolist()==['visible_end_support','foreign_front_support'],'Task order')
            av=bool(x['original_query_available'][idx]);raw=z['raw_support_logits'][idx];pred=z['support_prediction'][idx]
            require(raw.dtype==np.float32 and np.isfinite(raw).all() if av else np.isnan(raw).all(),'RawFP32/U')
            expected=(raw>0).astype(np.int8) if av else np.full(2,-1,np.int8);require(same(pred,expected),'Saved strict>0 support decoder')
            proxy=-1 if not av else 1 if pred[1]==1 else 0 if pred[0]==1 else -1
            require(proxy==int(z['regional_support_proxy_D0_E1_Uminus1'][idx]),'Saved proxy decoder')
            mods[arm]=dict(logits=raw.tolist(),prediction=pred.tolist(),proxy=proxy)
        image=x['rgb96'][idx];prefix='COLUMN%d'%(len(columns)+1)
        clean=clean_png(out/(prefix+'_CLEAN96.png'),image);center=clean_png(out/(prefix+'_CENTRAL32.png'),image[32:64,32:64])
        column=dict(title=label,q=x['query_xy'][idx].tolist(),L=float(x['L'][idx]),rgb=image,target=y['support_target'][idx].tolist(),available=bool(x['original_query_available'][idx]),models=mods)
        columns.append(column);facts.append(dict(title=label,row_id=str(x['row_id'][idx]),original_global_index=int(x['original_global_index'][idx]),native_slot=int(x['native_slot'][idx]),query_xy=column['q'],L=column['L'],support_target=column['target'],original_query_available=column['available'],models=mods,clean96=clean,central32=center,valid96_sha256=ash(x['valid96'][idx]),valid96_count=int(x['valid96'][idx].sum())))
    require(columns[0]['models'][ARMS[0]]['prediction']==chosen['original_prediction'] and columns[1]['models'][ARMS[0]]['prediction']==chosen['common_prediction'],'Selected C50 lost event facts')
    card=card_image(chosen,columns);card.save(out/'CARD.png')
    absolute=np.abs(columns[0]['rgb'].astype(np.int16)-columns[1]['rgb'].astype(np.int16)).astype(np.uint8)
    amplified=np.minimum(absolute.astype(np.uint16)*8,255).astype(np.uint8);Image.fromarray(amplified).save(out/'BEFORE_ABS_RGB_DIFF_X8_DISPLAY_ONLY.png')
    diff=dict(changed_RGB_pixels=int(np.any(absolute!=0,2).sum()),mean_abs_channel_difference=float(absolute.mean()),maximum_abs_channel_difference=int(absolute.max()),display_amplification=8,display_clipped_to255=True,raw_inputs_unchanged=True)
    write(out/'SAVED_FACTS.json',dict(selected_case=fixed,columns=facts,difference=diff,no_q_only_causal_or_representative_rate_claim=True,source_pixel_labels_not_physical_native_gold=True))
    page='<!doctype html><meta charset="utf-8"><title>Outcome-selected saved crop geometry</title><style>body{font:18px/1.6 sans-serif;background:#0e1828;color:#e2ebf6;margin:24px}img{max-width:100%;height:auto}a{color:#a4d0ff}.pixel{width:384px;image-rendering:pixelated}</style><h1>결과를 보고 선택한 source 오류 설명 사례</h1><p>C50의 같은 known support 정답에서 원 crop 정답→common crop 오답으로 바뀐 항목 중 양수 q 이동이 가장 작은 사례입니다. 성공률 대표 표본이나 q만 바꾼 인과 실험이 아닙니다. q와 L이 함께 달라졌고 C51은 SAM prompt도 다를 수 있습니다. 원 fullRGB 재읽기·새 crop sampling·모델 호출·새 라벨은 없습니다.</p><a href="CARD.png"><img src="CARD.png"></a><h2>표시용 차이</h2><p>같은 BEFORE 영상의 두 저장 crop |RGB 차이| ×8, 255에서 포화. 모델 입력이 아닙니다.</p><img class="pixel" src="BEFORE_ABS_RGB_DIFF_X8_DISPLAY_ONLY.png"><p><a href="SAVED_FACTS.json">정확한 저장 사실과 clean96/central32 SHA</a></p>'
    (out/'index.html').write_text(page)
    write(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),selected_case=bind(CASE),outcome_selected_error_illustration=True,representative_success_rate=False,case_count=1,columns=3,clean96_count=3,central32_count=3,all_cleanPNG_array_parity=True,card=bind(out/'CARD.png'),facts=bind(out/'SAVED_FACTS.json'),index=bind(out/'index.html'),actual_visual_QA='PENDING root actual view_image',browser_render_verified=False,new_source_RGB_read_crop_sampling_model_fit_or_labels=0,selected_model=None,fully_solved=False))
    print(json.dumps(dict(completed=True,PASS=True,complete=bind(out/'COMPLETE.json'))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--root-entry',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.root_entry)
