"""Saved V193 source pairs only. --prepare uses invented arrays; actual --entry is root gated."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,html,json,math,os,stat,textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont

H=Path(__file__).absolute().parent;V=H.parent
PRODUCER=V/'real_end_opposite_sham_preparation_c/build_real_end_opposite_sham_pairs.py'
PRODUCER_PROTOCOL=PRODUCER.with_name('PROTOCOL_BEFORE_REAL_END_PAIR_SOURCE_PIXELS.json')
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_SOURCE_PAIRS_OR_PIXELS.json'
OUT=V/'source_pair_visual_readout_v1'
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
KEYS=('rgb96','valid96','query_xy','L','native_slot','query_id','original_global_index','query_available','native_visibility_target','point_quality_target','owner_target')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def need(v,msg):
    if not bool(v):raise AssertionError(msg)
def safe(path):
    p=Path(path);need(p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Safe explicit path before access');return p
def opened(path):
    p=safe(path);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for name in p.parts[1:-1]:
            z=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=z
        z=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(z).st_mode),'Regular file');return os.fdopen(z,'rb')
def sha(path):
    h=hashlib.sha256()
    with opened(path) as f:
        for z in iter(lambda:f.read(1024**2),b''):h.update(z)
    return h.hexdigest()
def bind(path):return dict(path=str(path),sha256=sha(path))
def checked(binding):
    p=safe(binding['path']);need(sha(p)==binding['sha256'],'Bound bytes drift');return p
def read(binding):
    with opened(checked(binding)) as f:return json.load(f)
def write(path,obj):
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def validate_packet(a,frame):
    need(set(a)==set(KEYS),'Exact stored packet whitelist; fullRGB/alpha/context not decoded')
    need(a['rgb96'].dtype==np.uint8 and a['rgb96'].shape==(8,96,96,3),'All8 original RGB96')
    need(a['valid96'].dtype==bool and a['valid96'].shape==(8,96,96),'All8 original valid96')
    need(a['query_xy'].dtype==np.float64 and a['L'].dtype==np.float64,'Original F64 sampling metadata')
    need(np.array_equal(a['query_xy'],np.asarray(frame['query_xy'],np.float64),equal_nan=True) and np.array_equal(a['L'],np.asarray(frame['L'],np.float64),equal_nan=True),'Original q/L unchanged')
    need(a['native_slot'].tolist()==list(range(8)) and a['query_id'].tolist()==frame['query_id'] and a['original_global_index'].tolist()==frame['global_index'],'All8 original identities/order')
    need(a['query_available'].dtype==bool and a['query_available'].tolist()==frame['native_continuous_valid'],'Original inputU gate preserved')
    need(all(a[k].shape==(8,) and (a[k]==-1).all() for k in ('native_visibility_target','point_quality_target','owner_target')),'No new native/quality/owner labels')
    return True

def packet(binding,frame,producer_root):
    path=safe(binding['path']);need(producer_root in path.parents,'Only completed V193 saved pair packets')
    with opened(path) as f:
        h=hashlib.sha256()
        for z in iter(lambda:f.read(1024**2),b''):h.update(z)
        need(h.hexdigest()==binding['sha256'],'Saved packet SHA');f.seek(0)
        with np.load(f,allow_pickle=False) as z:
            need(set(KEYS)<=set(z.files),'Stored schema');a={k:z[k].copy() for k in KEYS}
    validate_packet(a,frame);return a

def save_png(path,a):
    Image.fromarray(a).save(path)
    with Image.open(path) as im:back=np.asarray(im)
    need(exact(back,a),'Clean PNG decoded bytes exact');return dict(**bind(path),decoded_array_sha256=ash(a))

def weak_state(row,variant,k):
    if k!=row['original_input']['native_slot']:return 'UNKNOWN'
    return row['direct_end_weak_states'][('SHAM' if variant=='ADJACENT_SHAM' else variant)]

def draw_sheet(path,row,packets,frame,natives):
    font=ImageFont.truetype(FONT,13);title=ImageFont.truetype(FONT,18)
    canvas=Image.new('RGB',(1140,130+len(natives)*230+95),'white');d=ImageDraw.Draw(canvas)
    d.text((12,10),row['pair_id']+' / saved source weak-end triplet',font=title,fill='black')
    d.text((12,35),'Stored status: '+row['status'],font=font,fill='black')
    d.text((12,56),'DISPLAY COPY: cyan cross = original q at RGB96 index (47.5,47.5). Separate CLEAN96 PNGs are unmarked.',font=font,fill='black')
    d.text((12,77),'Raw markers: copied producer context only. Original q is not an endpoint annotation; central32 is not the weak-end target.',font=font,fill='black')
    policy=row.get('opposite_sham_policy',{}).get('selection','NO_GEOMETRY_PLAN')
    d.text((12,99),'SHAM policy: '+policy+'; declared assistant weak end only, not physical native gold. No training admission.',font=font,fill='black')
    for rr,k in enumerate(natives):
        y=125+rr*230
        for col,name in enumerate(VARIANTS):
            x=12+col*378;d.text((x,y),name+' / native'+str(k)+' / '+weak_state(row,name,k),font=font,fill='black')
            q=frame['query_xy'][k];L=frame['L'][k]
            d.text((x,y+20),'raw q=(%.3f,%.3f); L=%.3f'%(q[0],q[1],L),font=font,fill='black')
            a=packets.get(name)
            if a is None:
                d.text((x+8,y+65),'NO SAVED RGB / not constructed',font=font,fill='black')
                reason=row['variants'][name].get('reason',row['status'])
                for j,line in enumerate(textwrap.wrap(str(reason),44)[:5]):d.text((x+8,y+88+j*17),line,font=font,fill='black')
                continue
            display=Image.fromarray(a['rgb96'][k]).resize((160,160),Image.Resampling.NEAREST)
            marker=ImageDraw.Draw(display);c=(47.5+.5)*160/96-.5
            marker.line((c-5,c,c+5,c),fill='cyan',width=1);marker.line((c,c-5,c,c+5),fill='cyan',width=1)
            canvas.paste(display,(x,y+47))
            canvas.paste(Image.fromarray(a['rgb96'][k,32:64,32:64]).resize((128,128),Image.Resampling.NEAREST),(x+176,y+61))
            d.text((x,y+211),'marked RGB96 / unmarked central32; original inputU='+str(not bool(a['query_available'][k])),font=font,fill='black')
    y=130+len(natives)*230
    d.text((12,y),'Clean/source RGB bytes are preserved separately. Display enlargement is nearest-neighbor; no new crop sampling.',font=font,fill='black')
    d.text((12,y+23),'Missing RGB is text only. InputU, weak UNKNOWN, geometry failure and closed pixel proof are distinct.',font=font,fill='black')
    d.text((12,y+46),'Source controls only: no model output, heldout accuracy, transfer improvement, adoption or full-goal claim.',font=font,fill='black')
    canvas.save(path);return bind(path)

def census_rows(row,frame,packets):
    out=[]
    for name in VARIANTS:
        for k in range(8):
            out.append(dict(pair_id=row['pair_id'],case_id=row['case_id'],donor_id=row['donor_id'],variant=name,native_slot=k,
                original_global_index=frame['global_index'][k],query_id=frame['query_id'][k],original_query_xy=json.dumps(frame['query_xy'][k]),original_L=frame['L'][k],
                original_input_available=frame['native_continuous_valid'][k],pixel_packet_present=name in packets,
                selected_weak_end_slot=(k==row['original_input']['native_slot']),declared_direct_weak_state=weak_state(row,name,k),
                native_quality_owner_targets='ALL_MINUS1',status=row['status'],training_admission=False))
    return out

def csv_write(path,rows):
    need(bool(rows),'Nonempty complete table')
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(entry):
    e=read(bind(entry));need(e['approved_saved_source_visual_readout'] is True and e['automatic_retry'] is False,'Separate approved root saved-pixel entry')
    need(e['code']==bind(Path(__file__).absolute()) and e['protocol']['path']==str(PROTOCOL),'Frozen renderer code/protocol')
    p=read(e['protocol']);need(p['code']==e['code'],'Code closure');read(e['root_scope'])
    done=read(e['source_generation_complete']);need(done['completed'] and done['PASS'] and done['rows66']==66 and done['cases22']==22 and done['training_admission'] is False,'Complete66/22 source only')
    need(done['protocol']==p['producer_protocol'],'Exact source producer protocol');producer=read(done['protocol']);need(producer['code']==p['producer_code'],'Exact frozen source producer')
    source=read(producer['source_plan']);frames={f['frame_index']:f for f in source['frames']};attempts=source['attempts']
    need(len(frames)==21 and len(attempts)==66 and all(f['split']=='SOURCE_DEVELOPMENT' for f in frames.values()),'Original22cases/21sourceframes/66attempts')
    with opened(checked(done['manifest'])) as f:rows=[json.loads(line) for line in f]
    need([r['pair_id'] for r in rows]==[r['pair_id'] for r in attempts] and len({r['case_id'] for r in rows})==22,'All original rows/order; no outcome-selected subset')
    need(e['output_directory']==str(OUT),'Fresh fixed output');OUT.mkdir(exist_ok=False)
    lookup={a['pair_id']:a for a in attempts};producer_root=Path(e['source_generation_complete']['path']).parent
    led=[];stats=[];cases={};cards=[];pngs=[];md=['# V193 공개 source 끝부위 대조 자료 전체66시도','',
        '원22사례·21개 고유 source프레임·66donor시도와 native8/입력U/UNKNOWN/실패를 모두 포함한다. 새 독립 holdout이나 모델 개선 증거가 아니다. 사진 없는 경우 이유만 표시한다. clean96은 저장 배열 그대로이며 표시용 q marker와 분리했다. 원 q, 관찰한 finite end, central32의 재료 존재는 서로 다른 명제다. COVER/SHAM 모두 central32에 donor가 보여도 특정 끝의 실제 보간·차폐 proof를 대체하지 않는다.','']
    old5=0
    for row in rows:
        name=row['pair_id'];need(Path(name).name==name and '..' not in name and 'v112' not in name.casefold(),'Safe fixed original pair ID')
        folder=OUT/name;folder.mkdir();a=lookup[name];frame=frames[a['frame_index']];k=row['original_input']['native_slot']
        need(row['original_input']['original_row_id']==frame['query_id'][k] and row['original_input']['query_xy']==frame['query_xy'][k] and row['original_input']['L']==frame['L'][k],'Original selected q/L identity')
        need(row['training_admission'] is False and row['new_native_quality_owner_all8']==-1,'Stored source weak-only authority')
        packets={};clean={}
        for variant in VARIANTS:
            b=row['variants'][variant]
            if 'path' not in b:
                need(b.get('constructed') is False,'Explicit missing packet');continue
            z=packet(b,frame,producer_root);packets[variant]=z
            for native in range(8):
                target=folder/(variant+'_NATIVE%d_CLEAN96.png'%native);cl=save_png(target,z['rgb96'][native]);pngs.append(cl)
                target=folder/(variant+'_NATIVE%d_VALID96.png'%native);va=save_png(target,z['valid96'][native].astype(np.uint8)*255);pngs.append(va)
                clean[variant+':'+str(native)]={'clean':cl,'valid':va,'source_rgb96_array_sha256':ash(z['rgb96'][native]),'source_valid96_array_sha256':ash(z['valid96'][native])}
        closed=row.get('closed14_actual_proof',{});proof14=len(closed)==15 and all(bool(v) for key,v in closed.items() if key!='evidence_bindings') and bool(closed.get('evidence_bindings'))
        was_old=row['predecessor_attempt']['PASS']
        if was_old:
            old5+=1;need(row['PASS'] and all(all(z.values()) for z in row['old5_decoded_packet_parity'].values()),'Old5 stored decoded parity')
        if row['PASS']:need(proof14 and set(packets)==set(VARIANTS),'Supported weak pair needs stored14proof/all3packets')
        if row['status']=='UNKNOWN_RETAINED':need(not packets,'SemanticUNKNOWN has no fabricated RGB')
        card=draw_sheet(folder/'CONTACT_ALL8.png',row,packets,frame,list(range(8)));selected=draw_sheet(folder/'SELECTED_END_TRIPLET.png',row,packets,frame,[k]);cards.extend([card,selected])
        context=None
        if row.get('contact'):
            src=checked(row['contact']);need(producer_root in src.parents,'Producer-generated raw context only')
            with opened(src) as f:raw=f.read()
            target=folder/'PRODUCER_RAW_CONTEXT_CONTACT.png'
            with target.open('xb') as f:f.write(raw)
            context=bind(target);pngs.append(context)
        item=dict(pair_id=name,case_id=row['case_id'],frame_index=frame['frame_index'],source_id=frame['source_id'],selected_native=k,status=row['status'],supported_weak_pair=row['PASS'],old_supported_pair=was_old,new_supported_pair=bool(row['PASS'] and not was_old),stored_closed14_PASS=proof14,old5_parity=('PASS' if was_old else 'NOT_OLD_SUPPORTED'),saved_variants='|'.join(packets),original_all8_inputU=sum(not x for x in frame['native_continuous_valid']),policy=row.get('opposite_sham_policy',{}).get('selection','NO_GEOMETRY_PLAN'),reason=json.dumps(row.get('reason',row['status']),ensure_ascii=False),training_admission=False)
        stats.append(item);cases.setdefault(row['case_id'],dict(case_id=row['case_id'],frame_index=frame['frame_index'],source_id=frame['source_id'],selected_native=k,original_query_id=frame['query_id'][k],source_role=frame['split']))
        led.extend(census_rows(row,frame,packets));write(folder/'SAVED_DISPLAY_PROVENANCE.json',dict(source_attempt=done['manifest'],summary=item,clean=clean,all8_contact=card,selected_triplet=selected,producer_raw_context=context))
        md+=['## '+name,'',str(item),'','![선택된 끝의 저장 triplet]('+name+'/SELECTED_END_TRIPLET.png)','', '[모든 native8 저장96 보기]('+name+'/CONTACT_ALL8.png) · [입력 byte·상태 연결]('+name+'/SAVED_DISPLAY_PROVENANCE.json)','']
        if context:md+=['[원 생성기의 raw-coordinate context]('+name+'/PRODUCER_RAW_CONTEXT_CONTACT.png)','']
    need(len(led)==1584 and len(stats)==66 and len(cases)==22 and old5==5,'Full native/variant/case census and original5')
    csv_write(OUT/'ALL1584_NATIVE_VARIANT_CENSUS.csv',led);csv_write(OUT/'ALL66_SOURCE_STATUS.csv',stats);csv_write(OUT/'ALL22_SOURCE_CASES.csv',list(cases.values()))
    summary=dict(completed=True,PASS=True,source_only=True,attempts66=66,cases22=22,unique_source_frames21=len({s['frame_index'] for s in stats}),native_variant_status_rows1584=len(led),saved_native_input_rows=sum(r['pixel_packet_present'] for r in led),supported_weak_pairs=sum(r['supported_weak_pair'] for r in stats),old_supported_pairs=old5,new_supported_pairs=sum(r['new_supported_pair'] for r in stats),all_status_counts=dict(Counter(r['status'] for r in stats)),all_case_frame_source_records=list(cases.values()),all_native_slots=list(range(8)),missing_saved_variants=sum(not r['pixel_packet_present'] for r in led)//8,original_inputU_native_variant_rows=sum(not r['original_input_available'] for r in led),saved_inputU_rows=sum(r['pixel_packet_present'] and not r['original_input_available'] for r in led),training_admission=False,native_quality_owner_all_minus1=True,query_endpoint_and_central32_presence_are_distinct=True,classification_or_transfer_improvement=False,selected_model=None,adopted=False,fully_solved=False)
    md[2]+='\n\n전체 저장 상태: '+json.dumps(summary,ensure_ascii=False)+'\n\n[전체66 표](ALL66_SOURCE_STATUS.csv) · [원22사례 목록](ALL22_SOURCE_CASES.csv) · [native/variant1584 상태행](ALL1584_NATIVE_VARIANT_CENSUS.csv) · [정확 coverage JSON](SOURCE_COVERAGE_KO.json)'
    write(OUT/'SOURCE_COVERAGE_KO.json',summary)
    with (OUT/'PORTABLE_GALLERY_KO.md').open('x') as f:f.write('\n'.join(md)+'\n')
    sections=''.join('<section><h2>'+html.escape(z['pair_id'])+'</h2><p>'+html.escape(str(z))+'</p><img loading="lazy" src="'+z['pair_id']+'/SELECTED_END_TRIPLET.png"><p><a href="'+z['pair_id']+'/CONTACT_ALL8.png">모든 native8</a></p></section>' for z in stats)
    with (OUT/'index.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8"><title>V193 source 전체66시도</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}img{max-width:100%;height:auto}section{margin:25px 0;border-top:1px solid #aaa}</style><h1>원22사례·21프레임·전체66시도</h1><p>source weak 대조 자료이며 새 모델 성능·독립 holdout·학습 승인이 아니다. UNKNOWN/미구성 영상은 이유만 표시한다.</p><a href="PORTABLE_GALLERY_KO.md">전체 자료·표·provenance</a>'+sections)
    write(OUT/'IMAGE_MANIFEST.json',dict(completed=True,PASS=True,cards=cards,clean_valid_and_context_PNG=pngs))
    result=dict(completed=True,PASS=True,entry=bind(entry),code=e['code'],protocol=e['protocol'],source_generation_complete=e['source_generation_complete'],coverage=bind(OUT/'SOURCE_COVERAGE_KO.json'),gallery=bind(OUT/'PORTABLE_GALLERY_KO.md'),HTML=bind(OUT/'index.html'),images=bind(OUT/'IMAGE_MANIFEST.json'),all66_and_native8_statuses_retained=True,no_new_crop_sampling_or_sourcefullRGB_alpha_or_model_reads=True,actual_visual_QA='PENDING_SEPARATE_TOOL_VIEW',HTML_browser_rendered=False,training_admission=False,adopted=False,fully_solved=False)
    write(OUT/'COMPLETE.json',result);print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'),summary=summary)),flush=True)

def prepare():
    import tempfile
    checks=[]
    def ck(name,v):need(v,name);checks.append(dict(name=name,PASS=True))
    frame=dict(query_xy=[[100.,70.]]*8,L=[96.]*8,native_slot=list(range(8)),query_id=['TOY/native'+str(k) for k in range(8)],global_index=list(range(8)),native_continuous_valid=[True]*7+[False])
    rgb=np.arange(8*96*96*3,dtype=np.uint32).astype(np.uint8).reshape(8,96,96,3)
    a=dict(rgb96=rgb,valid96=np.ones((8,96,96),bool),query_xy=np.array(frame['query_xy'],np.float64),L=np.array(frame['L'],np.float64),native_slot=np.arange(8,dtype=np.int8),query_id=np.array(frame['query_id']),original_global_index=np.arange(8,dtype=np.int32),query_available=np.array(frame['native_continuous_valid']),native_visibility_target=np.full(8,-1,np.int8),point_quality_target=np.full(8,-1,np.int8),owner_target=np.full(8,-1,np.int8))
    ck('all8_packet_schema_and_inputU',validate_packet(a,frame))
    bad={k:z.copy() for k,z in a.items()};bad['query_xy'][0,0]+=.01
    try:validate_packet(bad,frame);ok=False
    except AssertionError:ok=True
    ck('q_drift_rejected',ok)
    bad={k:z.copy() for k,z in a.items()};bad['native_visibility_target'][0]=1
    try:validate_packet(bad,frame);ok=False
    except AssertionError:ok=True
    ck('native_label_promotion_rejected',ok)
    row=dict(pair_id='INVENTED',case_id='TOY',donor_id='YCB:TOY',status='UNKNOWN_RETAINED',original_input={'native_slot':0},direct_end_weak_states=dict(BEFORE='UNKNOWN',COVER='UNKNOWN',SHAM='UNKNOWN'),variants={k:dict(constructed=False,reason='UNKNOWN no saved RGB') for k in VARIANTS})
    ck('all24_variant_native_rows_no_missing_drop',len(census_rows(row,frame,{}))==24)
    ck('UNKNOWN_and_U_separate',all(z['declared_direct_weak_state']=='UNKNOWN' for z in census_rows(row,frame,{})) and sum(not z['original_input_available'] for z in census_rows(row,frame,{}))==3)
    row['direct_end_weak_states']=dict(BEFORE='VISIBLE_WEAK',COVER='EXTERNALLY_BLOCKED_WEAK',SHAM='VISIBLE_WEAK')
    ck('SHAM_alias_explicit_selectedslot_only',weak_state(row,'ADJACENT_SHAM',0)=='VISIBLE_WEAK' and weak_state(row,'COVER',1)=='UNKNOWN')
    with tempfile.TemporaryDirectory(prefix='invented_source_gallery_',dir=H) as tmp:
        folder=Path(tmp);snapshot=rgb.copy();save_png(folder/'clean.png',rgb[0]);ck('clean_PNG_byteexact',True)
        card=draw_sheet(folder/'sheet.png',row,{'BEFORE':a},frame,[0]);ck('display_markers_do_not_change_RGB',exact(snapshot,rgb))
        with (H/'INVENTED_LAYOUT.png').open('xb') as f:f.write((folder/'sheet.png').read_bytes())
        ck('q_marker_center_original96',((47.5+.5)*160/96-.5)==79.5)
        ck('missing_COVER_SHAM_text_only',not any(p.name.startswith('COVER') for p in folder.iterdir()))
    for path in ['/tmp/V112/forbidden','/tmp/a/../b']:
        try:safe(path);ok=False
        except AssertionError:ok=True
        ck('unsafe_name_before_access_'+path,ok)
    write(H/'GENERATED_INVENTED_CHECKS.json',dict(completed=True,PASS=True,checks=checks,count=len(checks),invented_only=True,actual_source_output_asset_header_or_pixel_reads=0))
    protocol=dict(completed=True,prepared_only=True,code=bind(Path(__file__).absolute()),producer_code=bind(PRODUCER),producer_protocol=bind(PRODUCER_PROTOCOL),fixed_population='Original66 attempts/all22case rows/21 unique SOURCE_DEVELOPMENT frames/all8 native/all3variant1584statuses, no outcome selection.',actual_inputs='Only completed source manifest/protocol/source-plan metadata, V193 saved packet whitelist11 and optional producer-generated raw context CONTACT bytecopy. Never decode full_RGB/alpha/source originals, no new sampler/model.',pixel_rule='Separate decoded-byteexact CLEAN96 and VALID96 PNGs. Card cyan q is display-copy-only original96 coordinate47.5; central32 exact slice unmarked. Raw-coordinate markers only preserved producer context contact, never passed as96indices.',UNKNOWN='No pixel packet -> text-only reason, never a generated evidence image. Missing, inputU, weak UNKNOWN, failedgeometry and proof status separately retained.',declared_labels='Display saved weak states only for selected slot; other7 UNKNOWN; native/quality/owner storedall−1; no training admission or semantic promotion.',old5='Require saved original5 decoded packet parity PASS; report old/new support separately. Bbox-only hypotheses are not counted supported.',outputs=dict(root=str(OUT),cards='All66 CONTACT_ALL8 and selected-end triplet; separate clean/validPNG; optional bytecopied raw producerCONTACT',tables=['ALL1584_NATIVE_VARIANT_CENSUS.csv','ALL66_SOURCE_STATUS.csv','ALL22_SOURCE_CASES.csv'],reports=['SOURCE_COVERAGE_KO.json','PORTABLE_GALLERY_KO.md','index.html']),actual_entry_required=True,checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),model_forward_fit_eval_newGT_or_sourcepixels_during_preparation=0,HTML_browser_rendered=False,adopted=False,fully_solved=False)
    write(PROTOCOL,protocol)
    write(H/'ROOT_ACTUAL_SAVED_SOURCE_VISUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_source_visual_readout=False,automatic_retry=False,code=protocol['code'],protocol=bind(PROTOCOL),root_scope={'path':None,'sha256':None},source_generation_complete={'path':str(V/'real_end_opposite_sham_pairs_v1/COMPLETE.json'),'sha256':None},output_directory=str(OUT)))
    write(H/'FINAL_READY.json',dict(completed=True,PASS=True,FINAL_READY_FROZEN=True,code=protocol['code'],protocol=bind(PROTOCOL),checks=protocol['checks'],false_entry=bind(H/'ROOT_ACTUAL_SAVED_SOURCE_VISUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),invented_checks=len(checks),actual_production_source_or_output_pixels_headers_arrays_model_reads=0,actual_entry_required=True,no_edits_after_final_ready_without_root_request=True))
    print(json.dumps(dict(completed=True,PASS=True,FINAL_READY=bind(H/'FINAL_READY.json'),code=protocol['code'],protocol=bind(PROTOCOL),invented_checks=len(checks))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args()
    prepare() if a.prepare else run(a.entry)
