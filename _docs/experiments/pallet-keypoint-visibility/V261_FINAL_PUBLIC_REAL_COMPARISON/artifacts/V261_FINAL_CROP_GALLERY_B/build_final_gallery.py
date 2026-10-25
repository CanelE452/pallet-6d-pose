"""Fixed final cases, selection frozen before RGB decode. No model/fit code."""
import argparse, ast, csv, hashlib, html, io, json, math, os, stat, sys, traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode=True
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
POLICY={'path':str(R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON/PROSPECTIVE_FINAL_ONE_COMPARISON_POLICY.json'),'sha256':'5bfeaf8d690b39c6e9c287d1b25340a1ee7b538d90d2d9447be9fa61e94da54c','size_bytes':4732}
CROP={'path':str(R.parent/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'),'sha256':'6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b','size_bytes':4715}
CONTROL='CONTROL_ORIGINAL_SOURCE_FIT';TREAT='TREAT_ORIGINAL_PLUS_PUBLIC_SURFACE_AUX'
D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';GROUPS=('WOOD_PRIMARY105','SQUARE118')
SALT='V261_FINAL_CASE_GALLERY_V1|'

def need(ok,message):
    if not bool(ok):raise ValueError(message)

def safe(value,exists=True):
    s=str(value);need(not any(x in s.casefold()for x in('v112','cache','.pyc','source_holdout')),'Forbidden path before FS')
    p=Path(s);need(p.is_absolute()and'..'not in p.parts,'Closed absolute path')
    for a in [*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:need(not exists,'Missing exact artifact');continue
        need(not stat.S_ISLNK(mode),'Regular artifacts/ancestors only; no alias follow')
    return p

def bkey(b):
    need(set(b)=={'path','sha256','size_bytes'},'Exact binding fields')
    s=str(b['path']);need(not any(x in s.casefold()for x in('v112','cache','.pyc','source_holdout')),'Forbidden binding before FS')
    need(Path(s).is_absolute()and'..'not in Path(s).parts and len(b['sha256'])==64 and type(b['size_bytes'])is int,'Closed exact binding')
    return b['path'],b['sha256'],b['size_bytes']

def checked(b,white):
    need(bkey(b)in{bkey(x)for x in white},'Root exact whitelist before read');p=safe(b['path']);raw=p.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==b['sha256']and len(raw)==b['size_bytes'],'Exact file identity');return raw

def binding(p):
    raw=safe(p).read_bytes();return{'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def write_json(p,x):
    p=safe(p,False);need(not p.exists(),'Fresh output')
    with p.open('x',encoding='utf-8')as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return binding(p)

def finite(value):
    try:v=float(value)
    except(TypeError,ValueError):return None
    return v if math.isfinite(v)else None

def correct(row):
    return row[TREAT+'_D_E_U']==('D'if row['human_state']==D else'E')

def rank(row):return hashlib.sha256((SALT+row['row_id']).encode('utf-8')).hexdigest(),row['row_id']

def select_cases(ledger):
    """No images/probability ranking; keep every predefined cell including empty."""
    need(len(ledger)==1784,'Frozen all1784 ledger')
    for i,row in enumerate(ledger):
        need(int(row['global_index'])==i and int(row['frame_index'])==i//8 and int(row['native_corner'])==i%8,'Original row/native order')
        need(row[TREAT+'_D_E_U']in('D','E','U')and row[CONTROL+'_D_E_U']in('D','E','U')and row['v61_D_E_U']in('D','E','U'),'Stored decisions only')
    primary=[r for r in ledger if r['paired_same_query_RGB_D_vs_E']=='True'and r['human_state']in(D,E)]
    need((len(primary),sum(r['human_state']==D for r in primary),sum(r['human_state']==E for r in primary))==(1237,1147,90),'Unchanged primary')
    near=[r for r in ledger if r['human_state']==D and r.get('annotation_coordinate_source')=='manual_click'and r['original_query_valid']=='True'and finite(r.get('initial_reference_error_px'))is not None and float(r['initial_reference_error_px'])<=3]
    cells=[]
    def add(cell_id,pool,rule):
        winner=min(pool,key=rank)if pool else None
        cells.append({'cell_id':cell_id,'candidate_count':len(pool),'rule':rule,'selected_row':None if winner is None else dict(winner),'selected_row_id':None if winner is None else winner['row_id'],'selected_hash':None if winner is None else rank(winner)[0],'empty':winner is None})
    for group in GROUPS:
        for state,code in((D,'D'),(E,'E')):
            for outcome in(True,False):
                add(f'MAIN_{group}_GT{code}_TREAT_'+('CORRECT'if outcome else'WRONG'),[r for r in primary if r['group']==group and r['human_state']==state and correct(r)==outcome],'primary1237 x material x humanD/E x TREAT correctness; U=wrong')
    for group in GROUPS:
        for outcome in(True,False):
            add(f'MANUAL_LE3_{group}_GTD_TREAT_'+('CORRECT'if outcome else'WRONG'),[r for r in near if r['group']==group and correct(r)==outcome],'all1784 humanD/manual_click/error<=3/originalqueryvalid x material x TREAT correctness; no E forced')
    return {'completed':True,'selection_frozen_before_any_crop_RGB_decode':True,'salt':SALT,'selection':'smallest SHA256(salt+row_id), then row_id; no score sorting','primary_N':1237,'manual_click_LE3_ledger_derived_count':len(near),'manual_subset_authority':'Existing ledger annotation_coordinate_source/error/queryvalid fields, not new coordinategold; separate historical manualCSV count not silently substituted','slots':cells,'unique_selected_row_ids':sorted({c['selected_row_id']for c in cells if not c['empty']}),'empty_cells_preserved':sum(c['empty']for c in cells),'same_case_may_fill_multiple_cells':True,'new_model_or_threshold_selection':False}

def render_case(rgb,row,crop,expected_crop=None):
    """Pure RGB→sameq/L crop; existing reference is a ring, never crop centre."""
    need(rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3,'Original RGB uint8 HWC')
    q=[finite(row.get('query_x')),finite(row.get('query_y'))];L=finite(row.get('roi_side'))
    if None in q or L is None or L<=0:return None,None,{'unavailable':True,'reason':'Nonfinite/nonpositive original q/L','no_recrop_or_snap':True}
    raw,valid=crop(rgb,np.asarray(q,np.float64),L,96)
    sha=hashlib.sha256(raw.tobytes()).hexdigest();vsha=hashlib.sha256(valid.tobytes()).hexdigest()
    if expected_crop is not None:
        need(sha==expected_crop['RGB96_CHW_sha256']and vsha==expected_crop['valid96_bool_sha256'],'Exact classifier-generated RAW96/valid byte parity')
    original=Image.fromarray(raw.transpose(1,2,0),'RGB');marked=original.resize((384,384),Image.Resampling.NEAREST);draw=ImageDraw.Draw(marked)
    def xy(x):return(x+.5)*4-.5
    center=xy(47.5);draw.line((center-10,center,center+10,center),fill=(0,255,255),width=2);draw.line((center,center-10,center,center+10),fill=(0,255,255),width=2)
    reference=[finite(row.get('human_reference_x')),finite(row.get('human_reference_y'))];projected=None;inside=False
    if None not in reference:
        projected=((np.asarray(reference)-np.asarray(q))*96/L+47.5).tolist();inside=all(-.5<=v<95.5 for v in projected)
        if inside:
            x,y=map(xy,projected);draw.ellipse((x-8,y-8,x+8,y+8),outline=(255,0,255),width=2)
    fact={'unavailable':False,'q_xy_unchanged':q,'L_unchanged':L,'q_RAW96_xy':[47.5,47.5],'reference_xy_annotation_only':reference if None not in reference else None,'reference_RAW96_xy_annotation_only':projected,'reference_marker_inside_crop':inside,'reference_outside_crop':projected is not None and not inside,'RAW96_RGB_sha256_CHW':sha,'RAW96_valid_sha256_bool':vsha,'full_valid_fraction':float(valid.mean()),'centre32_valid_fraction':float(valid[:,32:64,32:64].mean()),'classifier_crop_byte_comparison':expected_crop is not None,'original_input_available':row['original_query_valid']=='True','TREAT_input_unavailability_reason':row.get(TREAT+'_U_reason',''),'no_recrop_or_snap':True,'raw_pixels_unmodified':True,'marked_view_only_changes':'Nearest display enlargement, cyan originalq cross, optional magenta existingreference ring'}
    return original,marked,fact

def run(entry_path):
    entry_raw=safe(entry_path).read_bytes();e=json.loads(entry_raw)
    need(e.get('approved_actual_final_gallery')is True,'Separate Root true gallery entry required before case data')
    scope=json.loads(checked(e['root_scope'],[e['root_scope']]))
    need(scope.get('approved_actual_final_gallery')is True and scope.get('entry_contract')=={k:v for k,v in e.items()if k!='root_scope'},'Root exact gallery scope')
    own=binding(Path(__file__).absolute());need(e['code']==own and e['policy']==POLICY and e['crop_code']==CROP,'Pinned fixed gallery/crop/policy')
    cw=e['code_whitelist'];mw=e['metadata_whitelist'];pw=e['image_whitelist']
    need({bkey(b)for b in cw}=={bkey(own),bkey(CROP)}and len(cw)==2,'Only gallery and pureV90crop CODE')
    checked(own,cw);checked(POLICY,mw)
    complete=json.loads(checked(e['comparison_complete'],mw));freeze=json.loads(checked(e['prediction_freeze'],mw))
    need(complete.get('completed')is True and complete.get('PASS')is True and freeze.get('completed')is True and freeze.get('PASS')is True,'Completed current comparison/predfreeze required')
    need(freeze.get('rows_each')==1784 and freeze.get('GT_cohort_or_v61_baseline_body_reads_before_this_freeze')==0,'Both newpredictions frozen before lateGT/v61 readout')
    need(complete['prediction_freeze']==e['prediction_freeze']and complete['input_facts']==e['input_facts']and complete['saved_evaluation_outputs']['ALL1784_FINAL_COMPARISON_LEDGER']==e['ledger'],'Exact fixed comparison outputs')
    ledger=list(csv.DictReader(io.StringIO(checked(e['ledger'],mw).decode('utf-8-sig'))))
    facts=json.loads(checked(e['input_facts'],mw))['frames'];need(len(facts)==223,'All original frames')
    need({bkey(b)for b in pw}<={bkey(f['canonical_encoded_binding'])for f in facts},'Only current-inference canonical image bindings; no fallback sources')
    selection=select_cases(ledger)
    out=safe(e['output_directory'],False);need(not out.exists(),'Fresh gallery output');out.mkdir()
    selection_binding=write_json(out/'CASE_SELECTION_BEFORE_RGB.json',selection)
    # No image stat/read/hash/decode before the selection JSON is persisted.
    ns={'np':np};raw=checked(CROP,cw);node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name=='crop');exec(compile(ast.Module(body=[node],type_ignores=[]),CROP['path'],'exec'),ns)
    rendered={};case_facts=[];outputs=[selection_binding];stage='selected_crop_RGB'
    try:
        by_id={r['row_id']:r for r in ledger}
        for row_id in selection['unique_selected_row_ids']:
            row=by_id[row_id];fi=int(row['frame_index']);native=int(row['native_corner']);fact=facts[fi]
            need(fact['frame_index']==fi and fact['id']==row['id'],'Exact selected frame association')
            image_binding=fact['canonical_encoded_binding']
            encoded=checked(image_binding,pw)
            with Image.open(io.BytesIO(encoded))as im:rgb=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
            need(hashlib.sha256(rgb.tobytes()).hexdigest()==fact['decoded_RGB_sha256'],'Same decoded original RGB as inference')
            expected=next((r for r in fact['local_crop_facts']if r['native_slot']==native),None)
            raw_image,marked,cf=render_case(rgb,row,ns['crop'],expected)
            cf.update({'global_index':int(row['global_index']),'row_id':row_id,'frame_index':fi,'native_slot':native,'encoded_original_binding':image_binding,'decoded_RGB_sha256':fact['decoded_RGB_sha256'],'GT_state':row['human_state'],'v61':row['v61_D_E_U'],'CONTROL':row[CONTROL+'_D_E_U'],'TREAT':row[TREAT+'_D_E_U'],'model_input_supported':row[TREAT+'_D_E_U']!='U','CONTROL_MAIN0_logit':finite(row.get(CONTROL+'_MAIN0_logit')),'TREAT_MAIN0_logit':finite(row.get(TREAT+'_MAIN0_logit')),'initial_reference_error_px':finite(row.get('initial_reference_error_px')),'annotation_coordinate_source':row.get('annotation_coordinate_source'),'mapping_status':row.get('mapping_status'),'labels_not_new_pixel_gold':True})
            if raw_image is not None:
                stem='case_%04d'%int(row['global_index']);raw_name=stem+'_RAW96.png';marked_name=stem+'_MARKED.png';raw_image.save(out/raw_name);marked.save(out/marked_name)
                outputs.extend([binding(out/raw_name),binding(out/marked_name)]);cf.update({'raw_png':raw_name,'marked_png':marked_name});rendered[row_id]=marked
            case_facts.append(cf)
        cf_binding=write_json(out/'CASE_FACTS.json',{'completed':True,'selection':selection_binding,'cases':case_facts,'legend':'Cyan unchanged original q47.5. Magenta optional existing annotation reference, no recrop/snap.','task_distinction':'Human registered-point D/E and reference association are inherited; no new q-pixel/nativegold. v61 stored presence0.5 target differs from SOURCE-trained MAIN D/E. Sameq does not certify identical model preprocessing.','no_model_or_baseline_forward':True,'model_or_threshold_changes':0});outputs.append(cf_binding)
        board=Image.new('RGB',(1260,12//3*490),(245,245,245));draw=ImageDraw.Draw(board);cards=[]
        cf_by_id={c['row_id']:c for c in case_facts}
        for index,cell in enumerate(selection['slots']):
            x=(index%3)*420;y=(index//3)*490;draw.rectangle((x,y,x+418,y+488),outline=(170,170,170),width=1)
            label=cell['cell_id'].replace('WOOD_PRIMARY105','WOOD').replace('SQUARE118','SQUARE');draw.text((x+10,y+8),label,fill=(0,0,0));draw.text((x+10,y+24),'Candidates: '+str(cell['candidate_count']),fill=(0,0,0))
            if cell['empty']:
                draw.text((x+30,y+170),'EMPTY / unavailable cell retained',fill=(150,30,30));cards.append('<article><h3>'+html.escape(label)+'</h3><p>EMPTY: no candidate. Cell retained.</p></article>');continue
            cf=cf_by_id[cell['selected_row_id']];image=rendered.get(cell['selected_row_id'])
            if image is not None:board.paste(image,(x+18,y+48))
            else:draw.text((x+30,y+170),'No finite original q/L: unavailable',fill=(150,30,30))
            lines=[f"GT:{cf['GT_state']} v61:{cf['v61']} C:{cf['CONTROL']} T:{cf['TREAT']}",f"C z0:{cf['CONTROL_MAIN0_logit']} T z0:{cf['TREAT_MAIN0_logit']}",f"qerr:{cf['initial_reference_error_px']} native:{cf['native_slot']}",f"original available:{cf.get('original_input_available',False)} model supported:{cf['model_input_supported']}"]
            for j,line in enumerate(lines):draw.text((x+10,y+438+j*12),line,fill=(0,0,0))
            image_html=''if'raw_png'not in cf else'<a href="'+cf['raw_png']+'"><img src="'+cf['marked_png']+'" alt="Original q crop with diagnostic markers"></a>'
            cards.append('<article><h3>'+html.escape(label)+'</h3>'+image_html+'<pre>'+html.escape(json.dumps(cf,ensure_ascii=False,indent=2))+'</pre></article>')
        board.save(out/'CONTACT_SHEET.png');outputs.append(binding(out/'CONTACT_SHEET.png'))
        text='<!doctype html><meta charset="utf-8"><title>V261 final fixed current crop cases</title><style>body{font:15px system-ui;margin:24px;background:#f5f5f5}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}article{background:white;padding:14px}img{width:100%;max-width:384px;image-rendering:pixelated}pre{white-space:pre-wrap;font-size:12px}</style><h1>Final fixed current crop cases</h1><p>Same original q/L; cyan q47.5, magenta inherited reference only. RAW96 PNG is unmarked and byte-checked against inference when a classifier crop existed. Unavailable inputs remain unavailable. No recrop, model rerun, training or threshold selection.</p><p>Inherited human D/E is registered-point authority, not new q-pixel/native gold. v61 is stored visible-target presence at its frozen0.5 threshold; C/T are SOURCE-trained MAIN0 decisions. Same query identity does not establish identical preprocessing. Repeated benchmark and legacy CAL exposure remain.</p><p>Selection was frozen before RGB decode: fixed hash per predefined cell. First8: primary material×D/E×TREAT correctness. Extra4: humanD/manual-click≤3px×material×TREAT correctness. Empty cells retained; ≤3px is inherited reference diagnosis, not new gold.</p><a href="CONTACT_SHEET.png">Contact sheet</a> · <a href="CASE_SELECTION_BEFORE_RGB.json">Frozen selection</a> · <a href="CASE_FACTS.json">Case facts</a><main>'+''.join(cards)+'</main>'
        (out/'GALLERY.html').write_text(text,encoding='utf-8');outputs.append(binding(out/'GALLERY.html'))
        cb=write_json(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':own,'policy':POLICY,'entry':{'path':str(entry_path),'sha256':hashlib.sha256(entry_raw).hexdigest(),'size_bytes':len(entry_raw)},'comparison_complete':e['comparison_complete'],'prediction_freeze':e['prediction_freeze'],'ledger':e['ledger'],'input_facts':e['input_facts'],'selection':selection_binding,'case_facts':cf_binding,'slots':12,'unique_selected_cases':len(case_facts),'empty_slots':selection['empty_cells_preserved'],'outputs':outputs,'original_fullframe_images_copied':0,'model_forwards':0,'optimizer_updates':0,'threshold_changes':0,'PnP_changes':0})
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':cb}),flush=True);return cb
    except BaseException as exc:
        write_json(out/'FAILED_NO_AUTOMATIC_RETRY.json',{'completed':False,'stage':stage,'error':str(exc),'traceback':traceback.format_exc(),'automatic_retry':False});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
