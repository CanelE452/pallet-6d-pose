"""Root-fixed4 TRAIN input gallery; original q/L/U, no visibility labels or model imports."""
import argparse,ast,csv,hashlib,html,io,json,os,stat
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

H=Path(__file__).absolute().parent
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility');R=P/'publication_root_preparation'
SAMPLER=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
SAMPLER_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'

def need(ok,msg):
    if not ok:raise ValueError(msg)

def safe(value,exists=True):
    s=str(value);need(not any(x in s.casefold() for x in ('v112','__pycache__','.pyc','current223','publicheld','source_holdout','oldval')),'lexical before FS')
    p=Path(s);need(p.is_absolute() and '..' not in p.parts,'literal absolute path')
    for q in reversed(p.parents):
        try:mode=os.lstat(q).st_mode
        except FileNotFoundError:continue
        need(stat.S_ISDIR(mode),'symlink/non-directory ancestor')
    if exists:need(stat.S_ISREG(os.lstat(p).st_mode),'symlink/nonregular leaf')
    elif p.exists():need(stat.S_ISDIR(os.lstat(p).st_mode),'output symlink/non-directory')
    return p

def raw(value):
    with os.fdopen(os.open(safe(value),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def checked(binding):
    b=raw(binding['path']);need(hashlib.sha256(b).hexdigest()==binding['sha256'],'exact SHA')
    if 'size_bytes' in binding:need(len(b)==binding['size_bytes'],'exact bytes')
    return b

def binding(p,b):return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}
def bitsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def run(entry_path):
    eb=raw(entry_path);e=json.loads(eb)
    need(e['approved_fixed4_TRAIN_RGB_gallery'] is True and e['automatic_retry'] is False,'Root actual entry required')
    need(e['code']==binding(Path(__file__).absolute(),raw(__file__)),'pinned renderer')
    plan=json.loads(checked(e['plan']));fixed=json.loads(checked(e['fixed_selection']))
    need(fixed['RGB_or_certificate_access_authorized_by_this_file'] is False and fixed['label_or_model_result_used_to_select'] is False,'before-truth fixed selection')
    need(fixed['frames']==[case['frame'] for case in plan['cases']] and len(fixed['frames'])==4 and len(fixed['pairs'])==2,'exact fixed4 cases')
    need(e['RGB_whitelist']==[{'path':f['image'],'sha256':f['image_sha256']} for f in fixed['frames']],'exact4 only RGB authority')
    complete=json.loads(checked(e['TRAIN_metadata_complete']));qa=json.loads(checked(e['TRAIN_metadata_QA']))
    need(complete['completed'] is True and complete['PASS'] is True and qa['completed'] is True and qa['PASS'] is True and qa['dataset_complete']==e['TRAIN_metadata_complete'],'independent TRAIN metadata gate')
    need(e['TRAIN_frames']==complete['outputs']['TRAIN44_FRAMES_INPUT_ONLY.jsonl'] and e['TRAIN_queries']==complete['outputs']['TRAIN352_ORIGINAL_NATIVE_QUERIES_INPUT_ONLY.jsonl'],'TRAIN output binding closure')
    need(all(e[k]==plan[k] for k in ('fixed_selection','TRAIN_metadata_complete','TRAIN_metadata_QA','TRAIN_frames','TRAIN_queries')),'frozen plan metadata closure')
    frames=[json.loads(x) for x in checked(e['TRAIN_frames']).splitlines()]
    queries=[json.loads(x) for x in checked(e['TRAIN_queries']).splitlines()]
    need((len(frames),len(queries))==(44,352) and all(x['split']=='TRAIN_LAB' for x in frames+queries),'TRAIN only metadata')
    for case in plan['cases']:
        f=case['frame'];need(f in frames,'fixed frame exact input metadata')
        qs=[q for q in queries if q['frame_input_id']==f['frame_input_id']]
        need(qs==case['queries'] and [q['native_slot'] for q in qs]==list(range(8)),'all original8 unchanged queries')
        need(all(q['native_semantic_state']=='UNKNOWN' and q['human_gold'] is False for q in qs),'no new visibility labels')
    mb=raw(SAMPLER);need(hashlib.sha256(mb).hexdigest()==SAMPLER_SHA and e['sampler']==binding(SAMPLER,mb),'exact pure V90 CODE')
    node=next(n for n in ast.parse(mb).body if isinstance(n,ast.FunctionDef) and n.name=='crop')
    scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SAMPLER),'exec'),scope);crop=scope['crop']
    out=safe(e['output_directory'],False);need(out==H/'gallery_v1' and not out.exists(),'fresh owned gallery output')
    out.mkdir();outputs={}
    def put(name,b):
        p=out/name
        with p.open('xb') as f:f.write(b)
        outputs[name]=binding(p,b)
    def js(name,v):put(name,(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    def png(name,img):
        b=io.BytesIO();img.save(b,format='PNG');put(name,b.getvalue())
    font=ImageFont.load_default();cards=[];rows=[];decoded=[]
    palette=['#00d5ff','#ffd400','#ff66cc','#72ff72','#ff9b35','#c4a3ff','#73ffe2','#ff7373']
    for fi,case in enumerate(plan['cases']):
        f=case['frame'];source=checked(e['RGB_whitelist'][fi])
        im=Image.open(io.BytesIO(source));need(im.format=='PNG','declared source PNG');im=im.convert('RGB')
        need([im.height,im.width]==f['raw_hw'],'raw H/W exact');rgb=np.asarray(im,dtype=np.uint8)
        decoded.append({'frame_input_id':f['frame_input_id'],'source':e['RGB_whitelist'][fi],'decoded_RGB_sha256':bitsha(rgb)})
        put(f'FRAME_{fi:02d}_SOURCE_RGB.png',source)
        shown=im.copy();draw=ImageDraw.Draw(shown)
        for q in case['queries']:
            if q['query_xy'] is None:continue
            x,y=q['query_xy'];k=q['native_slot'];color=palette[k]
            if 0<=x<im.width and 0<=y<im.height:
                draw.line((x-6,y,x+6,y),fill=color,width=2);draw.line((x,y-6,x,y+6),fill=color,width=2)
                draw.text((x+7,y+3),str(k),fill=color,font=font)
        width=max(840,im.width+24);height=im.height+600
        card=Image.new('RGB',(width,height),'#eeeeee');d=ImageDraw.Draw(card)
        d.text((12,10),f'FIXED TRAIN {fi} | {f["pair_id"]} | {f["view"]}',fill='black',font=font)
        d.text((12,28),'INPUTS ONLY: native semantics UNKNOWN; no D/E/SELF gold; display crosses do not alter input crops.',fill='black',font=font)
        d.text((12,46),'Original RGB at native resolution; unchanged q8. Clean96 tiles enlarged 2x NEAREST for display only.',fill='black',font=font)
        card.paste(shown,((width-im.width)//2,70))
        for q in case['queries']:
            k=q['native_slot'];xy=q['query_xy'];L=q['roi_side'];present=xy is not None
            if present:tile,valid=crop(rgb,np.asarray(xy,np.float64),float(L),96)
            else:tile=np.zeros((3,96,96),np.uint8);valid=np.zeros((1,96,96),bool)
            need(tile.shape==(3,96,96) and tile.dtype==np.uint8 and valid.shape==(1,96,96),'original96 sampler shapes')
            clean=Image.fromarray(tile.transpose(1,2,0));name=f'FRAME_{fi:02d}_N{k}_CLEAN96.png';png(name,clean)
            x=12+(k%4)*206;y=im.height+94+(k//4)*244
            card.paste(clean.resize((192,192),Image.Resampling.NEAREST),(x,y))
            caption='INPUT_AVAILABLE' if q['original_query_continuous_valid'] else 'INPUT_U'
            d.text((x,y+195),f'native{k} | {caption}',fill='black',font=font)
            d.text((x,y+211),'Original q/L: exact values in table',fill='black',font=font)
            if not q['original_query_continuous_valid']:
                reason='NO_DETECTION' if not present else 'OOB_OR_INVALID'
                d.text((x,y+225),reason,fill='black',font=font)
            rows.append({'frame_position':fi,**q,'native_semantic_state':'UNKNOWN','human_gold':False,'source_RGB_sha256':f['image_sha256'],'clean96_path':name,'RGB96_CHW_sha256':bitsha(tile),'valid96_bool_sha256':bitsha(valid),'valid_fraction':float(valid.mean()),'input_present':present,'absent_crop_is_blank':not present})
        png(f'FRAME_{fi:02d}_CARD.png',card);cards.append(card)
    need(len(rows)==32,'all32 rows retained')
    overview=Image.new('RGB',(cards[0].width*2,cards[0].height*2),'white')
    for i,card in enumerate(cards):overview.paste(card,((i%2)*card.width,(i//2)*card.height))
    png('FIXED4_OVERVIEW.png',overview);js('ALL32_CASE_INPUTS.json',{'completed':True,'rows':rows,'training_admission':False,'native_visibility_labels_created':False})
    cols=['frame_position','query_id','frame_index','frame_input_id','pair_id','view','split','native_slot','query_xy','roi_side','original_query_continuous_valid','query_nearest_pixel_valid','query_rgb96_centre_domain_valid','reason','native_semantic_state','human_gold','input_present','valid_fraction','RGB96_CHW_sha256','valid96_bool_sha256','clean96_path']
    text=io.StringIO(newline='');w=csv.DictWriter(text,fieldnames=cols,extrasaction='ignore',lineterminator='\n');w.writeheader()
    for row in rows:w.writerow(row)
    put('ALL32_INPUTS.csv',text.getvalue().encode())
    body='<h1>Fixed public TRAIN input crops</h1><p>Native semantics UNKNOWN; original q/L/U retained. This is input inspection, no visibility gold, certificate, model output or selection.</p><p><a href="ALL32_INPUTS.csv">all32 input table</a> | <a href="ALL32_CASE_INPUTS.json">metadata and pixel hashes</a> | <a href="FIXED4_OVERVIEW.png">overview</a></p>'
    for i in range(4):body+=f'<h2>Original fixed frame {i}</h2><a href="FRAME_{i:02d}_SOURCE_RGB.png">original source bytes</a><br><img style="max-width:100%" src="FRAME_{i:02d}_CARD.png">'
    put('PORTABLE_GALLERY_KO.html',('<!doctype html><meta charset="utf-8"><title>Fixed TRAIN inputs</title>'+body).encode())
    js('IMAGE_MANIFEST.json',{'completed':True,'source_RGB':decoded,'outputs':dict(outputs),'markers_on_training_inputs':False})
    js('COMPLETE.json',{'completed':True,'PASS':True,'entry':binding(Path(entry_path).absolute(),eb),'code':e['code'],'plan':e['plan'],'fixed_selection':e['fixed_selection'],'TRAIN_metadata_complete':e['TRAIN_metadata_complete'],'TRAIN_metadata_QA':e['TRAIN_metadata_QA'],'sampler':e['sampler'],'RGB_whitelist':e['RGB_whitelist'],'frames':4,'pairs':2,'native_queries':32,'q_L_U_native_unchanged':True,'native_semantic_state':'UNKNOWN','human_gold':False,'training_admission':False,'GT_certificate_model_score_or_other_RGB_reads':0,'image_rendered_not_actually_tool_viewed':True,'outputs':dict(outputs)})
    print(json.dumps({'completed':True,'PASS':True,'complete':outputs['COMPLETE.json']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
