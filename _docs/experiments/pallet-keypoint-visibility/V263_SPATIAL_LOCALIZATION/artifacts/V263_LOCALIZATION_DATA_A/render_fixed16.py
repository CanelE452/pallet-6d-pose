"""Fixed SOURCE-only qualitative gallery; no label, mask, crop or recipe changes."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.dont_write_bytecode = True

def need(v, m):
    if not bool(v): raise ValueError(m)

def safe(value):
    p=Path(value)
    need(p.is_absolute() and '..' not in p.parts, 'Absolute nontraversal')
    need(not any(s in str(p).casefold() for s in ('v112','cache','.pyc','current223','private_evaluation','source_holdout')), 'Forbidden before FS')
    for a in [*reversed(p.parents),p]:
        try: mode=os.lstat(a).st_mode
        except FileNotFoundError: continue
        need(not stat.S_ISLNK(mode), 'No symlink artifact/ancestor')
    return p

def binding(p):
    p=safe(p); b=p.read_bytes()
    return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'size_bytes':len(b)}

def read(b):
    raw=safe(b['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest()==b['sha256'] and ('size_bytes' not in b or len(raw)==b['size_bytes']), 'Exact SHA/size')
    return raw

def put(p,x):
    with safe(p).open('x',encoding='utf8')as f: json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return binding(p)

def records(b): return [json.loads(x)for x in read(b).splitlines()if x]

def rgbsha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def projection(x,g):
    K=np.asarray(g['K']);v=np.asarray(x)@np.asarray(g['GT_R_canonical']).T+g['GT_t']
    return(v[:,:2]/v[:,2,None])*[K[0,0],K[1,1]]+K[:2,2]

def font(n):
    # Display font only; never a data/model input.
    try: return ImageFont.truetype('DejaVuSans.ttf',n)
    except OSError: return ImageFont.load_default()

def marker(d,q,color,r=5):
    x,y=map(float,q);d.line((x-r,y,x+r,y),fill=color,width=2);d.line((x,y-r,x,y+r),fill=color,width=2)

def run(entry_path):
    e=json.loads(safe(entry_path).read_bytes());need(e.get('Root_SOURCE_RGB_gallery_authorized')is True,'Root authorization')
    out=safe(e['output_directory']);need(not out.exists(),'Fresh output');out.mkdir()
    try:
        complete=json.loads(read(e['label_complete']));groups=records(complete['outputs']['groups']);rows=records(complete['outputs']['rows'])
        need(complete['PASS'] and len(rows)==7040 and len(groups)==880,'Frozen original packet only')
        chosen=[]
        for j in range(8):
            rj=[r for r in rows if r['native_slot']==j]
            # Frozen deterministic display rule before any RGB decode; no image/score-based choice.
            if j in (2,3):
                d=[r for r in rj if not r['fit'] and r['point_combined_state']=='D' and len(r['visible_edge_axes'])==1]
                need(d,'Declared one-edge illustrative stratum exists')
                kind='DEV_D_SINGLE_EDGE'
            else:
                d=[r for r in rj if r['fit'] and r['point_combined_state']=='D' and r['direct_geometric_2edge']]
                kind='FIT_D_TWO_EDGE'
            chosen.append({'stratum':kind,'row':min(d,key=lambda r:r['source_index'])})
            ee=[r for r in rj if not r['fit'] and r['point_combined_state']=='E']
            if not ee: ee=[r for r in rj if r['fit'] and r['point_combined_state']=='E']
            need(ee,'Declared E illustrative stratum exists')
            chosen.append({'stratum':'DEV_E'if not ee[0]['fit']else'FIT_E','row':min(ee,key=lambda r:r['source_index'])})
        case_meta=[]
        for c in chosen:
            r=c['row'];g=groups[r['group_index']]
            case_meta.append({'stratum':c['stratum'],'row':r,'group_id':g['group_id'],'image':g['image'],'geometry_binding':g['geometry_binding']})
        selection=put(out/'CASE_IDS_BEFORE_RGB.json',{'policy':e['selection_policy'],'label_complete':e['label_complete'],'cases':case_meta,'RGB_decodes_before_this_file':0})
        source=groups[0]['RGB96_SOURCE_binding']
        need(all(g['RGB96_SOURCE_binding']==source for g in groups),'Same original RGB96 source')
        # Only RGB96/valid96 from original SYN-only packet; fullscene only RGB_uint8.
        with np.load(io.BytesIO(read(source)),allow_pickle=False)as z:
            rgb96=z['rgb96'];valid96=z['valid96']
        need(rgb96.shape==(7040,96,96,3) and rgb96.dtype==np.uint8 and valid96.dtype==bool,'Original crop schema')
        raw=read(e['V90_crop_code']);tree=ast.parse(raw)
        body=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='crop'];need(len(body)==1,'Pure crop AST')
        ns={'np':np};exec(compile(ast.Module(body=body,type_ignores=[]),e['V90_crop_code']['path'],'exec'),ns)
        crop=ns['crop'];sources={};geoms={};cards=[];outputs=[selection];cases=[]
        for n,c in enumerate(chosen):
            r=c['row'];g=groups[r['group_index']];i=r['source_index'];j=r['native_slot']
            ib=g['image'];key=ib['path']
            if key not in sources:
                with np.load(io.BytesIO(read(ib)),allow_pickle=False)as z: scene=z['RGB_uint8']
                need(scene.dtype==np.uint8 and scene.shape==(512,512,3),'Original SOURCE RGB only')
                sources[key]=scene
            scene=sources[key]
            gb=g['geometry_binding']
            if gb['path']not in geoms:
                with np.load(io.BytesIO(read(gb)),allow_pickle=False)as z: geoms[gb['path']]=z['native_witness_points']
            p=g['permCF_to_canonical'][j];uv=projection(geoms[gb['path']][p],g)
            need(np.max(np.abs(uv[0]-r['reference_xy']))<=1e-10,'Same declared anchor projection')
            replay,rv=crop(scene,r['q0'],r['L'],96)
            exact=np.array_equal(replay.transpose(1,2,0),rgb96[i])and np.array_equal(rv[0],valid96[i])
            need(exact,'Selected fullRGB -> original V90 RGB96 and valid96 byteexact')
            stem='N%d_%02d_S%06d'%(j,n,i);clean=out/(stem+'_CLEAN96.png')
            Image.fromarray(rgb96[i]).save(clean);outputs.append(binding(clean))
            # A separate display panel receives annotations; CLEAN96 remains exact raw pixels.
            full=Image.fromarray(scene.copy());fd=ImageDraw.Draw(full)
            for k,q in enumerate(g['q8']):
                marker(fd,q,'yellow'if k==j else'cyan',4);fd.text((q[0]+4,q[1]+4),str(k),font=font(13),fill='yellow'if k==j else'cyan')
            marker(fd,r['reference_xy'],'magenta',6)
            plain=Image.fromarray(rgb96[i]).resize((288,288),Image.Resampling.NEAREST)
            over=plain.copy();od=ImageDraw.Draw(over);qcrop=np.array([47.5,47.5]);rcrop=np.asarray(r['reference_RAW96_xy'])
            def display(x): return tuple((np.asarray(x)+.5)*3-.5)
            marker(od,display(qcrop),'cyan',8);marker(od,display(rcrop),'magenta',8)
            edge_rows=[]
            for a in range(3):
                endpoint=uv[3+3*a];ep=47.5+96*(endpoint-np.asarray(r['q0']))/r['L']
                visible=a in r['visible_edge_axes'];color=('yellow'if visible else'gray')
                od.line((*display(rcrop),*display(ep)),fill=color,width=2)
                edge_rows.append({'axis':a,'r_to_endpoint_RAW96':[rcrop.tolist(),ep.tolist()],'all3_active_D':visible})
            card=Image.new('RGB',(1120,660),'#15191d');d=ImageDraw.Draw(card)
            d.text((14,10),f"{stem} | {c['stratum']} | native address {j} | {'FIT'if r['fit']else'DEV'}",font=font(21),fill='white')
            d.text((14,42),f"point0 {r['point_combined_state']} | LOC {r['LOC_valid']} | input U {not r['original_available']} | axes {r['visible_edge_axes']}",font=font(17),fill='white')
            card.paste(full,(12,92));card.paste(plain,(536,92));card.paste(over,(824,92))
            d.text((12,72),'Original scene: q8 cyan; active q yellow; projected material anchor r magenta',font=font(14),fill='white')
            d.text((536,388),'Exact clean original96 (nearest display)',font=font(14),fill='white')
            d.text((824,388),'DISPLAY ONLY q cyan / r magenta',font=font(13),fill='white')
            d.text((536,418),'Yellow: projected edge with 3 saved active-D samples.',font=font(15),fill='yellow')
            d.text((536,444),'Gray: that geometric edge cue is not certified visible.',font=font(15),fill='white')
            d.text((536,480),'Geometry cue is not photometric observability proof.',font=font(15),fill='white')
            d.text((536,506),'Original q/L/native/split unchanged; no GT recrop.',font=font(15),fill='white')
            d.text((12,622),'Selected examples only. No population claim, new native gold, label correction, or recipe selection.',font=font(17),fill='white')
            path=out/(stem+'_CARD.png');card.save(path);outputs.append(binding(path));cards.append(card)
            cases.append(dict(case_index=n,stratum=c['stratum'],row=r,group_id=g['group_id'],image_binding=ib,
                clean_RGB96_HWC_sha256=rgbsha(rgb96[i]),clean_RGB96_CHW_sha256=rgbsha(replay),valid96_bool_sha256=rgbsha(valid96[i]),
                fullRGB_to_original_RGB96_and_valid96_byteexact=exact,projected_edges=edge_rows,clean=binding(clean),card=binding(path)))
        overview=Image.new('RGB',(1120,660),'#15191d')
        for i,card in enumerate(cards):overview.paste(card.resize((280,165),Image.Resampling.LANCZOS),((i%4)*280,(i//4)*165))
        overview.save(out/'OVERVIEW.png');outputs.append(binding(out/'OVERVIEW.png'))
        outputs.append(put(out/'CASE_DATA.json',{'cases':cases,'all16_fullRGB_V90_RGB96_valid96_byteexact':True,'source_roles':'Fixed original FIT86/DEV24; reused SOURCE DEV, not blind','labels_masks_recipe_changed':False,'no_model_or_real_image':True}))
        html='<!doctype html><meta charset="utf-8"><title>V263 fixed SOURCE geometric cue gallery</title><style>body{background:#15191d;color:white;font-family:sans-serif}img{max-width:100%}</style><h1>고정 SOURCE 16 사례</h1><p>기존 native별 D/E 사례이며 원q/L·라벨·고정 mask를 변경하지 않았습니다. 기하학적 edge 지지는 영상 대비나 실사 native 정답의 증명이 아닙니다. SOURCE DEV는 이미 노출된 내부 개발셋입니다.</p>'
        for c in cases: html+=f'<h2>native {c["row"]["native_slot"]}: {c["stratum"]}</h2><img src="{Path(c["card"]["path"]).name}">'
        with safe(out/'INDEX_KO.html').open('x',encoding='utf8')as f:f.write(html+'\n')
        outputs.append(binding(out/'INDEX_KO.html'))
        cb=put(out/'COMPLETE.json',dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),entry=binding(entry_path),label_complete=e['label_complete'],case_count=16,
            all16_fullRGB_V90_RGB96_valid96_byteexact=True,outputs=outputs,geometry_members_decoded=['native_witness_points'],fullscene_member_decoded='RGB_uint8',RGB96_members_decoded=['rgb96','valid96'],model_train_current_or_real=0,labels_masks_recipe_changed=False))
        print(json.dumps(cb),flush=True)
    except Exception as ex:
        put(out/'FAILURE.json',{'completed':False,'PASS':False,'error_type':type(ex).__name__,'error':str(ex),'partial_preserved':True,'automatic_retry':False});raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(safe(p.parse_args().entry))
