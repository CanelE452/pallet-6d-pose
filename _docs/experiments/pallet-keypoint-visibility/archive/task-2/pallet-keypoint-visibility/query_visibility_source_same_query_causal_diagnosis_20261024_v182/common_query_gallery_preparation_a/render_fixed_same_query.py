"""Outcome-blind saved common-query gallery; separate root entry before rendering."""
from pathlib import Path
import argparse,hashlib,json,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent;O=V/'fixed_common_query_gallery_v1'
SLOTS=V/'common_query_preparation_a/COMMON4096_SLOTS_INPUT_ONLY.jsonl'
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_COMMON_QUERY_OR_PUBLIC_PIXELS.json'
PLAN=H/'FIXED256_ROWS_BEFORE_GALLERY_PIXELS.json';PROTO=H/'PROTOCOL_BEFORE_GALLERY_PIXELS.json'
VARIANTS=['BEFORE','COVER','NEAR_SHAM','FAR_SHAM'];FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def need(x,msg):
    if not x:raise AssertionError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def binding(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def check(b):need(sha(b['path'])==b['sha256'],'SHA');return Path(b['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def selection():
    rr=rows(SLOTS);first={}
    for r in rr:
        f=r['source_image_original_metadata']['factors'];key=tuple(f[k] for k in ['geometry_family','material','aspect'])
        first.setdefault(key,r['base_index'])
    cases=[]
    for key,base in sorted(first.items()):
        for k in range(8):
            ids=[base*32+v*8+k for v in range(4)]
            cases.append(dict(case_id='BASE%03d_NATIVE%d'%(base,k),base_index=base,native_slot=k,factors=dict(zip(['geometry_family','material','aspect'],key)),diagnostic_rows=ids,row_ids=[rr[i]['row_id'] for i in ids]))
    need(len(first)==8 and len(cases)==64 and len({i for c in cases for i in c['diagnostic_rows']})==256,'8cells/64cards/256slots')
    return dict(rule='Lowest original base_index within each full geometry_family x material x aspect cell, before reading rendered pixels or model outcomes. All4variants/native8 retained; no replacement.',bases=sorted(first.values()),groups8=8,cards64=64,rows256=256,cases=cases,slots=binding(SLOTS),outcome_blind=True,selected_model=None)
def prepare():
    p=selection();write(PLAN,p)
    font=ImageFont.truetype(FONT,16);draw=ImageDraw.Draw(Image.new('RGB',(10,10)))
    strings=['raw96 x3; cyan=display only','valid96: white=valid','central32 x6 (32:64)','Original image q retained below','End support is separate GT']
    boxes={s:list(draw.textbbox((0,0),s,font=font)) for s in strings};need(all(b[2]-b[0]<=420 for b in boxes.values()),'column caption fit')
    protocol=dict(completed=True,prepared_only=True,code=binding(__file__),scope=binding(SCOPE),plan=binding(PLAN),font=binding(FONT),caption_boxes=boxes,
        layout='64nativecards, 4fixed variant columns, clean raw96+valid96+central32; exact raw PNGs separate. Cyan center2x2 footprint only on enlarged display copy. No RGB/fullframe reread; saved common NPZ only.',
        semantics='Common COVER q/L/inputflags immutable across4images, original image q preserved separately. End2 support and central4 literalowner5 shown as separate diagnostic GT, neither native E/physicalcorner gold nor modelinput. U retained.',
        actual_gate='--entry approved_common_query_gallery=true/source_TRAIN_only=true with exact code/protocol/plan/common_complete bindings; no automatic retry.',
        no_SAM_or_head_outputs=True,new_fit_or_forward=0,actual_rendered_images=0)
    write(PROTO,protocol);write(H/'ROOT_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_common_query_gallery=False,source_TRAIN_only=True,code=protocol['code'],protocol=binding(PROTO),plan=protocol['plan'],common_complete=None,automatic_retry=False))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=binding(PROTO),plan=protocol['plan'],actual_rendered_images=0,new_model_or_forward_or_fit=0))
def run(entry):
    e=read(entry);need(e.get('approved_common_query_gallery') is True and e.get('source_TRAIN_only') is True and e.get('automatic_retry') is False,'separate actual gallery entry')
    for k,p in [('code',Path(__file__)),('protocol',PROTO),('plan',PLAN)]:need(e[k]==binding(p),'entry '+k)
    c=read(check(e['common_complete']));need(c['completed'] and c['PASS'] and not c['production_query_replay'] and c['fit_admission'] is False,'diagnostic savedinput')
    p=read(PLAN);rr=rows(check(c['metadata']));mm=rows(check(c['measures']))
    need(p==selection(),'frozen outcome-blind selection')
    with np.load(check(c['inputs']),allow_pickle=False) as z:x={k:z[k].copy() for k in z.files}
    with np.load(check(c['targets']),allow_pickle=False) as z:y={k:z[k].copy() for k in z.files}
    need(not O.exists(),'preserve previous attempt');O.mkdir()
    font=ImageFont.truetype(FONT,16);small=ImageFont.truetype(FONT,14);title=ImageFont.truetype(FONT,23);manifest=[]
    for case in p['cases']:
        cd=O/case['case_id'];cd.mkdir();im=Image.new('RGB',(1820,805),(20,25,34));d=ImageDraw.Draw(im)
        d.text((15,10),case['case_id']+' | '+str(case['factors']),font=title,fill='white')
        first=rr[case['diagnostic_rows'][0]]
        d.text((15,47),'Common original COVER q='+str(first['query_xy'])+' L='+str(first['L'])+' | native'+str(case['native_slot']),font=font,fill='white')
        d.text((15,72),'SOURCE TRAIN diagnostic. Same q/L across images; no original-query replay. GT support != literal owner != native E.',font=font,fill='#edc98d')
        variant_rows=[]
        for vi,j in enumerate(case['diagnostic_rows']):
            r=rr[j];need(r['row_id']==case['row_ids'][vi] and r['variant']==VARIANTS[vi],'selected identity');left=15+450*vi
            raw=x['rgb96'][j];valid=x['valid96'][j];central=raw[32:64,32:64]
            paths={}
            for name,a in [('RAW96',raw),('VALID96',(valid*255).astype(np.uint8)),('CENTRAL32',central)]:
                q=cd/(r['variant']+'_'+name+'.png');Image.fromarray(a).save(q)
                need(np.array_equal(np.asarray(Image.open(q)),a),'PNG raw pixels exact');paths[name]=binding(q)
            need(ah(raw)==r['actual_rgb96_sha256'] and ah(valid)==r['actual_valid96_sha256'],'saved row raw facts')
            d.text((left,107),r['variant']+' | diag '+str(j)+' | input '+('VALID' if r['original_query_available'] else 'U'),font=font,fill='white')
            rgbcopy=Image.fromarray(raw).resize((288,288),Image.Resampling.NEAREST);cdraw=ImageDraw.Draw(rgbcopy);cdraw.rectangle((141,141,146,146),outline='#00ffff',width=1)
            im.paste(rgbcopy,(left,140));im.paste(Image.fromarray((valid*255).astype(np.uint8)).convert('RGB'),(left+312,140))
            d.text((left,437),'raw96 x3; cyan=display only',font=font,fill='#b9d4ed')
            d.text((left+302,246),'valid96',font=small,fill='white')
            im.paste(Image.fromarray(central).resize((192,192),Image.Resampling.NEAREST),(left,475))
            d.text((left,674),'central32 x6 (32:64)',font=font,fill='#b9d4ed')
            d.text((left+209,475),'Support visible / foreign',font=small,fill='white')
            d.text((left+209,498),str(y['support_target'][j].tolist())+' (1/0/-1)',font=small,fill='white')
            mass=mm[j]['end']['full96_average_mass']
            for k,label in enumerate(['outside','visible','foreign','self U','ambig U','register U','invalid']):
                s='missing' if mass is None else format(mass[k],'.6f');d.text((left+209,528+k*19),label+': '+s,font=small,fill='white')
            own=np.r_[y['owner_soft'][j],y['owner_ambiguous_mass'][j],y['owner_invalid_mass'][j]]
            d.text((left,706),'Owner T/F/BG/AMB/INVALID',font=small,fill='white')
            d.text((left,726),' / '.join('U' if not np.isfinite(v) else format(v,'.3f') for v in own),font=small,fill='white')
            iq=r['image_original_query_xy'];d.text((left,750),'Image-original q: '+('missing' if iq is None else ','.join(format(v,'.3f') for v in iq)),font=small,fill='#edc98d')
            variant_rows.append(dict(diagnostic_row_index=j,row_id=r['row_id'],variant=r['variant'],source_image_original_global_index=r['source_image_original_global_index'],anchor_original_global_index=r['anchor_original_global_index'],raw_actual_sha256=ah(raw),valid_actual_sha256=ah(valid),images=paths))
        card=cd/'CARD.png';im.save(card);manifest.append(dict(**case,card=binding(card),variants=variant_rows))
    mp=O/'ALL64_CARDS_256_SLOTS.json';write(mp,manifest)
    intro='<meta charset="utf-8"><title>고정 same-query SOURCE TRAIN</title><p>8개 factor 조합별 첫 base, 4variant×native8=256 crop. COVER 초기 q/L 고정 진단이며 production replay·native E gold·학습 채택이 아닙니다. 전체 crop 화상 검토 완료 주장은 별도 QA receipt에만 기록합니다.</p>'
    content=intro+''.join('<h3>'+html.escape(m['case_id'])+'</h3><a href="'+m['case_id']+'/CARD.png"><img style="width:100%;max-width:1820px" src="'+m['case_id']+'/CARD.png"></a>' for m in manifest)
    (O/'index.html').write_text(content)
    for m in manifest:need(Path(m['card']['path']).exists(),'local card link')
    write(O/'COMPLETE.json',dict(completed=True,PASS=True,code=binding(__file__),protocol=binding(PROTO),plan=binding(PLAN),entry=binding(entry),common_complete=e['common_complete'],cards64=64,crop_rows256=256,all_raw_valid_PNG_pixel_exact=True,manifest=binding(mp),index=binding(O/'index.html'),actual_view_tool_images=0,actual_visual_QA_pending=True,no_browser_execution=True,source_TRAIN_only=True,new_fit_or_forward=0,selected_model=None))
if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
