#!/usr/bin/env python3
"""Fixed SOURCE TRAIN displays; unchanged saved crops, no labels or fitting."""
import argparse
import ast
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_source_train_native_point_control_alignment_20261025_v209'
H=V/'fixed8_visual_preparation_c'
SCOPE=V/'ROOT_PROSPECTIVE_FIXED8_SOURCE_NATIVE_POINT_VISUAL_SCOPE.json'
OLD=V/'native_control_authority_preparation_c'
ALPHA_CODE=OLD/'check_saved_train_point_alpha_alignment.py'
WHITELIST=OLD/'TRAIN82_PACKET_FOREIGN_ALPHA_WHITELIST.json'
ASSOCIATIONS=OLD/'ALL1968_TRAIN_POINT_ALPHA_ASSOCIATIONS.jsonl'
SAMPLER=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'
PROTOCOL=H/'PROTOCOL_BEFORE_FIXED8_SOURCE_RGB_MEMBERS.json'
SELECTION=H/'SELECTED8_BEFORE_RGB.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
VARIANTS=('BEFORE','COVER','SHAM')
MEMBERS=('full_RGB','rgb96','valid96','query_xy','L','native_slot','original_global_index',
         'original_occurrence_id','original_query_available','foreign_alpha')
IDENTITY=('original_global_index','original_occurrence_id','query_xy','L','original_query_available')
CYAN=(0,235,245);GREEN=(80,255,80)


def need(value,message):
    if not value:raise AssertionError(message)


def safe(path):
    raw=str(path);need('v112'not in raw.lower(),'Forbidden lexical path before filesystem')
    p=Path(raw);need(p.is_absolute()and'..'not in p.parts,'Absolute literal path')
    for item in reversed((p,*p.parents)):need(not item.is_symlink(),'No symlink: '+str(item))
    return p


def sha(path,binary=False):
    p=safe(path);need(binary or p.suffix in('.json','.jsonl','.py','.md'),'Text-only preparation')
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def bind(path,binary=False):return dict(path=str(path),sha256=sha(path,binary))
def read(path):return json.loads(safe(path).read_text())
def lines(path):return [json.loads(s)for s in safe(path).read_text().splitlines()if s]
def check(b):need(bind(b['path'])==b,'Exact text binding');return b['path']
def write(path,data):
    with safe(path).open('x')as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def pure_functions():
    ns=dict(np=np,math=math,require=need)
    for path,names in [(SAMPLER,['crop']),(ALPHA_CODE,['alpha_relation'])]:
        tree=ast.parse(safe(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
        need(len(nodes)==len(names),'Exact frozen pure functions only')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path)+'::pure_only','exec'),ns)
    return ns['crop'],ns['alpha_relation']


def select_fixed8(packets):
    ranked=[]
    for p in packets:
        k=int(p['attempt_id'].rsplit('NATIVE',1)[1])
        if k in(0,1):ranked.append(dict(p,requested_native_slot=k,
            selection_sha256=hashlib.sha256(('V209_FIXED8_VISUAL_V1|'+p['attempt_id']).encode()).hexdigest()))
    selected=[];counts=Counter();bases=set()
    for p in sorted(ranked,key=lambda r:(r['selection_sha256'],r['attempt_id'])):
        if counts[p['requested_native_slot']]>=4 or p['base_index']in bases:continue
        selected.append(p);counts[p['requested_native_slot']]+=1;bases.add(p['base_index'])
        if len(selected)==8:break
    need(len(selected)==8 and counts=={0:4,1:4}and len(bases)==8,'4+4 unique bases, no replacement')
    return selected


def raw96_reference(q,L,reference):
    if reference is None:return None
    return (47.5+96*(np.asarray(reference,np.float64)-np.asarray(q,np.float64))/np.float64(L)).tolist()


def scaled_center(xy,scale):return tuple((float(x)+.5)*scale-.5 for x in xy)
def font(size):return ImageFont.truetype('DejaVuSans.ttf',size)


def wrapped(draw,text,x,y,width,size=18,fill=(235,235,235)):
    f=font(size);words=text.split();line=''
    for word in words:
        trial=(line+' '+word).strip()
        if line and draw.textlength(trial,font=f)>width:
            draw.text((x,y),line,font=f,fill=fill);y+=size+7;line=word
        else:line=trial
    if line:draw.text((x,y),line,font=f,fill=fill);y+=size+7
    return y


def cross(draw,xy,color,r=7,width=2):
    x,y=xy;draw.line((x-r,y,x+r,y),fill=color,width=width);draw.line((x,y-r,x,y+r),fill=color,width=width)


def crop_marked(rgb,q,reference,L):
    out=Image.fromarray(rgb).resize((384,384),Image.Resampling.NEAREST);d=ImageDraw.Draw(out)
    cross(d,scaled_center([47.5,47.5],4),CYAN,11,3)
    local=raw96_reference(q,L,reference)
    if local is not None and all(-.5<=z<95.5 for z in local):
        x,y=scaled_center(local,4);d.rectangle((x-10,y-10,x+10,y+10),outline=GREEN,width=3)
    return out


def full_marked(rgb,q8,available,selected,reference):
    out=Image.fromarray(rgb);d=ImageDraw.Draw(out);h,w=rgb.shape[:2]
    for k,(xy,av)in enumerate(zip(q8,available)):
        if not av:continue
        x,y=map(float,xy)
        if -.5<=x<w-.5 and -.5<=y<h-.5:
            cross(d,(x,y),CYAN,7,3 if k==selected else 2)
            d.text((min(w-44,max(0,x+8)),min(h-22,max(0,y+5))),f'q{k}',font=font(15),fill=CYAN,stroke_width=1,stroke_fill='black')
    if reference is not None:
        x,y=map(float,reference)
        if -.5<=x<w-.5 and -.5<=y<h-.5:
            d.rectangle((x-9,y-9,x+9,y+9),outline=GREEN,width=3)
            d.text((min(w-66,max(0,x+11)),min(h-22,max(0,y-22))),f'ref{selected}',font=font(15),fill=GREEN,stroke_width=1,stroke_fill='black')
    return out


def make_card(title,images,marked,rows):
    canvas=Image.new('RGB',(1216,1160),(22,24,28));d=ImageDraw.Draw(canvas)
    y=wrapped(d,title,16,12,1184,24)
    y=wrapped(d,'SOURCE TRAIN | cyan cross = unchanged initializer q; green square = saved synthetic point0 reference. Geometry display only.',16,y,1184,19)
    top=max(y+12,116)
    for j,var in enumerate(VARIANTS):
        x=16+j*400;d.text((x,top),var+' | saved CLEAN96',font=font(21),fill='white')
        canvas.paste(images[j].resize((384,384),Image.Resampling.NEAREST),(x,top+34))
        d.text((x,top+430),'Same crop + display markers',font=font(19),fill='white');canvas.paste(marked[j],(x,top+462))
        t=top+860;r=rows[j]
        t=wrapped(d,'point alpha: '+r['point_alpha_status']+' | query alpha: '+r['query_alpha_status'],x,t,384,18)
        t=wrapped(d,'original point0: '+r['original_point0_state']+' | reference raw96: '+str([round(z,4)for z in r['reference_raw96_xy']]) if r['reference_raw96_xy']is not None else 'reference raw96: UNKNOWN',x,t,384,17)
        need(t<1122,'Card caption fits fixed canvas')
    wrapped(d,'COVERED/CLEAR are saved foreign raster-opacity relations, separate from point D/S. No native E label, recentering, physical-gold or transfer claim.',16,1124,1184,16)
    return canvas


def fixture_checks():
    crop,alpha_relation=pure_functions()
    rgb=np.zeros((120,120,3),np.uint8);yy,xx=np.indices((120,120));rgb[:,:,0]=xx;rgb[:,:,1]=yy
    a,v=crop(rgb,np.array([59.5,59.5],np.float64),96.,96)
    need(np.array_equal(a.transpose(1,2,0),rgb[12:108,12:108])and v.all(),'Exact unchanged F64 integer-grid fixture')
    q=[50.,60.];ref=[55.,57.];loc=raw96_reference(q,48.,ref)
    need(loc==[57.5,41.5]and scaled_center([47.5,47.5],4)==(191.5,191.5),'Raw96 algebraic/display pixel centers')
    zero=np.zeros((96,96),bool);need(alpha_relation(zero,q)['status']=='CLEAR','Query and reference relation pure')
    clean=[Image.fromarray(np.transpose(a,(1,2,0)))]*3
    marked=[crop_marked(np.transpose(a,(1,2,0)),q,ref,48.)]*3
    rr=[dict(point_alpha_status='EDGE_UNKNOWN',query_alpha_status='EDGE_UNKNOWN',original_point0_state='D',reference_raw96_xy=loc)]*3
    card=make_card('INVENTED LAYOUT | V189_BASE000_NATIVE0 | base 0 | original native 0',clean,marked,rr)
    need(card.size==(1216,1160),'Bounded legible contact layout; no real image read')
    # Selection fixture verifies quotas, uniqueness and input-order invariance.
    candidates=[dict(attempt_id=f'FIXTURE_BASE{b:03d}_NATIVE{k}',base_index=b)for b in range(12)for k in(0,1)]
    s=select_fixed8(candidates);need(select_fixed8(list(reversed(candidates)))==s,'Stable metadata-only selection')
    return dict(completed=True,PASS=True,groups=4,assertions=6,invented_only=True,
        source_payload_stat_hash_header_decode=0,real_tool_views=0)


def prepare():
    scope=read(SCOPE);done=read(check(scope['actual_alignment_complete']));qa=read(check(scope['independent_actual_QA']))
    need(done['completed']and done['PASS']and qa['completed']and qa['PASS'],'Closed saved alpha metadata/independent QA')
    packets=read(WHITELIST)['packets'];allrows=lines(ASSOCIATIONS)
    need(len(packets)==82 and len(allrows)==1968,'Unchanged source TRAIN census')
    chosen=select_fixed8(packets)
    for c in chosen:
        rr=[r for r in allrows if r['attempt_id']==c['attempt_id']and r['variant']=='BEFORE']
        rr.sort(key=lambda r:r['native_slot']);need([r['native_slot']for r in rr]==list(range(8)),'All8 original identities')
        need(all(r['role']=='TRAIN'and r['source_packet']==c['binding']for r in rr),'Same fixed TRAIN packet')
        c['original_native8']=rr;c['selected_point']=rr[c['requested_native_slot']]
    write(SELECTION,dict(completed=True,PASS=True,rule=scope['selection_before_RGB'],
        exact_implementation='Global ascending (SHA256(V209_FIXED8_VISUAL_V1|attempt_id),attempt_id); accept if native quota<4 and base unseen; stop at8. No image/status/admission/alpha/value filter.',
        selection_count=8,native_counts={'0':4,'1':4},unique_bases=8,cases=chosen,
        whitelist=bind(WHITELIST),associations=bind(ASSOCIATIONS),payload_access=0))
    write(CHECKS,fixture_checks())
    sources={k:bind(p)for k,p in [('scope',SCOPE),('whitelist',WHITELIST),('associations',ASSOCIATIONS),('original_sampler',SAMPLER),('saved_alpha_reader',ALPHA_CODE)]}
    write(PROTOCOL,dict(completed=True,code=bind(__file__),sources=sources,selection=bind(SELECTION),checks=bind(CHECKS),
        actual_alignment_complete=scope['actual_alignment_complete'],independent_actual_QA=scope['independent_actual_QA'],
        allowed_members=list(MEMBERS),members_never_decoded=['direct_end_external_target','new_native_visibility_target','new_quality_target','new_owner_target','broad_support2_diagnostic','end_related96_mask'],
        input_parity='All8 native x3variants per selected packet exact V90 crop AST at original F64 q/L, uint8 floor(clip(acc)+.5), outside-domain zero; compare saved rgb96/valid96. Display uses saved bytes, never recomputed inputs.',
        markers='Clean PNGs untouched; cyan original query q, raw96 q=(47.5,47.5), green saved source point0 reference. raw96 reference=47.5+96*(reference-q)/L. Nearest-neighbor display x=(raw96_x+.5)*4-.5; markers separate display only. Fullimage q8/ref use original fullimage coordinates.',
        output_policy='Saved clean raw96+valid+marked, original fullRGB clean+q8/ref context, cards/CSV/relative MD+HTML/IMAGE_MANIFEST/COMPLETE. Fixed8 retained on visual issues; mismatch aborts with failure preserved. Root actual view separately required.',
        new_labels_or_admission=False,fit_forward=0,protected_access=0,original_q_L_native_U_sampler_unchanged=True))
    template=dict(approved_fixed8_SOURCE_TRAIN_native_point_visual=False,automatic_retry=False,
        new_labels_or_admission=False,fit=False,model_forward=False,root_actual_scope=None,
        code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),selection=bind(SELECTION),checks=bind(CHECKS),
        output_dir=str(V/'fixed8_source_native_point_visual_v1'))
    write(H/'ROOT_ACTUAL_ENTRY_FALSE_TEMPLATE.json',template)
    write(H/'FINAL_READY.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),
        selection=bind(SELECTION),checks=bind(CHECKS),false_entry=bind(H/'ROOT_ACTUAL_ENTRY_FALSE_TEMPLATE.json'),
        exact_selected_packets8=True,unique_bases8=True,actual_payload_stat_hash_header_decode=0,actual_views=0,
        new_labels_or_admission=False,fit_forward=0))
    print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),cases=[c['attempt_id']for c in chosen])))


def run(entry_path):
    e=read(entry_path);need(e.get('approved_fixed8_SOURCE_TRAIN_native_point_visual')is True and e.get('automatic_retry')is False,'Separate actual root entry')
    need(e.get('new_labels_or_admission')is False and e.get('fit')is False and e.get('model_forward')is False,'Display only')
    need(e.get('root_actual_scope')is not None,'Root actual scope binding required');rp=Path(check(e['root_actual_scope']));need(rp.parent==V,'Root current phase scope only')
    for k,p in [('code',Path(__file__)),('protocol',PROTOCOL),('scope',SCOPE),('selection',SELECTION),('checks',CHECKS)]:need(e[k]==bind(p),'Frozen '+k)
    protocol=read(PROTOCOL)
    for b in protocol['sources'].values():check(b)
    for key in('actual_alignment_complete','independent_actual_QA'):need(read(check(protocol[key]))['PASS']is True,'Closed '+key)
    need(read(CHECKS)['PASS']is True,'Frozen fixtures');selection=read(SELECTION);need(selection['selection_count']==8,'Same fixed8')
    out=safe(e['output_dir']);need(out.parent==V and not out.exists(),'Fresh current phase visual child');out.mkdir()
    write(out/'STARTED.json',dict(entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),selection=bind(SELECTION)))
    crop,relation=pure_functions();manifest=[];table=[];casepages=[];parity=0
    def save(im,path,role,case):
        need(not path.exists(),'No overwrite');im.save(path,format='PNG');b=bind(path,True)
        manifest.append(dict(path=str(path.relative_to(out)),binding=b,role=role,attempt_id=case));return b
    try:
        for number,c in enumerate(selection['cases']):
            b=c['binding'];need(sha(b['path'],True)==b['sha256'],'Exact selected packet opaque SHA')
            with np.load(safe(b['path']),allow_pickle=False)as z:x={key:z[key]for key in MEMBERS}
            h,w=c['raw_hw'];need(x['full_RGB'].dtype==np.uint8 and x['full_RGB'].shape==(3,h,w,3),'Original fullRGB3')
            need(x['rgb96'].dtype==np.uint8 and x['rgb96'].shape==(3,8,96,96,3),'Original saved RGB96')
            need(x['valid96'].dtype==bool and x['valid96'].shape==(3,8,96,96),'Original saved valid96')
            need(x['foreign_alpha'].dtype==bool and x['foreign_alpha'].shape==(3,h,w),'Saved foreground alpha3')
            need(x['query_xy'].dtype==np.float64 and x['query_xy'].shape==(8,2)and x['L'].dtype==np.float64 and x['L'].shape==(8,),'Original F64 q/L')
            need(x['native_slot'].dtype==np.int8 and np.array_equal(x['native_slot'],np.arange(8,dtype=np.int8)),'Original native8')
            need(x['original_query_available'].dtype==bool and x['original_query_available'].shape==(8,),'Original availability8')
            for k,r in enumerate(c['original_native8']):
                for key in IDENTITY:
                    a=x[key][k];expected=r[key]
                    need(np.array_equal(a,expected,equal_nan=True)if key in('query_xy','L')else a==expected,'Original metadata exact '+key)
                for vi in range(3):
                    if np.isfinite(x['query_xy'][k]).all()and np.isfinite(x['L'][k])and x['L'][k]>0:
                        rr,vv=crop(x['full_RGB'][vi],x['query_xy'][k],float(x['L'][k]),96)
                        rr=rr.transpose(1,2,0);vv=vv[0]
                    else:rr=np.zeros((96,96,3),np.uint8);vv=np.zeros((96,96),bool)
                    need(np.array_equal(rr,x['rgb96'][vi,k])and np.array_equal(vv,x['valid96'][vi,k]),'Exact V90 all8 crop parity, no input resampling replacement');parity+=1
            directory=out/f'CASE_{number:02d}_{c["attempt_id"]}';directory.mkdir();k=c['requested_native_slot'];r=c['selected_point'];q=r['query_xy'];ref=r['saved_anchor_projected_reference_xy'];L=r['L'];local=raw96_reference(q,L,ref)
            clean=[];marked=[];case_rows=[];fullcontexts=[]
            for vi,var in enumerate(VARIANTS):
                clean_im=Image.fromarray(x['rgb96'][vi,k]);clean.append(clean_im)
                mk=crop_marked(x['rgb96'][vi,k],q,ref,L);marked.append(mk)
                save(clean_im,directory/(var+'_CLEAN96.png'),'saved_unchanged_raw96',c['attempt_id'])
                save(Image.fromarray(x['valid96'][vi,k].astype(np.uint8)*255),directory/(var+'_VALID96.png'),'saved_unchanged_valid96',c['attempt_id'])
                save(mk,directory/(var+'_MARKED96_DISPLAY.png'),'display_only_q_reference_markers',c['attempt_id'])
                save(Image.fromarray(x['full_RGB'][vi]),directory/(var+'_FULL_RGB.png'),'saved_fullRGB_clean',c['attempt_id'])
                fm=full_marked(x['full_RGB'][vi],x['query_xy'],x['original_query_available'],k,ref);fullcontexts.append(fm)
                save(fm,directory/(var+'_FULL_Q8_REFERENCE_CONTEXT.png'),'fullimage_display_q8_and_saved_reference',c['attempt_id'])
                pa=relation(x['foreign_alpha'][vi],ref);qa=relation(x['foreign_alpha'][vi],q)
                row=dict(case_number=number,attempt_id=c['attempt_id'],source_base=c['base_index'],variant=var,native_slot=k,
                    original_query_xy=q,original_L=L,original_query_available=r['original_query_available'],
                    saved_reference_fullimage_xy=ref,reference_raw96_xy=local,
                    original_point0_state=r['saved_original_point0_combined_state'],point_alpha_status=pa['status'],query_alpha_status=qa['status'],
                    point_alpha_relation=pa,query_alpha_relation=qa,source_packet=b,training_admission=False,new_visibility_target_created=False)
                table.append(row);case_rows.append(row)
            title=f'{c["attempt_id"]} | SOURCE TRAIN base {c["base_index"]} | unchanged native {k}'
            save(make_card(title,clean,marked,case_rows),directory/'CARD.png','triplet_clean_and_marker_contact',c['attempt_id'])
            context=Image.new('RGB',(3*w+64,h+150),(22,24,28));d=ImageDraw.Draw(context)
            wrapped(d,title+' | original FULLIMAGE q8 context',16,10,context.width-32,22)
            wrapped(d,'Cyan q0..q7 = original initializer; green ref = saved synthetic point0 projection. No GT recenter / newly inferred coordinate.',16,48,context.width-32,18)
            for vi,var in enumerate(VARIANTS):
                xx=16+vi*(w+16);d.text((xx,106),var,font=font(22),fill='white');context.paste(fullcontexts[vi],(xx,138))
            save(context,directory/'FULL_CONTEXT_TRIPLET.png','original_fullimage_triplet_context',c['attempt_id'])
            casepages.append((c['attempt_id'],directory.name))
        need(parity==192 and len(table)==24 and len(casepages)==8,'8 cases x3variants x8 parity')
        write(out/'ALL24_POINT_QUERY_ALPHA_COORDINATES.json',dict(completed=True,PASS=True,rows=table))
        with (out/'ALL24_POINT_QUERY_ALPHA_COORDINATES.csv').open('x')as f:
            fields=['case_number','attempt_id','source_base','variant','native_slot','original_query_xy','original_L','original_query_available','saved_reference_fullimage_xy','reference_raw96_xy','original_point0_state','point_alpha_status','query_alpha_status']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in table:writer.writerow({k:json.dumps(row[k])if isinstance(row[k],list)else row[k]for k in fields})
        intro='# 고정 8개 SOURCE TRAIN point/query control 시각화\n\n원 native0·1 각4개, 서로 다른8 base를 RGB 열람 전에 hash로 고정했습니다. 표의 COVERED/CLEAR는 저장 foreign raster opacity이며 새 native D/E 라벨이 아닙니다. cyan은 원 initializer query, green은 저장 synthetic point0 reference입니다. 원 q/L/native/U·sampler는 그대로이며 clean PNG에 marker를 넣지 않았습니다. 실제 tool view는 별도 receipt로만 기록합니다.\n\n'
        md=intro+'[좌표/alpha 표](ALL24_POINT_QUERY_ALPHA_COORDINATES.csv) · [JSON](ALL24_POINT_QUERY_ALPHA_COORDINATES.json)\n\n'
        body='<h1>Fixed8 SOURCE TRAIN native point / query</h1><p>Cyan: original initializer query. Green: saved synthetic point0 reference. Raster-opacity diagnostic only; no new labels, recentering or fit.</p><p><a href="ALL24_POINT_QUERY_ALPHA_COORDINATES.csv">Coordinate/opacity table</a></p>'
        for name,directory in casepages:
            md+=f'## {name}\n\n![raw96 clean + markers]({directory}/CARD.png)\n\n![original fullimage q8 context]({directory}/FULL_CONTEXT_TRIPLET.png)\n\n'
            body+=f'<h2>{html.escape(name)}</h2><img style="max-width:100%" src="{directory}/CARD.png"><img style="max-width:100%" src="{directory}/FULL_CONTEXT_TRIPLET.png">'
        with (out/'PORTABLE_GALLERY_KO.md').open('x')as f:f.write(md)
        with (out/'index.html').open('x')as f:f.write('<!doctype html><meta charset="utf-8"><title>Fixed8 source point controls</title>'+body)
        write(out/'IMAGE_MANIFEST.json',dict(completed=True,PASS=True,images=manifest,source_packets=[c['binding']for c in selection['cases']],decoded_members=list(MEMBERS)))
        outputs={name:bind(out/name)for name in('ALL24_POINT_QUERY_ALPHA_COORDINATES.json','IMAGE_MANIFEST.json','PORTABLE_GALLERY_KO.md')}
        # HTML/CSV are generated text; hash directly through the bounded output path.
        for name in('index.html','ALL24_POINT_QUERY_ALPHA_COORDINATES.csv'):outputs[name]=bind(out/name,True)
        write(out/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry_path),selection=bind(SELECTION),
            selected_cases=8,unique_bases=8,native_counts={'0':4,'1':4},variants_per_case=3,all8_original_crop_parities=parity,
            PNGs=len(manifest),outputs=outputs,source_member_decode=list(MEMBERS),input_geometry_unchanged=True,
            actual_tool_views_by_renderer=0,new_labels_or_admission=False,fit_forward=0,protected_access=0))
        print(json.dumps(dict(completed=True,PASS=True,completion=bind(out/'COMPLETE.json'),parity192=parity,PNGs=len(manifest))))
    except BaseException:
        write(out/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,automatic_retry=False,traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry');a=p.parse_args()
    prepare()if a.prepare else run(a.entry)
