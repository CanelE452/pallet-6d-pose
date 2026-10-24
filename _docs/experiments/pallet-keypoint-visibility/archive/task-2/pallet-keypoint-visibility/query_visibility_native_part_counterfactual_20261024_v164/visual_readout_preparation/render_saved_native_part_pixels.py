"""Render saved V164 pixel inputs after an explicit completed-generation trigger.
No canonical source, donor, encoder, feature, weight, initializer or label reader.
Only bound saved input NPZ and proof metadata are used. Raw assets are untouched;
q/weak-region/actual-RGB-change overlays occur on separate display copies.
"""
from pathlib import Path
import argparse, hashlib, html, json, math, traceback
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw, ImageFont
H=Path(__file__).resolve().parent;V=H.parent
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_PIXEL_VISUAL_READOUT.json'
GEN=V/'generator_preparation/PROTOCOL_BEFORE_NATIVE_PART_SOURCE_PIXEL_BODIES.json'
GEN_SHA='6ecba959a7e38008bb7d7a05509122f3a293928eb4379855b516a00add524aea'
OUT=V/'visual_readout_v1'
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):
    p=Path(b['path']);assert sha(p)==b['sha256'],str(p);return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def font(n):return ImageFont.truetype(FONT,n)
def save_rgb(p,a):
    assert a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3
    Image.fromarray(a).save(p)
    with Image.open(p) as z:assert exact(np.array(z.convert('RGB')),a)
    return bind(p)
def target_display(a):
    assert a.dtype==np.float32 and a.shape==(224,224,3) and np.isfinite(a).all()
    return np.floor(np.clip(a,0,255)+.5).astype(np.uint8)
def fit_rgb(a,wh):
    im=Image.fromarray(a);im.thumbnail(wh,Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',wh,'#edf0f3');canvas.paste(im,((wh[0]-im.width)//2,(wh[1]-im.height)//2));return canvas
def overlay(a,before,q,box):
    # A separate display copy only. Exact raw PNG is exported separately.
    im=Image.fromarray(a.copy()).convert('RGB')
    changed=np.any(a!=before,axis=-1)
    if changed.any():
        z=np.array(im);z[changed]=np.floor(z[changed].astype(float)*.6+np.array([255,30,30])*.4+.5).astype(np.uint8);im=Image.fromarray(z)
    d=ImageDraw.Draw(im);x,y=map(float,q);r=9
    d.line((x-r,y,x+r,y),fill='#ff00ef',width=3);d.line((x,y-r,x,y+r),fill='#ff00ef',width=3)
    if box is not None:
        x0,y0,x1,y1=box;d.rectangle((x0,y0,x1-1,y1-1),outline='#00e7ff',width=3)
    return np.array(im),int(changed.sum())
def wrap(draw,text,x,y,width,f,fill='#18232e',spacing=4):
    words=str(text).split();line='';yy=y
    for word in words:
        trial=(line+' '+word).strip()
        if draw.textlength(trial,font=f)>width and line:
            draw.text((x,yy),line,font=f,fill=fill);yy+=f.size+spacing;line=word
        else:line=trial
    if line:draw.text((x,yy),line,font=f,fill=fill);yy+=f.size+spacing
    return yy
def draw_card(a,fr,r,arrays,assets,folder):
    W,Hh=2040,1510;card=Image.new('RGB',(W,Hh),'#f8fafc');d=ImageDraw.Draw(card)
    k=a['native_slot'];reg=a['region'];box=reg['original_fullRGB_bbox_xyxy'];q=fr['query_xy'][k];L=fr['L'][k]
    d.text((20,16),f"{a['pair_id']} | native{k} {reg['native_name']} | {reg['region_status']} | proof PASS={r['PASS']}",font=font(30),fill='#10293b')
    d.text((20,60),f"Same original q=({q[0]:.3f},{q[1]:.3f}), L={L:.3f}, instance={reg['selected_gt_instance_index']}, split={reg['split']}",font=font(23),fill='#10293b')
    d.text((20,95),f"Donor {a['donor_id']} | all new native / quality / owner targets = -1 | NOT GOLD | NOT TRAINING ADMITTED",font=font(22),fill='#a14700')
    d.text((20,130),"Weak coarse material-end box only. q is unchanged. Visibility and literal-center material/relation are different observations.",font=font(21),fill='#a14700')
    before=arrays.get('BEFORE',{}).get('full_RGB')
    for col,name in enumerate(VARIANTS):
        x=20+col*675;d.text((x,174),name,font=font(26),fill='#10293b')
        if name not in arrays:
            yy=wrap(d,"NO CONSTRUCTED PIXEL INPUT: "+r['reason'],x,230,640,font(25),fill='#7b3434')
            wrap(d,"No source image or guessed ROI is fabricated for this missing variant.",x,yy+15,640,font(22));continue
        z=arrays[name];raw=z['full_RGB']
        d.text((x,212),"SAVED full_RGB: clean display resize",font=font(21),fill='#10293b')
        card.paste(fit_rgb(raw,(640,360)),(x,242))
        d.text((x,615),"DISPLAY COPY: q / weak bbox / actual RGB changes",font=font(20),fill='#10293b')
        over,count=overlay(raw,before,q,box)
        card.paste(fit_rgb(over,(640,360)),(x,646))
        d.text((x,1018),f"Actual changed raw pixels vs BEFORE: {count}",font=font(21),fill='#10293b')
        d.text((x,1056),"RAW96 (2x nearest)",font=font(19),fill='#10293b')
        d.text((x+215,1056),"central32 (6x nearest)",font=font(19),fill='#10293b')
        d.text((x+430,1056),"target224 display",font=font(19),fill='#10293b')
        card.paste(Image.fromarray(z['rgb96'][k]).resize((192,192),Image.Resampling.NEAREST),(x,1086))
        card.paste(Image.fromarray(z['central32'][k]).resize((192,192),Image.Resampling.NEAREST),(x+215,1086))
        card.paste(Image.fromarray(target_display(z['target224'])).resize((200,200),Image.Resampling.NEAREST),(x+430,1086))
        center=z['valid96'][k,47:49,47:49]
        d.text((x,1302),f"center4 valid: {int(center.sum())}/4 {center.astype(int).tolist()}",font=font(19),fill='#10293b')
        d.text((x,1330),f"query_available continuous={bool(z['query_available'][k])}; construct={bool(z['crop_constructed'][k])}",font=font(18),fill='#10293b')
        d.text((x,1358),f"nearest={bool(z['nearest_valid'][k])}; context={bool(z['context_available'][k])}",font=font(18),fill='#10293b')
    d.text((20,1410),"Raw96 / central32 PNGs are exact saved arrays. target224 PNG rounds float32 for DISPLAY only; exact tensor remains in bound NPZ.",font=font(21),fill='#364859')
    d.text((20,1444),"Red = actual RGB difference, cyan = coarse weak box, magenta = original q. Overlays are separate display copies, never model inputs.",font=font(21),fill='#364859')
    path=folder/'CARD.png';card.save(path);return bind(path)
def census_png(attempts,records):
    W=1460;im=Image.new('RGB',(W,1010),'#f8fafc');d=ImageDraw.Draw(im)
    d.text((20,15),"ALL66 ATTEMPT CENSUS | 22 fixed cases x 3 fixed TRAIN donors",font=font(27),fill='#10293b')
    d.text((20,58),"Green: saved proof PASS | Red: retained failure | Gray: UNKNOWN region, no constructed intervention",font=font(20),fill='#253c50')
    by={r['pair_id']:r for r in records}
    for i in range(22):
        aa=attempts[i*3:(i+1)*3];reg=aa[0]['region'];y=105+i*38
        d.text((22,y),reg['case_id']+f" native{reg['native_slot']} "+reg['region_status'],font=font(21),fill='#10293b')
        for j,a in enumerate(aa):
            r=by[a['pair_id']];x=430+j*330
            color='#d8dde3' if reg['region_status']=='UNKNOWN' else ('#a2d9b1' if r['PASS'] else '#efb1ab')
            d.rounded_rectangle((x,y-1,x+308,y+29),radius=4,fill=color)
            text=a['pair_id']+' | '+('UNKNOWN' if reg['region_status']=='UNKNOWN' else ('PASS' if r['PASS'] else 'FAIL retained'))
            d.text((x+8,y+2),text,font=font(18),fill='#10293b')
    d.text((20,958),"No substitutions, no new labels/admission, no full WOOD/SQUARE/native8 qualification claim.",font=font(20),fill='#a14700')
    p=OUT/'ALL66_CENSUS.png';im.save(p);return bind(p)
def run(trigger_path):
    p=read(PROTOCOL);e=read(trigger_path)
    assert e['completed'] is True and e['source_only'] is True and e['allow_saved_pixel_visual_readout_only'] is True
    assert e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL)
    for k in ('encoder_allowed','fit_allowed','current_allowed','held_allowed','V112_allowed','PnP_allowed'):assert e[k] is False
    assert sha(GEN)==GEN_SHA and p['generator_protocol']==bind(GEN)
    gp=read(check(p['generator_protocol']));inputs=read(check(p['fixed_inputs']))
    complete_path=check(e['pixel_complete']);complete=read(complete_path)
    assert str(complete_path)==str(Path(gp['output'])/'COMPLETE.json') and complete['completed'] and complete['all66_attempts_retained']
    assert complete['protocol']==p['generator_protocol'] and complete['inputs']==p['fixed_inputs']
    proofpath=check(complete['proofs']);records=lines(proofpath);attempts=inputs['attempts']
    assert len(records)==len(attempts)==66 and [r['pair_id'] for r in records]==[a['pair_id'] for a in attempts]
    frames={r['frame_index']:r for r in inputs['frames']}
    OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',dict(trigger=bind(trigger_path),protocol=bind(PROTOCOL),pixel_complete=bind(complete_path),all66_retained=True))
    inventory=[];qa=[];started=datetime.now(timezone.utc).isoformat()
    try:
        for a,r in zip(attempts,records):
            fr=frames[a['frame_index']];k=a['native_slot']
            assert (r['case_id'],r['frame_index'],r['native_slot'],r['donor_id'])==(a['case_id'],a['frame_index'],a['native_slot'],a['donor_id'])
            assert r['region']==a['region'] and r['training_admitted'] is False and r['all_new_native_quality_owner_targets']==-1
            folder=OUT/a['pair_id'];folder.mkdir();arrays={};assets={};variantmeta={}
            for name in VARIANTS:
                vb=r['variants'][name]
                if 'path' not in vb:
                    assert vb['constructed'] is False
                    variantmeta[name]=dict(constructed=False,reason=vb['reason'],expected_query_available=fr['native_continuous_valid'])
                    continue
                path=check(vb)
                assert path.parent==Path(gp['output'])/a['pair_id'] and path.name==name+'_INPUTS.npz'
                with np.load(path,allow_pickle=False) as npz:
                    assert set(npz.files)==set(gp['output_schema']['NPZ_keys'])
                    z={key:npz[key].copy() for key in npz.files}
                assert z['full_RGB'].dtype==np.uint8 and list(z['full_RGB'].shape)==fr['raw_hw']+[3]
                assert z['rgb96'].dtype==np.uint8 and z['rgb96'].shape==(8,96,96,3)
                assert z['central32'].shape==(8,32,32,3) and exact(z['central32'],z['rgb96'][:,32:64,32:64])
                assert z['valid96'].dtype==bool and z['valid96'].shape==(8,96,96)
                for key in ('native_visibility_target','point_quality_target','owner_target'):assert z[key].shape==(8,) and (z[key]==-1).all()
                for key,expected in [('query_xy',np.array(fr['query_xy'],np.float64)),('L',np.array(fr['L'],np.float64)),
                    ('native_slot',np.array(fr['native_slot'],np.int8)),('query_available',np.array(fr['native_continuous_valid'],bool)),
                    ('crop_constructed',np.array(fr['feature_constructed'],bool)),('native_continuous_valid',np.array(fr['native_continuous_valid'],bool)),
                    ('nearest_valid',np.array(fr['nearest_valid'],bool))]:assert exact(z[key],expected),(a['pair_id'],name,key)
                assert z['query_id'].tolist()==fr['query_id'] and fr['native_slot']==list(range(8))
                arrays[name]=z
                assets[name]=dict(full_RGB=save_rgb(folder/(name+'_RAW_FULL_RGB.png'),z['full_RGB']),
                    selected_raw96=save_rgb(folder/(name+'_RAW96.png'),z['rgb96'][k]),
                    selected_central32=save_rgb(folder/(name+'_CENTRAL32.png'),z['central32'][k]),
                    target224_DISPLAY=save_rgb(folder/(name+'_TARGET224_DISPLAY.png'),target_display(z['target224'])))
                variantmeta[name]=dict(constructed=True,saved_NPZ=vb,raw_array_sha256={key:ash(z[key]) for key in ('full_RGB','rgb96','central32','target224','pasted_alpha')},
                    selected_raw96_array_sha256=ash(z['rgb96'][k]),selected_central32_array_sha256=ash(z['central32'][k]),all8_query_available=z['query_available'].tolist(),
                    all8_crop_constructed=z['crop_constructed'].tolist(),all8_nearest_valid=z['nearest_valid'].tolist(),
                    all8_center4_available=z['central4_available'].tolist(),all8_context_available=z['context_available'].tolist(),
                    all8_native_quality_owner_targets_unknown=True,center4_valid_bits=z['valid96'][k,47:49,47:49].astype(int).tolist())
            if a['region']['region_status']=='UNKNOWN':assert not arrays and all(not variantmeta[n]['constructed'] for n in VARIANTS)
            card=None
            if arrays:
                assert 'BEFORE' in arrays
                for name,z in arrays.items():
                    over,nchanged=overlay(z['full_RGB'],arrays['BEFORE']['full_RGB'],fr['query_xy'][k],a['region']['original_fullRGB_bbox_xyxy'])
                    assets[name]['full_RGB_DISPLAY_OVERLAY']=save_rgb(folder/(name+'_FULL_RGB_DISPLAY_OVERLAY.png'),over)
                    variantmeta[name]['actual_RGB_changed_pixel_count_vs_BEFORE']=nchanged
                card=draw_card(a,fr,r,arrays,assets,folder)
                qa.append(dict(pair_id=a['pair_id'],image=card,actually_viewed=False,required_tool='tools.view_image'))
            inventory.append(dict(pair_id=a['pair_id'],case_id=a['case_id'],native_slot=k,native_name=a['region']['native_name'],
                query_id=a['region']['query_id'],query_xy=a['region']['query_xy'],L=a['region']['L'],frame_index=a['frame_index'],
                source_id=a['region']['source_id'],split=a['region']['split'],selected_gt_instance_index=a['region']['selected_gt_instance_index'],
                donor_id=a['donor_id'],region_status=a['region']['region_status'],coarse_weak_original_fullRGB_bbox_xyxy=a['region']['original_fullRGB_bbox_xyxy'],
                generator_PASS=r['PASS'],generator_reason=r['reason'],generator_proof_binding=bind(Path(gp['output'])/a['pair_id']/'PROOF.json'),
                variants=variantmeta,assets=assets,card=card,new_native_quality_owner_targets=-1,training_admitted=False,
                display_overlays_separate_never_model_inputs=True,native_or_coordinate_gold=False))
        census=census_png(attempts,records)
        page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V164 all66 source pixel evidence</title>',
            '<style>body{font:16px system-ui;margin:24px;color:#142b3a}article{border-top:2px solid #ccd5dd;margin:30px 0;padding-top:16px}img.card{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccd5dd}.limit{color:#9c4700}.missing{padding:18px;background:#eef1f4}a{color:#135c93}</style>',
            '<h1>V164: 고정22사례 × 3 TRAIN donor = 전체66시도</h1>',
            '<p class="limit">약한 coarse 재료 끝 영역의 픽셀 개입 기록입니다. native/좌표/품질 gold가 아니며 새 native·quality·owner targets는 모두 −1, 학습 미승인입니다. 원래 q/L/native/instance/split/availability는 유지됩니다.</p>',
            '<p>청록 박스·자홍 q·빨강 변경 픽셀은 별도 표시 복사본입니다. 원본 fullRGB, raw96, central32, target224 표시 PNG와 정확한 저장 NPZ hash를 함께 제공합니다. target224 PNG는 float32 값을 반올림한 표시이며 정확한 입력 tensor는 NPZ에 있습니다. 부위 가시성과 literal 중심 재료/관계는 다릅니다.</p>',
            '<img class="card" src="ALL66_CENSUS.png" alt="All66 fixed attempt census"><ol>']
        for x in inventory:page.append('<li><a href="#'+x['pair_id']+'">'+x['pair_id']+'</a> '+x['region_status']+' | '+html.escape(x['generator_reason'])+'</li>')
        page.append('</ol>')
        for x in inventory:
            page.append('<article id="'+x['pair_id']+'"><h2>'+x['pair_id']+' | native'+str(x['native_slot'])+' '+x['native_name']+'</h2>')
            page.append('<p>'+html.escape('q='+str(x['query_xy'])+' L='+str(x['L'])+' donor='+x['donor_id']+' region='+x['region_status']+' reason='+x['generator_reason'])+'</p>')
            if x['card']:page.append('<img class="card" loading="lazy" src="'+x['pair_id']+'/CARD.png" alt="'+x['pair_id']+' three saved variants">')
            else:page.append('<div class="missing">UNKNOWN / no constructed pixel input. No guessed ROI or fabricated intervention image.</div>')
            page.append('<table><tr><th>Variant</th><th>Saved NPZ / exact SHA256</th><th>Saved clean display assets</th><th>Display-only overlay</th></tr>')
            for name in VARIANTS:
                z=x['variants'][name]
                if not z['constructed']:page.append('<tr><td>'+name+'</td><td colspan="3">No constructed input: '+html.escape(z['reason'])+'</td></tr>');continue
                b=z['saved_NPZ'];links=[]
                for key,item in x['assets'][name].items():
                    if key=='full_RGB_DISPLAY_OVERLAY':continue
                    links.append('<a href="'+x['pair_id']+'/'+Path(item['path']).name+'">'+key+'</a>')
                overlaylink=x['pair_id']+'/'+Path(x['assets'][name]['full_RGB_DISPLAY_OVERLAY']['path']).name
                page.append('<tr><td>'+name+'</td><td><a href="'+Path(b['path']).as_uri()+'">saved input</a><br><code>'+b['sha256']+'</code></td><td>'+' | '.join(links)+'</td><td><a href="'+overlaylink+'">q / weak box / actual RGB changes</a></td></tr>')
            page.append('</table></article>')
        page.append('</html>');gallery=OUT/'index.html';gallery.write_text('\n'.join(page))
        mp=OUT/'ALL66_VISUAL_MANIFEST.jsonl'
        with mp.open('x') as f:
            for x in inventory:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
        write(OUT/'PNG_ACTUAL_VIEW_QA_REQUIRED.json',dict(completed=False,actual_view_QA_pending=True,all_constructed_cards=qa,
            census=census,rule='Actually view every constructed-card PNG and census via tools.view_image; seal separate actual QA receipt. Rendering alone is not actual visual QA.'))
        assert len(inventory)==66 and len({x['pair_id'] for x in inventory})==66
        assert sum(x['region_status']=='UNKNOWN' for x in inventory)==15
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,render_only_complete=True,actual_view_QA_completed=False,
            count66=len(inventory),constructed_cards=len(qa),UNKNOWN15_retained_no_images=True,
            failed_attempts_retained=sum(not x['generator_PASS'] for x in inventory),manifest=bind(mp),gallery=bind(gallery),census=census,
            actual_QA_required=bind(OUT/'PNG_ACTUAL_VIEW_QA_REQUIRED.json'),trigger=bind(trigger_path),protocol=bind(PROTOCOL),
            generator_complete=bind(complete_path),no_source_donor_pixel_reread=True,no_models_features_weights_or_training=True,
            all_new_targets_minus1=True,training_admitted=False,native_or_coordinate_gold=False,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(completed=True,render_only=True,actual_QA_pending=True,complete=bind(OUT/'COMPLETE.json'))))
    except BaseException:
        write(OUT/'FAILED_RENDER.json',dict(completed=False,attempts_completed=len(inventory),traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--completed-trigger',type=Path,required=True);a=ap.parse_args();run(a.completed_trigger)

