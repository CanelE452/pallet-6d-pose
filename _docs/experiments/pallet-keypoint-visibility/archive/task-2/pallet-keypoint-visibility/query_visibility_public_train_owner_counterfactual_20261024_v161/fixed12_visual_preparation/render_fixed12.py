"""Fixed-before-pixels V161 gallery; source payloads open only after root entry.
No successful-case replacement, labels, model, encoder, or training.
"""
from pathlib import Path
import argparse, hashlib, html, importlib.util, json, sys, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL

H=Path(__file__).resolve().parent;V=H.parent
SELECTION=V/'metadata_assignment/FIXED12_VISUAL_SELECTION_BEFORE_PIXELS.json'
BUILDER=V/'pixel_builder_preparation/build_coherent_source_pixels.py'
BUILDER_SHA='835c2df24a1cad8bef97947bada0c28a2e8411b3115b757156bdaaa3d7a10875'
BUILDER_PROTOCOL=V/'pixel_builder_preparation/PROTOCOL_BEFORE_SOURCE_PIXEL_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_GENERATED_PAYLOADS.json';OUT=V/'fixed12_visual_v1'
FONT=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def builder():
    assert sha(BUILDER)==BUILDER_SHA
    spec=importlib.util.spec_from_file_location('v161_frozen_pixel_math',BUILDER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def font(n):return ImageFont.truetype(str(FONT),n)
def wrapped(d,xy,s,width=51,size=20,fill='#20344a',spacing=29):
    x,y=xy
    for line in textwrap.wrap(str(s),width=width,break_long_words=True):d.text((x,y),line,font=font(size),fill=fill);y+=spacing
    return y
def footprint_panel(mask,points,size=144):
    # Only contributing raw cells are colored; non-contributing bbox cells gray.
    xx=[p[0] for p in points];yy=[p[1] for p in points]
    x0,x1=min(xx),max(xx);y0,y1=min(yy),max(yy);ww=x1-x0+1;hh=y1-y0+1
    canvas=Image.new('RGB',(ww,hh),'#dbe2e8');a=np.asarray(canvas).copy()
    for x,y in points:a[y-y0,x-x0]=[16,150,112] if mask[y,x] else [246,164,56]
    return Image.fromarray(a).resize((size,size),Image.Resampling.NEAREST),(x0,y0,x1,y1)
def card(case,proof,payloads,m):
    c=Image.new('RGB',(1800,1450),'#f5f8fb');d=ImageDraw.Draw(c)
    status='픽셀 입력 증명 PASS' if proof['PASS'] else '실패 보존 · 대체/재시도 없음'
    color='#137d62' if proof['PASS'] else '#ad482b'
    d.text((22,20),f"{case['selection_rank']:02d} {case['pair_id']} · native {case['native_slot']} · {status}",font=font(31),fill=color)
    d.text((22,68),case['query_id'],font=font(23),fill='#1c3048')
    q=case['query_xy'];L=case['L']
    d.text((22,104),f"원래 q=({q[0]:.6f}, {q[1]:.6f})  L={L:.6f} · SOURCE_DEVELOPMENT / TRAIN · scale={case['scale']}",font=font(23),fill='#344c64')
    d.text((22,141),f"고정 donor: {case['donor_id']} · 전체 159 시도 중 사전고정12 · 실패도 원래 순서 유지",font=font(22),fill='#344c64')
    d.text((22,179),'원본 q/L 유지 · 아래 자홍색 십자는 화면 표시만 · 원래 약 owner TARGET는 계보이며 native D/E 정답이 아님',font=font(23),fill='#965111')
    points,inside=m.raw_footprint(q,L,case['raw_hw']);k=case['native_slot'];op=m.operators()
    for col,name in enumerate(VARIANTS):
        x=20+col*594
        d.rounded_rectangle((x-4,226,x+578,1322),radius=10,fill='white',outline='#cbd6df',width=2)
        d.text((x+8,238),name,font=font(27),fill='#1c3048')
        if name not in payloads:
            d.text((x+15,310),'미생성 · 실패 보존',font=font(28),fill='#ad482b')
            wrapped(d,(x+15,365),proof.get('reason','FAILURE_RETAINED'),width=38,size=22,fill='#ad482b')
            if proof.get('traceback'):wrapped(d,(x+15,470),proof['traceback'].strip().splitlines()[-1],width=38,size=20,fill='#6c3f30')
            continue
        a=payloads[name];rgb=a['full_RGB'];rh,rw=rgb.shape[:2]
        thumb=Image.fromarray(rgb).resize((560,315),Image.Resampling.BILINEAR);td=ImageDraw.Draw(thumb)
        mx,my=q[0]*560/rw,q[1]*315/rh;td.line((mx-5,my,mx+5,my),fill='#f000df',width=2);td.line((mx,my-5,mx,my+5),fill='#f000df',width=2)
        c.paste(thumb,(x+8,282));d.text((x+8,607),'정확한 저장 RGB96 · 3×',font=font(21),fill='#344c64')
        c.paste(Image.fromarray(a['rgb96'][k]).resize((288,288),Image.Resampling.NEAREST),(x+8,643))
        d.text((x+316,607),'동일 crop 중앙32 · 6×',font=font(20),fill='#344c64')
        c.paste(Image.fromarray(a['central32'][k]).resize((192,192),Image.Resampling.NEAREST),(x+316,643))
        mask=a['pasted_alpha'];sample=op['crop'](np.repeat(mask[:,:,None],3,2).astype(np.uint8)*255,q,L,96)[0][0]
        d.text((x+8,947),'붙인 alpha96 · 2×',font=font(20),fill='#344c64')
        c.paste(Image.fromarray(sample,'L').convert('RGB').resize((192,192),Image.Resampling.NEAREST),(x+8,981))
        d.text((x+229,947),'중심4의 raw 기여 셀',font=font(20),fill='#344c64')
        if inside:
            panel,box=footprint_panel(mask,points);c.paste(panel,(x+246,981))
            d.text((x+214,1135),f"{len(points)} cells · bbox {box}",font=font(16),fill='#344c64')
        else:d.text((x+229,995),'유효 영상 밖 기여 셀',font=font(20),fill='#ad482b')
        d.text((x+8,1190),f"중심4 alpha: {sample[47:49,47:49].tolist()}",font=font(19),fill='#344c64')
        d.text((x+8,1224),f"붙인 raw alpha mass: {int(mask.sum())}",font=font(19),fill='#344c64')
        if name=='ADJACENT_SHAM' and 'BEFORE' in payloads:
            same=exact(a['rgb96'][k,47:49,47:49],payloads['BEFORE']['rgb96'][k,47:49,47:49]);d.text((x+8,1258),f"BEFORE 중심4 RGB 동일: {same} · local alpha>0: {int((sample>0).sum())}",font=font(18),fill='#344c64')
    d.text((22,1343),'raw 기여 셀: 초록=붙인 alpha1 / 주황=alpha0 / 회색=비기여 bbox 셀 · BEFORE alpha0는 원래 TARGET 소유를 뜻하지 않음',font=font(21),fill='#344c64')
    d.text((22,1382),'약 owner 보조용 픽셀 증명만 · 모든 신규 native 가시성/좌표품질 UNKNOWN · 모델/학습/학습 편입 없음 · 자연 광학 가림 정답 아님',font=font(22),fill='#965111')
    return c
def invented():
    m=builder();checks={};rgb=np.full((720,1280,3),110,np.uint8);q8=np.tile([500.25,400.75],(8,1));q8[1:]+=np.arange(1,8)[:,None]*3
    fr=dict(query_xy=q8.tolist(),L=[100.]*8,feature_constructed=[True]*8,context_box_xyxy=[490.,390.,540.,440.],native_slot=list(range(8)),native_continuous_valid=[True]*8,nearest_valid=[True]*8)
    op=m.operators();a=m.materialize(rgb,fr,op);a['pasted_alpha']=np.zeros(rgb.shape[:2],bool)
    case=dict(selection_rank=1,pair_id='INVENTED',native_slot=0,query_id='INVENTED:native0',query_xy=q8[0].tolist(),L=100.,scale=.5,donor_id='INVENTED:DONOR',raw_hw=[720,1280])
    image=card(case,dict(PASS=True),{n:a for n in VARIANTS},m);checks['three_variant_card_size']=image.size==(1800,1450)
    fail=card(case,dict(PASS=False,reason='INVENTED_FAILURE_RETAINED',traceback='AssertionError: invented'),{'BEFORE':a},m);checks['failed_variants_draw_without_replacement']=fail.size==(1800,1450)
    checks['central32_display_exact_source_slice']=exact(a['central32'],a['rgb96'][:,32:64,32:64])
    checks['native_and_quality_UNKNOWN']=bool((a['native_visibility_target']==-1).all() and (a['point_quality_target']==-1).all())
    checks['proof_panel_alpha0_and1_distinct']=not np.array_equal(np.array(footprint_panel(np.zeros((3,3),bool),[(1,1)])[0]),np.array(footprint_panel(np.ones((3,3),bool),[(1,1)])[0]))
    assert all(checks.values());return dict(completed=True,PASS=True,invented_arrays_only=True,checks=checks,generated_source_payloads_read=0,encoder_model_fit_calls=0)
def prepare():
    sel=read(SELECTION);assert sel['before_any_new_pixels'] and len(sel['selected'])==12 and sel['no_failure_or_pixel_based_replacement']
    generated=H/'GENERATED_INVENTED_RENDER_CHECKS.json';write(generated,invented())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),selection=bind(SELECTION),builder=bind(BUILDER),builder_protocol=bind(BUILDER_PROTOCOL),font=bind(FONT),generated=bind(generated),runtime=dict(executable=sys.executable,numpy=np.__version__,Pillow=PIL.__version__),output=str(OUT),generated_source_payload_reads_during_preparation=0,selection_unchanged=True,failures_retained_no_replacement=True,source_generation_complete_required=True,complete_PASS_may_be_false_because_failures_are_retained=True,pages='12 individual1800x1450 PNG + 6 stacked2card contact pages + index.html + bound manifest',native_visibility_and_quality='UNKNOWN in every generated packet; no new label interpretation',owner_scope='weak auxiliary pixel/input proof only; no native/external optical occlusion truth or learned success claim',training_admitted=False)
    write(PROTOCOL,p);template=H/'ROOT_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(template,dict(approved_actual_fixed12_render=False,code=p['code'],protocol=bind(PROTOCOL),selection=p['selection'],source_generation_complete=None,source_only=True,encoder_allowed=False,model_allowed=False,fit_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,training_admitted=False,output=str(OUT)))
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(generated),entry_template=bind(template),generated_source_payloads_read=0,actual_rendering=False))
    print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)
def render(entry):
    e=read(entry);assert e['approved_actual_fixed12_render'] is True and e['source_only'] is True
    for k in ('encoder_allowed','model_allowed','fit_allowed','current_allowed','V112_allowed','PnP_allowed','training_admitted'):assert e[k] is False
    p=read(check(e['protocol']));assert e['protocol']==bind(PROTOCOL) and e['code']==p['code']==bind(__file__) and e['selection']==p['selection']
    for k in ('selection','builder','builder_protocol','font','generated'):check(p[k])
    assert read(check(p['generated']))['PASS'] and not OUT.exists() and e['output']==p['output']==str(OUT)
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve() and np.__version__==p['runtime']['numpy'] and PIL.__version__==p['runtime']['Pillow']
    complete=read(check(e['source_generation_complete']));assert complete['completed'] and complete['all159_attempts_retained'] and complete['source_only'] and not complete['training_admitted']
    assert complete['protocol']==p['builder_protocol'] and complete['fixed12']==p['selection']
    proofs=[json.loads(s) for s in check(complete['proofs']).read_text().splitlines() if s];assert len(proofs)==159
    index={r['pair_id']:r for r in proofs};selection=read(check(p['selection']))['selected'];m=builder();op=m.operators()
    OUT.mkdir();records=[];cards=[];htmlrows=[]
    for case in selection:
        r=index[case['pair_id']]
        assert r['pair_sha256']==case['pair_sha256'] and r['anchor_query_id']==case['anchor_query_id'] and r['donor_id']==case['donor_id'] and r['original_q']==case['query_xy'] and r['original_L']==case['L'] and r['native_slot']==case['native_slot'] and r['split']==case['split']
        payloads={};bindings={};k=case['native_slot'];bodychecks={}
        for name in VARIANTS:
            b=r['variants'][name]
            if 'path' not in b:continue
            with np.load(check(b),allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
            assert a['native_slot'].tolist()==list(range(8)) and a['query_id'][k]==case['query_id'] and a['query_xy'][k].tolist()==case['query_xy'] and float(a['L'][k])==case['L']
            assert (a['native_visibility_target']==-1).all() and (a['point_quality_target']==-1).all()
            assert exact(a['central32'],a['rgb96'][:,32:64,32:64])
            rebuilt,vm=op['crop'](a['full_RGB'],case['query_xy'],case['L'],96)
            assert exact(rebuilt.transpose(1,2,0).copy(),a['rgb96'][k]) and exact(vm[0],a['valid96'][k])
            bodychecks[name]=dict(selected_RGB96_from_same_fullRGB_exact=True,central32_direct_slice_exact=True,native_quality_UNKNOWN=True,RGB96_array_sha256=ash(a['rgb96'][k]),full_RGB_array_sha256=ash(a['full_RGB']))
            payloads[name]=a;bindings[name]=b
        image=card(case,r,payloads,m);path=OUT/(f"CASE_{case['selection_rank']:02d}_"+case['pair_id']+'.png');image.save(path);cards.append(image)
        records.append(dict(selection_rank=case['selection_rank'],pair_id=case['pair_id'],pair_sha256=case['pair_sha256'],query_id=case['query_id'],native_slot=case['native_slot'],original_q=case['query_xy'],original_L=case['L'],pixel_input_proof_PASS=r['PASS'],failure_reason=r['reason'] if not r['PASS'] else None,source_payloads=bindings,body_checks=bodychecks,card=bind(path),new_labels=False,training_admitted=False))
        htmlrows.append('<article><h2>'+html.escape(f"{case['selection_rank']:02d} {case['pair_id']} — "+('pixel proof PASS' if r['PASS'] else 'FAILURE RETAINED'))+'</h2><img src="'+path.name+'" alt="BEFORE COVER SHAM exact same-q source comparison"><p>'+html.escape(case['query_id'])+' · native/quality UNKNOWN · no training admission.</p></article>')
    contacts=[]
    for i in range(0,12,2):
        page=Image.new('RGB',(1800,2900),'white');page.paste(cards[i],(0,0));page.paste(cards[i+1],(0,1450));path=OUT/f'CONTACT_PAGE_{i//2+1:02d}.png';page.save(path);contacts.append(bind(path))
    page=OUT/'index.html';page.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V161 fixed12 source pixel proof</title><style>body{font:18px system-ui;max-width:1800px;margin:auto;padding:24px;background:#f5f8fb;color:#20344a}img{width:100%;height:auto}article{margin:36px 0}p{line-height:1.7}</style><h1>V161 사전고정12 · SOURCE TRAIN 픽셀 입력 증명</h1><p>예측/생성 픽셀 열람 전 선정한12개이며 실패도 대체하지 않습니다. 같은 원래 q/L의 BEFORE/COVER/ADJACENT_SHAM을 표시합니다. Owner 보조용 약증명만이며 모든 신규 native 가시성/좌표품질은 UNKNOWN입니다. 모델·학습·학습 편입 및 자연 광학 가림 정답 판정은 없습니다. absent native2/3은 성공/부정 근거가 아닙니다.</p>'+''.join(htmlrows)+'</html>')
    manifest=OUT/'FIXED12_RENDER_MANIFEST.json';write(manifest,dict(completed=True,selection=p['selection'],source_generation_complete=e['source_generation_complete'],records=records,contacts=contacts,HTML=bind(page)))
    write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),selection=p['selection'],source_generation_complete=e['source_generation_complete'],manifest=bind(manifest),cases=12,contact_pages=6,selected_pixel_proof_pass=sum(r['pixel_input_proof_PASS'] for r in records),selected_failures_retained=sum(not r['pixel_input_proof_PASS'] for r in records),no_failure_replacement=True,new_labels=False,training_admitted=False,encoder_model_fit_calls=0,actual_visual_QA_pending=True,HTML_browser_rendered=False))
    print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and not a.entry:prepare()
    elif a.entry and not a.prepare:render(a.entry)
    else:raise SystemExit('Use --prepare; actual payload rendering needs root --entry after generation completion')
