"""Blinded input-only review sheets. --prepare uses metadata and invented arrays only."""
from pathlib import Path
import argparse, ast, datetime, hashlib, html, json, sys, traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

H = Path(__file__).resolve().parent; V = H.parent
SCOPE = V/'ROOT_SCOPE_BEFORE_PUBLIC_HELD_PIXELS_OR_NEW_OWNER_ANNOTATIONS.json'
DEFINITION = V/'ROOT_WEAK_LABEL_DEFINITION_BEFORE_HELD_PIXELS.json'
READY = V/'metadata_readiness_c/REVIEW.json'
INPUT = V/'metadata_readiness_c/ALL360_HELD_INPUT_ONLY.jsonl'
PROTOCOL = H/'PROTOCOL_BEFORE_HELD_PIXEL_BODIES.json'
CHECKS = H/'GENERATED_INVENTED_CHECKS.json'
OUT = V/'blind_review_evidence_v1'
EXPECTED = {SCOPE:'c2018965ace4b5cbe4d4ef9a0ddc816d10514c73f4db3a0ff13c9f8a76ca9c70', DEFINITION:'fa9c00935a4c93d1af091eccba5aacda30bc98be8dcfaf205148eb2a3484df97', READY:'e43ace56dced63a9858e2e568634372f3b971c58cf5084845b75e5a1e63e7a5b', INPUT:'9c962418c6ba63589e9aaf8a64417bc26924ae4c8a6f448c7ad75ff00fba605f'}
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
SHEET_HW = (2440, 2400)
NEAREST = Image.Resampling.NEAREST
COLORS = ['#ff3737','#04bfe8','#ffe000','#ff69df','#48e577','#ff9933','#bfa4ff','#ffffff']

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def bind(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def ash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def checked(b): assert sha(b['path']) == b['sha256'], b['path']; return Path(b['path'])
def font(n): return ImageFont.truetype(str(FONT), n)
def rows():
    for p, h in EXPECTED.items(): assert sha(p) == h, str(p)
    rs = [json.loads(s) for s in INPUT.read_text().splitlines() if s]
    assert len(rs)==360 and [r['original_global_index'] for r in rs]==list(range(584,944))
    frames = [rs[i:i+8] for i in range(0,360,8)]
    assert len(frames)==45 and [f[0]['frame_index'] for f in frames]==list(range(73,118))
    for f in frames:
        assert [r['native_slot'] for r in f]==list(range(8))
        assert all(r['split']=='SOURCE_VALIDATION' and r['new_owner_annotation'] is None and not r['training_admission'] for r in f)
        for k in ('frame_id','source_id','frame_index','raw_hw','canonical_RGB','decoded_RGB_sha256','RGB96_packet','original_context_box_xyxy'):
            assert all(r[k]==f[0][k] for r in f)
    assert sum(not r['query_available'] for r in rs)==108
    assert sum(r['query_available'] and not r['local_central4_available'] for r in rs)==4
    assert sum(not r['feature_constructed'] for r in rs)==24
    return rs, frames

def context_operator(binding):
    p = checked(binding); tree = ast.parse(p.read_text())
    names = {'valid_box','target_context_grid','sample_target_context'}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==names
    ns={'np':np}; exec(compile(ast.Module(body=selected,type_ignores=[]),str(p),'exec'),ns)
    return ns

def force_center(r):
    if not r['query_available']: return 'INPUT_U'
    if not r['local_central4_available']: return 'CENTER4_INVALID'
    return None

def validate_packet(packet, f):
    assert set(packet)=={'rgb96','valid96','global_index','crop_constructed'}
    a, m = packet['rgb96'], packet['valid96']
    assert a.dtype==np.uint8 and a.shape==(8,96,96,3)
    assert m.dtype==np.bool_ and m.shape==(8,96,96)
    assert packet['global_index'].dtype==np.int32 and np.array_equal(packet['global_index'],[r['original_global_index'] for r in f])
    assert packet['crop_constructed'].dtype==np.bool_ and np.array_equal(packet['crop_constructed'],[r['feature_constructed'] for r in f])
    for k, r in enumerate(f):
        b=r['original_array_hashes_inherited_not_recomputed']
        assert ash(a[k])==b['RGB96_HWC_sha256'] and ash(a[k].transpose(2,0,1))==b['RGB96_CHW_sha256']
        assert ash(m[k])==b['valid96_sha256']
        assert bool(m[k,47:49,47:49].all())==r['local_central4_available']
        if not r['feature_constructed']: assert not a[k].any() and not m[k].any()
    return a, m

def context(rgb, box, op):
    if op['valid_box'](box): return op['sample_target_context'](rgb,box)
    return np.zeros((224,224,3),np.float32), np.zeros((224,224),bool)

def subsets(a, m):
    return dict(raw96=a, center16=a[40:56,40:56], center2=a[47:49,47:49], valid96=m.astype(np.uint8)*255, center2_valid=m[47:49,47:49].astype(np.uint8)*255)

def display_invalid(a, m):
    out=a.copy(); yy,xx=np.indices(m.shape); color=np.zeros_like(a)
    color[((xx//4+yy//4)%2)==0]=[240,0,240];color[((xx//4+yy//4)%2)==1]=[70,0,70]
    out[~m]=color[~m];return out

def png(p, a):
    Image.fromarray(a).save(p)
    with Image.open(p) as im: assert np.array_equal(np.asarray(im),a)
    return bind(p)

def sheet(rgb, t, vt, f, crops, masks, p):
    im=Image.new('RGB',(SHEET_HW[1],SHEET_HW[0]),'#18212b');d=ImageDraw.Draw(im)
    d.text((24,15),f"HELD INPUT ONLY | {f[0]['frame_id']} | original slots0..7",font=font(28),fill='white')
    d.text((24,55),'No scores / no native visibility labels / no GT or pose reference. Magenta=invalid, never background evidence.',font=font(20),fill='#ffec9f')
    full=Image.fromarray(rgb).resize((1280,720),NEAREST);fd=ImageDraw.Draw(full)
    sy=720/rgb.shape[0];sx=1280/rgb.shape[1]
    box=f[0]['original_context_box_xyxy']
    if box is not None:
        b=np.asarray(box,float)
        if b.shape==(4,) and np.isfinite(b).all():fd.rectangle(tuple(b*np.array([sx,sy,sx,sy])),outline='#ffb300',width=2)
    for k,r in enumerate(f):
        q=np.asarray(r['query_xy'],float)
        if q.shape==(2,) and np.isfinite(q).all() and not np.array_equal(q,[-1,-1]):
            x,y=q*[sx,sy]
            if 0<=x<1280 and 0<=y<720:
                fd.ellipse((x-5,y-5,x+5,y+5),outline=COLORS[k],width=3);fd.text((x+7,y-12),str(k),font=font(21),fill=COLORS[k],stroke_width=1,stroke_fill='black')
    im.paste(full,(24,100));d.text((24,825),'Full canonical RGB + unchanged original q8; orange box=input q8 extent, NOT GT instance annotation.',font=font(20),fill='white')
    td=np.floor(np.clip(t,0,255)+.5).astype(np.uint8)
    im.paste(Image.fromarray(td).resize((448,448),NEAREST),(1340,140));im.paste(Image.fromarray(vt.astype(np.uint8)*255).convert('RGB').resize((448,448),NEAREST),(1820,140))
    d.text((1340,100),'Exact original context224 (2x)',font=font(22),fill='white');d.text((1820,100),'Context validmask (white=valid)',font=font(22),fill='white')
    for y,txt in enumerate(['Context comes from initial q8, not GT or selected-instance mask.','Target identity unclear? Keep UNKNOWN; context does not certify it.','Center owner: original96[47:49,47:49], ALL four samples.','Crop presence: any selected target in valid96, separately.','InputU / center invalid => center owner UNKNOWN.','Black raw padding is not natural background.']):d.text((1340,620+y*31),txt,font=font(21),fill='#c7e0f5')
    for k,r in enumerate(f):
        x=24+(k%2)*1190;y=890+(k//2)*380
        forced=force_center(r);state='CENTER REVIEWABLE' if forced is None else 'CENTER UNKNOWN: '+forced
        q=r['query_xy'];qtext='null' if any(z is None for z in q) else ','.join(f'{z:.3f}' for z in q)
        d.text((x,y),f"slot{k} | g{r['original_global_index']} | {state}",font=font(23),fill='#ffe18b' if forced else 'white')
        d.text((x,y+31),f"q=[{qtext}] L={r['L']} | crop_constructed={r['feature_constructed']} | center valid={int(masks[k,47:49,47:49].sum())}/4",font=font(17),fill='#c7e0f5')
        a=crops[k];m=masks[k]
        for xx,title in ((x,'clean raw96 (3x)'),(x+310,'clean center16 (12x)'),(x+520,'clean center2 (48x)'),(x+630,'invalid display'),(x+860,'valid96 + footprint')):d.text((xx,y+58),title,font=font(17),fill='white')
        im.paste(Image.fromarray(a).resize((288,288),NEAREST),(x,y+84));im.paste(Image.fromarray(a[40:56,40:56]).resize((192,192),NEAREST),(x+310,y+84));im.paste(Image.fromarray(a[47:49,47:49]).resize((96,96),NEAREST),(x+520,y+84))
        im.paste(Image.fromarray(m[47:49,47:49].astype(np.uint8)*255).convert('RGB').resize((96,96),NEAREST),(x+520,y+200))
        overlay=Image.fromarray(display_invalid(a,m)).resize((216,216),NEAREST);od=ImageDraw.Draw(overlay);od.rectangle((47*2.25-2,47*2.25-2,49*2.25+1,49*2.25+1),outline='#00ffff',width=2);im.paste(overlay,(x+630,y+84))
        vm=Image.fromarray(m.astype(np.uint8)*255).convert('RGB').resize((216,216),NEAREST);vd=ImageDraw.Draw(vm);vd.rectangle((47*2.25-2,47*2.25-2,49*2.25+1,49*2.25+1),outline='#00ffff',width=2);im.paste(vm,(x+860,y+84))
        d.text((x+310,y+294),'Center16: exact40:56 subset.',font=font(16),fill='#c7e0f5');d.text((x+310,y+318),'Cyan footprint only on copies/mask.',font=font(16),fill='#c7e0f5');d.text((x+520,y+309),'2x2 valid',font=font(16),fill='#c7e0f5')
    im.save(p);return bind(p)

def invented(op):
    tests=[]
    def ok(name, cond): assert cond,name;tests.append(dict(name=name,PASS=True))
    a=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3);m=np.ones((96,96),bool);s=subsets(a,m)
    ok('exact_center16_and_center2_subsets',np.array_equal(s['center16'],a[40:56,40:56]) and np.array_equal(s['center2'],a[47:49,47:49]))
    ok('center2_is_middle2_of_center16',np.array_equal(s['center2'],s['center16'][7:9,7:9]))
    z=np.array(Image.fromarray(s['center2']).resize((96,96),NEAREST));ok('nearest_center2_preserves_each_sample48x48',all(np.array_equal(z[y*48:(y+1)*48,x*48:(x+1)*48],np.broadcast_to(s['center2'][y,x],(48,48,3))) for y in range(2) for x in range(2)))
    invalid=m.copy();invalid[47,48]=False;display=display_invalid(a,invalid);ok('invalid_display_only_changes_invalid_and_never_raw',np.array_equal(display[invalid],a[invalid]) and not np.array_equal(display[47,48],a[47,48]) and np.array_equal(a,s['raw96']))
    ok('inputU_precedes_center_support',force_center(dict(query_available=False,local_central4_available=True))=='INPUT_U')
    ok('invalid_center_forced_unknown',force_center(dict(query_available=True,local_central4_available=False))=='CENTER4_INVALID')
    ok('available_valid_center_reviewable',force_center(dict(query_available=True,local_central4_available=True)) is None)
    ok('hash_dtype_and_order_specific',ash(a)!=ash(a.transpose(2,0,1)) and ash(m)!=ash(m.astype(np.uint8)*255))
    fake=[dict(original_global_index=k,feature_constructed=True,local_central4_available=True,original_array_hashes_inherited_not_recomputed=dict(RGB96_HWC_sha256=ash(a),RGB96_CHW_sha256=ash(a.transpose(2,0,1)),valid96_sha256=ash(m))) for k in range(8)]
    packet=dict(rgb96=np.stack([a]*8),valid96=np.stack([m]*8),global_index=np.arange(8,dtype=np.int32),crop_constructed=np.ones(8,bool))
    aa,mm=validate_packet(packet,fake);ok('actual_packet_validator_accepts_exact_invented8',np.array_equal(aa[0],a) and np.array_equal(mm[0],m))
    corrupted={k:v.copy() for k,v in packet.items()};corrupted['rgb96'][4,47,48,0]^=1
    try:validate_packet(corrupted,fake);rejected=False
    except AssertionError:rejected=True
    ok('actual_packet_validator_rejects_one_center_pixel_change',rejected)
    reordered={k:v.copy() for k,v in packet.items()};reordered['global_index'][[0,1]]=reordered['global_index'][[1,0]]
    try:validate_packet(reordered,fake);rejected=False
    except AssertionError:rejected=True
    ok('actual_packet_validator_rejects_swapped_identity',rejected)
    rgb=np.arange(9*11*3,dtype=np.uint8).reshape(9,11,3);box=[1.25,2.5,8.75,7.5];out,v=op['sample_target_context'](rgb,box,(2,3));g=op['target_context_grid'](box,(2,3));manual=np.zeros_like(out,dtype=float)
    for y in range(2):
        for x in range(3):
            xx,yy=g[y,x];ix=int(np.floor(xx));iy=int(np.floor(yy));dx=xx-ix;dy=yy-iy
            for ox,oy,w in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)):manual[y,x]+=rgb[iy+oy,ix+ox].astype(float)*w
    ok('V107_context_exact_manual_four_support_bilinear',np.array_equal(out,manual.astype(np.float32)) and v.all())
    out,v=op['sample_target_context'](rgb,[-3,-3,4,4],(4,4));ok('context_outside_support_is_invalid',not v.all() and np.isfinite(out).all())
    out,v=context(rgb,None,op);ok('missing_context_zero_placeholder',out.shape==(224,224,3) and out.dtype==np.float32 and not out.any() and not v.any())
    tmp=H/'INVENTED_CENTER2_PNG_ROUNDTRIP.png';png(tmp,s['center2']);ok('invented_only_PNG_exact_roundtrip',np.array_equal(np.asarray(Image.open(tmp)),s['center2']))
    ok('sheet_geometry_8slots_nonoverlap',890+3*380+84+288<=SHEET_HW[0] and 24+1190+860+216<=SHEET_HW[1] and 1280+24<1340)
    ok('caption_widths_fit',font(23).getlength('slot7 | g943 | CENTER UNKNOWN: CENTER4_INVALID')<1160 and font(21).getlength('Target identity unclear? Keep UNKNOWN; context does not certify it.')<1060)
    return tests

def prepare():
    rs,frames=rows();ready=read(READY);opbind=ready['cache_contract']['pinned_context_crop'];op=context_operator(opbind)
    tests=invented(op)
    write(CHECKS,dict(completed=True,PASS=True,invented_only=True,checks=tests,checks_count=len(tests),code=bind(__file__),runtime=dict(executable=sys.executable,numpy=np.__version__),source_pixels_NPZ_headers_bodies_or_hashes_read=0,new_annotations=0,features_scores_weights_models_read_or_called=0))
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),definition=bind(DEFINITION),readiness=bind(READY),manifest=bind(INPUT),context_code=opbind,font=bind(FONT),checks=bind(CHECKS),rows=360,frames=45,original_inputU=108,available_center4_invalid_additional=4,all_rows_and_original_order_retained=True,output=str(OUT),runtime=dict(executable=str(Path(sys.executable).resolve()),numpy=np.__version__),schema='V177_BLIND_INPUT_RENDER_PROTOCOL_v1',raw96='Read only bound original8x96 cache; exact HWC/CHW/valid hashes, int32 globalID and constructed flags. Save clean PNGs and exact center16[40:56]/center2[47:49]. No raw overlay.',context='Read bound canonicalRGB only after entry; verify file+decoded hash. Exact pinned V107 context224+valid hash against all8 original facts. Initial q8 extent only, no GT/pose/reference/instance mask.',display='45 frame sheets2400x2440; fullframe1280x720 with unchanged q8+input box, context224+mask at2x;8 slots2cols4rows. Clean raw96at3x, center16at12x, center2at48x, separate validmask and magenta-invalid copy with cyan footprint.',blindness='No model outputs/logits/features/weights, old native D/E/owner annotations, GT/pose/geometry references loaded or displayed. Inputnative_slot is identity metadata only.',forced_center='Original inputU108=>UNKNOWN; another4 available center-invalid=>UNKNOWN. Other mixed/identity ambiguity reviewerUNKNOWN. No force for crop-presence except unconstructedcropUNKNOWN.',crop_presence='Independent ANY selected target in valid96 only, not named-native or exactpoint truth; does not repair center owner.',review='A/B each view all45 sheets/360rows independently; actual image bindings recorded later. No generated observation values in renderer. Per-field disagreementUNKNOWN; no fit/calibration/modelselection/admission.',historical_exposure='Prediction-blind new assistant-weak owner observations, not pristine blind dataset or human/pixel/native gold.',new_annotations=0,actual_pixel_read_calls=0,automatic_retry=False)
    write(PROTOCOL,protocol)
    template=dict(approved_blinded_public_held_pixel_render=False,code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),definition=bind(DEFINITION),manifest=bind(INPUT),checks=bind(CHECKS),output=str(OUT),model_forward_allowed=False,feature_weight_score_reads_allowed=False,old_native_label_or_GT_reference_reads_allowed=False,private_current_or_V112_allowed=False,fit_or_calibration_or_PnP_or_qL_change_allowed=False,automatic_retry=False)
    write(H/'ROOT_ACTUAL_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    write(H/'PREPARATION_COMPLETE.json',dict(completed=True,prepared_only=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),template=bind(H/'ROOT_ACTUAL_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE.json'),all360_metadata_checked=True,source_pixel_or_feature_score_weight_NPZ_header_body_hash_reads=0,actual_calls=0,new_annotations=0))
    print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),preparation=bind(H/'PREPARATION_COMPLETE.json'))),flush=True)

def run(entry):
    e=read(entry);assert e['approved_blinded_public_held_pixel_render'] is True
    for k in ('model_forward_allowed','feature_weight_score_reads_allowed','old_native_label_or_GT_reference_reads_allowed','private_current_or_V112_allowed','fit_or_calibration_or_PnP_or_qL_change_allowed','automatic_retry'):assert e[k] is False
    p=read(checked(e['protocol']));assert e['protocol']==bind(PROTOCOL) and e['output']==p['output']==str(OUT) and not OUT.exists()
    for k in ('code','scope','definition','manifest','checks'):assert e[k]==p[k];checked(e[k])
    assert p['code']==bind(__file__) and p['runtime']==dict(executable=str(Path(sys.executable).resolve()),numpy=np.__version__)
    checked(p['readiness']);checked(p['font']);op=context_operator(p['context_code']);rs,frames=rows()
    OUT.mkdir();write(OUT/'STARTED.json',dict(completed=False,entry=bind(entry),automatic_retry=False))
    try:
        reviewed=[];frame_records=[];source_bindings={}
        for f in frames:
            r=f[0];fi=r['frame_index'];folder=OUT/f'FRAME_{fi:04d}';folder.mkdir()
            for b in (r['canonical_RGB'],r['RGB96_packet']):checked(b);source_bindings[b['path']]=b
            rgb=np.load(r['canonical_RGB']['path'],allow_pickle=False);assert rgb.dtype==np.uint8 and list(rgb.shape)==r['raw_hw']+[3] and ash(rgb)==r['decoded_RGB_sha256']
            with np.load(r['RGB96_packet']['path'],allow_pickle=False) as z:packet={k:np.array(z[k],copy=True) for k in z.files}
            crops,masks=validate_packet(packet,f);t,vt=context(rgb,r['original_context_box_xyxy'],op)
            for q in f:
                b=q['original_array_hashes_inherited_not_recomputed'];assert ash(t)==b['target224_F32_HWC_sha256'] and ash(vt)==b['target_valid_sha256']
            context_png=png(folder/'CONTEXT224.png',np.floor(np.clip(t,0,255)+.5).astype(np.uint8));context_valid=png(folder/'CONTEXT_VALID224.png',vt.astype(np.uint8)*255)
            case_records=[]
            for k,q in enumerate(f):
                images={name:png(folder/f'SLOT{k}_{name}.png',a) for name,a in subsets(crops[k],masks[k]).items()}
                row=dict(q);row.update(images=images,forced_center_unknown_reason=force_center(q),crop_presence_forced_unknown=not q['feature_constructed'],center4_valid_bits=masks[k,47:49,47:49].tolist(),context224=context_png,context_valid224=context_valid,new_annotations_created=False)
                case_records.append(row)
            sb=sheet(rgb,t,vt,f,crops,masks,folder/'SHEET.png')
            for row in case_records:row['frame_sheet']=sb;reviewed.append(row)
            frame_records.append(dict(frame_index=fi,frame_id=r['frame_id'],source_id=r['source_id'],frame_sheet=sb,context224=context_png,context_valid224=context_valid,cases=case_records))
            print(json.dumps(dict(frames=len(frame_records),rows=len(reviewed))),flush=True)
        assert len(reviewed)==360 and [r['query_id'] for r in reviewed]==[r['query_id'] for r in rs]
        for b in source_bindings.values():checked(b)
        for path,h in EXPECTED.items():assert sha(path)==h
        ledger=OUT/'ALL360_BLINDED_REVIEW_INPUTS.jsonl';ledger.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in reviewed))
        deck=OUT/'REVIEW_INPUT_ALL45_FRAMES.json';write(deck,dict(completed=True,rows=360,frames=frame_records,annotation_values_written=False))
        index=OUT/'index.html';index.write_text('<!doctype html><meta charset="utf-8"><title>V177 blind held inputs</title><style>body{background:#18212b;color:white;font:18px sans-serif}img{width:100%;height:auto}article{margin:30px 0}</style><h1>45 original frames / 360 original slots — inputs only</h1><p>Original inputU and padding preserved; assistant weak observations remain separate from native visibility/coordinate gold. No predictions shown.</p>'+''.join('<article><h2>'+html.escape(x['frame_id'])+'</h2><a href="'+str(Path(x['frame_sheet']['path']).relative_to(OUT))+'">Exact sheet PNG</a><img src="'+str(Path(x['frame_sheet']['path']).relative_to(OUT))+'"></article>' for x in frame_records))
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),rows=360,frames=45,original_inputU=108,center4_invalid=112,available_center4_invalid_additional=4,original96_pixel_hash_rows_verified=360,context224_and_valid_hash_rows_verified=360,source_files_rechecked_after=len(source_bindings),original_metadata_qL_native_split_flags_preserved=True,source_unchanged=True,review_input=bind(deck),all360_rows=bind(ledger),HTML=bind(index),frame_sheets=[x['frame_sheet'] for x in frame_records],actual_visual_review='pending reviewers view tools; renderer makes no observations',HTML_browser_rendered=False,new_annotations=0,model_feature_weight_score_GT_reference_private_current_V112_fit_PnP_calls=0,training_admitted=False,selected_model=None,fully_solved=False))
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    else:run(a.entry)
