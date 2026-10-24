"""V215 factual SOURCE TRAIN previews; metadata prep never opens source binaries."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

H = Path(__file__).absolute().parent
V = H.parent
P = V.parent
OUT = V / 'fixed12_public_TRAIN_input_previews_v1'
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
SCOPE = {'path': str(V / 'ROOT_PROSPECTIVE_PUBLIC_SOURCE_TRAIN_COVERAGE_PILOT_SCOPE.json'), 'sha256': '417b08aa3bb0474aba448f2c25cd9099c1bf559a159e93538e3988e0e64a1383'}
SELECTOR_READY = {'path': str(V / 'selector_preparation_b/FINAL_READY.json'), 'sha256': 'f052d073fc49d2aa5f1129c22eb818ba08a72b888d11cea2bb7ffccf9112e714'}
SELECTION = {'path': str(V / 'selector_preparation_b/SELECTED12_SOURCE_TRAIN_FRAMES_BEFORE_PIXELS.json'), 'sha256': '9f067a6a707bde8d5a350ffff24ef5b9542f883e1994abd9e15d404bcb7cc224'}
PROTO = H / 'PROTOCOL_BEFORE_ANY_SELECTED_SOURCE_RGB_OR_FRAME8_BODIES.json'
PREFIXES = ('val/000000', 'val/000002', 'val/000005', 'val/000006')
COLORS = ((0,255,255), (255,80,80), (50,255,50), (255,160,0), (255,80,255), (255,255,0), (70,150,255), (240,240,240))


def need(value, message):
    if not bool(value):
        raise AssertionError(message)


def lexical(path):
    value = str(path)
    need('v112' not in value.casefold(), 'Forbidden V112 name before filesystem access')
    p = Path(value)
    need(p.is_absolute() and '..' not in p.parts and (P in p.parents or p == FONT), 'Exact allowed absolute path')
    return p


def parent_fd(path):
    p = lexical(path)
    fd = os.open(p.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
    except BaseException:
        os.close(fd)
        raise
    return p, fd


def opened(path, write=False):
    p, fd = parent_fd(path)
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL if write else os.O_RDONLY
        out = os.open(p.name, flags | os.O_NOFOLLOW, 0o644, dir_fd=fd)
    finally:
        os.close(fd)
    need(stat.S_ISREG(os.fstat(out).st_mode), 'Regular file, no symlink')
    return os.fdopen(out, 'wb' if write else 'rb')


def mkdir_new(path):
    p, fd = parent_fd(path)
    try:
        os.mkdir(p.name, dir_fd=fd)
    finally:
        os.close(fd)


def rb(path):
    with opened(path) as f:
        return f.read()


def bind(path):
    return {'path': str(lexical(path)), 'sha256': hashlib.sha256(rb(path)).hexdigest()}


def checked(binding):
    need(set(binding) == {'path', 'sha256'} and bind(binding['path']) == binding, 'Exact binding SHA')
    return binding['path']


def read(binding):
    return json.loads(rb(checked(binding)))


def write(path, value):
    with opened(path, True) as f:
        f.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def text(path, value):
    with opened(path, True) as f:
        f.write(value.encode())


def ash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def font(size):
    with opened(FONT) as f:
        return ImageFont.truetype(f, size)


def save_png(path, im):
    with opened(path, True) as f:
        im.save(f, format='PNG')


def png_array(path):
    with opened(path) as f:
        with Image.open(f) as im:
            return np.array(im)


def valid_q(q):
    return isinstance(q, list) and len(q) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in q)


def validate_frame(frame):
    rr = frame['native8']
    need(frame['role'] == 'TRAIN' and len(rr) == 8 and [r['native_slot'] for r in rr] == list(range(8)), 'Immutable TRAIN whole original native8 in order')
    for k, r in enumerate(rr):
        need(r['role'] == 'TRAIN' and r['raw_row'] == k and r['source_id'] == frame['source_id'] and r['component_id'] == frame['component_id'] and r['raw_packet'] == rr[0]['raw_packet'], 'Native cache/META frame identity')
        need(type(r['original_query_available']) is bool, 'Original availability boolean')
        if r['original_query_available']:
            need(valid_q(r['query_xy']) and isinstance(r['L'], (int, float)) and math.isfinite(r['L']) and r['L'] > 0, 'Available original q/L finite')
    need(frame['original_available_count'] == sum(r['original_query_available'] for r in rr), 'Original availability count, no U imputation')
    need(frame['image']['format'] == 'NPY_RGB_UINT8_HWC', 'Existing original canonicalRGB source format')
    lexical(frame['image']['binding']['path'])
    lexical(rr[0]['raw_packet']['path'])


def validate_selection(plan):
    need(plan['completed'] and plan['PASS'] and plan['selection_frozen_before_pixels'] and not plan['new_training_admission'], 'Accepted immutable prospective selection')
    ff = plan['frames']
    need(len(ff) == 12 and len({f['source_id'] for f in ff}) == 12 and len({f['component_id'] for f in ff}) == 12 and Counter(f['prefix'] for f in ff) == {p: 3 for p in PREFIXES}, 'Exactly prior4x3 selection / distinct components')
    for f in ff:
        validate_frame(f)
    need(sum(not r['original_query_available'] for f in ff for r in f['native8']) == 11, 'Original96 slots and U11 retained')
    binary = [b for f in ff for b in (f['image']['binding'], f['native8'][0]['raw_packet'])]
    need(binary == plan['future_payload_whitelist'] and len({b['path'] for b in binary}) == 24, 'Only selected12 RGB and12 frame8 bindings')
    return ff, binary


def payload_bytes(binding, whitelist):
    # Membership gate precedes all payload stat/hash/open/header/decode operations.
    need(binding in whitelist, 'Payload not in exact root-approved24 whitelist')
    data = rb(binding['path'])
    need(hashlib.sha256(data).hexdigest() == binding['sha256'], 'Original container SHA before decoding')
    return data


def validate_cache(frame, raw):
    need(raw['rgb96'].dtype == np.uint8 and raw['rgb96'].shape == (8,96,96,3), 'Original raw RGB96 schema')
    need(raw['valid96'].dtype == np.bool_ and raw['valid96'].shape == (8,96,96), 'Original valid96 schema')
    need(raw['crop_constructed'].dtype == np.bool_ and raw['crop_constructed'].shape == (8,), 'Original constructed flags')
    need(raw['global_index'].shape == (8,) and np.issubdtype(raw['global_index'].dtype, np.integer), 'Original global-index8')
    for k, r in enumerate(frame['native8']):
        need(int(raw['global_index'][k]) == r['original_global_index'] and bool(raw['crop_constructed'][k]) == r['original_flags']['feature_constructed'], 'Original frame8 META global/constructed parity')
        need(ash(raw['rgb96'][k]) == r['RGB96_sha256'] and ash(raw['valid96'][k]) == r['valid96_sha256'], 'Original per-native raw RGB/valid decoded SHA')


def input_overlay(rgb, frame):
    im = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(im)
    for r in frame['native8']:
        q = r['query_xy']
        if not valid_q(q) or not (0 <= q[0] < rgb.shape[1] and 0 <= q[1] < rgb.shape[0]):
            continue
        x, y = q
        color = COLORS[r['native_slot']]
        draw.line((x-8,y,x+8,y), fill=color, width=2)
        draw.line((x,y-8,x,y+8), fill=color, width=2)
        draw.text((x+7,y-22), str(r['native_slot']) + ('' if r['original_query_available'] else ' U'), fill=color, font=font(20))
    return im


def contact(rgb, frame, raw):
    width, image_height, top, tile_h = 2304, 700, 100, 430
    y0 = top + image_height + 38
    im = Image.new('RGB', (width, y0 + tile_h + 70), 'white')
    d = ImageDraw.Draw(im)
    d.text((18,12), 'V215 SOURCE TRAIN | ' + frame['source_id'] + ' | native0..7 / original q,L,U', font=font(28), fill='black')
    d.text((18,54), 'Factual inputs only. Original q8 markers are DISPLAY ONLY; no GT, teacher, old labels or model scores.', font=font(22), fill='black')
    display = input_overlay(rgb, frame)
    scale = min(width / display.width, image_height / display.height)
    display = display.resize((int(display.width * scale), int(display.height * scale)), Image.Resampling.LANCZOS)
    im.paste(display, ((width-display.width)//2, top + (image_height-display.height)//2))
    for k, r in enumerate(frame['native8']):
        x = 288*k
        d.line((x,y0,x,y0+tile_h), fill='#aaaaaa', width=1)
        d.text((x+7,y0+4), 'native %d | %s' % (k, 'AVAILABLE' if r['original_query_available'] else 'INPUT U'), font=font(21), fill='black')
        q = r['query_xy']
        d.text((x+7,y0+36), 'qx=' + str(None if q is None else q[0]), font=font(17), fill='black')
        d.text((x+7,y0+60), 'qy=' + str(None if q is None else q[1]), font=font(17), fill='black')
        d.text((x+7,y0+84), 'L=' + str(r['L']), font=font(17), fill='black')
        clean = Image.fromarray(raw['rgb96'][k])
        # All resize/markers stay on display copies. The clean96 ROI remains exact96 pixels.
        zoom = clean.resize((176,176), Image.Resampling.NEAREST)
        zd = ImageDraw.Draw(zoom)
        center = (47.5+.5)*176/96-.5
        zd.line((center-7,center,center+7,center), fill='cyan', width=1)
        zd.line((center,center-7,center,center+7), fill='cyan', width=1)
        im.paste(zoom, (x+6,y0+126))
        im.paste(clean, (x+184,y0+126))
        im.paste(Image.fromarray(raw['valid96'][k].astype(np.uint8)*255).convert('RGB'), (x+184,y0+252))
        d.text((x+184,y0+226), 'clean96', font=font(16), fill='black')
        d.text((x+184,y0+352), 'valid96', font=font(16), fill='black')
        d.text((x+7,y0+380), 'constructed=' + str(bool(raw['crop_constructed'][k])), font=font(17), fill='black')
    d.text((18,y0+tile_h+8), 'Cyan crop cross = original sample center47.5 DISPLAY ONLY. Black valid = padding/invalid. Off-frame/U slots are retained.', font=font(21), fill='black')
    return im


def invented_tests():
    raw = dict(rgb96=np.arange(8*96*96*3,dtype=np.uint32).reshape(8,96,96,3).astype(np.uint8), valid96=np.ones((8,96,96),bool), crop_constructed=np.array([True]*7+[False]), global_index=np.arange(8))
    raw['valid96'][7] = False
    frame = dict(source_id='INVENTED_ONLY', component_id='I', role='TRAIN', original_available_count=7, image={'format':'NPY_RGB_UINT8_HWC','binding':{'path':str(H/'UNOPENED_INVENTED_RGB.npy'),'sha256':'0'*64}}, native8=[])
    for k in range(8):
        frame['native8'].append(dict(role='TRAIN', source_id='INVENTED_ONLY', component_id='I', native_slot=k, raw_row=k, query_xy=[10.25,20.5] if k<7 else None, L=220.89751434326172 if k<7 else None, original_query_available=k<7, original_global_index=k, raw_packet={'path':str(H/'UNOPENED_INVENTED_FRAME.npz'),'sha256':'0'*64}, original_flags={'feature_constructed':k<7}, RGB96_sha256=ash(raw['rgb96'][k]), valid96_sha256=ash(raw['valid96'][k])))
    validate_frame(frame)
    validate_cache(frame, raw)
    bad = dict(raw, global_index=np.arange(8)[::-1])
    try:
        validate_cache(frame, bad)
        rejects = False
    except AssertionError:
        rejects = True
    need(rejects and frame['native8'][7]['query_xy'] is None and not frame['native8'][7]['original_query_available'], 'Mixed null/U preserved and wrong META rejected')
    rgb = np.full((144,256,3),127,np.uint8)
    before = raw['rgb96'].copy()
    im = contact(rgb, frame, raw)
    a = np.asarray(im)
    y = 100+700+38+126
    need(im.width == 2304 and np.array_equal(before,raw['rgb96']) and all(np.array_equal(a[y:y+96,k*288+184:k*288+280],raw['rgb96'][k]) for k in range(8)), 'Contact all8 clean96 exact / input unchanged / legible layout')
    line = 'qy=' + str(706.8020629882812)
    need(ImageDraw.Draw(im).textbbox((0,0),line,font=font(17))[2] < 278, 'Long F64 query text fits native column')
    with tempfile.TemporaryDirectory(dir=H, prefix='invented_nofollow_only_') as t:
        target = Path(t)/'target.txt'
        text(target,'invented fixture only')
        link = Path(t)/'link.txt'
        os.symlink('target.txt',link)
        try:
            rb(link)
            rejects = False
        except OSError:
            rejects = True
        need(rejects,'NOFOLLOW rejects actual invented symlink')
    try:
        lexical(H/'v112_before_stat')
        rejects = False
    except AssertionError:
        rejects = True
    need(rejects,'Forbidden lexical name before FS')
    return dict(completed=True,PASS=True,invented_only=True,count=3,checks=['Shape_META_and_mixed_null_U_preservation_reject_reordered_global8','Contact2304_all8_exactclean96_F64text_input_unchanged','Real_invented_symlink_NOFOLLOW_and_lexical_forbidden_precheck'],source_binary_stat_hash_header_decode=0), im


def prepare():
    scope, ready, plan = read(SCOPE), read(SELECTOR_READY), read(SELECTION)
    need(scope['approved_code_metadata_preparation_only'] and not scope['approved_actual_RGB_array_or_annotation'] and ready['completed'] and ready['PASS'] and ready['selection']==SELECTION, 'Exact source prep scope / frozen selector')
    frames, binary = validate_selection(plan)
    checks, invented = invented_tests()
    write(H/'INVENTED_PREVIEW_CHECKS.json',checks)
    save_png(H/'INVENTED_LAYOUT_ONLY.png',invented)
    protocol = dict(completed=True,preparation_only=True,code=bind(__file__),scope=SCOPE,selector_ready=SELECTOR_READY,selection=SELECTION,checks=bind(H/'INVENTED_PREVIEW_CHECKS.json'),binary_input_whitelist=binary,output=str(OUT),original_sampler='No resampling. Read existing per-frame rgb96/valid96/crop_constructed/global_index only; metadata q/L/native/U from frozen selection.',fullRGB='Selected12 canonical RGB NPY only, original containerSHA and decodedRGB SHA. Original encodedJPEG provenance is metadata only and is never opened.',display='2304px contact: original full-image display overlay plus native0..7 across one row with F64 q/L/U text, clean96 and valid96 tiles. q8 marker and crop47.5 cyan cross affect display copies only. Full unmarked RGB and each clean96/valid96 are separate byte-parity PNGs.',authority='Factual source inputs / observations pending. No GT/oldweak/teacher/classifier/SAM/labels/targetbbox fallback. Original unavailable/offframe rows kept, not E-imputed. No training admission.',actual_source_binary_stat_hash_header_or_decode=0,actual_requires_new_root_entry=True,automatic_retry=False)
    write(PROTO,protocol)
    write(H/'ROOT_ACTUAL_FIXED12_PREVIEW_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_selected12_SOURCE_TRAIN_factual_preview=False,code=protocol['code'],protocol=bind(PROTO),scope=SCOPE,selector_ready=SELECTOR_READY,selection=SELECTION,binary_input_whitelist=binary,models_labels_scores_GT_or_protected_allowed=False,automatic_retry=False))
    write(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=bind(PROTO),checks=protocol['checks'],selection=SELECTION,false_entry=bind(H/'ROOT_ACTUAL_FIXED12_PREVIEW_ENTRY_TEMPLATE_APPROVED_FALSE.json'),invented_layout=bind(H/'INVENTED_LAYOUT_ONLY.png'),selected_frames=12,native_rows=96,original_U=11,actual_source_binary_stat_hash_header_or_decode=0,training_admission=False,frozen_no_further_edits=True))
    print(json.dumps(dict(completed=True,PASS=True,ready=bind(H/'FINAL_READY.json'),code=protocol['code'],protocol=bind(PROTO)),ensure_ascii=False))


def run(entry_path):
    e = json.loads(rb(entry_path))
    need(e['approved_selected12_SOURCE_TRAIN_factual_preview'] is True and e['models_labels_scores_GT_or_protected_allowed'] is False and e['automatic_retry'] is False,'Root actual entry approval before payload access')
    p = read(e['protocol'])
    need(e['code']==p['code']==bind(__file__) and e['protocol']==bind(PROTO) and e['scope']==p['scope']==SCOPE and e['selector_ready']==p['selector_ready']==SELECTOR_READY and e['selection']==p['selection']==SELECTION and e['binary_input_whitelist']==p['binary_input_whitelist'],'Exact frozen entry/code/protocol/selection/24 whitelist')
    read(SCOPE)
    ready, plan = read(SELECTOR_READY), read(SELECTION)
    need(ready['completed'] and ready['PASS'] and ready['selection']==SELECTION,'Frozen selector closure')
    frames,binary = validate_selection(plan)
    need(binary==p['binary_input_whitelist'],'Immutable metadata selection before all24 bodies')
    mkdir_new(OUT)
    try:
        records, slots = [], []
        for pos,frame in enumerate(frames):
            # Every source container hash is checked before its decoder sees bytes.
            rgb = np.load(io.BytesIO(payload_bytes(frame['image']['binding'],binary)),allow_pickle=False)
            need(rgb.dtype==np.uint8 and rgb.shape==tuple(frame['image']['raw_hw'])+(3,) and ash(rgb)==frame['image']['decoded_RGB_sha256'],'Canonical fullRGB exact shape/dtype/decoded SHA')
            with np.load(io.BytesIO(payload_bytes(frame['native8'][0]['raw_packet'],binary)),allow_pickle=False) as z:
                raw = {k:z[k] for k in ('rgb96','valid96','crop_constructed','global_index')}
            validate_cache(frame,raw)
            folder=OUT/('FRAME_%02d_%04d'%(pos,frame['original_frame_index']))
            mkdir_new(folder)
            original=folder/'ORIGINAL_CANONICAL_RGB.png'
            overlay=folder/'ORIGINAL_Q8_DISPLAY_ONLY.png'
            card=folder/'FULLIMAGE_AND_ALL8_INPUT_CONTACT.png'
            save_png(original,Image.fromarray(rgb))
            save_png(overlay,input_overlay(rgb,frame))
            save_png(card,contact(rgb,frame,raw))
            need(np.array_equal(png_array(original),rgb),'Unmarked fullRGB PNG decoded byte identity')
            records.append(dict(position=pos,frame_key=frame['source_id'],original_frame_index=frame['original_frame_index'],prefix=frame['prefix'],component_id=frame['component_id'],role='TRAIN',original_fullRGB=bind(original),display_original_q8=bind(overlay),all8contact=bind(card),source_RGB=frame['image'],original_encoded_RGB_provenance_INERT=frame['original_encoded_RGB_provenance'],original_qL_native_U_preserved=True,observations='PENDING',fit_admission=False))
            for k,r in enumerate(frame['native8']):
                clean=folder/('NATIVE_%d_CLEAN96.png'%k)
                valid=folder/('NATIVE_%d_VALID96.png'%k)
                save_png(clean,Image.fromarray(raw['rgb96'][k]))
                save_png(valid,Image.fromarray(raw['valid96'][k].astype(np.uint8)*255))
                need(np.array_equal(png_array(clean),raw['rgb96'][k]) and np.array_equal(png_array(valid),raw['valid96'][k].astype(np.uint8)*255),'Every raw96/valid96 PNG decoded byte parity')
                slots.append(dict(position=pos,frame_key=frame['source_id'],original_frame_index=frame['original_frame_index'],component_id=frame['component_id'],role='TRAIN',native_slot=k,row_id=r['row_id'],original_global_index=r['original_global_index'],query_xy=r['query_xy'],L=r['L'],original_query_available=r['original_query_available'],original_flags=r['original_flags'],crop_constructed=bool(raw['crop_constructed'][k]),clean96=bind(clean),valid96=bind(valid),clean96_decoded_RGB_sha256=ash(raw['rgb96'][k]),valid96_decoded_bool_sha256=ash(raw['valid96'][k]),original_raw_byte_identity=True,observation='UNREVIEWED',fit_admission=False))
        need(len(records)==12 and len(slots)==96 and sum(not r['original_query_available']for r in slots)==11,'All12/96/U11, no missing native or invalid slot')
        write(OUT/'ANNOTATOR_INPUT_MANIFEST.json',dict(completed=True,PASS=True,selection=SELECTION,frames=records,slots96=slots,original_qL_native_U_preserved=True,no_GT_labels_teacher_SAM_or_model_scores=True,observations_pending=True,actual_toolview='NOT_CLAIMED'))
        fields=['position','frame_key','original_frame_index','native_slot','row_id','query_x','query_y','L','original_query_available','role','component_id','clean96','valid96']
        s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in slots:
            q=r['query_xy'];w.writerow({**{k:r[k]for k in fields if k in r and k not in('clean96','valid96')},'query_x':None if q is None else q[0],'query_y':None if q is None else q[1],'clean96':str(Path(r['clean96']['path']).relative_to(OUT)),'valid96':str(Path(r['valid96']['path']).relative_to(OUT))})
        text(OUT/'ALL96_ORIGINAL_INPUT_IDENTITIES.csv',s.getvalue())
        doc=['# V215 공개 SOURCE TRAIN 입력12프레임','','고정4prefix×3프레임·12component의 원입력96slot이다. 원 U11도 유지했다. 이 자료는 관찰 전 factual 입력이며 GT·모델·기존 약라벨·SAM을 표시하지 않는다. q8 표식과 crop47.5 십자는 표시용이며 수정 keypoint나 native 정답위치가 아니다. fullRGB와 원96crop/valid의 별도 PNG는 원바이트와 일치한다.','','[원96slot ID/qL/U CSV](ALL96_ORIGINAL_INPUT_IDENTITIES.csv) · [PNG 정합 manifest](ANNOTATOR_INPUT_MANIFEST.json)','','약관찰/학습 admission은 아직 없다. 이후 intended visiblepart와 q와의 관계를 분리하고 UNKNOWN·불일치를 보존한다. component 구분은 물리적 독립성을 인증하지 않는다.','']
        body='<h1>V215 SOURCE TRAIN factual inputs</h1><p>12 fixed frames / 96 native slots / original U11. DISPLAY ONLY query markers. No labels, scores, GT, SAM or new sampling.</p>'
        for r in records:
            doc += ['## '+r['frame_key'],'',f"![고정 fullimage 및 native8]({Path(r['all8contact']['path']).relative_to(OUT).as_posix()})",'',f"[표식 없는 원 fullRGB]({Path(r['original_fullRGB']['path']).relative_to(OUT).as_posix()}) · [원q8 표시 사본]({Path(r['display_original_q8']['path']).relative_to(OUT).as_posix()})",'']
            body+='<h2>'+html.escape(r['frame_key'])+'</h2><img style="max-width:100%;height:auto" src="'+Path(r['all8contact']['path']).relative_to(OUT).as_posix()+'">'
        text(OUT/'PORTABLE_GALLERY_KO.md','\n'.join(doc))
        text(OUT/'index.html','<!doctype html><meta charset="utf-8">'+body)
        for b in binary:
            payload_bytes(b,binary)  # End-of-run opaque source hash recheck, no extra decoding.
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTO),entry=bind(entry_path),scope=SCOPE,selector_ready=SELECTOR_READY,selection=SELECTION,manifest=bind(OUT/'ANNOTATOR_INPUT_MANIFEST.json'),CSV=bind(OUT/'ALL96_ORIGINAL_INPUT_IDENTITIES.csv'),gallery=bind(OUT/'PORTABLE_GALLERY_KO.md'),HTML=bind(OUT/'index.html'),frames=12,native_rows=96,original_inputU=11,distinct_components=12,prefixes={x:3 for x in PREFIXES},source_payloads_exact24=True,NPZ_members_decoded=['rgb96','valid96','crop_constructed','global_index'],all_original_fullRGB_RAW96_valid96_decoded_PNG_bytes_exact=True,models_labels_scores_GT_SAM_or_protected_reads=0,new_qL_sampling_or_labels_or_fit_admission=False,observations_pending=True,actual_toolviews='NOT_CLAIMED',automatic_retry=False))
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json')),ensure_ascii=False))
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry');args=ap.parse_args()
    prepare() if args.prepare else run(args.entry)
