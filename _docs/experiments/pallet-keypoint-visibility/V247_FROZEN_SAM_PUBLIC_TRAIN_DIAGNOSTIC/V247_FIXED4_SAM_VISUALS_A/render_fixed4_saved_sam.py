"""Fixed saved SOURCE output display only; no model or raw-source loader."""
from pathlib import Path
from io import BytesIO
import argparse, csv, hashlib, html, json, os, stat, sys, traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION

sys.dont_write_bytecode = True
R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V247_FIXED4_SAM_VISUALS_A'
SCOPE = {'path': str(R / 'V247_ROOT_PUBLIC_TRAIN_SAM/TRUE_FIXED4_SAVED_SAM_VISUALIZATION_SCOPE.json'),
         'sha256': '5b891964e3c0a2987b468b9a79a94bdb21f799f921a9d7a4fb92f079b7a1af89', 'size_bytes': 19115}


def guard(path):
    p = Path(path)
    if any(t in str(p).casefold() for t in ('v112', '__pycache__', '.pyc', 'current', 'held', 'private', 'holdout', 'oldval')):
        raise ValueError('Forbidden before filesystem')
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError('Absolute traversal-free path required')
    for a in reversed((p, *p.parents)):
        try: mode = os.lstat(a).st_mode
        except FileNotFoundError: continue
        if stat.S_ISLNK(mode): raise ValueError('Symlink forbidden')
    return p


def sha(raw): return hashlib.sha256(raw).hexdigest()


def identity(b): return str(b['path']), b['sha256']


def checked(b):
    raw = guard(b['path']).read_bytes()
    assert sha(raw) == b['sha256'], 'Input SHA mismatch'
    for key in ('bytes', 'size_bytes'):
        if key in b: assert b[key] == len(raw)
    return raw


def read(b): return json.loads(checked(b))


def bind(path):
    p = guard(path); raw = p.read_bytes()
    return {'path': str(p), 'sha256': sha(raw), 'size_bytes': len(raw)}


def write_json(path, obj):
    p = guard(path)
    with p.open('x') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return bind(p)


def font(size):
    return ImageFont.truetype(str(guard('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')), size)


def annotated(rgb, q8, box):
    image = Image.fromarray(rgb.copy()); d = ImageDraw.Draw(image)
    if box is not None:
        d.rectangle(tuple(map(float, box)), outline=(255, 213, 0), width=3)
    for k, q in enumerate(q8):
        if np.isfinite(q).all():
            x, y = map(float, q)
            d.line((x-6, y, x+6, y), fill=(0, 255, 255), width=2)
            d.line((x, y-6, x, y+6), fill=(0, 255, 255), width=2)
            d.text((x+5, y+3), str(k), font=font(17), fill=(0, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return image


def support(x): return 'U' if x is None else 'MASK+' if x else 'OUTSIDE'


def main(entry_path):
    e = read(bind(guard(entry_path)))
    assert e['approved'] is True and identity(e['root_scope']) == identity(SCOPE)
    assert identity(e['code']) == identity(bind(__file__))
    s = read(e['root_scope']); assert s['approved'] is True
    assert s['renderer_agent_authorized'] == '/root/owner_witness_a_v158'
    out = guard(e['output_directory'])
    assert str(out) == s['output_directory'] == str(H / 'gallery_v1') and not out.exists()
    frozen = read(s['prediction_freeze']); assert frozen['completed'] and frozen['PASS'] and frozen['model_state_byte_unchanged']
    old = read(s['V246_gallery_complete']); assert old['completed'] and old['PASS']
    fixed = read(s['fixed_selection']); assert fixed['only_SOURCE_TRAIN']
    frames = s['frames']; assert len(frames) == 4
    assert [x['original_frame_input'] for x in frames] == fixed['frames']
    allrows = [json.loads(x) for x in checked(s['all352_original_query_and_readout_rows']).splitlines() if x]
    assert len(allrows) == 352
    ids = {x['original_input_unchanged']['query_id'] for x in allrows}; assert len(ids) == 352
    fixed_ids = {x['original_frame_input']['frame_input_id'] for x in frames}
    rows = [x for x in allrows if x['original_input_unchanged']['frame_input_id'] in fixed_ids]
    assert len(rows) == 32
    byframe = {f: sorted([x for x in rows if x['original_input_unchanged']['frame_input_id'] == f], key=lambda x: x['original_input_unchanged']['native_slot']) for f in fixed_ids}
    frozen_byframe = {x['original_frame_input']['frame_input_id']: x for x in frozen['frame_outputs']}
    out.mkdir()
    outputs, case_data, cards = {}, [], []
    for i, f in enumerate(frames):
        original = f['original_frame_input']; fid = original['frame_input_id']; rr = byframe[fid]
        assert original['split'] == 'TRAIN_LAB' and [x['original_input_unchanged']['native_slot'] for x in rr] == list(range(8))
        assert identity(f['SAM_output']) == identity(frozen_byframe[fid]['output'])
        assert identity(f['source_RGB_copy']) == identity(old['outputs'][f'FRAME_{i:02d}_SOURCE_RGB.png'])
        source = checked(f['source_RGB_copy'])
        assert sha(source) == original['image_sha256']  # compare inherited encoded identity, never open original image path
        rgb_image = Image.open(BytesIO(source)); assert rgb_image.mode == 'RGB' and rgb_image.size == (640,480)
        rgb = np.asarray(rgb_image).copy()
        source_name = f'FRAME_{i:02d}_SOURCE_RGB.png'; (out/source_name).write_bytes(source); outputs[source_name] = bind(out/source_name)
        with np.load(BytesIO(checked(f['SAM_output'])), allow_pickle=False) as z:
            chosen=z['chosen_logit']; quality=z['quality_scores']; selected=int(z['selected_index'])
            q8=z['original_q8']; L8=z['original_L']; native=z['native_slot']; box=z['original_prompt_box']; hw=z['raw_hw']; qids=z['query_id']; exact=z['exact_logit']; near=z['radius2_logit']
        assert chosen.dtype == np.float32 and chosen.shape == (480,640) and np.isfinite(chosen).all()
        assert quality.shape == (3,) and quality.dtype == np.float32 and np.isfinite(quality).all() and selected == int(np.argmax(quality))
        assert q8.shape == (8,2) and L8.shape == (8,) and np.array_equal(native,np.arange(8)) and np.array_equal(hw,[480,640])
        assert np.array_equal(box,[q8[:,0].min(),q8[:,1].min(),q8[:,0].max(),q8[:,1].max()])
        mask = chosen > 0
        overlay = rgb.astype(np.float64); overlay[mask] = .62*overlay[mask]+.38*np.array([38,220,105]); overlay=np.floor(overlay+.5).astype(np.uint8)
        card=Image.new('RGB',(1600,1050),(247,249,251));d=ImageDraw.Draw(card)
        d.text((36,20), fid, font=font(27), fill=(12,25,43))
        d.text((36,62), 'Fixed SOURCE TRAIN case | frozen saved SAM2.1 | native/part gold UNKNOWN', font=font(20), fill=(40,56,72))
        d.text((36,94), f'Selected mask {selected}; predicted qualities '+', '.join(f'{x:.5f}' for x in quality)+' (model outputs, not confidence truth)', font=font(17), fill=(40,56,72))
        d.text((50,127),'Original copied RGB + unchanged q8 box',font=font(20),fill=(12,25,43))
        d.text((910,127),'Green: chosen logit > 0 proposal',font=font(20),fill=(12,25,43))
        card.paste(annotated(rgb,q8,box),(50,158));card.paste(annotated(overlay,q8,box),(910,158))
        d.text((36,653),'Cyan: original q0..7. Yellow: exact q8 min/max. No teacher/reference/GT box, no query correction.',font=font(18),fill=(40,56,72))
        d.text((36,681),'Unmarked CLEAN96 files below are byte copies; exact and near5 mask support are separate observations.',font=font(18),fill=(40,56,72))
        record={'frame_position':i,'original_frame_input':original,'saved_SAM':f['SAM_output'],'original_box':box.tolist(),'selected_index':selected,'quality3':quality.tolist(),'queries':[]}
        for k,row in enumerate(rr):
            q=row['original_input_unchanged'];assert q['query_id'] == str(qids[k]) and q['query_xy'] == q8[k].tolist() and q['roi_side'] == float(L8[k])
            assert row['selected_index'] == selected and row['quality3'] == quality.tolist() and row['box_xyxy'] == box.tolist()
            assert (row['exact_logit'] is None and np.isnan(exact[k])) or float(exact[k]) == row['exact_logit']
            assert (row['radius2_logit'] is None and np.isnan(near[k])) or float(near[k]) == row['radius2_logit']
            assert row['native_visibility_label'] is None and row['semantic_SELF'] is None
            clean_binding=f['CLEAN96'][str(k)];assert identity(clean_binding) == identity(old['outputs'][f'FRAME_{i:02d}_N{k}_CLEAN96.png'])
            clean_raw=checked(clean_binding);name=f'FRAME_{i:02d}_N{k}_CLEAN96.png';(out/name).write_bytes(clean_raw);outputs[name]=bind(out/name);assert outputs[name]['sha256']==clean_binding['sha256']
            clean=Image.open(BytesIO(clean_raw));assert clean.mode=='RGB' and clean.size==(96,96)
            x=48+k*188;d.text((x,716),f'Native slot {k}',font=font(19),fill=(12,25,43));card.paste(clean.resize((144,144),Image.Resampling.NEAREST),(x,745))
            d.text((x,899),'exact: '+support(row['exact_mask_support']),font=font(16),fill=(12,25,43))
            d.text((x,923),'near5: '+support(row['radius2_mask_support']),font=font(16),fill=(12,25,43))
            d.text((x,947),f'L={q["roi_side"]:.3f}',font=font(15),fill=(40,56,72))
            if row['U_reason']:d.text((x,969),'U: '+row['U_reason'][:18],font=font(12),fill=(120,55,30))
            record['queries'].append({'original_input_unchanged':q,'state':row['state'],'prediction_available':row['prediction_available'],'exact_mask_support':row['exact_mask_support'],'radius2_mask_support':row['radius2_mask_support'],'exact_logit':row['exact_logit'],'radius2_logit':row['radius2_logit'],'U_reason':row['U_reason'],'CLEAN96':outputs[name]})
        d.text((36,1009),'MASK+ / OUTSIDE are segmentation support only. OUTSIDE is not E / SELF / absence; 4 fixed cases are not population evidence.',font=font(17),fill=(115,42,32))
        name=f'FRAME_{i:02d}_CARD.png';card.save(out/name);outputs[name]=bind(out/name);cards.append(card);record['card']=outputs[name];record['source_RGB_copy']=outputs[source_name];case_data.append(record)
    overview=Image.new('RGB',(1600,1050),'white')
    for i,card in enumerate(cards):overview.paste(card.resize((800,525),Image.Resampling.LANCZOS),((i%2)*800,(i//2)*525))
    overview.save(out/'FIXED4_OVERVIEW.png');outputs['FIXED4_OVERVIEW.png']=bind(out/'FIXED4_OVERVIEW.png')
    outputs['CASE_DATA.json']=write_json(out/'CASE_DATA.json',{'completed':True,'frames':case_data,'original_fixed_cases':4,'queries':32,'no_native_labels':True,'training_admission':False})
    with (out/'QUERY32.csv').open('x',newline='') as stream:
        fields=['frame_position','query_id','native_slot','q_x','q_y','L','state','exact_mask_support','radius2_mask_support','exact_logit','radius2_logit','U_reason'];w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for frame in case_data:
            for row in frame['queries']:
                q=row['original_input_unchanged'];xy=q['query_xy'];w.writerow({'frame_position':frame['frame_position'],'query_id':q['query_id'],'native_slot':q['native_slot'],'q_x':None if xy is None else xy[0],'q_y':None if xy is None else xy[1],'L':q['roi_side'],**{k:row[k]for k in fields if k in row}})
    outputs['QUERY32.csv']=bind(out/'QUERY32.csv')
    page='<!doctype html><meta charset="utf-8"><title>고정 SOURCE SAM 4사례</title><style>body{font:18px sans-serif;max-width:1600px;margin:30px auto;line-height:1.6}img{width:100%;height:auto}a{color:#07589c}</style><h1>고정 공개 TRAIN SAM 관측 4사례</h1><p>원 q8 min/max box와 원 q/L을 유지한 frozen SAM2.1 저장 출력입니다. 초록은 선택 mask의 logit&gt;0입니다. MASK+·OUTSIDE는 proposal 지원 관측이며 native D/E·SELF·물리 부위 정답이 아닙니다. 모델 quality는 정답 확률이 아닙니다. 원 교사와의 instance 대응을 이 그림이 보증하지 않습니다.</p><p>같은 사전 고정 두 쌍·네 프레임·32 query만 사용했습니다. 원인·빈도·통계 일반화나 모델 선택 근거로 쓰지 않습니다. clean96 PNG와 SOURCE PNG는 기존 파일의 바이트 복사이며 표시 마커는 카드에만 있습니다. GT·교사 certificate·reference·모델 가중치는 읽지 않았고 새 inference·학습은 없습니다.</p><p><a href="CASE_DATA.json">사례 메타데이터</a> · <a href="QUERY32.csv">32 query 표</a> · <a href="FIXED4_OVERVIEW.png">개요</a></p>'
    for i,frame in enumerate(case_data):page+=f'<h2>{html.escape(frame["original_frame_input"]["frame_input_id"])}</h2><a href="FRAME_{i:02d}_CARD.png"><img src="FRAME_{i:02d}_CARD.png"></a><p>'+ ' '.join(f'<a href="FRAME_{i:02d}_N{k}_CLEAN96.png">원 crop {k}</a>'for k in range(8))+'</p>'
    (out/'PORTABLE_GALLERY_KO.html').write_text(page+'<p>Root의 실제 시각 QA는 별도 기록됩니다.</p>');outputs['PORTABLE_GALLERY_KO.html']=bind(out/'PORTABLE_GALLERY_KO.html')
    manifest=write_json(out/'MANIFEST.json',{'completed':True,'inputs_scope':e['root_scope'],'fixed_selection':s['fixed_selection'],'prediction_freeze':s['prediction_freeze'],'all352_readout_metadata':s['all352_original_query_and_readout_rows'],'input_frames':frames,'outputs':outputs})
    complete=write_json(out/'COMPLETE.json',{'completed':True,'PASS':True,'entry':bind(entry_path),'code':bind(__file__),'root_scope':e['root_scope'],'prediction_freeze':s['prediction_freeze'],'fixed_selection':s['fixed_selection'],'manifest':manifest,'outputs':outputs,'frames':4,'queries':32,'original_RGB_copy_count':4,'clean96_bytecopy_count':32,'saved_SAM_NPZ_count':4,'all_row_metadata_read':352,'selected_rows_used':32,'model_runtime_weight_teacher_current_held_or_extra_original_RGB_reads':0,'new_model_forward_or_training':0,'native_labels_created':0,'root_visual_QA_pending':True,'actually_tool_viewed_by_renderer':False,'runtime':{'numpy':np.__version__,'Pillow':PIL_VERSION},'automatic_retry':False})
    print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete,'overview':outputs['FIXED4_OVERVIEW.png'],'cards':[x['card']for x in case_data]}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);args=ap.parse_args()
    try: main(args.entry)
    except Exception:
        p=guard(H/'FAILURE.txt')
        if not p.exists():p.write_text(traceback.format_exc())
        raise
