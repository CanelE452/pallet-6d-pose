"""One admitted public frame: projected virtual AABB addresses, no labels."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import types
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIGNS = ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
         (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
COLORS = ('#ff3154','#ffa700','#f3dc00','#20d080','#00c4ed','#588bff','#c768ff','#ff6cc5')


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path, exists=True):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute path')
    need(not any(x in str(p).casefold() for x in ('v112','cache','.pyc','current223','publicheld','holdout','oldval')), 'Forbidden before FS')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Exact artifact absent')
            continue
        need(not stat.S_ISLNK(mode), 'Symlink artifact')
    return p


def raw(b):
    fd = os.open(safe(b['path']), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        v = f.read()
    need(len(v)==b['size_bytes'] and hashlib.sha256(v).hexdigest()==b['sha256'], 'SHA/size '+Path(b['path']).name)
    return v


def binding(p):
    v = safe(p).read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(v).hexdigest(), 'size_bytes': len(v)}


def dump(p, value):
    with safe(p, False).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return binding(p)


def marker(draw, x, y, color, size=5):
    draw.line((x-size,y,x+size,y), fill=color, width=2)
    draw.line((x,y-size,x,y+size), fill=color, width=2)


def font(size):
    # Pillow's bundled font, no local file discovery or font-path FS reads here.
    return ImageFont.load_default(size=size)


def observe_mask(mask, pixel, radius):
    x, y = pixel
    h, w = mask.shape
    left, top = max(0,x-radius), max(0,y-radius)
    right, bottom = min(w,x+radius+1), min(h,y+radius+1)
    a = mask[top:bottom,left:right]
    return {'radius': radius, 'bounds_xyxy_halfopen': [left,top,right,bottom],
            'pixels': int(a.size), 'support_pixels': int(a.sum()),
            'support_fraction': float(a.mean()), 'any_support': bool(a.any()),
            'all_support': bool(a.all()), 'clipped': a.size != (2*radius+1)**2}


def run(entry_path):
    cfg = json.loads(safe(entry_path).read_bytes())
    need(cfg.get('approved_actual_public_virtual_query_diagnostic') is True, 'Root bounded actual task required')
    need(cfg['code'] == binding(Path(__file__).absolute()), 'Pinned code')
    root = json.loads(raw(cfg['root_frame_entry']))
    need(root['approved_public_geometry_decode'] is True and root['training_or_model_inference'] is False, 'Admitted public frame only')
    qa = json.loads(raw(root['identity_QA']))
    need(qa['completed'] is True and qa['PASS'] is True and qa['permits_this_public_frame_geometry_diagnostic'] is True,
         'Opaque identity admission')
    need(all(qa['public_RGB_binding'][k] == root['inputs']['RGB'][k] for k in ('path','sha256','size_bytes')), 'Same admitted RGB')
    complete = json.loads(raw(cfg['geometry_complete']))
    need(complete['completed'] is True and complete['PASS'] is True and complete['new_point_labels'] == 0, 'Saved geometry complete')
    render_bindings = [x for x in complete['files'] if Path(x['path']).name == 'T_RENDER_DEPTH_M.npy']
    need(len(render_bindings) == 1, 'Only exact T render depth may be followed')
    render_binding = render_bindings[0]
    need(Path(render_binding['path']).parent == Path(root['output_directory']), 'Same saved geometry output')
    module = types.ModuleType('pinned_public_geometry_v2')
    module.__file__ = root['projection_code']['path']
    exec(compile(raw(root['projection_code']), module.__file__, 'exec'), module.__dict__)
    bs = {k: root['inputs'][k] for k in ('RGB','SAM','POSE','K','OBJ')}
    # Depth PNG, inverse-T arrays, old/protected identity pointers remain inert.
    rgb_bytes, mask_bytes = raw(bs['RGB']), raw(bs['SAM'])
    rgb = np.array(Image.open(io.BytesIO(rgb_bytes)).convert('RGB'), dtype=np.uint8)
    mask_u8 = np.array(Image.open(io.BytesIO(mask_bytes)))
    need(rgb.ndim == 3 and rgb.shape[2] == 3 and mask_u8.shape == rgb.shape[:2], 'Same full-frame RGB/mask size')
    need(mask_u8.dtype == np.uint8 and np.isin(mask_u8, [0,255]).all(), 'Publisher binary SAM values, no threshold fitting')
    mask = mask_u8 == 255
    K = np.loadtxt(io.BytesIO(raw(bs['K'])), dtype=np.float64)
    T = np.loadtxt(io.BytesIO(raw(bs['POSE'])), dtype=np.float64)
    K, T, pose_validation = module.validate_K_T(K,T)
    vertices, faces, mesh_report = module.parse_obj_text(raw(bs['OBJ']))
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    corners = np.asarray([[low[j] if sign[j] < 0 else high[j] for j in range(3)] for sign in SIGNS], np.float64)
    camera = corners@T[:3,:3].T + T[:3,3]
    uv, front = module.project_points(camera,K)
    render = np.load(io.BytesIO(raw(render_binding)), allow_pickle=False)
    need(render.dtype == np.float64 and render.shape == rgb.shape[:2], 'Saved full T render F64 geometry')
    h,w = mask.shape
    rows = []
    for k in range(8):
        continuous = uv[k].tolist() if front[k] else None
        pixel = [int(math.floor(float(v)+.5)) for v in uv[k]] if front[k] else None
        inside = pixel is not None and 0 <= pixel[0] < w and 0 <= pixel[1] < h
        presence = bool(mask[pixel[1],pixel[0]]) if inside else None
        support = bool(np.isfinite(render[pixel[1],pixel[0]]) and render[pixel[1],pixel[0]] > 0) if inside else None
        depth = float(render[pixel[1],pixel[0]]) if support else None
        rows.append({'index':k,'query_name':f'AUX_BBOX{k}','sign_order':list(SIGNS[k]),
                     'object_xyz':corners[k].tolist(),'camera_xyz_m':camera[k].tolist(),
                     'projected_query_xy':continuous,'query_source':'Exact T-projected public OBJ AABB virtual address; not initializer/native mapping',
                     'front_positive_Z':bool(front[k]),'rounded_pixel_floor_q_plus_half':pixel,'rounded_in_frame':inside,
                     'SAM_exact_pixel_presence':presence,'SAM_window3':observe_mask(mask,pixel,1) if inside else None,
                     'SAM_window7':observe_mask(mask,pixel,3) if inside else None,
                     'target_T_render_support':support,'target_T_render_depth_m':depth,
                     'virtual_corner_z_minus_render_m':float(camera[k,2]-depth) if depth is not None else None,
                     'native_mapping':None,'D_E_SELF_label':None,'training_admission':False})
    out = safe(cfg['output_directory'],False)
    need(not out.exists(), 'Fresh output, no automatic retry')
    out.mkdir();(out/'query_crops').mkdir()
    try:
        full = Image.new('RGB',(w,h+118),'white');full.paste(Image.fromarray(rgb),(0,62))
        d = ImageDraw.Draw(full)
        d.text((12,9),'Public object: virtual AABB query addresses AUX_BBOX0..7',fill='#111',font=font(18))
        d.text((12,34),'T unchanged; no snapping, native mapping, D/E/SELF or training labels',fill='#333',font=font(14))
        for a,b in EDGES:
            if front[a] and front[b]:
                d.line((uv[a,0],uv[a,1]+62,uv[b,0],uv[b,1]+62),fill='#ffffff',width=1)
        for r in rows:
            if r['front_positive_Z']:
                x,y=r['projected_query_xy'];marker(d,x,y+62,COLORS[r['index']],5)
                d.text((x+6,y+62+2),str(r['index']),fill=COLORS[r['index']],stroke_width=1,stroke_fill='black',font=font(16))
        d.text((12,h+70),'White lines = virtual cuboid. Colored crosses = exact projected addresses.',fill='#222',font=font(13))
        d.text((12,h+91),'Object/mask presence near a query does not establish semantic native-corner gold.',fill='#222',font=font(13))
        full.save(out/'FULL_RGB_VIRTUAL_QUERIES.png')
        overview=Image.new('RGB',(4*256,2*318+76),'#ededed');od=ImageDraw.Draw(overview)
        od.text((14,9),'All eight virtual queries; fixed96x96-source-pixel windows, display nearest x2',fill='#111',font=font(18))
        od.text((14,35),'Sorted index0..7; gray is out-of-frame padding. Marker remains continuous original q.',fill='#333',font=font(15))
        for r in rows:
            k=r['index'];card=Image.new('RGB',(256,318),'white');cd=ImageDraw.Draw(card)
            cd.text((10,8),r['query_name'],fill=COLORS[k],stroke_width=0,font=font(20))
            if r['front_positive_Z']:
                x,y=r['projected_query_xy'];left,top=math.floor(x)-48,math.floor(y)-48
                window=Image.new('RGB',(96,96),'#888888')
                sx0,sy0,sx1,sy1=max(0,left),max(0,top),min(w,left+96),min(h,top+96)
                if sx1>sx0 and sy1>sy0:
                    window.paste(Image.fromarray(rgb).crop((sx0,sy0,sx1,sy1)),(sx0-left,sy0-top))
                window=window.resize((192,192),Image.Resampling.NEAREST)
                wd=ImageDraw.Draw(window)
                marker(wd,2*(x-left+.5)-.5,2*(y-top+.5)-.5,COLORS[k],6)
                card.paste(window,(32,38))
                r['display_crop_original_integer_bounds_xyxy']=[left,top,left+96,top+96]
                r['display_crop_note']='Display crop integer anchor only; stored projected q unchanged. Nearest2x pixel-center display transform.'
            else:
                cd.rectangle((32,38,224,230),fill='#888888');cd.text((52,120),'Behind camera',fill='black',font=font(15))
                r['display_crop_original_integer_bounds_xyxy']=None
            if r['rounded_in_frame']:
                cd.text((10,238),f"Exact SAM {int(r['SAM_exact_pixel_presence'])} / render {int(r['target_T_render_support'])}",fill='#111',font=font(15))
                cd.text((10,261),f"3x3 SAM {r['SAM_window3']['support_pixels']}/{r['SAM_window3']['pixels']}",fill='#333',font=font(15))
                cd.text((10,284),f"7x7 SAM {r['SAM_window7']['support_pixels']}/{r['SAM_window7']['pixels']}",fill='#333',font=font(15))
            else:
                cd.text((10,243),'Rounded query not in-frame',fill='#333',font=font(15))
            card.save(out/f'query_crops/AUX_BBOX{k}.png')
            overview.paste(card,((k%4)*256,76+(k//4)*318))
        overview.save(out/'ALL8_FIXED_SCALE_QUERY_CROPS.png')
        counts={'all_virtual_queries':len(rows),'positive_Z':sum(r['front_positive_Z'] for r in rows),
                'rounded_in_frame':sum(r['rounded_in_frame'] for r in rows),
                'SAM_exact_present':sum(r['SAM_exact_pixel_presence'] is True for r in rows),
                'SAM_exact_absent_in_frame':sum(r['SAM_exact_pixel_presence'] is False for r in rows),
                'SAM3_any_support':sum(r['SAM_window3'] is not None and r['SAM_window3']['any_support'] for r in rows),
                'SAM7_any_support':sum(r['SAM_window7'] is not None and r['SAM_window7']['any_support'] for r in rows),
                'target_render_support':sum(r['target_T_render_support'] is True for r in rows),
                'SAM_exact_absent_but_local7_support':sum(r['SAM_exact_pixel_presence'] is False and r['SAM_window7']['any_support'] for r in rows),
                'D_E_SELF_native_labels_created':0}
        summary={'completed':True,'PASS':True,'frame':root['capture_frame'],'counts':counts,
                 'AABB_min':low.tolist(),'AABB_max':high.tolist(),'mesh_scale':1,'pose':'T unchanged, not inverse-selected',
                 'identity_limits':qa['limitations'],'pixel_center':'integer column,row; rounded floor(q+.5)',
                 'source_RGB_shape':list(rgb.shape),'source_RGB_decoded_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                 'raw_depth_png_or_inverse_render_or_other_geometry_outputs_read':0,
                 'native_mapping_certified':False,'virtual_query_is_initializer_prediction':False,
                 'negative_masks_create_E_SELF_or_native_gold':False,'no_search_or_snap_or_coordinate_correction':True,
                 'whole_target_visible_is_visual_observation_not_semantic_corner_gold':True,'training_admission':False,
                 'inputs':bs,'identity_QA':root['identity_QA'],'geometry_complete':cfg['geometry_complete'],
                 'saved_T_render_depth':render_binding,'projection_code':root['projection_code'],
                 'mesh_parser':mesh_report,'pose_validation':pose_validation}
        dump(out/'ALL8_VIRTUAL_QUERY_OBSERVATIONS.json',rows)
        dump(out/'SUMMARY.json',summary)
        flat=[]
        for r in rows:
            flat.append({'index':r['index'],'name':r['query_name'],'q_x':r['projected_query_xy'][0] if r['front_positive_Z'] else None,
                         'q_y':r['projected_query_xy'][1] if r['front_positive_Z'] else None,'in_frame':r['rounded_in_frame'],
                         'SAM_exact':r['SAM_exact_pixel_presence'],'SAM3_support':r['SAM_window3']['support_pixels'] if r['SAM_window3'] else None,
                         'SAM3_pixels':r['SAM_window3']['pixels'] if r['SAM_window3'] else None,
                         'SAM7_support':r['SAM_window7']['support_pixels'] if r['SAM_window7'] else None,
                         'SAM7_pixels':r['SAM_window7']['pixels'] if r['SAM_window7'] else None,'render_support':r['target_T_render_support'],
                         'render_depth_m':r['target_T_render_depth_m'],'virtual_corner_z_m':r['camera_xyz_m'][2],
                         'native_mapping':None,'D_E_SELF_label':None})
        with (out/'ALL8_VIRTUAL_QUERY_OBSERVATIONS.csv').open('x',newline='') as f:
            wri=csv.DictWriter(f,fieldnames=list(flat[0]));wri.writeheader();wri.writerows(flat)
        md=f'''# 공개 Pallox 한 frame의 가상 AABB query와 pixel support

선택된 공개 `0.55m_hall/0000`의 mesh AABB8개를 고정 sign 순서로 구성하고 원 T로 투영했습니다. 이름은 AUX_BBOX0..7이며 우리 pallet native 주소와 연결하지 않습니다. q는 정확한 가상주소 투영값으로, initializer/v61 예측이 아닙니다. snapping·좌표 보정·새 학습/추론은 없습니다.

모든8개를 순서대로 보존했습니다. 양의Z {counts['positive_Z']}, rounded in-frame {counts['rounded_in_frame']}, SAM exact pixel 있음 {counts['SAM_exact_present']}, 없음 {counts['SAM_exact_absent_in_frame']}, 3×3 중 SAM지원 {counts['SAM3_any_support']}, 7×7 중 지원 {counts['SAM7_any_support']}, target render exact support {counts['target_render_support']}입니다. exact SAM없지만7×7에 지원이 있는 주소는 {counts['SAM_exact_absent_but_local7_support']}개입니다. 이것은 이 frame/mask/render의 관측이며 negative mask를 E/SELF로 라벨링하거나 모델 정확도를 계산하지 않습니다.

![전체 RGB와 가상주소](FULL_RGB_VIRTUAL_QUERIES.png)

![모든8 query 고정scale crop](ALL8_FIXED_SCALE_QUERY_CROPS.png)

crop은 source96×96pixel 창을 nearest2배로 표시한 주석화 자료입니다. 원 q는 그대로이며 crop용 정수 anchor는 query 보정이 아닙니다. 회색은 frame 밖 padding입니다. 3×3/7×7은 rounded floor(q+.5)를 중심으로 frame에 clip한 실제 window 크기를 분모로 기록합니다. 전체 object가 보인다는 것은 시각 관측이며 semantic corner/native gold 인증이 아닙니다. 3D physical-native support는 원 query presence 판단의 필수조건으로 부과하지 않습니다.

원 mesh AABB는 외접 cuboid이므로 address에 실제 surface/target pixel이 존재한다는 보장이 없습니다. 주변 구조가 보이는지와 정확한 virtual pixel presence를 구분하기 위한 단일 공개 AUX 진단이며, native-point 정의·SELF·visibility gold를 만들지 않습니다. T의 옳음이나 mask 완전성을 이 표만으로 인증하지 않습니다. encoded-byte identity no-hit은 reencoding/공통capture 비중복 증명이 아니며 학습 admission도 아닙니다.

[모든8 observations CSV](ALL8_VIRTUAL_QUERY_OBSERVATIONS.csv) · [full rows JSON](ALL8_VIRTUAL_QUERY_OBSERVATIONS.json) · [요약·입력binding](SUMMARY.json) · [완료 영수증](COMPLETE.json)
'''
        with (out/'README_KO.md').open('x') as f:f.write(md)
        html='<!doctype html><html lang="ko"><meta charset="utf-8"><title>Public virtual-query AUX diagnostic</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;line-height:1.7}img{max-width:100%}</style><h1>Public AABB virtual addresses; not native gold</h1><p>All8, T unchanged, no snapping/training/D-E-SELF labels. q is a projected virtual cuboid address, not initializer output.</p><img src="FULL_RGB_VIRTUAL_QUERIES.png"><img src="ALL8_FIXED_SCALE_QUERY_CROPS.png"><p><a href="README_KO.md">한국어 한계·정의</a> · <a href="ALL8_VIRTUAL_QUERY_OBSERVATIONS.csv">All8 CSV</a> · <a href="SUMMARY.json">Bindings/summary</a></p></html>\n'
        with (out/'PORTABLE_GALLERY_KO.html').open('x') as f:f.write(html)
        paths=[out/x for x in ('ALL8_VIRTUAL_QUERY_OBSERVATIONS.json','ALL8_VIRTUAL_QUERY_OBSERVATIONS.csv','SUMMARY.json','README_KO.md','PORTABLE_GALLERY_KO.html','FULL_RGB_VIRTUAL_QUERIES.png','ALL8_FIXED_SCALE_QUERY_CROPS.png')]
        paths += [out/f'query_crops/AUX_BBOX{k}.png' for k in range(8)]
        receipt=dump(out/'COMPLETE.json',{'completed':True,'PASS':True,'code':cfg['code'],'entry':binding(entry_path),
                     'root_frame_entry':cfg['root_frame_entry'],'geometry_complete':cfg['geometry_complete'],
                     'counts':counts,'files':[binding(p) for p in paths], 'new_training_or_model_forward':0,
                     'native_or_D_E_SELF_labels_created':0,'training_admission':False,'current_or_held_access':False})
        print(json.dumps({'completed':True,'PASS':True,'COMPLETE':receipt,'counts':counts},sort_keys=True))
    except BaseException as e:
        dump(out/'FAILURE.json',{'completed':False,'PASS':False,'error':repr(e),'automatic_retry':False})
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True)
    run(p.parse_args().entry)
