"""Root-authorized C48/C49 saved-output display only; reuse V175 clean32 crops.

No source crop NPZ, source RGB, mask, feature or checkpoint is opened. Selection,
q/L, native, original semantic and point-owner soft mass are inherited unchanged.
"""
import hashlib
import html
import json
import platform
import runpy
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image,ImageDraw

sys.dont_write_bytecode=True
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V175=P/'query_visibility_direct_query_owner_control_20261024_v175'
V=P/'query_visibility_direct_owner_public_ycb_control_20261024_v176'
OUT=V/'fixed32_source_crop_review_v1'
N48='C48_PUBLIC_WEAK253_PLUS_LITERAL_SYNTHETIC'
N49='C49_LITERAL_SYNTHETIC_PLUS_PUBLIC_YCB_CF314'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def bind(p):return {'path':str(Path(p)),'sha256':sha(p)}
def load(p):return json.loads(Path(p).read_text())
def verify(b):
    p=Path(b['path']).resolve()
    assert p.is_relative_to(P) and not any(x in str(p).lower() for x in ['_v112','_v169','current223','private'])
    assert sha(p)==b['sha256'];return p


def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    assert not OUT.exists(),'preserve prior attempt; no automatic retry'
    assert Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/envs/pallet-yolo26/bin/python').resolve()
    assert platform.python_version()=='3.10.20' and np.__version__=='1.26.4' and PIL.__version__=='12.2.0'
    original=V175/'fixed32_source_crop_review_preparation/render_fixed32_source_crops.py'
    assert sha(original)=='779008f6613f38d46bc504dc89faac4bc0e30f5e345a01c31fabf91901e0a709'
    m=runpy.run_path(str(original))  # Pure plotting/validation functions only, no __main__ call.
    cp48=V175/'runs/C47_C48_DIRECT_OWNER_FIXED8000/COMPLETE.json'
    cp49=V/'runs/C49_PUBLIC_YCB_CF314_FIXED8000/COMPLETE.json'
    assert sha(cp48)=='4d2cdc4cd83bd422769c9bf39818460310e2c22907db9f0381f09cf9c37b8cb5'
    assert sha(cp49)=='1f530b0522192f32d1eb5d91caaf8c7367ba146d700de6c18e31800e2ad7eb35'
    c48,c49=load(cp48),load(cp49)
    assert c48['completed'] and c48['PASS'] and c48['all2_frozen_before_outputs']
    assert c49['completed'] and c49['PASS'] and c49['fixed_final_frozen_before_outputs']
    assert c49['baseline_C48_complete']==bind(cp48)
    assert c49['code']['sha256']=='77248b48771c942a8a13891744a1032bef6dd024604307308262aa8a30e5a161'
    assert c49['protocol']['sha256']=='c419a28f0bde78b590b3df6b63df34f5c0d7f0edaa5984c1717386289fea61f3'
    e49=load(verify(c49['entry']));f49=load(verify(c49['freeze']));f48=load(verify(c48['all2_freeze']))
    assert e49['approved_C49_source_fit'] and e49['source_only']
    assert f49['completed'] and f49['update']==8000 and not f49['saved_outputs_emitted']
    assert f48['completed'] and not f48['saved_outputs_emitted']
    parent=V175/'fixed32_source_crop_review_v2/COMPLETE.json'
    assert sha(parent)=='9b1bc2df902129288a6932b9ad1e2e477a53eafc0769825c78ef865fa79ba683'
    old=load(parent);inp=load(verify(old['fixed32_inputs']))
    saved=load(verify(old['saved_values_and_pixel_proofs']))['rows']
    assert len(inp['rows'])==len(saved)==32
    assert [r['original_global_index'] for r in inp['rows']]==[r['original_global_index'] for r in saved]
    assert inp['scope']['sha256']=='7ea5d7ca846b3f3f285028813b6c2cffeceb4044bf6e48ba455db0637fc49b91'
    output_bindings={'C48':c48['arms'][N48]['outputs']['OLD6144'],'C49':c49['arms'][N49]['outputs']['OLD6144']}
    packets={}
    for name,b in output_bindings.items():
        with np.load(verify(b),allow_pickle=False) as z:
            assert 'features' not in z.files
            o={k:z[k].copy() for k in z.files}
        assert o['owner_class_order'].tolist()==m['CLASSES']
        m['validate_predictions'](o['raw_owner_three_logits'],o['owner_prediction'],o['query_available'])
        o['display_probabilities']=m['diagnostic_probabilities'](o['raw_owner_three_logits'])
        for r in inp['rows']:m['validate_output_identity'](o,r['original_global_index'],r)
        packets[name]=o
    for r,v in zip(inp['rows'],saved):
        i=r['original_global_index']
        assert np.array_equal(packets['C48']['raw_owner_three_logits'][i],np.asarray(v['saved_outputs'][N48]['raw_logits'],np.float32))
        assert int(packets['C48']['owner_prediction'][i])==v['saved_outputs'][N48]['prediction']
    OUT.mkdir();(OUT/'clean96').mkdir();(OUT/'central32').mkdir()
    rows=[]
    for r,v in zip(inp['rows'],saved):
        i=r['original_global_index'];clean=verify(v['clean96']);central=verify(v['central32'])
        rgb=np.asarray(Image.open(clean)).copy()
        assert rgb.dtype==np.uint8 and rgb.shape==(96,96,3) and m['ash'](rgb)==v['rgb96_array_sha256']==r['rgb96_sha256']
        assert np.array_equal(np.asarray(Image.open(central)),rgb[32:64,32:64])
        dc=OUT/'clean96'/f'{i:06d}.png';ds=OUT/'central32'/f'{i:06d}.png'
        shutil.copyfile(clean,dc);shutil.copyfile(central,ds)
        assert sha(dc)==sha(clean) and sha(ds)==sha(central)
        r['_rgb']=rgb
        rows.append({'original_global_index':i,'original_occurrence_id':r['original_occurrence_id'],
             'original_semantic_state':r['original_semantic_state'],'point_owner_T_F_BG_unknown_mass':r['point_owner_T_F_BG_unknown_mass'],
             'native_slot':r['native_slot'],'query_xy':r['query_xy'],'L':r['L'],'split':r['split'],
             'original_input_flags':r['original_input_flags'],'clean96':bind(dc),'central32':bind(ds),
             'rgb96_array_sha256':m['ash'](rgb),'inherited_valid96_array_sha256':r['valid96_sha256'],
             'models':{name:{'raw_FP32_logits':o['raw_owner_three_logits'][i].tolist(),
                            'display_float64_probabilities':o['display_probabilities'][i].tolist(),
                            'saved_prediction':int(o['owner_prediction'][i])} for name,o in packets.items()}})
    sheets=[];maxwidth=0
    for base in [128,129,144,145]:
        br=[r for r in inp['rows'] if r['base_index']==base];assert len(br)==8
        W,H,cw,ch=1720,1260,420,530;im=Image.new('RGB',(W,H),'#f4f6f8');d=ImageDraw.Draw(im);font=m['font'];fac=br[0]['factors']
        d.text((20,14),f"V176 fixed VAL base{base} | {fac['material']} / {fac['aspect']}",fill='#101828',font=font(25))
        d.text((20,51),'Pre-C47 fixed source VAL examples | C48 vs C49 | not representative performance | original q/L RGB96',fill='#344054',font=font(19))
        for k,r in enumerate(br):
            i=r['original_global_index'];x=20+(k%4)*cw;y=90+(k//4)*ch
            d.rectangle((x,y,x+cw-10,y+ch-10),fill='white',outline='#cbd5e1');im.paste(m['marked_copy'](r['_rgb']),(x+60,y+10))
            mass=r['point_owner_T_F_BG_unknown_mass'];truth='UNKNOWN' if mass is None else '/'.join(f'{v:.3f}' for v in mass)
            text=[f"#{i} native{r['native_slot']} {r['variant']}",f"q=({r['query_xy'][0]:.3f},{r['query_xy'][1]:.3f}) L={r['L']:.3f}",f"Declared native semantic: {r['original_semantic_state']}",f'Point T/F/BG/U: {truth}']
            for name in ['C48','C49']:
                o=packets[name];p=o['display_probabilities'][i];j=int(o['owner_prediction'][i]);call='U' if j==-1 else m['SHORT'][j]
                ps='U' if not np.isfinite(p).all() else '/'.join(f'{a:.3f}' for a in p)
                text.append(f'{name}: {call} | T/F/BG {ps}')
            text.append(f"Original inputU={not r['original_input_flags']['query_available']} | q at raw(47.5,47.5)")
            for n,line in enumerate(text):
                width=d.textlength(line,font=font(16));maxwidth=max(maxwidth,width);assert width<=390
                d.text((x+10,y+306+n*27),line,fill='#111827',font=font(16))
        footer=['Original semantic D/E/S/U differs from literal point owner; FOREIGN does not imply native external occlusion E.',
                'C48: public weak253 + synthetic TRAIN4016; C49 adds controlled MR/YCB CF314 owner TRAIN loss.',
                'T=incident+other target; U=ambiguous+invalid. No truth cutoff, new labels/admission, model selection or physical-native certification.']
        for n,line in enumerate(footer):d.text((20,1160+n*27),line,fill='#344054',font=font(18))
        path=OUT/f'BASE_{base:03d}_8_FIXED_CROPS.png';im.save(path);sheets.append(bind(path))
    write(OUT/'FIXED32_SAVED_VALUES_AND_PIXEL_PARITY.json',{'rows':rows,'source_scope':inp['scope'],'output_bindings':output_bindings,'no_labels_changed':True})
    body=''.join(f'<section><h2>base {base}</h2><a href="{Path(b["path"]).name}"><img src="{Path(b["path"]).name}" alt="fixed8 base{base}"></a></section>' for base,b in zip([128,129,144,145],sheets))
    table=''.join(f'<tr><td>{r["original_global_index"]}</td><td>{html.escape(r["original_occurrence_id"])}</td><td>{r["original_semantic_state"]}</td><td><a href="clean96/{r["original_global_index"]:06d}.png">clean RGB96</a></td><td><a href="central32/{r["original_global_index"]:06d}.png">exact central32</a></td></tr>' for r in rows)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V176 C48 vs C49 fixed32</title><style>body{font-family:sans-serif;max-width:1740px;margin:24px auto;padding:0 16px;color:#17212e}img{width:100%;height:auto}td,th{padding:5px 12px;border-bottom:1px solid #ddd}</style><h1>V176 C48·C49 고정 source VAL32</h1><p>V175에서 C47 학습 전에 고정한 동일32 예시를 재사용했습니다. 전체 성능의 대표 표본이 아닙니다. C48은 public weak253+synthetic TRAIN4016, C49는 여기에 controlled MR/YCB CF314 owner loss를 더해 학습했습니다.</p><p>원 declared native semantic과 literal 점 소속 soft mass, 두 모델의 저장 owner argmax/U 및 raw FP32에서 계산한 표시용 float64 softmax를 구분합니다. owner FOREIGN은 native E의 정답이 아닙니다. 물리 native identity 미인증이며 이 readout은 새 라벨·학습 편입·모델/문턱 선택을 하지 않습니다.</p><p>q=(47.5,47.5)의 흰 표시는 표시 복사본에만 있습니다. clean96/central32는 V175 검증 PNG와 byte-identical합니다. <a href="FIXED32_SAVED_VALUES_AND_PIXEL_PARITY.json">저장값·pixel parity JSON</a></p>'+body+'<table><tr><th>index</th><th>occurrence</th><th>원 semantic</th><th>RGB96</th><th>central32</th></tr>'+table+'</table>')
    write(OUT/'COMPLETE.json',{'completed':True,'PASS':True,'UTC':datetime.now(timezone.utc).isoformat(),'code':bind(__file__),
        'C48_complete':bind(cp48),'C49_complete':bind(cp49),'source_V175_gallery_complete':bind(parent),
        'source_fixed32_inputs':old['fixed32_inputs'],'source_pixel_proofs':old['saved_values_and_pixel_proofs'],
        'saved_output_bindings':output_bindings,'contacts':sheets,'HTML':bind(OUT/'index.html'),
        'saved_values_and_pixel_parity':bind(OUT/'FIXED32_SAVED_VALUES_AND_PIXEL_PARITY.json'),
        'rows':32,'contacts_count':4,'clean96_and_central32_byte_identical_source_rows':32,
        'selected_output_metadata_qL_native_flags_classes_exact':True,'original_U_first_preserved':True,
        'max_panel_text_width_px':maxwidth,'original_source_crop_or_feature_or_weight_reads':0,
        'new_inference_fit_or_labels_or_admission_or_threshold_or_model_selection':False,
        'actual_visual_inspection':'pending separate actual tool view','HTML_browser_rendered':False,'automatic_retry':False})
    print(json.dumps({'completed':True,'output':str(OUT),'contacts':sheets}))


if __name__=='__main__':main()
