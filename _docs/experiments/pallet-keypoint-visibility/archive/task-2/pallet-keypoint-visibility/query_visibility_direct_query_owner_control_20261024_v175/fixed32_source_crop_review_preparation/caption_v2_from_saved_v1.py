"""Root-authorized display-only revision; v1 clean PNG/JSON only, no NPZ reads."""
import hashlib
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
V = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_direct_query_owner_control_20261024_v175')
H = V/'fixed32_source_crop_review_preparation'
OLD = V/'fixed32_source_crop_review_v1'
OUT = V/'fixed32_source_crop_review_v2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bind(p):
    return {'path':str(Path(p)), 'sha256':sha(p)}


def load(p):
    return json.loads(Path(p).read_text())


def verify(b):
    assert sha(b['path']) == b['sha256']


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')


def main():
    assert not OUT.exists(), 'preserve any previous attempt; no automatic retry'
    code=H/'render_fixed32_source_crops.py'
    assert sha(code)=='779008f6613f38d46bc504dc89faac4bc0e30f5e345a01c31fabf91901e0a709'
    m=runpy.run_path(str(code))  # Original pure display functions, no __main__/NPZ/run call.
    complete=load(OLD/'COMPLETE.json'); assert complete['completed'] and complete['PASS']
    for b in complete['contacts']+[complete['HTML'],complete['rendered_values_and_pixel_proofs'],complete['fixed32_inputs']]: verify(b)
    original_bindings=[bind(OLD/'COMPLETE.json'),*complete['contacts'],complete['HTML'],complete['rendered_values_and_pixel_proofs']]
    inp=load(complete['fixed32_inputs']['path']); vals=load(complete['rendered_values_and_pixel_proofs']['path'])['rows']
    assert [r['original_global_index'] for r in inp['rows']]==[r['original_global_index'] for r in vals]
    assert len(vals)==32
    outputs={name:{'probabilities':{},'prediction':{}} for name in m['ARMS']}
    for r,v in zip(inp['rows'],vals):
        i=r['original_global_index']; verify(v['clean96']); verify(v['central32'])
        rgb=np.asarray(Image.open(v['clean96']['path'])).copy()
        assert rgb.dtype==np.uint8 and rgb.shape==(96,96,3) and m['ash'](rgb)==v['rgb96_array_sha256']
        assert np.array_equal(np.asarray(Image.open(v['central32']['path'])),rgb[32:64,32:64])
        r['_rgb']=rgb
        for name in m['ARMS']:
            outputs[name]['probabilities'][i]=np.asarray(v['saved_outputs'][name]['probabilities'],np.float64)
            outputs[name]['prediction'][i]=v['saved_outputs'][name]['prediction']
    OUT.mkdir(); sheets=[]; max_text_width=0
    for base in [128,129,144,145]:
        rows=[r for r in inp['rows'] if r['base_index']==base]
        W,Hh,cw,ch=1720,1260,420,530
        im=Image.new('RGB',(W,Hh),'#f4f6f8');d=ImageDraw.Draw(im); font=m['font'];factor=rows[0]['factors']
        d.text((20,14),f"V175 fixed VAL base{base} | {factor['material']} / {factor['aspect']}",fill='#101828',font=font(25))
        d.text((20,51),'Pre-fit fixed examples, not representative performance | RGB96 original q/L; center cross is display only',fill='#344054',font=font(19))
        for k,r in enumerate(rows):
            x=20+(k%4)*cw;y=90+(k//4)*ch
            d.rectangle((x,y,x+cw-10,y+ch-10),fill='white',outline='#cbd5e1')
            im.paste(m['marked_copy'](r['_rgb']),(x+60,y+10))
            lines=m['display_lines'](r,outputs)
            # This is the only contact-sheet caption change; every numeric token is retained.
            lines=[s.replace('Exact-point mass T/F/BG/U:','Point T/F/BG/U:') for s in lines]
            for n,line in enumerate(lines):
                width=d.textlength(line,font=font(16));max_text_width=max(max_text_width,width)
                assert width<=cw-30, (base,r['original_global_index'],line,width)
                d.text((x+10,y+306+n*27),line,fill='#111827',font=font(16))
        footer=['Original declared D/E/S/U is a different task from literal point owner; owner FOREIGN does not mean native E.',
                'T = incident + other target; U mass = ambiguous + invalid. Soft masses have no truth cutoff.',
                'Source raster registration only; initializer physical native identity uncertified. No new labels, admission or model selection.']
        for n,line in enumerate(footer):d.text((20,1160+n*27),line,fill='#344054',font=font(18))
        path=OUT/f'BASE_{base:03d}_8_FIXED_CROPS.png';im.save(path);sheets.append(bind(path))
    page=Path(complete['HTML']['path']).read_text()
    page=page.replace('href="clean96/','href="../fixed32_source_crop_review_v1/clean96/')
    page=page.replace('href="central32/','href="../fixed32_source_crop_review_v1/central32/')
    page=page.replace('href="RENDERED32_SAVED_VALUES_AND_PIXEL_PROOFS.json"','href="../fixed32_source_crop_review_v1/RENDERED32_SAVED_VALUES_AND_PIXEL_PROOFS.json"')
    page=page.replace('<h1>V175 고정 source VAL32 crop 비교</h1>','<h1>V175 고정 source VAL32 crop 비교</h1><p>v2: 점 소속 행의 표시 문구만 줄여 panel 겹침을 고쳤습니다. 원 v1·32 crop·수치·선택은 그대로 보존했습니다.</p>')
    (OUT/'index.html').write_text(page)
    for b in original_bindings:verify(b)
    write(OUT/'COMPLETE.json',{'completed':True,'PASS':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'code':bind(__file__),'original_renderer':bind(code),'v1_complete':bind(OLD/'COMPLETE.json'),
        'fixed32_inputs':complete['fixed32_inputs'],'saved_values_and_pixel_proofs':complete['rendered_values_and_pixel_proofs'],
        'contacts':sheets,'HTML':bind(OUT/'index.html'),'fixed_rows':32,'clean96_verified':32,'central32_verified':32,
        'display_only_change':'Exact-point mass T/F/BG/U: -> Point T/F/BG/U:',
        'numeric_tokens_or_predictions_or_crops_changed':False,'v1_outputs_unchanged':True,
        'max_panel_line_width_px':max_text_width,'available_panel_line_width_px':390,
        'original_source_or_NPZ_or_feature_or_weight_reads':0,'new_model_or_threshold_or_label_or_admission':False,
        'automatic_retry':False,'HTML_browser_rendered':False,'actual_visual_inspection':'pending actual tool view'})
    print(json.dumps({'completed':True,'output':str(OUT),'contacts':sheets}))


if __name__=='__main__':main()
