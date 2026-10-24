"""Fixed metadata-selected source readout, after an explicit all4 root trigger.

Never imports torch, an encoder, initializer or model. Reuses exact uint8 saved
initial96 crops and original canonical RGB. No reference/GT recentering or labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

H=Path(__file__).resolve().parent
V=H.parent
SELECTION=H/'FIXED12_SELECTION_SEALED_BEFORE_RGB_OR_V160_PREDICTIONS.json'
SELECTION_SHA='da33bc0afe5c9c8092758f5222e2deeb9f76efbe5b3af9e9ec6c9cf66e9c1f2b'
RUN=V/'runs/C39_C40_C41_C42_NATIVE_SLOT_OWNER_FIXED8000'
OUT=H/'rendered_fixed12'
ARMS=('C39_BASE_NATIVE0_OWNER0','C40_NATIVE_SLOT8_OWNER0','C41_NATIVE0_OWNER3_AUX','C42_NATIVE_SLOT8_OWNER3_AUX')
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def check(binding):
    p=Path(binding['path']);assert '_v112' not in str(p).lower()
    assert sha(p)==binding['sha256'],str(p)
    return p
def read(path):return json.loads(Path(path).read_text())
def ash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def write(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def font(size):return ImageFont.truetype(str(FONT),size)
def probability(z):
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
def call(logit,available):return 'U' if not available else ('E' if logit>=0 else 'D')

def make_card(case,context,crop,measurements):
    canvas=Image.new('RGB',(1800,1130),'#f5f7fa');d=ImageDraw.Draw(canvas)
    f30,f23,f20,f18=font(30),font(23),font(20),font(18)
    d.text((25,18),f"{case['card_index']:02d} {case['family']} · native {case['native_slot']} · 기존 약역할 {case['weak_role_state']}",font=f30,fill='#172b42')
    d.text((25,65),case['query_id'],font=f20,fill='#42566c')
    d.text((25,96),f"검증 전용 {case['split']} · 학습 제외 · 원래 q=({case['query_xy'][0]:.3f},{case['query_xy'][1]:.3f}) L={case['L']:.3f}",font=f20,fill='#7b431b')
    d.text((25,135),'원래 전체 RGB · 자홍색 표시는 원래 q의 화면 표시만',font=f20,fill='#172b42')
    whole=Image.fromarray(context,'RGB');wd=ImageDraw.Draw(whole);x,y=case['query_xy'];radius=max(6,round(max(whole.size)/150));wd.line((x-radius,y,x+radius,y),fill='#e000df',width=3);wd.line((x,y-radius,x,y+radius),fill='#e000df',width=3)
    whole.thumbnail((960,540),Image.Resampling.LANCZOS);canvas.paste(whole,(25,174))
    d.text((1020,135),'정확한 저장 RGB96 · 최근접 4×',font=f20,fill='#172b42')
    canvas.paste(Image.fromarray(crop,'RGB').resize((384,384),Image.Resampling.NEAREST),(1020,174))
    d.text((1440,135),'같은 crop 중앙32 · 6×',font=f18,fill='#172b42')
    canvas.paste(Image.fromarray(crop[32:64,32:64],'RGB').resize((192,192),Image.Resampling.NEAREST),(1440,174))
    d.text((1020,584),'RGB96 원본 픽셀 유지 · q/L 재계산·이동 없음',font=f18,fill='#42566c')
    d.text((1020,620),'중심 소유 정답 미부여 · owner3는 보조 진단치',font=f18,fill='#42566c')
    d.rectangle((25,742,1775,787),fill='#e4ebf3')
    for text,x in [('고정 arm',40),('외부 E 헤드 p(E)',650),('원래 inputU 우선 D/E/U',1000),('owner T / F / BG 확률 · 정답 아님',1330)]:d.text((x,752),text,font=f18,fill='#172b42')
    for index,m in enumerate(measurements):
        y=800+index*51
        d.text((40,y),m['arm'],font=f20,fill='#172b42');d.text((690,y),f"{m['external_E_probability']:.4f}",font=f23,fill='#172b42');d.text((1110,y),m['primary_call'],font=f23,fill='#172b42')
        d.text((1350,y),' / '.join(f'{v:.3f}' for v in m['owner_three_probabilities']),font=f20,fill='#42566c')
        d.line((25,y+42,1775,y+42),fill='#d8e0ea')
    d.text((25,1024),'기존 D/E는 이전 assistant/root의 구조적 약라벨 · 공식 사람 키포인트 가시성 정답이 아님',font=f20,fill='#7b431b')
    d.text((25,1060),'고정12개 설명용 비교 · 새 가시성/소유 라벨 및 held 학습 편입 없음 · 성능/성공 사례 선정 아님',font=f20,fill='#42566c')
    return canvas

def invented_checks():
    assert call(100,False)=='U' and call(-1,True)=='D' and call(0,True)=='E'
    assert abs(probability(0)-.5)<1e-12 and probability(-100)<1e-30 and probability(100)>1-1e-12
    a=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3)
    assert np.array_equal(np.asarray(Image.fromarray(a).resize((384,384),Image.Resampling.NEAREST))[::4,::4],a)
    return {'PASS':True,'invented_arrays_only':True,'inputU_first_logit0_decoder_checked':True,'stable_sigmoid_checked':True,'nearest96_display_pixels_checked':True,'source_RGB_crop_or_V160_predictions_read':False}

def render(entry_path):
    # Before this guard, only the root entry and already-sealed selection/code are read.
    entry=read(entry_path)
    assert entry['approved_all4_completed_readout'] is True
    assert entry['source_only'] is True and entry['training_admitted'] is False
    assert entry['selection']==bind(SELECTION) and sha(SELECTION)==SELECTION_SHA
    assert entry['renderer']==bind(__file__)
    assert not OUT.exists(), 'Preserve prior artifacts/failures; no automatic replacement/retry'
    control_path=check(entry['all4_control_complete']);assert control_path==RUN/'CONTROL_COMPLETE.json'
    control=read(control_path);assert control['completed'] and control['PASS'] and control['source_only'] and control['updates_each']==8000
    assert tuple(control['arms'])==ARMS
    frozen=read(check(control['all4_frozen_before_source_comparison']))
    assert frozen['completed'] and tuple(frozen['arms'])==ARMS and set(frozen['checkpoints'])==set(ARMS)
    selection=read(SELECTION);assert selection['completed'] and len(selection['cases'])==12
    assert not selection['source_RGB_or_crop_bodies_read'] and not selection['V160_prediction_score_metric_checkpoint_bodies_read']
    for binding in selection['metadata_inputs'].values():check(binding)
    for arm in ARMS:
        fit=read(check(control['arms'][arm]['fit']));assert fit['completed'] and fit['optimizer_updates']==8000
        assert frozen['checkpoints'][arm]['path']==str(RUN/arm/'FINAL_CHECKPOINT.pt')
    # Never open/hash checkpoint bodies. FINAL output packets are read only after all4 guard.
    packets={};prediction_bindings=[]
    for arm in ARMS:
        for family in ('MR1840','PALLOX512'):
            path=RUN/arm/f'FINAL_{family}_OUTPUTS.npz';prediction_bindings.append(bind(path))
            with np.load(path,allow_pickle=False) as f:
                packets[(arm,family)]={k:f[k].copy() for k in ('global_index','query_id','native_slot','query_xy','L','query_available','raw_three_logits','raw_owner_three_logits','owner_three_probabilities','primary_visibility_prediction')}
    OUT.mkdir();records=[];views=[];body_bindings=[]
    try:
        for case in selection['cases']:
            rgb_path=check(case['canonical_RGB_NPY']);rgb=np.load(rgb_path,allow_pickle=False)
            assert rgb.dtype==np.uint8 and list(rgb.shape)==case['raw_hw']+[3] and ash(rgb)==case['decoded_RGB_sha256']
            crop_path=check(case['crop_packet'])
            with np.load(crop_path,allow_pickle=False) as f:
                indices=f['global_index'];positions=np.flatnonzero(indices==case['global_index']);assert len(positions)==1
                pos=int(positions[0]);crop=f['rgb96'][pos].copy();valid=f['valid96'][pos].copy()
            assert crop.shape==(96,96,3) and crop.dtype==np.uint8 and ash(crop)==case['RGB96_HWC_sha256'] and ash(valid)==case['valid96_sha256']
            body_bindings.extend([case['canonical_RGB_NPY'],case['crop_packet']]);measurements=[]
            i=case['global_index']
            for arm in ARMS:
                a=packets[(arm,case['family'])]
                assert int(a['global_index'][i])==i and str(a['query_id'][i])==case['query_id'] and int(a['native_slot'][i])==case['native_slot']
                assert np.array_equal(a['query_xy'][i],case['query_xy']) and float(a['L'][i])==case['L'] and bool(a['query_available'][i])==case['query_available']
                raw=a['raw_three_logits'][i];owner=a['raw_owner_three_logits'][i];op=a['owner_three_probabilities'][i]
                assert np.isfinite(raw).all() and np.isfinite(owner).all() and np.isfinite(op).all() and np.isclose(op.sum(),1)
                expected=-1 if not case['query_available'] else int(raw[0]>=0);assert int(a['primary_visibility_prediction'][i])==expected
                measurements.append({'arm':arm,'external_E_logit':float(raw[0]),'external_E_probability':probability(float(raw[0])),'primary_call':call(float(raw[0]),case['query_available']),'owner_three_logits':owner.tolist(),'owner_three_probabilities':op.tolist(),'owner_output_diagnostic_only':True})
            folder=OUT/f"CASE_{case['card_index']:02d}_{case['family']}_{i:04d}";folder.mkdir()
            raw_path=folder/'RAW_RGB96.png';Image.fromarray(crop,'RGB').save(raw_path)
            assert np.array_equal(np.asarray(Image.open(raw_path).convert('RGB')),crop)
            card=make_card(case,rgb,crop,measurements);card_path=folder/'COMPARISON_CARD.png';card.save(card_path);views.append(card)
            records.append({'case':case,'measurements':measurements,'raw_RGB96':bind(raw_path),'card':bind(card_path),'saved_RGB96_pixels_exact':True,'training_admitted':False,'new_labels':False})
        # Four pages of3cards preserve legibility; individual cards remain full resolution.
        contacts=[]
        for page in range(4):
            canvas=Image.new('RGB',(1800,3390),'white')
            for k in range(3):canvas.paste(views[page*3+k],(0,k*1130))
            path=OUT/f'CONTACT_PAGE_{page+1:02d}.png';canvas.save(path);contacts.append(bind(path))
        manifest=OUT/'FIXED12_RENDER_MANIFEST.json';write(manifest,{'completed':True,'selection':bind(SELECTION),'records':records,'contacts':contacts})
        write(OUT/'COMPLETE.json',{'completed':True,'PASS':True,'source_only':True,'fixed12':12,'renderer':bind(__file__),'entry':bind(entry_path),'selection':bind(SELECTION),'all4_control':bind(control_path),'all4_freeze':control['all4_frozen_before_source_comparison'],'prediction_packets':prediction_bindings,'source_body_bindings':body_bindings,'manifest':bind(manifest),'contacts':contacts,'exact_saved_RGB96_reused':True,'new_encoder_model_initializer_or_PnP_calls':0,'source_checkpoint_bodies_opened':False,'qL_native_split_or_weak_roles_changed':False,'new_visibility_or_owner_labels':False,'held_supervision_admitted':False,'training_admitted':False,'actual_visual_QA_pending':True,'HTML_browser_rendered':False})
        print(json.dumps({'completed':True,'result':bind(OUT/'COMPLETE.json')},ensure_ascii=False))
    except BaseException:
        import traceback
        write(OUT/'FAILED_RENDER.json',{'completed':False,'traceback':traceback.format_exc(),'automatic_retry':False});raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',type=Path);parser.add_argument('--invented-checks',action='store_true');args=parser.parse_args()
    if args.invented_checks and not args.entry:print(json.dumps(invented_checks()))
    elif args.entry and not args.invented_checks:render(args.entry)
    else:raise SystemExit('Use --invented-checks or root-approved all4 completion --entry')
