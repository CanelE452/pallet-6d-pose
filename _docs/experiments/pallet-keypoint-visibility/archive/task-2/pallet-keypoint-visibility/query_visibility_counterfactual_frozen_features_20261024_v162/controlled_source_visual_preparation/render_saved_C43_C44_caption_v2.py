"""Saved-output-only fixed C43/C44 source readout; no model/encoder/fit.
Preparation reads fixed selection/lineage JSON only. Rendering needs both frozen.
"""
from pathlib import Path
import argparse,hashlib,html,json,math,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import PIL
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V160=P/'query_visibility_native_slot_owner_factorial_20261024_v160'
V161=P/'query_visibility_public_train_owner_counterfactual_20261024_v161'
OLD_SELECTION=V160/'source_crop_readout/FIXED12_SELECTION_SEALED_BEFORE_RGB_OR_V160_PREDICTIONS.json'
CF_SELECTION=V161/'metadata_assignment/FIXED12_VISUAL_SELECTION_BEFORE_PIXELS.json'
CF_GALLERY=V161/'fixed12_visual_v1/FIXED12_RENDER_MANIFEST.json'
PROTOCOL=H/'PROTOCOL_BEFORE_C43_C44_OUTPUT_BODIES_CAPTION_V2.json';OUT=V/'controlled_source_visual_v2'
RUN=V/'runs/C43_C44_OWNER_CF_FIXED8000'
ARMS=('C43_REPLAY_C41_OWNER253_NATIVE0','C44_OWNER567_WITH_CF314_NATIVE0')
VARIANTS=('BEFORE','COVER','ADJACENT_SHAM')
FONT=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def font(n):return ImageFont.truetype(str(FONT),n)
def sigmoid(z):
    if not np.isfinite(z):return None
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
def primary(logit,available):return 'U' if not available else ('E' if logit>=0 else 'D')
def softmax(z):
    z=np.asarray(z,np.float64);assert z.shape==(3,) and np.isfinite(z).all();v=np.exp(z-z.max());return (v/v.sum()).tolist()
def measure(packet,index):
    avail=bool(packet['query_available'][index]);logit=float(packet['raw_three_logits'][index,0])
    probs=softmax(packet['raw_owner_three_logits'][index])
    if 'owner_three_probabilities' in packet:assert np.allclose(probs,packet['owner_three_probabilities'][index],atol=1e-6,rtol=1e-6)
    return dict(raw_external_E_logit=logit,raw_pE=sigmoid(logit),primary_call=primary(logit,avail),query_available=avail,owner_T_F_BG_probabilities=probs,owner_diagnostic_only=True)
def old_card(case,rgb,crop,measurements):
    c=Image.new('RGB',(1800,1060),'#f5f8fb');d=ImageDraw.Draw(c)
    d.text((24,22),f"고정 SOURCE {case['card_index']:02d} · {case['family']} · native{case['native_slot']} · 기존 약역할 {case['weak_role_state']}",font=font(31),fill='#20364c')
    d.text((24,66),case['query_id'],font=font(24),fill='#3b526a');q=case['query_xy']
    d.text((24,104),f"검증 전용 {case['split']} · q=({q[0]:.6f},{q[1]:.6f}) L={case['L']:.6f} · q/L 변경 없음",font=font(24),fill='#965111')
    d.text((24,146),'원래 전체 RGB · 자홍색 십자는 원래 q의 화면 표시만',font=font(22),fill='#20364c')
    im=Image.fromarray(rgb);im.thumbnail((960,540),Image.Resampling.BILINEAR);dd=ImageDraw.Draw(im);sx,sy=im.width/rgb.shape[1],im.height/rgb.shape[0];x,y=q[0]*sx,q[1]*sy;dd.line((x-5,y,x+5,y),fill='#f000df',width=2);dd.line((x,y-5,x,y+5),fill='#f000df',width=2);c.paste(im,(24,180))
    d.text((1020,146),'정확한 원래 RGB96 · 4×',font=font(22),fill='#20364c');c.paste(Image.fromarray(crop).resize((384,384),Image.Resampling.NEAREST),(1020,180))
    d.text((1440,146),'같은 crop 중앙32 · 6×',font=font(21),fill='#20364c');c.paste(Image.fromarray(crop[32:64,32:64]).resize((192,192),Image.Resampling.NEAREST),(1440,180))
    d.rectangle((24,742,1775,790),fill='#e4ebf3');d.text((38,752),'고정 조건',font=font(23),fill='#20364c');d.text((690,752),'raw p(E)',font=font(23),fill='#20364c');d.text((970,752),'입력U 우선 D/E/U',font=font(23),fill='#20364c');d.text((1285,752),'owner T/F/BG · held 정답 없음',font=font(23),fill='#20364c')
    for j,(arm,m) in enumerate(zip(ARMS,measurements)):
        yy=812+j*62;d.text((38,yy),arm,font=font(23),fill='#20364c');d.text((690,yy),f"{m['raw_pE']:.6f}",font=font(26),fill='#20364c');d.text((1055,yy),m['primary_call'],font=font(26),fill='#20364c');d.text((1285,yy),' / '.join(f'{x:.3f}' for x in m['owner_T_F_BG_probabilities']),font=font(24),fill='#3b526a')
    d.text((24,958),'기존 D/E/U는 prior assistant/root 구조 약라벨 · 공식 사람 keypoint 가시성 gold 아님 · held owner 새 라벨 없음',font=font(23),fill='#965111')
    d.text((24,1000),'사전고정 설명 사례만 · 새 성공 사례 선택/가시성·소유 재라벨/held 학습 편입 없음 · 전체 목표/채택 판정과 별도',font=font(23),fill='#3b526a')
    return c
def cf_card(case,source_card,measurements):
    # Preserve the V161 fullRGB/exact96/central32/alpha/raw-footprint figure.
    assert source_card.size==(1800,1450)
    c=Image.new('RGB',(1800,1850),'#f5f8fb');c.paste(source_card,(0,0));d=ImageDraw.Draw(c)
    # Caption-only revision: historical source proof is not a no-training claim for V162.
    d.rectangle((0,1374,1799,1449),fill='#f5f8fb')
    d.text((22,1382),'상단은 V161 생성 당시 픽셀 입력 증명 기록 · 신규 native 가시성/좌표품질 UNKNOWN · 자연 광학 가림 정답 아님',font=font(22),fill='#965111')
    d.text((22,1420),'아래는 V162 실제 학습 후 저장 출력: C43=CF0 / C44=CF314 owner 보조 학습 완료',font=font(22),fill='#20364c')
    d.text((22,1465),'C43/C44 저장 출력 비교 · 모든 CF native D/E/좌표품질 정답 UNKNOWN · 아래 primary는 예측값만',font=font(27),fill='#965111')
    weak=('TARGET 약관찰 계보 · 추가 loss 중복 제외','FOREIGN 통제 owner 약감독','TARGET 통제 owner 약감독')
    for col,name in enumerate(VARIANTS):
        x=22+594*col;d.text((x,1510),name,font=font(25),fill='#20364c');d.text((x,1550),weak[col],font=font(20),fill='#965111')
        for j,arm in enumerate(ARMS):
            m=measurements[name][arm];yy=1590+96*j;d.text((x,yy),('C43 owner253' if j==0 else 'C44 owner567 = 253+314'),font=font(23),fill='#20364c')
            d.text((x,yy+31),f"raw p(E)={m['raw_pE']:.6f} · 예측 {m['primary_call']} · owner "+'/'.join(f'{z:.3f}' for z in m['owner_T_F_BG_probabilities']),font=font(19),fill='#3b526a')
    d.text((22,1804),'Owner 약감독은 native 가시성 정답 아님 · 이 readout은 새 라벨·추가 학습 편입·모델/문턱/epoch 선택을 만들지 않음',font=font(24),fill='#965111')
    return c
def invented():
    p={'query_available':np.array([False,True,True]),'raw_three_logits':np.array([[99,0,0],[0,0,0],[-2,0,0]],np.float32),'raw_owner_three_logits':np.array([[0,0,0],[2,1,0],[1,1,1]],np.float32)}
    mm=[measure(p,i) for i in range(3)]
    assert [x['primary_call'] for x in mm]==['U','E','D'] and sigmoid(0)==.5 and abs(sum(mm[1]['owner_T_F_BG_probabilities'])-1)<1e-12
    assert sigmoid(1000)==1 and sigmoid(-1000)==0
    return dict(completed=True,PASS=True,invented_only=True,inputU_first_and_fixed_logit0_decoder=True,stable_sigmoid_softmax=True,no_source_or_modeloutput_bodies_read=True,encoder_classifier_fit_calls=0)
def prepare():
    old=read(OLD_SELECTION);cf=read(CF_SELECTION);gallery=read(CF_GALLERY)
    assert len(old['cases'])==len(cf['selected'])==len(gallery['records'])==12
    ci={r['pair_id']:r for r in gallery['records']};refs=[]
    for s in cf['selected']:
        r=ci[s['pair_id']];assert r['selection_rank']==s['selection_rank'] and r['query_id']==s['query_id'] and r['native_slot']==s['native_slot'] and r['original_q']==s['query_xy'] and r['original_L']==s['L']
        refs.append(dict(pair_id=s['pair_id'],selection_rank=s['selection_rank'],query_id=s['query_id'],native_slot=s['native_slot'],query_xy=s['query_xy'],L=s['L'],source_card=r['card'],pixel_input_proof_PASS=r['pixel_input_proof_PASS']))
    refsfile=H/'CF_DISPLAY_REFS12_METADATA_ONLY_CAPTION_V2.json';write(refsfile,dict(completed=True,records=refs,source_card_bodies_read=False))
    gen=H/'GENERATED_INVENTED_READOUT_CHECKS_CAPTION_V2.json';write(gen,invented())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),old_fixed12_selection=bind(OLD_SELECTION),CF_fixed12_selection=bind(CF_SELECTION),CF_original_gallery=bind(CF_GALLERY),CF_display_refs=bind(refsfile),font=bind(FONT),generated=bind(gen),arms=list(ARMS),run_directory=str(RUN),expected_output_paths={arm:{family:str(RUN/arm/f'FINAL_{family}_OUTPUTS.npz') for family in ('MR1840','PALLOX512','CF3816')} for arm in ARMS},output=str(OUT),runtime=dict(executable=sys.executable,numpy=np.__version__,Pillow=PIL.__version__),body_reads_during_preparation=0,all2_control_and_freeze_required=True,root_completion_trigger_required=True,checkpoint_feature_encoder_or_model_reads=False,selections_unchanged=True,old_source_semantics='Original existing weak D/E/U lineage only; held owner UNKNOWN/no new annotation',CF_semantics='BEFORE prior weak TARGET lineage not additional duplicate loss; COVER/SHAM controlled owner weak only; all native D/E/quality UNKNOWN; outputs diagnostic only',primary_decode='original inputU first else fixed raw external E logit>=0; no entity/quality/owner gate',source_only=True,training_admitted=False)
    write(PROTOCOL,p);template=H/'ROOT_ALL2_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE_CAPTION_V2.json';write(template,dict(approved_actual_saved_output_readout=False,code=p['code'],protocol=bind(PROTOCOL),root_completion_trigger=None,control_complete=None,prediction_packets=None,source_only=True,encoder_allowed=False,classifier_forward_allowed=False,fit_allowed=False,checkpoint_allowed=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,training_admitted=False,output=str(OUT)))
    write(H/'PREPARATION_COMPLETE_CAPTION_V2.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),generated=bind(gen),entry_template=bind(template),source_or_modeloutput_bodies_read=0,actual_rendering=False))
    print(json.dumps(dict(code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE_CAPTION_V2.json'))),flush=True)
def render(entry):
    e=read(entry);assert e['approved_actual_saved_output_readout'] is True and e['source_only'] is True
    for k in ('encoder_allowed','classifier_forward_allowed','fit_allowed','checkpoint_allowed','current_allowed','V112_allowed','PnP_allowed','training_admitted'):assert e[k] is False
    p=read(check(e['protocol']));assert e['protocol']==bind(PROTOCOL) and e['code']==p['code']==bind(__file__) and not OUT.exists() and e['output']==p['output']==str(OUT)
    for k in ('old_fixed12_selection','CF_fixed12_selection','CF_original_gallery','CF_display_refs','font','generated'):check(p[k])
    trigger=read(check(e['root_completion_trigger']));assert trigger.get('approved_saved_output_readout',trigger.get('approved_generated_source_readout',False)) is True
    control=read(check(e['control_complete']));assert control['completed'] and control['PASS'] and control['updates_each']==8000 and set(control['arms'])==set(ARMS)
    freeze=read(check(control['all2_frozen_before_source_comparison']));assert freeze['completed'] and set(freeze['arms'])==set(ARMS)
    packets={};outbindings=[]
    for arm in ARMS:
        packets[arm]={}
        for family in ('MR1840','PALLOX512','CF3816'):
            b=e['prediction_packets'][arm][family];assert b['path']==p['expected_output_paths'][arm][family];keys=('raw_three_logits','raw_owner_three_logits','owner_three_probabilities','owner_class_order','query_available','native_slot','query_xy','L','input_occurrence_id','original_query_id') if family=='CF3816' else ('raw_three_logits','raw_owner_three_logits','owner_three_probabilities','owner_class_order','query_available','native_slot','query_xy','L','query_id','global_index')
            with np.load(check(b),allow_pickle=False) as z:packets[arm][family]={k:z[k].copy() for k in keys}
            assert packets[arm][family]['owner_class_order'].tolist()==['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED'];outbindings.append(b)
    old=read(check(p['old_fixed12_selection']))['cases'];cf=read(check(p['CF_fixed12_selection']))['selected'];refs={r['pair_id']:r for r in read(check(p['CF_display_refs']))['records']};OUT.mkdir();records=[];oldcards=[];cfcards=[]
    for case in old:
        with np.load(check(case['crop_packet']),allow_pickle=False) as z:
            pos=np.flatnonzero(z['global_index']==case['global_index']);assert len(pos)==1;crop=z['rgb96'][pos[0]].copy();valid=z['valid96'][pos[0]].copy()
        assert ash(crop)==case['RGB96_HWC_sha256'] and ash(valid)==case['valid96_sha256']
        rgb=np.load(check(case['canonical_RGB_NPY']),allow_pickle=False);assert ash(rgb)==case['decoded_RGB_sha256'];ms=[]
        for arm in ARMS:
            z=packets[arm][case['family']];ii=np.flatnonzero(z['global_index']==case['global_index']);assert len(ii)==1;i=int(ii[0]);assert z['query_id'][i]==case['query_id'] and int(z['native_slot'][i])==case['native_slot'] and z['query_xy'][i].tolist()==case['query_xy'] and float(z['L'][i])==case['L'] and bool(z['query_available'][i])==case['query_available'];ms.append(measure(z,i))
        image=old_card(case,rgb,crop,ms);path=OUT/f'ORIGINAL_{case["card_index"]:02d}.png';image.save(path);oldcards.append(image);records.append(dict(kind='ORIGINAL_SOURCE_FIXED12',case_index=case['card_index'],query_id=case['query_id'],original_weak_role=case['weak_role_state'],measurements=dict(zip(ARMS,ms)),card=bind(path),new_owner_or_visibility_labels=False))
    for case in cf:
        ref=refs[case['pair_id']];ms={}
        for variant in VARIANTS:
            ms[variant]={}
            oid=case['pair_id']+'/'+variant+'/native'+str(case['native_slot'])
            for arm in ARMS:
                z=packets[arm]['CF3816'];ii=np.flatnonzero(z['input_occurrence_id']==oid);assert len(ii)==1;i=int(ii[0]);assert z['original_query_id'][i]==case['query_id'] and int(z['native_slot'][i])==case['native_slot'] and z['query_xy'][i].tolist()==case['query_xy'] and float(z['L'][i])==case['L'];ms[variant][arm]=measure(z,i)
        with Image.open(check(ref['source_card'])) as im:source=im.convert('RGB').copy()
        image=cf_card(case,source,ms);path=OUT/f'CF_TRIAD_{case["selection_rank"]:02d}.png';image.save(path);cfcards.append(image);records.append(dict(kind='CF_FIXED12_TRIAD',case_index=case['selection_rank'],pair_id=case['pair_id'],query_id=case['query_id'],native_visibility_and_quality_target='UNKNOWN',controlled_owner_only_weak=True,measurements=ms,card=bind(path),new_labels=False))
    contacts=[]
    for name,cards in (('ORIGINAL',oldcards),('CF_TRIAD',cfcards)):
        for i in range(0,12,2):
            page=Image.new('RGB',(1800,cards[i].height*2),'white');page.paste(cards[i],(0,0));page.paste(cards[i+1],(0,cards[i].height));path=OUT/f'{name}_CONTACT_{i//2+1:02d}.png';page.save(path);contacts.append(bind(path))
    hp=OUT/'index.html';hp.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>V162 C43 C44 fixed source readout</title><style>body{font:20px system-ui;max-width:1800px;margin:auto;padding:24px}img{width:100%;height:auto}article{margin:36px 0}</style><h1>C43/C44 · 고정 SOURCE/CF 설명 사례</h1><p>기존 V160 SOURCE12와 V161 CF12를 그대로 유지합니다. SOURCE의 D/E/U는 이전 약라벨 계보이며 held owner 정답은 없습니다. CF의 native 가시성·좌표품질은 UNKNOWN입니다. Controlled owner 약감독을 native 가시성 정답으로 바꾸지 않으며 이 그림은 전체 성능/모델 채택 판정과 별도입니다.</p>'+''.join('<article><h2>'+r['kind']+' '+str(r['case_index'])+'</h2><img src="'+Path(r['card']['path']).name+'" alt="fixed saved source comparison"></article>' for r in records)+'</html>')
    manifest=OUT/'READOUT_MANIFEST.json';write(manifest,dict(completed=True,records=records,contacts=contacts,HTML=bind(hp),prediction_packets=outbindings,control_complete=e['control_complete']))
    write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),entry=bind(entry),root_trigger=e['root_completion_trigger'],control_complete=e['control_complete'],manifest=bind(manifest),original_cases=12,CF_triads=12,selections_unchanged=True,new_labels=False,training_admitted=False,model_encoder_classifier_fit_calls=0,actual_visual_QA_pending=True,HTML_browser_rendered=False))
    print(json.dumps(dict(completed=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare and not a.entry:prepare()
    elif a.entry and not a.prepare:render(a.entry)
    else:raise SystemExit('Prepare first; actual saved source/output rendering requires root all2freeze trigger')
