"""Fixed19 V178 evaluation display from existing V169 PNGs and frozen CSV only.
No NPZ/checkpoint/source RGB/encoder/model/GTcenter crop reads.
"""
import csv,json,hashlib,math,html,os
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from datetime import datetime,timezone
D=Path(__file__).resolve().parent; V=D.parent; P=V.parent
A=P/'query_visibility_frozen_public_threshold_current_benchmark_20261024_v169/fixed_current_crop_review_v1'
E=V/'saved_five_model_evaluation_v1'
MODELS=['C47','C48','C49'];CLASS={-1:'U',0:'T',1:'F',2:'BG'}
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
BG='#f5f8fb';INK='#20364c';AMBER='#965111'

def bind(p):p=Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']
def put(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def font(n):return ImageFont.truetype(FONT,n)
def text(dr,xy,s,size=26,color=INK,maxw=2025):
    box=dr.textbbox(xy,s,font=font(size));assert box[2]<=xy[0]+maxw and box[3]<=1480,(s,box);dr.text(xy,s,font=font(size),fill=color)
def rel(p):return os.path.relpath(p,D)
def pixelsha(im):return hashlib.sha256(im.tobytes()).hexdigest()

def main():
    scope=V/'ROOT_SCOPE_BEFORE_NEW_CURRENT_FEATURE_CHECKPOINT_SCORE_OR_LABEL_BODIES.json'
    cause=V/'ROOT_CAUSE_SEPARATION_READOUT_BEFORE_CURRENT_LABEL_AND_SCORE_BODIES.json'
    selection=A/'CASE_IDS_BEFORE_ANY_RGB.json';source=A/'CROP_RENDER_COMPLETE.json';v2=A/'caption_aligned_v2/CAPTION_ALIGNMENT_COMPLETE.json'
    assert bind(selection)['sha256']=='db1f1303e9faebf675971696d2f60e54caf97a52beddd64157d11d2029a8735f'
    assert bind(source)['sha256']=='9f47236d47af91e3667cd18589106deed2e324072cf29291b96c6f1286b219cd'
    s=read(selection);sr=read(source);v2r=read(v2);es=read(E/'EVALUATION_SEAL.json');er=read(E/'CURRENT_FROZEN_BENCHMARK_RESULTS.json')
    assert sr['completed'] and sr['PASS'] and v2r['completed'] and v2r['PASS'] and es['completed'] and es['PASS'] and er['completed'] and er['PASS']
    assert er['selected_model'] is None and er['adopted'] is False
    check(es['scope']);assert es['scope']==bind(scope)
    for b in es['outputs']:
        if Path(b['path']).name in ['ALL1784_NATIVE_LEDGER.csv','CURRENT_FROZEN_BENCHMARK_RESULTS.json']:check(b)
    ids=s['selected_global_indices'];assert len(ids)==19 and len(set(ids))==19
    panels={r['global_index']:r for r in sr['panels']};v2cards={r['global_index']:r['caption_aligned'] for r in v2r['cards']};assert set(panels)==set(v2cards)==set(ids)
    with (E/'ALL1784_NATIVE_LEDGER.csv').open(newline='') as f:allrows=list(csv.DictReader(f))
    assert len(allrows)==1784;rows={int(r['global_index']):r for r in allrows if int(r['global_index']) in ids};assert set(rows)==set(ids)
    input_bindings={'scope':bind(scope),'cause_addendum':bind(cause),'selection':bind(selection),'original_render':bind(source),'caption_v2':bind(v2),'caption_v2_index':bind(A/'caption_aligned_v2/index.html'),'evaluation_seal':bind(E/'EVALUATION_SEAL.json'),'evaluation_results':bind(E/'CURRENT_FROZEN_BENCHMARK_RESULTS.json'),'saved_ledger':bind(E/'ALL1784_NATIVE_LEDGER.csv')}
    put(D/'DISPLAY_RULES_AND_INPUT_BINDINGS.json',{'UTC':datetime.now(timezone.utc).isoformat(),'fixed19':ids,'selection_unchanged':True,'inputs':input_bindings,'only_existing_saved_PNG_and_CSV':True,'raw96_c32_original_qL_native_unchanged':True,'display_aux_subsets':'Original saved96[47:49,47:49] and [40:56,40:56], nearest only; no new q/crop.','owner_is_prediction_not_GT':'T/F/BG/U owner argmax; T=>D,F/BG=>E,U=>U is a diagnostic binary mapping, not proof of physical native external occlusion.','no_new_labels_fit_model_threshold_selection':True,'raw_RGB_cache_NPZ_features_weights_models_V112_reads':False})
    records=[]
    content=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V178 fixed19 saved crop comparison</title><style>body{font:19px system-ui;max-width:2080px;margin:auto;padding:24px;background:#f5f8fb;color:#20364c}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bac5d0}article{margin:48px 0}</style><h1>동일 고정19 · C47/C48/C49 owner 예측 재대조</h1><p>V169에서 이미 선택한 19 ID 전부. 새 모델 결과에 따른 교체·추가 선택은 없습니다. 과거 결과에 노출된 설명 사례이며 전체 성능의 대표 표본이 아닙니다.</p><p>원 native 가시성 상태, literal q 소속 예측, crop 안 대상 존재, 기존 좌표 reference 오차는 서로 다른 진단입니다. F/BG 예측만으로 물리적 native external 가림을 확정하지 않습니다. reference≤3px는 기존 표시용 근거이며 물리 native 3px gold가 아닙니다.</p><p>원 V169 clean96/중앙32/context를 그대로 재사용했습니다. 중심2×2/16×16은 동일 saved96의 확대 부분집합입니다. 새 label/교정/학습 편입/모델·문턱 선택이 없습니다. HTML browser 렌더링 검증은 별도 주장하지 않습니다.</p><p>고정 ID: '+', '.join(f'<a href="#g{i}">g{i}</a>' for i in ids)+'</p>']
    for i in ids:
        p=panels[i];r=rows[i];k=int(r['native_corner']);assert p['id']==r['id'] and p['native_corner']==k
        assert p['original_query_xy']==[float(r['query_x']),float(r['query_y'])] and p['original_L']==float(r['roi_side'])
        for key in ['raw96','central32','valid96','card']:check(p[key])
        check(v2cards[i]);old=Image.open(v2cards[i]['path']).convert('RGB');assert old.size==(2080,1400)
        raw=Image.open(p['raw96']['path']).convert('RGB');c32=Image.open(p['central32']['path']).convert('RGB');valid=Image.open(p['valid96']['path']).convert('L')
        assert raw.size==(96,96) and c32.size==(32,32) and valid.size==(96,96)
        assert pixelsha(raw)==p['RGB96_HWC_sha256'] and pixelsha(c32)==p['central32_HWC_sha256'] and raw.crop((32,32,64,64)).tobytes()==c32.tobytes()
        rawdisp=raw.resize((384,384),Image.Resampling.NEAREST);cdisp=c32.resize((256,256),Image.Resampling.NEAREST)
        assert old.crop((1120,202,1504,586)).tobytes()==rawdisp.tobytes()
        assert old.crop((1595,202,1851,458)).tobytes()==cdisp.tobytes()
        board=Image.new('RGB',(2080,1490),BG);board.paste(old.crop((0,0,2080,840)),(0,0));dr=ImageDraw.Draw(board)
        # Restore complete validity display from the unchanged original PNG; v2 caption patch overlapped its last12 display rows.
        board.paste(valid.resize((192,192),Image.Resampling.NEAREST).convert('RGB'),(1120,660))
        text(dr,(1885,160),'center2 ×64',20,maxw=190);board.paste(raw.crop((47,47,49,49)).resize((128,128),Image.Resampling.NEAREST),(1885,202))
        text(dr,(1885,370),'center16 ×8',20,maxw=190);board.paste(raw.crop((40,40,56,56)).resize((128,128),Image.Resampling.NEAREST),(1885,412))
        text(dr,(24,882),'V178 고정 owner 모델 예측 · 원 native 상태는 위의 기존 관찰값',29)
        for x,t,w in [(24,'모델',180),(270,'owner argmax',220),(550,'binary 진단',220),(900,'저장 raw logits [T, F, BG]',1100)]:text(dr,(x,936),t,25,maxw=w)
        predictions={}
        for j,m in enumerate(MODELS):
            n=int(r[m+'_owner_prediction']);symbol=CLASS[n];assert symbol==r[m+'_owner_T_F_BG_U']
            diagnostic='U' if n==-1 else ('D' if n==0 else 'E');assert diagnostic==r[m+'_D_E_U']
            logits=[float(r[m+'_raw_owner_'+cl+'_logit']) for cl in ['T','F','BG']]
            if r['query_available']=='True':assert all(math.isfinite(x) for x in logits) and max(range(3),key=lambda n:logits[n])==n
            else:assert n==-1
            yy=984+j*55
            for x,t,w in [(24,m,180),(270,symbol,220),(550,diagnostic,220),(900,'['+', '.join(f'{z:.6g}' if math.isfinite(z) else 'NaN' for z in logits)+']',1100)]:text(dr,(x,yy),t,28,maxw=w)
            predictions[m]={'owner_prediction':n,'owner_T_F_BG_U':symbol,'binary_diagnostic':diagnostic,'raw_owner_logits_T_F_BG':[z if math.isfinite(z) else None for z in logits]}
        text(dr,(24,1166),f"기존 binary 참고: v61={r['v61_D_E_U']} / legacy={r['legacy_D_E_U']} · baseline 입력/의미 차이 유지",25)
        text(dr,(24,1222),'T/F/BG/U는 점 소속 예측. T→D, F/BG→E 매핑은 비교 진단이며 물리적 native external 정답이 아닙니다.',25,AMBER)
        text(dr,(24,1270),'대상 주변 재료가 crop에 보이는 것과 중심 네 표본의 소속, 원 native 끝의 가시성은 별개입니다.',25,AMBER)
        text(dr,(24,1325),'원 query/L·원 crop/중앙32/context 픽셀 유지 · 원 PNG/과거 카드 보존 · 새 GT-center crop/라벨/교정 없음',24)
        text(dr,(24,1374),'과거 결과 노출된 고정19 평가 예시 · 대표성/새 human·native gold/학습 편입/모델·문턱 선택 주장 없음',24)
        text(dr,(24,1422),'현재 그림의 시각 관찰만으로 목표 차이와 전이 실패의 원인을 모두 분리하거나 확정하지 않습니다.',24)
        op=D/f'CASE_{i:04d}_CARD.png';board.save(op)
        saved=Image.open(op).convert('RGB')
        checks={'full_context_display_exact':saved.crop((24,202,1074,792)).tobytes()==old.crop((24,202,1074,792)).tobytes(),'raw96_nearest_display_exact':saved.crop((1120,202,1504,586)).tobytes()==rawdisp.tobytes(),'central32_nearest_display_exact':saved.crop((1595,202,1851,458)).tobytes()==cdisp.tobytes(),'valid96_nearest_display_exact':saved.crop((1120,660,1312,852)).tobytes()==valid.resize((192,192),Image.Resampling.NEAREST).convert('RGB').tobytes(),'center2_exact_subset':saved.crop((1885,202,2013,330)).tobytes()==raw.crop((47,47,49,49)).resize((128,128),Image.Resampling.NEAREST).tobytes(),'center16_exact_subset':saved.crop((1885,412,2013,540)).tobytes()==raw.crop((40,40,56,56)).resize((128,128),Image.Resampling.NEAREST).tobytes()}
        assert all(checks.values())
        records.append({'global_index':i,'id':r['id'],'native_corner':k,'material':p['material'],'original_query_xy':p['original_query_xy'],'original_L':p['original_L'],'query_available':r['query_available']=='True','original_human_state':r['human_state'],'initial_reference_error_px':float(r['initial_reference_error_px']) if r['initial_reference_error_px'] else None,'annotation_coordinate_source':r['annotation_coordinate_source'],'physical_3px_gold':False,'predictions':predictions,'v61_binary':r['v61_D_E_U'],'legacy_binary':r['legacy_D_E_U'],'original_raw96':p['raw96'],'original_central32':p['central32'],'original_valid96':p['valid96'],'original_card':p['card'],'caption_v2_card':v2cards[i],'new_card':bind(op),'pixel_parity':checks,'original_selection_memberships':p['memberships'],'new_labels':False,'training_admission':False})
        content.append(f'<article id="g{i}"><h2>g{i} · {html.escape(r["id"])} · native{k}</h2><p>'+html.escape(' | '.join(f'{m}: owner {predictions[m]["owner_T_F_BG_U"]}, binary {predictions[m]["binary_diagnostic"]}' for m in MODELS))+'</p><img src="'+op.name+'"><p>'+f'<a href="{html.escape(rel(v2cards[i]["path"]))}">보존한 V169 caption-v2 전체 카드</a> · <a href="{html.escape(rel(p["raw96"]["path"]))}">원 clean96</a> · <a href="{html.escape(rel(p["central32"]["path"]))}">원 중앙32</a>'+'</p></article>')
    rp=D/'ALL19_DISPLAY_RECORDS.json';put(rp,records);index=D/'index.html';index.write_text('\n'.join(content)+'</html>')
    for i in ids:
        for key in ['raw96','central32','valid96','card']:check(panels[i][key])
        check(v2cards[i])
    for b in input_bindings.values():check(b)
    complete={'completed':True,'PASS':True,'UTC':datetime.now(timezone.utc).isoformat(),'code':bind(Path(__file__)),'display_rules':bind(D/'DISPLAY_RULES_AND_INPUT_BINDINGS.json'),'inputs':input_bindings,'fixed19_indices':ids,'selected_cases':19,'all_original_selection_and_metadata_exact':True,'original_query_crop_or_label_changes':0,'pixel_parity_all19_all6_ROIs':True,'raw96_central32_decoded_hashes_all19_match_original':True,'original_PNG_and_source_metadata_hashes_unchanged':True,'records':bind(rp),'cards':[r['new_card'] for r in records],'index':bind(index),'actual_visual_inspection':'pending','HTML_browser_rendered':False,'source_RGB_cache_feature_weight_NPZ_or_model_reads':False,'V112_access':False,'new_forward_fit_threshold_or_model_selection':False,'new_labels_or_training_admission':False,'selected_model':None,'adopted':False,'fully_solved':False}
    put(D/'COMPLETE.json',complete);print(json.dumps({'completed':True,'cases':19,'COMPLETE':bind(D/'COMPLETE.json'),'index':bind(index),'records':bind(rp)},ensure_ascii=False))

if __name__=='__main__':main()
