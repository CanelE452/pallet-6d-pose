"""Display-only exact fixed4 join to completed C14 scores; no model or packet load."""
from pathlib import Path
import csv, hashlib, json
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont

P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OLD=P/'query_visibility_success_failure_visual_review_20261023_v117'
C14=P/'query_visibility_consistent_query_context_20261023_v119'
OUT=Path(__file__).resolve().parent
EVAL=C14/'private_evaluation/C14_FROZEN_CURRENT223'
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
BOLD='/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
INK='#182d3b'; MUTED='#4a6171'; GREEN='#24654a'; RED='#a43e35'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path':str(p),'sha256':sha(p)}
def check(b):
    assert sha(b['path'])==b['sha256'], b['path']
    return Path(b['path'])
def txt(d,xy,s,n=20,color=INK,bold=False):
    d.text(xy,s,font=ImageFont.truetype(BOLD if bold else FONT,n),fill=color)

assert not (OUT/'C14_FIXED4_SUCCESS_FAILURE.png').exists(), 'Do not overwrite completed rendering'
lockp=OLD/'SELECTION_LOCK_BEFORE_PIXELS.json'
receiptp=OLD/'RENDER_RECEIPTS.json'
lock=json.loads(lockp.read_text()); prior=json.loads(receiptp.read_text())
scorep=EVAL/'HEAD_ONLY_SCORES.csv'
with scorep.open() as f: scores=list(csv.DictReader(f))
assert len(scores)==1784
assert len({(r['id'],int(r['native_corner'])) for r in scores})==1784
inferp=EVAL/'INFERENCE_SEAL.json'; evalp=EVAL/'evaluation/EVALUATION_SEAL.json'
assert sha(inferp)=='68c9dad674e1004ed3dd8347551ebef271168f48ead83a6f5f7ab752c82f5a45'
assert sha(evalp)=='a9a30f52d37f64bfaba26ed9910c33ed155bae92f8998b0841634c0d039e3954'
metricp=EVAL/'evaluation/CURRENT_PRIMARY_RESULTS.json'
metrics=json.loads(metricp.read_text())['tables']['metrics']
metric_rows=[]
for group,expected in [('WOOD_PRIMARY105',(599,616,9,29)),('SQUARE',(531,531,6,61))]:
    rows=[r for r in metrics if r['cohort']=='paired_same_query_RGB' and r['model']=='V119_C14_RAW' and r['task']=='D_vs_E' and (r['group']==group or (group=='SQUARE' and r['group'].startswith('SQUARE')))]
    assert len(rows)==1,(group,rows)
    r=rows[0]
    assert (int(r['TP']),int(r['visible_support']),int(r['TN']),int(r['occluded_support']))==expected
    cm=[[int(r['TP']),int(r['FN_decided']),int(r['U_visible'])],[int(r['FP']),int(r['TN']),int(r['U_occluded'])]]
    metric_rows.append({'group':r['group'],'model':'V119_C14_RAW','human_D_N':int(r['visible_support']),'human_E_N':int(r['occluded_support']),'CM_rows_D_E_cols_D_E_U':cm,'D_correct':int(r['TP']),'E_correct':int(r['TN']),'D_recall':float(r['visible_recall_all_GT']),'E_recall':float(r['occluded_recall_all_GT']),'D_precision':float(r['visible_precision']),'E_precision':float(r['occluded_precision']),'U_rate':float(r['abstention_rate'])})

chosen=[(0,200),(3,1416),(5,325),(8,1021)]
labels=[
 ('보임 성공 · 목재','노출된 테두리·모서리 구조가 보이는 사례.','보임 판단 일치가 초기 q의 좌표 정확성까지 입증하지는 않음.'),
 ('외부 가림 성공 · 정사각형','콘이 질의 부근을 차지하는 사례.','외부 가림 r은 기존 추정 좌표이며, 좌표 gold가 아님.'),
 ('외부 가림 실패 · 목재','콘과 남아 있는 목재가 함께 보이는데 D로 응답.','주변 재질의 존재와 기대한 구조 단서의 가시성은 다름.'),
 ('외부 가림 실패 · 정사각형','콘이 crop 중심을 크게 차지해도 D로 응답.','이 그림만으로 context·재질·위치의 단일 원인을 확정할 수 없음.'),
]
cases=[];cards=[]
for n,(ix,ordinal) in enumerate(chosen):
    item=lock['selected'][ix]; rec=prior['cases'][ix]
    assert item['case_index']==rec['case_index']==ix
    assert item['original_native_row']==rec['original_native_row']==ordinal
    r=scores[ordinal]
    assert r['id']==item['id']==rec['id'] and int(r['native_corner'])==item['native_slot']==rec['native_slot']
    q=[float(r['query_x']),float(r['query_y'])]; L=float(r['roi_side'])
    assert q==item['query_xy'] and L==item['original_L']
    z=float(r['raw_query_aux_hidden_logit']); available=r['valid']=='True'
    pred='U' if not available else ('D' if z<0 else 'E')
    assert pred==r['authoritative_decision']
    human='D' if item['human_state']=='DIRECT_VISIBLE' else 'E'
    correct=pred==human
    rawp=check(rec['raw_png']); markp=check(rec['marked_png']); fullp=check(item['full_RGB_binding'])
    with Image.open(rawp) as x:raw=x.convert('RGB')
    with Image.open(markp) as x:marked=x.convert('RGB')
    with Image.open(fullp) as x:full=x.convert('RGB')
    assert raw.size==marked.size==(96,96)
    assert hashlib.sha256(full.tobytes()).hexdigest()==item['decoded_RGB_sha256_expected']
    title,obs,limit=labels[n]
    card=Image.new('RGB',(800,605),'white');d=ImageDraw.Draw(card)
    color=GREEN if correct else RED
    d.rectangle((0,0,799,7),fill=color)
    txt(d,(20,22),title,26,color,True)
    txt(d,(20,65),f"사람 {'보임(D)' if human=='D' else '외부 가림(E)'}  /  C14 {'보임(D)' if pred=='D' else '외부 가림(E)'}",22,INK,True)
    txt(d,(20,101),f"원행 {ordinal} · 슬롯 {item['native_slot']} · 원 L={L:.2f} native px · zE={z:.4f}",17,MUTED)
    txt(d,(20,139),'전체 원 RGB · 무표시',18)
    txt(d,(356,139),'저장 RGB96 · 무표시',18)
    txt(d,(568,139),'같은 crop · 표시 별도',18)
    card.paste(full.resize((320,240),Image.Resampling.BILINEAR),(20,170))
    card.paste(raw.resize((192,192),Image.Resampling.NEAREST),(356,170))
    card.paste(marked.resize((192,192),Image.Resampling.NEAREST),(568,170))
    txt(d,(356,380),'노랑 + : 원 q / 하늘 원 : 기존 r',17,MUTED)
    txt(d,(20,438),obs,21,INK,True)
    txt(d,(20,475),limit,18,MUTED)
    txt(d,(20,515),f"ID: {item['id']}",17,MUTED)
    txt(d,(20,548),'표시는 설명용 복사본에만 추가 · 원 q/L·사람 라벨·저장 출력 유지',17,MUTED)
    txt(d,(20,577),'local RGB96는 입력 일부; C14는 별도 frozen context/features도 사용',16,MUTED)
    cardp=OUT/f'CASE_{ix:02d}_C14.png'; card.save(cardp);cards.append(card)
    cases.append({'case_index_in_original9':ix,'original_native_row':ordinal,'id':r['id'],'native_slot':int(r['native_corner']),'human_state':item['human_state'],'q':q,'L_native_pixels':L,'query_available':available,'C14_raw_zE':z,'C14_prediction':pred,'correct_against_existing_human_state':correct,'reference_xy':item['reference_xy'],'reference_source':item['reference_annotation_coordinate_source'],'E_reference_is_not_coordinate_gold':item['E_reference_is_occluded_estimated_not_coordinate_gold'],'original_selection_rule':item['rule'],'full_RGB':bind(fullp),'raw_RGB96_png':bind(rawp),'marked_RGB96_png':bind(markp),'new_card':bind(cardp),'visual_observation_KO':obs,'interpretation_limit_KO':limit})

board=Image.new('RGB',(1600,1435),'#edf2f5');d=ImageDraw.Draw(board)
txt(d,(24,17),'현재 C14 · 같은 고정 4사례의 실제 성공과 실패',31,INK,True)
txt(d,(24,65),'전체 paired cohort: 목재 D 599/616 · E 9/29  |  정사각형 D 531/531 · E 6/61',23)
txt(d,(24,105),'고정 C14 step1200 저장값 · 위 성공 / 아래 외부 가림 실패 · 재추론·좌표 교정 없음',20,MUTED)
for n,card in enumerate(cards):board.paste(card,((n%2)*800,155+(n//2)*605))
txt(d,(24,1380),'고정 4개는 빈도 표본이 아님. E reference는 추정값; 가시성 판정과 좌표 검증·기하 self 경로는 별개.',19,MUTED)
boardp=OUT/'C14_FIXED4_SUCCESS_FAILURE.png';board.save(boardp)
result={'completed':True,'scope':'Existing fixed4 display joined to completed saved C14 current output; no new inference or selection.','created_UTC':datetime.now(timezone.utc).isoformat(),'model':'C14','source_selected_update':1200,'original_native_row_order':[x[1] for x in chosen],'original_case_indices':[x[0] for x in chosen],'all4_exact_row_id_slot_q_L_join':True,'selection_lock':bind(lockp),'prior_render_receipts':bind(receiptp),'C14_saved_scores':bind(scorep),'C14_inference_seal':bind(inferp),'C14_evaluation_seal':bind(evalp),'C14_metrics_source':bind(metricp),'metrics':metric_rows,'cases':cases,'board':bind(boardp),'renderer':bind(Path(__file__)),'limitations_KO':['4개 사례는 기존 목적별 고정 표본이며 전체 빈도·일반화 성능을 추정하지 않는다.','D/E는 기존 사람 가시성 라벨과의 비교이고, E reference는 가려진 지점의 추정값이다.','local crop에 팔레트 재질이 남아 있어도 기대 부품이 보인다는 뜻은 아니다.','C14는 local RGB96 외 별도 context/features도 사용한다. 그림만으로 단일 실패 원인을 증명하지 않는다.','현재 자료는 반복 개발에 노출되었으며 fresh blind test가 아니다. 이 작업에서 학습 설계·문턱·모델·checkpoint를 선택하지 않았다.','기하학적 self filter는 별도 경로이고, 이 보드는 C14 raw D/E 응답이다.'],'new_model_encoder_fit_initializer_PnP_calls':0,'original_q_L_labels_outputs_changed':False,'fully_solved':False,'actual_board_viewed_by_C_at_write':False,'root_render_QA_complete':False}
(OUT/'C14_FIXED4_MAPPING_AND_METRICS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'completed':True,'board':bind(boardp),'mapping':bind(OUT/'C14_FIXED4_MAPPING_AND_METRICS.json'),'decisions':[r['C14_prediction'] for r in cases]},ensure_ascii=False))
