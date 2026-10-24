"""Report-only rendering from fixed V117 cases and saved C13 scores. No models."""
from pathlib import Path
import csv, hashlib, html, json
from PIL import Image, ImageDraw, ImageFont

P = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
OLD = P / 'query_visibility_success_failure_visual_review_20261023_v117'
V = P / 'query_visibility_same_material_query_head_20261023_v118'
OUT = Path(__file__).parent
FONT = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
BOLD = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
ink, muted = '#172c3a', '#465e6f'
green, red = '#24654a', '#ad3c32'

def digest(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def font(n, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, n)

def text(draw, xy, value, n=22, color=ink, bold=False):
    draw.text(xy, value, font=font(n,bold), fill=color)

lock = json.loads((OLD/'SELECTION_LOCK_BEFORE_PIXELS.json').read_text())
scores_path = V/'private_evaluation/C13_FROZEN_CURRENT223/HEAD_ONLY_SCORES.csv'
with scores_path.open() as f:
    scores = {(r['id'], int(r['native_corner'])): r for r in csv.DictReader(f)}
assert len(scores) == 1784
joined = []
cards = []
labels = [
    ('보임 성공 · 목재', '노출된 모서리 구조를 보임으로 유지'),
    ('보임 성공 · 정사각형', '근처에 콘이 있어도 해당 점이 가려진 것은 아님'),
    ('외부 가림 성공 · 목재', '질의 부근을 차지한 콘을 가림으로 판단'),
    ('외부 가림 성공 · 정사각형', '질의 부근을 차지한 콘을 가림으로 판단'),
    ('보임 개선 · 야간 목재', '작고 흐린 입력에서도 기존 사람 보임 라벨과 일치'),
    ('외부 가림 실패 · 목재', '콘과 남아 있는 목재가 함께 보이는데 보임으로 응답'),
    ('외부 가림 실패 · 정사각형', '콘과 초록 테두리가 섞인 경계에서 보임으로 응답'),
    ('외부 가림 실패 · 목재', 'C11은 가림, C12·C13은 보임'),
    ('외부 가림 실패 · 정사각형', '콘이 중심을 크게 차지해도 보임으로 응답'),
]

for item in lock['selected']:
    i = item['case_index']
    r = scores[(item['id'], item['native_slot'])]
    assert (float(r['query_x']),float(r['query_y'])) == tuple(item['query_xy'])
    assert float(r['roi_side']) == item['original_L']
    assert r['authoritative_decision'] == item['decisions']['C12']
    human = '보임' if item['human_state'] == 'DIRECT_VISIBLE' else '외부 가림'
    pred = {'D':'보임','E':'가림','U':'보류'}[r['authoritative_decision']]
    correct = (human == '보임' and pred == '보임') or (human == '외부 가림' and pred == '가림')
    row = {
        'case_index': i, 'original_native_row': item['original_native_row'],
        'id': item['id'], 'native_slot': item['native_slot'],
        'human_state': item['human_state'], 'C13_decision': r['authoritative_decision'],
        'C13_saved_logit': float(r['raw_query_aux_hidden_logit']),
        'q': item['query_xy'], 'L': item['original_L'],
        'reference_error_px': item['initial_reference_error_px'],
        'reference_source': item['reference_annotation_coordinate_source'],
        'E_reference_is_not_coordinate_gold': item['E_reference_is_occluded_estimated_not_coordinate_gold'],
        'historical_decisions': item['decisions'],
    }
    joined.append(row)
    card = Image.new('RGB', (800,590),'white')
    d = ImageDraw.Draw(card)
    d.rectangle((0,0,799,8), fill=green if correct else red)
    text(d,(20,24),labels[i][0],27,green if correct else red,True)
    text(d,(20,67),f'사람: {human}  /  최신 C13: {pred}',24,bold=True)
    text(d,(20,105),f"원행 {item['original_native_row']} · 슬롯 {item['native_slot']} · 원본 crop 한 변 {item['original_L']:.1f}px",18,muted)
    text(d,(20,145),'실제 저장 crop · 무표시',19)
    raw_path=OLD/f'exact_saved_crops/case_{i:02d}_RAW_RGB96.png'
    raw = Image.open(raw_path).convert('RGB')
    assert raw.size == (96,96)
    card.paste(raw.resize((288,288),Image.Resampling.NEAREST),(20,176))
    text(d,(332,145),'전체 영상 · 노랑 + 초기 위치',19)
    src_path = Path(item['full_RGB_binding']['path'])
    assert hashlib.sha256(src_path.read_bytes()).hexdigest() == item['full_RGB_binding']['sha256']
    full=Image.open(src_path).convert('RGB')
    preview=full.resize((320,240),Image.Resampling.BILINEAR)
    qx,qy=item['query_xy']; sx,sy=320/full.width,240/full.height
    px,py=qx*sx,qy*sy
    dd=ImageDraw.Draw(preview)
    half=item['original_L']/2
    dd.rectangle(((qx-half)*sx,(qy-half)*sy,(qx+half)*sx,(qy+half)*sy), outline='white',width=1)
    dd.line((px-6,py,px+6,py),fill='#ffd342',width=2)
    dd.line((px,py-6,px,py+6),fill='#ffd342',width=2)
    card.paste(preview,(332,176))
    marked=Image.open(OLD/f'exact_saved_crops/case_{i:02d}_MARKED_RGB96.png').convert('RGB')
    card.paste(marked,(676,176))
    text(d,(676,282),'q/r 표시',16)
    text(d,(332,429),'흰 사각형: 원 crop 범위',18,muted)
    text(d,(20,488),labels[i][1],21,bold=True)
    if human == '외부 가림':
        text(d,(20,527),'가려진 reference는 추정값: 좌표 정확성의 정답으로 보지 않음',17,muted)
    else:
        text(d,(20,527),'보임 성공은 초기 좌표의 정확성까지 검증한 출력은 아님',17,muted)
    text(d,(20,560),'기존 고정 9사례 · 새 모델 호출/학습/좌표 교정/PnP 0',16,muted)
    target=OUT/f'CASE_{i:02d}_C13.png'
    card.save(target)
    cards.append(card)

overview=Image.new('RGB',(1600,1280),'#edf2f5')
d=ImageDraw.Draw(overview)
text(d,(24,16),'최신 C13 · 실제 crop의 성공과 실패',32,bold=True)
text(d,(24,60),'위: 보임 성공 / 가림 성공    아래: 혼합 경계의 가림 실패 / 콘 중심의 가림 실패',22)
for n,i in enumerate([0,3,5,8]):
    overview.paste(cards[i],((n%2)*800,100+(n//2)*590))
overview.save(OUT/'LATEST_C13_SUCCESS_FAILURE.png')

results_path=V/'ROOT_RESULTS_KO.json'
results=json.loads(results_path.read_text())
table=[]
for r in results['current_primary_comparison']:
    cm=r['confusion_rows_D_E_cols_D_hidden_U']
    table.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [
        '목재' if r['group']=='WOOD_PRIMARY105' else '정사각형',r['display_model'],
        f"{cm[0][0]}/{r['human_D_N']} ({100*r['D_recall']:.1f}%)",
        f"{cm[1][1]}/{r['human_E_N']} ({100*r['E_recall']:.1f}%)",
        f"{100*r['E_precision']:.1f}%", f"{100*r['U_rate']:.1f}%"
    ])+'</tr>')
gallery=[]
for i, row in enumerate(joined):
    gallery.append(f'<figure><h3>{i+1}. {labels[i][0]}</h3><a href="CASE_{i:02d}_C13.png"><img src="CASE_{i:02d}_C13.png" alt="{labels[i][0]} · 원행{row["original_native_row"]}"></a><figcaption>{html.escape(row["id"])} · 원행 {row["original_native_row"]} · C13 저장 z={row["C13_saved_logit"]:.5f}</figcaption></figure>')
template=(OUT/'template.html').read_text()
rendered=template.replace('@@TABLE@@',''.join(table)).replace('@@GALLERY@@',''.join(gallery))
assert '@@' not in rendered
(OUT/'SAVED_C13_VISUAL_REVIEW_KO.html').write_text(rendered)
(OUT/'FIXED9_SAVED_C13_JOIN.json').write_text(json.dumps({
    'scope':'Report-only join of old fixed9 with source-frozen C13 saved outputs; no model calls/fit/private model selection.',
    'selection_lock':digest(OLD/'SELECTION_LOCK_BEFORE_PIXELS.json'),
    'scores':digest(scores_path),'metric_source':digest(results_path),
    'all9_same_decisions_as_C12':True,'selected_for_frequency_estimation':False,
    'original_q_L_labels_preserved':True,'new_model_encoder_fit_PnP':0,
    'overall_goal_solved':False,'cases':joined,
    'overview_display_case_indices':[0,3,5,8],
},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'completed':True,'cases':9,'overview':str(OUT/'LATEST_C13_SUCCESS_FAILURE.png'),
    'report':str(OUT/'SAVED_C13_VISUAL_REVIEW_KO.html'),'new_model_encoder_fit_PnP':0},ensure_ascii=False))
