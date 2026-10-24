"""V171 final Korean readout using existing selection/complete/visual receipt JSON only."""
from pathlib import Path
import hashlib,html,json
from html.parser import HTMLParser

V=Path(__file__).resolve().parent
SEL=V/'WITNESS_SELECTION_BEFORE_RGB.json';COMPLETE=V/'forensic_cards_v1/COMPLETE.json';VISUAL=V/'forensic_cards_v1/ACTUAL_VISUAL_OBSERVATIONS.json'
def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
s=read(SEL);c=read(COMPLETE);v=read(VISUAL)
assert s['PASS'] and c['PASS'] and v['PASS'] and v['actually_viewed_all_3_cards']
assert s['total_memberships']==4 and s['deduplicated_rows']==c['unique_cards']==3
assert v['complete']==bind(COMPLETE) and v['selection']==bind(SEL)
byid={x['global_index']:x for x in s['unique_witnesses']};observations={x['global_index']:x for x in v['viewed_images']}
cards=[]
for proof in c['cards']:
    i=proof['global_index'];row=byid[i];gt=row['source_GT_metadata'];obs=observations[i]
    assert proof['card']==obs['image'] and proof['frozen_crop_reconstruction_byte_exact']
    cards.append(dict(global_index=i,query_id=row['query_id'],semantic_D_E=gt['teacher_state'],native_slot=row['native_slot'],canonical_index=gt['canonical_index'],
        original_query_xy=row['query_xy'],original_L=row['L'],nominal_reference_xy=gt['reference_xy'],q_reference_error_px=gt['error_px'],correspondence_boundary=gt['correspondence_boundary'],
        both_fixed_head_raw_E_logits=row['both_fixed_head_raw_E_logits'],bound_memberships=row['bound_memberships'],image=proof['card'],literal_q_neighborhood_observation_KO=obs['observation_KO'],
        reconstruction_byte_exact=True,actual_viewed=True,no_new_label_or_gold=True))
results=dict(completed=True,phase='V171',scope='V170 공개 합성 OLD6144 constraint-bound 전체 tie3행의 source-only forensic 시각 진단',
    conclusion_KO='저장 crop 손상이나 q/L 재중심화 없이 세 행의 원본 픽셀이 정확히 재구성됐다. 원 semantic 역할·nominal reference·literal q 주변 appearance를 구분할 필요가 보이지만, 이 RGB3행만으로 instance/물리 native 가림/라벨 오류나 전체 실패 원인을 확정할 수 없다. 전체 목표는 미해결이다.',
    counts=dict(bound_memberships=4,unique_witnesses=3,tie_rows_replaced=0,unknown_selection=0,reconstruction_PASS=3,actually_viewed_cards=3),cards=cards,
    provenance=dict(selection=bind(SEL),input_path_readiness=bind(V/'INPUT_PATH_READINESS.json'),original_render_complete=bind(COMPLETE),actual_three_view_receipt=bind(VISUAL),render_code=c['code'],render_protocol=c['protocol'],invented_checks=c['tests']),
    temporal_record_note='COMPLETE.actual_visual_inspection=pending는 renderer 실행 당시 기록을 그대로 보존했다. 이후 ACTUAL_VISUAL_OBSERVATIONS.json가 tools.view_image 실제3뷰 완료를 별도 hash binding한다.',
    limitations_KO=['4개 고정 bound membership의 모든 tie를 dedup한3행이다. 좋은 예시로 교체하거나 수를 잘라내지 않았으며 대표성 표본이 아니다.',
        '세 행은 모두 correspondence_boundary=true이고 q-reference 오차13.59~16.91px이다. reference와 원 q는 표시용 복사본에서만 분리해 보였고 clean RGB96/central32는 바꾸지 않았다.',
        '원 semantic D/E와 literal q 중심 재료/owner는 같은 명제라고 가정하지 않는다. RGB만으로 같은 재질의 instance, 물리 native 지점 가림, teacher 오류를 확정하지 않는다.',
        'V170의 고정 점수·기준하 global threshold 교집합 EMPTY 판정은 유지한다. 이3행의 관찰만으로 모든 실패 원인 또는 미래 모델의 성공/불가능성을 판정하지 않는다.'],
    new_training_or_forward=0,new_threshold_or_model_selection=False,new_label_correction=0,qL_or_reference_recenter=0,new_training_admission=0,new_source_GTmask_model_private_current_V169_V112_bodies_read_for_this_report=0,
    selected_model=None,adopted=False,fully_solved=False,goal_complete=False,HTML_browser_rendered=False)
rp=V/'ROOT_RESULTS_KO.json';write(rp,results)
def e(x):return html.escape(str(x))
sections=[]
for row in cards:
    rel=Path(row['image']['path']).relative_to(V)
    scores=' / '.join(a.split('_')[0]+': '+f'{z:.9f}' for a,z in row['both_fixed_head_raw_E_logits'].items())
    sections.append('<article><h2>'+e(str(row['global_index'])+' · '+row['query_id'])+'</h2><p><strong>원 semantic '+e(row['semantic_D_E'])+'</strong> · native '+e(row['native_slot'])+' / canonical '+e(row['canonical_index'])+' · q-reference '+f"{row['q_reference_error_px']:.6f}"+'px · boundary=true<br>고정 raw E-logit '+e(scores)+'</p><p>'+e(row['literal_q_neighborhood_observation_KO'])+'</p><a href="'+e(rel)+'"><img src="'+e(rel)+'" alt="'+e(row['query_id'])+' 원본 scene와 clean 저장 crop forensic 카드"></a></article>')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V171 공개 constraint witness crop 진단</title><style>body{font:16px/1.65 sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#192536}h1{font-size:27px}h2{font-size:21px}img{display:block;max-width:100%;height:auto;border:1px solid #c5cdd7}article{margin:32px 0 44px}.note{padding:16px;background:#edf3fa}a{color:#155a8d}small{color:#46556a}</style><h1>V171 · 공개 합성 constraint witness crop 진단</h1>
<p class="note"><strong>실제 원본 재구성 3/3 PASS, 카드 실제 확인 3/3 완료.</strong> 전체 목표는 미해결입니다. 새 학습·좌표/라벨 교정·학습 편입은 없습니다.</p>
<p>V170에서 SAME_TARGET_MATERIAL D/E 조건의 상충 경계를 만든 각 head의 모든 동일점수 tie를 그대로 보존했습니다. 고정 bound membership4개를 원본3행으로 dedup했으며, 선택 상한·교체·좋은 사례 선별은 없습니다. 극단 constraint witness이므로 전체 데이터의 대표성 표본이 아닙니다.</p>
<p>원 PNG/chunk SHA, decoded RGB·saved96·valid96 array SHA를 검증했습니다. 원 q/L로 frozen crop 연산을 재실행한 결과와 실제 저장 RGB96/valid96가 세 행 모두 바이트 단위로 일치했습니다. central32는 저장96의 [32:64,32:64] 정확한 slice입니다. q(빨간 원)와 nominal GT reference(청록 십자)는 fullscene 표시용 복사본에만 그렸고 raw96/c32는 clean 상태입니다.</p>
<p><strong>원 semantic D/E, q-reference 차이, literal q 주변 appearance는 별도입니다.</strong> 세 행 모두 원 metadata의 boundary=true이며 q-reference 오차는13.59~16.91px입니다. 아래는 assistant의 RGB 관찰이지 새 owner/native 가림 gold가 아닙니다. RGB만으로 같은 재질의 instance, 물리 native 가림 또는 라벨 오류를 확정할 수 없습니다. V170의 고정 정의하 수학적 문턱 불가능성은 그대로이며, 세 관찰로 전체 원인이나 미래 모델의 가능성을 확정하지 않습니다.</p>
'''+''.join(sections)+'''<p>실행 당시 COMPLETE의 actual_visual_inspection=pending 필드는 보존했습니다. 이후 별도 실제3뷰 receipt가 완료를 바인딩합니다. 이 페이지 작성에서는 기존 JSON만 읽었고 새 source/GT mask/model/private 본문은 읽지 않았습니다.</p><p><a href="ROOT_RESULTS_KO.json">결과 JSON</a> · <a href="WITNESS_SELECTION_BEFORE_RGB.json">변경 없는 선택</a> · <a href="forensic_cards_v1/COMPLETE.json">원본 재구성 완료 기록</a> · <a href="forensic_cards_v1/ACTUAL_VISUAL_OBSERVATIONS.json">실제3뷰 관찰 receipt</a></p><small>PNG3장은 tools.view_image로 실제 확인했습니다. HTML은 소스와 링크만 점검했으며 브라우저 렌더링은 수행하지 않았습니다. 모델·문턱·데이터 정의 선택, 새 학습·교정·편입·성공 판정 없음.</small></html>'''
hp=V/'ROOT_READOUT_KO.html'
with hp.open('x') as f:f.write(page+'\n')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):self.refs.extend(val for k,val in attrs if k in ('href','src'))
parser=Links();parser.feed(page);assert all((V/x).is_file() for x in parser.refs)
phase=dict(completed=True,phase='V171',results=bind(rp),readout=bind(hp),selection=bind(SEL),original_execution_complete=bind(COMPLETE),actual_three_view_completion=bind(VISUAL),
    original_pending_execution_record_preserved=True,counts=results['counts'],source_only=True,representative_sample_claimed=False,new_source_bodies_read_for_final_report=0,
    reconstruction3of3=True,actually_viewed3of3=True,HTML_browser_rendered=False,local_links_and_thumbnail_refs_checked=len(parser.refs),
    label_or_coordinate_correction=False,new_training_or_admission=False,threshold_or_model_selection=False,selected_model=None,adopted=False,fully_solved=False,goal_complete=False)
pp=V/'ROOT_PHASE_RESULT.json';write(pp,phase)
print(json.dumps(dict(results=bind(rp),readout=bind(hp),phase=bind(pp),local_refs_checked=len(parser.refs))),flush=True)
