"""Close saved gallery integrity and manually viewed fixed-representative QA; no model calls."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
import collections,datetime,hashlib,json
from PIL import Image
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;O=V/'source_validation_full154_visual_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,r):
 with Path(p).open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
rows=lines(O/'ALL154_GALLERY_MANIFEST.jsonl');census=lines(O/'ALL360_SOURCE_VALIDATION_CENSUS.jsonl');selection=read(O/'DIAGNOSTIC_SELECTION_BEFORE_RGB.json');required=read(O/'ACTUAL_VIEW_QA_REQUIRED.json')
assert len(rows)==len({r['global_index'] for r in rows})==154 and len(census)==360
assert collections.Counter(r['fixed_weak_truth'] for r in rows)=={'D':102,'E':52}
assert sum(not r['query_available'] for r in rows)==9 and all(r['fixed_weak_truth']=='D' for r in rows if not r['query_available'])
assert sum(r['fixed_weak_truth'] is None for r in census)==206
assert [r['global_index'] for r in rows]==selection['all154_global_indices']
checks={};seen={};pngs=set()
for r in rows:
 for b in [r['card'],r['raw96'],r['central32'],r['valid96'],*r['assets'].values()]:
  if not isinstance(b,dict) or set(b)!= {'path','sha256'}:continue
  if b['path'] not in seen:assert sha(b['path'])==b['sha256'];seen[b['path']]=b['sha256']
  if b['path'].endswith('.png'):pngs.add(b['path'])
 with Image.open(r['card']['path']) as im:assert im.size==(2200,1410);im.verify()
 with Image.open(r['raw96']['path']) as im:a=np.asarray(im.convert('RGB'))
 with Image.open(r['central32']['path']) as im:b=np.asarray(im.convert('RGB'))
 with Image.open(r['valid96']['path']) as im:vm=np.asarray(im)>0
 assert a.shape==(96,96,3) and b.shape==(32,32,3) and np.array_equal(b,a[32:64,32:64])
 assert ash(a)==r['raw96_array_sha256'] and ash(b)==r['central32_array_sha256'] and ash(vm)==r['valid96_array_sha256']
checks['all154_card_and_saved_asset_bindings_exact']=True
checks['all154_raw96_central32_valid_pixels_match_bound_arrays']=True
checks['all154_PNG_dimensions_and_decode_integrity']=True
class Links(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('href','src') and v:self.refs.append(v)
localrefs=[]
for p in [O/'index.html',O/'diagnostic_representatives.html',O/'all_changed_appendix.html']:
 l=Links();l.feed(p.read_text())
 for s in l.refs:
  assert not s.startswith(('http:','https:','//'))
  target=p.parent/unquote(s.split('#')[0]);assert target.exists(),s;localrefs.append(str(target))
checks['all_HTML_local_links_and_images_exist']=True
arms=('C45_REPLAY_C43_OWNER253_NATIVE0','C46_PART_NATIVE92_OWNER253_NATIVE0')
cm={arm:[[sum(r['fixed_weak_truth']==t and r['measurements'][arm]['prediction']==pred for r in rows) for pred in ('D','E','U')] for t in ('D','E')] for arm in arms}
assert cm[arms[0]]==[[93,0,9],[50,2,0]] and cm[arms[1]]==[[93,0,9],[49,3,0]]
changed=[r['global_index'] for r in rows if r['measurements'][arms[0]]['prediction']!=r['measurements'][arms[1]]['prediction']]
assert changed==selection['all_changed_known_global_indices']==[844]
assert [r['global_index'] for r in required['cards']]==[592,593,596,668,844]
# Exactly these five original cards were displayed via tools.view_image by reviewer A.
notes={
592:'두 조건 TN. 원래 q가 영상 하단에 있고 raw96/중앙32는 바닥·얇은 목재 경계와 검은 padding을 함께 담는다. center4 2/4, nearest false여도 원래 continuous input-valid이므로 U로 새로 바꾸지 않는다. native D 약역할을 literal 중심 목재 소유/좌표정확도 gold로 등치하지 않는다.',
593:'두 조건 inputU. q는 영상 아래이며 중앙32가 검은 padding이다. raw pE가 약0.238→0.771로 바뀌어도 최종 U를 유지한다. padding은 자연 배경의 소유 정답이 아니며 inputU9 행을 제외하거나 confidence로 복구하지 않는다.',
596:'두 조건 FN. fullRGB에는 팔레트 위 갈색 짜임 바구니와 목재 상자가 있고 q crop은 바구니의 짜임/테두리 및 회색 배경에 걸친다. 이 자연 물체들은 선명한 노란/붉은 YCB 합성물보다 팔레트와 색·질감이 가깝고 주변 구조가 연속된다. 팔레트의 의도한 native 끝은 중심 crop만으로 직접 보이지 않아 selected part 대응이 모호하다. 원래 E 약역할을 유지하며 새 gold나 라벨 교정은 하지 않는다.',
668:'두 조건 TP. g596과 같은 갈색 바구니/목재 상자 유형이며 q는 바구니 측면 근처에 있다. 비슷한 재질·가림 장면에서 TP와 FN이 함께 존재하므로 재질 유사성 하나가 실패 원인이라고 확정할 수 없다. full target context와 literal q crop의 역할을 분리해 볼 필요가 있으나 이 관찰로 모델/문턱/평가행을 선택하지 않는다.',
844:'변경 전수의 유일한 행이며 원래 E 약역할. C45 FN(logit−0.5314556,pE0.3701774)→C46 TP(logit0.4194601,pE0.6033540). 원래 q와 중앙 crop은 갈색 바구니의 위/안쪽 모서리 부근을 담으며 실제 팔레트 코너 좌표를 증명하지 않는다. 한 경계 통과 사례이지 native 가림 문제 해결이나 폭넓은 일반화 성공이 아니다.'}
viewed=[]
by={r['global_index']:r for r in rows}
for i in [592,593,596,668,844]:
 r=by[i];assert sha(r['card']['path'])==r['card']['sha256']
 viewed.append(dict(global_index=i,query_id=r['query_id'],image=r['card'],tool='tools.view_image',actually_viewed=True,representative_categories=r['fixed_representative_categories'],allchanged_appendix=r['postresult_allchanged_appendix'],observation_KO=notes[i],assistant_weak_observation_only=True,new_native_owner_quality_label=False,coordinate_error_gold=False,training_admitted=False))
checks['fixed_representatives_and_allchanged_original_rows_actually_viewed']=True
protocol=read(H/'PROTOCOL_BEFORE_POSTFIT_SOURCE_VALIDATION_BODIES.json')
for key in ('code','trainer_protocol','parent_renderer','C9_pure_math','source_labels','source_inputfacts','source_frames','saved96_packets'):
 b=protocol[key];assert sha(b['path'])==b['sha256']
checks['original_bound_inputs_labels_code_protocol_unchanged']=True
receipt=dict(schema='V166_ACTUAL_ALL154_GALLERY_AND_FIXED_DIAGNOSTIC_VISUAL_QA_A',completed=True,PASS=all(checks.values()),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),renderer_complete=bind(O/'COMPLETE.json'),renderer_terminal=dict(session_id=65983,exit_code=0,actual_render_runs=1,automatic_retry=False),code=bind(H/'render_all154_saved_MR_C45_C46.py'),protocol=bind(H/'PROTOCOL_BEFORE_POSTFIT_SOURCE_VALIDATION_BODIES.json'),qa_code=bind(__file__),manifest=bind(O/'ALL154_GALLERY_MANIFEST.jsonl'),full360_census=bind(O/'ALL360_SOURCE_VALIDATION_CENSUS.jsonl'),selection_before_RGB=bind(O/'DIAGNOSTIC_SELECTION_BEFORE_RGB.json'),QA_required=bind(O/'ACTUAL_VIEW_QA_REQUIRED.json'),checks=checks,counts=dict(all360=360,known154=154,D102_E52=[102,52],known_inputU9_D=9,input_valid145_D93_E52=[93,52],truth_outside_DE=206,unique_asset_bindings_verified=len(seen),PNG_assets_hashed=len(pngs),local_HTML_refs_checked=len(localrefs),actual_view_image_tool_calls=5,distinct_cards_actually_viewed=5,other149_cards_not_claimed_actually_viewed=149),CM_rows_D_E_columns_D_E_U=cm,allchanged_known_global_indices=changed,empty_FP_strata_preserved_without_replacement=True,actually_viewed_images=viewed,conclusions_KO=['Full154에서 E2/52→3/52이며 D93/102와 D9U는 같다. 고정 대표와 변경 전수1행을 모두 실제로 확인했다.','V164의 같은 리뷰어 합성개입 관찰에서는 색/경계/그림자/범위 cue가 강했다. 여기 자연 바구니·목재 상자는 팔레트와 재질이 비슷하고 선택 native 부위와 q crop이 다를 수 있다. 이는 검토한 소수 사례의 약한 관찰이며 모델 실패 인과/전체52E의 대표성/gold가 아니다.','갈색 바구니 장면의 FN과 TP가 모두 존재한다. 합성 native92의 TRAIN 응답을 실제 held 가림 해결과 동일시하지 않는다.','새 native/owner/quality 라벨, q 보정, 평가 필터, 문턱/모델 선택 또는 추가 학습 편입을 만들지 않았다.'],prior_train_intervention_observation=bind(P/'query_visibility_native_part_counterfactual_20261024_v164/independent_part_intervention_visual_a/COMPLETE.json'),HTML_browser_rendered=False,new_model_feature_forward_fit_checkpoint_or_private_current_V112_PnP_calls=0,training_admitted=False,fully_solved=False)
write(O/'ACTUAL_VISUAL_QA_COMPLETE.json',receipt)
print(json.dumps(dict(PASS=True,receipt=bind(O/'ACTUAL_VISUAL_QA_COMPLETE.json'),CM=cm,actual_views=5)))
