"""Record completed tools.view_image observations; verify generated PNGs/links only."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
import collections,datetime,hashlib,json
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parent;V=H.parent;O=V/'fixed_current_crop_review_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,r):
 with Path(p).open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
receipt=O/'CROP_RENDER_COMPLETE.json';assert sha(receipt)=='9f47236d47af91e3667cd18589106deed2e324072cf29291b96c6f1286b219cd'
r=read(receipt);panels=r['panels'];selection=read(r['case_ids_before_RGB']['path'])
ids=[0,5,8,32,56,219,256,296,300,324,401,840,841,858,898,908,914,996,1009]
assert [p['global_index'] for p in panels]==ids and r['selected_cases']==19 and r['empty_buckets']==2
assert selection['selected_global_indices']==ids and len(selection['buckets'])==48
assert sum(len(p['memberships']) for p in panels)==46
notes={
0:('밝은 실내 목재 팔레트의 가까운 끝 블록/상판 경계가 crop에 들어오고 회색 바닥과 맞닿는다. 전체 팔레트의 연결 구조는 잘 읽히지만 확대 중앙32의 얇은 면·그림자는 부드럽다. 뚜렷한 별도 전경 물체는 이 카드에서 보이지 않는다.','기존 D와 네 조건 D를 기록할 뿐 이 crop을 정확한 native corner/좌표품질 gold로 확정하지 않는다.','BOUNDARY_SOFT_DETAIL'),
5:('같은 실내 프레임의 뒤쪽 작은 상판 끝을 바라보며 raw96 대부분은 회색 배경이고 중앙32에는 흐린 목재 끝 일부만 보인다. C45는 같은 저장 logit에서 threshold만 달라 D→E가 된다.','기존 reference오차 약1.82px의 출처 표시는 unknown이므로 manual<=3 추가 bucket과 다르다. 중심의 배경/끝 경계와 native D 상태를 등치하거나 좌표를 고치지 않는다.','SMALL_END_BACKGROUND_BOUNDARY'),
8:('실내 목재 끝 블록의 상면·수직 면·어두운 틈과 바닥 경계가 함께 보인다. 별도 물체보다 팔레트 자신의 입체 경계가 주된 crop 내용이며 다소 흐리다. 두 publicCAL 조건이 E로 바뀌어도 입력 이미지는 같다.','reference 값 약6.71px은 기존 진단일 뿐 새 좌표정확도 판정이 아니다. 이 사례로 threshold나 모델을 선택하지 않는다.','BOUNDARY_SOFT_DETAIL'),
32:('작은 목재 팔레트의 왼쪽 끝이 전체 장면과 crop에서 흐릿하고, 중앙32는 회색 바닥에서 어두운 목재 모서리로 넘어가는 경계다. 새 전경 물체를 명확히 식별할 수 있는 모습은 아니다.','기존 D에 대한 조건별 호출 차이를 보존한다. 낮은 선명도나 그림자가 모델 오판의 원인이라고 단정하지 않는다.','BLUR_SMALL_STRUCTURE'),
56:('원래 q가 이미지 하단 밖에 있어 raw96 아래와 중앙32가 검은 padding이다. validity도 중앙0/4이며 네 조건 모두 U를 유지한다. 위쪽에 남은 목재 일부는 중앙 입력의 유효성을 복구하지 않는다.','검은 padding을 자연 배경 소유나 E/D 증거로 읽지 않는다. 약498.82px 기존 reference값과 U는 별도 진단이며 새 quality label이나 q보정을 만들지 않는다.','PADDING_INPUT_U'),
219:('야외 작은 목재 받침의 하단과 아스팔트가 crop에 함께 있고 중앙은 블록 하단의 부드러운 경계 근처다. 전체 팔레트는 읽히지만 원래 q 주변만 보면 작은 갈색 면과 바닥의 구별이 제한된다.','manual_click reference오차 약0.75px인 기존 D가 publicCAL 두 조건에서 E로 변한다. 작은 fixed-reference 차이가 물리3px gold 또는 literal 중심의 정확한 재료 소유를 보증하지 않는다.','MANUAL_NEAR_BOUNDARY'),
256:('주황 cone이 팔레트 앞쪽 일부를 가리고 q crop과 중앙32에 선명한 붉은 면 및 회색 아스팔트 경계가 들어온다. 전체 장면에서 별도 전경 cone과 뒤 목재 구조의 관계를 읽기 쉽다.','기존 E 및 네 조건 E를 보존한다. 이 한 색상·형태 사례가 모든 외부가림 일반화를 대표하거나 정확한 숨은 corner 좌표를 증명하지 않는다.','CLEAR_FOREGROUND_CONE'),
296:('같은 야외 cone 장면에서 실제 q crop은 붉은 cone과 흰 띠, 일부 옆 목재를 담는다. 중앙32는 cone 면이 대부분이며 네 조건 모두 E다.','기존 native 상태는 D이고 manual reference오차가 약166.94px이다. Native 상태·원래 q 위치·literal crop의 대상이 다를 수 있는 사례로 남기며, 기존 D를 틀린 라벨로 고치거나 새 E gold를 만들지 않는다.','SEMANTIC_NATIVE_VS_LITERAL_Q'),
300:('같은 프레임이지만 q는 cone이 없는 뒤쪽 목재 판 끝/아스팔트 경계로 이동해 있다. raw96와 중앙32에는 붉은 cone이 보이지 않고 판·바닥이 보이며 네 조건 모두 D다.','기존 native 상태 E와 pnp_projected reference오차 약103.57px을 유지한다. 이를 같은 픽셀의 명백한 가림 실패로 단정하지 않고 native 의미·참조 불확실성과 literal q 입력의 차이로 기록한다.','SEMANTIC_NATIVE_VS_LITERAL_Q'),
324:('아주 먼 목재 팔레트의 작은 끝을 L약31px 창에서 확대했다. 중앙32는 회색 배경과 조그만 갈색 구조 경계이며, fullRGB에는 cone들이 있으나 q 부위의 상세 부품 관계는 알아보기 어렵다.','manual reference오차 약1.21px의 기존 D에도 C46_ORIGINAL0 등 E 호출이 있다. 작은 reference차이와 부품 가독성/정확한 native 가림은 별개다.','DISTANT_SMALL_END'),
401:('야간 장면은 강한 조명과 깊은 어둠·노이즈가 공존하고 선택된 오른쪽 끝은 작다. raw96의 갈색 상부와 검은 면/녹색 배경 경계가 흐리며 별도 물체와 자체 그림자를 확실히 가르기 어렵다.','manual reference오차 약1.71px의 기존 D에 네 새 조건 E, C16 D가 표시된다. 야간 조건과 호출 차이는 관찰하되 새 정답·인과·모델 선택으로 바꾸지 않는다.','NIGHT_NOISE_BOUNDARY'),
840:('초록 SQUARE 팔레트의 둥근 앞 모서리·수직 리브가 crop 중심에 있고 위쪽에는 cone 바닥과 밝은 상면이 보인다. 목재 판 끝과 다른 재료·형태이며 네 새 조건 E와 v61/C16 D가 함께 표시된다.','기존 D/manual reference 약2.70px을 보존한다. 가까운 cone의 장면 존재와 선택 끝의 실제 가림은 구분해야 하며 physical3px gold나 새 owner 라벨을 만들지 않는다.','GREEN_SQUARE_NEAR_CONE'),
841:('같은 초록 팔레트의 반대 위쪽 모서리는 회색 바닥과 맞닿고 crop 안에 cone의 붉은 면은 거의 없다. 중앙32는 초록 립/수직 면과 배경 경계다. 원 threshold D와 publicCAL E가 동일 이미지에서 나뉜다.','manual reference 약0.59px의 기존 D를 유지한다. 이는 작은 reference 오차라도 threshold 조정에 따른 오탐이 생길 수 있는 고정 사례이며 독립 정확좌표 gold가 아니다.','GREEN_SQUARE_BOUNDARY'),
858:('초록 팔레트의 낮은 오른쪽 끝은 raw96의 왼쪽 위에 있고 대부분은 회색 아스팔트다. 중앙32에는 얇은 초록 림과 바닥이 섞여 매우 작은 성분을 구분해야 한다. C45_PUBLICCAL은 D, C46_PUBLICCAL은 E다.','기존 D/reference 약6.55px과 원래 q를 그대로 둔다. 다른 호출을 전체모델 우열이나 성능 분모로 해석하지 않는다.','SMALL_END_BACKGROUND_BOUNDARY'),
898:('초록 팔레트의 낮은 오른쪽 끝 일부가 raw96 왼쪽 위에 있고 중앙은 바닥이 많다. fullRGB의 cone은 다른 위치에 있으며 이 crop의 선택 끝을 직접 가리는 것으로 단정할 수 없다. 네 새 조건은 모두 D다.','기존 D와 약9.51px reference 진단을 보존한다. 비슷해 보이는 858과 결과가 달라도 이미지 자세/초기 q·부위의 차이를 넘어 모델 원인을 단정하지 않는다.','GREEN_SQUARE_BOUNDARY'),
908:('선택 q와 중앙32가 주황 cone의 아래쪽 몸체/받침 부근에 놓이고 raw96 아래에는 초록 팔레트가 보인다. 전체 장면에서 cone과 팔레트의 앞뒤 관계가 읽히며 네 조건 E다.','기존 E를 유지하되 숨은 native 끝이나 물리 corner 좌표의 정확한 가림 gold로 확장하지 않는다.','CLEAR_FOREGROUND_CONE'),
914:('원래 q가 이미지 하단 밖이라 중앙32가 검은 padding이고 validity0/4이다. 원래 상태 UNANNOTATED이며 reference값도 없고 네 조건 모두 U다.','전체1784 U bucket의 사례로서 primary paired1237 성능 평가의 D/E 정답 사례처럼 사용하지 않는다. 검은 padding이나 blank reference를 새 라벨로 보완하지 않는다.','PADDING_INPUT_U'),
996:('q 부근은 아스팔트·얇은 초록 팔레트 상면·cone 받침의 경계다. 중앙32에서 초록과 회색 성분이 크고 붉은 가장자리는 한쪽에만 보여, 정확히 어느 native 끝이 가려졌는지 crop만으로 단정하기 어렵다.','기존 E/manual reference 약1.14px과 C45_ORIGINAL0 FN을 유지한다. Manual 가까움 자체가 물리3px corner gold나 모든 중심샘플 외부물체 소유를 보장하지 않는다.','OCCLUDER_PART_BOUNDARY'),
1009:('큰 cone이 팔레트 앞 오른쪽을 가리지만 실제 q crop은 cone 오른쪽 윤곽과 아스팔트 쪽에 걸쳐 있다. 중앙32에는 회색 바닥이 많고 얇은 붉은 경계가 남으며 fixed0 두 모델 D, publicCAL 두 모델 E다.','기존 E와 manual reference 약6.73px을 보존한다. fullRGB의 native 가림 의미와 literal q가 담는 픽셀 범위의 차이를 기록할 뿐 좌표를 보정하거나 라벨을 필터링하지 않는다.','OCCLUDER_PART_BOUNDARY')}
assert set(notes)==set(ids)
observations=[];checks={};asset_paths=set()
for panel in panels:
 idx=panel['global_index']
 for key in ('raw96','central32','valid96','card'):
  b=panel[key];assert sha(b['path'])==b['sha256'];asset_paths.add(b['path'])
 with Image.open(panel['raw96']['path']) as im:raw=np.asarray(im.convert('RGB'))
 with Image.open(panel['central32']['path']) as im:center=np.asarray(im.convert('RGB'))
 with Image.open(panel['valid96']['path']) as im:valid=np.asarray(im)>0
 with Image.open(panel['card']['path']) as im:card=np.asarray(im.convert('RGB'));assert im.size==(2080,1400)
 assert raw.shape==(96,96,3) and center.shape==(32,32,3) and valid.shape==(96,96)
 assert np.array_equal(center,raw[32:64,32:64]) and ash(raw)==panel['RGB96_HWC_sha256'] and ash(raw.transpose(2,0,1))==panel['RGB96_CHW_sha256'] and ash(center)==panel['central32_HWC_sha256'] and ash(valid[None])==panel['valid96_sha256']
 assert np.array_equal(card[202:586,1120:1504],np.repeat(np.repeat(raw,4,0),4,1)) and np.array_equal(card[202:458,1595:1851],np.repeat(np.repeat(center,8,0),8,1))
 obs,lim,kind=notes[idx]
 observations.append(dict(global_index=idx,id=panel['id'],native_corner=panel['native_corner'],material=panel['material'],memberships=panel['memberships'],actually_viewed=dict(tool='tools.view_image',image=panel['card'],viewed=True,tool_call_count=1),visual_observation_KO=obs,interpretation_limit_KO=lim,diagnostic_visual_topic_not_training_label=kind,original_q_L_retained=True,physical_3px_or_native_or_owner_gold_created=False,labels_corrected=False,evaluation_filter_or_model_threshold_selection_created=False,training_admitted=False,reviewer_is_assistant_not_independent_human=True,review_not_blind_to_historical_context=True))
checks['all19_selected_unique_cards_actually_viewed']=True;checks['all19_raw96_central32_validity_hash_and_exact_slice']=True;checks['all19_card_raw_nearest_display_pixel_equality']=True;checks['all19_original_bound_card_and_cleanPNG_file_hashes_unchanged']=True
class Links(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.ids=set()
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('href','src') and v:self.refs.append(v)
   if k=='id':self.ids.add(v)
index=Path(r['index']['path']);assert sha(index)==r['index']['sha256'];l=Links();l.feed(index.read_text());local_refs=0
for s in l.refs:
 assert not s.startswith(('http:','https:','//'))
 target,_,anchor=s.partition('#');assert (index.parent/unquote(target)).exists() if target else True
 if anchor:assert anchor in l.ids
 local_refs+=1
checks['HTML_local_file_and_anchor_links_exist_no_browser_execution']=True
for key in ('case_ids_before_RGB','all1784_csv','all1784_jsonl','all_changed_pairs_csv','all_changed_pairs_jsonl'):
 b=r[key];assert sha(b['path'])==b['sha256']
checks['selection_and_full_changed_ledgers_bound_unchanged']=True
out=H/'ALL19_VISUAL_OBSERVATIONS.jsonl'
with out.open('x') as f:
 for obs in observations:f.write(json.dumps(obs,ensure_ascii=False,allow_nan=False)+'\n')
qa=dict(schema='V169_ACTUAL_FIXED19_CROP_VISUAL_REVIEW_A',completed=True,PASS=all(checks.values()),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_renderer_complete=bind(receipt),original_case_selection=r['case_ids_before_RGB'],original_render_entry=r['entry'],observations=bind(out),qa_code=bind(Path(__file__).resolve()),checks=checks,counts=dict(selected_unique_cards=19,actually_viewed_card_images=19,actual_view_image_tool_calls=19,distinct_original_frames=len({p['id'] for p in panels}),WOOD_cards=11,SQUARE_cards=8,buckets=48,empty_buckets=2,hashed_card_and_cleanPNG_files=len(asset_paths),local_HTML_file_or_anchor_refs_checked=local_refs,all1784_crops_actually_viewed_claimed=False),actual_view_inventory=[o['actually_viewed']['image'] for o in observations],
selection_scope_KO='대표 선택은 full1784의 기존 상태/각 조건 호출을 대상으로 한 고정 strata이다. Primary paired1237 성능 eligibility와 다를 수 있다. Bucket eligible count(예WOOD TP30)와 paired 성능 분모(예29)를 같은 표본수로 취급하지 않는다. 19개는 성능 추정용 무작위/전수 이미지가 아니다.',
observed_common_topics_KO=['흐림·먼 작은 부위·야간 노이즈와 판/블록/바닥/그림자 경계 때문에 crop만으로 부품 끝을 읽기 어려운 사례가 있다.','cone처럼 눈에 띄는 전경이 중심에 있는 경우와, 가림 의미는 native에 있으나 q crop은 다른 판/바닥이나 cone 경계를 담는 경우가 분리된다.','초록 SQUARE의 리브/둥근 면은 목재 판 끝과 시각 형태가 다르지만 이 그림만으로 전이 실패의 원인이나 다음 모델 설계를 결정하지 않는다.','manual-reference<=3 사례에도 E 호출이 보이지만 fixed reference 진단은 physical3px/정확한 corner gold가 아니다.','원래 inputU의 검은 padding을 자연 배경 또는 새로운 가시성 라벨로 해석하지 않는다.','threshold0와publicCAL의 호출 차이는 같은 저장 입력에서 생긴다. 모든 조건을 보존하며 이 private 진단으로 문턱·모델·학습·데이터를 선택하지 않는다.'],
display_issues=[dict(issue='Original v1 table header single proportional string is left-shifted relative to fixed row columns.',severity='DISPLAY_ONLY',values_and_clean_pixels_unchanged=True,root_planned_caption_only_v2_separate_folder=True,original19_cards_preserved=True)],
limits_KO='같은 assistant의 사후 설명 관찰이며 root가 일부 카드 해석을 먼저 공유했다. 19개 모두 직접 보았지만 독립 인간/블라인드 gold가 아니고 현재223 전체1784 crop을 실제로 본 결과가 아니다. 실패/과제 차이/정확좌표를 단정하거나 라벨·평가 필터를 새로 만들지 않는다. 향후 모델 방향은 public V168/V170 근거에서 정하며 본 private 관찰은 진단 전용이다.',
HTML_browser_executed=False,additional_original_RGB_or_cache_or_feature_bodies_read=0,new_model_encoder_fit_PnP_qL_calls=0,labels_or_models_or_thresholds_modified=False,selected_model=None,training_admitted=False,fully_solved=False)
q=H/'ACTUAL_VISUAL_QA_COMPLETE.json';write(q,qa);print(json.dumps(dict(PASS=True,receipt=bind(q),observations=bind(out),actual_cards=19)))
