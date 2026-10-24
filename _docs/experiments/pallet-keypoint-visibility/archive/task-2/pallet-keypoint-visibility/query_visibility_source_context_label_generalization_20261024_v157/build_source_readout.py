"""Publish source evidence without using private outcomes for next decisions."""
from pathlib import Path
import datetime, hashlib, html, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

V=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p).resolve();assert '_v112'not in str(p).lower()
 return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,d):
 with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 label=read(V/'root_source_label_support/LABEL_SUPPORT_DIAGNOSIS.json')
 nominal=read(V/'root_source_label_support/NOMINAL_REFERENCE_CROP_SUPPORT_APPENDIX.json')
 feat=read(V/'source_feature_coverage_agent/SOURCE_SUPPORTED_DIAGNOSIS.json')
 crop=read(V/'source_crop_review_agent/SOURCE_CROP_REVIEW_FINDINGS.json')
 qa=read(V/'source_crop_review_agent/FINAL_VISUAL_QA.json')
 assert label['PASS']and feat['completed']and qa['PASS']
 assert label['missing_public_TRAIN_E_native_slots']==[0,1,2,6,7]
 font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
 font_manager.findfont('NanumGothic',fallback_to_default=False)
 plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':12})
 fig,ax=plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
 mr=[label['MR_available_TRAIN_E_by_native'][str(i)]for i in range(8)]
 pal=[label['all_existing_public_TRAIN_visibility_E_by_native'][str(i)]-mr[i]for i in range(8)]
 ax[0].bar(range(8),mr,label='MR 공개 실사',color='#367bb5')
 ax[0].bar(range(8),pal,bottom=mr,label='Pallox 공개 실사',color='#df934c')
 for i,(a,b)in enumerate(zip(mr,pal)):ax[0].text(i,a+b+.6,str(a+b),ha='center')
 ax[0].set(xticks=range(8),xlabel='원래 native 역할 번호',ylabel='외부가림 학습 query 수',ylim=(0,44),title='공개 실사 E57개는 3개 역할에 집중')
 ax[0].legend(frameon=False)
 names=['학습 D','학습 E','미사용 D','미사용 E'];keys=['SOURCE_DEVELOPMENT::D','SOURCE_DEVELOPMENT::E','SOURCE_VALIDATION::D','SOURCE_VALIDATION::E']
 groups=[nominal['groups'][k]for k in keys]
 bars=ax[1].bar(names,[100*g['inside32']/g['N']for g in groups],color=['#367bb5','#dc8d36','#6b9ec9','#e8b875'])
 for bar,g in zip(bars,groups):ax[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{g['inside32']}/{g['N']}",ha='center')
 ax[1].set(ylabel='중앙32 범위 안에 있는 명목 참조 (%)',ylim=(0,100),title='역할 라벨과 query 중심 위치를 구분해야 함')
 fig.suptitle('공개 source 감독의 역할 범위와 query 위치 불일치',fontsize=18)
 fig.text(.5,-.045,'제공 pose/CAD의 명목 참조 진단이며 실제 픽셀·키포인트 정답이 아닙니다. 원래 라벨·좌표·split은 유지했습니다.',ha='center',fontsize=11)
 plot=V/'SOURCE_LABEL_AND_QUERY_ALIGNMENT_KO.png';fig.savefig(plot,dpi=160,bbox_inches='tight');plt.close(fig)
 commit={
  'completed':True,'UTC':now,'source_only_design_commitment':True,
  'evidence':[bind(V/'root_source_label_support/LABEL_SUPPORT_DIAGNOSIS.json'),bind(V/'root_source_label_support/NOMINAL_REFERENCE_CROP_SUPPORT_APPENDIX.json'),bind(V/'source_feature_coverage_agent/SOURCE_SUPPORTED_DIAGNOSIS.json'),bind(V/'source_crop_review_agent/SOURCE_CROP_REVIEW_FINDINGS.json')],
  'next_action':'TRAIN-only query-local visible-owner witness audit before any new fit or controlled augmentation.',
  'scope':'All390 available SOURCE_DEVELOPMENT known weak-role queries (D341/E49), original indices/q/L fixed. No held45 or private images used to create training labels.',
  'sidecar_fields':['global_index','original_q_L_binding','original_native_role_D_E_unchanged','center_four_saved_samples_owner','witness_reason','confidence_or_UNKNOWN','source_only_weak_observation_not_pixel_or_native_gold'],
  'owner_states':['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED','UNKNOWN'],
  'definition':'Read actual original RGB96 + central32 + whole-source context. Assign supported owner only when the central2x2 saved sample neighborhood and connected scene structure clearly support it; boundaries, same-material uncertain owner, ambiguous role/entity remain UNKNOWN. This is coarse visual support, not exact native-coordinate truth.',
  'semantic_separation':'Visible role D does not imply target is visible at q; foreign/background pixels do not by themselves imply native external E. Keep owner/role/point-quality targets separate. No automatic primary gate or relabeling.',
  'controlled_pair_feasibility':'Only after the source witness audit, consider unchanged-q before/cover/adjacent controls from public TRAIN evidence with certified visible support. Record failed feasibility as UNKNOWN and keep all original rows. A query-pixel cover alone must not become a native E label unless the actual native support is also certified covered.',
  'previous_counterfactual_limit':'V95 existing controlled source had9 TRAIN accepted pairs, all one physical Pallox bin and conditional cue-set semantics/native identity unknown. Repeating this and calling it native visibility would not fix the target gap.',
  'prohibited':['No private outcome/model/threshold/architecture selection','No held relabeling or TRAIN reassignment','No V112','No q/L/xy or PnP correction','No new training or model-forward authorization in this commitment'],
  'new_fit_authorized':False,'goal_status':'active_unsolved','selected_model':None,
 }
 write(V/'ROOT_NEXT_SOURCE_ACTION_BEFORE_ANY_NEW_FIT.json',commit)
 result={
  'schema':'V157_SOURCE_CONTEXT_LABEL_GENERALIZATION_DIAGNOSIS_COMPLETE_GOAL_UNSOLVED','completed':True,'stage_complete':True,'UTC':now,'goal_status':'active_unsolved','fully_solved':False,'selected_model':None,
  'previous_goal_turn_classification':'progress','scope':bind(V/'ROOT_SOURCE_ONLY_SCOPE_BEFORE_BODIES.json'),
  'lead_KO':'공개 source에서 실제 전이 차이와 query/역할 라벨의 목표 차이를 함께 확인했습니다. 한 원인으로 확정하거나 새 모델 성공으로 해석하지 않습니다.',
  'analyzed_rows':feat['retained_rows'],'all_source_rows_labels_splits_qL_preserved':True,
  'public_TRAIN_visibility_counts':{'D':376,'E':57,'E_by_native':label['all_existing_public_TRAIN_visibility_E_by_native'],'missing_E_native':label['missing_public_TRAIN_E_native_slots']},
  'occluder_coverage':nominal['occluder_coverage'],'nominal_reference_crop_support':nominal['groups'],
  'feature_findings':{k:feat[k]for k in ['frame_balanced_target_context','query_weighted_median_nearest_TRAIN_cosine_distances','tie_aware_E_source_label_overlap','numerical_tie_tolerance','target_context_same_frame_DE_unordered_pairs','target_context_same_frame_DE_identical_pairs','source_supported_findings','interpretive_limits']},
  'direct_observed_example_KO':'TRAIN g1044/native4는 역할D지만 원query=(242.44,192.67), 명목ref=(243.68,290.36), L201.22이며 q주변에는파란상자가보입니다. 기존역할D는밖으로노출된목재deck의구조적보임을의미했고 정확한query위치정답은아닙니다. 따라서D라벨을중심픽셀팔레트존재정답으로복사하면안됩니다.',
  'actual_source_visual_scope':{'full_boards':63,'all_held_frames':45,'initial_TRAIN_frames':12,'extra_category_frames':6,'unique_category_query_rows':32,'exact_saved_crop_cells':504,'agent_actually_viewed_PNGs':21,'root_actually_viewed_selected_contact_sheets':4},
  'limits_KO':['public E가5개native에없지만 heldE도native4/5라 그것만으로held실패를설명하지않습니다.','TRAIN 가림물은회색/파란플라스틱계열, held는바구니/슬랫목재계열입니다. texture/scene 차이는관찰됐지만단독인과증명은아닙니다.','target-context descriptor는같은frame의D/E에서동일합니다. 그러나현모델은query-token/local96/central32도사용하므로context만사용하는모델이라고주장하지않습니다.','같은frame제외TRAIN의이웃은시간/장면이유사할수있어거리차이를독립물리샘플의일반화증명으로보지않습니다.','명목참조는cm오차가능pose/CAD이며pixel/nativegold가아닙니다. 기존역할라벨도assistant약한감독으로유지합니다.','기존E/위치오류/자기가림을합치거나보조owner음수를외부가림정답으로만들지않습니다.'],
  'evidence':{'labels':bind(V/'root_source_label_support/LABEL_SUPPORT_DIAGNOSIS.json'),'nominal':bind(V/'root_source_label_support/NOMINAL_REFERENCE_CROP_SUPPORT_APPENDIX.json'),'features':bind(V/'source_feature_coverage_agent/SOURCE_SUPPORTED_DIAGNOSIS.json'),'crop_review':bind(V/'source_crop_review_agent/SOURCE_CROP_REVIEW_FINDINGS.json'),'crop_QA':bind(V/'source_crop_review_agent/FINAL_VISUAL_QA.json'),'previous_C2_design':bind(V.parent/'query_visibility_same_q_counterfactual_20261023_v95/ROOT_C2_ACTUAL_CORPUS_ADMISSION_BEFORE_FIT.json')},
  'next_source_action':bind(V/'ROOT_NEXT_SOURCE_ACTION_BEFORE_ANY_NEW_FIT.json'),
  'new_model_training_encoder_R0_private_V112_PnP_qL_calls':0,'live_processes':[],
 }
 write(V/'ROOT_RESULTS_KO.json',result)
 styles='body{max-width:1150px;margin:32px auto;padding:0 24px 60px;font-family:sans-serif;line-height:1.7;color:#192838}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:8px;border:1px solid #cad5df;text-align:left}section{margin:30px 0}small{color:#596775}a{color:#245c98}'
 page=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V157 source query/label 진단</title><style>'+styles+'</style><body>',
  '<h1>공개 학습자료의 역할 가시성과 query 위치 상태를 구분해야 합니다</h1><p>V157 source 진단 완료 · 전체 목표 미해결 · 새 모델/학습/실사223 접근0</p>',
  '<p>'+result['lead_KO']+'</p><img src="SOURCE_LABEL_AND_QUERY_ALIGNMENT_KO.png" alt="source 역할 및 query 라벨 범위">',
  '<section><h2>확인한 데이터 빈틈</h2><p>공개 실사 TRAIN 가림 감독57개는 native3/4/5에만 있습니다. MR의49개는 회색통23·파란KLT8·회색상자/흰포장블록18개이고, held52개는 바구니28·슬랫목재상자24개입니다. held도native4/5라서 역할 부족만으로 실패를 설명하지는 않습니다.</p>',
  '<p>명목참조가 중앙32 범위 안에 있는 MR TRAIN은 D242/341·E13/49입니다. 원래q와 같은-native 제공pose/CAD 사이의 거리 진단이며, 실제 위치의 정확한 정답이 아닙니다. 원래 라벨을 변경하거나 표본을 걸러내지 않았습니다.</p></section>',
  '<section><h2>실제 query crop과 역할 라벨의 목표 차이</h2><p>'+result['direct_observed_example_KO']+'</p><img src="source_crop_review_agent/contact_sheets/CATEGORY_FIRST_ROWS_04.png" alt="역할D와 query위치차이 포함 source사례"><small>g1044는 왼쪽 팔레트 역할의 구조적 보임과, 파란 상자 위에 놓인 초기q의 상태를 구분해야 하는 TRAIN 사례입니다. 다른 사례도 원래D/E/S/OOF/U를 그대로 표시했습니다.</small></section>',
  '<section><h2>단순 라벨 차이만 남은 것은 아닙니다</h2><p>저장된 C37/C38은 TRAIN의 약한 D/E390개를 모두 맞췄지만 held에서는145개 중46/50개를 틀렸습니다. Local96 특징에서 같은 E TRAIN 예가 더 가까운 경우는 TRAIN48/49와 held2/52였습니다. 중앙32에서는41/49와1/52였습니다. 이는 기존 특징·라벨의 관측이며 새 분류기를 학습한 결과나 인과 증명이 아닙니다.</p><p>같은 frame의D/E295쌍은 target-context 특징이 같았습니다. query-specific 입력이 필요한 이유를 보여 주지만, 현모델에query입력이 없다는 뜻은 아닙니다. context의 nearest-class tie를E부재로 해석하지 않았습니다.</p><img src="source_feature_coverage_agent/SOURCE_COVERAGE_SAVED_ERRORS.png" alt="source feature coverage 및 저장오류"></section>',
  '<section><h2>실제 학습·미사용 장면</h2><img src="source_crop_review_agent/contact_sheets/FIRST12TRAIN_01.png" alt="첫학습4장"><img src="source_crop_review_agent/contact_sheets/HELD45_01.png" alt="미사용첫4장"><p>선택 규칙은 이미지 읽기 전에 봉인했습니다. 전체held45와 첫TRAIN12, 규칙으로 추가된6프레임을 확인했습니다. LOCAL96은 실제 저장된 원본 입력이고, context는 원RGB에q를 표시한 그림이며 target224 encoder tensor와는 구분합니다.</p></section>',
  '<section><h2>다음은 TRAIN query 중심의 실제 보이는 대상부터 확인합니다</h2><p>원래 TRAIN D341/E49의390개 q를 옮기지 않고, 중심 주변이 대상팔레트·외부물체·배경·미확정 중 무엇인지 별도 약한 관찰 sidecar에 기록합니다. 기존 native역할D/E와 분리하고, 경계·같은목재의소유개체모호함은UNKNOWN으로 남깁니다. 이 기록을 새 위치정답이나 외부가림정답으로 자동 승격하지 않습니다.</p><p>확실한 visible-support가 있는 경우에만 원래q에서 before/cover/adjacent 대조의 구성 가능성을 확인합니다. query픽셀을 덮었다는 이유만으로 native전체가림E를 만들지 않습니다. 새fit과 모델변경은 아직 시작하지 않았습니다.</p></section>',
  '<section><h2>원래 범위와 제한</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>'for x in result['limits_KO'])+'</ul><p>원래목표인WOOD/SQUARE/native8 외부가림 판단을 유지합니다. 자기 가림은상위기하범위로남기고PnP·q/L·xy를수정하지않습니다. current223과기존held45를학습에편입하지않으며모델채택은없습니다.</p></section>',
  '<p><a href="ROOT_RESULTS_KO.json">구조화 결과</a> · <a href="ROOT_NEXT_SOURCE_ACTION_BEFORE_ANY_NEW_FIT.json">다음 source 작업 범위</a> · <a href="source_crop_review_agent/FINAL_VISUAL_QA.json">21개 그림 시각 확인</a> · <a href="source_feature_coverage_agent/SOURCE_SUPPORTED_DIAGNOSIS.json">특징·오류 진단</a></p></body></html>']
 with (V/'ROOT_READOUT_KO.html').open('x')as f:f.write('\n'.join(page))
 print(json.dumps(dict(completed=True,results=bind(V/'ROOT_RESULTS_KO.json'),report=bind(V/'ROOT_READOUT_KO.html'),plot=bind(plot))))
if __name__=='__main__':main()
