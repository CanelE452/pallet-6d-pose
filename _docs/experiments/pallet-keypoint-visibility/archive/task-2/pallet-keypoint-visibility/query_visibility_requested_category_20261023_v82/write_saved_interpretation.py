"""Additive interpretation of completed saved V82 report; no model/RGB I/O."""
from pathlib import Path
import hashlib,html,json
R=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def check(b):assert bind(Path(b['path']))==b
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def table(head,rows):return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'

report=read(R/'RESULTS_KO.json')
for b in report['bindings'].values():
    if isinstance(b,dict) and set(b)>={'path','sha256'}:check(b)
assert report['completed'] and report['no_automatic_current_evaluation'] and report['current223']==0
old=report['V81_benchmarks'][0];new=report['V82_benchmarks'][0]
a=old['strict535_same_frozen_IDs'];b=new['strict535_same_frozen_IDs']
assert a['CM_rows_GT_CV_cols_pred_CVU']==[[15,56,5],[83,259,117]]
assert b['CM_rows_GT_CV_cols_pred_CVU']==[[22,24,30],[47,251,161]]
assert report['prospective_component_screen']['component_screen_PASS'] is False
cases=[
 {'id':'YCB:000009:000241:000000','native_k':6,'observation_KO':'파란 pitcher 뒤/앞의 드릴이 함께 보이며 q/r는 빨간 드릴 부분에 놓입니다. 요청 phrase는 mug에서 pitcher base로 바뀌었지만 예측 mask는 계속 pitcher를 분리하고 q는 mask0입니다. source C와의 일치는 유지됐습니다.','limit_KO':'전체 mask IoU는 약0.771로 비슷하되 bit-identical mask라고 주장하지 않습니다. mask0는 별도의 semantic E/S 원인 판정이 아닙니다.'},
 {'id':'YCB:000008:001872:000004','native_k':6,'observation_KO':'하얀 bleach cleanser 앞에 노란 mustard bottle이 보입니다. V82는 bleach cleanser를 요청하고 같은 phrase를 출력했지만 선택 box와 mask는 노란 앞 물체 쪽에 남았습니다. q는 두 버전 모두 visible-mask로 잘못 읽고 intended-target IoU는 약0.027입니다.','limit_KO':'요청 이름/예측 phrase 일치가 instance 소유권의 증거는 아닙니다. 저장된 detector-box+SAM 파이프라인 결과이며 detector 단독 원인이 모든 오차를 설명한다고 단정하지 않습니다.'},
 {'id':'YCB:000008:001299:000002','native_k':1,'observation_KO':'낮은 tuna fish can이 실제 crop과 target crop에 보입니다. V81과 V82 모두 입력 q-box에 대한 최대 IoU가 고정0.1에 미달해 U/no mask입니다. category 제공으로 복구되지 않았습니다.','limit_KO':'U는 absent 또는 all-zero mask로 바꾸지 않았고 source V recall의 miss 분모에 남겼습니다.'},
 {'id':'YCB:000009:000225:000002','native_k':6,'observation_KO':'드릴 아래 목재 block의 밝은 윗면이 보입니다. phrase는 cracker box에서 wood block으로 바뀌었으나 q의 visible-mask 정답과 약0.821의 whole-mask IoU는 이미 V81에서 성립했습니다.','limit_KO':'이 고정 사례의 새 성공이나 category 제공의 필수 효과라고 주장하지 않습니다.'}]
assert [(x['id'],x['native_k']) for x in cases]==[(x['id'],x['native_k']) for x in report['fixed_four']]
transition=report['paired_comparison']['same_strict535_prediction_transitions'][0]['V81_to_V82_CVU_counts']
assert transition==[[4,5,6],[17,17,22],[1,2,2]]
owner=report['ownership_diagnostic']
oldbins={}
for x in owner['all800_V81_to_V82_bin_transitions']:oldbins[x['V81']]=oldbins.get(x['V81'],0)+x['N']
v=dict(completed=True,PASS=True,status='HOLD_SOURCE_COMPONENT_SCREEN_0_OF_4',fully_solved=False,
    interpretation_KO='더 강한 category 제공 조건에서도 source component 기준4개가 모두 실패했습니다. 조건부 mask IoU 향상을 실제 C/V 판별 해결로 부르지 않습니다.',
    strict535=dict(V81=a,V82=b,GT_C76_net_distribution_change=dict(pred_C=7,pred_V=-32,pred_U=25),net_change_is_not_individual_transition=True,actual_C76_V81row_to_V82column_transition_CVU=transition),
    whole_mask=dict(V81=old['mask_IoU'],V82=new['mask_IoU']),
    all800_ownership=dict(V81_bins=oldbins,V82_bins=owner['all800_bins'],new_falsevisible24_all_have_other_visible_at_q=True,new_falsevisible_global_other_IoU_greater23_of24=True),
    cases=cases,all_four_actual_panels_directly_viewed=True,new_case_selection=False,
    source_RGB_GT_reads_by_this_addendum=0,new_model_forward_fit_current_PnP=0,
    bindings=dict(report=bind(R/'RESULTS_KO.json'),main_HTML=bind(R/'RESULTS_KO.html'),build_seal=bind(R/'fixed_natural_gallery/BUILD_SEAL.json'),code=bind(Path(__file__))))
write(R/'RESULTS_INTERPRETATION_ADDENDUM_KO.json',v)
style='body{font-family:Arial,"NanumGothic",sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#182431;line-height:1.7}h1{font-size:30px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px;border:1px solid #bbc9d2;text-align:left}th{background:#eaf0f4}.warn{background:#fff0e8;border-left:5px solid #bd6c48;padding:20px}img{width:100%}a{color:#145b7e}'
body=f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>V82 결과 해석과 실제 네 사례</title><style>{style}</style><h1>V82: source component HOLD — 사전 기준 0/4 통과</h1>'
body+='<p class="warn">요청 category를 추가해도 해결되지 않았습니다. 사람의 corner/부분선 cue D/E/S 목표와 별개인 source mask component 진단이며, 현재223장 사용·새 학습·PnP는0입니다. SYNTH는 pallet 상수, 공개 YCB는 GT metadata로 요청 category를 제공한 더 강한 조건입니다. 동일 입력의 V81→V82 ablation으로 해석하지 않습니다.</p>'
body+=table(['같은 strict535: GT C76/V459','V81','V82'],[['혼동행렬: GT C/V × pred C/V/U',a['CM_rows_GT_CV_cols_pred_CVU'],b['CM_rows_GT_CV_cols_pred_CVU']],['C recall(U miss)',f'{15/76:.1%}',f'{22/76:.1%}'],['V recall(U miss)',f'{259/459:.1%}',f'{251/459:.1%}'],['C precision: GT C/V-known',f'{15/98:.1%}',f'{22/69:.1%}'],['보류 U /535',f'{122}/535 ({122/535:.1%})',f'{191}/535 ({191/535:.1%})']])
body+='<p><b>C76의 false-visible 56→24 감소는 32개가 정답 C로 바뀌었다는 뜻이 아닙니다.</b> 전체 C76 예측 분포의 순변화는 C15→22(+7), V56→24(−32), U5→30(+25)입니다. 실제 같은 행의 전이는 아래와 같아, 원래 V였던56개 중 C17·V17·U22로 바뀌었습니다. 다른 원래 C/U 행에서 V로 이동한7개도 보존했습니다.</p>'
body+=table(['GT C76: V81 pred → V82 pred','C','V','U'],[['C',*transition[0]],['V',*transition[1]],['U',*transition[2]]])
body+='<h2>조건부 mask 품질과 전체 판단을 분리</h2>'+table(['전체1035 instance','V81','V82'],[['전체 mIoU(U=0)',f'{old["mask_IoU"]["all_rows_no_mask0_mean"]:.6f}',f'{new["mask_IoU"]["all_rows_no_mask0_mean"]:.6f}'],['mask 있음 조건부 mIoU',f'{old["mask_IoU"]["usable_selected_mask_conditional_mean"]:.6f}',f'{new["mask_IoU"]["usable_selected_mask_conditional_mean"]:.6f}'],['mask 있음 / U',f'{823} / {212}',f'{790} / {245}']])
body+='<p>조건부 mask IoU는 개선됐지만 mask를 제공하는 instance 수는823→790으로 줄었습니다. 공개 YCB800개에서 OTHER_IOU_GREATER는194→103으로 줄었고, INTENDED_IOU_GREATER는484→479, no-association U는122→218입니다. 다른 물체를 고르는 오류 감소만으로 intended instance를 더 많이 성공했다고 말할 수 없습니다.</p><p>새 strict C의 false-visible24개 모두 다른 annotated visible instance가 q를 차지한 근거가 있으며,23개는 whole-mask IoU도 intended보다 다른 instance에서 더 컸습니다. 이는 소유권 혼동 진단을 지지하지만 detector 단독의 인과 실패를 증명하지 않습니다.</p>'
body+='<h2>직접 확인한 같은 네 그림</h2>'
for text,record in zip(cases,report['fixed_four']):
    body+=f'<h3>{html.escape(text["id"])} / native{text["native_k"]}</h3><p>{html.escape(text["observation_KO"])}</p><p>{html.escape(text["limit_KO"])}</p><img src="{html.escape(record["figure"]["path"])}" alt="고정 source 사례">'
body+='<h2>남은 목표</h2><p>mask0는 semantic E/S의 원인 판정이 아니고, virtual cuboid reference의 구멍/공중 위치를 실제 외부 가림으로 바꿔 부를 수 없습니다. category는 instance ID도 아닙니다. 이번 cohort는 image/category 중복이 없어 같은 종류 여러 instance의 분리 능력을 검증하지 못합니다. 반복 사용한 source VAL 개발 진단이며 독립 final test가 아닙니다. 기준에 맞춰 새 caption·문턱·query를 고르거나 현재 데이터로 다음 모델을 선택하지 않았습니다.</p>'
body+=f'<p><a href="{R/"RESULTS_KO.html"}">전체 표·소유권 진단 보고서</a> · <a href="{R/"RESULTS_KO.json"}">전체 JSON</a> · <a href="{R/"RESULTS_INTERPRETATION_ADDENDUM_KO.json"}">해석 근거 JSON</a></p></html>'
with (R/'RESULTS_INTERPRETATION_ADDENDUM_KO.html').open('x') as f:f.write(body)
print(json.dumps(bind(R/'RESULTS_INTERPRETATION_ADDENDUM_KO.html')))
