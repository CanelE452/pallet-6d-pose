"""Short Korean readout from frozen V168/V166 JSON and completed saved-array receipt."""
from pathlib import Path
import hashlib,html,json
from urllib.parse import unquote
from html.parser import HTMLParser

V=Path(__file__).resolve().parent.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def cmmetric(r):return r['CM_D_E_rows_D_E_U_columns']
completepath=V/'retrospective_full_source_v1/COMPLETE.json';complete=read(completepath)
inputspath=V/'readout_preparation/FIXED_OUTPUT_HEAD_SCALAR_AND_LABEL_BINDINGS.json';inputs=read(inputspath)
auditpath=V/'saved_array_review/REVIEW.json';audit=read(auditpath)
assert complete['PASS'] and audit['PASS'] and len(complete['results'])==2
models=[]
for model,result in zip(inputs['models'],complete['results']):
    old=read(model['source_results']['path']);new=read(result['report']['path'])
    panels=read(result['panels']['path'])['panels']
    models.append(dict(arm=model['arm'],fixed_E_logit_threshold_float64=model['threshold_float64'],
        native35=dict(original0_passed=old['native']['native35_passed'],fixed_scalar_passed=new['native']['native35_passed'],N=35),
        synthetic_VALIDATION={family:dict(original0_CM_D_E_by_D_E_U=cmmetric(old['native'][family]['VALIDATION']),fixed_scalar_CM_D_E_by_D_E_U=cmmetric(new['native'][family]['VALIDATION'])) for family in ('old_native','new_native')},
        Pallox_inputvalid_sanity=dict(original0=old['Pallox']['sanity'],fixed_scalar=new['Pallox']['sanity']),
        MR_PUBLIC_CAL_APPARENT=dict(original0=old['MR']['SOURCE_VALIDATION']['visibility'],fixed_scalar=new['MR']['PUBLIC_CAL_APPARENT']['visibility'],inputvalid145=new['MR']['PUBLIC_CAL_APPARENT']['visibility_inputvalid']),
        controlled_weakTRAIN92_response=dict(original0=old['CF_part_native_diagnostic'],fixed_scalar=new['CF_weakTRAIN92_response']),
        panel_count=len(panels),source_report=result['report'],factor_native8_panels=result['panels'],
        retrospective_existing_source_gates_all_pass=new['existing_source_gates_all_pass'],independent_qualification_claimed=False))
results=dict(completed=True,phase='V168',scope='동결된 V166 두 head의 저장 출력 전체에 V167 full PUBLIC_CAL scalar를 그대로 적용한 사후 공개 소스 판독',
    conclusion_KO='MR 공개 한 시퀀스에 맞춘 문턱은 합성·Pallox 보임 판정을 크게 잃었다. 두 조건 모두 기존 전체 source qualification 실패이며 채택할 모델은 없다.',
    fixed_rules=dict(raw='원 FP32 E-logit',comparison='float64(raw_FP32) >= unchanged float64 scalar',threshold_search_calls=0,head_or_model_selection=False,entity_quality_owner_raw_unchanged=True),
    models=models,
    counts=dict(original_each=28976,PART_each=1144,total_each=30120,original_inputU_each=535,PART_inputU_each=178,all_variant_statuses=198,missing_null_variants=55,actual_saved_retrospective_NPZ_reloaded=10,original_V166_NPZ_reloaded=10),
    audit_kind='root와 작성자가 코드 공동 검토 후 저장 배열을 실제 재로딩하여 검산함. fresh independent 성능 실험 또는 별도 독립 모델 검증이 아님.',
    original0_parity_first='실제 producer가 두 조건의 native/Pallox/MR/quality/CF 전체 원0 보고서 일치를 확인한 뒤 scalar 적용. 이번 저장배열 감사는 원 필드·판정·패널·gate 요약을 검산.',
    limitations_KO=[
        'MR PUBLIC_CAL은 45프레임 한 공개 시퀀스의 입력 유효 known145(D93/E52)에 맞춘 apparent 결과다. 전체360 중 known154(D102/E52), known-inputU9는 모두 D이고 unknown truth206이다.',
        '이 공개 시퀀스의 수치·crop·ranking은 V153~V166에서 반복 관찰됐다. V167의 purged OOF도 같은 시퀀스 내 문턱 안정성 진단이며 fresh independent model validation이 아니다.',
        'scalar는 동작 문턱 조정이며 보정된 확률 p(E)를 뜻하지 않는다. MR apparent 이득을 다른 재료·형상·native의 성공으로 일반화할 수 없다.',
        'MR/Pallox 역할 및 controlled PART92는 assistant/source weak supervision이다. PART92 반응은 학습 소스 반응이며 held 성능이나 물리 native point gold가 아니다.',
        '이 단계는 새 threshold/모델 선택·새 학습·새 라벨 편입 없이 저장값만 판독했다. 전체 WOOD/SQUARE/native8 목표는 미해결이다.'
    ],
    bindings=dict(complete=bind(completepath),inputs=bind(inputspath),saved_array_review=bind(auditpath),producer_code=complete['code'],producer_protocol=complete['protocol'],actual_entry=complete['entry'],fixed_scalar_source=complete['fixed_scalar_source']),
    source_qualification=False,selected_model=None,adopted=False,fully_solved=False,goal_complete=False,current_performance_claimed=False,
    forbidden_RGB_features_weights_private_current_V112_access=0,new_model_forward_fit_or_threshold_search=0,HTML_browser_rendered=False)
resultpath=V/'ROOT_RESULTS_KO.json';write(resultpath,results)

def cell(x):return html.escape(str(x))
def fraction(cm,row):return f'{cm[row][row]}/{sum(cm[row])}'
rows=[]
for m in models:
    name=m['arm'].split('_')[0];old=m['synthetic_VALIDATION']['old_native'];new=m['synthetic_VALIDATION']['new_native']
    p=m['Pallox_inputvalid_sanity'];mr=m['MR_PUBLIC_CAL_APPARENT'];cf=m['controlled_weakTRAIN92_response']
    rows.extend([
        [name,'native 기존 기준',f"{m['native35']['original0_passed']}/35",f"{m['native35']['fixed_scalar_passed']}/35"],
        [name,'합성 OLD VAL: D / E',fraction(old['original0_CM_D_E_by_D_E_U'],0)+' / '+fraction(old['original0_CM_D_E_by_D_E_U'],1),fraction(old['fixed_scalar_CM_D_E_by_D_E_U'],0)+' / '+fraction(old['fixed_scalar_CM_D_E_by_D_E_U'],1)],
        [name,'합성 NEW VAL: D / E',fraction(new['original0_CM_D_E_by_D_E_U'],0)+' / '+fraction(new['original0_CM_D_E_by_D_E_U'],1),fraction(new['fixed_scalar_CM_D_E_by_D_E_U'],0)+' / '+fraction(new['fixed_scalar_CM_D_E_by_D_E_U'],1)],
        [name,'Pallox 입력 유효 sanity: D / E',f"{p['original0']['D_correct']}/{p['original0']['D_N']} / {p['original0']['E_correct']}/{p['original0']['E_N']}",f"{p['fixed_scalar']['D_correct']}/{p['fixed_scalar']['D_N']} / {p['fixed_scalar']['E_correct']}/{p['fixed_scalar']['E_N']}"],
        [name,'MR PUBLIC_CAL apparent: D / E',fraction(mr['original0']['CM_rows0_1_columns0_1_U'],0)+' / '+fraction(mr['original0']['CM_rows0_1_columns0_1_U'],1),fraction(mr['fixed_scalar']['CM_rows0_1_columns0_1_U'],0)+' / '+fraction(mr['fixed_scalar']['CM_rows0_1_columns0_1_U'],1)],
        [name,'controlled weak TRAIN92: D / E',fraction(cf['original0']['CM_rows0_1_columns0_1_U'],0)+' / '+fraction(cf['original0']['CM_rows0_1_columns0_1_U'],1),fraction(cf['fixed_scalar']['CM_rows0_1_columns0_1_U'],0)+' / '+fraction(cf['fixed_scalar']['CM_rows0_1_columns0_1_U'],1)],
    ])
table=''.join('<tr>'+''.join('<td>'+cell(v)+'</td>' for v in row)+'</tr>' for row in rows)
link=lambda p,label:'<a href="'+html.escape(str(Path(p).relative_to(V)),quote=True)+'">'+cell(label)+'</a>'
links=' · '.join([link(resultpath,'기계 판독 결과'),link(auditpath,'저장 배열 검산'),link(completepath,'실제 완료 receipt')]+[link(m['factor_native8_panels']['path'],m['arm'].split('_')[0]+' 전체 249 패널') for m in models])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V168 동결 문턱 전체 공개 소스 판독</title><style>body{font:16px/1.6 sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#182334}h1{font-size:26px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border:1px solid #cbd3de;text-align:left}th{background:#edf1f7}.result{background:#fff0ea;padding:16px;border-left:5px solid #a44118}small{color:#465469}a{color:#16558e}</style><h1>V168 · 동결 문턱의 전체 공개 소스 적용</h1>
<p class="result"><strong>두 조건 모두 기존 전체 source qualification 실패. 모델 채택 없음, 전체 목표 미해결.</strong><br>MR 한 공개 시퀀스에 맞춘 문턱을 다른 공개 소스에 적용하자 보임(D) 판정을 크게 잃었습니다.</p>
<p>V166 C45/C46 head와 저장 logits를 바꾸지 않고 V167 full PUBLIC_CAL scalar <code>−10.97239875793457</code> / <code>−9.855539321899414</code>를 적용했습니다. 원 FP32 logit를 float64로 올린 뒤 <code>logit ≥ scalar</code>로 판정했습니다. 새 학습·문턱 탐색·모델 선택은 없습니다. 아래는 각 역할에 대한 맞은 수/해당 역할 분모이며, D는 외부 가림 없음 역할, E는 외부 가림 역할입니다.</p>
<table><thead><tr><th>조건</th><th>공개 소스 패널</th><th>기존 문턱 0</th><th>동결 PUBLIC_CAL scalar</th></tr></thead><tbody>'''+table+'''</tbody></table>
<p>합성 VAL의 D 분모에는 기존 입력 U도 포함됩니다. OLD는 828 중 U3, NEW는 1772 중 U2를 보존했습니다. Pallox 표는 기존 sanity의 입력 유효 D33/E6이며 전체 VAL weak known41에는 D-inputU2도 그대로 남습니다.</p>
<p><strong>MR 행은 PUBLIC_CAL apparent입니다.</strong> 문턱을 맞춘 같은 한 공개 시퀀스의 360행 중 known154(D102/E52), 입력 유효 known145(D93/E52)입니다. known D의 입력 U9와 unknown truth206을 유지했습니다. 반복 관찰한 공개 자료이므로 fresh independent model validation을 뜻하지 않습니다. V167의 purged OOF는 별도로 같은 시퀀스 내 문턱 안정성만 진단했고, 이 표의 full CAL apparent와 구분됩니다. scalar 조정은 확률 p(E)의 보정을 뜻하지 않습니다.</p>
<p>controlled weak TRAIN92는 학습 소스 반응으로 held 성능과 구분했습니다. MR/Pallox 역할과 PART92는 assistant/source weak supervision이며 물리 native point gold가 아닙니다. entity·quality·owner raw 값은 변하지 않았고 primary 필터로 추가하지 않았습니다.</p>
<p>원 28,976행 + PART 1,144행을 조건별 모두 보존했습니다. 입력 U는 원535 + PART178, variant status는 198개 중 missing55의 null을 유지했습니다. 두 조건 각각 249개의 split/material/aspect/foreign-family/geometry/native0..7 패널을 보존했습니다.</p>
<p>root와 작성자가 producer 코드를 공동 검토했고, 실제 완료 뒤 새 NPZ10개와 원 V166 NPZ10개를 재로딩했습니다. 원 전 필드 바이트 일치, direct float64 판정, 보류·status·패널·gate 요약 검산은 PASS입니다. 이는 저장값 검산이며 fresh independent 성능 연구가 아닙니다. 원0 전체 보고서 exact parity와 V167 MR full CAL exact 일치는 실제 producer receipt에 기록되어 있습니다.</p>
<p>'''+links+'''</p><small>HTML 소스와 로컬 링크를 점검했습니다. 브라우저 렌더링은 수행하지 않았습니다. 이 판독에서 RGB/feature/weight/current/private/V112 접근, encoder/head forward, PnP, q/L 변경은 없습니다.</small></html>'''
htmlpath=V/'ROOT_READOUT_KO.html'
with htmlpath.open('x') as f:f.write(page+'\n')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href')
parser=Links();parser.feed(page)
assert all((V/unquote(href)).is_file() for href in parser.links)
phase=dict(completed=True,phase='V168',actual_saved_readout_once_complete=bind(completepath),saved_array_verification=bind(auditpath),results=bind(resultpath),readout=bind(htmlpath),
    report_kind='Retrospective source readout; jointly reviewed code plus saved-array verification, no fresh independent performance claim',
    native35=[m['native35'] for m in models],source_qualification=False,selected_model=None,adopted=False,fully_solved=False,goal_complete=False,
    threshold_or_model_selection=False,private_current_evaluation_claimed=False,new_model_forward_fit_threshold_search=0,
    original_logits_metadata_auxiliary_outputs_unchanged=True,original_inputU535_PART178_preserved=True,status198_missing55_null_preserved=True,
    local_HTML_links_checked=len(parser.links),HTML_browser_rendered=False,limitations='MR full CAL apparent from repeatedly observed one public sequence; weak controlled TRAIN response is not held performance; cross-source D losses remain.',
    source_bodies_accessed_for_report='JSON receipts only; saved-array verifier reloaded exactly ten retrospective and ten original saved output NPZ. No RGB/features/weights/model.')
phasepath=V/'ROOT_PHASE_RESULT.json';write(phasepath,phase)
print(json.dumps({'completed':True,'results':bind(resultpath),'html':bind(htmlpath),'phase':bind(phasepath),'links_checked':len(parser.links)},ensure_ascii=False))
