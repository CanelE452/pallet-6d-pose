"""Korean C5 readout, saved source outputs only. No model/feature/private reader."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,html,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
RUN=V/'runs/C5_FROZEN_C1_CENTRAL_RGB_SOURCE_V1';NOLD=31680
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(x):assert sha(x['path'])==x['sha256'];return Path(x['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def arrayeq(a,c):return np.array_equal(a,c,equal_nan=True)if np.asarray(a).dtype.kind in'fc'else np.array_equal(a,c)
def ratio(a,c):return float(a/c)if c else None
def cm(y,p,mask):
    x=np.zeros((2,3),np.int64);q=mask&np.isin(y,[0,1]);cols=np.where(p[q]<0,2,p[q]);np.add.at(x,(y[q],cols),1)
    return dict(matrix=x.tolist(),known_DE_N=int(x.sum()),D_N=int(x[0].sum()),E_N=int(x[1].sum()),D_recall=ratio(x[0,0],x[0].sum()),E_recall=ratio(x[1,1],x[1].sum()),D_precision=ratio(x[0,0],x[:,0].sum()),E_precision=ratio(x[1,1],x[:,1].sum()),prediction_U=int(x[:,2].sum()),U_rate=ratio(x[:,2].sum(),x.sum()))
def pair(p,pp,mask):
    v=p[pp[mask]];n=len(v)
    return dict(N=n,before_D=int((v[:,0]==0).sum()),cover_E=int((v[:,1]==1).sum()),sham_D=int((v[:,2]==0).sum()),all3=int(np.all(v==[0,1,0],axis=1).sum()),any_U=int((v<0).any(1).sum()),all3_rate=float(np.all(v==[0,1,0],axis=1).mean())if n else None)
def pct(x):return '—'if x is None else f'{100*x:.1f}%'
def esc(x):return html.escape(str(x))
def link(path,label):return f'<a href="{esc(path)}">{esc(label)}</a>'
def table(headers,rows):return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table></div>'

def main():
    fit=read(RUN/'FIT_COMPLETE.json');assert fit['completed']and fit['PASS_execution']
    for x in fit.values():
        if isinstance(x,dict)and'path'in x and'sha256'in x:check(x)
    report=read(check(fit['source_results']));gates=read(check(fit['source_gates']));freeze=read(check(fit['MODEL_FREEZE']))
    audit=read(V/'independent_saved_review/REVIEW.json');assert audit['PASS']and audit['score_binding']==fit['scores']
    steering=read(V/'ROOT_USER_STEERING_EXTERNAL_ONLY.json');assert steering['next_source_supervision']['S'].startswith('preserved geometric')
    base_receipt=read(V/'v61_source_baseline/FINAL_BASELINE_RECEIPT.json');assert base_receipt['PASS_execution_and_saved_replay']
    baseline_seal=read(check(base_receipt['baseline_seal']));check(baseline_seal['scores']);check(baseline_seal['results'])
    diagnosis_seal=V/'saved_source_failure_diagnosis/SEAL.json';ds=read(diagnosis_seal);assert ds['completed']
    diagnosis=read(V/'saved_source_failure_diagnosis/RESULT.json');assert diagnosis['PASS_saved_diagnosis']
    with np.load(check(fit['scores']),allow_pickle=False)as f:s={k:f[k]for k in f.files}
    with np.load(check(baseline_seal['scores']),allow_pickle=False)as f:v={k:f[k]for k in f.files}
    assert len(s['query_logit'])==33984 and np.array_equal(s['query_logit'],s['base_logit']+s['delta_logit'])
    for vk,sk in [('label','scalar_target_D0_EorS1_Uminus1'),('available','query_available'),('typed_state','typed_state'),('domain','domain'),('source','source'),('split','split'),('original_split','original_split'),('variant','variant'),('appearance','appearance'),('aspect','aspect'),('input_unit_index','input_unit_index'),('query_xy','query_xy_raw'),('L','L_raw')]:assert arrayeq(v[vk],s[sk]),(vk,sk)
    assert np.array_equal(v['occurrence_index'],s['global_occurrence_index'])
    assert np.array_equal(v['occurrence_id'][:NOLD],s['occurrence_id'][:NOLD])
    assert np.array_equal(v['occurrence_id'][NOLD:],np.array([x.replace('::AUX',':AUX')for x in s['occurrence_id'][NOLD:]]))
    assert np.array_equal(v['old_pairs'],s['old_accepted_triplets'])and np.array_equal(v['new_pairs'],s['new_selected_triplets'])
    a=s['query_available'];pred=dict(v61=np.where(a,np.where(v['raw_visible_logit']>=0,0,1),-1).astype(np.int8),C1=np.where(a,(s['base_logit']>=0).astype(np.int8),-1),C5=np.where(a,(s['query_logit']>=0).astype(np.int8),-1))
    assert np.array_equal(pred['v61'],v['decoded_aux_prediction'])and np.array_equal(pred['C5'],s['prediction'])
    labels=s['scalar_target_D0_EorS1_Uminus1'];de=np.isin(s['typed_state'],['D','E']);val=s['split']=='VALIDATION';comparison={};pairtables={}
    for domain in('V96_REAL','V99_SYNTH'):
        comparison[domain]={};pairtables[domain]={};pp=s['old_accepted_triplets'if domain=='V96_REAL'else'new_selected_triplets']
        for split in('TRAIN','VALIDATION'):
            mask=(s['domain']==domain)&(s['split']==split)&de
            comparison[domain][split]={k:cm(labels,p,mask)for k,p in pred.items()}
            pm=s['split'][pp[:,0]]==split;pairtables[domain][split]={k:pair(p,pp,pm)for k,p in pred.items()}
    expected={'V96_REAL':{'v61':[[25,127,0],[11,110,0]],'C1':[[147,5,0],[117,4,0]],'C5':[[141,11,0],[110,11,0]]},'V99_SYNTH':{'v61':[[37,172,0],[5,29,0]],'C1':[[189,20,0],[32,2,0]],'C5':[[166,43,0],[1,33,0]]}}
    for dom,models in expected.items():
        for name,matrix in models.items():assert comparison[dom]['VALIDATION'][name]['matrix']==matrix
    assert [pairtables['V96_REAL']['VALIDATION'][m]['all3']for m in pred]==[8,2,7]
    assert pairtables['V99_SYNTH']['VALIDATION']['v61']['all3']==7 and pairtables['V99_SYNTH']['VALIDATION']['C5']['all3']==26
    axes={}
    npairs=s['new_selected_triplets'];npval=s['split'][npairs[:,0]]=='VALIDATION'
    for axis in('appearance','aspect'):
        axes[axis]={}
        for value in np.unique(s[axis][NOLD:]):
            mask=(s['domain']=='V99_SYNTH')&val&de&(s[axis]==value);pm=npval&(s[axis][npairs[:,0]]==value)
            axes[axis][str(value)]={k:dict(DE=cm(labels,p,mask),selected_pairs=pair(p,npairs,pm))for k,p in pred.items()}
    oldvalpairs=s['old_accepted_triplets'][s['split'][s['old_accepted_triplets'][:,0]]=='VALIDATION'];basequeries=np.unique(oldvalpairs[:,0]//33)
    oldmanifest=read(read(V/'source_fit_preparation/FIT_PROTOCOL_BEFORE_SOURCE_HEAD_FORWARD.json')['bindings']['old_manifest']['path'])
    lineage=[json.loads(x)for x in Path(check(oldmanifest['input_lineage'])).read_text().splitlines()]
    imageN=len({lineage[int(x*33)]['original_image_sha256']for x in basequeries});assert len(basequeries)==18 and imageN==9
    known=s['candidate_known'];allrow=dict(total_occurrences=33984,old_occurrences=31680,new_occurrences=2304,typed_counts={t:int((s['typed_state']==t).sum())for t in('D','E','S','U')},teacher_U=int((~known).sum()),factual_prediction_U=int((~a).sum()),known_prediction_U=int((known&~a).sum()),by_domain_split=report['counts'])
    auximages=[V/'saved_v99_contacts/V99_FIXED_BASE_016_SAVED_CONTACT.png',V/'saved_source_failure_diagnosis/case_00_v61_comparison.png',V/'saved_source_failure_diagnosis/case_01_v61_comparison.png']
    assert all(x.is_file()for x in auximages)
    bindings=dict(FIT_COMPLETE=b(RUN/'FIT_COMPLETE.json'),MODEL_FREEZE=fit['MODEL_FREEZE'],C5_scores=fit['scores'],C5_results=fit['source_results'],C5_frozen_gates=fit['source_gates'],history=fit['history'],TRAIN_trace=fit['train_updates'],audit=b(V/'independent_saved_review/REVIEW.json'),audit_seal=b(V/'independent_saved_review/SEAL.json'),A_model_preparation=b(V/'model/PREPARATION_SEAL.json'),C_trainer_protocol=b(V/'source_fit_preparation/FIT_PROTOCOL_BEFORE_SOURCE_HEAD_FORWARD.json'),v61_receipt=b(V/'v61_source_baseline/FINAL_BASELINE_RECEIPT.json'),v61_baseline_seal=base_receipt['baseline_seal'],v61_scores=baseline_seal['scores'],v61_saved_replay=base_receipt['saved_replay'],B_failure_diagnosis=b(V/'saved_source_failure_diagnosis/RESULT.json'),B_failure_seal=b(diagnosis_seal),latest_user_steering=b(V/'ROOT_USER_STEERING_EXTERNAL_ONLY.json'),A_V99_renderer=b(P/'query_visibility_diverse_expected_cue_source_20261023_v99/render_preparation/FINAL_PREPARATION_AND_CORPUS_SEAL.json'),B_V99_matching_features=b(P/'query_visibility_diverse_expected_cue_source_20261023_v99/matching_feature_pack/INPUT_PACK_SEAL.json'),B_V99_geometry_teacher=b(P/'query_visibility_diverse_expected_cue_source_20261023_v99/geometry_query_preparation/GEOMETRY96_SEAL.json'),fixed_images=[b(x)for x in auximages])
    data=dict(completed=True,report_language='ko',main_conclusion_KO='단순 합성 외부 COVER는 구분했지만 실사 기반 외부 가림은 해결하지 못했습니다.',candidate='C5 frozenC1+central32RGBresidual',selected_update=report['selected_update'],optimizer_updates=2000,trainable_parameters=3221,frozen_parameters=26573,total_parameters=29794,source_only=True,model_gate_PASS=False,original_gate_summary=dict(PASS=sum(x['PASS']for x in gates['gates'].values()),FAIL=sum(not x['PASS']for x in gates['gates'].values()),gates_unchanged=True),latest_user_workflow=['initialq+existinggeometry','geometricSELFfilter','learnedvisibleDvsEXTERNAL_E'],DE_only_comparison_scope='Retrospective saved-output diagnostic on known typedD/E rows; excludes typedS/U only from these tables. Not an external-only refit, reselection or changedC5gate.',decoder=dict(v61='Original visible_logit>=0 D, including signed0; <0hidden; unavailableU',C1_C5='Hiddenlogit<0 D, >=0hidden; unavailableU; base/delta are diagnostic only'),same33984_rows_q_L_folds_units_available=True,v61_new_id_alias='Frozen v61 generated IDs use :AUX; C5 uses ::AUX; same domain occurrence index/image/AUX/qL join verified',DE_only_comparison=comparison,selected_triplets=pairtables,new_crossed_axes_DE_only=axes,all_original_mixed_S_rows_and_counts=allrow,all13_historical_frozen_gates=gates,original_9cohort_selector=report['VAL_cohorts'],failure_diagnosis=diagnosis,old_VAL_accepted_pair_clustering=dict(rank_variants=119,distinct_original_queries=18,distinct_original_images=9,all_PALLOX=True,LOCO_controlled_pairs=0),old_natural_E_VAL_N=2,current223_new_access_in_this_phase=0,current223_new_evaluation=0,latest_actual_current_evaluation='V94_C1 unchanged',new_PnP=0,fully_solved=False,native8_mapping='UNKNOWN',human_gold=False,source_emulated_q=True,limits_KO=['V96는공개실사RGB에외부물체를합성한통제쌍이며자연가림일반성공을뜻하지않습니다.','V99는동일팔레트메시에단순큐빅물체와공개배경을합친RGBA소스입니다.새팔레트인스턴스/실사전이증거가아닙니다.','AUX조건부주석/유한기하witness는인간native8정답이아닙니다.','반복소스개발/VAL사용으로blindgeneralization검증이아닙니다.','새합성모양/표면축은서로교차하며현재WOOD/SQUARE코호트와동등하지않습니다.','감사PASS는저장/연산/절차일치이며모델성능PASS가아닙니다.'],bindings=bindings,builder=b(__file__))
    assert data['original_gate_summary']['PASS']==7 and data['original_gate_summary']['FAIL']==6
    write(V/'ROOT_RESULTS_KO.json',data)
    labelsdisplay={'v61':'v61 원형','C1':'고정 C1 (저장 base)','C5':'C5 선택 step1900'}
    parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C5 소스 결과 — 외부 가림은 미해결</title><style>
@font-face{font-family:Nanum;src:url(file:///usr/share/fonts/truetype/nanum/NanumGothic.ttf)}*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#182b3e;font:16px/1.75 Nanum,sans-serif}main{max-width:1180px;margin:auto;padding:34px 32px 70px}h1{font-size:31px;line-height:1.4;margin:10px 0 18px}h2{font-size:23px;margin:42px 0 15px}h3{font-size:19px;margin-top:26px}p{margin:13px 0}.kicker{font-size:13px;letter-spacing:.06em;color:#556b80}.lead{font-size:19px;background:#fff;padding:20px 24px;border-left:5px solid #b3493d;border-radius:5px}.note,.card{background:white;padding:16px 20px;border-radius:8px;margin:16px 0}.note{border:1px solid #d4dce7}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.6;background:#fff}th,td{padding:10px 11px;border-bottom:1px solid #dce3eb;text-align:left;vertical-align:top}th{background:#e8eef5}td{font-variant-numeric:tabular-nums}.tablewrap{max-width:100%;overflow-x:auto;margin:14px 0 18px}a{color:#145c8c}figure{margin:22px 0}img{max-width:100%;height:auto;border:1px solid #d9e0e8;display:block;background:white}figcaption{font-size:14px;color:#506173;margin-top:9px}.muted{color:#506173;font-size:14px}code{font:13px/1.5 monospace;overflow-wrap:anywhere}.fail{color:#9e3026}.pass{color:#286450}details{margin-top:24px}summary{cursor:pointer;font-weight:bold}li{margin:6px 0}@media(max-width:700px){main{padding:20px 14px}h1{font-size:25px}table{font-size:12px}th,td{padding:7px}}</style><main>''']
    parts+=['<div class="kicker">V101 · C5 · 저장된 source 출력만 분석 · current223 새 평가 0</div>','<h1>단순 합성 COVER는 구분했지만<br>실사 기반 외부 가림은 해결하지 못했습니다</h1>',f'<p class="lead">C5는 C1을 고정하고 중앙 RGB32 잔차만 학습했습니다. 새 합성 VAL의 선택 COVER는 <b>32/32</b>를 맞혔지만, 실사 RGB 기반 통제 COVER는 <b>9/119</b>에 머물렀습니다. 기존 동결 기준은 <b class="fail">7/13 통과·6개 실패</b>이며, current223 실행이나 해결 선언으로 이어지지 않습니다.</p>',f'<p>한 번의 2,000 update에서 9개 VAL 코호트 평균 BCE의 최초 최소인 <b>step1900</b>을 선택했습니다. 26,573개 C1 파라미터는 고정하고 3,221개 잔차 파라미터만 학습했습니다. 원래 q/L·입력·분할을 유지했고 좌표를 보정하지 않았습니다.</p>','<div class="note"><b>최신 사용자 방향</b>: 초기 팔레트 추론 → 기존 기하학적 자기 가림(S) 필터 → 남은 query의 학습 D/E 판별. 다음 학습은 S를 제외합니다. 이미 수행한 C5의 D/E/S 혼합 학습·선택·13개 게이트는 그대로 남깁니다. 아래 D/E 표는 저장 결과의 <b>사후 진단</b>이며 외부 가림 전용으로 다시 학습한 모델이 아닙니다.</div>']
    parts+=['<h2>같은 소스 입력에서의 D/E 정량</h2>','<p>정답이 알려진 D/E 행에 한정한 비교입니다. S와 teacher-U는 이 표에서만 제외하며 원래 모든 행은 부록/JSON에 보존했습니다. 행렬의 행은 [D,E], 열은 [D,E,예측U]입니다. P/R도 같은 알려진 D/E 분모입니다. v61은 원래 visible logit ≥0을 D로 읽어 동점 규칙을 유지했습니다. v61의 원래 학습 표적은 C5의 물리 cue AUX와 다르므로 통제된 구조 ablation으로 해석하지 않습니다.</p>']
    for dom,title in [('V96_REAL','공개 실사 기반 V96 VAL — D152 / E121'),('V99_SYNTH','단순 합성 V99 VAL — D209 / E34')]:
        rr=[]
        for name,r in comparison[dom]['VALIDATION'].items():rr.append([labelsdisplay[name],pct(r['D_recall']),pct(r['E_recall']),pct(r['D_precision']),pct(r['E_precision']),pct(r['U_rate']),'<code>'+esc(str(r['matrix']))+'</code>'])
        parts+=['<h3>'+title+'</h3>',table(['모델','D 재현율','E 재현율','D 정밀도','E 정밀도','예측 U율','혼동행렬'],rr)]
    parts+=['<h2>BEFORE·COVER·SHAM을 모두 맞히는가</h2>','<p>동일 q/L에서 BEFORE=D, COVER=E, SHAM=D를 모두 맞혀야 쌍 성공입니다. 실사 기반 통제 쌍은 모두 PALLOX이며 LOCO 외부 통제 쌍은 0개로 미시험입니다.</p>']
    for dom,title in [('V96_REAL','V96 VAL: 119 rank 변형'),('V99_SYNTH','V99 VAL: 32개 선택 표적 triplet')]:
        rr=[]
        for name,r in pairtables[dom]['VALIDATION'].items():rr.append([labelsdisplay[name],f'{r["before_D"]}/{r["N"]}',f'{r["cover_E"]}/{r["N"]}',f'{r["sham_D"]}/{r["N"]}',f'{r["all3"]}/{r["N"]} ({pct(r["all3_rate"])})'])
        parts+=['<h3>'+title+'</h3>',table(['모델','BEFORE D','COVER E','SHAM D','세 단계 모두 정답'],rr)]
    oldtrain=pairtables['V96_REAL']['TRAIN'];parts+=['<p class="note">기존 실사 기반 COVER는 TRAIN에서도 C5 <b>4/75</b>에 불과합니다. 따라서 VAL 실패를 전이 문제 하나로 단정할 수 없습니다. v61은 VAL COVER <b>108/119</b>를 hidden으로 읽지만 SHAM D가 <b>18/119</b>뿐이며, triplet 전체 성공은 <b>8/119</b>입니다. 가림 물체에 대한 높은 반응만으로 이 cue 과제를 해결했다고 할 수 없습니다.</p>']
    parts+=['<h2>새 합성의 모양·표면별 결과</h2>','<p>다음은 C5의 알려진 D/E VAL 행만 본 주변 집계입니다. 두 축은 서로 교차하며 합산 대상이 아닙니다. 합성 직사각/정사각 mesh 변형을 현재 private WOOD/SQUARE 코호트와 동일시하지 않습니다. v61/C1의 같은 축 결과도 JSON에 보존했습니다.</p>']
    for axis,title in [('aspect','기하 모양'),('appearance','표면')]:
        rr=[]
        for value,models in axes[axis].items():
            r=models['C5'];x=r['DE'];pp=r['selected_pairs'];rr.append([esc(value),f'D{x["D_N"]} / E{x["E_N"]}',pct(x['D_recall']),pct(x['E_recall']),f'{pp["all3"]}/{pp["N"]}', '<code>'+esc(str(x['matrix']))+'</code>'])
        parts+=['<h3>'+title+'</h3>',table(['축 값','알려진 D/E','D 재현율','E 재현율','triplet 성공','혼동행렬'],rr)]
    parts+=['<h2>고정 source 그림과 실제 실패 crop</h2>','<figure><img loading="eager" src="saved_v99_contacts/V99_FIXED_BASE_016_SAVED_CONTACT.png" alt="사전에 고정한 첫 VAL base16의 세 단계와 원RGB96"><figcaption>사전에 고정한 첫 VAL base16. 같은 q/L의 원 RGB96을 확대한 보기입니다. 단순 procedural mesh·cuboid·공개 RGB 배경의 RGBA 합성이며, photorealism이나 실사 전이 성능의 증명이 아닙니다. AUX/native8 mapping은 UNKNOWN입니다.</figcaption></figure>','<p>아래 두 그림은 B가 사전 규칙으로 선택한 첫 TRAIN 실패 쌍과 첫 VAL 실패 쌍입니다. 실제 저장 RGB96을 그대로 표시하고 텍스트를 바깥에 붙였습니다. 결과에 맞춘 좌표·crop 재선택이나 모델 재호출은 없습니다.</p>','<figure><img loading="eager" src="saved_source_failure_diagnosis/case_00_v61_comparison.png" alt="고정한 TRAIN 실패 triplet"><figcaption>TRAIN 실패: 상대적인 COVER–SHAM 순서가 개선되어도 합계 logit이 0을 넘지 못하면 COVER를 D로 남깁니다.</figcaption></figure>','<figure><img loading="eager" src="saved_source_failure_diagnosis/case_01_v61_comparison.png" alt="고정한 VAL 실패 triplet"><figcaption>VAL 실패: v61의 높은 외부 물체 반응과 C5의 D 보존 성향을 같은 저장 crop에서 비교합니다. 단일 원인이나 올바른 native 부분 대응을 증명하는 그림은 아닙니다.</figcaption></figure>','<p>저장값에서 기존 TRAIN COVER의 필요한 양의 잔차 중앙값은 <b>11.960</b>, 실제 잔차 중앙값은 <b>3.246</b>입니다. VAL은 각각 <b>11.291</b>(base가 D인 행)과 <b>1.522</b>입니다. TRAIN의 67/75 쌍은 상대 margin ≥2를 만족하지만 그중 <b>63개</b>는 COVER를 여전히 D로 읽습니다. 이는 저장된 <code>z=base+delta</code>의 설명이며, base 고정이 실패의 유일한 원인이라는 개입 증거는 아닙니다.</p>']
    parts+=['<h2>해석의 한계와 현재 상태</h2>','<ul><li>기존 VAL 119개 통제 쌍은 <b>18개 원 query·9장 RGB</b>에 군집합니다. 서로 독립인 119개 장면이 아닙니다. 자연 BEFORE-E VAL은 <b>2개</b>뿐입니다.</li><li>V99는 같은 팔레트 자산의 시점·모양·표면·배경 변형입니다. 새 팔레트 인스턴스의 일반화나 실사 전이 검증이 아닙니다. q는 source-emulated projection+jitter이며 실제 initializer 산출 query가 아닙니다.</li><li>조건부 AI cue와 유한 기하 witness는 인간 native8 D/E/S gold가 아닙니다. 반복된 source 개발/VAL 노출도 공개합니다.</li><li>독립 감사 PASS는 저장된 조인·수식·샘플러·선택·게이트의 일치입니다. 과학적 일반화나 모델 gate 통과를 뜻하지 않습니다.</li><li><b>본 단계 current223 새 접근·평가 0, PnP 0.</b> 마지막 실제 현재 평가는 V94_C1 그대로입니다. 전체 목표는 미해결 상태입니다.</li></ul>']
    parts+=['<details><summary>원래 혼합 D/E/S 결과와 13개 동결 게이트</summary><p>S 평가는 기존 C5 실험의 역사적 진단이며 다음 학습 표적에서 요구하지 않습니다. 이 변경으로 과거 실패 게이트를 삭제하거나 C5를 통과로 바꾸지 않았습니다.</p>']
    rr=[]
    for dom in('V96_REAL','V99_SYNTH'):
        for split,r in report['counts'][dom].items():
            t=r['typed_states'];rr.append([dom,split,r['rows'],t['D'],t['E'],t['S'],t['U']])
    parts+=[table(['소스','분할','전체 행','D','E','S','teacher U'],rr),f'<p>전체33984행: teacher-U <b>{allrow["teacher_U"]}</b>, 입력 support에 따른 예측-U <b>{allrow["factual_prediction_U"]}</b>, 정답이 알려진 행의 예측-U <b>{allrow["known_prediction_U"]}</b>. 서로 다른 분모입니다.</p>']
    rr=[]
    for name,g in gates['gates'].items():rr.append([esc(name),pct(g['value']),('≤ 'if name.endswith('_max')else'≥ ')+pct(g['limit']),'<span class="'+('pass'if g['PASS']else'fail')+'">'+('PASS'if g['PASS']else'FAIL')+'</span>'])
    parts+=[table(['원래 동결 기준','값','기준','판정'],rr),'<p>기존 학습 SELF의 VAL hidden 반응은44/192입니다. 학습된 외부/자기 가림 원인 분류로 해석하지 않습니다. 최신 방향에서는 초기 추론 기하의 S 필터가 별도 역할을 맡습니다.</p></details>','<h2>근거 파일</h2>','<p>'+ ' · '.join([link('ROOT_RESULTS_KO.json','재계산 결과·분모·전체 binding'),link('independent_saved_review/REVIEW.json','독립 저장 감사'),link('saved_source_failure_diagnosis/RESULT.json','실패 진단'),link('ROOT_USER_STEERING_EXTERNAL_ONLY.json','사용자 최신 방향'),link('runs/C5_FROZEN_C1_CENTRAL_RGB_SOURCE_V1/MODEL_FREEZE.json','선택 checkpoint 동결')])+'</p>','<p class="muted">이 보고서 생성은 저장 점수/메타데이터와 기존 보고용 PNG만 사용했습니다. 모델·optimizer·backbone·private 자료·PnP를 실행하지 않았고, 부모 상태 포인터를 갱신하지 않았습니다.</p></main></html>']
    with(V/'ROOT_READOUT_KO.html').open('x')as f:f.write('\n'.join(parts))
    write(H/'BUILD_RECEIPT.json',dict(completed=True,builder=b(__file__),HTML=b(V/'ROOT_READOUT_KO.html'),JSON=b(V/'ROOT_RESULTS_KO.json'),same_saved_rows_joined=33984,metrics_recomputed_independently=True,source_gates_unchanged=True,input_images=bindings['fixed_images'],actual_model_features_fit_current_PnP=0))
    print(json.dumps(dict(HTML=b(V/'ROOT_READOUT_KO.html'),JSON=b(V/'ROOT_RESULTS_KO.json'),C1_new_VAL_triplets=pairtables['V99_SYNTH']['VALIDATION']['C1'],source_gates=data['original_gate_summary'])))

if __name__=='__main__':main()
