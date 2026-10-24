"""Concise Korean saved-only source readout. Run only after both root terminal notices."""
import argparse,hashlib,html,json,sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
C5=P/'query_visibility_local_residual_head_20261023_v101';NOLD=31680;N=33984
ARMS=('C6_EXTERNAL_ONLY_RAW_PRIOR','C7_EXTERNAL_ONLY_CLIPPED_PRIOR2');MODELS=('v61','C1','C6','C7')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(x):assert sha(x['path'])==x['sha256'];return Path(x['path'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def arr(p):
 with np.load(p,allow_pickle=False)as f:return {k:f[k]for k in f.files}
def eq(a,c):return np.array_equal(a,c,equal_nan=True)if np.asarray(a).dtype.kind in'fc'else np.array_equal(a,c)
def ratio(a,n):return float(a/n)if n else None
def cm(raw,p,mask):
 known=mask&np.isin(raw,['D','E']);y=(raw=='E').astype(np.int8);x=np.array([[np.sum(known&(y==k)&(p==v))for v in(0,1,-1)]for k in(0,1)],np.int64)
 return dict(matrix=x.tolist(),rows=['D','external_E'],columns=['D','external_E','U'],known_DE_N=int(x.sum()),D_N=int(x[0].sum()),E_N=int(x[1].sum()),D_recall=ratio(x[0,0],x[0].sum()),E_recall=ratio(x[1,1],x[1].sum()),D_precision=ratio(x[0,0],x[:,0].sum()),E_precision=ratio(x[1,1],x[:,1].sum()),U_N=int(x[:,2].sum()),U_rate=ratio(x[:,2].sum(),x.sum()),known_geometric_S_outside_task=int((mask&(raw=='S')).sum()),unresolved_teacher_U=int((mask&(raw=='U')).sum()))
def pair(p,pp,mask):
 q=p[pp[mask]];n=len(q);hit=np.all(q==[0,1,0],axis=1)
 return dict(N=n,before_D=int((q[:,0]==0).sum()),cover_E=int((q[:,1]==1).sum()),sham_D=int((q[:,2]==0).sum()),all3=int(hit.sum()),any_U=int((q<0).any(1).sum()),all3_rate=float(hit.mean())if n else None)
def esc(x):return html.escape(str(x))
def pct(x):return '—'if x is None else f'{100*x:.1f}%'
def num(x):return '—'if x is None else f'{x:.4f}'
def fraction(n,d):return f'{n}/{d}'
def table(headers,rr):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>'for x in r)+'</tr>'for r in rr)+'</tbody></table></div>'
def link(p,label):return '<a href="'+esc(p)+'">'+esc(label)+'</a>'
def main(c6sha,c7sha):
 assert not(V/'ROOT_RESULTS_KO.json').exists()and not(V/'ROOT_READOUT_KO.html').exists(),'Report outputs preserved; no silent overwrite'
 fit=[];reports=[];gates=[];saved=[];freezes=[]
 for arm,expected in zip(ARMS,(c6sha,c7sha)):
  path=V/'runs'/arm/'FIT_COMPLETE.json';assert sha(path)==expected;f=read(path);assert f['completed']and f['PASS_execution']and f['optimizer_updates']==2000
  for x in f.values():
   if isinstance(x,dict)and'path'in x and'sha256'in x:check(x)
  fit.append(f);reports.append(read(check(f['source_results'])));gates.append(read(check(f['source_gates'])));saved.append(arr(check(f['scores'])));freezes.append(read(check(f['MODEL_FREEZE'])))
  assert f['arm']==arm and freezes[-1]['arm']==arm and len(saved[-1]['query_logit'])==N
 basefit=read(C5/'runs/C5_FROZEN_C1_CENTRAL_RGB_SOURCE_V1/FIT_COMPLETE.json');base=arr(check(basefit['scores']))
 vreceipt=read(C5/'v61_source_baseline/FINAL_BASELINE_RECEIPT.json');assert vreceipt['PASS_execution_and_saved_replay'];vs=read(check(vreceipt['baseline_seal']));v61=arr(check(vs['scores']))
 s=saved[0];raw=s['raw_teacher_state'];available=s['query_available'];assert np.array_equal(np.unique(raw),['D','E','S','U'])
 for a in saved:
  assert np.array_equal(a['prediction'],np.where(a['query_available'],(a['query_logit']>=0).astype(np.int8),-1))
  assert np.array_equal(a['query_logit'],a['base_logit']+a['delta_logit'])
  for k in('global_occurrence_index','domain','source','split','original_split','variant','appearance','aspect','azimuth','elevation','raw_teacher_state','external_loss_candidate','query_available','input_unit_index','query_xy_raw','L_raw','occurrence_id','old_accepted_triplets','new_selected_triplets'):assert eq(a[k],s[k]),k
  assert np.array_equal(a['external_loss_candidate'],np.isin(raw,['D','E']))and np.sum(a['known_geometric_S_out_of_task'])==396 and np.sum(a['unresolved_teacher_U'])==32070
 for k,basekey in [('domain','domain'),('source','source'),('split','split'),('original_split','original_split'),('variant','variant'),('appearance','appearance'),('aspect','aspect'),('azimuth','azimuth'),('elevation','elevation'),('raw_teacher_state','typed_state'),('query_available','query_available'),('input_unit_index','input_unit_index'),('query_xy_raw','query_xy_raw'),('L_raw','L_raw')]:assert eq(s[k],base[basekey]),k
 for vk,sk in [('typed_state','raw_teacher_state'),('domain','domain'),('source','source'),('split','split'),('original_split','original_split'),('variant','variant'),('appearance','appearance'),('aspect','aspect'),('available','query_available'),('input_unit_index','input_unit_index'),('query_xy','query_xy_raw'),('L','L_raw')]:assert eq(v61[vk],s[sk]),vk
 assert np.array_equal(v61['occurrence_index'],s['global_occurrence_index'])and np.array_equal(v61['old_pairs'],s['old_accepted_triplets'])and np.array_equal(v61['new_pairs'],s['new_selected_triplets'])
 assert np.array_equal(v61['occurrence_id'][:NOLD],s['occurrence_id'][:NOLD])and np.array_equal(v61['occurrence_id'][NOLD:],np.array([x.replace('::AUX',':AUX')for x in s['occurrence_id'][NOLD:]]))
 pred=dict(v61=v61['decoded_aux_prediction'],C1=np.where(available,(base['base_logit']>=0).astype(np.int8),-1),C6=saved[0]['prediction'],C7=saved[1]['prediction'])
 assert np.array_equal(pred['v61'],np.where(available,np.where(v61['raw_visible_logit']>=0,0,1),-1))
 for a in saved:assert np.array_equal(a['reference_base_logit'],base['base_logit'])
 initial=[read(check(f['update0_parity']))for f in fit];assert initial[0]['initial_residual_state_sha256']==initial[1]['initial_residual_state_sha256']and reports[0]['all2000_draws_sha256']==reports[1]['all2000_draws_sha256']
 comparison={};pairs={};axes={}
 for domain in('V96_REAL','V99_SYNTH'):
  comparison[domain]={};pairs[domain]={};axes[domain]={};pp=s['old_accepted_triplets'if domain=='V96_REAL'else'new_selected_triplets']
  for split in('TRAIN','VALIDATION'):
   m=(s['domain']==domain)&(s['split']==split);comparison[domain][split]={k:cm(raw,p,m)for k,p in pred.items()};pm=s['split'][pp[:,0]]==split;pairs[domain][split]={k:pair(p,pp,pm)for k,p in pred.items()};axes[domain][split]={}
   for axis in(('source',)if domain=='V96_REAL'else('appearance','aspect','azimuth','elevation')):axes[domain][split][axis]={str(v):{k:cm(raw,p,m&(s[axis]==v))for k,p in pred.items()}for v in np.unique(s[axis][m])}
 oldm=s['domain']=='V96_REAL';newm=~oldm
 counts={dom:{sp:{t:int(((s['domain']==dom)&(s['split']==sp)&(raw==t)).sum())for t in('D','E','S','U')}for sp in('TRAIN','VALIDATION')}for dom in('V96_REAL','V99_SYNTH')}
 arms={}
 for name,f,r,g in zip(('C6','C7'),fit,reports,gates):
  assert len(g['gates'])==12 and g['all_gates_pass']==all(x['PASS']for x in g['gates'].values())
  arms[name]=dict(selected_update=f['selected_update'],criterion_eight_VAL_BCE=r['criterion_eight_VAL_cohorts_BCE'],VAL_cohorts=r['VAL_cohorts'],gates=g['gates'],all_gates_pass=g['all_gates_pass'],passed_gates=sum(x['PASS']for x in g['gates'].values()),MODEL_FREEZE=f['MODEL_FREEZE'],scores=f['scores'])
 qualifying=[k for k in('C6','C7')if arms[k]['all_gates_pass']];candidate=min(qualifying,key=lambda k:(arms[k]['criterion_eight_VAL_BCE'],('C6','C7').index(k)))if qualifying else None
 # Same already-fixed old TRAIN/VAL C5 cases; never choose new C6/C7 wins.
 policy=read(C5/'saved_source_failure_diagnosis/SCOPE_AND_SELECTION_BEFORE_RGB96_READ.json');cropmeta=read(C5/'saved_source_failure_diagnosis/NUMERICAL_AND_CROP_RESULTS.json');byindex={r['global_occurrence_index']:r for r in cropmeta['selected_crop_records']};fixed=[]
 zdisplay={'v61':v61['raw_visible_logit'],'C1':base['base_logit'],'C6':saved[0]['query_logit'],'C7':saved[1]['query_logit']}
 for case in policy['selections'][:2]:
  ix=case['indices'];assert len(ix)==3 and [raw[i]for i in ix]==['D','E','D']
  fixed.append(dict(kind=case['kind'],indices=ix,rows=[dict(index=i,raw_image=byindex[i]['exact_RGB96_png'],q=s['query_xy_raw'][i].tolist(),L=float(s['L_raw'][i]),teacher=str(raw[i]),predictions={k:int(v[i])for k,v in pred.items()},display_logits={k:float(v[i])for k,v in zdisplay.items()})for i in ix]))
 for c in fixed:
  for r in c['rows']:check(r['raw_image'])
 diagnosis=dict(question='Unbounded C1 raw prior versus fixed clamp[-2,+2], both external-only with identical residual initialization/data/draws/loss.',main_conclusion='No arm passes all12 source gates; no new current inference authorized.'if candidate is None else'At least one arm passes all12 prospective source gates; current inference still needs a separate exact root entry.',source_qualified_candidate=candidate,current_inference_authorized=False,latest_actual_current_model='V94_C1',goal_complete=False)
 bindings=dict(root_design=b(V/'ROOT_C6_C7_DESIGN_BEFORE_SOURCE_FORWARD.json'),trainer_protocol=b(V/'source_fit_preparation/FIT_PROTOCOL_BEFORE_SOURCE_FORWARD.json'),preparation=b(V/'source_fit_preparation/PREPARATION_SEAL.json'),C6_fit=b(V/'runs'/ARMS[0]/'FIT_COMPLETE.json'),C7_fit=b(V/'runs'/ARMS[1]/'FIT_COMPLETE.json'),C5_saved_base_reference=basefit['scores'],v61_saved_baseline=vs['scores'],original_fixed_crop_policy=b(C5/'saved_source_failure_diagnosis/SCOPE_AND_SELECTION_BEFORE_RGB96_READ.json'),fixed_crop_metadata=b(C5/'saved_source_failure_diagnosis/NUMERICAL_AND_CROP_RESULTS.json'))
 result=dict(completed=True,source_only=True,report_kind='Controlled external-only source phase, not current benchmark',diagnosis=diagnosis,counts=counts,arms=arms,external_DE_comparison=comparison,matched_triplets=pairs,source_axes=axes,fixed_old_source_cases=fixed,source_bindings=bindings,all33984_preserved=True,teacherS396_known_outside_learned_task=True,teacherU32070_unresolved=True,source_development_exposure=True,legacy_or_current_superiority_claim=False,no_new_model_forward_fit_private_current_PnP=True)
 write(V/'ROOT_RESULTS_KO.json',result)
 # Compact HTML; all detail tables are derived above and JSON stays exportable.
 css='''body{font:16px/1.65 system-ui,sans-serif;margin:0;color:#152536;background:#f2f5f8}main{max-width:1180px;margin:auto;padding:38px 24px 70px}h1{font-size:32px;line-height:1.3}h2{margin-top:34px;font-size:23px}h3{font-size:18px}p{max-width:1050px}.lead{font-size:19px}.card{background:white;border-radius:12px;padding:20px 24px;margin:18px 0;border:1px solid #dae2ea}.warn{border-left:5px solid #bf6227}.muted,figcaption{color:#536579;font-size:14px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:9px 10px;text-align:left;border-bottom:1px solid #dce4eb;white-space:nowrap}th{background:#e8eff5}.scroll{overflow:auto}code{font:12px ui-monospace,monospace}a{color:#075fa1}details{margin:16px 0}summary{cursor:pointer;font-weight:650}figure{margin:18px 0}.triplet{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.triplet img{width:100%;max-width:240px;image-rendering:pixelated}.PASS{color:#087248}.FAIL{color:#aa3828}@media(max-width:700px){main{padding:22px 14px}h1{font-size:25px}.triplet{grid-template-columns:1fr}}'''
 parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>C6/C7 외부 가림 전용 소스 비교</title><style>'+css+'</style><main><p class="muted">V102 · source-only · 원래 q/L 고정 · 목표 진행 중</p><h1>외부 가림 전용: C1 prior를 제한하면 달라지는가</h1>']
 if candidate is None:parts.append('<div class="card warn"><p class="lead"><strong>두 후보 모두 소스 12개 조건을 통과하지 못했습니다.</strong> 새 현재 데이터 추론은 허용되지 않으며, 마지막 실제 현재 비교는 V94 C1입니다.</p></div>')
 else:parts.append('<div class="card"><p class="lead">소스 조건을 통과한 후보는 <strong>'+esc(candidate)+'</strong>입니다. 현재 데이터 추가 추론은 별도 고정 entry가 필요하며, 이 보고서는 소스 결과만 다룹니다.</p></div>')
 parts.append('<p>C6는 원래 C1 logit, C7은 그 logit을 <code>[-2,+2]</code>로 제한한 prior를 사용합니다. 두 후보는 같은 3개 입력(P3 ROI·RGB96·valid), 같은 q/L, 같은 새 residual 3,221개 parameter 초기값, 같은 2,000회 D32/E32 draw와 loss를 사용했습니다. C1 26,573개 parameter는 고정했습니다.</p><p>사용자 방침에 따라 <strong>학습은 보임 D 대 외부 가림 E</strong>입니다. geometric S 396행은 알려진 별도 상태로 유지하고 학습·selector·gate에서 제외했습니다. unresolved U 32,070행도 음성으로 사용하지 않았습니다. 모델은 좌표를 찾거나 보정하지 않습니다.</p>')
 rr=[]
 for k,a in arms.items():rr.append([k,str(a['selected_update']),num(a['criterion_eight_VAL_BCE']),str(a['passed_gates'])+'/12','PASS'if a['all_gates_pass']else'FAIL'])
 parts.append('<h2>선택 결과</h2>'+table(['후보','선택 update / 2000','8-cohort VAL BCE','통과 조건','판정'],rr)+'<p class="muted">각 후보는 update0,100,…,2000에서 동일 8-cohort BCE의 엄격한 최소값을 선택했습니다. 후보 간에는 12개 조건을 모두 통과한 후보 중 BCE가 낮은 것을 택하고, 정확한 동률은 C6를 우선합니다. source VAL은 반복 개발에 노출되었습니다.</p>')
 for split,title in(('VALIDATION','소스 VAL'),('TRAIN','소스 TRAIN — 학습 내 실패와 전이를 구분')):
  rr=[]
  for domain,label in(('V96_REAL','기존 공개실사'),('V99_SYNTH','새 합성')):
   for name in MODELS:
    c=comparison[domain][split][name];rr.append([label,name,str(c['known_DE_N']),pct(c['D_recall']),pct(c['E_recall']),pct(c['D_precision']),pct(c['E_precision']),fraction(c['U_N'],c['known_DE_N']),'<code>'+esc(c['matrix'])+'</code>'])
  parts.append('<h2>'+title+'</h2>'+table(['source','모델','D/E N','D recall','E recall','D precision','E precision','U','CM'],rr))
 parts.append('<p class="muted">CM의 행은 GT [D,E], 열은 예측 [D,E,U]입니다. precision은 알려진 D/E 부분집합에 한정합니다. v61은 원래 visible logit ≥0을 D로 판정하며, C1/C6/C7은 hidden logit ≥0을 E로 판정합니다. v61의 원래 학습 목표와 이 조건부 물리 cue AUX 목표는 다르므로 소스 표만으로 legacy 우위를 주장하지 않습니다.</p><h2>같은 before / cover / sham</h2>')
 rr=[]
 for domain,label in(('V96_REAL','공개실사'),('V99_SYNTH','합성')):
  for split in('TRAIN','VALIDATION'):
   for name in MODELS:
    p=pairs[domain][split][name];rr.append([label,split,name,str(p['N']),fraction(p['before_D'],p['N']),fraction(p['cover_E'],p['N']),fraction(p['sham_D'],p['N']),fraction(p['all3'],p['N'])])
 parts.append(table(['source','fold','모델','쌍 N','before D','cover E','sham D','세 장 모두'],rr)+'<p class="muted">기존 공개실사 control 194쌍은 TRAIN 10개 base query의 75 donor-rank쌍, VAL 18개 base query의 119쌍으로 군집화되어 있습니다. 독립 194장면으로 해석하지 않습니다. 이 control은 모두 Pallox이며 LOCO control은 0입니다.</p>')
 parts.append('<details><summary>고정 12개 source gate</summary>')
 rr=[]
 for key in arms['C6']['gates']:
  row=[esc(key)]
  for name in('C6','C7'):
   g=arms[name]['gates'][key];row.append(num(g['value'])+' / '+num(g['limit'])+' <span class="'+('PASS'if g['PASS']else'FAIL')+'">'+('PASS'if g['PASS']else'FAIL')+'</span>')
  rr.append(row)
 parts.append(table(['고정 조건 (min/max는 식별자 참고)','C6 값 / 기준','C7 값 / 기준'],rr)+'</details><details><summary>소스 appearance / aspect별 VAL 결과</summary>')
 rr=[]
 for domain in('V96_REAL','V99_SYNTH'):
  for axis,groups in axes[domain]['VALIDATION'].items():
   if axis not in('source','appearance','aspect'):continue
   for value,ms in groups.items():
    for name,c in ms.items():rr.append([axis,esc(value),name,str(c['known_DE_N']),pct(c['D_recall']),pct(c['E_recall']),fraction(c['U_N'],c['known_DE_N'])])
 parts.append(table(['source 축','값','모델','D/E N','D recall','E recall','U'],rr)+'</details>')
 parts.append('<h2>이전에 고정한 공개실사 두 쌍에서의 절대 판정</h2><p class="muted">V101 C5 진단 때 이미 선택된 TRAIN/VAL 각 1쌍을 그대로 재사용했습니다. C6/C7 성공 여부로 사례를 바꾸지 않았습니다. 원래 RGB96과 q/L은 그대로이며 아래 수치만 새 저장 결과를 연결했습니다. 빈도나 단일 원인의 증거로 해석하지 않습니다.</p>')
 for case in fixed:
  parts.append('<div class="card"><h3>'+esc(case['kind'])+'</h3><div class="triplet">')
  for stage,r in zip(('BEFORE · D','COVER · E','SHAM · D'),case['rows']):parts.append('<figure><img src="'+esc(r['raw_image']['path'])+'" alt="'+stage+' 고정 원본 RGB96"><figcaption>'+stage+' · index '+str(r['index'])+'</figcaption></figure>')
  parts.append('</div>');rr=[]
  for stage,r in zip(('BEFORE','COVER','SHAM'),case['rows']):rr.append([stage]+[num(r['display_logits'][k])+' → '+('U'if r['predictions'][k]<0 else'D'if r['predictions'][k]==0 else'E')for k in MODELS])
  parts.append(table(['고정 단계','v61 visible z','C1 hidden z','C6 hidden z','C7 hidden z'],rr)+'</div>')
 parts.append('<h2>해석 범위</h2><p>clamp 비교는 prior의 범위를 바꾼 소스 통제 실험입니다. 특정 오류의 유일한 원인, 실제 팔레트의 native8 visibility 정답, 현재 평가 성능 개선을 증명하지 않습니다. 합성은 공개실사 배경에 RGBA를 합성한 단순화된 분포이며 q는 source-emulated입니다. 기존 공개실사는 주로 한 Pallox bin의 조건부 AI cue이고 natural E VAL은 2행입니다.</p><p>자기 가림은 초기 추론의 별도 기하 필터로 다룹니다. synthetic teacher S가 그 실제 기하 필터의 품질을 검증하지는 않습니다. 원래 좌표 오차와 대상 cue 결합 문제도 별도로 남아 있습니다. <strong>전체 목표는 아직 완료되지 않았습니다.</strong></p><p>'+link(V/'ROOT_RESULTS_KO.json','전체 수치 JSON')+' · '+link(C5/'saved_source_failure_diagnosis/RESULT.json','이전 source 실패 진단')+' · '+link(P/'query_visibility_matched_query_scalar_20261023_v94/ROOT_READOUT_KO.html','마지막 실제 현재 비교: V94 C1')+'</p></main></html>')
 with(V/'ROOT_READOUT_KO.html').open('x')as f:f.write(''.join(parts))
 write(H/'BUILD_RECEIPT.json',dict(completed=True,code=b(__file__),results=b(V/'ROOT_RESULTS_KO.json'),HTML=b(V/'ROOT_READOUT_KO.html'),bindings=bindings,all_same33984_q_L_split_order_checked=True,current_private_model_fit_PnP_reads_or_runs=0,root_terminal_fit_hashes=[c6sha,c7sha]))
 print(json.dumps(dict(results=b(V/'ROOT_RESULTS_KO.json'),HTML=b(V/'ROOT_READOUT_KO.html'),candidate=candidate,arms=arms),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--c6-fit-sha',required=True);p.add_argument('--c7-fit-sha',required=True);a=p.parse_args();main(a.c6_fit_sha,a.c7_fit_sha)
