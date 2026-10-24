"""Derive V213 report from exactly bound V203 code; no result or image reads."""
from pathlib import Path
import ast,hashlib,os,stat
H=Path(__file__).absolute().parent;P=H.parent.parent
OLD=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203/report_preparation_a/build_fixed12_and_korean_report.py'
def rb(p):
    assert 'v112'not in str(p).casefold();c=Path('/')
    for x in p.parts[1:]:
        c=c/x;assert not stat.S_ISLNK(os.lstat(c).st_mode)
    return p.read_bytes()
def fn(s,name,new):
    n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name==name)
    a=s.splitlines(keepends=True);return ''.join(a[:n.lineno-1])+new.rstrip()+'\n\n'+''.join(a[n.end_lineno:])
EXTRA=r'''
def complete_alias(review,expected):
 bindings=[review[k]for k in ('evaluation_complete','complete')if k in review]
 require(bindings and all(b==expected for b in bindings),'Independent scalar review exact evaluation binding, no conflicting alias')
 return True

def supplementary_tables(current,rows):
 t=current['tables'];refs=[]
 for name,predicate in [('HUMAN_D_EXISTING_REFERENCE_LE3',lambda r:r['human_state']=='DIRECT_VISIBLE'and number(r['initial_reference_error_px'])is not None and number(r['initial_reference_error_px'])<=3),('HUMAN_D_MISSING_REFERENCE',lambda r:r['human_state']=='DIRECT_VISIBLE'and number(r['initial_reference_error_px'])is None)]:
  rr=[r for r in rows if predicate(r)]
  if name=='HUMAN_D_EXISTING_REFERENCE_LE3':require(len(rr)==461,'Existing reference association461 is not manualnear376')
  for m in MODELS:
   c=Counter(r[m+'_D_E_U']for r in rr);refs.append(dict(association=name,model=m,N=len(rr),D=c['D'],E=c['E'],U=c['U'],scope='Existing reference association only; not coordinate error gold or primary gate'))
 return dict(native8=t['primary_native8_D_E_with_U'],type_U_inventory=t['all1784_model_inventory'],reference_association=refs,manual_near_D=t['manual_near_D'],coordinate_error_bins=t['coordinate_error_bins'],paired_recovery_damage=t['paired_recovery_and_damage'])
'''
GATE=r'''
def evaluation_gate(e,p):
 pred=completed(e['prediction_seal']);ev=completed(e['evaluation_complete'])
 require(pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']is True and pred['annotations_reference_GT_oldreal_scores_or_fullRGB_read']is False,'Both predictions sealed before labels')
 require(ev['prediction_seal']==e['prediction_seal']and ev['code']==p['evaluation_code']and ev['protocol']==p['evaluation_protocol']and ev['models']==MODELS and ev['old_twelve_primary_metrics_exact']is True,'Exact V213 all14 evaluation and old12 saved metrics')
 require(pred['source_fit_complete']['path']==p['source_fit_complete_path']and pred['original_rows2144']==2144 and pred['model_domain_rows4288']==4288,'Source finals/full original input census')
 require(set(pred['arms'])=={r[0]for r in p['new_arms'].values()},'Both new fixed arms')
 for a,weak,weight in p['new_arms'].values():
  r=pred['arms'][a];require(r['updates']==4000 and r['use_sam']is False and r['spatial_q8']is True and type(r['weak_bridge'])is bool and r['weak_bridge']is weak and r['weak_loss_weight']==weight,'Same C61 RAW/q8, fixed point/weak roles')
 review=completed(e['independent_numeric_review']);complete_alias(review,e['evaluation_complete'])
 return pred,ev,review
'''
PREP=r'''
def prepare():
 scope=read(SCOPE);require(scope['approved_actual_forward_or_evaluation']is False and scope['status']=='PROSPECTIVE_CODE_PREPARATION_ONLY','Prospective code-only scope')
 oldp=read(OLD_PROTOCOL);require(oldp['code']['sha256']=='c3560149cfd79fee14e8d3d1d662432d81c1a3e2fb9e7202c7bcfc234d75d3dc','Exact inherited report code');checked(oldp['code'])
 er=read(EVAL_READY);ep=read(checked(er['evaluation']['protocol']));require(ep['models']==MODELS,'All14 evaluator schema')
 source=read(SOURCE_PROTOCOL);require(source['primary_predeclared']=='C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8'and source['parameters_each']==118359 and source['training']['steps_each']==4000,'Declared source loss authority only')
 checks,card=invented();write(CHECKS,checks);card.save(H/'INVENTED_LAYOUT_ONLY.png')
 p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),checks=bind(CHECKS),inherited_visual_code=oldp['code'],inherited_visual_protocol=bind(OLD_PROTOCOL),evaluation_code=er['evaluation']['code'],evaluation_protocol=er['evaluation']['protocol'],evaluation_schema=er['schema'],source_fit_complete_path=ep['source_fit_complete_path'],fixed_original_ids=FIXED_IDS,fixed_rule=oldp['fixed_rule'],prior_gallery_complete_path=str(OLD/'fixed12_spatial_q8_real_visual_v1/COMPLETE.json'),output=str(OUT),models=MODELS,
  new_arms={'C62':['C62_DENSE_POINT_RAW_Q8',False,0.],'C63':['C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8',True,.25]},primary_a_priori='C63',ablation_a_priori='C62',
  original_clean96_source='Only exact V203 fixed12 CLEAN_RAW96/VALID96 PNG bytes via sealed gallery COMPLETE/manifest/selection; no source RGB/cache NPZ. Actual root entry only. Encoded bytes unchanged, decoded schema/hash recorded, clean card ROI exact.',
  future_metadata_whitelist='New sealed evaluation+prediction+independent scalar review; V203 gallery COMPLETE/manifest/selection/prediction-seal metadata; twelve bound CLEAN/VALID PNG pairs. Never follow source/model/cache payloads.',
  context_contract='Both same C61 RAW/native/original availability mask/original q8 F64 XY; no GT/reference/region/teacher feature, no crop/recenter/permutation changes.',
  source_protocol=bind(SOURCE_PROTOCOL),source_losses=source['losses'],source_point_mask_authority=source['masks'],source_weak_authority=source['weak_authority'],
  report_targets=['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'],limits=[
  'Existing human D/E/S remains authority; source procedural point and public weak visible-feature labels are distinct and are not human native gold.',
  'Repeatedly exposed current223/publicheld45, not fresh blind. Legacy realCAL history remains; v61 presence is a proxy, not an independently trained D/E classifier.',
  'Zero output U means the fixed decoder returned a finite decision, not confidence, calibration or solved uncertainty. Original inputU45/1784 remains distinct from primary U.',
  'Existing reference <=3px humanD461 is association only, distinct from manualnear376; no GT recrop or coordinate correction. Cyan marks initial q only; reference is numeric text, no reference marker.',
  'Self paired21 uses D_vs_ES; full humanSELF136/same-slot response is separate. Binary E/ownerF does not certify external-vs-self cause.',
  'Public360/U108 weakcenter204/UNKNOWN156 is separate, legacy owner missing; never native D/E gold.',
  'Twelve frozen descriptive cases, not representative/all1784 inspection. Rendering does not claim actual toolview or browser rendering.',
  'C63 primary and C62 ablation fixed before results. No arm adoption or model/data/architecture/threshold selection. No established performance/causal improvement or solved-goal claim.'
  ],selected_model=None,adopted=False,fully_solved=False)
 write(PROTOCOL,p)
 shared=dict(code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],prediction_seal=None,evaluation_complete=None,independent_numeric_review=None)
 write(H/'ROOT_ACTUAL_FIXED12_GALLERY_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(shared,approved_fixed12_saved_png_gallery=False,source_RGB_cache_NPZ_weights_models_or_forward_allowed=False,prior_gallery_complete=dict(path=p['prior_gallery_complete_path'],sha256=None)))
 write(H/'ROOT_ACTUAL_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(shared,approved_closed_saved_report=False,gallery_complete=None,actual_visual_review_bindings=[],root_confirmed_fixed12_actual_visual_QA=False,new_metrics_fit_or_source_arrays_allowed=False))
 ready=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],scope=p['scope'],gallery_entry_template=bind(H/'ROOT_ACTUAL_FIXED12_GALLERY_ENTRY_TEMPLATE_APPROVED_FALSE.json'),report_entry_template=bind(H/'ROOT_ACTUAL_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json'),invented_layout=bind(H/'INVENTED_LAYOUT_ONLY.png'),actual_source_or_real_metadata_array_header_weight_image_label_score_result_reads=0,new_model_forward_fit=0,automatic_retry=False)
 write(H/'FINAL_READY.json',ready);print(json.dumps(ready))
'''
REPORT=r'''
def report(entry):
 e,p=read_entry(entry,'approved_closed_saved_report');require(e['new_metrics_fit_or_source_arrays_allowed']is False,'Closed saved report only')
 pred,ev,qa=evaluation_gate(e,p);gal=completed(e['gallery_complete']);require(gal['evaluation_complete']==e['evaluation_complete']and gal['prediction_seal']==e['prediction_seal']and gal['exact_original_ids']==FIXED_IDS,'Exact current fixed12 gallery')
 data=read(checked(gal['summary']));require(data['models']==MODELS and data['selected12']==FIXED_IDS,'All14 model report schema')
 require(e['root_confirmed_fixed12_actual_visual_QA']is True and e['actual_visual_review_bindings'],'Separate root actual fixed12 visual QA required')
 visual=[dict(binding=b,receipt=completed(b))for b in e['actual_visual_review_bindings']]
 targets=[V/x for x in p['report_targets']];require(all(not x.exists()for x in targets),'Exclusive report targets')
 main=[x for x in data['primary42']if x['group']=='ALL'];by={x['model']:x for x in main}
 headline={m:dict(D_correct=by[m]['CM_D_E_by_D_E_U'][0][0],E_correct=by[m]['CM_D_E_by_D_E_U'][1][1],D_recall=by[m]['D_recall'],E_recall=by[m]['E_recall'],primary_U=by[m]['U'],primary_U_rate=by[m]['U_rate'])for m in ['C62','C63']}
 filelinks=[('고정12 gallery',gal['gallery']['path']),('gallery HTML',gal['HTML']['path']),('14모델 CM/P/R/U',OUT/'PRIMARY_ALL14_CM_P_R_U.csv'),('native8 원표',OUT/'PRIMARY_NATIVE8_ALL14.csv'),('전체1784 유형/U',OUT/'ALL1784_TYPE_U_INVENTORY.csv'),('SELF 두 cohort',OUT/'SELF_TWO_COHORTS_ALL14.csv'),('reference461/missing 연관',OUT/'REFERENCE_ASSOCIATIONS_ALL14.csv'),('manualnear376 원표',OUT/'MANUAL_NEAR_D376_ALL14.csv'),('기존 error-bin 연관',OUT/'COORDINATE_ERROR_ASSOCIATIONS_ALL14.csv'),('paired recovery/damage 원표',OUT/'PAIRED_RECOVERY_DAMAGE_ALL14.csv'),('public weak13',OUT/'PUBLIC_WEAK_CENTER_ONLY13.csv'),('저장 evaluator',ev['code']['path']),('평가 protocol',ev['protocol']['path']),('독립 수치감사',e['independent_numeric_review']['path']),('prediction 봉인',e['prediction_seal']['path']),('source 감독 protocol',p['source_protocol']['path']),('보고서 코드',__file__),('보고서 protocol',PROTOCOL)]
 links=[(label,relative(path,V))for label,path in filelinks]
 intro='C63은 결과 전 지정한 주 모델이고 C62는 point-only 대조 팔이다. 두 모델은 같은 C61 RAW/native/original q8 구조(118,359개 파라미터)로 각각 4,000회 학습했다. 두 SOURCE 최종 상태와 두 실사 출력을 먼저 봉인하고 같은 고정 평가에서 모두 비교했다. 결과에 따른 채택·선택을 하지 않는다.'
 quantitative='C62: D%d/1147·E%d/90, C63: D%d/1147·E%d/90. 저장 CM의 정답 수이며, 단일 원인의 효과나 성능향상 입증을 뜻하지 않는다.'%(headline['C62']['D_correct'],headline['C62']['E_correct'],headline['C63']['D_correct'],headline['C63']['E_correct'])
 authority='두 팔은 기존 old5 평균과 원영상 own-scene procedural point D/E BCE를 사용하고, C63만 공개 SOURCE TRAIN의 local-feature weak D/E bridge BCE를 가중치0.25로 추가한다. Point target은 자체 합성 장면의 점 상태이고 public weak target은 보이는 국소 feature 개입이다. Region/finite-end/center-owner를 인간 native D/E 정답으로 복사하지 않는다. 원 q/L/native/sampler를 유지하며 GT reference·라벨은 입력 특징이 아니다.'
 doc=['# V213 두 동결 dense-point head 실사 비교','',intro,'',quantitative,'',authority,'','주평가1237(D1147/E90)의 기존12+신규2모델 전체를 표시한다. CM 행은 기존 human D/E, 열은 D/E/U다. 원 inputU45/1784와 주평가 U율은 별개다.','',table_md(main),'','## 목재·정사각형별 주평가','',table_md([x for x in data['primary42']if x['group']!='ALL']),'','## SELF 두 별도 cohort','','paired21은 D_vs_ES eligibility이며 전체 humanSELF136/same-slot과 다르다. Binary hidden 응답은 SELF 학습판정이나 실제 차폐 원인 인증이 아니다.','',table_md(data['self28']),'','## reference 연관만','','기존 human-D/reference≤3px461과 manualnear376은 다르다. 아래 표는 위치오류 gold나 주평가 gate가 아니다.','',table_md(data['supplementary']['reference_association']),'','## public weak center-owner만','','원360/inputU108/known204/UNKNOWN156이다. v61 presence proxy 및 owner 응답을 native D/E 성능과 섞지 않는다. legacy owner는 결측을 유지한다.','',table_md(data['public_weak13']),'','## 고정 입력 관찰 범위','','V203에서 이미 고정한 동일12 crop/valid PNG를 원byte 그대로 사용했다. Cyan은 초기 q47.5 표시이고 기존 reference는 숫자로만 별도 표시한다. GT recenter는 없다. 실제 visual receipt '+str(len(visual))+'개를 인용하며, 렌더링과 실제 toolview 및 HTML browser 렌더링을 구분한다. 전체68/1784를 봤다고 주장하지 않는다.','']
 images=read(checked(gal['manifest']))['images']
 for im in images:
  if im['original_global_index']in ROOT6:doc += [f"![고정 CASE_{im['original_global_index']:04d}]({relative(im['card']['path'],V)})",'']
 doc+=['## 한계','',*['- '+x for x in p['limits']],'','## 표·코드·감사 연결','',*['- [%s](%s)'%(a,b)for a,b in links],'']
 md='\n'.join(doc);text(V/'ROOT_REPORT_KO.md',md);text(V/'PORTABLE_REPORT_KO.md',md)
 page='<!doctype html><meta charset="utf-8"><style>body{font:18px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}img{max-width:100%;height:auto}</style><h1>V213 고정14 모델 실사 비교</h1><p>'+html.escape(intro)+'</p><p>'+html.escape(quantitative)+'</p><p>'+html.escape(authority)+'</p>'+table_html(main)+'<h2>목재·정사각형</h2>'+table_html([x for x in data['primary42']if x['group']!='ALL'])+'<h2>SELF paired21 / full136</h2>'+table_html(data['self28'])+'<h2>기존 reference association</h2>'+table_html(data['supplementary']['reference_association'])+'<h2>public weak center-owner</h2>'+table_html(data['public_weak13'])+'<h2>한계</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>'for x in p['limits'])+'</ul><ul>'+''.join('<li><a href="'+b+'">'+html.escape(a)+'</a></li>'for a,b in links)+'</ul>'
 text(V/'ROOT_READOUT_KO.html',page)
 result=dict(completed=True,PASS=True,phase='V213',code=bind(__file__),protocol=bind(PROTOCOL),scope=p['scope'],prediction_seal=e['prediction_seal'],evaluation_complete=e['evaluation_complete'],independent_numeric_review=e['independent_numeric_review'],gallery_complete=e['gallery_complete'],actual_visual_review_bindings=e['actual_visual_review_bindings'],all_models14=MODELS,table_primary42=data['primary42'],table_SELF28=data['self28'],table_public_weak13=data['public_weak13'],supplementary=data['supplementary'],original_input_U=data['original_input_U'],primary1237=data['primary1237'],headline=headline,human_state_inventory=data['human_state_inventory'],fixed12_ids=FIXED_IDS,primary_a_priori='C63',ablation_a_priori='C62',source_authority=authority,source_protocol=p['source_protocol'],local_crop_cause_or_physical_native_gold_claim=False,source_results_selected_arm=False,repeated_exposure_not_freshblind=True,legacy_realCAL_history_preserved=True,v61_presence_proxy=True,zero_U_is_not_confidence=True,reference461_not_manual376=True,original_qL_native_and_sampler_unchanged=True,new_model_forward_or_fit=0,limits=p['limits'],links=[dict(label=a,relative=b)for a,b in links],selected_model=None,adopted=False,fully_solved=False)
 write(V/'ROOT_RESULTS_KO.json',result)
 n=sum(verify_links(V,(V/x).read_text(),x.endswith('.html'))for x in ['ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'])
 require((V/'ROOT_REPORT_KO.md').read_bytes()==(V/'PORTABLE_REPORT_KO.md').read_bytes(),'MD portable exact')
 write(H/'ROOT_REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),outputs={x.name:bind(x)for x in targets},relative_link_occurrences=n,actual_model_array_or_original_pixel_reads=0,browser_rendered=False,actual_visual_receipts_count=len(visual),selected_model=None,adopted=False,fully_solved=False))
 print(json.dumps(dict(completed=True,PASS=True,complete=bind(H/'ROOT_REPORT_BUILD_COMPLETE.json'))))
'''
def main():
 raw=rb(OLD);assert hashlib.sha256(raw).hexdigest()=='c3560149cfd79fee14e8d3d1d662432d81c1a3e2fb9e7202c7bcfc234d75d3dc'
 s=raw.decode().replace('V203 closed-result','V213 closed-result')
 s=s.replace("SCOPE=V/'ROOT_PROSPECTIVE_FROZEN_REAL_BENCHMARK_SCOPE.json'","SCOPE=V/'ROOT_PROSPECTIVE_DENSE_POINT_HEADS_FROZEN_BENCHMARK_SCOPE.json'")
 s=s.replace("OLD=P/'query_visibility_native_heads_frozen_real_benchmark_20261024_v200'","OLD=P/'query_visibility_spatial_q8_heads_frozen_real_benchmark_20261025_v203'\nOLD_PROTOCOL=OLD/'report_preparation_a/PROTOCOL_BEFORE_SAVED_RESULTS_OR_FIXED_PNG_BODIES.json'\nSOURCE_PROTOCOL=P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json'")
 s=s.replace("OLD_READY=OLD/'fixed_visual_preparation_a/FINAL_READY.json'\n",'')
 s=s.replace("EVAL_READY=V/'evaluation_preparation_a/FINAL_READY.json'","EVAL_READY=V/'inference_evaluation_preparation_b/FINAL_READY.json'")
 s=s.replace('fixed12_spatial_q8_real_visual_v1','fixed12_dense_point_real_visual_v1')
 s=s.replace("'C58','C59','C60','C61']","'C58','C59','C60','C61','C62','C63']")
 s=s.replace("(1600,1440)","(1600,1540)").replace("'V203 | original","'V213 | original").replace('24,1200+i*39','24,1300+i*39')
 s=s.replace('len(metric)==24','len(metric)==28').replace('All12 x2','All14 x2').replace('all12_ALLplus','all14_ALLplus').replace('all12_ALL','all14_ALL')
 s=s.replace('all eleven available owner routes','all thirteen available owner routes').replace('Every saved twelve-model','Every saved fourteen-model')
 s=s.replace('594+11*47+30<1200 and 1200+3*39+30<1440','594+13*47+30<1300 and 1300+3*39+30<1540').replace('Twelve_model_rows','Fourteen_model_rows')
 s=s.replace('len(pp)==36','len(pp)==42').replace("x['model']!='C61'","x['model']!='C63'")
 s=s.replace('len(st)==24 and st[0]', 'len(st)==28 and st[0]').replace("st[12]['N']","st[14]['N']")
 s=s.replace('Actual_public11_adapter','Actual_public13_adapter').replace('len(wt)==11','len(wt)==13')
 s=fn(s,'evaluation_gate',EXTRA+'\n'+GATE)
 s=fn(s,'prepare',PREP);s=fn(s,'report',REPORT)
 s=s.replace("old=completed(e['prior_gallery_complete']);require(e['prior_gallery_complete']['path']==p['prior_gallery_complete_path']and old['code']==p['inherited_visual_code']and old['protocol']==p['inherited_visual_protocol'],'Exact prior V200 gallery')", "old=completed(e['prior_gallery_complete']);require(e['prior_gallery_complete']['path']==p['prior_gallery_complete_path']and old['code']==p['inherited_visual_code']and old['protocol']==p['inherited_visual_protocol']and old['exact_original_ids']==FIXED_IDS and old['all12_original_encoded_and_decoded_clean_valid_parity']is True,'Exact prior V203 fixed12 gallery')")
 s=s.replace("oldentry=read(checked(old['entry']));require(oldentry['input_complete']==pred['input_complete']and old['all_selected_original_RGB96_valid96_png_parity']is True,'Same original raw input cache, old exact clean/valid PNG parity')", "oldpred=completed(old['prediction_seal']);require(oldpred['input_complete']==pred['input_complete']and oldpred['original_raw_complete']==pred['original_raw_complete'],'Same immutable original input caches; metadata only')")
 s=s.replace("priorselection=read(checked(p['prior_selection']));require(all(i in priorselection['unique_original_indices']for i in FIXED_IDS),'All12 belong to immutable old68')", "priorselection=read(checked(old['selection']));require(priorselection['original_global_indices']==FIXED_IDS,'Exact inherited12, no new score selection')")
 s=s.replace("prior_selection=p['prior_selection']","prior_selection=old['selection']")
 s=s.replace("require(len(byid)==68,'Old68 unique manifest images')","require(len(byid)==12 and sorted(byid)==FIXED_IDS,'Old fixed12 unique manifest images')")
 s=s.replace("src['clean_pixel_parity']is True", "src['input_available']==r['query_available']and src['original_clean_and_valid_encoded_and_pixel_parity']is True")
 s=s.replace("require(ash(rgb)==src['original_RGB96_array_sha256']and ash(valid)==src['original_valid96_array_sha256'],'Decoded original clean96/valid byte parity')", "decoded_rgb=ash(rgb);decoded_valid=ash(valid) # Exact encoded source PNG identity already checked; record decoded hashes without reopening original cache.")
 s=s.replace("clean96_card_ROI_exact=True))","clean96_card_ROI_exact=True,original_RGB96_array_sha256=decoded_rgb,original_valid96_array_sha256=decoded_valid))")
 s=s.replace('1440*len(cards','1540*len(cards').replace('(0,1440*k)','(0,1540*k)')
 s=s.replace('PRIMARY_ALL12','PRIMARY_ALL14').replace('SELF_TWO_COHORTS_ALL12','SELF_TWO_COHORTS_ALL14').replace('PUBLIC_WEAK_CENTER_ONLY11','PUBLIC_WEAK_CENTER_ONLY13')
 s=s.replace('primary36','primary42').replace('self24','self28').replace('public_weak11','public_weak13')
 s=s.replace("pt=pretty_primary(primary_tables(current));st=self_tables(current);wt=public_table(public)","pt=pretty_primary(primary_tables(current));st=self_tables(current);wt=public_table(public);supp=supplementary_tables(current,rows)\n   for key,name in [('native8','PRIMARY_NATIVE8_ALL14.csv'),('type_U_inventory','ALL1784_TYPE_U_INVENTORY.csv'),('reference_association','REFERENCE_ASSOCIATIONS_ALL14.csv'),('manual_near_D','MANUAL_NEAR_D376_ALL14.csv'),('coordinate_error_bins','COORDINATE_ERROR_ASSOCIATIONS_ALL14.csv'),('paired_recovery_damage','PAIRED_RECOVERY_DAMAGE_ALL14.csv')]:csvfile(OUT/name,supp[key])")
 s=s.replace('public_weak13=wt,original_input_U','public_weak13=wt,supplementary=supp,original_input_U')
 s=s.replace('# V203 고정12 저장 crop 진단','# V213 고정12 저장 crop 진단').replace('V203 fixed12 saved original crops','V213 fixed12 saved original crops').replace('PNG 원문과 픽셀은 V200과 같다','PNG 원문과 픽셀은 V203과 같다').replace('[12모델 주평가 CSV]','[14모델 주평가 CSV]')
 ast.parse(s)
 out=H/'build_fixed12_and_korean_report.py'
 with out.open('x')as f:f.write(s)
 print(hashlib.sha256(s.encode()).hexdigest())
if __name__=='__main__':main()
