"""Seal preparation artifacts only. Does not access any source/evaluation payload."""
from pathlib import Path
import ast,hashlib,json,os,stat
H=Path(__file__).absolute().parent
V=H.parent
P=V.parent
def need(ok,msg):
    if not ok:raise AssertionError(msg)
def rb(p):
    p=Path(p).absolute();need(P in p.parents and '..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Lexical project path')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for s in p.parts[1:-1]:
            n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file only')
    with os.fdopen(n,'rb')as f:return f.read()
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def read(p):return json.loads(rb(p))
def write(name,data):
    with(H/name).open('x')as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    pred=read(H/'PREDICTION_PREPARATION_COMPLETE.json');ev=read(H/'PREPARATION_COMPLETE.json')
    checks=[read(H/n)for n in ('GENERATED_INVENTED_CHECKS.json','GENERATED_MEANINGFUL_CHECKS.json','NEW_SOURCE_AND_SAVED_ROUTES_CHECKS.json')]
    need(all(x['completed']and x['PASS']for x in [pred,ev,*checks]),'All preparation checks pass')
    for d in (pred,ev):
        for k in ('code','protocol','checks','entry_template'):need(bind(Path(d[k]['path']))==d[k],'Current preparation binding '+k)
    pp=read(Path(pred['protocol']['path']));ep=read(Path(ev['protocol']['path']))
    need(ep['prediction_producer_code']==pred['code']and ep['prediction_producer_protocol']==pred['protocol']and ep['prediction_producer_preparation']==bind(H/'PREDICTION_PREPARATION_COMPLETE.json'),'Evaluator exact current predictor closure')
    future=('source_fit_protocol','source_fit_trainer','source_fit_complete','source_both_final_freeze','independent_source_fit_review')
    need(all(pp[k]is None for k in future),'Root actual source binding placeholders')
    need(read(Path(pred['entry_template']['path']))['approved_both_frozen_direct_prediction']is False and read(Path(ev['entry_template']['path']))['approved_saved_real_evaluation']is False,'Actual entry approval false')
    need(len(ep['models'])==14 and ep['primary_a_priori']=='C63'and ep['ablation_a_priori']=='C62','Fixed14/current model contract')
    need([d['count']for d in checks]==[18,24,15],'Prepared18+24+15 checks')
    schema=dict(completed=True,prepared_only=True,phase=str(V),primary_a_priori='C63',ablation_a_priori='C62',
      source_interface=dict(fit_complete_path=pp['source_fit_complete_path'],trainer_path=pp['source_fit_trainer_path'],actual_root_entry_required_bindings=list(future),
        source_implementation_ready_declared_by_A=dict(path=str(P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/FINAL_READY.json'),sha256='1ef23fb991e042b26c1670154f1764a8dd34c7a6eec8e943bd68f6574163a073'),
        trainer_declared_by_A=dict(path=pp['source_fit_trainer_path'],sha256='35854077f43180f7cb47504b21a448b08e4008764c98a9af119c1fdd6df345e0'),
        protocol_declared_by_A=dict(path=str(P/'query_visibility_dense_point_weak_bridge_source_training_20261025_v212/fit_preparation_a/PROTOCOL_BEFORE_DENSE_POINT_WEAK_BRIDGE_SOURCE_FIT.json'),sha256='fa0946967d4ec5ea845ba1df31a4f1e6961e31467d0a9e62c91f5463ee1c32c7'),
        declarations_not_opened_by_finalizer=True,required_review=dict(completed=True,PASS=True,fit_complete='exact actual source COMPLETE binding'),
        required_source_barrier=dict(completed=True,checkpoints='exact two arm checkpoint bindings'),parameters_each=118359,updates_each=4000,both_RAW=True,both_spatial_q8=True,
        arms={'C62_DENSE_POINT_RAW_Q8':dict(weak_bridge=False,weak_loss_weight=0.),'C63_DENSE_POINT_WEAK_BRIDGE_RAW_Q8':dict(weak_bridge=True,weak_loss_weight=.25)}),
      prediction=dict(seal_path=ep['prediction_seal_path'],domains={'CURRENT223_1784':dict(rows=1784,original_global_start=0,inputU=45),'PUBLIC_HELD45_360':dict(rows=360,original_global_start=584,inputU=108)},original_rows=2144,model_domain_rows=4288,output_count=4,
        packet_keys=['raw_support_logits','raw_owner_logits','raw_direct_end_external_logit','support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order','direct_end_order','row_id','original_global_index','native_slot','query_xy','L','original_query_available'],
        direct_rule='finite raw logit >=0 => E1; <0 => D0; original unavailable or nonfinite => U-1; raw is primary, no rounded sigmoid',
        context_rule=pp['context_contract'],all_new_predictions_before_annotation_reference_or_old_score_reads=True),
      evaluation=dict(output=ep['output'],complete_path=str(V/'saved_real_head_evaluation_v1/COMPLETE.json'),models=ep['models'],current_models=14,public_models=13,old_saved_models=ep['models'][:-2],
        old_twelve_saved_metrics_must_match_exactly=True,CM_orientation=dict(truth_rows=['D','E'],prediction_columns=['D','E','U']),
        current=dict(all_rows=1784,original_input_U=45,primary=dict(N=1237,D=1147,E=90),all_native_slots=list(range(8)),self_all=136,self_paired_D_vs_ES=21,existing_reference_LE3_humanD=461,manual_near_D=376),
        public=dict(rows=360,known_weak_center_owner=204,unknown_weak_center_owner=156,input_U=108,legacy='missing; no imputation',authority='weak center-owner only; not native/support gold'),
        metadata_and_all_existing_cohorts_unchanged=True,old_metric_complete_binding='Root actual saved-evaluation entry supplies exact V203 COMPLETE after new prediction seal',
        output_bindings='Actual COMPLETE.outputs relative filepath to {path,sha256}; current1784 and publicheld360 tables/ledgers preserved'),
      interpretation=dict(repeated_exposure_not_fresh_blind=True,source_includes_public_real_weak=True,finite_end_point_teacher_not_human_native_gold=True,model_selection=False,adoption=False,both_arms_carry_forward=True),
      actual_payload_header_or_asset_hash_or_body_reads=0,actual_model_weight_reads_or_forward=0)
    write('PREPARED_OUTPUT_SCHEMA.json',schema)
    ko='''# V213 고정 실사 평가 코드 준비

C63은 사전 지정 주 모델이고 C62는 dense-point만 사용하는 대조 팔입니다. 두 팔은 같은 C61 RAW·native·q8 구조(118,359개 파라미터)와 4,000회 갱신 상태를 사용합니다. C63의 source weak-bridge 가중치 0.25는 학습 프로토콜의 차이이며, 평가 시 입력·문턱·구조를 바꾸지 않습니다.

원래 RAW10/META6·F64 q/L·native8·input U를 유지합니다. 다른 7개 초기 query의 availability와 F64 상대좌표 계산도 V203 함수를 그대로 사용하며, GT endpoint·reference·라벨은 특징에 들어가지 않습니다. 두 SOURCE 최종 상태와 독립 exact-fit PASS를 먼저 확인하고, 두 실사 출력 전체가 봉인된 뒤에만 별도 entry로 저장 평가를 실행합니다.

현재 14개 모델과 public weak 13개 모델을 비교합니다. 기존 v61/legacy/C52~C61은 저장 출력만 사용하고 이전 12개 모델의 표는 완전히 같아야 합니다. 주평가 1,237개(D 1,147/E 90), 원1784/U45, self 전체136과 paired21, reference ≤3px human-D 461과 manual-near376을 구분합니다. Public360의 weak204/UNKNOWN156 및 legacy 결측도 보존합니다.

예측 가상검사 18개, 저장 평가 가상검사 24개, 새 source 봉인·역할·기존 C60/C61 경로 검사 15개가 통과했습니다. 원 입력·문맥·추론·순수 평가 커널의 함수 AST 동일성도 기록했습니다. 실제 데이터·가중치·라벨·점수·NPZ 헤더/본문을 열지 않았고, 실제 forward는 0회입니다. 두 actual entry는 approved=false입니다.

실제 실행은 root가 V212 최종 COMPLETE·독립 REVIEW·trainer/protocol·두 최종 checkpoint barrier의 정확 바인딩과 terminal exit0를 채운 후에만 가능합니다. 출처의 point/finite-end 약한 감독은 인간 native visibility gold가 아니며, 반복 노출 평가를 새 blind 검증이나 개선 원인 증명으로 해석하지 않습니다. 결과에 따른 모델 채택·선택은 하지 않습니다.
'''
    with(H/'KO.md').open('x')as f:f.write(ko)
    ready=dict(completed=True,PASS=True,prepared_only=True,scope=pp['scope'],prediction=pred,evaluation=ev,
      schema=bind(H/'PREPARED_OUTPUT_SCHEMA.json'),new_source_and_saved_routes_checks=bind(H/'NEW_SOURCE_AND_SAVED_ROUTES_CHECKS.json'),new_checks_code=bind(H/'check_new_source_and_saved_routes.py'),
      notes=bind(H/'KO.md'),finalizer=bind(Path(__file__)),check_counts=dict(prediction=18,evaluation=24,new_source_and_saved_routes=15),
      original_context_input_inference_metric_kernel_AST_unchanged=True,source_final_bindings_root_actual_entry_only=True,
      predictor_cli=str(pp['code']['path'])+' --entry ABSOLUTE_ROOT_ENTRY',evaluator_cli=str(ep['code']['path'])+' --entry ABSOLUTE_ROOT_ENTRY',pinned_python='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',
      actual_forward=0,actual_source_or_real_payload_header_hash_or_body_reads=0,actual_weight_or_score_or_annotation_reads=0,approved_actual_execution=False,
      initial_draft_provenance=bind(H/'provenance/INITIAL_DRAFT_BEFORE_NEW_WEAK_ROLE_SEAL_GUARD/PRESERVATION.json'),initial_derive_script_is_initial_draft_only=True,
      final_draft_delta='Explicit C62/C63 weak-role seal and source-fit guard; matching authored seal fixture; stale count/doc strings corrected before final seal. No input/scalar kernel/decoder change.',
      selected_model=None,adopted=False,fully_solved=False)
    write('FINAL_READY.json',ready)
    print(json.dumps(dict(completed=True,PASS=True,FINAL_READY=bind(H/'FINAL_READY.json'),predictor=pred['code'],prediction_protocol=pred['protocol'],evaluator=ev['code'],evaluation_protocol=ev['protocol'],schema=bind(H/'PREPARED_OUTPUT_SCHEMA.json'))))
if __name__=='__main__':main()
