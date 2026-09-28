"""Prospective selection and six-fit closure allocation; no further hypothesis."""
from pathlib import Path
from . import common as C
from . import metric_baseline as M
from .student import protocol

def main():
    destination=C.DOC/'FINAL_SELECTION.json'
    if destination.exists(): print('SELECTION_ALREADY_FROZEN'); return
    candidates=[]; bindings=[]
    for cycle in ('A_INPUT_OCCLUSION','B_COORDINATE_SUPPLEMENT','C_EXPOSURE'):
        path=C.DOC/'cycles'/cycle/'RESULTS_PLASTIC_S42.json'; result=C.read(path)
        row=result['groups'][M.PRIMARY]['NEW_REF']; judgment=result['classification']['versus']['OLD_REF']
        candidates.append(dict(card_id=cycle,translation_cm=row['conditional']['translation_cm']['median'],
            rotation_deg=row['conditional']['rotation_deg']['median'],eligible_joint=judgment['eligible_joint'],
            added_inference_seconds=0,new_training_GPU_seconds=result['fit_seconds'],judgment=judgment))
        bindings.append(C.bind(path))
    chosen=M.choose_repeat([row for row in candidates if row['eligible_joint']])
    assert chosen=='C_EXPOSURE', 'Reinspect rather than silently choose a different posthoc candidate'
    source=C.read(C.DOC/'cycles'/chosen/'PROTOCOL.json')
    repeat_prior=C.ROOT/'scripts/research/pallet_type_selftrain_v1/recovery_repeat.py'
    selection=dict(created_utc=C.now(),selected_cycle=chosen,selected_material='PLASTIC',selected_seed=42,selected_target='REF',
        comparison='OLD_REF',repetition_cycle='RECIPE_REPEAT',control_repeat_cycle='BASELINE_REPEAT',Wood_cycle='WOOD_APPLICABILITY',
        repeat_seed=43,control_recipe='Original no-occlusion no-supplement RAW/REF main; same R0, actual training RNG43',
        status='REPEAT_AND_WOOD_PENDING',outcome='TINY_JOINT_SIGN_ONLY_NOT_PROMOTED',candidates=candidates,sources=bindings,
        criterion=C.bind(C.DOC/'METRIC_AND_SELECTION_LOCK.json'),
        reason='Only C has both negative median deltas vs OLD_REF with unchanged coverage. No AUC/PCK selection. Sign only, not meaningful gain.',
        limitations=['C seed42 is worse than R0 and matched NEW_RAW on both primary medians',
            'Old REF difference about0.03573cm and0.01210deg is tiny and may be training/reference variability',
            'All development reused; no independent physical pose truth or pretrained seed replication'],
        historical_seed_audit=dict(source=C.bind(repeat_prior),
            finding='Historical ORDER43/44 permuted indexed1024aliases with optimizer seed42. New repeats set actual trainer seed43 and retain original file aliases.'),
        further_method_search=False,additional_fits=6,additional_updates=1920,total_planned_fits=12,total_planned_updates=3840,
        teacher_fits=0,new_manual=0,loss_B_combination='NOT_RUN: B failed both OLD_REF primary medians; no useful second principle',
        gradient_scale_control='NOT_NEEDED_FOR_B_NO_GAIN; scale confounding disclosed, not identified causally')
    C.save(destination,selection,True)
    for cycle,material,training_seed,occlusion,role in [
        ('BASELINE_REPEAT','PLASTIC',43,False,'same-seed main control'),
        ('RECIPE_REPEAT','PLASTIC',43,True,'selected C recipe repeat'),
        ('WOOD_APPLICABILITY','WOOD',42,True,'selected fixed C rule transferred to Wood')]:
        pp,p=protocol(material); folder=C.DOC/'cycles'/cycle
        C.save(folder/'SPEC.md',f'''# {cycle} — {role}

새 주 가설이 아닌 사전 선택 후 확인이다. 선택: C의 median 부호만 joint이고 차이가 매우 작다. 이미 확인된 matched RAW 및 R0 대비 악화도 보존한다. 이 후속 결과를 보고 recipe/seed/threshold를 다시 고르지 않는다.

{material}, seed{training_seed}, RAW/REF 각320update, 동일 R0, 기존 real pool/512real+512source, AdamW1e−5, pose+flow-only, last checkpoint만 사용한다. 입력 가림={occlusion}; 위치 손실은 기존 그대로다. 가림이 있으면 C의 schedule1 및 나머지 A 파라미터를 그대로 사용하며 teacher나 support를 다시 선택하지 않는다.

학습난수43은 기존 ORDER43(optimizer42+순서별명 변경)과 다르다. 같은 pretrained R0의 augmentation/batch 학습변동 확인일 뿐 독립 pretrained model이나 새 recording TEST가 아니다. BASELINE_REPEAT와 RECIPE_REPEAT는 같은seed/데이터 기본변환을 사용하고 가림만 다르다. RAW/REF 각 쌍은 원래 좌표값만 다르다.

Wood는 기존361real 및45evaluation/38Clean+7Moderate, Severe없음을 유지한다. Plastic에서 고른 가림 규칙을 Wood 결과를 보기 전에 그대로 적용하며 teacher교체·threshold튜닝·평가제거는없다. 주 발견은Plastic99이고Wood는별도적용성이다.

세 가설 cycle은종료했다. 이 단계는예약된4repeatfit+2Woodfit 안에서닫는다. 추가manual0/teacherfit0/추론비용추가0. 모든seed와손익을보고하고모델자동승격하지않는다.
''',True)
        spec=dict(source,cycle=cycle,materials=[material],seeds=[training_seed],created_utc=C.now(),
            occlusion=occlusion,exposure=occlusion,source_cycle=chosen if occlusion else 'ORIGINAL_MAIN',
            parent_protocols={material:C.bind(pp)},initialization=p['initialization'],
            is_replication=material=='PLASTIC',is_material_applicability=material=='WOOD',
            selection=C.bind(destination),spec=C.bind(folder/'SPEC.md'),
            implementation=source['implementation']+[C.bind(Path(__file__))])
        C.save(folder/'PROTOCOL.json',spec,True)
    ledger=C.read(C.DOC/'RESOURCE_LEDGER.json')
    ledger['closure_reservation']=dict(selection=C.bind(destination),completed_hypothesis_cycles=3,
        remaining_fits=[dict(cycle='BASELINE_REPEAT',fits=2,updates=640),dict(cycle='RECIPE_REPEAT',fits=2,updates=640),
            dict(cycle='WOOD_APPLICABILITY',fits=2,updates=640)],method_search_closed=True)
    C.save(C.DOC/'RESOURCE_LEDGER.json',ledger)
    C.state('CONTROLLED_REPEAT_AND_WOOD',['A_complete','B_complete','C_complete','selection_frozen'],
        'Run BASELINE_REPEAT+RECIPE_REPEAT seed43 and fixed WOOD_APPLICABILITY seed42; no new method')
    print('SELECTED_FOR_CHECK_NOT_PROMOTION',chosen)

if __name__=='__main__': main()
