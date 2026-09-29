"""Record evidence-limited human-readable closure, not a tuned success test."""
from pathlib import Path
from . import common as C


def main():
    if (C.DOC / 'FINAL_DECISION.json').exists():
        for b in C.read(C.DOC / 'FINAL_DECISION.json')['inputs']:
            C.verify(b)
        print('FINAL_DECISION_REUSED'); return
    result = C.read(C.DOC / 'FINAL_RESULTS.json'); g = result['groups']['NATURAL99']
    assert len(g) == 33
    strongest = 'OLD_REF_GEO'
    def med(arm, key):
        return g[arm]['full_population'][key]['median']
    def delta(a, b):
        return {k: med(b, k) - med(a, k) for k in ('translation_cm', 'rotation_deg')}
    dominating = [a for a in g if a != strongest and all(v < 0 for v in delta(strongest, a).values())]
    assert not dominating, 'Reconsider prose rather than silently changing a success threshold'
    ledger = C.read(C.DOC / 'RESOURCE_LEDGER.json')
    assert ledger['totals']['student_fits'] == 8 and ledger['totals']['selector_fits'] == 1
    assert C.read(C.DOC / 'TRAIN_CROSS_INPUT_RESULTS.json')['new_fits'] == 0
    assert C.read(C.DOC / 'CALIBRATION_INTEGRITY_AUDIT.json')['passed']
    assert C.read(C.DOC / 'CLEAR43_EVALUATION_AUDIT.json')['passed']
    key_pairs = [('R0_GEO', 'OLD_REF_GEO'), ('OLD_RAW_GEO', 'OLD_REF_GEO'), ('OLD_REF_D9', 'OLD_REF_GEO')]
    for seed in (42, 43):
        key_pairs += [(f'RAW_CLEAR_S{seed}_GEO', f'REF_CLEAR_S{seed}_GEO'),
                      (f'RAW_OCC_S{seed}_GEO', f'REF_OCC_S{seed}_GEO'),
                      (f'REF_CLEAR_S{seed}_GEO', f'REF_OCC_S{seed}_GEO'),
                      ('OLD_REF_GEO', f'REF_CLEAR_S{seed}_GEO'),
                      ('OLD_REF_GEO', f'REF_OCC_S{seed}_GEO'),
                      (f'RAW_OCC_S{seed}_NEWGEO', f'REF_OCC_S{seed}_NEWGEO')]
    key_pairs.append(('OLD_REF_GEO', 'R0_NEWGEO'))
    decision = dict(created_at=C.now(), status='COMPLETED_NEGATIVE_BOUNDED_EXTENSION',
        retained_fixed_reference=strongest, globally_best_or_deployment_guarantee=False,
        new_jointly_dominating_configurations=dominating,
        delta_vs_retained_strong_simple_reference=dict(translation_cm=0., translation_mm=0., rotation_deg=0.),
        interpretation='No replacement established. Retain the existing fixed reference; zero delta is not a newly improved method.',
        reference_vs_same_GEO_R0=delta('R0_GEO', strongest),
        correction_value_in_matched_old217=delta('OLD_RAW_GEO', strongest),
        closest_clear43_tradeoff_vs_reference=delta(strongest, 'REF_CLEAR_S43_GEO'),
        headline_ko='이번 추가 실험에서 기존217장 보정학생+기존GEO보다 위치·회전 중앙값을 함께 낮춘 구성은 없었다. 기존 고정 대조를 유지하고 입력가림·새GEO·clean78로의 교체는 채택하지 않는다. 전역최적 또는 실제배포 충분성을 입증한 것은 아니다.',
        component_conclusions_ko=[
            '- **유지:** 기존217장 REF_LR5 학생+기존GEO. 자연99 위치11.4101cm/회전4.9553°. 추가 실험 전 강한 대조 그대로이므로 그 대조 대비 새 개선량은0cm/0°다. 운영중인 모델파일은 자동 교체하지 않았다.',
            '- **자기학습:** 같은GEO의R0보다 위치중앙값−0.9931cm(−9.931mm), 회전−0.2623°. 41/99장은 둘다개선,23장은 둘다악화. 선택기만 붙인 R0와 구분되는 이득 신호지만 독립반복으로 필요성을 증명한 것은 아니다.',
            '- **좌표 보정:** 기존217장 RAW+같은GEO 대비−0.8419cm(−8.419mm)/−0.0872°. 두중앙값은 낮으나 회전효과가작고 T P90은 악화한다. clean78 CLEAR는seed42회전미세악화/seed43개선으로 일반화된 공동이득을 주장하지 않는다.',
            '- **입력 가림 제외:** 기존GEO의 REF OCC−CLEAR는seed42 T−0.1302cm/R+0.0609°, seed43 T+0.0802cm/R+0.0895°. 두흐름에서 R악화, T일관이득없음. 이설정의 가림필요성은 지지되지 않는다.',
            '- **기존GEO 유지, 새GEO 제외:** 기존217REF는 D9→기존GEO로T−1.0129cm/R−3.7394°. 단독효과를 보존한다. 새GEO에서OCC 보정효과는seed42두지표개선,seed43두지표악화로 반전했다. 선택기를 다시학습하지 않았다.',
            '- **clean78 교체 보류:** CLEAR43+기존GEO는 대조보다T−0.2260cm(−2.260mm), R+0.02994°의trade-off다. paired Δ중앙값은 오히려T+0.1354cm/R+0.0945°이며16장공동개선/43장공동악화다. seed43의 낮은T만 골라 업그레이드라고 하지 않는다.'
        ],
        tail_and_recording_ko=[
            '모든33구성은현재128/128자세를산출했다. 산출성공은정확함이아니다. 유지대조의자연99 T P90은127.1431cm/R P90은88.8204°로 큰오류가남는다. 같은GEO R0의T P90 120.4714cm, RAW217의119.3087cm보다 악화하므로 중앙값이득과함께보고한다.',
            '유지대조−R0의전체자연99 T중앙값은−0.9931cm지만 Moderate21은+0.4247cm, Severe78은+0.1850cm다. 결합분포의중앙값은 부분집합중앙값의평균이아니며 모든난도에일관된T개선으로읽으면안된다. Full128의R중앙값도+0.0145°로미세악화다.',
            '기록별로REC007은T/R−5.4558cm/−13.6628°인반면 REC022는T−3.8778cm/R+0.2524°이고 REC021은T−0.4286cm/R+0.0198°다. pooled와recording평균은구분하며 작은기록의영향을남긴다.',
            '[후보의 T-tail 상한 진단](CANDIDATE_TAIL_CEILING_KO.md): 명시한6구성 모두 두후보중 T만최선으로골라도P90은현재와같다. selected worst10중6~7장은이미T최선후보이며 기록구성은REC007 2/REC022 5/REC027 3이다. 더좋은선택만으로이P90을해결할수없다. T-onlyoracle는진단전용이며 R최솟값과합쳐가상자세를만들지않았다.'
        ],
        execution_and_stop_ko=[
            '순서: frozen동일GEO대조 보완 → 한종류합성source-only선택기재보정 → 빠졌던CLEAR43 RAW/REF 320update각1회 → 동일가림TRAIN입력 교차추론(학습0) → 최종33구성과오류사례정리. 기존음성결과를오류라고부르며재학습하지않았다.',
            '새선택기는합성TRAIN4096/VAL1024의현재RAW/REF CLEAR42 feature를사용했다. 동일Linear94→1, 기존pairwiseBCE, seed42, 기존학습규칙을유지했다. bestVAL0.9375로기존0.94043보다낮고실사반복도반전했으므로업그레이드로채택하지않았다.',
            '[동일 입력 교차 진단](TRAIN_CROSS_INPUT_KO.md): 모든8모델에같은62 real occurrence/45unique,476감독점(covered21/unmasked455)을입력했다. 같은maskedRGB에서OCC학습은가린점own-target평균오차를0.0247~0.0777px줄였지만나머지점이0.0113~0.0207px나빠져전체평균은0.0097~0.0174px악화했다. 가림을전혀학습하지않았다는주장도, 노출을늘리면실사T/R가개선된다는주장도지지하지않는다.',
            'mask노출은seed42 real2560회중418회이며가려진감독은632/19017점이었다. ROI기반placement확대는미검증가설로남기지만,같은maskedTRAIN의미세복원이자연pose개선을예측한다는근거가없다. 과거고빈도실패/학습량·LR·loss시도와현재tail상한까지읽고남은2fit을맹목적으로소진하지않는다. 미검증과불가능을구분한다.',
            '[후속 가설 종료 판단](NEXT_HYPOTHESIS_DECISION_KO.md)에추가자료가있다면필요한증거를남겼다. 추가촬영/레이블/새loss/새selector/다른estimator로확대하지않고현재배치를종료한다.'
        ], key_pairs=key_pairs, resource_totals=ledger['totals'],
        method_development_stopped=True, additional_manual=0, additional_RGB=0,
        evidence_status='REUSED_DEV_ONLY', independent_confirmation=False,
        automatic_background_resume=False, user_action_required=False,
        untested='ROI-conditioned masking coverage and other architectures are not proven impossible; no grounded next paired fit was selected.',
        inputs=[C.bind(C.DOC / n) for n in ('FINAL_RESULTS.json', 'FINAL_ROBUSTNESS.json', 'RESOURCE_LEDGER.json',
            'CALIBRATION_INTEGRITY_AUDIT.json', 'CLEAR43_EVALUATION_AUDIT.json', 'TRAIN_CROSS_INPUT_RESULTS.json',
            'CANDIDATE_TAIL_CEILING.json', 'NEXT_HYPOTHESIS_DECISION_KO.md')])
    # Insert existing approved public examples, never raw RGB/private target arrays.
    case_md = (C.DOC / 'CASES_FINAL.md').read_text()
    images = [line for line in case_md.splitlines() if line.startswith('![')]
    assert images, 'Approved improvement/worsening/tail examples required'
    decision['case_embeds_ko'] = images[:8]
    decision['inputs'].append(C.bind(C.DOC / 'CASES_FINAL.json'))
    C.save(C.DOC / 'FINAL_DECISION.json', decision, True)
    fit_path = C.ROOT / '_docs/experiments/pallet_type_selftrain_v1/selftrain_recovery_v1/pose_only/FIT_REF_LR5.json'
    fit = C.read(fit_path)
    geo_lock = C.read(C.DOC / 'SAME_GEO_CONTROLS_LOCK.json')
    associations = C.read(C.RAW / 'FINAL_CASE_BINDINGS_PRIVATE.json')
    config = dict(name=strongest, purpose='Retained fixed practical reference, not new improvement/global optimum/deployment guarantee',
        checkpoint=fit['checkpoint'], fit_record=C.bind(fit_path), training_protocol=fit['protocol'],
        actual_args=C.bind(C.ROOT / Path(fit['checkpoint']['path']).parents[1] / 'args.yaml'),
        initialization=C.read(C.OLD.DOC / 'PRIMARY_PROTOCOL.json')['initialization'],
        real_unique=217, real_source_slots_per_epoch=[512,512], optimizer='AdamW', lr=1e-5, updates=320,
        teacher_budget='9images/38manualcorners', historical_indirect_union='19images/86manualcorners with oldGEO',
        trainable='pose branches and flow only; backbone/detector/buffers fixed',
        input_occlusion=False, selector=geo_lock['same_frozen_scorer'],
        final_pose='Existing physical dimensions / D9 candidates / learned GEO chooses one complete pose / C2 reference',
        model_routing='None by severity/GT/seed. Ordinary plastic input contract remains fixed.',
        bindings=associations[strongest], cached_final_poses=C.bind(C.RAW / 'controls/OLD_REF/POSES.json'),
        original_main_inputs=C.bind(C.ROOT / '_docs/experiments/pallet_selftraining_paper_closure_v1/INPUT_BINDINGS.json'),
        runtime_teacher_needed=False, production_configuration_modified=False,
        verification_command='python -m scripts.research.pallet_clean_pose_minimal_v1.controls freeze',
        scoring_command='python -m scripts.research.pallet_clean_pose_minimal_v1.controls score')
    for b in (config['checkpoint'], config['training_protocol'], config['actual_args'], config['selector']): C.verify(b)
    C.save(C.DOC / 'FINAL_CONFIGURATION.json', config, True)
    C.save(C.DOC / 'STATE.json', dict(stage='COMPLETED_REPORT_AND_AUDIT_PENDING', method_development_stopped=True,
        automatic_background_resume=False, completed_expensive_stages=['sameGEOcontrols','single_selector','CLEAR43pair','sameTRAINcross'],
        retained_configuration=strongest, cumulative_resource_ledger=C.bind(C.DOC / 'RESOURCE_LEDGER.json')))
    print('FINAL_DECISION_RECORDED')


if __name__ == '__main__':
    main()
