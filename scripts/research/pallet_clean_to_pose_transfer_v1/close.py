"""End the bounded branch after its single predeclared actual-seed repetition."""
from . import common as C


def run():
    paths=[C.DOC/f'SELECTOR_PAIR_RESULTS_S{s}.json' for s in (42,43)]
    values=[C.read(p) for p in paths]
    rows=[]
    for seed,result in zip((42,43),values):
        for binding in result['sources']+result['private_artifacts']: C.verify(binding)
        g=result['groups']['NATURAL99']
        def med(arm,key): return g[arm]['conditional'][key]['median']
        coverage=g['REF_GEO']['valid_pose']==g['RAW_GEO']['valid_pose']==g['R0']['valid_pose']==99
        both_r0=coverage and all(med('REF_GEO',k)<med('R0',k) for k in ('translation_cm','rotation_deg'))
        both_raw=coverage and all(med('REF_GEO',k)<med('RAW_GEO',k) for k in ('translation_cm','rotation_deg'))
        rows.append(dict(seed=seed,REF_GEO_beats_R0_medians=both_r0,
            REF_GEO_beats_RAW_GEO_medians=both_raw,
            medians={arm:{k:med(arm,k) for k in ('translation_cm','rotation_deg')}
                     for arm in ('R0','RAW_D9','REF_D9','RAW_GEO','REF_GEO')},
            full_population=99,pose_coverage={arm:g[arm]['valid_pose'] for arm in g},
            P90={arm:{k:g[arm]['conditional'][k]['P90'] for k in ('translation_cm','rotation_deg')}
                 for arm in ('R0','RAW_GEO','REF_GEO')}))
    repeated=all(r['REF_GEO_beats_R0_medians'] for r in rows)
    corrected=all(r['REF_GEO_beats_RAW_GEO_medians'] for r in rows)
    ledger=C.read(C.DOC/'RESOURCE_LEDGER.json')
    assert ledger['totals']['student_fits']==6 and ledger['totals']['optimizer_updates']==1920
    assert ledger['totals']['selector_fits']==0
    audit_path=C.DOC/'REPEAT_AUDIT_S43.json'
    assert C.read(audit_path)['passed']
    decision=dict(status='FINAL',finalized=True,
        final_verdict='BEATS_R0_ON_OCCLUDED_POSE' if repeated else 'CLEAN_TO_OCCLUSION_TRADEOFF',
        replication_status='TWO_VALID_STREAMS_SAME_MEDIAN_DIRECTION' if repeated else 'MIXED_OR_NOT_REPLICATED',
        corrected_target_with_common_GEO='JOINT_MEDIAN_SIGNAL_BOTH_STREAMS' if corrected else 'MIXED_OR_NOT_REPLICATED',
        scope='Ordinary Plastic, reused DEV natural99. Median sign, not all-tail improvement or independent confirmation.',
        primary_D9_status='NO_JOINT_GAIN: original four-arm table preserved',
        per_seed=rows,resources=ledger['totals'],method_development_stopped=True,
        current_teacher_manual=dict(images=9,corners=38),
        full_reused_selector_pipeline_manual_union=dict(images=19,corners=86),
        newly_added_manual_coordinates=0,newly_added_training_RGB=0,
        branches=dict(
            selector=dict(status='FROZEN_OLD_GEO_ONLY',reason='기존 scorer에 의한 회복 신호가 있어 새 selector 학습은 불필요했다.'),
            bridge=dict(status='NOT_RUN',reason='selector-first 분기에 신호가 있어 LR·scope·노출 변경을 추가하지 않았다.'),
            seed=dict(status='COMPLETED_ONE_MATCHED_PAIR',reason='같은 OCC RAW/REF를 실제 다른 seed43 스트림으로 한 번만 반복했다. 결과와 무관하게 종료한다.')),
        report_sections=dict(
            selector='기존 frozen GEO를 같은 RAW/REF에 적용했다. keypoint 및 후보 pose를 다시 보정하지 않고 두 고정 후보 중 선택만 바꿨다. 새 selector 학습은0회다. 기존 D9 실패 표를 대체하지 않는다.',
            bridge='실행0회. 기존 GEO에서 회복 신호가 먼저 관찰되어 추가 LR·노출·학습범위 실험의 필요성이 낮아졌다. 예산이 남았어도 bridge를 실행하지 않았다.',
            seed='기존 R0 초기 가중치는 동일하며 데이터 순서·기본 증강·가림 RNG만 실제로 달라지는 seed43을 사용했다. 독립 초기화나 독립 평가셋을 추가한 실험이라고 부르지 않는다. 반복 결과가 약해도 추가 seed로 구제하지 않는다.',
            explained='기존217장은 전부 clean이 아니었다. 보정 타깃은 학생에게 일부 전달되며 Clean 평가에서 이득도 있었다. 그러나 D9의 자연 가림 T/R 동시 개선으로는 이어지지 않았다. 고정 후보 선택을 바꿔 회복할 수 있는 부분을 확인했다.',
            unresolved='과거 S1 성공의 단일 원인은 확정하지 못했다. 수동감독 예산·학습 범위·학습률·RGB 분포·가림 구현이 달랐다. 현재 clean78도 저앙각·잘림이 남고 대부분 야간이다. 의사 타깃의 물리적 정확도, 큰 꼬리오차, 새로운 recording에서의 독립 검증은 미확정이다.',
            next_step='추가 방법 개발을 멈추고, 원래 D9 결과와 수동감독 계보를 공개한 GEO 보완 결과를 구분해 보고한다. 일반화 주장이 필요하면 이 고정 경로의 독립 recording 평가를 별도 사전계획으로 수행하되 이번 작업에서 새 촬영·레이블링을 요구하지 않는다.'),
        result_paths=[str(p.relative_to(C.ROOT)) for p in paths]+[
            str((C.DOC/n).relative_to(C.ROOT)) for n in ('ZERO_FIT_SELECTOR_RESULTS.json',
                'SELECTOR_SUPERVISION_PROVENANCE.json','REPEAT_AUDIT_S43.json',
                'TRAIN_AUGMENTED_FOLLOWING.json')],
        inputs=[C.bind(p) for p in paths]+[C.bind(audit_path),C.bind(C.DOC/'REPLICATION_DECISION.json'),C.bind(__file__)],
        user_action_required=False)
    C.save(C.DOC/'FINAL_DECISION.json',decision,True)
    print(decision['final_verdict'],decision['replication_status'],decision['corrected_target_with_common_GEO'],flush=True)


if __name__=='__main__': run()
