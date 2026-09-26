from . import common as C

def main():
    r=C.read(C.stage(3)/'REAL_SELECTOR_RESULTS.json');d=C.read(C.stage(3)/'DECISION.json');s=C.read(C.stage(2)/'SYNTH_COMPATIBILITY_MATRIX.json')['combinations']
    groups=r['groups'];twod=r['twoD'];oracle=r['model_oracles'];tt=C.read(C.stage(3)/'SELECTOR_TRANSITIONS.json');stage1=C.read(C.stage(1)/'BASELINE_PARITY.json')['groups'];gain=C.read(C.stage(1)/'MANUAL_GAIN_DECOMPOSITION.json');cats=C.read(C.stage(1)/'SELECTOR_CATEGORY_COUNTS.json')
    lines=['# Model-conditioned selector compatibility after minimal hard supervision',
        '## 1. 한 줄 결론',
        f'`{d["Q_SELECTOR_COMPATIBILITY"]}` / `{d["Q_PIPELINE"]}`.',
        '**사전 기준을 통과했다. H_MANUAL + HMAN_SPECIFIC_GEO_LINEAR를 재사용 DEV에서의 최종 후보로 동결한다.** 실제 배포 설정을 교체한 것은 아니며, 독립 확인은 아직 없다. 새 hard label·키포인트 학습은 0회, 동일 구조 selector 학습은 2회다.',
        C.table(['Population','n','S1 + old GEO','H_MANUAL + old GEO','H_MANUAL + own GEO','vs S1 (%p)'],[[g,groups[g]['S1_OLD_GEO']['frames'],groups[g]['S1_OLD_GEO']['ADDsym_AUC'],groups[g]['H_MANUAL_OLD_GEO']['ADDsym_AUC'],groups[g]['H_MANUAL_HMANSPEC_GEO']['ADDsym_AUC'],100*(groups[g]['H_MANUAL_HMANSPEC_GEO']['ADDsym_AUC']-groups[g]['S1_OLD_GEO']['ADDsym_AUC'])] for g in C.PRIMARY]),
        '지표는 기존과 동일한 ADDsym AUC(0~1)다. `%p`는 AUC 차이에 100을 곱한 값이며, 프레임 정답률이나 상대 개선율이 아니다. 심한 난도 개선은 작으며 통계적 유의성 주장을 하지 않는다.',
        '## 2. 출발점',
        '기존 최선은 S1 + frozen old GEO_LINEAR였다. 최소 hard8장/direct36점으로 학습 완료된 H_MANUAL은 2D와 candidate oracle에서 이득을 보였으나, 같은 old GEO를 붙인 최종 6D 결과는 낮았다. 두 키포인트 모델은 재학습하지 않았다.',
        C.table(['Moderate model','D9','Old GEO','Posthoc oracle'],[[m,stage1['MODERATE'][m]['D9']['ADDsym_AUC'],stage1['MODERATE'][m]['current']['ADDsym_AUC'],stage1['MODERATE'][m]['oracle']['ADDsym_AUC']] for m in C.MODELS]),
        '## 3. Stage1 frozen failure decomposition',
        '기존 후보 pose의 reference metric을 새로 재계산하고 기존 발표값과 90개 비교를 통과했다. valid 두 후보의 ADDnorm 차이 1e-12 이하는 동률이다. 두 후보가 모두 나쁘다는 새 threshold는 만들지 않았으며 연속 best ADDnorm을 저장했다.',
        C.table(['Group','Model','Selector-recoverable','D9 right / GEO wrong','GEO right / D9 wrong'],[[g,m,cats[g][m]['selector'].get('SELECTOR_RECOVERABLE',0),cats[g][m]['compatibility'].get('D9_RIGHT_OLD_GEO_WRONG',0),cats[g][m]['compatibility'].get('OLD_GEO_RIGHT_D9_WRONG',0)] for g in C.PRIMARY for m in C.MODELS]),
        C.table(['Group','Oracle gain frames','Oracle loss frames','Unrealized manual gain','Their oracle AUC contribution','Their current AUC contribution'],[[g,gain[g]['categories'].get('ORACLE_GAIN',0),gain[g]['categories'].get('ORACLE_LOSS',0),gain[g]['unrealized_count'],gain[g]['unrealized_oracle_AUC_contribution'],gain[g]['unrealized_current_AUC_contribution']] for g in C.PRIMARY]),
        '![Selector categories](stage1/figures/01_selector_categories.png)',
        '[상세 1단계 보고서 및 사례 이미지](stage1/STAGE1_REPORT_KO.md)',
        '## 4. Synthetic controlled selector training',
        '순서: upstream SHA 잠금 → 기존 baseline 재현/분해 → H_MANUAL 합성 frozen inference(6,144장), S1 exact cache 재사용 → predictions/features lock → 기존 exact renderer parity 연결 → 모델별 TRAIN 정규화 → 동일 GEO_LINEAR 학습/VAL early-stop → 두 checkpoint lock → synthetic TEST 1회 → 실사 8조합 decision lock → reference 결합 및 평가.',
        '기존 renderer-group-disjoint TRAIN4096 / VAL1024 / TEST1024, seed20260925, RGB/K/치수/labels를 정확히 재사용했다. 두 모델 모두 TRAIN 유효 쌍 4095, VAL1024. 모델별 TRAIN 양 후보 공유 mean/std(floor1e-6), 94 features, 공유 Linear(94,1), lower-score selection, name tie-break. AdamW lr1e-3, weight_decay1e-4, batch256, seed42, max30epoch/patience5, earliest best VAL. S1 best epoch10/15, H_MANUAL best epoch15/20.',
        'Old GEO는 S0+S1 pooled(동일 frame당 두 모델 출력)로 학습된 control이고, 이번 모델별 selector는 해당 모델만 본다. 따라서 모델별 학습은 pooled보다 학습 출력 수가 절반이며, “분포 호환성만 완벽히 고립한 효과”라고 과장하지 않는다. 실제 가설 검정은 이 고정된 model-specific fitting 설정에 한정된다.',
        '![Synthetic VAL](stage2/figures/01_synth_val.png)',
        '## 5. Synthetic compatibility matrix',
        C.table(['Combination','Correct/1024','Parity accuracy','Brier'],[[k,v['correct'],v['accuracy'],v['brier']] for k,v in s.items()]),
        '합성 TEST에서는 H_MANUAL 전용이 old보다 낮다(936/1024 vs 952/1024). 이 결과로 추가 학습·설정 변경을 하지 않았다. 합성에서 개선됐지만 실사에서 안 된 SYNTH_REAL_COMPATIBILITY_GAP 분기는 해당하지 않는다.',
        '![Synthetic TEST](stage2/figures/02_synth_test_matrix.png)',
        '![Feature shift](stage2/figures/03_model_feature_shift.png)']
    for sec,g in [(6,'CLEAN'),(7,'MODERATE'),(8,'SEVERE')]:
        lines += [f'## {sec}. Real {g}',C.table(['Combination','AUC','Axis correct','Coverage','Selection loss','Selector correct'],[[k,v['ADDsym_AUC'],v['axis_correct_count'],v['pose_coverage'],v['selection_loss'],v['selector_correct_count']] for k,v in groups[g].items()]),
            C.table(['Model','Oracle AUC (GT-dependent)','Old→own wrong→correct','Old→own correct→wrong'],[[m,oracle[g][m]['ADDsym_AUC'],len(tt[g][m]['wrong_to_correct']),len(tt[g][m]['correct_to_wrong'])] for m in C.MODELS])]
    lines += ['![Current matrix](stage3/figures/01_current_auc_matrix.png)',
        '## 9. Selector vs candidate/localization decomposition',
        'H_MANUAL 전용 selector는 old 대비 중간 AUC +0.060905, 심함 +0.007750으로 두 hard 난도 모두 회복했다. 중간에서는 D9 및 S1-specific과 같은 0.490286이다. 따라서 “새 selector가 중간에서 D9보다 뛰어나다”는 주장은 불가하다. 심함에서는 D9 0.189327보다 높다.',
        'S1-specific control은 S1 old 대비 중간·심함 모두 낮다. H_MANUAL 전용의 회복은 관찰되지만, 전체 cross-table은 완전한 모델 특이성 증명을 뜻하지 않는다. 예를 들어 S1+HMAN-specific은 중간이 더 좋고 심함은 더 나빠서 별도의 새 winner로 선택하거나 난도별 routing을 만들지 않았다.',
        'H_MANUAL의 심함 selector correct count는 old와 같아도 AUC는 증가한다. 맞춘/틀린 frame의 수뿐 아니라 바뀐 ADD 값과 AUC 범위 내의 기여가 다르기 때문이다. Oracle AUC와 selector accuracy는 별개이며, 낮은 ADD가 반드시 W/D axis label 정답과 같은 것은 아니다.',
        '![Transitions](stage3/figures/06_wrong_to_correct.png)',
        '아래는 old→HMAN-specific의 실제 ADDnorm 변화가 큰 개선/악화 사례 각 최대3장이다. 사후 설명용으로 선택했으며, 학습 또는 checkpoint 선택에 사용하지 않았다. 초록 X=기존 annotation reference, 청록 점=raw keypoints, 노랑 선=PnP projection. 가운데·오른쪽의 raw keypoint는 동일하고 후보 선택만 다르다.']
    manifest=C.read(C.stage(3)/'EXAMPLE_MANIFEST.json')
    for row in manifest['rows']:lines += [f'### {row["category"]}: {row["id"]} (ΔADDnorm {row["ADDnorm_delta"]:+.4f})',f'![{row["category"]}](stage3/examples/{row["image"]["path"].split("/")[-1]})']
    rec=[g for g in groups if g.startswith('REC_')]
    lines += ['## 10. Per-recording',
        C.table(['Recording','Model','D9','Old GEO','S1-specific','HMAN-specific','Oracle','PCK10'],[[g,m,*[groups[g][m+'_'+s]['ADDsym_AUC'] for s in C.SELECTORS],oracle[g][m]['ADDsym_AUC'],twod[g][m]['PCK']['10']] for g in rec for m in C.MODELS]),
        '![Recordings](stage3/figures/07_recording_breakdown.png)',
        'REC_022의 candidate quality 손실과 REC_025의 oracle/current 차이도 같은 전체 표에 포함한다. 특정 recording을 제외하지 않았다. 자세한 selector correct count/margin은 [RECORDING_BREAKDOWN.json](stage3/RECORDING_BREAKDOWN.json)에 저장했다.',
        '## 11. 2D tail limitation',
        C.table(['Group','Model','PCK5','PCK10','PCK20','Median px','P90 px','Gross20','Detected','Matched','Missing'],[[g,m,*[twod[g][m]['PCK'][str(t)] for t in (5,10,20)],twod[g][m]['matched_pooled_corner8_median_px'],twod[g][m]['matched_pooled_corner8_P90_px'],twod[g][m]['gross20'],twod[g][m]['detected'],twod[g][m]['matched'],twod[g][m]['missing']] for g in C.PRIMARY for m in C.MODELS]),
        '특히 MODERATE raw P90은 S1 22.48px → H_MANUAL 57.34px로 악화됐다. selector 교체는 좌표를 수정하지 않으므로 이 큰 오차를 해결한 것이 아니다. 고정 verified HARD36도 S1 19/36 → H_MANUAL 24/36 그대로이며 selector variant별로 중복 이득을 세지 않는다.',
        '![Tail warning](stage3/figures/08_localization_tail_warning.png)',
        '### 6D error tails (all eight combinations)',
        C.table(['Group','Combo','R med/P90 °','Yaw med/P90 °','t med/P90 cm','IoU3D med/P90'],[[g,k,*[f'{v[n]["median"]:.3f} / {v[n]["P90"]:.3f}' if v[n]['median'] is not None else 'NA' for n in ('rotation_deg','yaw_deg','translation_cm','IoU3D')]] for g in C.PRIMARY for k,v in groups[g].items()]),
        '2D gained/lost10, 20→10 recovery, 5→10 damage 및 verified visible 전체 표는 [REAL_SELECTOR_RESULTS.json](stage3/REAL_SELECTOR_RESULTS.json)의 model-level 항목에 저장했다. Selector별 raw 2D는 동일하기 때문에 별도 증분으로 표시하지 않는다.',
        '## 12. Objective decision',
        f'Compatibility: `{d["Q_SELECTOR_COMPATIBILITY"]}`. Pipeline: `{d["Q_PIPELINE"]}`.',
        '사전 규칙: H_MANUAL own selector가 old 대비 hard 한 난도 strict 개선 + 다른 hard 비악화이면 compatibility recovery. 최종 pipeline은 S1 old 대비 Clean/Moderate/Severe 모두 비악화 + hard 적어도 하나 strict 개선. 이 조건을 모두 만족했다. 효과 크기 cutoff 또는 결과 기반 threshold sweep은 없다.',
        '**FINAL_CANDIDATE_ON_REUSED_DEV = frozen H_MANUAL + frozen HMAN_SPECIFIC_GEO_LINEAR.** 추가 키포인트 학습·selector sweep·hard annotation을 실행하지 않는다. 기존 baseline과 모든 결과는 보존한다.',
        '![Decision](stage3/figures/09_final_decision.png)',
        '## 13. Next one step',
        'Untouched/reserved recording이 실제로 남아 있는지 먼저 감사하고 독립 확인을 설계한다. 독립 데이터가 없을 때만 새 촬영을 논의한다. 이번 작업은 1~3단계까지이므로 추가 평가/촬영/학습은 자동 실행하지 않았다.',
        '## 14. Limitations',
        '- Real HELDOUT128은 recording-disjoint이지만 이미 반복 열람한 DEV이며 독립 final TEST가 아니다.\n- Synthetic TEST도 이미 연구에서 쓰인 development-heldout이고 base R0는 더 넓은 합성 population을 학습했다.\n- Single-seed keypoint models, hard8/direct36 pilot이며 일반화 성공을 확정할 근거는 부족하다.\n- Human PnP-assisted labels 및 기존 6D reference는 독립 physical full6D GT가 아니다.\n- 기존 94-feature GEO_LINEAR contract를 그대로 사용했고 selector는 synthetic only로 학습했다.\n- Candidate ORACLE는 POSTHOC / GT-dependent / NONDEPLOYABLE이다.\n- Camera-facing/180° symmetry role convention 경고가 남아 있다.\n- Selector는 missing detection, raw keypoint tail, 잘못된 candidate 자체를 복구하지 않는다.\n- Old pooled 대 model-specific 학습의 sample-output 수 차이를 통제한 추가 실험은 수행하지 않았다.\n- S1-specific 대조군/cross 조합을 숨기지 않았고 난도별 oracle routing이나 real-GT selector fit은 없다.',
        '따라서 결론은 “model-conditioned synthetic selector recovered the localization/candidate gains of the minimal-hard model on the reused recording-disjoint development set” 수준으로 제한한다. “hard supervision solved pallet pose” 또는 모든 코너 오차가 해결됐다는 주장은 하지 않는다.',
        '## Reproduction / audit',
        '[입력 bindings](INPUT_BINDINGS.json) · [고정 프로토콜](PROTOCOL_LOCK.json) · [Stage2 audit](STAGE2_AUDIT.json) · [최종 audit](FINAL_AUDIT.json) · [Stage3 상세](stage3/STAGE3_REPORT_KO.md).']
    C.save(C.DOC/'REPORT_KO.md','\n\n'.join(lines)+'\n')
    C.save(C.DOC/'FINAL_CANDIDATE_LOCK.json',dict(created_at=C.now(),status='FINAL_CANDIDATE_ON_REUSED_DEV',decision=C.bind(C.stage(3)/'DECISION.json'),
        keypoint_checkpoint=C.read(C.DOC/'INPUT_BINDINGS.json')['checkpoints']['H_MANUAL'],scorer=C.read(C.stage(2)/'SCORER_LOCK.json')['checkpoints']['H_MANUAL'],
        inference_contract=C.read(C.DOC/'INPUT_BINDINGS.json')['contract'],independent_confirmed=False,deployment_configuration_changed=False,no_more_training=True))
    print('INTEGRATED_REPORT_WRITTEN',flush=True)

if __name__=='__main__':main()
