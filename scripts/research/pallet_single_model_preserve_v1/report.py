from . import common as C
def main():
    base=C.read(C.DOC/'BASELINE_LOCK.json');par=C.read(C.DOC/'ZERO_INIT_PARITY.json');fit=C.read(C.DOC/'FIT.json');res=C.read(C.DOC/'RESULTS.json');aa=C.read(C.DOC/'VERIFIED_VISIBLE.json');src=C.read(C.DOC/'SOURCE_PRESERVATION.json');cost=C.read(C.DOC/'COMPUTE_COST.json');dec=C.read(C.DOC/'DECISION.json');audit=C.read(C.DOC/'AUDIT.json');gap=C.read(C.DOC/'SUPERVISION_GAP_DIAGNOSTIC.json') if (C.DOC/'SUPERVISION_GAP_DIAGNOSTIC.json').exists() else None
    lines=['# S1 + GEO_LINEAR — single-model clean-preservation pilot','',
        '## 1. 결론','',f"**{dec['primary']}**. 기준은 새 어댑터가 없는 frozen S1 + GEO_LINEAR다. Clean AUC 상승과 Moderate/Severe AUC 비감소를 모두 요구하며, 결과를 보고 효과크기 임계값을 추가하지 않았다.",'',
        '|group|BASE AUC|PRES1 AUC|Δ AUC|','|---|---:|---:|---:|']
    for g in ('CLEAN','MODERATE','SEVERE','ALL'):
        r=res['groups'][g];lines.append(f"|{g}|{r['BASE']['current']['ADDsym_AUC']:.8f}|{r['PRES1']['current']['ADDsym_AUC']:.8f}|{dec['delta_AUC'][g]:+.8f}|")
    lines+=['',f"verified HARD36 PCK10 감소 경고: {dec['verified_HARD_PCK10_warning']} (정답 개수 변화 {dec['hard_correct_delta']:+d}). 이 warning으로 6D 판정을 뒤집지 않는다. 이번 결과로 기존 production/final 모델을 교체하지 않았다.",'',
        '## 2. Baseline S1+GEO_LINEAR','',
        '원본 S1 가중치와 합성 학습 GEO_LINEAR 가중치/feature 계약을 그대로 고정했다. 실제 S1 재추론과 기존 캐시 일치를 확인했고, 기존 PCK/pose/verified FINAL_V2 baseline을 재현했다. S0+D9/R0/PRES1+D9는 보조표에만 포함한다. 기존128장은 recording-disjoint이지만 이미 열람한 DEV다.','',
        '## 3. Zero-init 구조와 parity','',
        f"설치된 Pose26의 end-to-end one2one keypoint projection `one2one_cv4_kpts` 직전 feature를 각 scale에서 사용했다. 독립 `Conv1x1(C,32) → SiLU → Conv1x1(32,27)` 3개이며 마지막 Conv weight/bias0. raw x/y 채널에만 더하고 visibility/conf 채널은 mask0이다. 추가 파라미터 **{par['adapter_params']:,}개**. backbone/neck/box/class/기존 keypoint/scorer는 전부 고정했다.",'',
        f"고정32장의 최대 xy 차이 **{par['max_xy_px']}px**. bbox/conf/선택검출/D9 pose/GEO_LINEAR 결정 동일. clean loss의 adapter gradient L1={par['adapter_gradient_L1']:.6f}, base grad0. 원래 Pose26 inference decoder의 in-place sigmoid/slice 연산은 학습 역전파에서 version 오류를 내므로, 학습 중만 동일 `(raw+anchor)*stride` / sigmoid의 functional 식으로 바꿨다. stock 대비 bit-exact 출력 검증을 통과했고 어떤 optimizer step 전에도 모델/손실 수식은 바꾸지 않았다.",'',
        '![parity](figures/01_zero_init_parity.png)','',
        '## 4. 고정 학습 계약','',
        '320 updates / batch16 /5epochs /seed42 /AdamW lr1e-3 wd1e-4 /last checkpoint only. 각 batch 4 real-clean +4 synth-clean +8 synth-occluded-preserve. 총 REAL1280 +SYNTH_CLEAN1280 +SYNTH_OCC2560=5120 occurrence. original S1의 immutable base-augmentation cache를 재사용하고, real-clean은 원래 Clean10 teacher pseudo xy/mask를 사용했다. DEV/anchor/T2/hard manual을 학습하지 않았다.','',
        '합성 clean256/epoch는 기존 source512의 SHA 고정 순서로 선택했다. 합성 preserve512/epoch는 기존 S1 random rectangle policy의 size/fill/schedule/coverage/paired-placement 조건을 cached640 canvas에서 적용했다. 계획은 모델 출력과 무관하게 먼저 고정했다. 적용 불가/미예약은 원래 policy처럼 clean RGB를 그대로 보존한다. 원본 source512의 모든 epoch 캐시가 hash 검증됐다.','',
        '좌표 task는 지시문에서 미지정된 구체식을 학습 전에 명시적으로 고정했다: frozen top1 검출의 decoded xy를 640-input pixel 단위 SmoothL1(beta1)로 감독하며, clean 유효 x/y scalar 수로 나눈다. 원래 mask v==2만 사용한다. 같은 occluded input의 frozen S1 top1 decoded9점 xy를 detach하여 동일 단위 SmoothL1로 보존한다. `L=L_clean+1.0*L_occ_preserve`. pose/RLE/LoRA/추가 선택기 loss는 없다.','',
        '|epoch|L_clean mean|L_occ mean|residual norm px|real / synth clean / synth preserve|','|---|---:|---:|---:|---|']
    for e in fit['epochs']:lines.append(f"|{e['epoch']}|{e['L_clean']:.6f}|{e['L_occ_preserve']:.6f}|{e['residual_xy_mean_px']:.6f}|256 /256 /512|")
    lines+=['','각 epoch는 서로 다른 cached augmentation을 쓰므로 loss의 epoch 간 차이를 같은 입력의 train-fit 개선으로 해석하지 않는다. magnitude cap/8px 제한은 없지만 보존 loss가 실제 이동을 작게 만들 수 있다. 결과에 맞춰 lr/epoch/lambda를 다시 고르지 않았다.','', '![loss](figures/02_training_losses.png)']
    for section,g,fname in [(5,'CLEAN','03_clean_recovery.png'),(6,'MODERATE','04_moderate_preservation.png'),(7,'SEVERE','05_severe_preservation.png')]:
        lines+=['',f'## {section}. {g}','', '|arm|PCK5|PCK10|PCK20|median px|P90 px|>20|CURRENT AUC|ORACLE AUC|selection loss|axis count|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for a,r in res['groups'][g].items():
            k=r['twoD'];p=r['current'];lines.append(f"|{a}|{k['PCK']['5']:.4f}|{k['PCK']['10']:.4f}|{k['PCK']['20']:.4f}|{k['full_penalty_median_px']:.3f}|{k['full_penalty_P90_px']:.3f}|{k['corners']-k['correct']['20']}|{p['ADDsym_AUC']:.6f}|{r['oracle']['ADDsym_AUC']:.6f}|{r['selection_loss']:.6f}|{p['axis_correct_count']}/{p['frames']}|")
        lines+=['','|arm|R med° /P90|yaw med° /P90|t med cm /P90|IoU3D median|','|---|---|---|---|---|']
        for a,r in res['groups'][g].items():
            p=r['current'];lines.append(f"|{a}|{p['rotation_deg']['median']:.3f} /{p['rotation_deg']['P90']:.3f}|{p['yaw_deg']['median']:.3f} /{p['yaw_deg']['P90']:.3f}|{p['translation_cm']['median']:.3f} /{p['translation_cm']['P90']:.3f}|{p['IoU3D']['median']:.4f}|")
        lines+=['',f"canonical GT ID 대응 후 loss/gain transitions: `{res['transitions'][g]}`",'',f'![{g}](figures/{fname})']
    lines+=['','## 8. Verified visible — fixed FINAL_V2','', '|group|N|BASE correct10|PRES1 correct10|BASE PCK10|PRES1 PCK10|BASE median/P90 px|PRES1 median/P90 px|PRES1 >20|','|---|---:|---:|---:|---:|---:|---|---|---:|']
    for g,r in aa['groups'].items():
        b,n=r['BASE'],r['PRES1'];lines.append(f"|{g}|{b['n']}|{b['PCK']['10']['correct']}|{n['PCK']['10']['correct']}|{b['PCK']['10']['fraction']:.4f}|{n['PCK']['10']['fraction']:.4f}|{b['median_px']:.3f}/{b['p90_px']:.3f}|{n['median_px']:.3f}/{n['p90_px']:.3f}|{n['gt20']}|")
    lines+=['','PCK5/20와 전체 정량은 [VERIFIED_VISIBLE.json](VERIFIED_VISIBLE.json). visible direct clicks만 fixed-ID로 평가하며 PnP 완성점이나 새 remapping을 정답으로 추가하지 않았다.','', '![visible](figures/06_verified_visible.png)','',
        '## 9. Source preservation','', '|arm|N frames|PCK5|PCK10|PCK20|median/P90 px|>20|exact ADD AUC|','|---|---:|---:|---:|---:|---|---:|---:|']
    for a,r in src['groups'].items():
        k=r['twoD'];lines.append(f"|{a}|256|{k['PCK']['5']['fraction']:.4f}|{k['PCK']['10']['fraction']:.4f}|{k['PCK']['20']['fraction']:.4f}|{k['median_px']:.3f}/{k['p90_px']:.3f}|{k['gt20']}|{r['pose']['ADDsym_AUC']:.6f}|")
    lines+=['',f"기존 고정 source256 그대로. 이전 단계의 exact renderer evaluator를 재사용했고, 투영-라벨 최대 오차 {src['exact_projection_max_px']:.6f}px를 확인했다. 새로운 synthetic TEST나 가림 recipe를 결과 보고 고르지 않았다.",'','![source](figures/07_source_preservation.png)','',
        '## 10. Compute','',f"Frozen fused base {cost['model_params']:,} params +adapter {cost['additional_params']:,}. 학습 {fit['seconds']:.2f}s. inference GPU peak allocated {cost['GPU_peak_allocated_MiB']:.1f}MiB. median latency S1 {cost['timings_ms']['BASE']['median']:.2f}ms → PRES1 {cost['timings_ms']['PRES1']['median']:.2f}ms. RTX3080 batch1, 먼저10 pair 제외, decode/checkpoint load 제외, RGB predictor 시간. PnP/scorer는 별도 총 {cost['pose_seconds']:.2f}s/256 model-frame. Jetson 측정이 아니다.",'','![compute](figures/08_latency_params.png)','',
        '## 11. 조건부 supervision gap','']
    if gap:
        lines += [f"**{gap['decision']}**. Frozen S1 기준 HARD36 중 >10px {gap['student_wrong10']}, TEACHER_CAN_TEACH {gap['teacher_can_teach']}, BOTH_FAIL_VISIBLE {gap['both_fail_visible']}, GROSS_BOTH_FAIL {gap['gross_both_fail']}. Missing trusted support cell의 both-fail {gap['missing_trusted_support']}점, {gap['distinct_frames']} frames/{gap['distinct_corners']} corner IDs. gate를 사후 변경하지 않았다.",'', '[점별 오류·trusted support 제외 사유·고정 gate](SUPERVISION_GAP_REPORT_KO.md)']
    else:lines+=['PRESERVATION_SUPPORTED에 따라 생략. **MIN_HARD_LABELING_NOT_NEEDED_NOW**. 신규 hard manual 없이 이번 DEV에서 보존 조건을 만족했다는 뜻이지 독립 최종 일반화를 검증했다는 뜻은 아니다.']
    lines+=['','## 12. 다음 행동','']
    if dec['primary']=='PRESERVATION_SUPPORTED':lines+=['PRES1을 FINAL_CANDIDATE로 동결하고, 다음은 새 독립 recording 평가 계획이다. 지금 새 라벨이나 추가 학습을 요청하지 않는다.']
    elif gap['decision']=='MIN_HARD_LABELING_JUSTIFIED':lines+=['고정 gate가 최소 hard-label 파일럿을 지지했지만, 후보 metadata 감사는 **HARD_CANDIDATE_METADATA_INSUFFICIENT**로 멈췄다. 평가/anchor/Clean10/H10/SHA/MAD 제외 후 기존 hard 태그 후보16장은 REC_001/002 두 recording뿐이다(Moderate15 /Severe1). 최소3 recordings·recording당 최대3장·initial8을 충족할 수 없다. 전체 adaptation RGB pool8031장도 기존 hard 태그가 없어 추가 후보0장이다. 모델 예측/오차/confidence로 대신 고르지 않았다.','', '따라서 선택된 큐0장, 새 클릭0개, GUI/label lock 미생성. 지금 사용자에게 어노테이션을 요청하지 않는다. 다음에는 적어도 한 개의 추가 비평가·비예약 recording에서 model-independent hard metadata가 필요하다. 기존2개 recording으로 조건을 완화하거나 새 라벨로 재학습하지 않았다. [조건부 후보 감사와 그림](../pallet_min_hard_labels_v1/REPORT_KO.md).']
    else:lines+=['추가 어노테이션을 요구하지 않는다. 보존 실패와 수동 라벨 필요성을 구별하여 남은 병목만 보고한다. 별도 지시 없이 추가 학습/하이퍼파라미터 탐색은 하지 않았다.']
    lines+=['','![decision](figures/09_decision.png)','',
        '아래는 각 난도에서 PRES1−BASE ADDnorm 변화가 가장 좋은/나쁜 사례다. 전체 빈도가 아니라 사후 설명용이며, raw2D(노랑)와 선택 PnP(빨강)를 구분한다. GT(초록)는 평가/시각화에만 사용한다.','',
        '![improved and failed examples](figures/10_improved_and_failed_cases.jpg)','',
        '## 13. 한계와 재현','',
        '- reused recording-disjoint HELDOUT128이며 독립 final TEST가 아니다. plastic-only, single seed42, 작은 visible anchor, geometry-derived real6D reference다.',
        '- camera-facing 역할 규약과 물리적180° C2 동치는 다르다. 기존 role warning 유지, C4 remap/GT 수정/점별 free matching 없음.',
        '- 추가 capacity와 loss/fit 신호의 충분성은 이번 한 어댑터/320step으로만 판단한다. 작은 변화나 실패를 구조 전체의 불가능성으로 일반화하지 않는다.',
        '- synthetic occlusion 보존과 실제 hard 보존이 동일하다고 가정하지 않는다. teacher-can-teach는 수동 정답 부족과 다른 병목이다.',
        f"- 자동 감사 {audit['n_tests']}개 PASS. 기존 hash-bound 입력 {audit['immutable_inputs']}개 불변. [감사](AUDIT.json), [프로토콜](PROTOCOL_LOCK.json), [학습노출](OCCURRENCES_LOCK.json), [실행코드](../../../scripts/research/pallet_single_model_preserve_v1/README.md).",
        '- 큰 tensor/checkpoint/좌표 캐시는 로컬 private namespace에 보존하고 MD·작은 ROI 이미지·표만 공개한다. 어노테이션 package는 조건부 gate를 통과할 때만 만든다.','']
    C.save(C.DOC/'REPORT_KO.md','\n'.join(lines))
if __name__=='__main__':main()
