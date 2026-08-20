# FINAL DATA ROLE DECISION

```
MAIN TRAIN
  Corner = BROAD_40K  (MH_TRAIN 33,758)
  Line   = BROAD_40K  (MH_TRAIN 33,758)
  manifest FINAL_POSITIVE_V1_manifest.json   sha256 4554d9e8ee9bf63a...

NEGATIVE
  status        미도착 (NEGATIVE_SYNTH_V1)
  arrival 후    NEGATIVE_TRAIN_CANDIDATE — 별도 stream, positive 노출 대체 금지
                contract: NEGATIVE_READY_TRAIN_CONTRACT.md

ABLATION ONLY
  - CORNER_LA_Y30_PLUS  2,500   CORNER_TARGET_ABLATION   benefit NOT_ESTABLISHED
  - CORNER_LA_Y15_30    2,500   EASY_CONTROL_ABLATION    겨냥 cell 이 near-baseline
  - EDGE_HARD_TRAIN    10,000   LINE_HARD_ABLATION       corner 감독 불가(설계상)
  - CORNER_LA_FRONTAL       0   PRESERVE_UNUSED          미도착, 자동 투입 금지

EVAL ONLY
  - MH_DEV 6,242 (D2/D3/D4/D5/lcurve 는 전부 여기서 뽑음, train 누수 0)
  - EDGE_HARD_DEV 1,000            line-hard 평가 전용
  - EDGE_HARD_UNTOUCHED             ★도착 완료 (2026-08-19/20, sha256 검증 OK)
      trunc_untouched 1,000  visible_kp 1~3   line-hard 평가용
      clean_untouched 1,000  visible_kp >=4   ★corner PnP 평가 가능 (도착)
  - EDGE clean train/dev 11,000     ★의도적 미배포. README 가 CLEAN 을
                                    '대조군(감사·figure 용)' 으로 규정. 요청 대상 아님

EXCLUDED / ARCHIVE
  - mixed_v8_train, mixed_v8_train_2k        v8 계열, 폐기 실패작
  - aug_scale_v2 / aug_squash_v2 / aug_trunc_v2
  - paper_s2_pl_* / paper_s2_plrf_* / paper_s2_full7_* / paper_s2_fullpool_*
  - pallet6d_v2, v4_split_base, paper_4pallet_mask_v1*
  이유: 구세대 convention·구 self-training PL. 현 실험 체계와 불일치
```

## 왜 concat 하지 않는가

각 셋을 main 에 넣지 않은 이유는 "아직 안 해봐서" 가 아니라 **실험 결과가 있어서**다.

```
CORNER_LA_Y30_PLUS   C1(12.5% 혼합)·C1_RESCUE(노출 보존 + dose 1,500) 두 설계 모두
                     두 seed 방향 충돌, bootstrap CI 가 0 포함
CORNER_LA_Y15_30     canonical failure map 에서 겨냥 cell 이 9.62px x1.14 = 거의 정상
EDGE_HARD            라벨 자체가 G1_Vvis>=4 = False. corner 감독을 붙일 수 없다
CORNER_LA_FRONTAL    데이터가 없다
```

## 확인된 무결성

```
train × eval 누수     평가 고유 프레임 2,885 (D2/D3/D4/D5/lcurve) × MH_TRAIN → 0
BROAD × CORNER_LA     rgb 서명 40,000 × 5,000 → collision 0
BROAD × EDGE_HARD     생성 run 분리 (v2_prod40k vs v2_edgecomp_s9601~9608) → 0
CORNER_LA 내부        고유 5,000 / 5,000
old→canonical 버킷    Y15_30 → 100% 15-30,  Y30_PLUS → 100% >=30,  누출 0
```

## NEXT

negative 도착 시 `NEGATIVE_READY_TRAIN_CONTRACT.md` 의 N0/N1 로 바로 착수 가능.
그 전에 새 학습·새 렌더 없음.
