# PAPER_S2 DiffPnP3D — 결론 (partial transfer)

> 2026-07-09 확정. canonical 결론. 상세: PLAN.md, quick_screen_results.md,
> ../../eval_results/paper_s2_scratch_diffpnp/{real_eval_summary.md, regression_3to8_diagnosis.md}

## 한 줄
DiffPnP3D(dims-aware 3D-corner regularizer)는 **버릴 실패는 아니고 깨끗한 성공도 아닌 PARTIAL 전이**.
rear robustness·검출 강건성·극저앙각(<3°)은 real로 전이됐으나 **전체 pose 정확도(honest8)는 개선 못 함**.

## 최종 real 결과 (Stage B vs paper_s1, filterval N=123 저앙각)
```
개선:  rear -1.5px / det +6%p / PnP +4%p / gross -1.1%p
flat~악화:  corner +1.0px / honest8 +0.2px / good +1.1%p
앙각별:  <3°(n46) 광범위 승 / 3-8°(n67) MIXED(rear median↓지만 corner/honest8 회귀) / 8-15°(n10) 노이즈
```

## Claim / No-Claim (그대로 사용)
**EN (paper)**:
> DiffPnP3D improved rear localization and robustness in extreme low-angle cases (<3°),
> but did not improve overall pose accuracy. In the 3-8° transition band, the regression
> was already present in raw heatmap predictions, indicating that the degradation is not
> caused by the PnP re-projection step.

**KR**:
> DiffPnP3D는 극저앙각에서 rear 안정성과 검출 강건성 일부를 개선했지만, 전체 pose
> 정확도(honest8)는 개선하지 못했다. 3-8° 회귀는 PnP 후처리 문제가 아니라 raw heatmap
> 단계에서 이미 발생했다.

**✅ 클레임 가능**: DiffPnP3D가 synthetic Q1 + real 극저앙각(<3°)에서 rear/검출 robustness 신호.
**❌ 클레임 금지**: "rear/depth collapse를 해결" / "전 구간 pose accuracy 개선".

## 3-8° 회귀 진단 (핵심)
- **회귀는 RAW heatmap에 있다** (PnP 이전 이미 발생, raw +1.0 paired / PnP는 +0.7만 추가). → **DiffPnP/PnP 후처리 문제 아님** [확인].
- decode-PnP mismatch도 아님 (raw-PnP displacement 오히려 낮음) [확인].
- 원인은 **paper_s2 recipe 묶음(squash·scratch·데이터·DiffPnP)에서 옴 — squash 유력 가설이나 scratch/데이터와 미분리**. ★"Stage A=λ0" 은 오류(Stage A도 λ0.005) → "squash 단독범인/DiffPnP 중립" 주장 불가.

## 결정 (2026-07-09 사용자)
1. PAPER_S2 = partial transfer 로 기록(위 문구). ✅
2. "Stage A λ0" 오류 정정. ✅ (regression_3to8_diagnosis.md)
3. 3-8° raw-vs-PnP failure montage 저장. (진행)
4. **다음 메인 = 데이터 트랙**: paper-safe **각도-bin(<3/3-8/8-15/15+)** 저앙각·near·transition 데이터 생성 또는 real 촬영 계획. 3-8° transition 회귀 구간 우선.
5. λ0 vs λ0.005 full ablation = **보류**(논문 reviewer가 요구할 때만; 회귀는 입력측이라 이걸론 안 풀림).

## 교훈
표현/후처리(loss·PnP) 트랙은 이번에도 rear 절대정확도 천장을 못 뚫음 → **rear/저앙각의 진짜 레버는 데이터/appearance** (STAGE15/16, corner01, stage22 반복 결론과 정합). DiffPnP는 부분 효과는 있으나 데이터 레버를 대체하지 못한다.
