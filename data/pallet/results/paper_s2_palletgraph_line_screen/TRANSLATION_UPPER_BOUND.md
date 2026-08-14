# G1 — Translation Identifiability UPPER BOUND

> ## ⚠ 이 결과를 능력으로 읽으면 안 된다

> line map 을 **GT pose 로 그리고**, roll/pitch/ty 도 **GT 로 고정**한 뒤 남은 yaw/tx/tz 3 자유도만 찾았다.
> 즉 GT 에서 만든 신호에서 GT 를 되찾은 것이라, corner 5.1mm 같은 값은 물리적 정확도가 아니라
> **자기참조(oracle self-consistency)** 다.  이 단계가 보여주는 것은 오직 **정보량 상한**이다.


search prior 출처: `data/pallet/training_data/paper_4pallet_mask_v1` (N87 GT 미사용, 무결성 규칙 7)
  tz search 1.481..4.677 m (train p1..p99 = 1.747..4.411)


## Slice 결과
```
slice                   n  yaw_med(°)  t_med(m)  corner(m)  reproj(px)
────────────────────────────────────────────────────────────────────────
ALL                    87       0.188    0.0044     0.0051        2.32
point_fail             17       0.250    0.0036     0.0055        2.77
point_success          70       0.188    0.0044     0.0051        2.26
truncated              17       0.313    0.0043     0.0055        2.77
F1_NO_RESPONSE         24       0.250    0.0040     0.0056        3.06
F2_CONFIDENT_WRONG     35       0.188    0.0051     0.0061        2.37
```


## Gate
```
기준: point-fail 17 중 yaw<=10° 인 valid pose >= 8
결과: 17/17  (yaw<=5° 도 17/17, positive_depth 17/17)
판정: PASS
```


[확인] point-PnP 가 실패한 17 프레임 전부에서 line-only 로 valid pose 가 나왔다.
[확인] point-success 70 프레임에서도 corner error 가 70/70 개선(0.4516 -> 0.0051 m).
[주의] 위 경고대로, 이 수치는 oracle 자기참조라 **deployable 성능이 아니다**.  진짜 관문은 G2 다.

