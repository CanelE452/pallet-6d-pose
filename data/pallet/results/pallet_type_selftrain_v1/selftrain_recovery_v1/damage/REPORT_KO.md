# 일반 플라스틱 self-training: 큰 오차 감사

학습 완료 후 GT를 이용한 진단이다. 평가 이미지/코너를 학습 대상으로 바꾸지 않았고 새 필터나 모델 교체도 하지 않았다.

## 학습 신호와 학생 이동량

```json
{
  "pool_trusted_correction": {
    "n": 1863,
    "quantiles": {
      "p50": 3.011220795792939,
      "p90": 7.782789310132541,
      "p99": 15.208033905192714,
      "max": 24.510086173066952
    },
    "over20": 6
  },
  "sampled_unique_trusted_correction": {
    "n": 1627,
    "quantiles": {
      "p50": 3.010221916197084,
      "p90": 7.676710964188291,
      "p99": 15.238980849340408,
      "max": 24.510086173066952
    },
    "over20": 6
  },
  "student_native_displacement": {
    "n": 1552,
    "quantiles": {
      "p50": 1.1723248709318908,
      "p90": 3.5135613973918347,
      "p99": 5.3089494445177365,
      "max": 7.242536687097176
    },
    "over20": 0
  }
}
```

## 세 반복의 개선·악화 및 큰 오차

```json
{
  "REF_LR5": {
    "damage": {
      "improved_frames": 143,
      "harmed_frames": 43,
      "unchanged_frames": 8,
      "good5_to_bad10": 0,
      "bad20_to_good10": 0,
      "reverse_good5_to_bad10": 0,
      "evaluation_branch_changed": 2,
      "canonical_GT_identity_aligned": true
    },
    "tail_counts_including_miss_penalties": {
      "20": {
        "R0": 335,
        "student": 322
      },
      "40": {
        "R0": 209,
        "student": 214
      },
      "80": {
        "R0": 133,
        "student": 135
      },
      "100": {
        "R0": 112,
        "student": 114
      }
    },
    "matched_only_tail_counts": {
      "20": {
        "R0": 281,
        "student": 268
      },
      "40": {
        "R0": 155,
        "student": 160
      },
      "80": {
        "R0": 79,
        "student": 81
      },
      "100": {
        "R0": 58,
        "student": 60
      }
    },
    "unmatched_frames": 8
  },
  "REF_ORDER43": {
    "damage": {
      "improved_frames": 143,
      "harmed_frames": 43,
      "unchanged_frames": 8,
      "good5_to_bad10": 0,
      "bad20_to_good10": 0,
      "reverse_good5_to_bad10": 0,
      "evaluation_branch_changed": 2,
      "canonical_GT_identity_aligned": true
    },
    "tail_counts_including_miss_penalties": {
      "20": {
        "R0": 335,
        "student": 323
      },
      "40": {
        "R0": 209,
        "student": 214
      },
      "80": {
        "R0": 133,
        "student": 134
      },
      "100": {
        "R0": 112,
        "student": 114
      }
    },
    "matched_only_tail_counts": {
      "20": {
        "R0": 281,
        "student": 269
      },
      "40": {
        "R0": 155,
        "student": 160
      },
      "80": {
        "R0": 79,
        "student": 80
      },
      "100": {
        "R0": 58,
        "student": 60
      }
    },
    "unmatched_frames": 8
  },
  "REF_ORDER44": {
    "damage": {
      "improved_frames": 141,
      "harmed_frames": 45,
      "unchanged_frames": 8,
      "good5_to_bad10": 0,
      "bad20_to_good10": 0,
      "reverse_good5_to_bad10": 0,
      "evaluation_branch_changed": 2,
      "canonical_GT_identity_aligned": true
    },
    "tail_counts_including_miss_penalties": {
      "20": {
        "R0": 335,
        "student": 324
      },
      "40": {
        "R0": 209,
        "student": 213
      },
      "80": {
        "R0": 133,
        "student": 135
      },
      "100": {
        "R0": 112,
        "student": 114
      }
    },
    "matched_only_tail_counts": {
      "20": {
        "R0": 281,
        "student": 270
      },
      "40": {
        "R0": 155,
        "student": 159
      },
      "80": {
        "R0": 79,
        "student": 81
      },
      "100": {
        "R0": 58,
        "student": 60
      }
    },
    "unmatched_frames": 8
  }
}
```

8개의 매칭 실패 이미지 벌점과 실제 매칭된 코너의 큰 오차를 분리해 기록했다. 대칭 분기 변화는 canonical GT identity 기준으로 비교했다.

작은 보정 신호, 제한된 학습량, 필터를 통과한 쉬운 이미지의 편향, 동결한 모델 부분 등이 원인 후보다. 이 감사만으로 한 가지 원인으로 확정할 수 없다.

전체 194장 비교 HTML: outputs/pallet_type_selftrain_v1/selftrain_recovery_v1/damage/index.html
