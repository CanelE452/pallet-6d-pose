# Combined evaluation target progress

```text
Status                    READY
Positive                   305 / 300
Negative                  2688 / 1500
UNKNOWN_METADATA           305
Counting population       ALL_AVAILABLE
Counting policy           One combined collection; SHA256-deduplicated
New annotation required   NO
```

목표 진행률은 역할별 counter를 만들지 않고 `ALL_AVAILABLE` 통합 view 하나만 사용한다.
같은 image는 SHA256으로 한 번만 센다. 이 목표 미달은 새 annotation을 의무화하지
않는다.
