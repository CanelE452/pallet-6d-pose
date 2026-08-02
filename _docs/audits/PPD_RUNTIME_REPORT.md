# PPD runtime

warmup 20 / measure 100, input 1x3x400x400, device cuda.

```
base (frozen ep57) only        21.05 ms
base + polarity line head      21.39 ms
head overhead                  0.34 ms  (1.6%)
head parameters                181,573
```

[확인] head 는 forward 비용이 사실상 무시할 수준(1.6%)이다.
[확인] 이 수치는 **forward 만** 포함한다.  SAI candidate 생성과 line-energy scoring 은
CPU 측 기하 연산이며 여기 포함되지 않았다.
[확인] 비용이 싸다는 사실은 채택 근거가 되지 못한다 — real gate 를 통과하지 못했다.
