# FIXED_5EP_CONFIG_DIFF

baseline `runs_broad40k/b_yolo26n_broad40k_5ep/args.yaml` 과의 **실제** 차이.
기억으로 만들지 않고 args.yaml 을 읽어 diff 했다.

```
항목          baseline (CF 5ep)                         fixed 5ep
──────────────────────────────────────────────────────────────────────────────
data          datasets/broad40k/data.yaml               datasets/broad40k_fixed/data.yaml
                                                        ★ 유일한 의미 있는 차이
                                                          = DATA LABEL CONTRACT
project       runs_broad40k                             runs_fixed        (경로만)
name          b_yolo26n_broad40k_5ep                    FIXED_OBJECT_BROAD40K_5EP_SEED42 (이름만)
```

동일하게 유지한 것 (baseline args 그대로):
```
model yolo26n-pose.pt · epochs 5 · batch 32 · imgsz 640 · optimizer SGD
lr0 0.01 · seed 42 · patience 0 · workers 4 · pretrained True · fliplr 0.0
device 0 · cache False · save_period -1
```

RGB membership: **완전히 동일**.
`train_rgb_membership_sha256 = 412e73a1e789347c` 가 10 bundle manifest 와 일치.
이미지는 broad40k 의 실제 파일을 symlink 로 가리킨다(재인코딩 없음).

참고: baseline 5ep 와 paper 60ep 의 args 차이도 실측했다 —
`epochs / name / project / save_dir / save_period` 뿐이고 나머지는 전부 같다.
