# weights/_archive

2026-08-15. `weights/` 가 97개 폴더 142G 로 불어나 현역 체크포인트가 묻혔다.
여기 있는 16개(11G)는 **저장소 어디에서도 참조되지 않는** 것들이다 — 코드·설정·문서를
전수로 훑어 `weights/<이름>` 문자열이 한 번도 나오지 않은 것만 골랐다.

## 구성

계열은 이름이 아니라 각 run 의 `header.txt` 에 적힌 **학습 입력**으로 묶었다.

```
selftrain_r1_cf/     2.9G  5   R1 self-training, cf 필터 PL 풀
                             r1_{indoor,night}_cf_strict · r1_outside_cf_{loo,loo_fast,v2}
                             data = pl_{indoor,night,outside}_R0_cf_*

filter_ransac_loo/   4.9G  4   ransac_loo PL 풀로 학습한 필터 계열
                             f3_realonly_ransac_loo · f5_{ep100,ep125,reproduce}
                             data = {f3_pl,pl_noapril}_ransac_loo_only

loss_screen/         1.2G  2   loss 스크리닝 A/C arm
                             data = mixed_v8_train

── 계열이 없어 단독으로 둔 것 ──
ep65_pl_realonly     385M      data = pl_noapril_clean
pallet_category_test 385M      data = train
paper_s2_rgb1_tz_v3  838M      header 없음
legacy_filter_100    207M      header 없음
challenge_ft_mp40    3.4M      header 없음
```

## 지운 게 아니라 옮긴 것

`weights/` 는 `.gitignore` 대상이라 git 이 보호하지 않는다. 체크포인트는 재학습에
GPU 시간이 드는 물건이고 복구 경로가 없으므로, **삭제하지 않고 옮기기만 했다.**
필요하면 해당 폴더를 `weights/` 로 다시 올리면 된다.

## 남긴 기준

```
코드·설정이 참조   60개 121.7G   → 루트 유지
문서에만 언급       7개   8.3G   → 루트 유지 (실험 재현에 필요할 수 있다)
어디에도 없음      16개  11.0G   → 여기
```

memory 가 이름을 못박은 것들은 전부 "코드 참조" 쪽에 있다:
`paper_s2_stageB`(55회) · `challenge0123`(16) · `stage11_16k_B2_maskaux`(6) 등.

## 다시 정리할 때

판정은 이름이 아니라 참조로 한다.

```bash
grep -rl "weights/<이름>" --include="*.py" --include="*.sh" --include="*.yaml" \
     --include="*.json" --include="*.md" . | grep -v "/.git/"
```

계열을 나눌 때는 `<run>/header.txt` 의 `data=` 를 본다. 이름이 비슷해도 학습 입력이
다르면 다른 실험이고, 그 반대도 있다.


---

## weights/ 전체 구조 (2026-08-15)

루트가 81개로 흩어져 있어 10개 계열로 묶었다.

```
paper_s2/         20  45G   PAPER_S2 스크린
stage_screens/    11  26G   stage11~24
challenge_track/   9  18G   과제용 트랙
selftrain/        11  13G   R1~R3 self-training 라운드
misc/              6  12G   계열이 없는 것
_archive/          9  11G   참조 0 (아래 참조)
dope/              3 7.4G   dope_cropaug 계열
paper_base/        4 6.3G
paper_s1/          3 4.9G
_logs/            13         학습 로그
paper_s2_stageB/  10 1.5G   ★루트에 남는다 — 아래 이유
```

### ★paper_s2_stageB 만 루트에 남는 이유

`scripts/stage0/paper_s2_frozen_diagnostic.py` 가 이 경로를 상수로 들고 있고,
그 파일의 sha256 이 `cache_key` 에 들어간다. 경로를 한 글자만 고쳐도 파일 해시가
바뀌어 **저장된 진단 캐시가 전부 무효**가 된다. 실제로 옮겼다가
`test_cache_reuse_performs_zero_model_forwards` 가 잡아냈고 되돌렸다.

옮길 수 없는 파일은 자기가 가리키는 것까지 붙잡아 둔다 — 코드에서 이미 겪은 것과
같은 규칙이 가중치에도 적용된다.
