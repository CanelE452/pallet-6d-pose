# Clean78 학습 타깃 전달: native 입력 진단

실제78장과epoch512 real slot에서복원했다. 새학습은없다. 이수치는의사타깃추종이지정답정확도나가림증강입력의실제TRAIN loss가아니다.

| 모델 | raw타깃 코너평균px | corrected타깃 코너평균px | corrected타깃 occurrence가중평균px | corrected타깃 이미지균등평균px |
|---|---:|---:|---:|---:|
| R0 | 0.0000 | 3.3586 | 3.3382 | 3.4119 |
| CLEAN_RAW_CLEAR | 0.6554 | 3.4813 | 3.4519 | 3.5226 |
| CLEAN_REF_CLEAR | 1.6810 | 2.6242 | 2.6107 | 2.6314 |
| CLEAN_RAW_OCC | 0.6590 | 3.4631 | 3.4319 | 3.5039 |
| CLEAN_REF_OCC | 1.6959 | 2.6490 | 2.6333 | 2.6541 |

최고confidence 검출을 그대로사용했다. teacher타깃에가장가까운후보로교체하지않았다. center는별도집계이며v1 ignored점은감독점으로세지않는다.
원래nativeRGB에reflection100 padding을정확히한번적용한다. 이미padded된export RGB에다시padding을더하지않는다.
가림/affine실제TRAIN입력에대한검증은아니므로이진단만으로최적화량·표현한계·전이실패중하나를확정하지않는다.

[recording·코너·중앙값/P90·가중집계](TRAIN_TARGET_FOLLOWING_S42.json)
