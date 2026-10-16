# A 실행 결과

[확인] source oracle, 동결 N3 세 seed의 GEO/PERM×I/J, GEO/PERM 세 seed의 6,000회 정식 학습과 마지막 checkpoint 평가를 완료했어. 큰 특징 cache와 가중치는 저장소 밖에 있고 기존 N3/PoseFix 결과를 동일 입력 해시로 재사용했어. 실제 DEV는 독립 확인 자료가 아니야.

[확인] 실제 개발 자료의 GEO--J minus N3/PoseFix 전체 방법 비교 판정은 각각 악화/악화야. GEO--J minus RAW/PERM의 평균 정규화 오차 차이 구간이 음수여도 고정 손상/보존 규칙을 함께 적용해 각각 이익 미확립/이익 미확립라는 판정으로 기록했어. synthetic 개발 진단의 GEO--J minus PERM 판정은 악화야.

[확인] 아래 중앙값/90백분위는 매칭되고 관측된 코너를 모은 조건부 통계야. 실제 DEV는 검출 319/319, 8유효 예측 코너 319/319, GT instance 매칭 311/319를 구분해. 전체 penalty 포함 평균 정규화 거리와 PCK는 참조 코너 전체 분모를 사용하고, F 자세 coverage는 전체 frame을 분모로 사용해. 실제 참조 자세는 같은 2D 레이블·치수에서 재구성했어.

[확인] 실제 DEV의 유효 참조 코너는 2,499개이고 매칭된 관측 거리의 분모는 2,445개야. synthetic 개발 진단의 유효 참조/관측 코너는 각각15,658개야. 전체 참조와 조건부 관측 분모를 서로 바꾸지 않아.

[확인] PCK(Percentage of Correct Keypoints, 허용 거리 안의 코너 비율), ADD(Average Distance of Model Points, 모델 점 평균 거리)의 대칭 적용 진단 ADDsym, PnP(Perspective-n-Point, 2D/3D 대응 자세 계산)를 사용해. ADDsym은 승인된 proper rotation group에서 정준 8코너 대응 평균 거리 m이며 표면 ADD-S가 아니야. I는 코너별 hard argmax, J는 한 action의 hard argmax야.

## 동일 분모의 세 seed 원결과

### 실제 개발 319장/13세션

| 방법/seed | 코너 med px | P90 px | PCK10 | gross20 | T med cm | R med deg | F coverage | NoOp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | 6.7207 | 43.8900 | 0.63425 | 0.19928 | 7.8969 | 2.5389 | 319/319 | N/A |
| N3_seed1 | 5.7446 | 42.3204 | 0.68828 | 0.17567 | 7.0392 | 2.1104 | 319/319 | N/A |
| N3_seed2 | 5.7677 | 41.8721 | 0.68267 | 0.17767 | 6.7902 | 2.0727 | 319/319 | N/A |
| N3_seed3 | 5.8222 | 42.2087 | 0.68667 | 0.17367 | 7.3735 | 2.0281 | 319/319 | N/A |
| PoseFix_seed1 | 5.5626 | 43.5346 | 0.68547 | 0.17967 | 7.1854 | 1.9540 | 319/319 | N/A |
| PoseFix_seed2 | 5.5284 | 44.0303 | 0.69228 | 0.18087 | 6.8438 | 1.9871 | 319/319 | N/A |
| PoseFix_seed3 | 5.5911 | 44.1571 | 0.68347 | 0.18167 | 6.8250 | 2.1422 | 319/319 | N/A |
| FIT_GEO_J_seed1 | 6.4291 | 44.7775 | 0.64826 | 0.18647 | 7.2888 | 2.4026 | 319/319 | 82 |
| FIT_GEO_J_seed2 | 6.4614 | 43.0952 | 0.64906 | 0.18888 | 7.6630 | 2.4674 | 319/319 | 100 |
| FIT_GEO_J_seed3 | 6.4908 | 43.9856 | 0.64226 | 0.19088 | 7.8845 | 2.4918 | 319/319 | 129 |
| FIT_PERM_J_seed1 | 6.6265 | 44.1047 | 0.63665 | 0.19288 | 8.2266 | 2.5502 | 319/319 | 140 |
| FIT_PERM_J_seed2 | 6.7655 | 43.5901 | 0.63826 | 0.19168 | 8.3237 | 2.5579 | 319/319 | 121 |
| FIT_PERM_J_seed3 | 6.5640 | 43.6749 | 0.64346 | 0.19248 | 7.8969 | 2.3581 | 319/319 | 109 |
| FROZEN_GEO_I_seed1 | 6.5809 | 42.7268 | 0.63786 | 0.19368 | 8.1923 | 2.4993 | 319/319 | 147 |
| FROZEN_GEO_J_seed1 | 6.7045 | 43.8900 | 0.63505 | 0.19808 | 7.8969 | 2.5389 | 319/319 | 294 |
| FROZEN_PERM_I_seed1 | 6.5809 | 42.7268 | 0.63786 | 0.19368 | 8.1923 | 2.4993 | 319/319 | 147 |
| FROZEN_PERM_J_seed1 | 6.7050 | 43.8900 | 0.63545 | 0.19808 | 7.8936 | 2.4717 | 319/319 | 301 |
| FROZEN_GEO_I_seed2 | 6.6776 | 43.2799 | 0.63265 | 0.19488 | 7.4698 | 2.3924 | 319/319 | 141 |
| FROZEN_GEO_J_seed2 | 6.7207 | 43.8900 | 0.63465 | 0.19928 | 7.8936 | 2.5389 | 319/319 | 308 |
| FROZEN_PERM_I_seed2 | 6.6776 | 43.2799 | 0.63265 | 0.19488 | 7.4698 | 2.3924 | 319/319 | 141 |
| FROZEN_PERM_J_seed2 | 6.7181 | 43.8900 | 0.63425 | 0.19928 | 7.8969 | 2.5389 | 319/319 | 317 |
| FROZEN_GEO_I_seed3 | 6.6244 | 44.6366 | 0.63625 | 0.19328 | 8.2917 | 2.4717 | 319/319 | 167 |
| FROZEN_GEO_J_seed3 | 6.6944 | 43.8900 | 0.63585 | 0.19928 | 7.8969 | 2.5192 | 319/319 | 307 |
| FROZEN_PERM_I_seed3 | 6.6244 | 44.6366 | 0.63625 | 0.19328 | 8.2917 | 2.4717 | 319/319 | 167 |
| FROZEN_PERM_J_seed3 | 6.7181 | 43.8900 | 0.63425 | 0.19888 | 7.8969 | 2.5356 | 319/319 | 313 |

### 기존에 소진된 synthetic 개발 진단 1,985장

| 방법/seed | 코너 med px | P90 px | PCK10 | gross20 | T med cm | R med deg | F coverage | NoOp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RAW | 1.9784 | 7.0566 | 0.93639 | 0.02727 | 2.5036 | 1.2631 | 1985/1985 | N/A |
| N3_seed1 | 1.7638 | 6.7106 | 0.93997 | 0.02644 | 2.4172 | 1.1913 | 1985/1985 | N/A |
| N3_seed2 | 1.7962 | 6.8658 | 0.94048 | 0.02663 | 2.4917 | 1.1794 | 1985/1985 | N/A |
| N3_seed3 | 1.7541 | 6.7893 | 0.94131 | 0.02631 | 2.3508 | 1.1944 | 1985/1985 | N/A |
| PoseFix_seed1 | 1.3893 | 6.3952 | 0.94303 | 0.02587 | 1.8903 | 0.9779 | 1985/1985 | N/A |
| PoseFix_seed2 | 1.3797 | 6.4974 | 0.94425 | 0.02567 | 1.9256 | 1.0052 | 1985/1985 | N/A |
| PoseFix_seed3 | 1.4087 | 6.4985 | 0.94303 | 0.02606 | 1.9686 | 1.0030 | 1985/1985 | N/A |
| FIT_GEO_J_seed1 | 2.1889 | 7.8229 | 0.93115 | 0.02753 | 2.6695 | 1.3843 | 1985/1985 | 1036 |
| FIT_GEO_J_seed2 | 2.0791 | 7.6301 | 0.93294 | 0.02785 | 2.5830 | 1.3431 | 1985/1985 | 1435 |
| FIT_GEO_J_seed3 | 2.0934 | 7.7058 | 0.93294 | 0.02759 | 2.5969 | 1.3163 | 1985/1985 | 1387 |
| FIT_PERM_J_seed1 | 2.0401 | 7.3949 | 0.93409 | 0.02733 | 2.5540 | 1.2832 | 1985/1985 | 1681 |
| FIT_PERM_J_seed2 | 2.0618 | 7.4138 | 0.93479 | 0.02746 | 2.5841 | 1.3060 | 1985/1985 | 1526 |
| FIT_PERM_J_seed3 | 2.1085 | 7.5697 | 0.93358 | 0.02733 | 2.5916 | 1.3046 | 1985/1985 | 1435 |
| FROZEN_GEO_I_seed1 | 2.0047 | 7.1971 | 0.93665 | 0.02772 | 2.4398 | 1.2706 | 1985/1985 | 919 |
| FROZEN_GEO_J_seed1 | 1.9938 | 7.1009 | 0.93607 | 0.02740 | 2.5036 | 1.2771 | 1985/1985 | 1900 |
| FROZEN_PERM_I_seed1 | 2.0047 | 7.1971 | 0.93665 | 0.02772 | 2.4398 | 1.2706 | 1985/1985 | 919 |
| FROZEN_PERM_J_seed1 | 1.9849 | 7.0904 | 0.93639 | 0.02733 | 2.5040 | 1.2685 | 1985/1985 | 1958 |
| FROZEN_GEO_I_seed2 | 2.0097 | 7.1879 | 0.93601 | 0.02753 | 2.5460 | 1.2816 | 1985/1985 | 950 |
| FROZEN_GEO_J_seed2 | 1.9883 | 7.0915 | 0.93626 | 0.02727 | 2.5010 | 1.2685 | 1985/1985 | 1909 |
| FROZEN_PERM_I_seed2 | 2.0097 | 7.1879 | 0.93601 | 0.02753 | 2.5460 | 1.2816 | 1985/1985 | 950 |
| FROZEN_PERM_J_seed2 | 1.9824 | 7.0694 | 0.93633 | 0.02727 | 2.5040 | 1.2684 | 1985/1985 | 1961 |
| FROZEN_GEO_I_seed3 | 1.9945 | 7.1918 | 0.93703 | 0.02670 | 2.5261 | 1.2883 | 1985/1985 | 970 |
| FROZEN_GEO_J_seed3 | 1.9862 | 7.0781 | 0.93658 | 0.02714 | 2.5036 | 1.2712 | 1985/1985 | 1916 |
| FROZEN_PERM_I_seed3 | 1.9945 | 7.1918 | 0.93703 | 0.02670 | 2.5261 | 1.2883 | 1985/1985 | 970 |
| FROZEN_PERM_J_seed3 | 1.9816 | 7.0644 | 0.93645 | 0.02727 | 2.5040 | 1.2638 | 1985/1985 | 1964 |

## practical 기준선 및 결합/decoder 대조

### 실제 개발: seed 평균 paired E_sym 차이

| 새 방법 minus 기준 | 평균 차이 | 95% CI | seed1/2/3 차이 | 판정 | 손상/coverage 보존 |
| --- | --- | --- | --- | --- | --- |
| FIT_GEO_J_minus_RAW | -0.00031911 | [-0.00065231,-0.00005401] | -0.00035925/-0.00031139/-0.00028670 | UNRESOLVED | False |
| FIT_GEO_J_minus_N3 | 0.00078570 | [0.00045428,0.00125593] | 0.00079150/0.00075803/0.00080756 | WORSENED | False |
| FIT_GEO_J_minus_PoseFix | 0.00090093 | [0.00050038,0.00141278] | 0.00085103/0.00098791/0.00086385 | WORSENED | False |
| FIT_GEO_J_minus_FIT_PERM_J | -0.00019133 | [-0.00039871,-0.00002461] | -0.00020883/-0.00023799/-0.00012717 | UNRESOLVED | False |
| FIT_PERM_J_minus_RAW | -0.00012779 | [-0.00026173,-0.00001490] | -0.00015042/-0.00007340/-0.00015954 | UNRESOLVED | False |
| FIT_PERM_J_minus_N3 | 0.00097703 | [0.00066160,0.00144119] | 0.00100034/0.00099602/0.00093472 | WORSENED | False |
| FIT_PERM_J_minus_PoseFix | 0.00109226 | [0.00071768,0.00159640] | 0.00105986/0.00122589/0.00099102 | WORSENED | False |
| FROZEN_GEO_I_minus_RAW | -0.00007255 | [-0.00018424,0.00000458] | -0.00010767/-0.00004326/-0.00006671 | UNRESOLVED | False |
| FROZEN_GEO_I_minus_N3 | 0.00103227 | [0.00067239,0.00156545] | 0.00104309/0.00102616/0.00102755 | WORSENED | False |
| FROZEN_GEO_I_minus_PoseFix | 0.00114750 | [0.00077161,0.00164855] | 0.00110261/0.00125603/0.00108384 | WORSENED | False |
| FROZEN_GEO_J_minus_RAW | -0.00000820 | [-0.00002680,0.00000643] | -0.00000900/-0.00001393/-0.00000167 | UNRESOLVED | True |
| FROZEN_GEO_J_minus_N3 | 0.00109661 | [0.00070057,0.00164239] | 0.00114176/0.00105549/0.00109259 | WORSENED | False |
| FROZEN_GEO_J_minus_PoseFix | 0.00121184 | [0.00080799,0.00171356] | 0.00120129/0.00128536/0.00114888 | WORSENED | False |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | 0.00006435 | [-0.00001740,0.00018203] | 0.00009867/0.00002933/0.00006504 | UNRESOLVED | False |
| FROZEN_PERM_I_minus_RAW | -0.00007255 | [-0.00018424,0.00000458] | -0.00010767/-0.00004326/-0.00006671 | UNRESOLVED | False |
| FROZEN_PERM_I_minus_N3 | 0.00103227 | [0.00067239,0.00156545] | 0.00104309/0.00102616/0.00102755 | WORSENED | False |
| FROZEN_PERM_I_minus_PoseFix | 0.00114750 | [0.00077161,0.00164855] | 0.00110261/0.00125603/0.00108384 | WORSENED | False |
| FROZEN_PERM_J_minus_RAW | -0.00000392 | [-0.00001013,0.00000068] | -0.00000341/-0.00000599/-0.00000236 | UNRESOLVED | True |
| FROZEN_PERM_J_minus_N3 | 0.00110089 | [0.00070966,0.00164351] | 0.00114735/0.00106343/0.00109190 | WORSENED | False |
| FROZEN_PERM_J_minus_PoseFix | 0.00121612 | [0.00081663,0.00171374] | 0.00120688/0.00129330/0.00114820 | WORSENED | False |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | 0.00006863 | [-0.00000907,0.00018156] | 0.00010426/0.00003727/0.00006435 | UNRESOLVED | False |

| 대조 | seed | good<5 → bad>10 코너 | bad>20 → good<10 코너 | 개선/손상/동일 frame | 양성공/새것만/기준만/양실패 |
| --- | --- | --- | --- | --- | --- |
| FIT_GEO_J_minus_RAW | 1 | 6 | 1 | 143/91/85 | 319/0/0/0 |
| FIT_GEO_J_minus_RAW | 2 | 6 | 1 | 132/82/105 | 319/0/0/0 |
| FIT_GEO_J_minus_RAW | 3 | 5 | 1 | 113/73/133 | 319/0/0/0 |
| FIT_GEO_J_minus_N3 | 1 | 17 | 0 | 77/234/8 | 319/0/0/0 |
| FIT_GEO_J_minus_N3 | 2 | 15 | 0 | 75/236/8 | 319/0/0/0 |
| FIT_GEO_J_minus_N3 | 3 | 19 | 0 | 77/234/8 | 319/0/0/0 |
| FIT_GEO_J_minus_PoseFix | 1 | 28 | 2 | 95/216/8 | 319/0/0/0 |
| FIT_GEO_J_minus_PoseFix | 2 | 26 | 3 | 78/233/8 | 319/0/0/0 |
| FIT_GEO_J_minus_PoseFix | 3 | 26 | 2 | 86/225/8 | 319/0/0/0 |
| FIT_GEO_J_minus_FIT_PERM_J | 1 | 5 | 1 | 141/98/80 | 319/0/0/0 |
| FIT_GEO_J_minus_FIT_PERM_J | 2 | 15 | 1 | 145/85/89 | 319/0/0/0 |
| FIT_GEO_J_minus_FIT_PERM_J | 3 | 22 | 1 | 123/96/100 | 319/0/0/0 |
| FIT_PERM_J_minus_RAW | 1 | 3 | 2 | 109/67/143 | 319/0/0/0 |
| FIT_PERM_J_minus_RAW | 2 | 6 | 1 | 104/90/125 | 319/0/0/0 |
| FIT_PERM_J_minus_RAW | 3 | 7 | 1 | 122/84/113 | 319/0/0/0 |
| FIT_PERM_J_minus_N3 | 1 | 19 | 0 | 58/253/8 | 319/0/0/0 |
| FIT_PERM_J_minus_N3 | 2 | 15 | 0 | 56/255/8 | 319/0/0/0 |
| FIT_PERM_J_minus_N3 | 3 | 16 | 0 | 65/246/8 | 319/0/0/0 |
| FIT_PERM_J_minus_PoseFix | 1 | 31 | 1 | 65/246/8 | 319/0/0/0 |
| FIT_PERM_J_minus_PoseFix | 2 | 28 | 1 | 48/263/8 | 319/0/0/0 |
| FIT_PERM_J_minus_PoseFix | 3 | 20 | 1 | 78/233/8 | 319/0/0/0 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | 1 | 3 | 0 | 64/94/161 | 319/0/0/0 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | 2 | 0 | 0 | 80/82/157 | 319/0/0/0 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | 3 | 1 | 0 | 61/79/179 | 319/0/0/0 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | 1 | 3 | 0 | 63/95/161 | 319/0/0/0 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | 2 | 0 | 0 | 77/85/157 | 319/0/0/0 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | 3 | 1 | 0 | 59/81/179 | 319/0/0/0 |

| 대조 | 자세 지표 | 3seed 공동성공 frame | 평균 paired 차이 | 95% CI | seed1/2/3 차이 |
| --- | --- | --- | --- | --- | --- |
| FIT_GEO_J_minus_RAW | translation_cm | 319 | -0.32394565 | [-0.85860552,0.02723033] | -0.21968174/-0.72279419/-0.02936102 |
| FIT_GEO_J_minus_RAW | rotation_deg | 319 | -0.06005721 | [-0.23641777,0.08044957] | -0.04796406/-0.07708382/-0.05512375 |
| FIT_GEO_J_minus_RAW | ADDsym_m | 319 | -0.00425504 | [-0.00922241,-0.00147987] | -0.00340309/-0.00795606/-0.00140598 |
| FIT_GEO_J_minus_N3 | translation_cm | 319 | -0.13077053 | [-3.23232519,1.44114461] | -0.24909291/-0.62291598/0.47969731 |
| FIT_GEO_J_minus_N3 | rotation_deg | 319 | 2.90862070 | [0.81998777,4.77961122] | 2.60403845/3.29899735/2.82282628 |
| FIT_GEO_J_minus_N3 | ADDsym_m | 319 | 0.02271392 | [-0.01455669,0.05187466] | 0.01731824/0.02314275/0.02768077 |
| FIT_GEO_J_minus_PoseFix | translation_cm | 319 | -0.46863196 | [-3.22431336,1.05764616] | -0.14158234/-1.16545799/-0.09885554 |
| FIT_GEO_J_minus_PoseFix | rotation_deg | 319 | 3.29275212 | [0.99433607,5.44099199] | 3.17933638/3.13624241/3.56267757 |
| FIT_GEO_J_minus_PoseFix | ADDsym_m | 319 | 0.02490137 | [-0.01693441,0.05843033] | 0.02770462/0.01566284/0.03133667 |
| FIT_GEO_J_minus_FIT_PERM_J | translation_cm | 319 | -0.59795046 | [-1.37872118,-0.14810737] | -0.38920201/-1.06883411/-0.33581528 |
| FIT_GEO_J_minus_FIT_PERM_J | rotation_deg | 319 | -0.10725308 | [-0.96296086,0.41770516] | 0.20704409/-0.27658191/-0.25222143 |
| FIT_GEO_J_minus_FIT_PERM_J | ADDsym_m | 319 | -0.00604722 | [-0.02040714,0.00258599] | -0.00146623/-0.01122507/-0.00545037 |
| FIT_PERM_J_minus_RAW | translation_cm | 319 | 0.27400482 | [-0.06726466,0.79111456] | 0.16952026/0.34603992/0.30645426 |
| FIT_PERM_J_minus_RAW | rotation_deg | 319 | 0.04719587 | [-0.57999994,0.93338462] | -0.25500816/0.19949810/0.19709768 |
| FIT_PERM_J_minus_RAW | ADDsym_m | 319 | 0.00179218 | [-0.00552781,0.01427960] | -0.00193686/0.00326901/0.00404439 |
| FIT_PERM_J_minus_N3 | translation_cm | 319 | 0.46717994 | [-1.89637664,1.72773198] | 0.14010910/0.44591812/0.81551259 |
| FIT_PERM_J_minus_N3 | rotation_deg | 319 | 3.01587378 | [1.31729603,4.51719009] | 2.39699436/3.57557927/3.07504771 |
| FIT_PERM_J_minus_N3 | ADDsym_m | 319 | 0.02876115 | [0.00158492,0.05166778] | 0.01878447/0.03436782/0.03313114 |
| FIT_PERM_J_minus_PoseFix | translation_cm | 319 | 0.12931851 | [-1.95566589,1.35956733] | 0.24761967/-0.09662388/0.23695974 |
| FIT_PERM_J_minus_PoseFix | rotation_deg | 319 | 3.40000520 | [1.58337154,5.16783973] | 2.97229229/3.41282432/3.81489900 |
| FIT_PERM_J_minus_PoseFix | ADDsym_m | 319 | 0.03094860 | [0.00010909,0.05813505] | 0.02917085/0.02688790/0.03678704 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | translation_cm | 319 | 0.07378234 | [-1.57798354,1.27774242] | -0.17274888/0.59486468/-0.20076879 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | rotation_deg | 319 | 1.47623450 | [-0.25262982,3.03687452] | 2.70102454/1.27889870/0.44878027 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | ADDsym_m | 319 | 0.01096088 | [-0.01024112,0.02905203] | 0.01565081/0.01477515/0.00245668 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | translation_cm | 319 | 0.26217224 | [-0.87567470,1.18053125] | -0.70918686/1.07040484/0.42529875 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | rotation_deg | 319 | 1.38251759 | [-0.24860065,3.05629759] | 2.32001921/1.29917034/0.52836322 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | ADDsym_m | 319 | 0.01286348 | [-0.00612157,0.03101871] | 0.01038525/0.01954618/0.00865901 |

### synthetic 개발 진단: seed 평균 paired E_sym 차이

| 새 방법 minus 기준 | 평균 차이 | 95% CI | seed1/2/3 차이 | 판정 | 손상/coverage 보존 |
| --- | --- | --- | --- | --- | --- |
| FIT_GEO_J_minus_RAW | 0.00025434 | [0.00022038,0.00028804] | 0.00032602/0.00021397/0.00022303 | WORSENED | False |
| FIT_GEO_J_minus_N3 | 0.00048924 | [0.00044939,0.00052833] | 0.00057664/0.00042398/0.00046711 | WORSENED | False |
| FIT_GEO_J_minus_PoseFix | 0.00079897 | [0.00073658,0.00085502] | 0.00088184/0.00077147/0.00074360 | WORSENED | False |
| FIT_GEO_J_minus_FIT_PERM_J | 0.00010170 | [0.00007133,0.00013190] | 0.00021396/0.00005955/0.00003158 | WORSENED | False |
| FIT_PERM_J_minus_RAW | 0.00015264 | [0.00013270,0.00017232] | 0.00011206/0.00015442/0.00019145 | WORSENED | False |
| FIT_PERM_J_minus_N3 | 0.00038754 | [0.00035538,0.00041896] | 0.00036268/0.00036443/0.00043552 | WORSENED | False |
| FIT_PERM_J_minus_PoseFix | 0.00069727 | [0.00063740,0.00074959] | 0.00066788/0.00071192/0.00071202 | WORSENED | False |
| FROZEN_GEO_I_minus_RAW | 0.00004673 | [0.00003218,0.00006139] | 0.00004879/0.00005459/0.00003680 | WORSENED | False |
| FROZEN_GEO_I_minus_N3 | 0.00028163 | [0.00025528,0.00030922] | 0.00029941/0.00026460/0.00028088 | WORSENED | False |
| FROZEN_GEO_I_minus_PoseFix | 0.00059135 | [0.00053013,0.00064297] | 0.00060460/0.00061209/0.00055737 | WORSENED | False |
| FROZEN_GEO_J_minus_RAW | 0.00001045 | [0.00000428,0.00001684] | 0.00001586/0.00000961/0.00000586 | WORSENED | False |
| FROZEN_GEO_J_minus_N3 | 0.00024535 | [0.00021557,0.00027619] | 0.00026649/0.00021962/0.00024993 | WORSENED | False |
| FROZEN_GEO_J_minus_PoseFix | 0.00055507 | [0.00049513,0.00060819] | 0.00057168/0.00056712/0.00052643 | WORSENED | False |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | -0.00003628 | [-0.00005103,-0.00002128] | -0.00003292/-0.00004497/-0.00003095 | UNRESOLVED | False |
| FROZEN_PERM_I_minus_RAW | 0.00004673 | [0.00003218,0.00006139] | 0.00004879/0.00005459/0.00003680 | WORSENED | False |
| FROZEN_PERM_I_minus_N3 | 0.00028163 | [0.00025528,0.00030922] | 0.00029941/0.00026460/0.00028088 | WORSENED | False |
| FROZEN_PERM_I_minus_PoseFix | 0.00059135 | [0.00053013,0.00064297] | 0.00060460/0.00061209/0.00055737 | WORSENED | False |
| FROZEN_PERM_J_minus_RAW | 0.00000553 | [0.00000185,0.00000958] | 0.00000715/0.00000527/0.00000418 | WORSENED | False |
| FROZEN_PERM_J_minus_N3 | 0.00024043 | [0.00021074,0.00027131] | 0.00025777/0.00021528/0.00024825 | WORSENED | False |
| FROZEN_PERM_J_minus_PoseFix | 0.00055016 | [0.00049001,0.00060363] | 0.00056296/0.00056278/0.00052475 | WORSENED | False |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | -0.00004119 | [-0.00005603,-0.00002665] | -0.00004164/-0.00004931/-0.00003262 | UNRESOLVED | False |

| 대조 | seed | good<5 → bad>10 코너 | bad>20 → good<10 코너 | 개선/손상/동일 frame | 양성공/새것만/기준만/양실패 |
| --- | --- | --- | --- | --- | --- |
| FIT_GEO_J_minus_RAW | 1 | 63 | 0 | 272/671/1035 | 1985/0/0/0 |
| FIT_GEO_J_minus_RAW | 2 | 43 | 0 | 143/401/1434 | 1985/0/0/0 |
| FIT_GEO_J_minus_RAW | 3 | 46 | 0 | 170/422/1386 | 1985/0/0/0 |
| FIT_GEO_J_minus_N3 | 1 | 79 | 1 | 557/1421/0 | 1985/0/0/0 |
| FIT_GEO_J_minus_N3 | 2 | 53 | 0 | 654/1324/0 | 1985/0/0/0 |
| FIT_GEO_J_minus_N3 | 3 | 62 | 0 | 626/1352/0 | 1985/0/0/0 |
| FIT_GEO_J_minus_PoseFix | 1 | 103 | 0 | 296/1682/0 | 1985/0/0/0 |
| FIT_GEO_J_minus_PoseFix | 2 | 82 | 1 | 350/1628/0 | 1985/0/0/0 |
| FIT_GEO_J_minus_PoseFix | 3 | 77 | 0 | 356/1622/0 | 1985/0/0/0 |
| FIT_GEO_J_minus_FIT_PERM_J | 1 | 59 | 0 | 348/613/1017 | 1985/0/0/0 |
| FIT_GEO_J_minus_FIT_PERM_J | 2 | 36 | 1 | 313/352/1313 | 1985/0/0/0 |
| FIT_GEO_J_minus_FIT_PERM_J | 3 | 37 | 1 | 400/379/1199 | 1985/0/0/0 |
| FIT_PERM_J_minus_RAW | 1 | 18 | 0 | 66/234/1678 | 1985/0/0/0 |
| FIT_PERM_J_minus_RAW | 2 | 13 | 0 | 112/341/1525 | 1985/0/0/0 |
| FIT_PERM_J_minus_RAW | 3 | 25 | 0 | 129/415/1434 | 1985/0/0/0 |
| FIT_PERM_J_minus_N3 | 1 | 35 | 0 | 627/1351/0 | 1985/0/0/0 |
| FIT_PERM_J_minus_N3 | 2 | 24 | 0 | 647/1331/0 | 1985/0/0/0 |
| FIT_PERM_J_minus_N3 | 3 | 46 | 0 | 575/1403/0 | 1985/0/0/0 |
| FIT_PERM_J_minus_PoseFix | 1 | 55 | 0 | 343/1635/0 | 1985/0/0/0 |
| FIT_PERM_J_minus_PoseFix | 2 | 39 | 0 | 332/1646/0 | 1985/0/0/0 |
| FIT_PERM_J_minus_PoseFix | 3 | 59 | 0 | 332/1646/0 | 1985/0/0/0 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | 1 | 8 | 0 | 607/440/931 | 1985/0/0/0 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | 2 | 12 | 0 | 595/419/964 | 1985/0/0/0 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | 3 | 9 | 0 | 568/425/985 | 1985/0/0/0 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | 1 | 8 | 0 | 624/423/931 | 1985/0/0/0 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | 2 | 12 | 0 | 597/417/964 | 1985/0/0/0 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | 3 | 9 | 0 | 574/419/985 | 1985/0/0/0 |

| 대조 | 자세 지표 | 3seed 공동성공 frame | 평균 paired 차이 | 95% CI | seed1/2/3 차이 |
| --- | --- | --- | --- | --- | --- |
| FIT_GEO_J_minus_RAW | translation_cm | 1985 | 0.18000014 | [0.04245969,0.33935895] | 0.18994392/0.22557702/0.12447948 |
| FIT_GEO_J_minus_RAW | rotation_deg | 1985 | 0.13034414 | [-0.06748920,0.33830786] | 0.12694630/0.19295461/0.07113149 |
| FIT_GEO_J_minus_RAW | ADDsym_m | 1985 | 0.00242083 | [0.00026605,0.00483782] | 0.00256237/0.00300366/0.00169645 |
| FIT_GEO_J_minus_N3 | translation_cm | 1985 | 0.14907636 | [-0.22548566,0.47754893] | 0.18951450/0.13612951/0.12158508 |
| FIT_GEO_J_minus_N3 | rotation_deg | 1985 | -0.01566233 | [-0.56028966,0.53456746] | -0.06099803/0.22133501/-0.20732397 |
| FIT_GEO_J_minus_N3 | ADDsym_m | 1985 | 0.00061018 | [-0.00613983,0.00720881] | 0.00049965/0.00280024/-0.00146935 |
| FIT_GEO_J_minus_PoseFix | translation_cm | 1985 | -2.62470362 | [-3.95662118,-1.43314112] | -2.38232384/-2.81768553/-2.67410148 |
| FIT_GEO_J_minus_PoseFix | rotation_deg | 1985 | -0.82354389 | [-1.65692102,-0.04133493] | -1.01031428/-0.84650359/-0.61381380 |
| FIT_GEO_J_minus_PoseFix | ADDsym_m | 1985 | -0.02351672 | [-0.03695338,-0.01102180] | -0.02208963/-0.02745934/-0.02100119 |
| FIT_GEO_J_minus_FIT_PERM_J | translation_cm | 1985 | 0.07187679 | [-0.08473125,0.23620996] | 0.00104253/0.13182380/0.08276402 |
| FIT_GEO_J_minus_FIT_PERM_J | rotation_deg | 1985 | 0.05980468 | [-0.22910031,0.34981166] | 0.08263988/0.13358660/-0.03681244 |
| FIT_GEO_J_minus_FIT_PERM_J | ADDsym_m | 1985 | 0.00067994 | [-0.00285753,0.00419366] | 0.00050771/0.00130026/0.00023186 |
| FIT_PERM_J_minus_RAW | translation_cm | 1985 | 0.10812335 | [0.02500094,0.20459365] | 0.18890139/0.09375322/0.04171546 |
| FIT_PERM_J_minus_RAW | rotation_deg | 1985 | 0.07053946 | [-0.17791491,0.32371733] | 0.04430642/0.05936801/0.10794394 |
| FIT_PERM_J_minus_RAW | ADDsym_m | 1985 | 0.00174088 | [-0.00133790,0.00487713] | 0.00205466/0.00170341/0.00146458 |
| FIT_PERM_J_minus_N3 | translation_cm | 1985 | 0.07719958 | [-0.26620793,0.37481609] | 0.18847197/0.00430570/0.03882106 |
| FIT_PERM_J_minus_N3 | rotation_deg | 1985 | -0.07546701 | [-0.59762705,0.46116712] | -0.14363791/0.08774840/-0.17051152 |
| FIT_PERM_J_minus_N3 | ADDsym_m | 1985 | -0.00006977 | [-0.00632136,0.00631989] | -0.00000807/0.00149998/-0.00170122 |
| FIT_PERM_J_minus_PoseFix | translation_cm | 1985 | -2.69658041 | [-4.00869174,-1.50676275] | -2.38336638/-2.94950934/-2.75686551 |
| FIT_PERM_J_minus_PoseFix | rotation_deg | 1985 | -0.88334856 | [-1.72783551,-0.08596426] | -1.09295415/-0.98009019/-0.57700135 |
| FIT_PERM_J_minus_PoseFix | ADDsym_m | 1985 | -0.02419666 | [-0.03760209,-0.01171559] | -0.02259734/-0.02875960/-0.02123305 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | translation_cm | 1985 | -0.31469280 | [-0.73556563,0.07212631] | -0.24525997/-0.29606889/-0.40274954 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | rotation_deg | 1985 | -0.28333922 | [-0.66663347,0.10432399] | -0.28933690/-0.38916318/-0.17151760 |
| FROZEN_GEO_J_minus_FROZEN_GEO_I | ADDsym_m | 1985 | -0.00576670 | [-0.01092610,-0.00085730] | -0.00550650/-0.00676827/-0.00502533 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | translation_cm | 1985 | -0.20842290 | [-0.57539358,0.11350385] | -0.05631661/-0.31647758/-0.25247450 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | rotation_deg | 1985 | -0.22796887 | [-0.62665325,0.18369970] | -0.19421752/-0.31902696/-0.17066214 |
| FROZEN_PERM_J_minus_FROZEN_PERM_I | ADDsym_m | 1985 | -0.00442587 | [-0.00912975,0.00025055] | -0.00324367/-0.00613920/-0.00389473 |

## 고정 bank의 실제 F oracle와 선택 gap

[확인] source 1,031장은 학습 진입 판단용 개발 oracle이며 이전 heldout 1,985장/실제 319장 진단과 분리해. NoOp 반복 F의 ADDsym 차이는 0m였고 첫 metric 전 고정한 numerical tie 허용오차 1e-7m보다 큰 여지만 비영으로 세었어. 진입은 GEO 여지만으로 판단했어. oracle 선택에는 참조를 쓰지만 후보 생성·특징 점수·배포 J에는 쓰지 않아.

| 자료 | bank | 전체 | RAW 성공 | oracle 성공 | 비영 여지 | 여지 med m | 같은 후보 T/R 동시 비악화 | 후보 F 호출 | 후보 F 실패 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SOURCE_SELECTION | GEO | 1031 | 1030 | 1030 | 954 | 0.01191545 | 641 | 207031 | 2 |
| SOURCE_SELECTION | PERM | 1031 | 1030 | 1030 | 1003 | 0.01328241 | 684 | 207031 | 1 |
| SYNTH_HELDOUT | GEO | 1985 | 1985 | 1985 | 1840 | 0.01247555 | 1241 | 398985 | 0 |
| SYNTH_HELDOUT | PERM | 1985 | 1985 | 1985 | 1929 | 0.01439369 | 1353 | 398985 | 0 |
| REAL_DEV | GEO | 319 | 319 | 319 | 319 | 0.03124382 | 211 | 64119 | 0 |
| REAL_DEV | PERM | 319 | 319 | 319 | 319 | 0.02739230 | 221 | 64119 | 0 |

| 자료 | bank | 바로 그 선택 RAW/oracle med px | RAW/oracle P90 px | RAW/oracle PCK10 | good5 → bad10 | bad20 → good10 |
| --- | --- | --- | --- | --- | --- | --- |
| SOURCE | GEO | 1.9148 / 2.5348 | 6.9441 / 9.0892 | 0.93945 / 0.91886 | 159 | 0 |
| SOURCE | PERM | 1.9148 / 3.1670 | 6.9441 / 8.0339 | 0.93945 / 0.93452 | 56 | 0 |
| SYNTH_HELDOUT | GEO | 1.9784 / 2.4989 | 7.0566 / 9.0588 | 0.93639 / 0.91698 | 325 | 0 |
| SYNTH_HELDOUT | PERM | 1.9784 / 3.1850 | 7.0566 / 8.2100 | 0.93639 / 0.93262 | 97 | 0 |
| REAL_DEV | GEO | 6.7207 / 6.9304 | 43.8900 / 44.5959 | 0.63425 / 0.64946 | 23 | 1 |
| REAL_DEV | PERM | 6.7207 / 6.6357 | 43.8900 / 42.3936 | 0.63425 / 0.64426 | 15 | 0 |

### REAL_DEV: 선택 minus 같은 bank oracle ADDsym

| 방법/seed | 공동성공 | gap med m | gap P90 m | 선택 F 실패 | oracle F 실패 | 음수 > 허용오차 |
| --- | --- | --- | --- | --- | --- | --- |
| FIT_GEO_J_seed1 | 319 | 0.02906926 | 0.14258612 | 0 | 0 | 0 |
| FIT_GEO_J_seed2 | 319 | 0.02706045 | 0.16420994 | 0 | 0 | 0 |
| FIT_GEO_J_seed3 | 319 | 0.03004041 | 0.16420994 | 0 | 0 | 0 |
| FIT_PERM_J_seed1 | 319 | 0.02739230 | 0.90401651 | 0 | 0 | 0 |
| FIT_PERM_J_seed2 | 319 | 0.02928776 | 0.97490652 | 0 | 0 | 0 |
| FIT_PERM_J_seed3 | 319 | 0.02666890 | 0.97311609 | 0 | 0 | 0 |
| FROZEN_GEO_I_seed1 | 319 | 0.03005251 | 0.12939408 | 0 | 0 | 6 |
| FROZEN_GEO_J_seed1 | 319 | 0.03171948 | 0.13999093 | 0 | 0 | 0 |
| FROZEN_PERM_I_seed1 | 319 | 0.02680132 | 0.58083247 | 0 | 0 | 4 |
| FROZEN_PERM_J_seed1 | 319 | 0.02666890 | 0.85538492 | 0 | 0 | 0 |
| FROZEN_GEO_I_seed2 | 319 | 0.03018083 | 0.14089751 | 0 | 0 | 2 |
| FROZEN_GEO_J_seed2 | 319 | 0.03117862 | 0.14064120 | 0 | 0 | 0 |
| FROZEN_PERM_I_seed2 | 319 | 0.02560692 | 0.72233979 | 0 | 0 | 2 |
| FROZEN_PERM_J_seed2 | 319 | 0.02739230 | 0.94906228 | 0 | 0 | 0 |
| FROZEN_GEO_I_seed3 | 319 | 0.03090571 | 0.16029421 | 0 | 0 | 4 |
| FROZEN_GEO_J_seed3 | 319 | 0.03117862 | 0.14064120 | 0 | 0 | 0 |
| FROZEN_PERM_I_seed3 | 319 | 0.02647274 | 0.90315566 | 0 | 0 | 3 |
| FROZEN_PERM_J_seed3 | 319 | 0.02739230 | 0.94906228 | 0 | 0 | 0 |

### SYNTH_HELDOUT: 선택 minus 같은 bank oracle ADDsym

| 방법/seed | 공동성공 | gap med m | gap P90 m | 선택 F 실패 | oracle F 실패 | 음수 > 허용오차 |
| --- | --- | --- | --- | --- | --- | --- |
| FIT_GEO_J_seed1 | 1985 | 0.01442058 | 0.08467143 | 0 | 0 | 0 |
| FIT_GEO_J_seed2 | 1985 | 0.01351534 | 0.08482702 | 0 | 0 | 0 |
| FIT_GEO_J_seed3 | 1985 | 0.01351534 | 0.08499014 | 0 | 0 | 0 |
| FIT_PERM_J_seed1 | 1985 | 0.01477014 | 0.21199745 | 0 | 0 | 0 |
| FIT_PERM_J_seed2 | 1985 | 0.01510812 | 0.21199745 | 0 | 0 | 0 |
| FIT_PERM_J_seed3 | 1985 | 0.01452492 | 0.21199745 | 0 | 0 | 0 |
| FROZEN_GEO_I_seed1 | 1985 | 0.01212463 | 0.09085980 | 0 | 0 | 66 |
| FROZEN_GEO_J_seed1 | 1985 | 0.01254280 | 0.08487384 | 0 | 0 | 0 |
| FROZEN_PERM_I_seed1 | 1985 | 0.01399849 | 0.23365637 | 0 | 0 | 31 |
| FROZEN_PERM_J_seed1 | 1985 | 0.01434929 | 0.21091847 | 0 | 0 | 0 |
| FROZEN_GEO_I_seed2 | 1985 | 0.01228986 | 0.08942230 | 0 | 0 | 60 |
| FROZEN_GEO_J_seed2 | 1985 | 0.01254169 | 0.08417831 | 0 | 0 | 0 |
| FROZEN_PERM_I_seed2 | 1985 | 0.01399849 | 0.25200046 | 0 | 0 | 19 |
| FROZEN_PERM_J_seed2 | 1985 | 0.01439369 | 0.21060932 | 0 | 0 | 0 |
| FROZEN_GEO_I_seed3 | 1985 | 0.01235302 | 0.09315504 | 0 | 0 | 69 |
| FROZEN_GEO_J_seed3 | 1985 | 0.01254280 | 0.08487384 | 0 | 0 | 0 |
| FROZEN_PERM_I_seed3 | 1985 | 0.01398592 | 0.22026927 | 0 | 0 | 27 |
| FROZEN_PERM_J_seed3 | 1985 | 0.01436281 | 0.21678376 | 0 | 0 | 0 |

[확인] 선택된 단일 oracle의 바로 그 2D 결과/손상과 이동·회전은 `A_SOURCE_ORACLE_CORNERS.json`, `A_SYNTH_HELDOUT_ORACLE_CORNERS.json`, `A_REAL_DEV_ORACLE_CORNERS.json`에 남아. index/bank 재사용 보완의 추가 F 호출은 0회야. I는 bank 밖의 코너 조합이라 oracle보다 좋을 수 있지만 J의 음수 gap은 허용오차 밖에서 0건이어야 해.

## 실행량과 검증

```json
{
  "formal_updates": 36000,
  "formal_exposures": 576000,
  "formal_fit_seconds": 2383.508212399902,
  "formal_failed_or_discarded_updates": 0,
  "smoke_updates": 4,
  "smoke_seconds": 0.448090102057904,
  "smoke_weights_discarded_before_formal": true,
  "offline_oracle_F_calls": 1340270,
  "offline_oracle_F_failures": 3,
  "offline_oracle_seconds": 234.70949828694575,
  "oracle_corner_enrichment_new_F_calls": 0,
  "bank_precompute_seconds": "NOT_RECORDED: precompute retries retained valid banks before any optimizer update;fit/oracle/evaluation times have individual receipts",
  "method_evaluation_F_calls_from_stored_rows": 57600,
  "frozen_and_fitted_evaluation_seconds": 232.99585729581304,
  "baseline_evaluation_seconds": "NOT_RECORDED: baseline outputs reused or evaluated once,actual F calls counted by rows",
  "source_full_partition_counts": {
    "train": 55980,
    "calibration": 1004,
    "selection": 1031,
    "heldout": 1985
  },
  "source_cache_partition_counts": {
    "train": 55980,
    "calibration": 1004,
    "selection": 1031,
    "heldout": 1985
  },
  "source_cache_omitted_rows": 0,
  "train_excluded_by_prior_matched_target_rules": 65
}
```

[확인] 6 fit×6,000 update=36,000 update, batch16의 576,000 노출이야. smoke는 각 arm 2회씩 총4회이며 가중치를 폐기하고 같은 seed의 원래 무작위 초기화로 정식 fit을 시작했어. seed별 두 arm의 초기 state SHA/순서 SHA/20,259 parameter가 같음을 `A_manifest.json`에서 검증해. 실제 TRAIN 제외 target 노출 수와 최종 checkpoint SHA는 각 `A_fits/*.json`에 있어. GPU는 RTX3080, FP32, matmul TF32 off/cudnn TF32 on의 원래 수치 계약이야.

[확인] 첫 optimizer 이전의 NPZ 반복 materialization/인위적 NaN backward/프로세스 import 정합성 수정과 재시작에는 정식·smoke update가 각각 0회였어. 유효 oracle/bank를 재사용했어. NoOp·cap·중심 회전·양의 깊이·원본/네트워크 affine·PERM 주변 multiset·기존 radial API·결측 backward 검증은 `A_NUMERICAL_PARITY.json`, `A_INDEPENDENT_PRELIMINARY_QA.json`, 독립 검토 영수증에 연결돼. trained dynamic radial adapter의 최대 logits 차이 4.482269287109375e-5는 FP32 coordinate 재구성 범위이며 기존 radial API exact parity와 구분해.

[확인] 마지막 fresh 분석은 `A_ANALYSIS_REAGGREGATION.json`의 wall126.04초로 측정됐어. 완료fit6개/예측파일18개/oracle3개를 재사용해 추가 optimizer·CNN 점수 추론·최종 F 채점은 모두0회야. 기존 평가기 구조 때문에 실제319개 bank를 3번(957개 bank) 다시 구성했고 내부의 초기 prediction-only selector/PnP는 수행됐어. 이 초기 PnP의 개별 호출수는 미계측이므로 0회라고 쓰지 않아. 분석 보완을 새 학습이나 최종 F 성능 실험으로 세지 않아.

[확인] `results/A_SAMPLING_MASK_RECEIPT.json`은 synthetic1,985+real319의 2,304개 frame에서 원시좌표/affine/box/input bounds/기존32점 stencil만으로 sampling mask·coverage·공통support를 CPU FP32로 재구성했어. 원래 실행한 GPU tensor는 보존하지 않았으므로 원tensor 저장본이라고 쓰지 않아. GEO/PERM mask 주변 multiset과9개checkpoint stencil은같아. source저장bank는초기PnP0회로재사용했고realbank1회재생성은실측solvePnP1,276회/RefineLM1,276회, 전체보완wall20.449초야. 추가CNN/최종F/update는0회이며bit-packed mask와realbank는외부cache에해시와함께저장했어. 보조mask/subgroup receipt는parentmanifest가따로바인딩해.

[확인] 실제는 session, synthetic 주 분석은 frame으로 10,000회/seed20260917의 같은 bootstrap 추출을 공유했어. synthetic scenario secondary는 미실행이야. 여러 대조의 multiplicity 보정 없는 탐색적 구간이며 독립 확증이 아니야.

## 연구 판단과 남은 의존성

[추정] 비영 oracle 여지는 이 고정 bank 안에 더 나은 최종 F 출력이 존재한다는 진단이야. practical N3/PoseFix와의 방법 전체 비교, GEO/PERM 유한 결합 대조, 같은 bank I/J decoder 대조의 CI·손상·실패를 각각 읽어야 해. 좋은 seed·중앙값·인위적 대조 승리만으로 센서 기반 6D 개선을 선언하지 않아. 동결 scorer의 실패는 특징 정보 부재의 증명이 아니며 이번 6 fit 결과는 고정 후보/특징/참조/예산 범위만 제한해.

[확인] 물리 가림의 독립 상태와 clean/실물 가림 동일 상대 자세 pair·독립 센서 참조는 BLOCKED_DATA야. human grade/가시성 상태/기존 거리 tag의 보조 재집계는 `A_DESCRIPTIVE_SUBGROUPS_KO.md`와 연결하지만 물리 가림 정도나 독립 거리 계측으로 승격하지 않아. 새로운 모델·cap·seed·epoch 탐색은 추가하지 않았어.

[확인] 배포 비용은 root의 동일 26frame/20warmup/5repeat 전체 경계 A runtime 영수증에서 별도로 통합해. 위 전체 후보 oracle F 탐색은 오프라인 비용이며 배포 비용과 합치지 않아.

## 결과와 재개

`A_protocol.json`, `results/A_ID_MANIFEST.json`, `results/A_summary.json`, `results/A_*_COMPARISONS.json`, `results/A_*_METRIC_ROWS.tsv`, `A_manifest.json`, `A_RESULT_SCHEMA_KO.md`를 함께 읽어. 원본 private 입력이 없으면 모델/특징 재실행 의존성이 막히며 작은 집계 파일만으로 새 성능을 생성하지 않아.

주 재개 명령은 `python -m scripts.research.pallet_joint_action_handoff_20261006_v1.run --stage resume --source-root ORIGINAL_READ_ONLY_CHECKOUT --cache-dir EXTERNAL_CACHE`야. 실제 경로는 상위 README에 있어. 이 진입점은 추가663개 입력의 전량 SHA 및 source17배열 receipt를 먼저 검증한 뒤 완료된 6 fit와 원결과를 재사용해.
