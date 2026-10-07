# 팔레트 비학습 보정 고정 비교

[확인] 비학습 보정이 N3를 대체할 만큼 좋아졌는가: 이번 두 알고리즘·네 출력에서 N3 seed1보다 T/R 중앙값이 함께 낮은 방법은 없다. 전체 방법군이나 동등성 결론으로 확대하지 않는다.

[확인] SUBPIX NATIVE는 N3 seed1 대비 T 중앙값 7.039→5.794cm, ADD 중앙값 0.080549→0.065734m이다. R 중앙값은 2.110→2.541degree, 2D 중앙값은 5.745→7.020px다. 위치·ADD 이득과 회전·코너 손상이 공존해 전체 대체나 동등성을 입증하지 않는다.

[확인] 같은 초기 YOLO319장·13세션. 추가 보정 학습0이며 전체 시스템에는 기존 YOLO/N3/PoseFix의 학습이 있다. 2D 참조2499점·매칭311장·관측2445점, 자세분모319장. NATIVE/CAP1 네 출력을 모두 보존했다.

[확인] CVRANK의 반경12는 기존 지시문 기본값이고 x/y 사각 창이다. 후보 설정은 과거 SOURCE_DEV 합성 val에서 cov5 최대·후보수 최소 grid로 선택했고 네 rank 항은 동일 비중으로 수동 고정했다. 현재 새 설정 선택·가중치 학습은0이다. SUBPIX win5/count40/epsilon0.001과CAP1=원영상대각1%를 사전 고정했다.

| 방법 | 2D median/P90 px | full PCK10/gross20 | T median/P90 cm | R median/P90 degree | ADD median/P90 m | F성공/319 | full ADD AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 6.7207/43.8900 | 0.634254/0.199280 | 7.8969/40.5302 | 2.5389/86.5273 | 0.087137/1.198890 | 319/319 | 0.376603 |
| N3_seed1 | 5.7446/42.3204 | 0.688275/0.175670 | 7.0392/38.1358 | 2.1104/85.9305 | 0.080549/1.192205 | 319/319 | 0.409627 |
| PoseFix_seed1 | 5.5626/43.5346 | 0.685474/0.179672 | 7.1854/43.8098 | 1.9540/85.9950 | 0.078512/1.185009 | 319/319 | 0.418621 |
| N0_seed1 | 5.8845/43.8365 | 0.675470/0.183273 | 7.0830/36.9030 | 2.2299/85.7719 | 0.077688/1.191935 | 319/319 | 0.410981 |
| SUBPIX_NATIVE | 7.0196/43.6671 | 0.622249/0.201281 | 5.7942/39.3900 | 2.5409/86.9971 | 0.065734/1.200072 | 319/319 | 0.432951 |
| SUBPIX_CAP1 | 7.0196/43.6671 | 0.622249/0.201281 | 5.7942/39.3900 | 2.5409/86.9971 | 0.065734/1.200072 | 319/319 | 0.432951 |
| CVRANK_NATIVE | 8.2462/44.1378 | 0.562225/0.215286 | 6.3765/42.2261 | 3.0000/86.2282 | 0.070677/1.199627 | 319/319 | 0.411502 |
| CVRANK_CAP1 | 8.0623/43.8471 | 0.578231/0.208483 | 6.1875/37.9807 | 2.8805/86.3527 | 0.067014/1.199718 | 319/319 | 0.422734 |

[확인] PCK/gross/AUC는0..1분율. 2D quantile은관측조건부,PCK/gross는전체참조+결측페널티. ADD는정준8점proper-group대응평균이며표면ADD-S가아니다. 기존정규화ADD AUC는직경대비0..0.1/1001임계값trapz이고실패inf를포함한다.

| TF−대조 | T paired mean [session95CI] cm | R paired mean [session95CI] degree | ADD paired mean [session95CI] m | T/R difference-of-medians | F실패Δ |
|---|---:|---:|---:|---:|---:|
| SUBPIX_NATIVE_minus_BASE | -0.865675 [-2.717050, 1.226406] | -0.828839 [-2.435686, 1.549486] | -0.011592 [-0.041504, 0.021315] | -2.1027/0.0020 | 0 |
| SUBPIX_NATIVE_minus_N3_seed1 | -0.895086 [-2.754176, 0.559914] | 1.823163 [0.754120, 3.250245] | 0.009129 [-0.006505, 0.027091] | -1.2450/0.4305 | 0 |
| SUBPIX_NATIVE_minus_PoseFix_seed1 | -0.787575 [-2.266144, 0.306440] | 2.398461 [0.444375, 4.828473] | 0.019516 [-0.003680, 0.047873] | -1.3912/0.5868 | 0 |
| SUBPIX_NATIVE_minus_N0_seed1 | -0.715927 [-2.472648, 0.700484] | 2.761933 [0.970139, 4.686229] | 0.021798 [0.003993, 0.044062] | -1.2888/0.3110 | 0 |
| SUBPIX_CAP1_minus_BASE | -0.865675 [-2.717050, 1.226406] | -0.828839 [-2.435686, 1.549486] | -0.011592 [-0.041504, 0.021315] | -2.1027/0.0020 | 0 |
| SUBPIX_CAP1_minus_N3_seed1 | -0.895086 [-2.754176, 0.559914] | 1.823163 [0.754120, 3.250245] | 0.009129 [-0.006505, 0.027091] | -1.2450/0.4305 | 0 |
| SUBPIX_CAP1_minus_PoseFix_seed1 | -0.787575 [-2.266144, 0.306440] | 2.398461 [0.444375, 4.828473] | 0.019516 [-0.003680, 0.047873] | -1.3912/0.5868 | 0 |
| SUBPIX_CAP1_minus_N0_seed1 | -0.715927 [-2.472648, 0.700484] | 2.761933 [0.970139, 4.686229] | 0.021798 [0.003993, 0.044062] | -1.2888/0.3110 | 0 |
| CVRANK_NATIVE_minus_BASE | -1.209180 [-3.695257, 1.633200] | -0.107660 [-4.411017, 5.489998] | -0.012055 [-0.070450, 0.055980] | -1.5204/0.4611 | 0 |
| CVRANK_NATIVE_minus_N3_seed1 | -1.238591 [-5.441649, 1.923462] | 2.544342 [-0.666021, 7.011809] | 0.008667 [-0.032298, 0.061078] | -0.6627/0.8896 | 0 |
| CVRANK_NATIVE_minus_PoseFix_seed1 | -1.131080 [-4.040742, 1.440598] | 3.119640 [-0.072867, 7.819410] | 0.019053 [-0.023600, 0.073812] | -0.8089/1.0459 | 0 |
| CVRANK_NATIVE_minus_N0_seed1 | -1.059432 [-5.097560, 2.117127] | 3.483112 [0.655034, 7.478660] | 0.021335 [-0.010868, 0.065390] | -0.7065/0.7701 | 0 |
| CVRANK_CAP1_minus_BASE | -1.684887 [-4.088209, 1.052038] | -0.157380 [-3.913309, 4.988767] | -0.014161 [-0.068359, 0.049983] | -1.7094/0.3416 | 0 |
| CVRANK_CAP1_minus_N3_seed1 | -1.714298 [-5.656488, 1.284042] | 2.494623 [-0.148545, 6.538386] | 0.006561 [-0.032522, 0.057123] | -0.8517/0.7701 | 0 |
| CVRANK_CAP1_minus_PoseFix_seed1 | -1.606788 [-4.350994, 0.873018] | 3.069920 [0.138787, 7.453046] | 0.016947 [-0.025584, 0.071844] | -0.9980/0.9265 | 0 |
| CVRANK_CAP1_minus_N0_seed1 | -1.535139 [-5.311284, 1.435279] | 3.433392 [0.981950, 7.034255] | 0.019229 [-0.013445, 0.062860] | -0.8955/0.6506 | 0 |

[확인] A=TF−BASE, B=TF−N3/PoseFix. 음수는낮은오차방향이며CI가0을포함하면확정개선이아니다. 모든비교·seed가원13세션동일draw10000회/seed20260917을공유한다. paired차이중앙값과marginal중앙값차이는다르다. 전체mean/P90·각seed/seed통계평균·13세션결과는RESULTS에보존했고TF를가짜seed3개로복제하지않았다. TF NoOp는8코너정확무이동프레임수이며보존된검출객체선택index0을actionNoOp로세지않는다.

[재집계] N0는 동일319장 saved pose/2D를 뒤늦게 연결한 참고행이다. 같은 원파일의N3 3seed가a22참조를완전히재현함을검산했고원PROTOCOL의NOT_LINKED를보존했다. 성능에따른선별이나추가N0 F/NN은0이며[N0_LINK_RECEIPT](N0_LINK_RECEIPT.json)에근거를남겼다.

| 방법 | 수정/초기유지 코너 | 이동 mean/median px | RAW good<5→bad>10 / bad>20→good<10 | fallback |
|---|---:|---:|---:|---|
| SUBPIX_NATIVE | 1206/1346 | 1.5696/0.0000 | 0/0 | {'outside_initial': 106, 'function_error': 1} |
| SUBPIX_CAP1 | 1206/1346 | 1.5696/0.0000 | 0/0 | {'outside_initial': 106, 'function_error': 1} |
| CVRANK_NATIVE | 2480/72 | 5.5554/5.1644 | 72/0 | {'no_candidate': 72} |
| CVRANK_CAP1 | 2480/72 | 5.1698/5.1644 | 40/0 | {'no_candidate': 72} |

[확인] NATIVE/CAP1 좌표·저장 자세의 정확 일치 집계: {'SUBPIX': {'frames': 319, 'native_points_exact_equal': 319, 'stored_pose_and_hypothesis_exact_equal': 319, 'scope': 'two retained deterministic output variants; not two independent trials'}, 'CVRANK': {'frames': 319, 'native_points_exact_equal': 90, 'stored_pose_and_hypothesis_exact_equal': 90, 'scope': 'two retained deterministic output variants; not two independent trials'}}. 같은 출력은 독립 반복이나 두 번의 성공으로 세지 않는다.

| 방법 | 가시성 | GT/관측 코너 | 2D median/P90 px | full PCK10 | RAW 양호손상/큰오류복구 |
|---|---|---:|---:|---:|---:|
| BASE | DIRECT_VISIBLE | 1776/1759 | 5.8291/25.7608 | 0.709459 | 0/0 |
| BASE | SELF_OCCLUDED | 462/448 | 8.8738/41.1316 | 0.532468 | 0/0 |
| BASE | EXTERNAL_OCCLUDED | 218/195 | 19.4851/80.8964 | 0.275229 | 0/0 |
| BASE | OUT_OF_FRAME | 43/43 | 12.1585/224.3253 | 0.441860 | 0/0 |
| N3_seed1 | DIRECT_VISIBLE | 1776/1759 | 4.8703/24.6385 | 0.779842 | 0/0 |
| N3_seed1 | SELF_OCCLUDED | 462/448 | 8.1740/37.9001 | 0.560606 | 0/0 |
| N3_seed1 | EXTERNAL_OCCLUDED | 218/195 | 19.2665/81.6416 | 0.256881 | 0/0 |
| N3_seed1 | OUT_OF_FRAME | 43/43 | 11.2727/223.4315 | 0.465116 | 0/0 |
| PoseFix_seed1 | DIRECT_VISIBLE | 1776/1759 | 4.7417/22.7453 | 0.775901 | 2/1 |
| PoseFix_seed1 | SELF_OCCLUDED | 462/448 | 8.8946/40.2939 | 0.541126 | 0/0 |
| PoseFix_seed1 | EXTERNAL_OCCLUDED | 218/195 | 17.4457/81.1897 | 0.293578 | 0/0 |
| PoseFix_seed1 | OUT_OF_FRAME | 43/43 | 10.2669/224.1964 | 0.488372 | 0/0 |
| N0_seed1 | DIRECT_VISIBLE | 1776/1759 | 5.0287/23.3780 | 0.768581 | 0/0 |
| N0_seed1 | SELF_OCCLUDED | 462/448 | 8.7460/40.1661 | 0.525974 | 0/0 |
| N0_seed1 | EXTERNAL_OCCLUDED | 218/195 | 20.1414/81.7818 | 0.279817 | 0/0 |
| N0_seed1 | OUT_OF_FRAME | 43/43 | 11.3056/223.8145 | 0.441860 | 0/0 |
| SUBPIX_NATIVE | DIRECT_VISIBLE | 1776/1759 | 5.7405/26.6647 | 0.695383 | 0/0 |
| SUBPIX_NATIVE | SELF_OCCLUDED | 462/448 | 9.2268/41.1316 | 0.521645 | 0/0 |
| SUBPIX_NATIVE | EXTERNAL_OCCLUDED | 218/195 | 19.7855/73.6052 | 0.275229 | 0/0 |
| SUBPIX_NATIVE | OUT_OF_FRAME | 43/43 | 13.0489/222.7928 | 0.441860 | 0/0 |
| SUBPIX_CAP1 | DIRECT_VISIBLE | 1776/1759 | 5.7405/26.6647 | 0.695383 | 0/0 |
| SUBPIX_CAP1 | SELF_OCCLUDED | 462/448 | 9.2268/41.1316 | 0.521645 | 0/0 |
| SUBPIX_CAP1 | EXTERNAL_OCCLUDED | 218/195 | 19.7855/73.6052 | 0.275229 | 0/0 |
| SUBPIX_CAP1 | OUT_OF_FRAME | 43/43 | 13.0489/222.7928 | 0.441860 | 0/0 |
| CVRANK_NATIVE | DIRECT_VISIBLE | 1776/1759 | 6.4031/27.3276 | 0.655405 | 41/0 |
| CVRANK_NATIVE | SELF_OCCLUDED | 462/448 | 11.8834/41.5575 | 0.372294 | 27/0 |
| CVRANK_NATIVE | EXTERNAL_OCCLUDED | 218/195 | 20.9068/72.9007 | 0.243119 | 2/0 |
| CVRANK_NATIVE | OUT_OF_FRAME | 43/43 | 14.4625/205.2812 | 0.372093 | 2/0 |
| CVRANK_CAP1 | DIRECT_VISIBLE | 1776/1759 | 6.3246/27.3276 | 0.666667 | 26/0 |
| CVRANK_CAP1 | SELF_OCCLUDED | 462/448 | 11.3059/41.5703 | 0.406926 | 13/0 |
| CVRANK_CAP1 | EXTERNAL_OCCLUDED | 218/195 | 19.8077/72.7715 | 0.256881 | 0/0 |
| CVRANK_CAP1 | OUT_OF_FRAME | 43/43 | 13.0489/205.2812 | 0.395349 | 1/0 |

[확인] 직접가시0..8점 영상히스토그램: {'0': 0, '1': 4, '2': 7, '3': 24, '4': 32, '5': 49, '6': 109, '7': 93, '8': 1}. 미주석/UNKNOWN/가림/실패를구분한다. 전체객체대칭대응후GT정준상태를붙였고층마다대칭을재선택하지않았다. 가림인과효과나독립참조가아니다.

[확인] 시간은새RUNTIME의동일환경패널만사용하며과거14.47/25.24ms에새CPU비용을더하지않는다. 보정단독과RAM원영상→전처리/YOLO→보정→실제F를구분한다.

| 경로 | 전체 median/P90 ms | 보정단독 median/P90 ms |
|---|---:|---:|
| BASE | 12.0821/13.7317 | NA/NA |
| N3_seed1 | 15.2230/17.1872 | 3.4469/4.8057 |
| PoseFix_seed1 | 25.8289/28.1500 | 14.0957/15.7435 |
| SUBPIX_NATIVE | 12.9673/14.2331 | 0.6009/0.9527 |
| SUBPIX_CAP1 | 12.2161/14.4161 | 0.6542/1.0843 |
| CVRANK_NATIVE | 27.2927/29.7644 | 15.0178/17.5041 |
| CVRANK_CAP1 | 26.9356/30.0628 | 14.9139/17.3566 |

[확인] 시간 계측 pipeline/F 1050회: 이전 실패 준비 1회와 유효 1049회. 측정 910행(7경로×130); BASE 유효 준비 19회·실패 소비 1회, 나머지6경로 준비 각20회. detector predict 1050회, N3/PoseFix head 각 150/150회, 추가 stage F 0회. 이전 실패·CPU fallback은 표 통계에 합치지 않았으며 내부 초기화 forward·수치 parity·원측정은 RUNTIME.json에 구분했다.

[확인] 정사각형119 상태: DONE. 단일세션2D보조이며새clean집합으로대체하지않는다.

| square119 모드 | 방법 | GT/관측 코너 | 2D median/P90 px | full PCK10/gross20 |
|---|---|---:|---:|---:|
| manual_declared | BASE | 602/597 | 5.5257/11.4390 | 0.852159/0.033223 |
| manual_declared | N3_seed1 | 602/597 | 5.0241/9.9928 | 0.892027/0.029900 |
| manual_declared | N3_seed2 | 602/597 | 4.9192/9.8711 | 0.893688/0.026578 |
| manual_declared | N3_seed3 | 602/597 | 5.0689/9.9324 | 0.892027/0.021595 |
| manual_declared | SUBPIX_NATIVE | 602/597 | 4.9034/11.9431 | 0.833887/0.033223 |
| manual_declared | SUBPIX_CAP1 | 602/597 | 4.9034/11.9431 | 0.833887/0.033223 |
| manual_declared | CVRANK_NATIVE | 602/597 | 4.1231/13.6015 | 0.812292/0.038206 |
| manual_declared | CVRANK_CAP1 | 602/597 | 4.1231/13.0841 | 0.832226/0.036545 |
| manual_in_frame | BASE | 600/595 | 5.5214/11.4732 | 0.853333/0.033333 |
| manual_in_frame | N3_seed1 | 600/595 | 5.0066/9.9641 | 0.893333/0.030000 |
| manual_in_frame | N3_seed2 | 600/595 | 4.8857/9.8351 | 0.895000/0.026667 |
| manual_in_frame | N3_seed3 | 600/595 | 5.0469/9.9565 | 0.891667/0.021667 |
| manual_in_frame | SUBPIX_NATIVE | 600/595 | 4.8727/11.9621 | 0.835000/0.033333 |
| manual_in_frame | SUBPIX_CAP1 | 600/595 | 4.8727/11.9621 | 0.835000/0.033333 |
| manual_in_frame | CVRANK_NATIVE | 600/595 | 4.1231/13.6015 | 0.813333/0.038333 |
| manual_in_frame | CVRANK_CAP1 | 600/595 | 4.1231/13.1069 | 0.833333/0.036667 |

[확인] declared602/inframe600를분리하고한세션일반화CI·자세평가는없다. 기존N3seed별/가시성/hist는RESULTS.square119에보존한다.

[기존 결과 인용] old visible==2 supervised1594 corners; BASE median/P90 6.36/43.89px, selector7.07/45.03px; source calibrated square patch x/y radius12, not Euclidean12 limit. 현재2499참조와합치지않고과거원인단정을복사하지않는다.

[기존 결과 인용] E/F: old clean78 native REF-target mean R0 3.3586px to REF OCC2.6490px; partial imitation, not physical-reference accuracy or augmented loss. same REF OCC S42 coordinates/two poses natural99: D9 T/R12.5120cm/6.1832deg to oldGEO11.3696cm/5.2635deg; TP90unchanged128.6450cm; Clean T/R worsen. old217 AGREE1218/1627 points, all six>20px corrections removed; large recovery0; oldDEV194/159, not319. clean78 S42 scheduled1282/2560, applied418/2560=16.328125%; planned and actual differ. pseudo추종/평가오차,D9→GEO선택변경,planned/applied를구분하고99/128/217의조건을현재319과합치지않는다.

[기존 결과 인용] EXPECT6D final100 NoOp=1.000000, mean gradientnorm=1.879794386556305e-09. 이전2304NoOp=RAW는그학습퇴화관측이며이번TF나모든보정실패의공통원인증명이아니다.

[확인] 반복DEV탐색적구간이고다중비교보정·사전실용동등성폭이없다. 유의하지않음은동등성입증이아니다. 2D/등록치수재구성참조이며독립물리실측0쌍/BLOCKED_DATA다. opticalcorner와semanticcorner는일치보장이없다. 원설정CV는source calibration이력이있으며사후rank/가중/반경변경으로구제하지않았다. 무이동=BASE를개선으로부르지않고이번두영상기반방법의범위만판단한다. 렌더링/CAD/다중시점/기초모델방식전반에대한결론이아니다.

[확인] 현재N3는 객체/박스/점수/중심8/결측을 고정한다. 정확도 F 1276회 + BASE parity 26회; 정확도 detector/N3/PoseFix 재추론0, 새 학습0. 보고 CPU 집계 3.271초, 보고 자체 NN/F/optimizer0. 추가 pose-cost 생성·논문/LaTeX/PDF/bib 수정·빌드0.

[RESULTS](RESULTS.json) · [PROTOCOL](PROTOCOL.json) · [PREDICTIONS](PREDICTIONS.json) · [RUNTIME](RUNTIME.json) · [CHECKS](CHECKS.json)
