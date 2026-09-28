# C2: 실사 입력의 affine 증강만 끈 짝지은 대조

기존 REF를 개선하지 못했다. Plastic NEW_REF의 PCK10은 503/985로 기존507/985보다4점 낮고, Wood는164/346으로 기존165/346보다1점 낮다. ADDsym AUC도 각각 −0.000281250, −0.002466667이다. Plastic의 verified66 보조값과 PCK20에는 작은 이득이 있지만 전체 주목표에서 이득을 입증하지 못했으므로 기존 main을 교체하거나 이 설정을 재현 대상으로 선택하지 않는다.

## 실제 변경과 유효성

[사전 명세](SPEC.md)에 따라 R0에서 시작하는 PLASTIC_RAW/REF, WOOD_RAW/REF의4개 fit을 실행했다. 각각 seed42,320 optimizer update, 기존217/361 real sample 및 multiplicity,512real+512source slots/epoch, 같은 수동정보0추가·교사·bbox·common support·loss·pose/flow trainable 범위를 유지했다. 유일한 변경은 실사 `RandomPerspective`의 translate0.1과 scale0.25를 해당 호출 동안0으로 만드는 것이다. HSV, synthetic 증강, degree/shear/perspective, LR1e-5와 schedule을 유지했다.

소스 transform 및 RNG 소비를 보존하는 wrapper 검사는4 tests에 통과했다. fit 전 재료별32 source+32 real exposure에서 source RGB/boxes/keypoints bit-exact, real 입력32개 변경, RNG64개 일치, RAW/REF RGB·bbox·support64개 일치를 확인했다. 기존1024개 slot 전체의 image/order/bbox/support/source hash도 짝에서 일치했다.

실제 학습에서는 재료별320개 모든 batch의 이름·RGB·bbox·mask·batch_idx·노출이 RAW/REF에서 일치하고 좌표만 달랐다. source/real 입력 노출은 각2560회다. pose/flow 이외 가중치 및 모든 buffer747개를 초기 checkpoint와 비교해 저장된4개 checkpoint에서 exact 일치를 확인했다. 기존 per-fit 기록의 간단한 protected count744는 explicit BN 검사만 센 값이고, 최종 검사는 나머지 pose/flow buffer까지 포함한747개를 검사한다. 각 arm은 실제320update와 trainable tensor 변화가 확인됐고 마지막 `last.pt`만 사용했다. 최종 온도는56/58/61/62°C였다.

| 재료/역할 | 이미지 노출 | supervised 점 노출 | ignore 점 노출 | invisible 점 노출 |
|---|---:|---:|---:|---:|
| Plastic source | 2560 | 22531 | 0 | 509 |
| Plastic real | 2560 | 21855 | 1185 | 0 |
| Wood source | 2560 | 22531 | 0 | 509 |
| Wood real | 2560 | 22770 | 270 | 0 |

이 수는9점 visibility sentinel의 실제 batch 노출이며 독립적인 GT 점 수가 아니다. 실사 affine을 없애면 transformed-out-of-view 지원점 감소도 달라질 수 있으므로 입력분포와 기하 변형 제거의 총효과를 검사한 것으로 해석한다.

4개 학생의 raw native 예측과 기존 D9 pose를 먼저 잠그고 별도 score process에서 참조를 읽었다. 검출 candidate 순서·box·confidence는 기존 R0와 exact 일치했다. Plastic128/985, Wood45/346의 전체 분모, fixed-ID 보조 분모, detection-matched120/45와 pose coverage1.0을 확인했다. 기존 full-denominator·whole-object symmetry·실패 패널티·0–0.1 normalized ADDsym AUC 계약을 유지했다. legacy pose reference는 독립 물리6D가 아니며 두 평가 집합은 반복 사용 DEV다.

## 주 결과와 손익

median/P90은 검출 mismatch의 기존 native diagonal penalty까지 포함한 전체 코너 값이다. 모든 행에서 detection128/45, match120/45, pose coverage1.0이며 match 변화는0이다.

| 재료/arm | PCK5 % | PCK10 % | PCK20 % | median/P90 px | >20px count | ADDsym AUC |
|---|---:|---:|---:|---:|---:|---:|
| Plastic R0 | 20.812 | 49.137 | 72.690 | 10.18/70.63 | 269 | 0.337964844 |
| Plastic old RAW | 20.305 | 47.513 | 72.589 | 10.46/70.14 | 270 | 0.334722656 |
| Plastic old REF | 24.670 | 51.472 | 73.909 | 9.64/70.36 | 257 | 0.359015625 |
| Plastic SYN | 20.609 | 48.629 | 72.792 | 10.30/70.29 | 268 | 0.338945313 |
| Plastic NEW_RAW | 20.000 | 47.614 | 72.284 | 10.60/70.59 | 273 | 0.331742188 |
| Plastic NEW_REF | 24.162 | 51.066 | 74.619 | 9.76/70.71 | 250 | 0.358734375 |
| Wood R0 | 17.341 | 48.266 | 76.012 | 10.45/66.78 | 83 | 0.670500000 |
| Wood old RAW | 16.474 | 47.110 | 75.434 | 10.64/67.84 | 85 | 0.656433333 |
| Wood old REF | 17.341 | 47.688 | 75.434 | 10.62/68.98 | 85 | 0.665033333 |
| Wood SYN | 17.341 | 47.399 | 74.855 | 10.57/68.19 | 87 | 0.658122222 |
| Wood NEW_RAW | 17.052 | 48.266 | 75.723 | 10.51/67.41 | 84 | 0.655055556 |
| Wood NEW_REF | 16.474 | 47.399 | 75.145 | 10.79/69.01 | 86 | 0.662566667 |

Plastic에서는 NEW_REF−NEW_RAW가 PCK10 +34/985(+3.4518pp), AUC +0.026992188로 보정 타깃의 이득 자체는 남는다. NEW_REF−R0는 +19/985(+1.9289pp), AUC +0.020769531다. 그러나 oldREF보다 낮으므로 이를 새 recipe의 성공으로 쓰지 않는다. Wood NEW_REF는 NEW_RAW보다 PCK10 −3/346, AUC +0.007511111로 지표 간 부호가 다르며 R0 대비 PCK10 −3/346, AUC −0.007933333이다.

oldREF→NEW_REF의 canonical-identity 정답10px 진입/이탈은 Plastic10/14개, Wood2/3개다. >20→≤10 회복 및 <5→>10 손상은 두 재료 모두0이다. Plastic PCK20 정답은728→735(+7)로 늘었지만 PCK5는243→238(−5)이다. Wood는 PCK5 −3, PCK20 −1이다. >50px/>100px count는 Plastic REF122/82, Wood REF43/28로 기존과 같다.

REF의 pose-axis 정답은 Plastic82/128→84/128, Wood40/45→40/45다. Plastic R median/P90은3.766/88.809°→3.661/88.614°, yaw2.550/88.656°→2.435/88.378°, t9.186/78.938cm→8.988/78.383cm, IoU median0.592778→0.590359다. Wood R1.605/9.337°→1.620/9.213°, yaw0.499/3.588°→0.482/3.551°, t2.071/7.985cm→2.036/8.006cm, IoU median0.773980→0.774248다. 보조 지표의 작은 움직임을 전체 자세 개선으로 합치지 않았다.

## Recording·난도 및 검수66점

아래는 NEW_REF−oldREF이며 원시 recording 결과를 그대로 남긴다. 전체 LORO와 각 arm의 모든 요약은 [RESULTS.json](RESULTS.json)에 있다. Wood2recording과 단일 randomness seed에서 정밀 모집단 추정이나 통계적 일반화를 주장하지 않는다.

| 재료/recording | 정답10px 변화 | PCK10 변화 pp | AUC 변화 |
|---|---:|---:|---:|
| Plastic REC_007 | +3 | +1.1450 | +0.005803030 |
| Plastic REC_021 | −3 | −2.2059 | −0.006388889 |
| Plastic REC_022 | −5 | −4.0984 | 0 |
| Plastic REC_025 | −1 | −0.5076 | +0.022759259 |
| Plastic REC_027 | +2 | +2.1739 | −0.020500000 |
| Plastic REC_041 | 0 | 0 | −0.004700000 |
| Plastic REC_044 | 0 | 0 | −0.036166667 |
| Wood REC_039 | −1 | −0.5348 | −0.002040000 |
| Wood REC_042 | 0 | 0 | −0.003000000 |

Plastic CLEAN/MODERATE/SEVERE의 PCK10 변화는 −3/+2/−3코너, AUC 변화는 −0.018758621/−0.007238095/+0.008461538이다. Wood CLEAN/MODERATE는0/−1코너 및 AUC −0.002078947/−0.004571429다. Wood severe는 표본이 없으므로 NA다.

Plastic fixed-identity verified66점에서 old RAW/REF의 PCK10은43/43, NEW_RAW/REF는43/44다. R0와 SYN도44/66이다. NEW_REF의 PCK5는26→23, PCK20은63→63, median7.097→7.063px, P9017.343→16.094px다. 이1점 이득은16frame/66점 선택 subset의 보조 결과이며 전체128장 개선을 대신하지 않는다. Wood strict verified subset은 `NA_REFERENCE_NOT_VERIFIED`다.

## 사전 증강 진단의 큰 오차는 무엇이었나

기존 TRAIN32 exposure 진단에서 Plastic 평균 normalized residual0.05873→0.01276은 한 frame의1.385 오차가 지배했다. [TRAIN_OUTLIER_BOX_AUDIT.json](TRAIN_OUTLIER_BOX_AUDIT.json)은 이 frame을 기존 REF checkpoint·동일 RNG로만 재추론한2입력 진단이며 optimizer step0이다.

| 입력 | 후보 수 | highest-score box와 TRAIN pseudo box IoU | 후보 중 최대 IoU | mean normalized target residual |
|---|---:|---:|---:|---:|
| 기존 affine | 8 | 0.000000 | 0.875285 | 1.384746 |
| 실사 affine off | 5 | 0.969505 | 0.969505 | 0.018639 |

따라서 해당 큰 잔차는 affine 입력에서 최고 confidence 검출이 의사 타깃과 다른 instance를 선택한 것과 연결된다. 의사 bbox와의 일치가 GT 검출 정확도를 뜻하는 것은 아니지만, 이 관측은 평균 차이를 광범위한 좌표 회귀/감독 전달 실패의 증거로 쓰는 설명을 반박한다. 이 진단에서도 candidate를 타깃 기준으로 선택한 예측을 배포 성능으로 보고하지 않았다. 나머지 TRAIN 분포·tail·표현력·타깃 품질에 대한 원인은 여전히 분리되지 않았다.

## Source 보존과 비용

추가 source GPU benchmark를 돌리지 않고 기존과 동일32 synthetic validation의 마지막 CSV 값을 비교했다. Plastic old/new RAW/REF pose mAP50–95는 모두0.98863, Wood RAW old/new도0.98863이며 Wood REF는0.98478→0.98863이다. detector mAP50–95는 전부0.94734다. pose validation loss는 Plastic RAW0.09621→0.09527, REF0.09624→0.09638, Wood RAW0.09556→0.09464, REF0.09797→0.09654다. 이는 작고 반복 사용된 framework validation의 보조 요약이며 source6D 일반화나 checkpoint 선택 근거가 아니다.

fit GPU 예약시간은 Plastic RAW75.6466초, REF72.4486초, Wood RAW74.0307초, REF74.0296초로 총296.1555초다. 동결 DEV inference7.9575초, TRAIN outlier 검사1.0877초를 더하면 계측 GPU 사용 구간은305.2007초(약0.08478시간)다. 준비 CPU preflight의 성공 실행11.8181초, score CPU wall3.2605초다. Python startup/import와 문서 작성은 이 함수 내부 계측과 전체 프로젝트 wall에서 구분한다.

CPU preflight 첫 시도는 모든 parity 검사를 수행한 뒤 설치된 외부 module의 경로를 repo-relative hash helper에 전달해 metadata serialization ValueError로 종료했다. 외부 absolute-path/hash binding으로 수정하고 같은 명세·동일 fingerprint를 다시 확인했다. 이때 fit0/update0/GPU0이며 첫 실패의 wall 계측은 저장되지 않아 결측으로 남긴다.4개 fit은 모두 첫 실행에서320update로 완료했고 미완료 fit·숨긴 재학습은 없다. 시작 시 sandbox CUDA 접근이 없어 승인된 host 실행을 사용했다. 기존 설치된 Albumentations의 ImageCompression 인자 호환 경고는 source와 두 paired arm에서 동일하게 나타났으며 해당 선택적 변환이 비활성화된 기존 런타임을 바꾸지 않았다.

실행 전4 tests(0.017초), 완료 artifact·747buffer를 포함한6 tests(0.342초), 공개 주표12행과 저장 결과의 수치 일치까지 포함한 최종7 tests(0.364초)가 통과했다. 참조·기존 per-frame metric·solver/evaluator·checkpoint source hash를 결과에 기록하고 검증했다. 원시 batch fingerprint·좌표·예측·가중치는 private 결과 namespace에만 저장했다.

## 판정과 다음 결정

`EVIDENCE_VALIDITY=VALID`(짝지은 개입 계약; 정확도 해석은 legacy reference 제한), `HEADROOM=UNRESOLVED`(이 실험은 oracle를 재정의하지 않음), `RECOVERY=NOT_DEMONSTRATED`, `CAUSE=MULTIPLE_EXPLANATIONS`다. 작은 보조 개선이 있어도 두 재료 주목표에서 기존 REF를 넘지 못했고, 사전 진단의 주 outlier는 instance 선택 문제였다. 이 설정에서 real affine 제거가 자기학습의 제한된 이득을 해결한다는 가설은 지지되지 않았다. 모든 증강 또는 최적화 계열이 무효라는 결론은 아니다.

예측점을 바꾸는 학습 실험이므로 이전 고정 pose 후보 집합의 oracle gap 회수율을 적용하지 않았다. 결과 확인 전 정한 Plastic 주목표 개선 조건이 성립하지 않아 seed43 짝지은4fit 재현은 자동 실행하지 않는다. 남은 별도 가설의 실행 여부는 parent의 독립 명세에 따른다. 기존 논문 main·baseline·교사·checkpoint는 보존했다.
