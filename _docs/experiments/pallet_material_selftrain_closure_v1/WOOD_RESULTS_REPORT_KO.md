# Wood matched RAW/corrected self-training 결과

**MATERIAL_GENERAL_SIGNAL**. Plastic corrected−raw: PCK10 +3.959pp / ADDsym AUC +0.02429. Wood corrected−raw: PCK10 +0.578pp / ADDsym AUC +0.00860. 이 판정은 material 내부의 같은 RAW/REF 학습 계약에 대한 기술적 비교이며, 모든 지표 동시 개선을 요구하지 않는다.

동일 Replay9/38 교사를 고정했다. Wood 후보 1000장 중 raw confidence 탈락 266장, raw flip/LOO 탈락 30장, 보정 후 all8 LOO 탈락 28장으로 공통 승인 676장이다. 실제 real512 replacement 슬롯에 노출된 고유 영상은 361장이다. 기존 synthetic512와 매 epoch 혼합하여 R0 복제 학생2개를 각5epoch/320update 학습했다. RAW/REF의 RGB·박스·support·순서·augmentation·optimizer·학습량은 같고 유효 pseudo 좌표만 다르다. 교사는 student 평가/배포 추론에 붙이지 않았다.

## 평가 population과 역할

Wood116의 교사 감독 exact-ID/SHA 중복은5장이고, 교사 recording에 속한 day20+night51장 전체를 평가에서 제외했다. Plastic night 교사의 REC_002도 Wood night와 같은 촬영이므로 함께 제외했다. 결과를 보기 전에 Wood main을 REC_039의25장 + REC_042의20장 =45장으로 고정했다. Clean38/Moderate7이며 Severe는 없음(0% 아님). 미주석 Wood 학습 후보는 REC_001/002에서만 고르고 기존319개 GT 영상의 exact-ID/SHA를 모두 제외했다. 교사-학생 source recording은 겹치지만 두 source recording은 Wood45 평가와 겹치지 않는다. 과거 Wood116 결과는 삭제하거나 새45장 수치로 교체하지 않았다.

## Wood 결과

| Material | Method | Images | Corners | PCK10 % | Med px | ADDsym AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Wood | R0 | 45 | 346 | 48.27 | 10.453 | 0.67050 |
| Wood | Raw ST | 45 | 346 | 47.11 | 10.639 | 0.65643 |
| Wood | Corrected ST | 45 | 346 | 47.69 | 10.623 | 0.66503 |

Wood corrected−raw 정답 진입 8 / 이탈 6점; 순변화 +2 / 분모 346. 프레임 평균오차 개선/악화/동일: 22/23/0. 큰 오류 >20→≤10 복구 0점, <5→>10 손상 0점이다. paired 변화는 같은 canonical GT point identity로 대응한다. 개별 후보/체크포인트/threshold를 결과로 다시 고르지 않았다.

## 모든2D/6D 지표

| Material | Method | Detected | Matched | Correct10 | PCK5 % | PCK10 % | PCK20 % | Med px | P90 px | >20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wood | R0 | 45/45 | 45/45 | 167/346 | 17.34 | 48.27 | 76.01 | 10.453 | 66.777 | 83 |
| Wood | Raw ST | 45/45 | 45/45 | 163/346 | 16.47 | 47.11 | 75.43 | 10.639 | 67.839 | 85 |
| Wood | Corrected ST | 45/45 | 45/45 | 165/346 | 17.34 | 47.69 | 75.43 | 10.623 | 68.985 | 85 |
| Wood | Source-only update | 45/45 | 45/45 | 164/346 | 17.34 | 47.40 | 74.86 | 10.570 | 68.191 | 87 |

| Material | Method | Pose coverage | Axis correct | R med/P90 deg | Yaw med/P90 deg | t med/P90 cm | IoU3D med | ADDsym AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wood | R0 | 45/45 | 40/45 | 1.545 / 9.190 | 0.471 / 3.583 | 2.095 / 8.011 | 0.7898 | 0.67050 |
| Wood | Raw ST | 45/45 | 40/45 | 1.598 / 9.084 | 0.536 / 3.469 | 2.349 / 8.259 | 0.7704 | 0.65643 |
| Wood | Corrected ST | 45/45 | 40/45 | 1.605 / 9.337 | 0.499 / 3.588 | 2.071 / 7.985 | 0.7740 | 0.66503 |
| Wood | Source-only update | 45/45 | 40/45 | 1.612 / 9.277 | 0.613 / 3.271 | 2.195 / 7.963 | 0.7713 | 0.65812 |

## Recording별 변화 및 LORO

| Material | Recording/session | N | Raw PCK10 % | Corr PCK10 % | Raw AUC | Corr AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Wood | REC_039 | 25 | 48.13 | 50.80 | 0.77990 | 0.78514 |
| Wood | REC_042 | 20 | 45.91 | 44.03 | 0.50210 | 0.51490 |
| Wood | SESSION_wood_183705 | 25 | 48.13 | 50.80 | 0.77990 | 0.78514 |
| Wood | SESSION_wood_184309 | 20 | 45.91 | 44.03 | 0.50210 | 0.51490 |

| Excluded recording | Remaining N | ΔPCK10 pp | ΔAUC |
| --- | --- | --- | --- |
| REC_039 | 20 | -1.887 | +0.01280 |
| REC_042 | 25 | +2.674 | +0.00524 |

두 recording의 기술적 민감도이며 독립 초기화 반복 또는 유의성 검정이 아니다.

## 감독·한계

Wood45에는 출처가 확인되는 직접 클릭 가시점이0개이므로 **Q1_WOOD=UNRESOLVED**다. legacy teacher 정합 점수가 좋아도 verified pseudo-quality로 승격하지 않는다. Plastic66 가시점 teacher44→50/66과 학생43/66 동률은 그대로 유지하며, 후속640-update 학생의44/66 결과로 원래320-update main을 대체하지 않는다. 두 재료 모두 반복 DEV이며 새로운 독립 TEST가 아니다. Wood는 단2recording이고 프레임/코너를 독립 반복으로 세지 않는다. 6D는 geometry-derived annotation reference 정합도이며 독립 측정된 물리 pose 정확도가 아니다. 재료별 별도 학생은 외부 material metadata로 선택한다(material type is externally provided for routed evaluation). 자동 material 분류·unknown-material 대응·DOPE 등 estimator-generalization은 검증하지 않았다. 재료별 절대 수치 차이는 난도·카메라·세션·학습 pool 차이도 포함하므로 material 자체의 인과효과로 단정하지 않는다.

## 학습 전 정정

학습 전 공통 support가6개 이상이어야 한다는 추가 gate가 잘못 들어가 전체 Wood pool을 거절한 기록이 있었다. Plastic 원래 export는 공통 support4/5개도 허용하므로 이 gate를 제거해 원래 계약을 복원했다. 당시 새 fit0·평가 scoring 전이었고, 필터 threshold나 승인영상676개는 바꾸지 않았다. 원래 결정·정정 이유·정정 결정 JSON을 모두 로컬에 보존했다. 이는 성능 결과 기반 재시도가 아니다. 공개 범위는 [안내](PUBLICATION_NOTICE_KO.md)를 참조한다.
