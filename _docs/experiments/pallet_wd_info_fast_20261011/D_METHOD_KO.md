치수와 후보별 영상 특징을 직접 연결한 소형 보정기는 실제 구현·학습됐지만, SubPix 이후 주 대비에서 W/D 혼동은 같고 성공은 1장 감소했다.

[확인] 허용된 실사 173장·9세션만 새로 평가했다. 축이 불명확한 3장은 W/D 정답률에서 UNCLEAR로 남기고 자세 성공의 분모 173장에는 포함했다. 보호 중인 물리적 FINAL 146장은 새 특징 추출·평가에 사용하지 않았다. 모집단 근거는 [INPUT_CONTRACT.json](INPUT_CONTRACT.json)이다.

[확인] D_JOINT_DIM_SUBPIX와 D_CONTROL_SUBPIX의 혼동은 모두 41/170, strict T<5cm AND R<5° 성공은 85/173 대 86/173이다. 평균 T는 58.363 대 58.480cm, 평균 R은 21.788 대 21.815°이지만 T 중앙값은 4.675 대 4.610cm, P90은 51.284 대 51.063cm로 악화했다. Stage1에서 혼동은 40/170 대 42/170으로 줄었지만 성공은 77/173 대 81/173으로 감소했다. 결과 근거는 **[D_REPAIRED_RESULTS.json](D_REPAIRED_RESULTS.json)**이며 최초 adapter 오류 산출물은 아래 설명대로 구분한다.

![Implemented local residual scorer](FIGURES/D_LOCAL_METHOD.png)

[확인] 기존 OLD_P seed1 checkpoint의 18,962개 파라미터를 동결했다. 일반 후보 221개의 기존 scorer 입력 64차원은 영상 pooled48, 문맥13, 변위2, coverage1이다. null 후보의 입력61은 영상 평균48과 문맥13이며 0 세 칸으로 패딩했다. null flag1와 metadata5를 더해 두 arm 모두 `70→32(SiLU)→1`, 학습 파라미터 2,305개를 가진다. 원본 source를 수정하지 않고 read-only pre-hook으로 이 입력을 얻었다. 마지막 층 weight와 bias는 0이며 초기 logits, decoded 좌표와 hook 원출력이 bitwise 일치했다. [D_CONTRACT.json](D_CONTRACT.json), [D_THROUGHPUT.json](D_THROUGHPUT.json).

[확인] CONTROL은 metadata 다섯 칸을 0으로 고정하고 JOINT는 TRAIN-only 정규화된 canonical `logW, logD, logH, log(W/D), log(H/√WD)`를 넣었다. 두 scorer의 초기값·파라미터 수·seed1·batch16·원래 TRAIN order의 첫 1,000 batch·대칭 감독·loss·optimizer가 같다. AdamW lr0.001, weight decay0.0001, betas(0.9,0.999), gradient clip5, warmup100 이후 1,000-step cosine final fraction0.1이다. 각 arm을 정확히 1,000 updates 학습했고 마지막 checkpoint만 썼다. 고유 TRAIN 노출은 16,000장이다. R0와 OLD_P를 추가 학습하지 않았다. [D_TRAINING_RECEIPT.json](D_TRAINING_RECEIPT.json).

[확인] 전체 기존 synthetic calibration 1,004장에서 원래 fixed-index Gaussian candidate CE로 T={0.5,1,2,4}만 비교했다. 두 arm의 T는 모두 1, lambda1과 원영상 대각선1% cap은 기존 값을 유지했다. 실사 결과로 선택하지 않았다. [D_CALIBRATION.json](D_CALIBRATION.json).

[확인] 새로운 예측은 CONTROL/JOINT의 Stage1·SubPix, JOINT의 mean/shuffle 입력, OLD_P→SubPix의 7경로다. 원래 N3·N3→SubPix와 OLD_P Stage1의 자세는 같은 seed1에서 기존 cache를 재사용했다. 기존 N3 seed1·2·3은 동결된 별도 대조군이며 D 파일럿은 독립 학습 seed가 1개뿐이다. SubPix는 기존 (5,5), (-1,-1), EPS|COUNT 40/0.001이다. SubPix 시작점은 기존 cap을 가진 Stage1 좌표이고, SubPix 반환의 마지막 cap은 원래 Base q0 기준으로 한 번 적용한다. centroid8·결측 mask·box·confidence를 보존한다.

[확인] 치수를 mean으로 바꾸면 좌표 변화 중앙값0.470px/P90 1.754px, 선택 전환2장; TRAIN 지원 치수 shuffle에서는 0.406px/P90 1.946px, 선택 전환3장이었다. 이때 downstream PnP에는 계속 원래 올바른 치수를 썼다. 민감도는 있지만, mean/shuffle의 Stage1 성공이 각각79/80장으로 올바른 치수의77장보다 많아 실제 치수 활용 이득을 단독 입증하지 못한다. [D_REPAIRED_SENSITIVITY.json](D_REPAIRED_SENSITIVITY.json).

[확인] Joint SubPix의 코너 평균 오차는 CONTROL보다 107장에서 개선, 58장에서 악화, 8장에서 같았다. 양호 코너<5px가 >10px로 손상된 사례는 1개이며 `eval_cad:1778653055801377280` corner3은 4.554→10.882px다. 큰 오류>20px의 <10px 회복은 0개다. [전체 짝 전환](D_ALL_PAIRED_TRANSITIONS.csv), [극단 코너 손상](D_EXTREME_CORNER_TRANSITIONS.csv).

[확인] 첫 평가 adapter는 JSON의 K를 list로 전달해 final OpenCV solve에서 실패했다. selector 후보는 계산됐지만 1,211개 최종 자세가 모두 무효였고 6,055 solver 호출이 소비됐다. **D_ROWS.jsonl.gz·D_RESULTS.json은 이 무효 첫 시도를 보존한 파일이며 최종 방법 성능이 아니다.** 예측 좌표·후보 점수·치수·학습 checkpoint를 유지하고 봉인된 원래 selector 선택만 재사용해 K를 float64 ndarray로 변환하여 original F의 final8 fit을 재개했다. 2,422 호출과 fresh full F 검산18호출을 더해 누적8,495 호출이다. 검산은 첫 영상의 세 arm 예측을 비교한 것이며 세 독립 영상은 아니다. [D_REPAIR_LOCK.json](D_REPAIR_LOCK.json), [D_REPAIRED_POSE_SEAL.json](D_REPAIRED_POSE_SEAL.json), [D_VERIFICATION.json](D_VERIFICATION.json). 수정 후 모든 경로가 173/173 자세를 산출한다.

[확인] 최종 원행은 [D_REPAIRED_ROWS.jsonl.gz](D_REPAIRED_ROWS.jsonl.gz)이다. 독립 arithmetic으로 400개 평균·SD·중앙값·P90·P95를 검산했고 최대 절대 차이는 5.684e-14다. 기존 source와 checkpoint hash는 같고 adapter 수정 전후 코너 지표도 같다. 후보 표의 두 R/t는 원래 selector fit이며, 원래 F가 추가로 계산하는 final8-corner fit과 같다고 표현하지 않는다.

[확인] C 분류기의 분할과 D의 기존 보정기 TRAIN 계약은 전 실험 단일 분할이 아니다. D의 추가 fitting에 C TRAIN1,133·VAL253·TEST271장이 노출됐다. D pair 내부의 분할은 동일하고 실사로 fitting/선택하지 않았지만, C와 D의 합쳐진 독립 synthetic TEST 또는 직접 우위 비교는 유효하지 않다. 데이터를 다시 나누거나 추가 fitting하지 않았다. [D_CROSS_TASK_EXPOSURE.json](D_CROSS_TASK_EXPOSURE.json).

[추정] 현재 예산의 결과는 치수 민감도를 확인하되 실제 6D 개선은 미해결이며 손실도 있는 파일럿이다. 1,000 updates·seed1과 개발 모집단만으로 이 연결 위치 자체를 기각할 수 없다. 추가 설정 탐색 없이 여기서 종료한다. 세션 짝 CI와 정해진 Holm family의 결과는 상위 종합 보고서에서 함께 해석한다.
