# 전체 점 묶음 정렬: 최초 기준 범위와 정정 기록

## 최초 실행

동결된 SYN/MIX 영상 전용 모델의 공간 확률을 평균하고, 각 코너의 5×5 국소 확률 질량을 계산했다. 원래 8개 점의 상대 배치를 유지한 채 전체 index 순열 4개와 공통 2D 평행이동을 탐색했다. 새 CAD·depth·수작업 레이블이나 프레임 탈락 필터는 추가하지 않았다. 중심점 및 검출 메타데이터는 원래 값으로 유지했다.

합성 32장 calibration / 32장 test를 사용했다. 각 이미지에 원래 입력, 두 크기의 전체 이동, 전체 quarter index 재배열, quarter+이동의 5개 view를 만들었다. 정답은 proposal 생성 입력이 아니라 채점·기준 설정에만 사용했다.

최초 후보 gain 기준은 0~16을 0.25 간격으로 나눈 값과 null(1e9)이었다. 이 범위에서는 normal 보존과 선택된 case의 개선 precision 95%를 함께 만족하는 non-null 기준이 없었다. 따라서 당시 고정된 정책은 모든 입력에 대해 R0 보존이다.

## 실행 장치와 null 정책 처리

시작 직전 별도 `pallet_direct_dimension_v1.evaluate synthetic` GPU 작업이 감지돼 보호 검사가 실행을 중단했다. 그 프로세스를 종료하거나 보호 검사를 우회하지 않았다. 동일한 frozen weights·저장 특징·수식으로 합성 320개 case의 proposal 및 calibration을 **CPU float32, 2 threads**에서 완료했다. 이후 실사 GPU 시도도 별도 평가의 다음 단계와 겹쳐 시작 전 중단됐다.

원래 기준 1e9는 가능한 점수 증가량보다 크다. 확률 질량을 1e-12~1로 제한하므로 평균 log score의 증가 상한은 약 27.631이며, float32 여유를 고려한 보수 상한 64도 훨씬 작다. 따라서 해당 null 정책의 실사 결과는 신경망을 다시 계산하지 않아도 정확히 R0다. `dino_joint_translate_null.py`에서 이 조건을 검증한 뒤 194장 원본 예측을 그대로 보존해 채점했다. **실사 registration proposal을 GPU로 계산했다고 보고하지 않는다.**

결과는 변화 0장, 큰 오류 복구 0개, 정상 손상 0개다. 안전한 보존일 뿐 개선이나 전체 목표 완료가 아니다.

## 중요한 정정

감사에서 모든 실제 source gain 경계를 확인하자 **16보다 큰 값에서 통과하는 조건이 존재**했다. 최선의 source-only 기준은 16.976975560188293이며, calibration의 인공 오류 포함 160개 case 중 25개를 선택했고 개선 precision 96%, >20→≤10px 복구 175개였다. 원래 raw calibration 32장은 여전히 하나도 바꾸지 않았다.

따라서 **"유효한 기준이 전혀 없다"는 결론은 부정확하다. 최초 기준 범위가 잘렸다.** 최초 실행·결과를 고치거나 삭제하지 않고, 같은 source proposal과 모델을 유지한 채 모든 gain 경계를 사용하는 [정정 비교](../dino_joint_translate_exact/INTERPRETATION_KO.md)를 별도 phase에서 수행했다. 이 정정에 실사 GT는 사용하지 않았다.

검사: 회귀 39개, CPU source 320개 정확 재현, 전체 calibration grid 재계산, 실사 194장 원본과 정확 일치, score 상한에 따른 null shortcut 확인. GPU 미실행 proposal 결과를 만들어낸 것이 아니다.

근거: [원래 결과](RESULTS.json), [CPU 실행 기록](CPU_SOURCE_ADAPTER.json), [null 처리 근거](NULL_POLICY_SHORTCUT.json), [전체 source 경계 진단](EXACT_SOURCE_FRONTIER_DIAGNOSTIC.json), [검사](COMPLETION_AUDIT.json).
