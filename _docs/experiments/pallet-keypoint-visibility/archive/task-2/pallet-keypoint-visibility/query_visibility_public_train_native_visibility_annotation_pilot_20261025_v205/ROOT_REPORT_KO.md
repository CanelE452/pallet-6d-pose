# V205 공개 TRAIN 원 query/native 보임·가림 약관찰

고정한 6프레임·48 native에서 두 assistant 관찰을 완료했다. 사전 규칙을 만족한 같은 D/E 약합의 후보는 D 1개, E 0개다. 원 입력 U 8개를 포함해 48개를 모두 유지했다. 새 학습·모델 평가·학습 채택은 수행하지 않았다.

이번 결과는 학습 감독을 준비하는 SOURCE TRAIN 진단이다. 실제 모델 precision/recall이나 라벨 정확도가 아니며, 독립 인간 평가·물리 native gold도 아니다. 관찰자 A/C는 같은 자료를 독립적으로 보았고 서로의 값·GT·기존 약라벨·모델 점수를 보지 않은 상태에서 각각 봉인했다. 두 assistant의 일치가 정확성을 보증하지 않는다.

## 데이터와 보존 조건

기존 공개 MR6D의 SOURCE TRAIN 155 원프레임 중, 기존 direct 통제쌍에 쓰인 12프레임을 제외하고 원본 frame key의 SHA 순서·prefix 할당 2/2/1/1·서로 다른 그래프 성분 6개로 미리 선택했다. 그래프 성분 6개는 독립 촬영 6회라는 뜻이 아니다. 6개를 보고 선택을 바꾸지 않았으며 155개 전체의 빈도나 성능으로 일반화하지 않는다.

원 fullRGB, 원 native0..7의 q/L, raw96 crop, validity, available/U를 유지했다. 생성 PNG 114개의 저장 해시·decoded RGB/crop/valid 및 contact 안의 clean96가 원 입력 메타데이터와 일치했다. fullframe의 q8 표시는 표시용이다. 원 선택 검출 bbox는 메타데이터에 없어 추가하지 않았고 GT target으로 대체하지 않았다.

현재223·publicheld45·SOURCE HOLDOUT·COMMON·기존 VAL은 이번 관찰과 학습 준비에 사용하지 않았다. 경로의 val/000000 등은 공개 배포 이름이며, 사용 여부는 고정 SOURCE TRAIN 역할로 결정했다. 새 initializer·crop sampling·q/L 보정·PnP·SAM·모델 forward는 없다.

## 사전 결합 규칙과 결과

D는 의도한 대상 부위가 원 crop에서 보이는 근거, E는 식별된 부위를 외부 전경이 가리는 근거가 필요하다. 단순한 빈 공간·다른 목재·query 이탈을 E로 바꾸지 않는다. 자기 가림 S, 좌표 이탈 가능성, 부위 불명확은 분리한다. D/E 후보는 두 관찰의 같은 D/E·target/part YES·같은 ON 또는 같은 NEAR·location NO와 각 시각 근거를 모두 요구한다. E는 양쪽 외부 차폐 근거도 필요하다.

| 결합 분류 | 수 |
|---|---:|
| ALIGNED_WEAK_CONSENSUS_CANDIDATE | 1 |
| DISAGREEMENT_OR_UNRESOLVED | 1 |
| INSUFFICIENT_ALIGNED_EVIDENCE | 1 |
| LOCATION_ERROR_OR_OFF | 37 |
| ORIGINAL_INPUT_U | 8 |

A의 전체 상태: D=12, E=1, UNKNOWN=35. C의 전체 상태: D=13, E=2, UNKNOWN=33. 이 원상태 수와 엄격한 합의 후보 수는 서로 다른 분모/조건이다.

관찰 후 축별 기술 집계를 별도로 확인했다. 같은 D 판단은 11개, 같은 E는 1개다. 같은 D 중 query도 같은 ON/NEAR인 것은 8개이며, 그중 7개는 location_error_possible 표시가 다르다. 이 집계는 사전 결합 규칙을 변경하거나 새 학습 채택 기준을 정한 것이 아니다.

LOCATION_ERROR_OR_OFF 분류는 한 관찰자라도 OFF 또는 위치 오류 가능성 YES를 기록하면 붙는다. 따라서 이 수는 확인된 좌표 오류의 개수가 아니다. 예를 들어 같은 NEAR·같은 D에도 한쪽은 작은 이탈을 허용하고 다른 쪽은 위치 오류 가능성을 표시했다. 양쪽 NO를 요구한 엄격한 준비 수를 전체 보임 부위의 수로 바꾸어 말하면 안 된다.

| 원 frame key | 슬롯/U | A 상태 | C 상태 | 결합 분류 |
|---|---:|---|---|---|
| val/000000:17 | 8/1 | D=5, UNKNOWN=3 | D=4, UNKNOWN=4 | LOCATION_ERROR_OR_OFF=7, ORIGINAL_INPUT_U=1 |
| val/000000:67 | 8/0 | UNKNOWN=8 | UNKNOWN=8 | LOCATION_ERROR_OR_OFF=8 |
| val/000002:32 | 8/2 | D=3, E=1, UNKNOWN=4 | D=4, E=2, UNKNOWN=2 | ALIGNED_WEAK_CONSENSUS_CANDIDATE=1, DISAGREEMENT_OR_UNRESOLVED=1, INSUFFICIENT_ALIGNED_EVIDENCE=1, LOCATION_ERROR_OR_OFF=3, ORIGINAL_INPUT_U=2 |
| val/000002:37 | 8/4 | UNKNOWN=8 | UNKNOWN=8 | LOCATION_ERROR_OR_OFF=4, ORIGINAL_INPUT_U=4 |
| val/000005:21 | 8/0 | UNKNOWN=8 | UNKNOWN=8 | LOCATION_ERROR_OR_OFF=8 |
| val/000006:6 | 8/1 | D=4, UNKNOWN=4 | D=5, UNKNOWN=3 | LOCATION_ERROR_OR_OFF=7, ORIGINAL_INPUT_U=1 |

## 해석과 다음 행동

V204에서 확인한 finite-end 통제 라벨과 query 중심 개입의 차이를, 자연 공개 실사의 의도한 부위·query 관계와 구분해 기록할 수 있게 됐다. 이 파일은 원인을 확인하기 위한 자료이며, 기존 모델 실패의 인과효과를 측정한 결과는 아니다.

현재 같은 D 약합의 후보 1개는 후속 source 감독의 출발점으로 검토할 수 있다. 외부 E 약합의 후보 0개만으로 자연 외부 가림의 범위를 충분히 확보했다고 할 수 없다. 위치 이탈·불명확·불일치는 원 상태와 근거를 남기고 E로 강제하지 않는다.

후속 작업은 SOURCE TRAIN에서 식별 가능한 부위의 시각 근거를 원 crop 안에 더 구체적으로 남기고, 부위의 가시성과 q의 위치 관계를 각각 기록하는 것이다. 이를 바탕으로 해당 부위를 실제로 덮는 외부 가림과 같은 물체를 다른 곳에 놓는 대조쌍을 준비할 수 있다. 완벽한 q 정합을 가시성의 선결조건으로 올리지 않으며, 원 q/L/crop은 바꾸지 않는다. 기존 finite-end 라벨을 native 정답으로 복사하지 않는다. 이 후속 부위 표시·개입·학습은 아직 구현하거나 실행하지 않았다.

현재 평가셋 성능은 V203의 이미 동결된 비교가 최신이다. V205는 성능 개선·모델 채택·목표 해결을 주장하지 않는다.

## 직접 확인

[6프레임 원 q8·crop 및 48개 관찰표](PORTABLE_GALLERY_KO.md) · [전체 결합표](joined_source_observations_v1/JOINED_OBSERVATIONS48.csv)

공개 이미지 출처: [MR6D 데이터셋](https://huggingface.co/datasets/anas-gouda/mr6d), Anas Gouda 등, pinned revision `3c4ed8453312655499c3d3a3cf38b4929fae77ee`, 배포 라이선스 CC-BY-4.0. 기존 SOURCE 원본만 사용했으며 새 데이터 다운로드는 없다. 원 출처·라이선스의 상세 메타데이터는 앞선 V204 공개 기록에 보존했다.
