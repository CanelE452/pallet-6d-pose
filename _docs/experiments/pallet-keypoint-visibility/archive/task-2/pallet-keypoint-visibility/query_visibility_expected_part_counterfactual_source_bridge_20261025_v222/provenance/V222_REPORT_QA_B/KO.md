# V222 저장 SOURCE 결과·보고서 QA

**PASS.** 최종 두 Markdown은 byte동일(9bfc7baf…)이며, CSV16행 및 JSON의 CM/P/R/U·paired 회복/손상 산술이 맞고 A 완료 감사의 SOURCE 집계와 일치한다. 비교 V1/V2의 숫자 JSON·CSV도 각각 byte동일하다.

| 분리한 대상 | C66 | C67 |
|---|---:|---:|
| CF FIT 성공 triplet | 1/39 | 39/39 |
| CF FIT 전체 행 정답 | 78/117 | 117/117 |
| CF DEV 성공 triplet | 1/9 | 2/9 |
| CF DEV 전체 행 정답 | 19/27 | 17/27 |
| natural DEV 전체 정답 | 11/12 | 9/12 |
| strong known TRAIN | 6096/6096 | 6096/6096 |

CF DEV는 회복2/손상4(순−2)이며 D회복0/손상3, E회복2/손상1이다. triplet 정답 증가만으로 전체 개선이라고 할 수 없다. natural DEV는 D손상2이며 E정답은 그대로다. CF DEV는 실제2 component의9상관 triplet이며 native1/4만 포함한다. strong6096은 known apparent TRAIN 분모로 input7040/UNKNOWN944와 구분된다.

CF DEV CM은 C66 `[[17,1,0],[7,2,0]]`, C67 `[[14,4,0],[6,3,0]]`다. D precision/recall은70.83/94.44%→70.00/77.78%, E는66.67/22.22%→42.86/33.33%로 최종 표가 정확하다. natural DEV D는90/100%→87.5/77.78%, E는100/66.67%→50/66.67%다.

각1100회 실제 continuation과 동일 C65 초기 상태·optimizerreset·공통draw·118359파라미터·두 final 선동결은 fit COMPLETE/A 완료 감사에 근거한다. probe64는 이번 task에서 원receipt를 재개봉하지 않아 Root 실행기록에 귀속한다. A는 old192 조건부 확률 전체나 neural training을 재현한 것이 아니며, 이 한계가 보고서에 명시됐다.

훈련 V1의2schema gate는 실행 전에 발견되어 새V2에서만 고쳤다. 모델 재학습 실패/선택을 숨긴 일이 아니다. 비교 그림 V1/V2는 여백수리이며 숫자파일 byte는 같다. 보고서의 판정경계도 실제코드와 같은 `z>=0`으로 교정됐다.

부모 q8은 GT-derived training family이고 finite-part weak는 physical native-point gold가 아니다. 원initializer-query9는 쓰지 않았다. 추가자료+손실가지 bundle이며 같은 예제 수/주소화 단독 인과 비교가 아니다. SOURCE 개선·프로덕션 q 성공·새current223추론·실사 일반화·채택은 입증되지 않았다.

B는 저장 JSON/CSV와 fit/A감사 metadata만 읽었다. 새이미지나 source input/GT/NPZ/PT/모델/보호셋은 열지 않았다. 사례 이미지는 Root의9장 실제관찰에 귀속하며 B가 재시각검증했다는 뜻은 아니다. provenance/ARCHIVE_MANIFEST.json은 Root가 아직 생성 전이라고 명시한 게시용 대기 산출물이므로 전체링크존재 PASS라고 주장하지 않는다. 과학원본 수정0. 정확 binding·세부표·한계는 [QA.json](QA.json)에 있다.
