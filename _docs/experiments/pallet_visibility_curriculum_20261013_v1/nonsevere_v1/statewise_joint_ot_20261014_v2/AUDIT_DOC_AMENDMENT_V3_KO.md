# 최종 v3 문서 해시 보충 감사

최종 README/RESULTS의 추가 설명은 기존 독립 완료 감사와 일치한다. 문서 관련 10개 확인은 모두 통과했으며 새로운 material finding은 없다. 원 감사 JSON·한국어 문서·숫자 근거는 수정하지 않았다. 전체 4,137개 감사나 추론·평가·테스트를 다시 실행하지 않았다.

- 자세 차이는 **수치적 차이**이며 ΔR matrix norm>1e-6 또는 Δt>1e-7m 기준이다. accepted CAL2/TEST7 중 TEST 한 건 Δt≈3.74e-7m을 별도 명시하고, 개선 사례 수로 해석하지 않는다.
- S 영상 배제는 **초기 기하가 S로 정한 ID**에 대한 보장이다. 실제 사람 SELF 중 CAL13/TEST13점이 BI로 분류된 한계를 두 문서 모두 밝힌다.
- 감사 링크와 33묶음·4,137확인·실패0 기록이 기존 감사에 맞는다. RESULTS는 감사 SHA를 역참조하지 않아 순환 해시를 만들지 않는다.

| 최종 문서 | SHA256 |
|---|---|
| [README_KO.md](README_KO.md) | `9765c3c63091ff920837cd2cc4d5993fb0d648efe7b527e8d4d1c23f3a5482b2` |
| [RESULTS_V3.json](RESULTS_V3.json) | `275a364402712ef948752f3baafd0f8aca486e4e07ad3889cc7a1875bafe565a` |

이 두 SHA는 원 감사에 기록된 해당 문서의 이전 SHA만 대체한다. 원 코어·모델·봉인된 예측·그림에 대한 기존 감사 결론을 변경하지 않는다. 상세 기록은 [AUDIT_DOC_AMENDMENT_V3.json](AUDIT_DOC_AMENDMENT_V3.json)에 있다.
