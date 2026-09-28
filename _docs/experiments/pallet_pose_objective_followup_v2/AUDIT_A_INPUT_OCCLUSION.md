# A 입력 가림 독립 CPU 감사

결론: `PASS_WITH_EXPLICIT_LIMITATIONS`. [기계 판독 결과와 SHA binding](AUDIT_A_INPUT_OCCLUSION.json). 이 감사는 학습/GT reference를 실행하지 않고 구현·기존 preflight·완료된 paired trace를 읽었다. fit0/update0/GPU0, 실제 CPU process wall 약1.39초(내부 검사0.50초). 평가 성과나 checkpoint tensor 동결에 대한 독립 검증은 이 범위에 없다.

## 독립 검사

- `random_plan`에 실제 데이터가 아닌 합성9점 fixture, support0–9개, 각300seed로 총3,000계획을 실행했다. 같은 seed의 계획 동일, 입력 mask 불변, center 계획 제외, 적용927개의 canvas 경계/bbox overlap/covered≥1/remaining≥2를 확인했다. 3점 미만이면 적용되지 않는다.
- RAW/REF 실제320batch, 각16occurrence를 전수 읽었다. batch·epoch·RGB digest·이름/순서·bbox·batch index가 동일했고 각5120occurrence의 모든 공통 계획 필드가 동일했다. 이는 manifest 수준뿐 아니라 학습 trace 수준 비교다.
- SOURCE2,560 occurrence는 모두 `source_unchanged`, 추가 가림0이었다. baseline 원본 RGB/RNG 보존은 [기존128표본 preflight](OCCLUSION_PREFLIGHT_PLASTIC.json)의 검사와 해당 assert 코드를 확인했다. 이번 독립 작업에서 그 전체 transform을 다시 돌렸다고 주장하지 않는다.

| 항목 | 실제 수 |
| --- | ---: |
| Real occurrence | 2,560 |
| Scheduled | 1,302 |
| 실제 가림 적용 | 542 (21.17%) |
| 32개 후보 내 유효 위치 없음 | 760 |
| 미선정 | 1,258 |
| 가려진 감독점 RAW / REF | 781 / 824 |
| support가 다른 batch | 3 / 320 |
| Real v2 RAW / REF | 21,819 / 21,823 |
| Real v1 RAW / REF | 1,185 / 1,185 |
| Real v0 RAW / REF | 36 / 32 |

## 해석의 한계

기존 affine의 좌표별 out-of-frame 처리를 보존했으므로 저장 공통 mask가 모든 증강 support tensor의 동등성을 뜻하지 않는다. 같은 RGB mask에서도 RAW/REF 위치가 달라 covered supervision 양이 다르다. 따라서 동일 masked-target 노출량을 강제한 실험이 아니다.

`SharedOcclusion`은 기본 transform을 실행한 뒤, 같은 초기 RNG로 REF 좌표 복사본만 dry-run하고 실행 후 RNG를 복원한다. 계획은 고정 REF 위치 조건을 양 팔에 공통 사용한다. RAW가 teacher 정보를 전혀 사용하지 않는 대조라고 하지 않는다. 현재 가림 위치 영역은 최종 입력 전체 canvas이며 기존 Clean19의 변환된 native-canvas 교집합과 다르다. S2 구조 pairing·edge 의존은 제거되었고 원래 target/v 값을 다시 쓰는 코드는 없다. center는 계획에서만 제외된다.

이 감사는 문헌 원리의 faithful reproduction 판정이 아니다. A는 고정 pseudo의 input-only perturbation을 검사하는 새 팔레트 대조이며 원논문의 dual-model heatmap 또는 다른 이미지 limb patch를 재현하지 않는다. 자연가림의 T/R 성과는 별도 동결 evaluator 결과로 판정해야 한다.
