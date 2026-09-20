# 일반 플라스틱 극저각 제외 subset v1

194장 중 RGB 정성 기준으로 명확한 극저각 35장을 제외한 **159장**이다.
초록150·목재125는 유지하며 전체 양성은434장이다. Negative는 변경하지 않았다.

- [사용할 전체434장 manifest 및 프레임별 판정](SUBSET_MANIFEST.json)
- [제외35장 원본 RGB 갤러리](../../../../../outputs/pallet_type_selftrain_v1/large_corner_recovery_v1/extreme_low_subset_v1/excluded.html)
- [전체194 / 유지159 / 제외35 모델별 비교](../../../../../outputs/pallet_type_selftrain_v1/large_corner_recovery_v1/extreme_low_subset_v1/RESULTS_KO.md)
- [동일 비교의 수치·출처 해시](RESULTS.json)

원본 파일·주석·split·기존 평가 기본값과 기존 논문 표는 바꾸지 않았다.
후속 subset 평가는 위 manifest의 `records`를 명시적으로 사용한다.
제외35장도 학습에 넣지 않으며, 전체194 결과를 함께 보존한다.
평가 범위의 사후 제한이지 모델 개선이나 큰 오차 복구가 아니다.

출력 폴더의 RESULTS_KO.md 본문에 쓰인 “이 디렉터리의 manifest”는 이 문서가 있는 문서 폴더를 뜻한다.
그 보고서 맨 아래 상대 링크 대신 위 갤러리 링크를 사용한다.
