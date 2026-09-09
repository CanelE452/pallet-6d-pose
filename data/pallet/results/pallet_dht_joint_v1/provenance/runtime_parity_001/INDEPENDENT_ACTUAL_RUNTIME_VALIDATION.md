# 실제 RUNTIME 독립 검증

검증 기록은 일치합니다. **원래 strict parity 검사는 여전히 실패입니다(PASS=false, 7/702회).**

- RUNTIME SHA256: `6b40769bcc35dc946cc39a2d7943f28f173efe41863ff946b23f5dfff945a11e`
- 702개 시간 관측과 9개 모델의 평균·중앙값·P90을 재계산했으며 저장값과 차이는 모두 0입니다.
- 실패 7건을 원본 정확도 캐시로 재비교했습니다. 기준 후보, 이미지 key↔frame ID, 오류 문구, 좌표 차이, observation 연결이 모두 정확히 일치합니다.
- compare 함수는 원본 바이트와 같으며 atol=1e-4, rtol=0을 유지합니다. 후보 수·shape·availability·비유한 값은 여전히 fatal입니다.
- 보존 대상으로 지정된 정확도 관련 29개 artifact SHA가 모두 유지됐습니다.
- CPU 읽기 검증만 수행했고 신경망 실행·GPU 사용·bound source 변경은 없습니다.

세부 근거는 같은 폴더의 `INDEPENDENT_ACTUAL_RUNTIME_VALIDATION.json`에 있습니다.
