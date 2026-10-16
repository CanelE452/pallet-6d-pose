# 이 실험은 외부 v5 번들의 재현이 아니다

외부 지시문 `pallet_point_line_v5_CLI_INSTRUCTIONS_KO.md` 는 저자가 제공하는
`plpose_v5/` 코어(72 tests 통과, 3 arm x 8 = 24 AdamW updates 로 검증됨)를 로컬 데이터에
**연결**하라고 요구한다. 그 코어는 전달되지 않았다(`../pallet_point_line_v5/BLOCKED_MISSING_BUNDLE.json`).

사용자 지시로 여기서는 **같은 과학적 질문에 답하기 위한 독립 구현**을 만들어 돌린다.
따라서 다음을 명시한다.

- 이 결과는 **저자의 v5 코어를 검증한 것이 아니다.** 코드가 다르므로 저자의 테스트·
  파라미터 수(point 24,927 / direct 26,879 / hough 32,683)·수치와 대응하지 않는다.
- P/D/H 세 군의 상대 비교는 **이 구현 안에서만** 유효하다. 저자 구현에서 같은 결론이
  나온다는 보장이 없다.
- 실험 이름을 `pallet_point_line_v5_local` 로 분리해 저자 v5 산출물과 섞이지 않게 한다.
- 외부 v5 지시문이 정한 **계약과 금지사항은 그대로 따른다**: 새 object frame 정의,
  치수 conditioning, 대칭 min-over-g, native-only 데이터, 예산 상한(3 arm x 3 seed x
  2,000 = 18,000 optimizer steps), 실사 평가·online 전체학습·FINAL·commit/push·외부알림 금지.
- 대칭: 로컬 계약상 **C4 표본 0** 이므로 C1/C2 만 실행된다
  (`../pallet_point_line_v5/DIMENSION_SYMMETRY_AUDIT.json`).
